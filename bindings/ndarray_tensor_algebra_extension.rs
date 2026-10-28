//! General tensor algebra through maintained Rust `ndarray` kernels.

use ndarray::{ArrayD as DenseArray, IxDyn};
use rustfft::num_complex::Complex64;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CStr;
use std::panic::{catch_unwind, AssertUnwindSafe};

use super::native_support::{
    element_count, numeric, owned_int_as, owned_numeric, owned_numeric_as, promoted_dtype, shape,
    write_result, Array, AXIS_OUT_OF_RANGE, DOMAIN_ERROR, INVALID_ARGUMENT, SHAPE_MISMATCH,
    SIZE_OVERFLOW,
};

/// One parsed Einstein-summation term.
#[derive(Clone, Debug, Eq, PartialEq)]
enum EquationToken {
    /// One caller-visible alphabetic axis label.
    Label(char),
    /// One NumPy-style broadcast ellipsis.
    Ellipsis,
}

/// A normalized equation and the broadcast shape required for each operand.
#[derive(Debug)]
struct NormalizedEquation {
    /// Lowercase equation accepted by the maintained contraction kernel.
    equation: String,
    /// Dense shape to which each source operand must be broadcast.
    operand_shapes: Vec<Vec<usize>>,
}

/// Parses one equation term without accepting malformed dots or punctuation.
fn parse_term(term: &str) -> Result<Vec<EquationToken>, i32> {
    let mut tokens = Vec::new();
    let mut chars = term.chars().peekable();
    let mut has_ellipsis = false;
    while let Some(character) = chars.next() {
        if character == '.' {
            if chars.next() != Some('.') || chars.next() != Some('.') || has_ellipsis {
                return Err(SHAPE_MISMATCH);
            }
            has_ellipsis = true;
            tokens.push(EquationToken::Ellipsis);
        } else if character.is_ascii_alphabetic() {
            tokens.push(EquationToken::Label(character));
        } else {
            return Err(SHAPE_MISMATCH);
        }
    }
    Ok(tokens)
}

/// Expands one parsed term into concrete logical labels aligned to its rank.
fn expand_term(
    tokens: &[EquationToken],
    rank: usize,
    ellipsis_rank: usize,
    ellipsis_labels: &[char],
) -> Result<Vec<char>, i32> {
    let explicit = tokens
        .iter()
        .filter(|token| matches!(token, EquationToken::Label(_)))
        .count();
    let has_ellipsis = tokens.contains(&EquationToken::Ellipsis);
    if (!has_ellipsis && explicit != rank) || (has_ellipsis && explicit > rank) {
        return Err(SHAPE_MISMATCH);
    }
    let local_ellipsis = if has_ellipsis { rank - explicit } else { 0 };
    let ellipsis_start = ellipsis_rank
        .checked_sub(local_ellipsis)
        .ok_or(SHAPE_MISMATCH)?;
    let mut labels = Vec::with_capacity(rank);
    for token in tokens {
        match token {
            EquationToken::Label(label) => labels.push(*label),
            EquationToken::Ellipsis => labels.extend_from_slice(
                ellipsis_labels
                    .get(ellipsis_start..)
                    .ok_or(SHAPE_MISMATCH)?,
            ),
        }
    }
    Ok(labels)
}

/// Merges one labeled dimension under NumPy's singleton-broadcasting rule.
fn merge_dimension(
    dimensions: &mut BTreeMap<char, usize>,
    label: char,
    size: usize,
) -> Result<(), i32> {
    match dimensions.get(&label).copied() {
        None => {
            dimensions.insert(label, size);
            Ok(())
        }
        Some(current) if current == size || size == 1 => Ok(()),
        Some(1) => {
            dimensions.insert(label, size);
            Ok(())
        }
        Some(_) => Err(SHAPE_MISMATCH),
    }
}

/// Normalizes NumPy equation syntax into the maintained kernel's explicit form.
fn normalize_equation(equation: &str, shapes: &[Vec<usize>]) -> Result<NormalizedEquation, i32> {
    let compact = equation
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect::<String>();
    let mut arrow = compact.split("->");
    let inputs = arrow.next().ok_or(SHAPE_MISMATCH)?;
    let explicit_output = arrow.next();
    if arrow.next().is_some() {
        return Err(SHAPE_MISMATCH);
    }
    let input_terms = inputs
        .split(',')
        .map(parse_term)
        .collect::<Result<Vec<_>, _>>()?;
    if input_terms.len() != shapes.len() || input_terms.is_empty() {
        return Err(SHAPE_MISMATCH);
    }
    let output_tokens = explicit_output.map(parse_term).transpose()?;
    let ellipsis_rank = input_terms
        .iter()
        .zip(shapes)
        .map(|(tokens, shape)| {
            let explicit = tokens
                .iter()
                .filter(|token| matches!(token, EquationToken::Label(_)))
                .count();
            if tokens.contains(&EquationToken::Ellipsis) {
                shape.len().checked_sub(explicit).ok_or(SHAPE_MISMATCH)
            } else if explicit == shape.len() {
                Ok(0)
            } else {
                Err(SHAPE_MISMATCH)
            }
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(0);

    let caller_labels = input_terms
        .iter()
        .flat_map(|term| term.iter())
        .filter_map(|token| match token {
            EquationToken::Label(label) => Some(*label),
            EquationToken::Ellipsis => None,
        })
        .collect::<BTreeSet<_>>();
    if caller_labels.len() + ellipsis_rank + shapes.iter().filter(|shape| shape.is_empty()).count()
        > 26
    {
        return Err(SHAPE_MISMATCH);
    }
    let synthetic = (0..26)
        .map(|index| char::from(b'a' + index))
        .filter(|label| !caller_labels.contains(label))
        .collect::<Vec<_>>();
    let ellipsis_labels = synthetic
        .get(..ellipsis_rank)
        .ok_or(SHAPE_MISMATCH)?
        .to_vec();
    let mut expanded = input_terms
        .iter()
        .zip(shapes)
        .map(|(tokens, shape)| expand_term(tokens, shape.len(), ellipsis_rank, &ellipsis_labels))
        .collect::<Result<Vec<_>, _>>()?;

    let mut private_index = ellipsis_rank;
    let mut private_labels = BTreeSet::new();
    for labels in &mut expanded {
        if labels.is_empty() {
            let label = *synthetic.get(private_index).ok_or(SHAPE_MISMATCH)?;
            labels.push(label);
            private_labels.insert(label);
            private_index += 1;
        }
    }
    let mut dimensions = BTreeMap::new();
    let mut counts = BTreeMap::new();
    for ((labels, shape), tokens) in expanded.iter().zip(shapes).zip(&input_terms) {
        let scalar_expansion = shape.is_empty() && labels.len() == 1;
        let effective_shape = if scalar_expansion {
            &[1][..]
        } else {
            shape.as_slice()
        };
        let mut local = BTreeMap::new();
        for (&label, &size) in labels.iter().zip(effective_shape) {
            if let Some(previous) = local.insert(label, size) {
                if previous != size {
                    return Err(SHAPE_MISMATCH);
                }
            }
            merge_dimension(&mut dimensions, label, size)?;
            *counts.entry(label).or_insert(0usize) += 1;
        }
        if !tokens.contains(&EquationToken::Ellipsis)
            && !scalar_expansion
            && labels.len() != shape.len()
        {
            return Err(SHAPE_MISMATCH);
        }
    }

    let output_labels = if let Some(tokens) = output_tokens {
        let explicit = tokens
            .iter()
            .filter(|token| matches!(token, EquationToken::Label(_)))
            .count();
        let output_rank = explicit
            + if tokens.contains(&EquationToken::Ellipsis) {
                ellipsis_rank
            } else {
                0
            };
        let labels = expand_term(&tokens, output_rank, ellipsis_rank, &ellipsis_labels)?;
        let unique = labels.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != labels.len()
            || labels.iter().any(|label| !dimensions.contains_key(label))
        {
            return Err(SHAPE_MISMATCH);
        }
        labels
    } else {
        let mut labels = ellipsis_labels.clone();
        labels.extend(counts.iter().filter_map(|(&label, &count)| {
            (count == 1 && !ellipsis_labels.contains(&label) && !private_labels.contains(&label))
                .then_some(label)
        }));
        labels
    };

    let all_labels = expanded
        .iter()
        .flatten()
        .chain(&output_labels)
        .copied()
        .collect::<BTreeSet<_>>();
    if all_labels.len() > 26 {
        return Err(SHAPE_MISMATCH);
    }
    let remapping = all_labels
        .into_iter()
        .zip((0..26).map(|index| char::from(b'a' + index)))
        .collect::<BTreeMap<_, _>>();
    let render = |labels: &[char]| -> String {
        labels
            .iter()
            .filter_map(|label| remapping.get(label))
            .collect()
    };
    let operand_shapes = expanded
        .iter()
        .map(|labels| labels.iter().map(|label| dimensions[label]).collect())
        .collect::<Vec<Vec<usize>>>();
    let inputs = expanded
        .iter()
        .map(|labels| render(labels))
        .collect::<Vec<_>>()
        .join(",");
    Ok(NormalizedEquation {
        equation: format!("{inputs}->{}", render(&output_labels)),
        operand_shapes,
    })
}

/// Builds a shared view for a single-operand permutation or diagonal equation.
unsafe fn singleton_view(array: Array, normalized_equation: &str) -> Result<Option<Array>, i32> {
    let (input, output) = normalized_equation.split_once("->").ok_or(SHAPE_MISMATCH)?;
    if input.contains(',') {
        return Ok(None);
    }
    let input_set = input.chars().collect::<BTreeSet<_>>();
    let output_set = output.chars().collect::<BTreeSet<_>>();
    if input_set != output_set || output_set.len() != output.chars().count() {
        return Ok(None);
    }
    let mut labels = input.chars().collect::<Vec<_>>();
    let mut current = array;
    let mut owned = false;
    loop {
        let duplicate = labels.iter().enumerate().find_map(|(first, label)| {
            labels
                .iter()
                .enumerate()
                .skip(first + 1)
                .find_map(|(second, candidate)| {
                    (candidate == label).then_some((first, second, *label))
                })
        });
        let Some((first, second, label)) = duplicate else {
            break;
        };
        let mut next = std::ptr::null_mut();
        let status = unsafe {
            crate::ffi::terlan_ndarray_v1_diagonal(
                current,
                0,
                first as i64,
                second as i64,
                &mut next,
            )
        };
        if status != super::native_support::OK {
            if owned {
                let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(current) };
            }
            return Err(status);
        }
        if owned {
            let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(current) };
        }
        labels.remove(second);
        labels.remove(first);
        labels.push(label);
        current = next;
        owned = true;
    }
    for (destination, label) in output.chars().enumerate() {
        let source = labels
            .iter()
            .position(|candidate| *candidate == label)
            .ok_or(SHAPE_MISMATCH)?;
        if source == destination {
            continue;
        }
        let mut next = std::ptr::null_mut();
        let status = unsafe {
            crate::ffi::terlan_ndarray_v1_transpose(
                current,
                destination as i64,
                source as i64,
                &mut next,
            )
        };
        if status != super::native_support::OK {
            if owned {
                let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(current) };
            }
            return Err(status);
        }
        if owned {
            let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(current) };
        }
        labels.swap(destination, source);
        current = next;
        owned = true;
    }
    if !owned {
        let dimensions = unsafe { shape(array)? };
        let mut view = std::ptr::null_mut();
        let status = unsafe {
            crate::ffi::terlan_ndarray_v1_reshape(
                array,
                dimensions.as_ptr(),
                dimensions.len() as i64,
                &mut view,
            )
        };
        if status != super::native_support::OK {
            return Err(status);
        }
        current = view;
    }
    Ok(Some(current))
}

/// Converts an admitted package array into owned row-major Rust storage.
unsafe fn dense(array: Array) -> Result<(DenseArray<Complex64>, bool, i32), i32> {
    let dimensions = unsafe { shape(array)? };
    let dense_shape: Vec<usize> = dimensions
        .iter()
        .map(|&dimension| usize::try_from(dimension).map_err(|_| SHAPE_MISMATCH))
        .collect::<Result<_, _>>()?;
    let values = unsafe { numeric(array, false)? };
    let dtype = values.dtype;
    let dense = DenseArray::from_shape_vec(IxDyn(&dense_shape), values.values)
        .map_err(|_| SHAPE_MISMATCH)?;
    Ok((dense, values.complex, dtype))
}

/// Converts owned Rust storage back into one Float64 package array.
unsafe fn owned_dense(array: DenseArray<Complex64>, dtype: i32) -> Result<Array, i32> {
    let dimensions: Vec<i64> = array
        .shape()
        .iter()
        .map(|&dimension| i64::try_from(dimension).map_err(|_| SIZE_OVERFLOW))
        .collect::<Result<_, _>>()?;
    let values: Vec<Complex64> = array.iter().copied().collect();
    unsafe { owned_einsum_values(&values, &dimensions, dtype) }
}

/// Stores contraction results with NumPy's fixed-width integer wrap semantics.
unsafe fn owned_einsum_values(
    values: &[Complex64],
    dimensions: &[i64],
    dtype: i32,
) -> Result<Array, i32> {
    let (bits, signed) = match dtype {
        4 => (8, true),
        5 => (16, true),
        6 => (32, true),
        2 => (64, true),
        7 => (8, false),
        8 => (16, false),
        9 => (32, false),
        10 => (64, false),
        _ => return unsafe { owned_numeric_as(values, dimensions, dtype) },
    };
    let modulus = 1_i128 << bits;
    let sign = modulus / 2;
    let values = values
        .iter()
        .map(|value| {
            if value.im != 0.0 || !value.re.is_finite() || value.re.fract() != 0.0 {
                return Err(DOMAIN_ERROR);
            }
            let integer = value.re as i128;
            let raw = ((integer % modulus) + modulus) % modulus;
            let wrapped = if signed && raw >= sign {
                raw - modulus
            } else {
                raw
            };
            i64::try_from(wrapped).map_err(|_| DOMAIN_ERROR)
        })
        .collect::<Result<Vec<_>, _>>()?;
    unsafe { owned_int_as(&values, dimensions, dtype) }
}

/// Normalizes and validates a caller-owned axis list.
unsafe fn axes(values: *const i64, count: i64, rank: usize) -> Result<Vec<usize>, i32> {
    if count < 0 || (count > 0 && values.is_null()) {
        return Err(INVALID_ARGUMENT);
    }
    let count = usize::try_from(count).map_err(|_| SIZE_OVERFLOW)?;
    let values = if count == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(values, count) }
    };
    let rank_i64 = i64::try_from(rank).map_err(|_| AXIS_OUT_OF_RANGE)?;
    let mut output = Vec::with_capacity(count);
    for &axis in values {
        let axis = if axis < 0 { axis + rank_i64 } else { axis };
        if axis < 0 || axis >= rank_i64 {
            return Err(AXIS_OUT_OF_RANGE);
        }
        let axis = usize::try_from(axis).map_err(|_| AXIS_OUT_OF_RANGE)?;
        if output.contains(&axis) {
            return Err(INVALID_ARGUMENT);
        }
        output.push(axis);
    }
    Ok(output)
}

/// Decodes one row-major linear index into coordinates.
fn coordinates(mut index: usize, dimensions: &[usize]) -> Vec<usize> {
    let mut output = vec![0; dimensions.len()];
    for axis in (0..dimensions.len()).rev() {
        if dimensions[axis] > 0 {
            output[axis] = index % dimensions[axis];
            index /= dimensions[axis];
        }
    }
    output
}

/// Encodes row-major coordinates as one linear index.
fn linear_index(coordinates: &[usize], dimensions: &[usize]) -> usize {
    coordinates
        .iter()
        .zip(dimensions)
        .fold(0usize, |index, (&coordinate, &dimension)| {
            index * dimension + coordinate
        })
}

/// Computes a generalized tensor contraction over explicit axis lists.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_tensordot(
    lhs: Array,
    rhs: Array,
    lhs_axes: *const i64,
    lhs_axis_count: i64,
    rhs_axes: *const i64,
    rhs_axis_count: i64,
    output: *mut Array,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    let lhs_shape = match unsafe { shape(lhs) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let rhs_shape = match unsafe { shape(rhs) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if lhs_axis_count != rhs_axis_count {
        return SHAPE_MISMATCH;
    }
    let lhs_axes = match unsafe { axes(lhs_axes, lhs_axis_count, lhs_shape.len()) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let rhs_axes = match unsafe { axes(rhs_axes, rhs_axis_count, rhs_shape.len()) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if lhs_axes
        .iter()
        .zip(&rhs_axes)
        .any(|(&left, &right)| lhs_shape[left] != rhs_shape[right])
    {
        return SHAPE_MISMATCH;
    }
    let lhs_free: Vec<usize> = (0..lhs_shape.len())
        .filter(|axis| !lhs_axes.contains(axis))
        .collect();
    let rhs_free: Vec<usize> = (0..rhs_shape.len())
        .filter(|axis| !rhs_axes.contains(axis))
        .collect();
    let output_shape: Vec<i64> = lhs_free
        .iter()
        .map(|&axis| lhs_shape[axis])
        .chain(rhs_free.iter().map(|&axis| rhs_shape[axis]))
        .collect();
    let contract_shape: Vec<i64> = lhs_axes.iter().map(|&axis| lhs_shape[axis]).collect();
    let lhs_values = match unsafe { numeric(lhs, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let rhs_values = match unsafe { numeric(rhs, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let output_count = match element_count(&output_shape) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let contract_count = match element_count(&contract_shape) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let lhs_dimensions: Vec<usize> = lhs_shape.iter().map(|&value| value as usize).collect();
    let rhs_dimensions: Vec<usize> = rhs_shape.iter().map(|&value| value as usize).collect();
    let output_dimensions: Vec<usize> = output_shape.iter().map(|&value| value as usize).collect();
    let contract_dimensions: Vec<usize> =
        contract_shape.iter().map(|&value| value as usize).collect();
    let complex = lhs_values.complex || rhs_values.complex;
    let narrow =
        matches!(lhs_values.dtype, 11 | 12 | 14) && matches!(rhs_values.dtype, 11 | 12 | 14);
    let mut values = vec![Complex64::new(0.0, 0.0); output_count];
    for (output_index, value) in values.iter_mut().enumerate() {
        let output_coordinates = coordinates(output_index, &output_dimensions);
        let mut lhs_coordinates = vec![0; lhs_shape.len()];
        let mut rhs_coordinates = vec![0; rhs_shape.len()];
        for (index, &axis) in lhs_free.iter().enumerate() {
            lhs_coordinates[axis] = output_coordinates[index];
        }
        for (index, &axis) in rhs_free.iter().enumerate() {
            rhs_coordinates[axis] = output_coordinates[lhs_free.len() + index];
        }
        for contract_index in 0..contract_count {
            let contract_coordinates = coordinates(contract_index, &contract_dimensions);
            for (index, &axis) in lhs_axes.iter().enumerate() {
                lhs_coordinates[axis] = contract_coordinates[index];
            }
            for (index, &axis) in rhs_axes.iter().enumerate() {
                rhs_coordinates[axis] = contract_coordinates[index];
            }
            *value += lhs_values.values[linear_index(&lhs_coordinates, &lhs_dimensions)]
                * rhs_values.values[linear_index(&rhs_coordinates, &rhs_dimensions)];
        }
    }
    unsafe {
        write_result(
            owned_numeric(&values, &output_shape, complex, narrow),
            output,
        )
    }
}

/// Computes a NumPy-style Kronecker product over rank-padded operands.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_kron(lhs: Array, rhs: Array, output: *mut Array) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    let lhs_shape = match unsafe { shape(lhs) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let rhs_shape = match unsafe { shape(rhs) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let rank = lhs_shape.len().max(rhs_shape.len());
    let mut lhs_padded = vec![1i64; rank - lhs_shape.len()];
    lhs_padded.extend(lhs_shape);
    let mut rhs_padded = vec![1i64; rank - rhs_shape.len()];
    rhs_padded.extend(rhs_shape);
    let output_shape: Vec<i64> = lhs_padded
        .iter()
        .zip(&rhs_padded)
        .map(|(&left, &right)| left.checked_mul(right).ok_or(SIZE_OVERFLOW))
        .collect::<Result<_, _>>()
        .unwrap_or_default();
    if output_shape.len() != rank {
        return SIZE_OVERFLOW;
    }
    let lhs_values = match unsafe { numeric(lhs, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let rhs_values = match unsafe { numeric(rhs, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let count = match element_count(&output_shape) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let output_dimensions: Vec<usize> = output_shape.iter().map(|&value| value as usize).collect();
    let lhs_dimensions: Vec<usize> = lhs_padded.iter().map(|&value| value as usize).collect();
    let rhs_dimensions: Vec<usize> = rhs_padded.iter().map(|&value| value as usize).collect();
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let coordinate = coordinates(index, &output_dimensions);
        let lhs_coordinate: Vec<usize> = coordinate
            .iter()
            .zip(&rhs_dimensions)
            .map(|(&value, &right)| value / right)
            .collect();
        let rhs_coordinate: Vec<usize> = coordinate
            .iter()
            .zip(&rhs_dimensions)
            .map(|(&value, &right)| value % right)
            .collect();
        values.push(
            lhs_values.values[linear_index(&lhs_coordinate, &lhs_dimensions)]
                * rhs_values.values[linear_index(&rhs_coordinate, &rhs_dimensions)],
        );
    }
    let complex = lhs_values.complex || rhs_values.complex;
    let narrow =
        matches!(lhs_values.dtype, 11 | 12 | 14) && matches!(rhs_values.dtype, 11 | 12 | 14);
    unsafe {
        write_result(
            owned_numeric(&values, &output_shape, complex, narrow),
            output,
        )
    }
}

/// Computes a rank-preserving vector cross product along one axis.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_cross(
    lhs: Array,
    rhs: Array,
    axis: i64,
    output: *mut Array,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    let lhs_shape = match unsafe { shape(lhs) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let rhs_shape = match unsafe { shape(rhs) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if lhs_shape.len() != rhs_shape.len() || lhs_shape.is_empty() {
        return SHAPE_MISMATCH;
    }
    let axis = match unsafe { axes(&axis, 1, lhs_shape.len()) } {
        Ok(value) => value[0],
        Err(status) => return status,
    };
    if !matches!(lhs_shape[axis], 2 | 3) || !matches!(rhs_shape[axis], 2 | 3) {
        return SHAPE_MISMATCH;
    }
    let lhs_values = match unsafe { numeric(lhs, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let rhs_values = match unsafe { numeric(rhs, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut batch_shape = lhs_shape.clone();
    for current_axis in 0..batch_shape.len() {
        if current_axis == axis {
            continue;
        }
        batch_shape[current_axis] = match (lhs_shape[current_axis], rhs_shape[current_axis]) {
            (left, right) if left == right => left,
            (1, right) => right,
            (left, 1) => left,
            _ => return SHAPE_MISMATCH,
        };
    }
    let scalar_output = lhs_shape[axis] == 2 && rhs_shape[axis] == 2;
    let mut output_shape = batch_shape.clone();
    if scalar_output {
        output_shape.remove(axis);
    } else {
        output_shape[axis] = 3;
    }
    let count = match element_count(&output_shape) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let lhs_dimensions: Vec<usize> = lhs_shape.iter().map(|&value| value as usize).collect();
    let rhs_dimensions: Vec<usize> = rhs_shape.iter().map(|&value| value as usize).collect();
    let output_dimensions: Vec<usize> = output_shape.iter().map(|&value| value as usize).collect();
    let mut values = vec![Complex64::new(0.0, 0.0); count];
    for (index, value) in values.iter_mut().enumerate() {
        let output_coordinate = coordinates(index, &output_dimensions);
        let component = if scalar_output {
            2
        } else {
            output_coordinate[axis]
        };
        let mut batch_coordinate = output_coordinate;
        if scalar_output {
            batch_coordinate.insert(axis, 0);
        }
        let component_value = |source: &super::native_support::NumericValues,
                               source_shape: &[i64],
                               source_dimensions: &[usize],
                               requested: usize| {
            if requested >= source_shape[axis] as usize {
                return Complex64::new(0.0, 0.0);
            }
            let mut coordinate = batch_coordinate.clone();
            for current_axis in 0..coordinate.len() {
                if current_axis != axis && source_shape[current_axis] == 1 {
                    coordinate[current_axis] = 0;
                }
            }
            coordinate[axis] = requested;
            source.values[linear_index(&coordinate, source_dimensions)]
        };
        let next = (component + 1) % 3;
        let following = (component + 2) % 3;
        *value = component_value(&lhs_values, &lhs_shape, &lhs_dimensions, next)
            * component_value(&rhs_values, &rhs_shape, &rhs_dimensions, following)
            - component_value(&lhs_values, &lhs_shape, &lhs_dimensions, following)
                * component_value(&rhs_values, &rhs_shape, &rhs_dimensions, next);
    }
    let complex = lhs_values.complex || rhs_values.complex;
    let narrow =
        matches!(lhs_values.dtype, 11 | 12 | 14) && matches!(rhs_values.dtype, 11 | 12 | 14);
    unsafe {
        write_result(
            owned_numeric(&values, &output_shape, complex, narrow),
            output,
        )
    }
}

/// Executes an Einstein summation expression over one or more arrays.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_einsum(
    equation: *const std::ffi::c_char,
    operands: *const u64,
    operand_count: i64,
    output: *mut Array,
) -> i32 {
    if equation.is_null() || output.is_null() || operand_count <= 0 || operands.is_null() {
        return INVALID_ARGUMENT;
    }
    let equation = match unsafe { CStr::from_ptr(equation) }.to_str() {
        Ok(value) if !value.is_empty() => value,
        _ => return INVALID_ARGUMENT,
    };
    let operand_count = match usize::try_from(operand_count) {
        Ok(value) => value,
        Err(_) => return SIZE_OVERFLOW,
    };
    let identities = unsafe { std::slice::from_raw_parts(operands, operand_count) };
    let mut arrays = Vec::with_capacity(operand_count);
    let mut output_dtype = None;
    for &identity in identities {
        let address = match usize::try_from(identity) {
            Ok(0) | Err(_) => return INVALID_ARGUMENT,
            Ok(value) => value,
        };
        let (array, _array_complex, array_dtype) = match unsafe { dense(address as Array) } {
            Ok(value) => value,
            Err(status) => return status,
        };
        arrays.push(array);
        output_dtype = match output_dtype {
            None => Some(array_dtype),
            Some(dtype) => match unsafe { promoted_dtype(dtype, array_dtype) } {
                Ok(dtype) => Some(dtype),
                Err(status) => return status,
            },
        };
    }
    let shapes = arrays
        .iter()
        .map(|array| array.shape().to_vec())
        .collect::<Vec<_>>();
    let normalized = match normalize_equation(equation, &shapes) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if arrays.len() == 1 && arrays[0].shape() == normalized.operand_shapes[0] {
        match unsafe { singleton_view(identities[0] as usize as Array, &normalized.equation) } {
            Ok(Some(view)) => return unsafe { write_result(Ok(view), output) },
            Ok(None) => {}
            Err(status) => return status,
        }
    }
    let arrays = match arrays
        .into_iter()
        .zip(normalized.operand_shapes)
        .map(|(array, target)| {
            if array.shape() == target {
                Ok(array)
            } else if array.ndim() == 0 && target == [1] {
                array
                    .into_shape_with_order(IxDyn(&target))
                    .map_err(|_| SHAPE_MISMATCH)
            } else {
                array
                    .broadcast(IxDyn(&target))
                    .map(|view| view.to_owned())
                    .ok_or(SHAPE_MISMATCH)
            }
        })
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(value) => value,
        Err(status) => return status,
    };
    let references = arrays
        .iter()
        .map(|array| array as &dyn ndarray_einsum::ArrayLike<Complex64>)
        .collect::<Vec<_>>();
    let result = match catch_unwind(AssertUnwindSafe(|| {
        ndarray_einsum::einsum(&normalized.equation, &references)
    })) {
        Ok(Ok(value)) => value,
        Ok(Err(_)) | Err(_) => return SHAPE_MISMATCH,
    };
    unsafe { write_result(owned_dense(result, output_dtype.unwrap_or(3)), output) }
}

/// Executes Einstein summation and atomically writes the result into an array.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_einsum_into(
    destination: Array,
    equation: *const std::ffi::c_char,
    operands: *const u64,
    operand_count: i64,
) -> i32 {
    if destination.is_null() {
        return INVALID_ARGUMENT;
    }
    let mut result = std::ptr::null_mut();
    let status =
        unsafe { terlan_ndarray_v1_einsum(equation, operands, operand_count, &mut result) };
    if status != super::native_support::OK {
        return status;
    }
    let write_status = unsafe { crate::ffi::terlan_ndarray_v1_copyto(destination, result) };
    let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(result) };
    write_status
}

/// Reorders a rank-two-or-higher result into a stable noncanonical shared view.
unsafe fn keep_ordered_result(result: Array, output: *mut Array) -> i32 {
    let dimensions = match unsafe { shape(result) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if dimensions.len() < 2 || dimensions[0] <= 1 || dimensions[1] <= 1 {
        let status = unsafe { crate::ffi::terlan_ndarray_v1_copy_ordered(result, 1, output) };
        let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(result) };
        return status;
    }
    let mut transposed = std::ptr::null_mut();
    let mut backing = std::ptr::null_mut();
    let status = unsafe { crate::ffi::terlan_ndarray_v1_transpose(result, 0, 1, &mut transposed) };
    let status = if status == super::native_support::OK {
        unsafe { crate::ffi::terlan_ndarray_v1_copy_ordered(transposed, 0, &mut backing) }
    } else {
        status
    };
    let status = if status == super::native_support::OK {
        unsafe { crate::ffi::terlan_ndarray_v1_transpose(backing, 0, 1, output) }
    } else {
        status
    };
    if !backing.is_null() {
        let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(backing) };
    }
    if !transposed.is_null() {
        let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(transposed) };
    }
    let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(result) };
    status
}

/// Executes Einstein summation with an explicit output memory-order policy.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_einsum_ordered(
    equation: *const std::ffi::c_char,
    operands: *const u64,
    operand_count: i64,
    order: i32,
    output: *mut Array,
) -> i32 {
    if output.is_null() || !matches!(order, 0..=3) {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    let mut result = std::ptr::null_mut();
    let status =
        unsafe { terlan_ndarray_v1_einsum(equation, operands, operand_count, &mut result) };
    if status != super::native_support::OK {
        return status;
    }
    let all_fortran = if matches!(order, 2 | 3) {
        let identities = unsafe { std::slice::from_raw_parts(operands, operand_count as usize) };
        identities.iter().all(|identity| {
            let array = *identity as usize as Array;
            let mut contiguous = 0_i32;
            (unsafe { crate::ffi::terlan_ndarray_v1_is_fortran_contiguous(array, &mut contiguous) })
                == super::native_support::OK
                && contiguous == 1
        })
    } else {
        false
    };
    if order == 2 && all_fortran {
        return unsafe { keep_ordered_result(result, output) };
    }
    let selected_order = if order == 1 || (order == 3 && all_fortran) {
        1
    } else {
        0
    };
    if selected_order == 0 {
        unsafe { *output = result };
        return super::native_support::OK;
    }
    let copy_status =
        unsafe { crate::ffi::terlan_ndarray_v1_copy_ordered(result, selected_order, output) };
    let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(result) };
    copy_status
}

#[cfg(test)]
#[path = "ndarray_tensor_algebra_tests.rs"]
mod tests;
