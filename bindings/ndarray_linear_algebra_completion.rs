//! NumPy-compatible orchestration over the maintained `faer` linear algebra core.

use faer::linalg::solvers::{DenseSolveCore, Solve};
use faer::{c64, Mat};

use super::norm_support::{
    flat_coordinates, matrix_norm, output_index, vector_norm, GeneralNormOrder,
};

use super::native_support::{
    delete_arrays, element_count, normalize_axis, numeric, owned_numeric, owned_numeric_as,
    owned_real, shape, transfer_arrays, write_result, Array, DOMAIN_ERROR, INVALID_ARGUMENT, OK,
    SHAPE_MISMATCH, SIZE_OVERFLOW,
};

/// Reads a finite rank-two array into a row-major complex matrix.
unsafe fn matrix(array: Array) -> Result<(Mat<c64>, Vec<i64>, bool, bool, i32), i32> {
    let dimensions = unsafe { shape(array)? };
    if dimensions.len() != 2 {
        return Err(SHAPE_MISMATCH);
    }
    let rows = usize::try_from(dimensions[0]).map_err(|_| SHAPE_MISMATCH)?;
    let columns = usize::try_from(dimensions[1]).map_err(|_| SHAPE_MISMATCH)?;
    let source = unsafe { numeric(array, true)? };
    let narrow = matches!(source.dtype, 11 | 12);
    Ok((
        Mat::from_fn(rows, columns, |row, column| {
            source.values[row * columns + column]
        }),
        dimensions,
        source.complex,
        narrow,
        source.dtype,
    ))
}

/// Copies one matrix into row-major storage.
fn matrix_values(matrix: &Mat<c64>) -> Vec<c64> {
    let mut values = Vec::with_capacity(matrix.nrows().saturating_mul(matrix.ncols()));
    for row in 0..matrix.nrows() {
        for column in 0..matrix.ncols() {
            values.push(matrix[(row, column)]);
        }
    }
    values
}

/// Raises one square matrix to an integer power.
fn integer_power(mut base: Mat<c64>, exponent: i64) -> Result<Mat<c64>, i32> {
    let dimension = base.nrows();
    if dimension != base.ncols() {
        return Err(SHAPE_MISMATCH);
    }
    if exponent < 0 {
        let singular = singular_values(&base)?;
        let threshold = tolerance(&base, &singular, 0.0)?;
        if singular.last().copied().unwrap_or(0.0) <= threshold {
            return Err(DOMAIN_ERROR);
        }
        base = base.partial_piv_lu().inverse();
    }
    let mut result = Mat::identity(dimension, dimension);
    let mut power = exponent.unsigned_abs();
    while power > 0 {
        if power & 1 == 1 {
            result = &result * &base;
        }
        power >>= 1;
        if power > 0 {
            base = &base * &base;
        }
    }
    Ok(result)
}

/// Requires a nonempty square matrix.
fn square(matrix: &Mat<c64>) -> Result<usize, i32> {
    if matrix.nrows() == 0 || matrix.nrows() != matrix.ncols() {
        Err(SHAPE_MISMATCH)
    } else {
        Ok(matrix.nrows())
    }
}

/// Returns singular values through `faer`'s maintained decomposition.
fn singular_values(matrix: &Mat<c64>) -> Result<Vec<f64>, i32> {
    matrix.singular_values().map_err(|_| DOMAIN_ERROR)
}

/// Computes the default tolerance used by rank-sensitive operations.
fn tolerance(matrix: &Mat<c64>, singular: &[f64], requested: f64) -> Result<f64, i32> {
    if !requested.is_finite() || requested < 0.0 {
        return Err(DOMAIN_ERROR);
    }
    if requested == 0.0 {
        Ok(f64::EPSILON
            * matrix.nrows().max(matrix.ncols()) as f64
            * singular.first().copied().unwrap_or(0.0))
    } else {
        Ok(requested)
    }
}

/// Computes a matrix condition number.
///
/// Orders `0` and `2` select the spectral norm, `-2` selects the reciprocal
/// singular-value ratio, `1`/`-1` select maximum/minimum column sums, and
/// `3`/`-3` select maximum/minimum row sums, and `4` selects Frobenius norm.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_cond(
    array: Array,
    order: i64,
    out_value: *mut f64,
) -> i32 {
    unsafe { super::linalg_system::condition_scalar(array, order, out_value) }
}

/// Computes general complex eigenvalues and right eigenvectors.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_eig(
    array: Array,
    out_values: *mut *mut u64,
    out_length: *mut usize,
) -> i32 {
    unsafe { super::linalg_system::eig(array, out_values, out_length) }
}

/// Computes general complex eigenvalues without materializing eigenvectors.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_eigvals(array: Array, output: *mut Array) -> i32 {
    unsafe { super::linalg_system::eigvals(array, output) }
}

/// Computes ascending eigenvalues of a finite Hermitian matrix.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_eigvalsh(array: Array, output: *mut Array) -> i32 {
    unsafe { super::linalg_decomposition::eigvalsh(array, false, output) }
}

/// Raises a square matrix to an integer power by checked exponentiation.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_matrix_power(
    array: Array,
    exponent: i64,
    output: *mut Array,
) -> i32 {
    let dimensions = match unsafe { shape(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if dimensions.len() < 2 || dimensions[dimensions.len() - 2] != dimensions[dimensions.len() - 1]
    {
        return SHAPE_MISMATCH;
    }
    let dimension = match usize::try_from(dimensions[dimensions.len() - 1]) {
        Ok(value) => value,
        Err(_) => return SHAPE_MISMATCH,
    };
    let batch_count = match element_count(&dimensions[..dimensions.len() - 2]) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let matrix_length = match dimension.checked_mul(dimension) {
        Some(value) => value,
        None => return SIZE_OVERFLOW,
    };
    let source = match unsafe { numeric(array, true) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut values = Vec::with_capacity(source.values.len());
    for batch in 0..batch_count {
        let offset = batch * matrix_length;
        let base = Mat::from_fn(dimension, dimension, |row, column| {
            source.values[offset + row * dimension + column]
        });
        let powered = match integer_power(base, exponent) {
            Ok(value) => value,
            Err(status) => return status,
        };
        values.extend(matrix_values(&powered));
    }
    let output_dtype = if exponent < 0 && matches!(source.dtype, 1 | 2 | 4..=10) {
        3
    } else {
        source.dtype
    };
    unsafe { write_result(owned_numeric_as(&values, &dimensions, output_dtype), output) }
}

/// Multiplies a nonempty chain of arrays from left to right with package matmul semantics.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_multi_dot(
    arrays: *const u64,
    array_count: i64,
    output: *mut Array,
) -> i32 {
    if output.is_null() || arrays.is_null() || array_count < 2 {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    let count = match usize::try_from(array_count) {
        Ok(value) => value,
        Err(_) => return SIZE_OVERFLOW,
    };
    let identities = unsafe { std::slice::from_raw_parts(arrays, count) };
    let resolve = |identity: u64| -> Result<Array, i32> {
        match usize::try_from(identity) {
            Ok(0) | Err(_) => Err(INVALID_ARGUMENT),
            Ok(value) => Ok(value as Array),
        }
    };
    let mut current = match (resolve(identities[0]), resolve(identities[1])) {
        (Ok(lhs), Ok(rhs)) => {
            let mut value = std::ptr::null_mut();
            let status = unsafe { crate::ffi::terlan_ndarray_v1_matmul(lhs, rhs, &mut value) };
            if status != OK {
                return status;
            }
            value
        }
        (Err(status), _) | (_, Err(status)) => return status,
    };
    for &identity in &identities[2..] {
        let rhs = match resolve(identity) {
            Ok(value) => value,
            Err(status) => {
                unsafe { delete_arrays(&[current]) };
                return status;
            }
        };
        let mut next = std::ptr::null_mut();
        let status = unsafe { crate::ffi::terlan_ndarray_v1_matmul(current, rhs, &mut next) };
        unsafe { delete_arrays(&[current]) };
        if status != OK {
            return status;
        }
        current = next;
    }
    unsafe { *output = current };
    OK
}

/// Multiplies a chain and atomically copies the result into a writable destination.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_multi_dot_into(
    destination: Array,
    arrays: *const u64,
    array_count: i64,
) -> i32 {
    if destination.is_null() {
        return INVALID_ARGUMENT;
    }
    let mut product = std::ptr::null_mut();
    let status = unsafe { terlan_ndarray_v1_multi_dot(arrays, array_count, &mut product) };
    if status != OK {
        return status;
    }
    let status = unsafe { crate::ffi::terlan_ndarray_v1_copyto(destination, product) };
    unsafe { delete_arrays(&[product]) };
    status
}

/// Computes the NumPy default vector or matrix norm.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_norm(array: Array, out_value: *mut f64) -> i32 {
    unsafe { super::linear_algebra_extension::terlan_ndarray_v1_frobenius_norm(array, out_value) }
}

/// Computes a generalized vector or matrix norm over one or two normalized axes.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_norm_general(
    array: Array,
    order_kind: i64,
    order_value: f64,
    axes: *const i64,
    axis_count: i64,
    keepdims: bool,
    output: *mut Array,
) -> i32 {
    if output.is_null() || axis_count < 0 || (axis_count > 0 && axes.is_null()) {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    let order = match order_kind {
        0 => GeneralNormOrder::Default,
        1 => GeneralNormOrder::NegativeInfinity,
        2 => GeneralNormOrder::Infinity,
        3 => GeneralNormOrder::Frobenius,
        4 => GeneralNormOrder::Nuclear,
        5 if order_value.is_finite() => GeneralNormOrder::Finite(order_value),
        _ => return INVALID_ARGUMENT,
    };
    let input_shape = match unsafe { shape(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let dimensions = match input_shape
        .iter()
        .map(|dimension| usize::try_from(*dimension).map_err(|_| SHAPE_MISMATCH))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(value) => value,
        Err(status) => return status,
    };
    let axis_count = match usize::try_from(axis_count) {
        Ok(value) => value,
        Err(_) => return INVALID_ARGUMENT,
    };
    let requested = if axis_count == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(axes, axis_count) }.to_vec()
    };
    let normalized = if requested.is_empty() {
        (0..dimensions.len()).collect::<Vec<_>>()
    } else {
        let mut output_axes = Vec::with_capacity(requested.len());
        for axis in requested {
            let axis = match normalize_axis(axis, dimensions.len()) {
                Ok(value) => value,
                Err(status) => return status,
            };
            if output_axes.contains(&axis) {
                return INVALID_ARGUMENT;
            }
            output_axes.push(axis);
        }
        output_axes
    };
    let matrix_mode = normalized.len() == 2 && (axis_count > 0 || dimensions.len() == 2);
    if (!matrix_mode && normalized.len() > 1 && !matches!(order, GeneralNormOrder::Default))
        || (matrix_mode
            && matches!(order, GeneralNormOrder::Finite(value) if !matches!(value, 1.0 | -1.0 | 2.0 | -2.0)))
    {
        return INVALID_ARGUMENT;
    }
    let mut reduced = vec![false; dimensions.len()];
    for &axis in &normalized {
        reduced[axis] = true;
    }
    let output_shape = input_shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &dimension)| {
            if reduced[axis] {
                keepdims.then_some(1)
            } else {
                Some(dimension)
            }
        })
        .collect::<Vec<_>>();
    let output_count = match element_count(&output_shape) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let source = match unsafe { numeric(array, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut groups = vec![Vec::<(usize, usize, c64)>::new(); output_count];
    for (flat, value) in source.values.iter().copied().enumerate() {
        let coordinates = flat_coordinates(flat, &dimensions);
        let group = output_index(&coordinates, &dimensions, &reduced, keepdims);
        let row = normalized
            .first()
            .map(|axis| coordinates[*axis])
            .unwrap_or(0);
        let column = normalized
            .get(1)
            .map(|axis| coordinates[*axis])
            .unwrap_or(0);
        groups[group].push((row, column, value));
    }
    let mut values = Vec::with_capacity(output_count);
    for group in groups {
        let value = if matrix_mode {
            let rows = dimensions[normalized[0]];
            let columns = dimensions[normalized[1]];
            let mut matrix = Mat::zeros(rows, columns);
            for (row, column, value) in group {
                matrix[(row, column)] = value;
            }
            matrix_norm(&matrix, order)
        } else {
            let values = group
                .into_iter()
                .map(|(_, _, value)| value)
                .collect::<Vec<_>>();
            vector_norm(&values, order)
        };
        match value {
            Ok(value) => values.push(value),
            Err(status) => return status,
        }
    }
    let output_dtype = if matches!(source.dtype, 11 | 12) {
        11
    } else {
        3
    };
    unsafe { write_result(owned_real(&values, &output_shape, output_dtype), output) }
}

/// Computes an SVD pseudoinverse with an explicit singular-value cutoff.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_pinv(
    array: Array,
    requested_tolerance: f64,
    output: *mut Array,
) -> i32 {
    let (matrix, dimensions, complex, narrow, _) = match unsafe { matrix(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return SHAPE_MISMATCH;
    }
    let factor = match matrix.svd() {
        Ok(value) => value,
        Err(_) => return DOMAIN_ERROR,
    };
    let singular: Vec<f64> = factor
        .S()
        .column_vector()
        .iter()
        .map(|value| value.re)
        .collect();
    let threshold = match tolerance(&matrix, &singular, requested_tolerance) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut result = Mat::zeros(matrix.ncols(), matrix.nrows());
    for index in 0..singular.len() {
        if singular[index] > threshold {
            let reciprocal = 1.0 / singular[index];
            for row in 0..matrix.ncols() {
                for column in 0..matrix.nrows() {
                    result[(row, column)] +=
                        factor.V()[(row, index)] * factor.U()[(column, index)].conj() * reciprocal;
                }
            }
        }
    }
    let result_shape = [dimensions[1], dimensions[0]];
    unsafe {
        write_result(
            owned_numeric(&matrix_values(&result), &result_shape, complex, narrow),
            output,
        )
    }
}

/// Computes determinant sign and logarithmic absolute determinant as scalar arrays.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_slogdet(
    array: Array,
    out_values: *mut *mut u64,
    out_length: *mut usize,
) -> i32 {
    let (matrix, _, complex, narrow, _) = match unsafe { matrix(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let dimension = match square(&matrix) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let factor = matrix.partial_piv_lu();
    let (permutation, _) = factor.P().arrays();
    let mut visited = vec![false; dimension];
    let mut transpositions = 0usize;
    for start in 0..dimension {
        if visited[start] {
            continue;
        }
        let mut length = 0usize;
        let mut index = start;
        while !visited[index] {
            visited[index] = true;
            index = permutation[index];
            length += 1;
        }
        transpositions += length.saturating_sub(1);
    }
    let mut sign = if transpositions % 2 == 0 {
        c64::new(1.0, 0.0)
    } else {
        c64::new(-1.0, 0.0)
    };
    let mut logarithm = 0.0;
    for index in 0..dimension {
        let diagonal = factor.U()[(index, index)];
        let magnitude = diagonal.norm();
        if magnitude == 0.0 {
            sign = c64::new(0.0, 0.0);
            logarithm = f64::NEG_INFINITY;
            break;
        }
        sign *= diagonal / magnitude;
        logarithm += magnitude.ln();
    }
    let sign_array = match unsafe { owned_numeric(&[sign], &[], complex, narrow) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let log_array = match unsafe { owned_real(&[logarithm], &[], if narrow { 11 } else { 3 }) } {
        Ok(value) => value,
        Err(status) => {
            unsafe { delete_arrays(&[sign_array]) };
            return status;
        }
    };
    unsafe { transfer_arrays(vec![sign_array, log_array], out_values, out_length) }
}

/// Computes row-major strides for a checked shape.
fn strides(dimensions: &[i64]) -> Result<Vec<usize>, i32> {
    let mut output = vec![1usize; dimensions.len()];
    for index in (0..dimensions.len().saturating_sub(1)).rev() {
        let next = usize::try_from(dimensions[index + 1]).map_err(|_| SHAPE_MISMATCH)?;
        output[index] = output[index + 1].checked_mul(next).ok_or(SIZE_OVERFLOW)?;
    }
    Ok(output)
}

/// Permutes row-major values according to a complete output-to-input axis order.
fn permuted(values: &[c64], dimensions: &[i64], axes: &[usize]) -> Result<Vec<c64>, i32> {
    if axes.len() != dimensions.len() {
        return Err(SHAPE_MISMATCH);
    }
    let source_strides = strides(dimensions)?;
    let output_shape: Vec<i64> = axes.iter().map(|&axis| dimensions[axis]).collect();
    let output_strides = strides(&output_shape)?;
    let count = element_count(dimensions)?;
    let mut output = vec![c64::new(0.0, 0.0); count];
    for (flat, value) in output.iter_mut().enumerate() {
        let mut remainder = flat;
        let mut source = 0usize;
        for (output_axis, &source_axis) in axes.iter().enumerate() {
            let coordinate = if output_shape[output_axis] == 0 {
                0
            } else {
                remainder / output_strides[output_axis]
            };
            remainder %= output_strides[output_axis];
            source += coordinate * source_strides[source_axis];
        }
        *value = values[source];
    }
    Ok(output)
}

/// Computes the inverse of a tensor interpreted as a square matrix split at `index`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_tensorinv(
    array: Array,
    index: i64,
    output: *mut Array,
) -> i32 {
    let dimensions = match unsafe { shape(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let index = match usize::try_from(index) {
        Ok(value) if value > 0 && value < dimensions.len() => value,
        _ => return INVALID_ARGUMENT,
    };
    let rows = match element_count(&dimensions[..index]) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let columns = match element_count(&dimensions[index..]) {
        Ok(value) if value == rows && value > 0 => value,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let source = match unsafe { numeric(array, true) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let matrix = Mat::from_fn(rows, columns, |row, column| {
        source.values[row * columns + column]
    });
    let singular = match singular_values(&matrix) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let threshold = match tolerance(&matrix, &singular, 0.0) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if singular.last().copied().unwrap_or(0.0) <= threshold {
        return DOMAIN_ERROR;
    }
    let inverse = matrix.partial_piv_lu().inverse();
    let result_shape: Vec<i64> = dimensions[index..]
        .iter()
        .chain(&dimensions[..index])
        .copied()
        .collect();
    unsafe {
        write_result(
            owned_numeric(
                &matrix_values(&inverse),
                &result_shape,
                source.complex,
                matches!(source.dtype, 11 | 12),
            ),
            output,
        )
    }
}

/// Solves a tensor equation after optionally moving selected coefficient axes to the end.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_tensorsolve(
    coefficients: Array,
    rhs: Array,
    axes: *const i64,
    axis_count: i64,
    output: *mut Array,
) -> i32 {
    if axis_count < 0 || (axis_count > 0 && axes.is_null()) {
        return INVALID_ARGUMENT;
    }
    let coefficient_shape = match unsafe { shape(coefficients) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let rhs_shape = match unsafe { shape(rhs) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if rhs_shape.len() >= coefficient_shape.len() {
        return SHAPE_MISMATCH;
    }
    let axis_count = match usize::try_from(axis_count) {
        Ok(value) => value,
        Err(_) => return SIZE_OVERFLOW,
    };
    let requested = if axis_count == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(axes, axis_count) }
    };
    let rank = coefficient_shape.len();
    let mut moved = Vec::with_capacity(axis_count);
    for &axis in requested {
        let normalized = if axis < 0 { axis + rank as i64 } else { axis };
        let normalized = match usize::try_from(normalized) {
            Ok(value) if value < rank && !moved.contains(&value) => value,
            _ => return INVALID_ARGUMENT,
        };
        moved.push(normalized);
    }
    let mut order: Vec<usize> = (0..rank).filter(|axis| !moved.contains(axis)).collect();
    order.extend(&moved);
    let ordered_shape: Vec<i64> = order.iter().map(|&axis| coefficient_shape[axis]).collect();
    let trailing_rank = rank - rhs_shape.len();
    let result_shape = &ordered_shape[rank - trailing_rank..];
    let rows = match element_count(&rhs_shape) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let columns = match element_count(result_shape) {
        Ok(value) if value == rows && value > 0 => value,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let left = match unsafe { numeric(coefficients, true) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let right = match unsafe { numeric(rhs, true) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = match permuted(&left.values, &coefficient_shape, &order) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let matrix = Mat::from_fn(rows, columns, |row, column| values[row * columns + column]);
    let singular = match singular_values(&matrix) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let threshold = match tolerance(&matrix, &singular, 0.0) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if singular.last().copied().unwrap_or(0.0) <= threshold {
        return DOMAIN_ERROR;
    }
    let rhs_matrix = Mat::from_fn(rows, 1, |row, _| right.values[row]);
    let solution = matrix.partial_piv_lu().solve(&rhs_matrix);
    let solution_values = (0..columns)
        .map(|row| solution[(row, 0)])
        .collect::<Vec<_>>();
    unsafe {
        write_result(
            owned_numeric(
                &solution_values,
                result_shape,
                left.complex || right.complex,
                matches!(left.dtype, 11 | 12) && matches!(right.dtype, 11 | 12),
            ),
            output,
        )
    }
}

#[cfg(test)]
#[path = "ndarray_linear_algebra_completion_tests.rs"]
mod tests;
