//! NumPy-compatible tensor calculus, root evaluation, and polynomial fitting.

use faer::Mat;
use rustfft::num_complex::Complex64;

use super::native_support::{
    delete_arrays, dtype, element_count, ints, normalize_axis, numeric, owned_numeric, shape,
    transfer_arrays, write_result, Array, NumericValues, INVALID_ARGUMENT, SHAPE_MISMATCH,
    SIZE_OVERFLOW, UNSUPPORTED_DTYPE,
};
use super::polynomial_extension::{least_squares, narrow_dtype};
use super::polynomial_power_extension::{broadcast_offset, broadcast_shape, strides};

/// Returns checked row-major calculus dimensions around one coefficient axis.
fn calculus_layout(dimensions: &[i64], axis: i64) -> Result<(usize, usize, usize, usize), i32> {
    let axis = normalize_axis(axis, dimensions.len())?;
    let axis_length = usize::try_from(dimensions[axis]).map_err(|_| SHAPE_MISMATCH)?;
    if axis_length == 0 {
        return Err(SHAPE_MISMATCH);
    }
    let outer = element_count(&dimensions[..axis])?;
    let inner = element_count(&dimensions[axis + 1..])?;
    Ok((axis, axis_length, outer, inner))
}

/// Returns one row-major coefficient offset within a tensor calculus lane.
fn coefficient_offset(outer: usize, coefficient: usize, inner: usize, axis_length: usize) -> usize {
    (outer * axis_length + coefficient) * inner
}

/// Reads zero or more scalar integration constants.
unsafe fn integration_constants(array: Array, order: usize) -> Result<NumericValues, i32> {
    let dimensions = unsafe { shape(array)? };
    if dimensions.len() > 1 {
        return Err(SHAPE_MISMATCH);
    }
    let values = unsafe { numeric(array, true)? };
    if values.values.len() > order {
        return Err(INVALID_ARGUMENT);
    }
    Ok(values)
}

/// Solves one complex least-squares system through a real block expansion.
pub(crate) fn least_squares_complex(
    matrix: &[Complex64],
    rows: usize,
    columns: usize,
    rhs: &[Complex64],
) -> Result<Vec<Complex64>, i32> {
    if matrix.len() != rows.saturating_mul(columns) || rhs.len() != rows {
        return Err(SHAPE_MISMATCH);
    }
    let expanded = Mat::from_fn(rows * 2, columns * 2, |row, column| {
        let source_row = row % rows;
        let source_column = column % columns;
        let value = matrix[source_row * columns + source_column];
        match (row < rows, column < columns) {
            (true, true) | (false, false) => value.re,
            (true, false) => -value.im,
            (false, true) => value.im,
        }
    });
    let expanded_rhs = rhs
        .iter()
        .map(|value| value.re)
        .chain(rhs.iter().map(|value| value.im))
        .collect::<Vec<_>>();
    let solution = least_squares(&expanded, &expanded_rhs)?;
    Ok((0..columns)
        .map(|column| Complex64::new(solution[column], solution[column + columns]))
        .collect())
}

/// Differentiates tensor coefficients along one axis with linear-variable scaling.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyder_axis(
    coefficients: Array,
    order: i64,
    scale: f64,
    axis: i64,
    output: *mut Array,
) -> i32 {
    if order < 0 || !scale.is_finite() {
        return INVALID_ARGUMENT;
    }
    let dimensions = match unsafe { shape(coefficients) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let (axis, axis_length, outer, inner) = match calculus_layout(&dimensions, axis) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let coefficients = match unsafe { numeric(coefficients, true) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let order = order as usize;
    let output_axis = if order >= axis_length {
        1
    } else {
        axis_length - order
    };
    let output_count = match outer
        .checked_mul(output_axis)
        .and_then(|value| value.checked_mul(inner))
    {
        Some(value) => value,
        None => return SIZE_OVERFLOW,
    };
    let mut values = vec![Complex64::new(0.0, 0.0); output_count];
    if order < axis_length {
        for outer_index in 0..outer {
            for coefficient in 0..output_axis {
                let factor = ((coefficient + 1)..=(coefficient + order))
                    .fold(scale.powi(order as i32), |value, degree| {
                        value * degree as f64
                    });
                let source =
                    coefficient_offset(outer_index, coefficient + order, inner, axis_length);
                let destination = coefficient_offset(outer_index, coefficient, inner, output_axis);
                for inner_index in 0..inner {
                    values[destination + inner_index] =
                        coefficients.values[source + inner_index] * factor;
                }
            }
        }
    }
    let mut output_shape = dimensions;
    output_shape[axis] = output_axis as i64;
    unsafe {
        write_result(
            owned_numeric(
                &values,
                &output_shape,
                coefficients.complex,
                narrow_dtype(coefficients.dtype),
            ),
            output,
        )
    }
}

/// Integrates tensor coefficients along one axis with explicit NumPy constants.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyint_axis(
    coefficients: Array,
    order: i64,
    constants: Array,
    lower_bound: f64,
    scale: f64,
    axis: i64,
    output: *mut Array,
) -> i32 {
    if order < 0 || !lower_bound.is_finite() || !scale.is_finite() {
        return INVALID_ARGUMENT;
    }
    let mut dimensions = match unsafe { shape(coefficients) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let (axis, mut axis_length, outer, inner) = match calculus_layout(&dimensions, axis) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let coefficients = match unsafe { numeric(coefficients, true) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let order = order as usize;
    let constants = match unsafe { integration_constants(constants, order) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut values = coefficients.values;
    for integration in 0..order {
        let constant = constants
            .values
            .get(integration)
            .copied()
            .unwrap_or_else(|| Complex64::new(0.0, 0.0));
        for value in &mut values {
            *value *= scale;
        }
        let all_constant_terms_zero = (0..outer).all(|outer_index| {
            let offset = coefficient_offset(outer_index, 0, inner, axis_length);
            values[offset..offset + inner]
                .iter()
                .all(|value| *value == Complex64::new(0.0, 0.0))
        });
        if axis_length == 1 && all_constant_terms_zero {
            for outer_index in 0..outer {
                let offset = coefficient_offset(outer_index, 0, inner, axis_length);
                for inner_index in 0..inner {
                    values[offset + inner_index] += constant;
                }
            }
            continue;
        }
        let next_axis = match axis_length.checked_add(1) {
            Some(value) => value,
            None => return SIZE_OVERFLOW,
        };
        let count = match outer
            .checked_mul(next_axis)
            .and_then(|value| value.checked_mul(inner))
        {
            Some(value) => value,
            None => return SIZE_OVERFLOW,
        };
        let mut integrated = vec![Complex64::new(0.0, 0.0); count];
        for outer_index in 0..outer {
            for coefficient in 0..axis_length {
                let source = coefficient_offset(outer_index, coefficient, inner, axis_length);
                let destination =
                    coefficient_offset(outer_index, coefficient + 1, inner, next_axis);
                for inner_index in 0..inner {
                    integrated[destination + inner_index] =
                        values[source + inner_index] / (coefficient + 1) as f64;
                }
            }
            for inner_index in 0..inner {
                let base = coefficient_offset(outer_index, 0, inner, next_axis) + inner_index;
                let evaluated =
                    (1..next_axis)
                        .rev()
                        .fold(Complex64::new(0.0, 0.0), |value, coefficient| {
                            value * lower_bound
                                + integrated[coefficient_offset(
                                    outer_index,
                                    coefficient,
                                    inner,
                                    next_axis,
                                ) + inner_index]
                        })
                        * lower_bound;
                integrated[base] = constant - evaluated;
            }
        }
        values = integrated;
        axis_length = next_axis;
    }
    dimensions[axis] = axis_length as i64;
    unsafe {
        write_result(
            owned_numeric(
                &values,
                &dimensions,
                coefficients.complex || constants.complex,
                narrow_dtype(coefficients.dtype),
            ),
            output,
        )
    }
}

/// Evaluates one or more root-defined polynomials with NumPy tensor semantics.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyvalfromroots_tensor(
    coordinates: Array,
    roots: Array,
    tensor: i32,
    output: *mut Array,
) -> i32 {
    if !matches!(tensor, 0 | 1) {
        return INVALID_ARGUMENT;
    }
    let coordinate_shape = match unsafe { shape(coordinates) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut root_shape = match unsafe { shape(roots) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let coordinates = match unsafe { numeric(coordinates, true) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let roots = match unsafe { numeric(roots, true) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if root_shape.is_empty() {
        root_shape.push(1);
    }
    let root_count = match usize::try_from(root_shape[0]) {
        Ok(value) => value,
        Err(_) => return SHAPE_MISMATCH,
    };
    let polynomial_shape = &root_shape[1..];
    let polynomial_count = match element_count(polynomial_shape) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let (output_shape, values) = if tensor == 1 {
        let output_shape = polynomial_shape
            .iter()
            .chain(&coordinate_shape)
            .copied()
            .collect::<Vec<_>>();
        let mut values =
            Vec::with_capacity(polynomial_count.saturating_mul(coordinates.values.len()));
        for polynomial in 0..polynomial_count {
            for coordinate in &coordinates.values {
                let value = (0..root_count).fold(Complex64::new(1.0, 0.0), |value, root| {
                    value * (coordinate - roots.values[root * polynomial_count + polynomial])
                });
                values.push(value);
            }
        }
        (output_shape, values)
    } else {
        if coordinate_shape.len() >= root_shape.len() {
            return SHAPE_MISMATCH;
        }
        let output_shape = match broadcast_shape(&[polynomial_shape, &coordinate_shape]) {
            Ok(value) => value,
            Err(status) => return status,
        };
        let output_count = match element_count(&output_shape) {
            Ok(value) => value,
            Err(status) => return status,
        };
        let output_strides = match strides(&output_shape) {
            Ok(value) => value,
            Err(status) => return status,
        };
        let polynomial_strides = match strides(polynomial_shape) {
            Ok(value) => value,
            Err(status) => return status,
        };
        let coordinate_strides = match strides(&coordinate_shape) {
            Ok(value) => value,
            Err(status) => return status,
        };
        let values = (0..output_count)
            .map(|flat| {
                let polynomial = broadcast_offset(
                    flat,
                    &output_shape,
                    &output_strides,
                    polynomial_shape,
                    &polynomial_strides,
                );
                let coordinate = broadcast_offset(
                    flat,
                    &output_shape,
                    &output_strides,
                    &coordinate_shape,
                    &coordinate_strides,
                );
                (0..root_count).fold(Complex64::new(1.0, 0.0), |value, root| {
                    value
                        * (coordinates.values[coordinate]
                            - roots.values[root * polynomial_count + polynomial])
                })
            })
            .collect::<Vec<_>>();
        (output_shape, values)
    };
    unsafe {
        write_result(
            owned_numeric(
                &values,
                &output_shape,
                coordinates.complex || roots.complex,
                narrow_dtype(coordinates.dtype) && narrow_dtype(roots.dtype),
            ),
            output,
        )
    }
}

/// Implements selected-degree fitting with one reusable basis sequence.
pub(crate) unsafe fn fit_selected_with<F>(
    coordinates: Array,
    values: Array,
    degrees: Array,
    weights: Option<Array>,
    sequence: F,
    output: *mut Array,
) -> i32
where
    F: Fn(Complex64, usize) -> Vec<Complex64>,
{
    let coordinate_shape = match unsafe { shape(coordinates) } {
        Ok(value) if value.len() == 1 && value[0] > 0 => value,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let value_shape = match unsafe { shape(values) } {
        Ok(value) if matches!(value.len(), 1 | 2) && value.first() == coordinate_shape.first() => {
            value
        }
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    match unsafe { shape(degrees) } {
        Ok(value) if value.len() == 1 && value[0] > 0 => {}
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    }
    match unsafe { dtype(degrees) } {
        Ok(value) if matches!(value, 2 | 4..=10) => {}
        Ok(_) => return UNSUPPORTED_DTYPE,
        Err(status) => return status,
    }
    let mut degrees = match unsafe { ints(degrees) } {
        Ok(value) if value.iter().all(|degree| *degree >= 0) => value,
        Ok(_) => return INVALID_ARGUMENT,
        Err(status) => return status,
    };
    degrees.sort_unstable();
    if degrees.windows(2).any(|pair| pair[0] == pair[1]) {
        return INVALID_ARGUMENT;
    }
    let coordinates = match unsafe { numeric(coordinates, true) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = match unsafe { numeric(values, true) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let observation_count = coordinates.values.len();
    let output_columns = if value_shape.len() == 1 {
        1
    } else {
        value_shape[1] as usize
    };
    let weights = if let Some(weights) = weights {
        let weight_shape = match unsafe { shape(weights) } {
            Ok(value) if value == coordinate_shape => value,
            Ok(_) => return SHAPE_MISMATCH,
            Err(status) => return status,
        };
        let _ = weight_shape;
        match unsafe { numeric(weights, true) } {
            Ok(value) if !value.complex => value.values.iter().map(|value| value.re).collect(),
            Ok(_) => return UNSUPPORTED_DTYPE,
            Err(status) => return status,
        }
    } else {
        vec![1.0; observation_count]
    };
    if observation_count < degrees.len() {
        return SHAPE_MISMATCH;
    }
    let maximum_degree = *degrees.last().unwrap() as usize;
    let mut design = Vec::with_capacity(observation_count.saturating_mul(degrees.len()));
    for (row, coordinate) in coordinates.values.iter().enumerate() {
        let basis_values = sequence(*coordinate, maximum_degree);
        for degree in &degrees {
            design.push(basis_values[*degree as usize] * weights[row]);
        }
    }
    let mut scales = vec![1.0; degrees.len()];
    for column in 0..degrees.len() {
        let norm = (0..observation_count)
            .map(|row| design[row * degrees.len() + column].norm_sqr())
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            scales[column] = norm;
            for row in 0..observation_count {
                design[row * degrees.len() + column] /= norm;
            }
        }
    }
    let mut coefficients =
        vec![Complex64::new(0.0, 0.0); (maximum_degree + 1).saturating_mul(output_columns)];
    for output_column in 0..output_columns {
        let rhs = (0..observation_count)
            .map(|row| values.values[row * output_columns + output_column] * weights[row])
            .collect::<Vec<_>>();
        let solution = match least_squares_complex(&design, observation_count, degrees.len(), &rhs)
        {
            Ok(value) => value,
            Err(status) => return status,
        };
        for (column, degree) in degrees.iter().enumerate() {
            coefficients[*degree as usize * output_columns + output_column] =
                solution[column] / scales[column];
        }
    }
    let output_shape = if value_shape.len() == 1 {
        vec![(maximum_degree + 1) as i64]
    } else {
        vec![(maximum_degree + 1) as i64, output_columns as i64]
    };
    unsafe {
        write_result(
            owned_numeric(
                &coefficients,
                &output_shape,
                coordinates.complex || values.complex,
                false,
            ),
            output,
        )
    }
}

/// Fits selected power-basis degrees without observation weights.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyfit_degrees(
    coordinates: Array,
    values: Array,
    degrees: Array,
    output: *mut Array,
) -> i32 {
    unsafe {
        fit_selected_with(
            coordinates,
            values,
            degrees,
            None,
            |coordinate, maximum| {
                let mut values = Vec::with_capacity(maximum + 1);
                let mut value = Complex64::new(1.0, 0.0);
                for _ in 0..=maximum {
                    values.push(value);
                    value *= coordinate;
                }
                values
            },
            output,
        )
    }
}

/// Fits selected power-basis degrees with one weight per observation.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyfit_weighted_degrees(
    coordinates: Array,
    values: Array,
    degrees: Array,
    weights: Array,
    output: *mut Array,
) -> i32 {
    unsafe {
        fit_selected_with(
            coordinates,
            values,
            degrees,
            Some(weights),
            |coordinate, maximum| {
                let mut values = Vec::with_capacity(maximum + 1);
                let mut value = Complex64::new(1.0, 0.0);
                for _ in 0..=maximum {
                    values.push(value);
                    value *= coordinate;
                }
                values
            },
            output,
        )
    }
}

/// Fits contiguous power degrees and returns coefficients plus covariance.
unsafe fn fit_covariance(
    coordinates: Array,
    observations: Array,
    requested_degree: i64,
    weights: Option<Array>,
    scale_covariance: i32,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let degree = match usize::try_from(requested_degree) {
        Ok(value) => value,
        Err(_) => return INVALID_ARGUMENT,
    };
    if !matches!(scale_covariance, 0 | 1) {
        return INVALID_ARGUMENT;
    }
    let coordinate_shape = match unsafe { shape(coordinates) } {
        Ok(value) if value.len() == 1 && value[0] > 0 => value,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let observation_count = coordinate_shape[0] as usize;
    let column_count = match degree.checked_add(1) {
        Some(value) => value,
        None => return SIZE_OVERFLOW,
    };
    if observation_count < column_count
        || (scale_covariance == 1 && observation_count <= column_count)
    {
        return SHAPE_MISMATCH;
    }
    let coordinates = match unsafe { numeric(coordinates, true) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let observation_shape = match unsafe { shape(observations) } {
        Ok(value)
            if matches!(value.as_slice(), [rows] if *rows == coordinate_shape[0])
                || matches!(value.as_slice(), [rows, columns] if *rows == coordinate_shape[0] && *columns > 0) =>
        {
            value
        }
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let observations = match unsafe { numeric(observations, true) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let output_columns = if observation_shape.len() == 1 {
        1
    } else {
        observation_shape[1] as usize
    };
    let weights = if let Some(weights) = weights {
        match unsafe { shape(weights) } {
            Ok(value) if value == coordinate_shape => {}
            Ok(_) => return SHAPE_MISMATCH,
            Err(status) => return status,
        }
        match unsafe { numeric(weights, true) } {
            Ok(value)
                if !value.complex && value.values.iter().all(|value| value.re.is_finite()) =>
            {
                value
                    .values
                    .iter()
                    .map(|value| value.re)
                    .collect::<Vec<_>>()
            }
            Ok(_) => return UNSUPPORTED_DTYPE,
            Err(status) => return status,
        }
    } else {
        vec![1.0; observation_count]
    };

    let mut design = Vec::with_capacity(observation_count.saturating_mul(column_count));
    for (row, coordinate) in coordinates.values.iter().enumerate() {
        let mut power = Complex64::new(1.0, 0.0);
        for _ in 0..column_count {
            design.push(power * weights[row]);
            power *= coordinate;
        }
    }
    let mut scales = vec![1.0; column_count];
    for column in 0..column_count {
        let norm = (0..observation_count)
            .map(|row| design[row * column_count + column].norm_sqr())
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return super::native_support::DOMAIN_ERROR;
        }
        scales[column] = norm;
        for row in 0..observation_count {
            design[row * column_count + column] /= norm;
        }
    }

    let mut coefficients =
        vec![Complex64::new(0.0, 0.0); column_count.saturating_mul(output_columns)];
    let mut residual_scales = vec![1.0; output_columns];
    for output_column in 0..output_columns {
        let rhs = (0..observation_count)
            .map(|row| observations.values[row * output_columns + output_column] * weights[row])
            .collect::<Vec<_>>();
        let solution = match least_squares_complex(&design, observation_count, column_count, &rhs) {
            Ok(value) => value,
            Err(status) => return status,
        };
        for column in 0..column_count {
            coefficients[column * output_columns + output_column] =
                solution[column] / scales[column];
        }
        if scale_covariance == 1 {
            let residual = (0..observation_count)
                .map(|row| {
                    let coordinate = coordinates.values[row];
                    let prediction =
                        (0..column_count)
                            .rev()
                            .fold(Complex64::new(0.0, 0.0), |value, column| {
                                value * coordinate
                                    + coefficients[column * output_columns + output_column]
                            });
                    ((prediction - observations.values[row * output_columns + output_column])
                        * weights[row])
                        .norm_sqr()
                })
                .sum::<f64>();
            residual_scales[output_column] = residual / (observation_count - column_count) as f64;
        }
    }

    let gram = (0..column_count)
        .flat_map(|row| {
            let design = &design;
            (0..column_count).map(move |column| {
                (0..observation_count)
                    .map(|sample| {
                        design[sample * column_count + row].conj()
                            * design[sample * column_count + column]
                    })
                    .sum::<Complex64>()
            })
        })
        .collect::<Vec<_>>();
    let mut inverse = vec![Complex64::new(0.0, 0.0); column_count * column_count];
    for column in 0..column_count {
        let mut unit = vec![Complex64::new(0.0, 0.0); column_count];
        unit[column] = Complex64::new(1.0, 0.0);
        let solution = match least_squares_complex(&gram, column_count, column_count, &unit) {
            Ok(value) => value,
            Err(status) => return status,
        };
        for row in 0..column_count {
            inverse[row * column_count + column] = solution[row] / (scales[row] * scales[column]);
        }
    }
    let covariance = (0..column_count)
        .flat_map(|row| {
            let inverse = &inverse;
            let residual_scales = &residual_scales;
            (0..column_count).flat_map(move |column| {
                (0..output_columns).map(move |output_column| {
                    inverse[row * column_count + column] * residual_scales[output_column]
                })
            })
        })
        .collect::<Vec<_>>();
    let coefficient_shape = if observation_shape.len() == 1 {
        vec![column_count as i64]
    } else {
        vec![column_count as i64, output_columns as i64]
    };
    let covariance_shape = if observation_shape.len() == 1 {
        vec![column_count as i64, column_count as i64]
    } else {
        vec![
            column_count as i64,
            column_count as i64,
            output_columns as i64,
        ]
    };
    let complex = coordinates.complex || observations.complex;
    let coefficient_array =
        match unsafe { owned_numeric(&coefficients, &coefficient_shape, complex, false) } {
            Ok(value) => value,
            Err(status) => return status,
        };
    let covariance_array =
        match unsafe { owned_numeric(&covariance, &covariance_shape, complex, false) } {
            Ok(value) => value,
            Err(status) => {
                unsafe { delete_arrays(&[coefficient_array]) };
                return status;
            }
        };
    unsafe {
        transfer_arrays(
            vec![coefficient_array, covariance_array],
            output,
            output_length,
        )
    }
}

/// Fits contiguous power degrees and returns coefficients plus scaled or unscaled covariance.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyfit_covariance(
    coordinates: Array,
    observations: Array,
    degree: i64,
    scale_covariance: i32,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    unsafe {
        fit_covariance(
            coordinates,
            observations,
            degree,
            None,
            scale_covariance,
            output,
            output_length,
        )
    }
}

/// Fits weighted contiguous power degrees and returns coefficients plus covariance.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyfit_weighted_covariance(
    coordinates: Array,
    observations: Array,
    degree: i64,
    weights: Array,
    scale_covariance: i32,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    unsafe {
        fit_covariance(
            coordinates,
            observations,
            degree,
            Some(weights),
            scale_covariance,
            output,
            output_length,
        )
    }
}

#[cfg(test)]
#[path = "ndarray_polynomial_numpy_completion_tests.rs"]
mod tests;
