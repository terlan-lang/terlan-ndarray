//! NumPy-compatible tensor semantics shared by orthogonal polynomial bases.

use rustfft::num_complex::Complex64;

use super::native_support::{
    delete_arrays, element_count, normalize_axis, numeric, owned_numeric, shape, write_result,
    Array, INVALID_ARGUMENT, OK, SHAPE_MISMATCH, SIZE_OVERFLOW,
};
use super::polynomial_basis_support::{self as basis, Basis};
use super::polynomial_extension::narrow_dtype;
use super::polynomial_numpy_completion::fit_selected_with;
use super::polynomial_power_extension::{broadcast_offset, broadcast_shape, narrow, strides};

/// Returns one row-major coefficient offset within a tensor lane.
fn coefficient_offset(outer: usize, coefficient: usize, inner: usize, axis_length: usize) -> usize {
    (outer * axis_length + coefficient) * inner
}

/// Evaluates one coefficient lane at one coordinate.
fn evaluate_lane(
    coefficients: &[Complex64],
    polynomial_count: usize,
    polynomial: usize,
    basis: Basis,
    coordinate: Complex64,
) -> Complex64 {
    let basis_values =
        basis::sequence(basis, coordinate, coefficients.len() / polynomial_count - 1);
    (0..basis_values.len()).fold(Complex64::new(0.0, 0.0), |value, degree| {
        value + coefficients[degree * polynomial_count + polynomial] * basis_values[degree]
    })
}

/// Evaluates coefficient tensors with NumPy's `tensor` flag semantics.
pub(crate) unsafe fn evaluate_tensor(
    coordinates: Array,
    coefficients: Array,
    selected_basis: Basis,
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
    let mut coefficient_shape = match unsafe { shape(coefficients) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if coefficient_shape.is_empty() {
        coefficient_shape.push(1);
    }
    if coefficient_shape[0] <= 0 {
        return SHAPE_MISMATCH;
    }
    let coordinates = match unsafe { numeric(coordinates, true) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let coefficients = match unsafe { numeric(coefficients, true) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let polynomial_shape = &coefficient_shape[1..];
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
                values.push(evaluate_lane(
                    &coefficients.values,
                    polynomial_count,
                    polynomial,
                    selected_basis,
                    *coordinate,
                ));
            }
        }
        (output_shape, values)
    } else {
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
                evaluate_lane(
                    &coefficients.values,
                    polynomial_count,
                    polynomial,
                    selected_basis,
                    coordinates.values[coordinate],
                )
            })
            .collect::<Vec<_>>();
        (output_shape, values)
    };
    unsafe {
        write_result(
            owned_numeric(
                &values,
                &output_shape,
                coordinates.complex || coefficients.complex,
                narrow(&coordinates) && narrow(&coefficients),
            ),
            output,
        )
    }
}

/// Applies power-basis calculus independently to every tensor coefficient lane.
unsafe fn calculus(
    coefficients: Array,
    order: i64,
    constants: Option<Array>,
    lower_bound: f64,
    scale: f64,
    axis: i64,
    selected_basis: Basis,
    output: *mut Array,
) -> i32 {
    if order < 0 || !lower_bound.is_finite() || !scale.is_finite() {
        return INVALID_ARGUMENT;
    }
    let mut dimensions = match unsafe { shape(coefficients) } {
        Ok(value) if !value.is_empty() => value,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let axis = match normalize_axis(axis, dimensions.len()) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let axis_length = match usize::try_from(dimensions[axis]) {
        Ok(value) if value > 0 => value,
        _ => return SHAPE_MISMATCH,
    };
    let outer = match element_count(&dimensions[..axis]) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let inner = match element_count(&dimensions[axis + 1..]) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let coefficients = match unsafe { numeric(coefficients, true) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let lane_count = match outer.checked_mul(inner) {
        Some(value) => value,
        None => return SIZE_OVERFLOW,
    };
    let mut lanes = Vec::with_capacity(lane_count);
    let mut output_axis = None;
    let mut output_complex = coefficients.complex;
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let lane = (0..axis_length)
                .map(|coefficient| {
                    coefficients.values[coefficient_offset(
                        outer_index,
                        coefficient,
                        inner,
                        axis_length,
                    ) + inner_index]
                })
                .collect::<Vec<_>>();
            let lane = match unsafe {
                owned_numeric(
                    &lane,
                    &[axis_length as i64],
                    coefficients.complex,
                    narrow_dtype(coefficients.dtype),
                )
            } {
                Ok(value) => value,
                Err(status) => return status,
            };
            let power = match unsafe { basis::converted(lane, selected_basis.code(), 0) } {
                Ok(value) => value,
                Err(status) => {
                    unsafe { delete_arrays(&[lane]) };
                    return status;
                }
            };
            let mut transformed = std::ptr::null_mut();
            let status = if let Some(constants) = constants {
                unsafe {
                    super::polynomial_numpy_completion::terlan_ndarray_v1_polyint_axis(
                        power,
                        order,
                        constants,
                        lower_bound,
                        scale,
                        0,
                        &mut transformed,
                    )
                }
            } else {
                unsafe {
                    super::polynomial_numpy_completion::terlan_ndarray_v1_polyder_axis(
                        power,
                        order,
                        scale,
                        0,
                        &mut transformed,
                    )
                }
            };
            unsafe { delete_arrays(&[power, lane]) };
            if status != OK {
                return status;
            }
            let converted = match unsafe { basis::converted(transformed, 0, selected_basis.code()) }
            {
                Ok(value) => value,
                Err(status) => {
                    unsafe { delete_arrays(&[transformed]) };
                    return status;
                }
            };
            unsafe { delete_arrays(&[transformed]) };
            let converted_shape = match unsafe { shape(converted) } {
                Ok(value) if value.len() == 1 => value,
                Ok(_) => {
                    unsafe { delete_arrays(&[converted]) };
                    return SHAPE_MISMATCH;
                }
                Err(status) => {
                    unsafe { delete_arrays(&[converted]) };
                    return status;
                }
            };
            let values = match unsafe { numeric(converted, false) } {
                Ok(value) => value,
                Err(status) => {
                    unsafe { delete_arrays(&[converted]) };
                    return status;
                }
            };
            unsafe { delete_arrays(&[converted]) };
            let lane_axis = converted_shape[0] as usize;
            if output_axis.is_some_and(|value| value != lane_axis) {
                return SHAPE_MISMATCH;
            }
            output_axis = Some(lane_axis);
            output_complex |= values.complex;
            lanes.push(values.values);
        }
    }
    let output_axis = output_axis.unwrap_or_else(|| {
        if constants.is_some() {
            axis_length.saturating_add(order as usize)
        } else {
            axis_length.saturating_sub(order as usize).max(1)
        }
    });
    dimensions[axis] = output_axis as i64;
    let count = match outer
        .checked_mul(output_axis)
        .and_then(|value| value.checked_mul(inner))
    {
        Some(value) => value,
        None => return SIZE_OVERFLOW,
    };
    let mut values = vec![Complex64::new(0.0, 0.0); count];
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let lane = &lanes[outer_index * inner + inner_index];
            for coefficient in 0..output_axis {
                values[coefficient_offset(outer_index, coefficient, inner, output_axis)
                    + inner_index] = lane[coefficient];
            }
        }
    }
    unsafe {
        write_result(
            owned_numeric(
                &values,
                &dimensions,
                output_complex,
                narrow_dtype(coefficients.dtype),
            ),
            output,
        )
    }
}

/// Fits selected orthogonal-basis degrees with optional weights.
unsafe fn fit(
    coordinates: Array,
    values: Array,
    degrees: Array,
    weights: Option<Array>,
    selected_basis: Basis,
    output: *mut Array,
) -> i32 {
    unsafe {
        fit_selected_with(
            coordinates,
            values,
            degrees,
            weights,
            |coordinate, maximum| basis::sequence(selected_basis, coordinate, maximum),
            output,
        )
    }
}

macro_rules! basis_completion {
    ($prefix:ident, $basis:expr, $derivative:ident, $integral:ident, $evaluate:ident, $fit:ident, $weighted_fit:ident) => {
        #[doc = concat!("Differentiates ", stringify!($prefix), " coefficient tensors along one axis.")]
        #[no_mangle]
        pub unsafe extern "C" fn $derivative(
            coefficients: Array,
            order: i64,
            scale: f64,
            axis: i64,
            output: *mut Array,
        ) -> i32 {
            unsafe { calculus(coefficients, order, None, 0.0, scale, axis, $basis, output) }
        }

        #[doc = concat!("Integrates ", stringify!($prefix), " coefficient tensors along one axis.")]
        #[no_mangle]
        pub unsafe extern "C" fn $integral(
            coefficients: Array,
            order: i64,
            constants: Array,
            lower_bound: f64,
            scale: f64,
            axis: i64,
            output: *mut Array,
        ) -> i32 {
            unsafe {
                calculus(
                    coefficients,
                    order,
                    Some(constants),
                    lower_bound,
                    scale,
                    axis,
                    $basis,
                    output,
                )
            }
        }

        #[doc = concat!("Evaluates ", stringify!($prefix), " coefficient tensors.")]
        #[no_mangle]
        pub unsafe extern "C" fn $evaluate(
            coordinates: Array,
            coefficients: Array,
            tensor: i32,
            output: *mut Array,
        ) -> i32 {
            unsafe { evaluate_tensor(coordinates, coefficients, $basis, tensor, output) }
        }

        #[doc = concat!("Fits selected ", stringify!($prefix), " degrees.")]
        #[no_mangle]
        pub unsafe extern "C" fn $fit(
            coordinates: Array,
            values: Array,
            degrees: Array,
            output: *mut Array,
        ) -> i32 {
            unsafe { fit(coordinates, values, degrees, None, $basis, output) }
        }

        #[doc = concat!("Fits selected weighted ", stringify!($prefix), " degrees.")]
        #[no_mangle]
        pub unsafe extern "C" fn $weighted_fit(
            coordinates: Array,
            values: Array,
            degrees: Array,
            weights: Array,
            output: *mut Array,
        ) -> i32 {
            unsafe { fit(coordinates, values, degrees, Some(weights), $basis, output) }
        }
    };
}

basis_completion!(
    Chebyshev,
    Basis::Chebyshev,
    terlan_ndarray_v1_chebder_axis,
    terlan_ndarray_v1_chebint_axis,
    terlan_ndarray_v1_chebval_tensor,
    terlan_ndarray_v1_chebfit_degrees,
    terlan_ndarray_v1_chebfit_weighted_degrees
);

basis_completion!(
    Legendre,
    Basis::Legendre,
    terlan_ndarray_v1_legder_axis,
    terlan_ndarray_v1_legint_axis,
    terlan_ndarray_v1_legval_tensor,
    terlan_ndarray_v1_legfit_degrees,
    terlan_ndarray_v1_legfit_weighted_degrees
);

#[cfg(test)]
#[path = "ndarray_polynomial_basis_numpy_completion_tests.rs"]
mod tests;
