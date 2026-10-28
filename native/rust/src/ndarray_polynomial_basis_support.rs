//! Shared Chebyshev and Legendre polynomial implementation.

use faer::{Mat, Side};
use rustfft::num_complex::Complex64;

use super::native_support::{
    delete_arrays, element_count, numeric, owned_numeric, owned_real, shape, transfer_arrays,
    write_result, Array, DOMAIN_ERROR, INVALID_ARGUMENT, OK, SHAPE_MISMATCH, SIZE_OVERFLOW,
};
use super::polynomial_extension::{narrow_dtype, numeric_vector};
use super::polynomial_power_extension::{
    broadcast_offset, broadcast_shape, degree, narrow, shaped, strides,
};

/// Supported orthogonal polynomial basis.
#[derive(Clone, Copy)]
pub(crate) enum Basis {
    /// Chebyshev basis of the first kind.
    Chebyshev,
    /// Legendre basis.
    Legendre,
}

impl Basis {
    /// Returns the package basis code accepted by `polyconvert`.
    pub(crate) fn code(self) -> i32 {
        match self {
            Self::Chebyshev => 1,
            Self::Legendre => 2,
        }
    }
}

/// Converts one coefficient array and returns an owned result.
pub(crate) unsafe fn converted(array: Array, source: i32, target: i32) -> Result<Array, i32> {
    let mut output = std::ptr::null_mut();
    let status = unsafe {
        super::polynomial_extension::terlan_ndarray_v1_polyconvert(
            array,
            source,
            target,
            -1.0,
            1.0,
            -1.0,
            1.0,
            &mut output,
        )
    };
    if status == OK {
        Ok(output)
    } else {
        Err(status)
    }
}

/// Converts one coefficient array between the power and selected basis.
pub(crate) unsafe fn convert(
    array: Array,
    source: Basis,
    to_power: bool,
    output: *mut Array,
) -> i32 {
    let (from, to) = if to_power {
        (source.code(), 0)
    } else {
        (0, source.code())
    };
    unsafe { write_result(converted(array, from, to), output) }
}

/// Invokes one unary power-basis operation around basis conversions.
unsafe fn unary_power(
    coefficients: Array,
    basis: Basis,
    operation: impl FnOnce(Array, *mut Array) -> i32,
    output: *mut Array,
) -> i32 {
    let power = match unsafe { converted(coefficients, basis.code(), 0) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut power_result = std::ptr::null_mut();
    let status = operation(power, &mut power_result);
    unsafe { delete_arrays(&[power]) };
    if status != OK {
        return status;
    }
    let result = unsafe { converted(power_result, 0, basis.code()) };
    unsafe { delete_arrays(&[power_result]) };
    unsafe { write_result(result, output) }
}

/// Invokes one binary power-basis operation around basis conversions.
unsafe fn binary_power(
    lhs: Array,
    rhs: Array,
    basis: Basis,
    operation: impl FnOnce(Array, Array, *mut Array) -> i32,
    output: *mut Array,
) -> i32 {
    let lhs_power = match unsafe { converted(lhs, basis.code(), 0) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let rhs_power = match unsafe { converted(rhs, basis.code(), 0) } {
        Ok(value) => value,
        Err(status) => {
            unsafe { delete_arrays(&[lhs_power]) };
            return status;
        }
    };
    let mut power_result = std::ptr::null_mut();
    let status = operation(lhs_power, rhs_power, &mut power_result);
    unsafe { delete_arrays(&[lhs_power, rhs_power]) };
    if status != OK {
        return status;
    }
    let result = unsafe { converted(power_result, 0, basis.code()) };
    unsafe { delete_arrays(&[power_result]) };
    unsafe { write_result(result, output) }
}

/// Adds basis coefficients degree by degree.
pub(crate) unsafe fn add(lhs: Array, rhs: Array, output: *mut Array) -> i32 {
    unsafe { super::polynomial_extension::terlan_ndarray_v1_polyadd(lhs, rhs, output) }
}

/// Subtracts basis coefficients degree by degree.
pub(crate) unsafe fn subtract(lhs: Array, rhs: Array, output: *mut Array) -> i32 {
    unsafe { super::polynomial_extension::terlan_ndarray_v1_polysub(lhs, rhs, output) }
}

/// Multiplies two coefficient vectors in one orthogonal basis.
pub(crate) unsafe fn multiply(lhs: Array, rhs: Array, basis: Basis, output: *mut Array) -> i32 {
    unsafe {
        binary_power(
            lhs,
            rhs,
            basis,
            |a, b, out| super::polynomial_extension::terlan_ndarray_v1_polymul(a, b, out),
            output,
        )
    }
}

/// Divides two coefficient vectors and transfers quotient and remainder.
pub(crate) unsafe fn divide(
    lhs: Array,
    rhs: Array,
    basis: Basis,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let lhs_power = match unsafe { converted(lhs, basis.code(), 0) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let rhs_power = match unsafe { converted(rhs, basis.code(), 0) } {
        Ok(value) => value,
        Err(status) => {
            unsafe { delete_arrays(&[lhs_power]) };
            return status;
        }
    };
    let mut identities = std::ptr::null_mut();
    let mut length = 0usize;
    let status = unsafe {
        super::polynomial_extension::terlan_ndarray_v1_polydiv_all(
            lhs_power,
            rhs_power,
            &mut identities,
            &mut length,
        )
    };
    unsafe { delete_arrays(&[lhs_power, rhs_power]) };
    if status != OK {
        return status;
    }
    let powers = unsafe { std::slice::from_raw_parts(identities, length) }
        .iter()
        .map(|&identity| identity as usize as Array)
        .collect::<Vec<_>>();
    unsafe { crate::ffi::terlan_ndarray_v1_delete_handle_values(identities) };
    let mut results = Vec::with_capacity(powers.len());
    for power in &powers {
        match unsafe { converted(*power, 0, basis.code()) } {
            Ok(value) => results.push(value),
            Err(status) => {
                unsafe { delete_arrays(&powers) };
                unsafe { delete_arrays(&results) };
                return status;
            }
        }
    }
    unsafe { delete_arrays(&powers) };
    unsafe { transfer_arrays(results, output, output_length) }
}

/// Multiplies one coefficient vector by `x` in the selected basis.
pub(crate) unsafe fn multiply_x(coefficients: Array, basis: Basis, output: *mut Array) -> i32 {
    unsafe {
        unary_power(
            coefficients,
            basis,
            |value, out| super::polynomial_power_extension::terlan_ndarray_v1_polymulx(value, out),
            output,
        )
    }
}

/// Raises one coefficient vector to a nonnegative power.
pub(crate) unsafe fn power(
    coefficients: Array,
    exponent: i64,
    basis: Basis,
    output: *mut Array,
) -> i32 {
    unsafe {
        unary_power(
            coefficients,
            basis,
            |value, out| {
                super::polynomial_power_extension::terlan_ndarray_v1_polypow(value, exponent, out)
            },
            output,
        )
    }
}

/// Differentiates one coefficient vector by a nonnegative order.
pub(crate) unsafe fn derivative(
    coefficients: Array,
    order: i64,
    basis: Basis,
    output: *mut Array,
) -> i32 {
    unsafe {
        unary_power(
            coefficients,
            basis,
            |value, out| super::polynomial_extension::terlan_ndarray_v1_polyder(value, order, out),
            output,
        )
    }
}

/// Integrates one coefficient vector by a nonnegative order.
pub(crate) unsafe fn integral(
    coefficients: Array,
    order: i64,
    constant: f64,
    basis: Basis,
    output: *mut Array,
) -> i32 {
    unsafe {
        unary_power(
            coefficients,
            basis,
            |value, out| {
                super::polynomial_extension::terlan_ndarray_v1_polyint(value, order, constant, out)
            },
            output,
        )
    }
}

/// Fits coefficients in one orthogonal basis.
pub(crate) unsafe fn fit(
    coordinates: Array,
    values: Array,
    requested_degree: i64,
    basis: Basis,
    output: *mut Array,
) -> i32 {
    let mut power = std::ptr::null_mut();
    let status = unsafe {
        super::polynomial_extension::terlan_ndarray_v1_polyfit(
            coordinates,
            values,
            requested_degree,
            &mut power,
        )
    };
    if status != OK {
        return status;
    }
    let result = unsafe { converted(power, 0, basis.code()) };
    unsafe { delete_arrays(&[power]) };
    unsafe { write_result(result, output) }
}

/// Builds basis coefficients from roots.
pub(crate) unsafe fn from_roots(roots: Array, basis: Basis, output: *mut Array) -> i32 {
    let mut power = std::ptr::null_mut();
    let status = unsafe {
        super::polynomial_power_extension::terlan_ndarray_v1_polyfromroots(roots, &mut power)
    };
    if status != OK {
        return status;
    }
    let result = unsafe { converted(power, 0, basis.code()) };
    unsafe { delete_arrays(&[power]) };
    unsafe { write_result(result, output) }
}

/// Returns a roots-compatible companion matrix for basis coefficients.
pub(crate) unsafe fn companion(coefficients: Array, basis: Basis, output: *mut Array) -> i32 {
    let power = match unsafe { converted(coefficients, basis.code(), 0) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let status =
        unsafe { super::polynomial_extension::terlan_ndarray_v1_polycompanion(power, output) };
    unsafe { delete_arrays(&[power]) };
    status
}

/// Returns roots for basis coefficients.
pub(crate) unsafe fn roots(coefficients: Array, basis: Basis, output: *mut Array) -> i32 {
    let power = match unsafe { converted(coefficients, basis.code(), 0) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let status = unsafe { super::polynomial_extension::terlan_ndarray_v1_polyroots(power, output) };
    unsafe { delete_arrays(&[power]) };
    status
}

/// Trims high-order basis coefficients under a tolerance.
pub(crate) unsafe fn trim(coefficients: Array, tolerance: f64, output: *mut Array) -> i32 {
    unsafe {
        super::polynomial_power_extension::terlan_ndarray_v1_polytrim(
            coefficients,
            tolerance,
            output,
        )
    }
}

/// Returns coefficients for `offset + scale * x` in either admitted basis.
pub(crate) unsafe fn line(offset: f64, scale: f64, output: *mut Array) -> i32 {
    unsafe { super::polynomial_power_extension::terlan_ndarray_v1_polyline(offset, scale, output) }
}

/// Returns basis values from degree zero upward at one coordinate.
pub(crate) fn sequence(basis: Basis, coordinate: Complex64, maximum: usize) -> Vec<Complex64> {
    let mut values = vec![Complex64::new(1.0, 0.0); maximum + 1];
    if maximum == 0 {
        return values;
    }
    values[1] = coordinate;
    for order in 2..=maximum {
        values[order] = match basis {
            Basis::Chebyshev => 2.0 * coordinate * values[order - 1] - values[order - 2],
            Basis::Legendre => {
                ((2 * order - 1) as f64 * coordinate * values[order - 1]
                    - (order - 1) as f64 * values[order - 2])
                    / order as f64
            }
        };
    }
    values
}

/// Evaluates one rank-matched coefficient tensor over broadcast or Cartesian coordinates.
pub(crate) unsafe fn evaluate(
    coordinates: &[Array],
    coefficients: Array,
    basis: Basis,
    grid: bool,
    output: *mut Array,
) -> i32 {
    if coordinates.is_empty() {
        return INVALID_ARGUMENT;
    }
    let coefficient_shape = match unsafe { shape(coefficients) } {
        Ok(value)
            if value.len() == coordinates.len() && value.iter().all(|dimension| *dimension > 0) =>
        {
            value
        }
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let coefficients = match unsafe { numeric(coefficients, true) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut inputs = Vec::with_capacity(coordinates.len());
    for &coordinate in coordinates {
        match unsafe { shaped(coordinate) } {
            Ok(value) => inputs.push(value),
            Err(status) => return status,
        }
    }
    let shapes = inputs
        .iter()
        .map(|(_, dimensions)| dimensions.as_slice())
        .collect::<Vec<_>>();
    let output_shape = if grid {
        shapes
            .iter()
            .flat_map(|shape| shape.iter())
            .copied()
            .collect()
    } else {
        match broadcast_shape(&shapes) {
            Ok(value) => value,
            Err(status) => return status,
        }
    };
    let count = match element_count(&output_shape) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let output_strides = match strides(&output_shape) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let input_strides = match shapes
        .iter()
        .map(|shape| strides(shape))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(value) => value,
        Err(status) => return status,
    };
    let input_counts = match shapes
        .iter()
        .map(|shape| element_count(shape))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(value) => value,
        Err(status) => return status,
    };
    let coefficient_strides = match strides(&coefficient_shape) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let coefficient_count = match element_count(&coefficient_shape) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut result = Vec::with_capacity(count);
    for flat in 0..count {
        let mut basis_values = Vec::with_capacity(inputs.len());
        for axis in 0..inputs.len() {
            let input_index = if grid {
                let trailing = input_counts[axis + 1..].iter().product::<usize>();
                (flat / trailing) % input_counts[axis]
            } else {
                broadcast_offset(
                    flat,
                    &output_shape,
                    &output_strides,
                    shapes[axis],
                    &input_strides[axis],
                )
            };
            basis_values.push(sequence(
                basis,
                inputs[axis].0.values[input_index],
                usize::try_from(coefficient_shape[axis] - 1).unwrap_or(0),
            ));
        }
        let mut value = Complex64::new(0.0, 0.0);
        for coefficient_index in 0..coefficient_count {
            let mut remainder = coefficient_index;
            let mut term = coefficients.values[coefficient_index];
            for axis in 0..coefficient_shape.len() {
                let index = remainder / coefficient_strides[axis];
                remainder %= coefficient_strides[axis];
                term *= basis_values[axis][index];
            }
            value += term;
        }
        result.push(value);
    }
    let complex = coefficients.complex || inputs.iter().any(|(values, _)| values.complex);
    let use_narrow = narrow(&coefficients) && inputs.iter().all(|(values, _)| narrow(values));
    unsafe {
        write_result(
            owned_numeric(&result, &output_shape, complex, use_narrow),
            output,
        )
    }
}

/// Builds a one-, two-, or three-dimensional orthogonal Vandermonde matrix.
pub(crate) unsafe fn vander(
    coordinates: &[Array],
    degrees: &[i64],
    basis: Basis,
    output: *mut Array,
) -> i32 {
    if coordinates.len() != degrees.len() || coordinates.is_empty() {
        return INVALID_ARGUMENT;
    }
    let degrees = match degrees
        .iter()
        .map(|&value| degree(value))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut inputs = Vec::with_capacity(coordinates.len());
    for &coordinate in coordinates {
        match unsafe { shaped(coordinate) } {
            Ok(value) => inputs.push(value),
            Err(status) => return status,
        }
    }
    let shapes = inputs
        .iter()
        .map(|(_, dimensions)| dimensions.as_slice())
        .collect::<Vec<_>>();
    let mut output_shape = match broadcast_shape(&shapes) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let count = match element_count(&output_shape) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let width = match degrees.iter().try_fold(1usize, |value, degree| {
        value.checked_mul(degree + 1).ok_or(SIZE_OVERFLOW)
    }) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let output_strides = match strides(&output_shape) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let input_strides = match shapes
        .iter()
        .map(|shape| strides(shape))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut result = Vec::with_capacity(count.saturating_mul(width));
    for flat in 0..count {
        let mut row = vec![Complex64::new(1.0, 0.0)];
        for axis in 0..inputs.len() {
            let index = broadcast_offset(
                flat,
                &output_shape,
                &output_strides,
                shapes[axis],
                &input_strides[axis],
            );
            let axis_values = sequence(basis, inputs[axis].0.values[index], degrees[axis]);
            row = row
                .iter()
                .flat_map(|lhs| axis_values.iter().map(move |rhs| lhs * rhs))
                .collect();
        }
        result.extend(row);
    }
    output_shape.push(width as i64);
    let complex = inputs.iter().any(|(values, _)| values.complex);
    let use_narrow = inputs.iter().all(|(values, _)| narrow(values));
    unsafe {
        write_result(
            owned_numeric(&result, &output_shape, complex, use_narrow),
            output,
        )
    }
}

/// Returns Chebyshev points of the first kind in ascending order.
pub(crate) unsafe fn chebyshev_points_first(count: i64, output: *mut Array) -> i32 {
    let count = match usize::try_from(count) {
        Ok(value) if value > 0 => value,
        _ => return INVALID_ARGUMENT,
    };
    let values = (0..count)
        .map(|index| {
            let odd = -(count as i64) + 1 + 2 * index as i64;
            (0.5 * std::f64::consts::PI / count as f64 * odd as f64).sin()
        })
        .collect::<Vec<_>>();
    unsafe { write_result(owned_real(&values, &[count as i64], 3), output) }
}

/// Returns Chebyshev points of the second kind in ascending order.
pub(crate) unsafe fn chebyshev_points_second(count: i64, output: *mut Array) -> i32 {
    let count = match usize::try_from(count) {
        Ok(value) if value >= 2 => value,
        _ => return INVALID_ARGUMENT,
    };
    let values = (0..count)
        .map(|index| {
            (-std::f64::consts::PI + index as f64 * std::f64::consts::PI / (count - 1) as f64).cos()
        })
        .collect::<Vec<_>>();
    unsafe { write_result(owned_real(&values, &[count as i64], 3), output) }
}

/// Returns Gaussian quadrature nodes and weights for one orthogonal basis.
pub(crate) unsafe fn gauss(requested_degree: i64, basis: Basis, output: *mut Array) -> i32 {
    let dimension = match usize::try_from(requested_degree) {
        Ok(value) if value > 0 => value,
        _ => return INVALID_ARGUMENT,
    };
    let (nodes, weights) = match basis {
        Basis::Chebyshev => {
            let nodes = (0..dimension)
                .map(|index| {
                    (std::f64::consts::PI * (2 * index + 1) as f64 / (2 * dimension) as f64).cos()
                })
                .collect::<Vec<_>>();
            let weights = vec![std::f64::consts::PI / dimension as f64; dimension];
            (nodes, weights)
        }
        Basis::Legendre => {
            let mut jacobi = Mat::<f64>::zeros(dimension, dimension);
            for order in 1..dimension {
                let value = order as f64 / (4.0 * (order * order) as f64 - 1.0).sqrt();
                jacobi[(order - 1, order)] = value;
                jacobi[(order, order - 1)] = value;
            }
            let factor = match jacobi.self_adjoint_eigen(Side::Lower) {
                Ok(value) => value,
                Err(_) => return DOMAIN_ERROR,
            };
            let nodes = factor
                .S()
                .column_vector()
                .iter()
                .copied()
                .collect::<Vec<_>>();
            let weights = (0..dimension)
                .map(|column| 2.0 * factor.U()[(0, column)].powi(2))
                .collect::<Vec<_>>();
            (nodes, weights)
        }
    };
    let values = nodes.into_iter().chain(weights).collect::<Vec<_>>();
    unsafe { write_result(owned_real(&values, &[2, dimension as i64], 3), output) }
}

/// Evaluates the selected orthogonality weight function.
pub(crate) unsafe fn weight(coordinates: Array, basis: Basis, output: *mut Array) -> i32 {
    let (coordinates, dimensions) = match unsafe { shaped(coordinates) } {
        Ok(value) if !value.0.complex => value,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let values = coordinates
        .values
        .iter()
        .map(|value| match basis {
            Basis::Chebyshev => 1.0 / (1.0 - value.re * value.re).sqrt(),
            Basis::Legendre => 1.0,
        })
        .collect::<Vec<_>>();
    unsafe {
        write_result(
            owned_real(
                &values,
                &dimensions,
                if narrow_dtype(coordinates.dtype) {
                    11
                } else {
                    3
                },
            ),
            output,
        )
    }
}

/// Interpolates values sampled at first-kind Chebyshev points.
pub(crate) unsafe fn chebyshev_interpolate(values: Array, output: *mut Array) -> i32 {
    let values_data = match unsafe { numeric_vector(values) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let count = values_data.values.len();
    let points = (0..count)
        .map(|index| {
            let odd = -(count as i64) + 1 + 2 * index as i64;
            (0.5 * std::f64::consts::PI / count as f64 * odd as f64).sin()
        })
        .collect::<Vec<_>>();
    let points = match unsafe { owned_real(&points, &[count as i64], 3) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let status = unsafe { fit(points, values, (count - 1) as i64, Basis::Chebyshev, output) };
    unsafe { delete_arrays(&[points]) };
    status
}
