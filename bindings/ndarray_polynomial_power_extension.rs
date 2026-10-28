//! Extended power-basis polynomial operations.

use rustfft::num_complex::Complex64;

use super::native_support::{
    element_count, numeric, owned_numeric, shape, write_result, Array, INVALID_ARGUMENT,
    SHAPE_MISMATCH, SIZE_OVERFLOW,
};
use super::polynomial_extension::{multiplied_numeric, narrow_dtype, numeric_vector};

/// Reads one finite rank-one numeric vector, including an empty vector.
unsafe fn root_vector(array: Array) -> Result<super::native_support::NumericValues, i32> {
    let dimensions = unsafe { shape(array)? };
    if dimensions.len() != 1 {
        return Err(SHAPE_MISMATCH);
    }
    unsafe { numeric(array, true) }
}

/// Returns a checked degree as `usize`.
pub(crate) fn degree(value: i64) -> Result<usize, i32> {
    usize::try_from(value).map_err(|_| INVALID_ARGUMENT)
}

/// Returns whether one source precision policy is narrow.
pub(crate) fn narrow(values: &super::native_support::NumericValues) -> bool {
    narrow_dtype(values.dtype)
}

/// Computes a checked broadcast shape.
pub(crate) fn broadcast_shape(shapes: &[&[i64]]) -> Result<Vec<i64>, i32> {
    let rank = shapes.iter().map(|shape| shape.len()).max().unwrap_or(0);
    let mut output = vec![1i64; rank];
    for shape in shapes {
        for offset in 0..rank {
            let source = shape
                .len()
                .checked_sub(rank - offset)
                .map(|index| shape[index])
                .unwrap_or(1);
            if source < 0 {
                return Err(SHAPE_MISMATCH);
            }
            if output[offset] == 1 {
                output[offset] = source;
            } else if source != 1 && output[offset] != source {
                return Err(SHAPE_MISMATCH);
            }
        }
    }
    Ok(output)
}

/// Computes row-major strides for a shape.
pub(crate) fn strides(shape: &[i64]) -> Result<Vec<usize>, i32> {
    let mut values = vec![1usize; shape.len()];
    for index in (0..shape.len().saturating_sub(1)).rev() {
        let dimension = usize::try_from(shape[index + 1]).map_err(|_| SHAPE_MISMATCH)?;
        values[index] = values[index + 1]
            .checked_mul(dimension)
            .ok_or(SIZE_OVERFLOW)?;
    }
    Ok(values)
}

/// Maps one broadcast output offset to an input offset.
pub(crate) fn broadcast_offset(
    flat: usize,
    output_shape: &[i64],
    output_strides: &[usize],
    input_shape: &[i64],
    input_strides: &[usize],
) -> usize {
    let leading = output_shape.len() - input_shape.len();
    let mut remainder = flat;
    let mut input = 0usize;
    for output_axis in 0..output_shape.len() {
        let coordinate = if output_shape[output_axis] == 0 {
            0
        } else {
            remainder / output_strides[output_axis]
        };
        remainder %= output_strides[output_axis];
        if output_axis >= leading {
            let input_axis = output_axis - leading;
            if input_shape[input_axis] != 1 {
                input += coordinate * input_strides[input_axis];
            }
        }
    }
    input
}

/// Evaluates low-to-high coefficients at one coordinate.
fn evaluate(coefficients: &[Complex64], coordinate: Complex64) -> Complex64 {
    coefficients
        .iter()
        .rev()
        .fold(Complex64::new(0.0, 0.0), |value, coefficient| {
            value * coordinate + coefficient
        })
}

/// Builds powers from degree zero through a requested degree.
fn powers(coordinate: Complex64, degree: usize) -> Vec<Complex64> {
    let mut output = Vec::with_capacity(degree + 1);
    let mut value = Complex64::new(1.0, 0.0);
    for _ in 0..=degree {
        output.push(value);
        value *= coordinate;
    }
    output
}

/// Reads one finite numeric array and its shape.
pub(crate) unsafe fn shaped(
    array: Array,
) -> Result<(super::native_support::NumericValues, Vec<i64>), i32> {
    let dimensions = unsafe { shape(array)? };
    let values = unsafe { numeric(array, true)? };
    Ok((values, dimensions))
}

/// Returns low-to-high coefficients for a monic polynomial with the supplied roots.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyfromroots(roots: Array, output: *mut Array) -> i32 {
    let roots = match unsafe { root_vector(roots) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut coefficients = vec![Complex64::new(1.0, 0.0)];
    for root in &roots.values {
        coefficients = match multiplied_numeric(&coefficients, &[-*root, Complex64::new(1.0, 0.0)])
        {
            Ok(value) => value,
            Err(status) => return status,
        };
    }
    unsafe {
        write_result(
            owned_numeric(
                &coefficients,
                &[coefficients.len() as i64],
                roots.complex,
                narrow(&roots),
            ),
            output,
        )
    }
}

/// Returns low-to-high coefficients for `offset + scale * x`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyline(
    offset: f64,
    scale: f64,
    output: *mut Array,
) -> i32 {
    if !offset.is_finite() || !scale.is_finite() {
        return INVALID_ARGUMENT;
    }
    let values = if scale == 0.0 {
        vec![offset]
    } else {
        vec![offset, scale]
    };
    unsafe {
        write_result(
            super::native_support::owned_real(&values, &[values.len() as i64], 3),
            output,
        )
    }
}

/// Multiplies one low-to-high coefficient vector by `x`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polymulx(
    coefficients: Array,
    output: *mut Array,
) -> i32 {
    let coefficients = match unsafe { numeric_vector(coefficients) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let source = super::polynomial_extension::trim_numeric(&coefficients.values);
    let values = if source.len() == 1 && source[0] == Complex64::new(0.0, 0.0) {
        source.to_vec()
    } else {
        std::iter::once(Complex64::new(0.0, 0.0))
            .chain(source.iter().copied())
            .collect()
    };
    unsafe {
        write_result(
            owned_numeric(
                &values,
                &[values.len() as i64],
                coefficients.complex,
                narrow(&coefficients),
            ),
            output,
        )
    }
}

/// Raises one low-to-high coefficient vector to a nonnegative integer power.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polypow(
    coefficients: Array,
    exponent: i64,
    output: *mut Array,
) -> i32 {
    let exponent = match degree(exponent) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let coefficients = match unsafe { numeric_vector(coefficients) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut result = vec![Complex64::new(1.0, 0.0)];
    let mut base = coefficients.values.clone();
    let mut power = exponent;
    while power > 0 {
        if power & 1 == 1 {
            result = match multiplied_numeric(&result, &base) {
                Ok(value) => value,
                Err(status) => return status,
            };
        }
        power >>= 1;
        if power > 0 {
            base = match multiplied_numeric(&base, &base) {
                Ok(value) => value,
                Err(status) => return status,
            };
        }
    }
    unsafe {
        write_result(
            owned_numeric(
                &result,
                &[result.len() as i64],
                coefficients.complex,
                narrow(&coefficients),
            ),
            output,
        )
    }
}

/// Removes trailing coefficients whose magnitude does not exceed a tolerance.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polytrim(
    coefficients: Array,
    tolerance: f64,
    output: *mut Array,
) -> i32 {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return INVALID_ARGUMENT;
    }
    let coefficients = match unsafe { numeric_vector(coefficients) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = coefficients
        .values
        .iter()
        .rposition(|value| value.norm() > tolerance)
        .map(|last| coefficients.values[..=last].to_vec())
        .unwrap_or_else(|| vec![Complex64::new(0.0, 0.0)]);
    unsafe {
        write_result(
            owned_numeric(
                &values,
                &[values.len() as i64],
                coefficients.complex,
                narrow(&coefficients),
            ),
            output,
        )
    }
}

/// Evaluates a monic polynomial directly from roots at every coordinate.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyvalfromroots(
    coordinates: Array,
    roots: Array,
    output: *mut Array,
) -> i32 {
    let (coordinates, dimensions) = match unsafe { shaped(coordinates) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let roots = match unsafe { root_vector(roots) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = coordinates
        .values
        .iter()
        .map(|coordinate| {
            roots
                .values
                .iter()
                .fold(Complex64::new(1.0, 0.0), |value, root| {
                    value * (coordinate - root)
                })
        })
        .collect::<Vec<_>>();
    unsafe {
        write_result(
            owned_numeric(
                &values,
                &dimensions,
                coordinates.complex || roots.complex,
                narrow(&coordinates) && narrow(&roots),
            ),
            output,
        )
    }
}

/// Builds a power Vandermonde matrix with columns from degree zero upward.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyvander(
    coordinates: Array,
    requested_degree: i64,
    output: *mut Array,
) -> i32 {
    let degree = match degree(requested_degree) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let (coordinates, mut dimensions) = match unsafe { shaped(coordinates) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let width = match degree.checked_add(1) {
        Some(value) => value,
        None => return SIZE_OVERFLOW,
    };
    let mut values = Vec::with_capacity(coordinates.values.len().saturating_mul(width));
    for coordinate in &coordinates.values {
        values.extend(powers(*coordinate, degree));
    }
    dimensions.push(width as i64);
    unsafe {
        write_result(
            owned_numeric(
                &values,
                &dimensions,
                coordinates.complex,
                narrow(&coordinates),
            ),
            output,
        )
    }
}

/// Evaluates a two-dimensional low-to-high coefficient grid with broadcasting.
unsafe fn evaluate_2d(
    x: Array,
    y: Array,
    coefficients: Array,
    grid: bool,
) -> Result<(Vec<Complex64>, Vec<i64>, bool, bool), i32> {
    let (x, x_shape) = unsafe { shaped(x)? };
    let (y, y_shape) = unsafe { shaped(y)? };
    let coefficient_shape = unsafe { shape(coefficients)? };
    if coefficient_shape.len() != 2 || coefficient_shape.iter().any(|&value| value <= 0) {
        return Err(SHAPE_MISMATCH);
    }
    let coefficients = unsafe { numeric(coefficients, true)? };
    let rows = usize::try_from(coefficient_shape[0]).map_err(|_| SHAPE_MISMATCH)?;
    let columns = usize::try_from(coefficient_shape[1]).map_err(|_| SHAPE_MISMATCH)?;
    let output_shape = if grid {
        x_shape.iter().chain(&y_shape).copied().collect::<Vec<_>>()
    } else {
        broadcast_shape(&[&x_shape, &y_shape])?
    };
    let count = element_count(&output_shape)?;
    let output_strides = strides(&output_shape)?;
    let x_strides = strides(&x_shape)?;
    let y_strides = strides(&y_shape)?;
    let mut output = Vec::with_capacity(count);
    for flat in 0..count {
        let (x_index, y_index) = if grid {
            let y_count = element_count(&y_shape)?;
            (flat / y_count, flat % y_count)
        } else {
            (
                broadcast_offset(flat, &output_shape, &output_strides, &x_shape, &x_strides),
                broadcast_offset(flat, &output_shape, &output_strides, &y_shape, &y_strides),
            )
        };
        let mut value = Complex64::new(0.0, 0.0);
        for row in (0..rows).rev() {
            value = value * x.values[x_index]
                + evaluate(
                    &coefficients.values[row * columns..(row + 1) * columns],
                    y.values[y_index],
                );
        }
        output.push(value);
    }
    Ok((
        output,
        output_shape,
        x.complex || y.complex || coefficients.complex,
        narrow(&x) && narrow(&y) && narrow(&coefficients),
    ))
}

/// Evaluates a three-dimensional low-to-high coefficient grid with broadcasting.
unsafe fn evaluate_3d(
    x: Array,
    y: Array,
    z: Array,
    coefficients: Array,
    grid: bool,
) -> Result<(Vec<Complex64>, Vec<i64>, bool, bool), i32> {
    let (x, x_shape) = unsafe { shaped(x)? };
    let (y, y_shape) = unsafe { shaped(y)? };
    let (z, z_shape) = unsafe { shaped(z)? };
    let coefficient_shape = unsafe { shape(coefficients)? };
    if coefficient_shape.len() != 3 || coefficient_shape.iter().any(|&value| value <= 0) {
        return Err(SHAPE_MISMATCH);
    }
    let coefficients = unsafe { numeric(coefficients, true)? };
    let a = usize::try_from(coefficient_shape[0]).map_err(|_| SHAPE_MISMATCH)?;
    let b = usize::try_from(coefficient_shape[1]).map_err(|_| SHAPE_MISMATCH)?;
    let c = usize::try_from(coefficient_shape[2]).map_err(|_| SHAPE_MISMATCH)?;
    let output_shape = if grid {
        x_shape
            .iter()
            .chain(&y_shape)
            .chain(&z_shape)
            .copied()
            .collect::<Vec<_>>()
    } else {
        broadcast_shape(&[&x_shape, &y_shape, &z_shape])?
    };
    let count = element_count(&output_shape)?;
    let output_strides = strides(&output_shape)?;
    let x_strides = strides(&x_shape)?;
    let y_strides = strides(&y_shape)?;
    let z_strides = strides(&z_shape)?;
    let y_count = element_count(&y_shape)?;
    let z_count = element_count(&z_shape)?;
    let mut output = Vec::with_capacity(count);
    for flat in 0..count {
        let (xi, yi, zi) = if grid {
            (
                flat / (y_count * z_count),
                (flat / z_count) % y_count,
                flat % z_count,
            )
        } else {
            (
                broadcast_offset(flat, &output_shape, &output_strides, &x_shape, &x_strides),
                broadcast_offset(flat, &output_shape, &output_strides, &y_shape, &y_strides),
                broadcast_offset(flat, &output_shape, &output_strides, &z_shape, &z_strides),
            )
        };
        let mut x_value = Complex64::new(0.0, 0.0);
        for i in (0..a).rev() {
            let mut y_value = Complex64::new(0.0, 0.0);
            for j in (0..b).rev() {
                y_value = y_value * y.values[yi]
                    + evaluate(
                        &coefficients.values[(i * b + j) * c..(i * b + j + 1) * c],
                        z.values[zi],
                    );
            }
            x_value = x_value * x.values[xi] + y_value;
        }
        output.push(x_value);
    }
    Ok((
        output,
        output_shape,
        x.complex || y.complex || z.complex || coefficients.complex,
        narrow(&x) && narrow(&y) && narrow(&z) && narrow(&coefficients),
    ))
}

macro_rules! evaluation_api {
    ($name:ident, $helper:ident, $($argument:ident),+) => {
        #[doc = "Evaluates a multivariate power-basis coefficient grid."]
        #[no_mangle]
        pub unsafe extern "C" fn $name($($argument: Array,)+ coefficients: Array, output: *mut Array) -> i32 {
            let result = unsafe { $helper($($argument,)+ coefficients, false) }.and_then(
                |(values, dimensions, complex, narrow)| unsafe {
                    owned_numeric(&values, &dimensions, complex, narrow)
                },
            );
            unsafe { write_result(result, output) }
        }
    };
}

evaluation_api!(terlan_ndarray_v1_polyval2d, evaluate_2d, x, y);
evaluation_api!(terlan_ndarray_v1_polyval3d, evaluate_3d, x, y, z);

macro_rules! grid_api {
    ($name:ident, $helper:ident, $($argument:ident),+) => {
        #[doc = "Evaluates a multivariate power-basis coefficient grid over a Cartesian product."]
        #[no_mangle]
        pub unsafe extern "C" fn $name($($argument: Array,)+ coefficients: Array, output: *mut Array) -> i32 {
            let result = unsafe { $helper($($argument,)+ coefficients, true) }.and_then(
                |(values, dimensions, complex, narrow)| unsafe {
                    owned_numeric(&values, &dimensions, complex, narrow)
                },
            );
            unsafe { write_result(result, output) }
        }
    };
}

grid_api!(terlan_ndarray_v1_polygrid2d, evaluate_2d, x, y);
grid_api!(terlan_ndarray_v1_polygrid3d, evaluate_3d, x, y, z);

/// Builds a broadcast multivariate power Vandermonde matrix.
unsafe fn multivariate_vander(arrays: &[Array], degrees: &[i64], output: *mut Array) -> i32 {
    if arrays.len() != degrees.len() || arrays.is_empty() {
        return INVALID_ARGUMENT;
    }
    let mut inputs = Vec::with_capacity(arrays.len());
    for &array in arrays {
        match unsafe { shaped(array) } {
            Ok(value) => inputs.push(value),
            Err(status) => return status,
        }
    }
    let shapes = inputs
        .iter()
        .map(|(_, shape)| shape.as_slice())
        .collect::<Vec<_>>();
    let mut output_shape = match broadcast_shape(&shapes) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let degrees = match degrees
        .iter()
        .map(|&value| degree(value))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(value) => value,
        Err(status) => return status,
    };
    let width = match degrees.iter().try_fold(1usize, |count, value| {
        count.checked_mul(value + 1).ok_or(SIZE_OVERFLOW)
    }) {
        Ok(value) => value,
        Err(status) => return status,
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
    let mut values = Vec::with_capacity(count.saturating_mul(width));
    for flat in 0..count {
        let basis = inputs
            .iter()
            .zip(&degrees)
            .zip(&input_strides)
            .map(|(((input, shape), degree), strides)| {
                let index = broadcast_offset(flat, &output_shape, &output_strides, shape, strides);
                powers(input.values[index], *degree)
            })
            .collect::<Vec<_>>();
        let mut row = vec![Complex64::new(1.0, 0.0)];
        for axis in basis {
            row = row
                .iter()
                .flat_map(|lhs| axis.iter().map(move |rhs| lhs * rhs))
                .collect();
        }
        values.extend(row);
    }
    output_shape.push(width as i64);
    let complex = inputs.iter().any(|(values, _)| values.complex);
    let narrow = inputs.iter().all(|(values, _)| narrow(values));
    unsafe {
        write_result(
            owned_numeric(&values, &output_shape, complex, narrow),
            output,
        )
    }
}

/// Builds a two-dimensional broadcast power Vandermonde matrix.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyvander2d(
    x: Array,
    y: Array,
    x_degree: i64,
    y_degree: i64,
    output: *mut Array,
) -> i32 {
    unsafe { multivariate_vander(&[x, y], &[x_degree, y_degree], output) }
}

/// Builds a three-dimensional broadcast power Vandermonde matrix.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyvander3d(
    x: Array,
    y: Array,
    z: Array,
    x_degree: i64,
    y_degree: i64,
    z_degree: i64,
    output: *mut Array,
) -> i32 {
    unsafe { multivariate_vander(&[x, y, z], &[x_degree, y_degree, z_degree], output) }
}

#[cfg(test)]
#[path = "ndarray_polynomial_power_tests.rs"]
mod tests;
