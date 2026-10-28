//! NumPy-compatible numerical completion operations.

use rustfft::num_complex::Complex64;

use super::native_support::{
    Array, DOMAIN_ERROR, INVALID_ARGUMENT, OK, SHAPE_MISMATCH, dtype, numeric, owned_bool,
    owned_numeric, owned_numeric_as, owned_real, promoted_dtype, shape, transfer_arrays,
    write_result,
};
use super::polynomial_power_extension::{broadcast_offset, broadcast_shape, narrow, strides};

#[cfg(test)]
use super::numpy_comparison_extension::{
    terlan_ndarray_v1_array_equal_nan, terlan_ndarray_v1_isclose,
};

/// Applies NumPy's finite, infinity, and optional NaN closeness rules.
pub(crate) fn close_values(
    lhs: Complex64,
    rhs: Complex64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    equal_nan: bool,
) -> bool {
    if lhs == rhs {
        return true;
    }
    let lhs_nan = lhs.re.is_nan() || lhs.im.is_nan();
    let rhs_nan = rhs.re.is_nan() || rhs.im.is_nan();
    if lhs_nan || rhs_nan {
        return equal_nan && lhs_nan && rhs_nan;
    }
    if !lhs.re.is_finite() || !lhs.im.is_finite() || !rhs.re.is_finite() || !rhs.im.is_finite() {
        return false;
    }
    (lhs - rhs).norm() <= absolute_tolerance + relative_tolerance * rhs.norm()
}

/// Writes one scalar boolean result through the C status ABI.
pub(crate) unsafe fn write_bool(value: bool, output: *mut bool) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = value };
    OK
}

/// Reads two numeric arrays and prepares checked broadcasting metadata.
unsafe fn binary_inputs(
    lhs: Array,
    rhs: Array,
) -> Result<
    (
        super::native_support::NumericValues,
        Vec<i64>,
        Vec<usize>,
        super::native_support::NumericValues,
        Vec<i64>,
        Vec<usize>,
        Vec<i64>,
        Vec<usize>,
    ),
    i32,
> {
    let lhs_shape = unsafe { shape(lhs)? };
    let rhs_shape = unsafe { shape(rhs)? };
    let output_shape = broadcast_shape(&[&lhs_shape, &rhs_shape])?;
    let lhs_strides = strides(&lhs_shape)?;
    let rhs_strides = strides(&rhs_shape)?;
    let output_strides = strides(&output_shape)?;
    Ok((
        unsafe { numeric(lhs, false)? },
        lhs_shape,
        lhs_strides,
        unsafe { numeric(rhs, false)? },
        rhs_shape,
        rhs_strides,
        output_shape,
        output_strides,
    ))
}

/// Applies one broadcast-aware numeric binary operation.
unsafe fn binary_numeric(
    lhs: Array,
    rhs: Array,
    operation: impl Fn(Complex64, Complex64) -> Complex64,
    preserve_narrow: bool,
    output: *mut Array,
) -> i32 {
    let (lhs, lhs_shape, lhs_strides, rhs, rhs_shape, rhs_strides, output_shape, output_strides) =
        match unsafe { binary_inputs(lhs, rhs) } {
            Ok(value) => value,
            Err(status) => return status,
        };
    let count = output_shape.iter().try_fold(1usize, |count, dimension| {
        count.checked_mul(*dimension as usize).ok_or(DOMAIN_ERROR)
    });
    let count = match count {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = (0..count)
        .map(|index| {
            let lhs_index = broadcast_offset(
                index,
                &output_shape,
                &output_strides,
                &lhs_shape,
                &lhs_strides,
            );
            let rhs_index = broadcast_offset(
                index,
                &output_shape,
                &output_strides,
                &rhs_shape,
                &rhs_strides,
            );
            operation(lhs.values[lhs_index], rhs.values[rhs_index])
        })
        .collect::<Vec<_>>();
    unsafe {
        write_result(
            owned_numeric(
                &values,
                &output_shape,
                lhs.complex || rhs.complex,
                preserve_narrow && narrow(&lhs) && narrow(&rhs),
            ),
            output,
        )
    }
}

/// Applies one broadcast-aware predicate.
pub(crate) unsafe fn binary_predicate(
    lhs_array: Array,
    rhs_array: Array,
    predicate: impl Fn(Complex64, Complex64) -> bool,
) -> Result<(Vec<u8>, Vec<i64>), i32> {
    let (lhs, lhs_shape, lhs_strides, rhs, rhs_shape, rhs_strides, output_shape, output_strides) =
        unsafe { binary_inputs(lhs_array, rhs_array)? };
    let count = super::native_support::element_count(&output_shape)?;
    let values = (0..count)
        .map(|index| {
            let lhs_index = broadcast_offset(
                index,
                &output_shape,
                &output_strides,
                &lhs_shape,
                &lhs_strides,
            );
            let rhs_index = broadcast_offset(
                index,
                &output_shape,
                &output_strides,
                &rhs_shape,
                &rhs_strides,
            );
            u8::from(predicate(lhs.values[lhs_index], rhs.values[rhs_index]))
        })
        .collect();
    Ok((values, output_shape))
}

/// Returns whether every pair is within the supplied tolerances.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_allclose(
    lhs: Array,
    rhs: Array,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    equal_nan: i32,
    output: *mut bool,
) -> i32 {
    if relative_tolerance < 0.0 || absolute_tolerance < 0.0 {
        return INVALID_ARGUMENT;
    }
    let result = unsafe {
        binary_predicate(lhs, rhs, |lhs, rhs| {
            close_values(
                lhs,
                rhs,
                relative_tolerance,
                absolute_tolerance,
                equal_nan != 0,
            )
        })
    };
    match result {
        Ok((values, _)) => unsafe { write_bool(values.iter().all(|value| *value != 0), output) },
        Err(status) => status,
    }
}

/// Returns the phase angle of every real or complex element.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_angle(
    array: Array,
    degrees: i32,
    output: *mut Array,
) -> i32 {
    let dimensions = match unsafe { shape(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = match unsafe { numeric(array, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let scale = if degrees != 0 {
        180.0 / std::f64::consts::PI
    } else {
        1.0
    };
    let result = values
        .values
        .iter()
        .map(|value| value.arg() * scale)
        .collect::<Vec<_>>();
    unsafe {
        write_result(
            owned_real(&result, &dimensions, if narrow(&values) { 11 } else { 3 }),
            output,
        )
    }
}

/// Returns strict shape-and-value equality.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_array_equal(
    lhs: Array,
    rhs: Array,
    output: *mut bool,
) -> i32 {
    let lhs_shape = match unsafe { shape(lhs) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let rhs_shape = match unsafe { shape(rhs) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if lhs_shape != rhs_shape {
        return unsafe { write_bool(false, output) };
    }
    let result = unsafe { binary_predicate(lhs, rhs, |lhs, rhs| lhs == rhs) };
    match result {
        Ok((values, _)) => unsafe { write_bool(values.iter().all(|value| *value != 0), output) },
        Err(status) => status,
    }
}

/// Returns whether two arrays are broadcast-equivalent.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_array_equiv(
    lhs: Array,
    rhs: Array,
    output: *mut bool,
) -> i32 {
    match unsafe { binary_predicate(lhs, rhs, |lhs, rhs| lhs == rhs) } {
        Ok((values, _)) => unsafe { write_bool(values.iter().all(|value| *value != 0), output) },
        Err(SHAPE_MISMATCH) => unsafe { write_bool(false, output) },
        Err(status) => status,
    }
}

/// Copies one array after rejecting every non-finite element.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_asarray_chkfinite(
    array: Array,
    output: *mut Array,
) -> i32 {
    if let Err(status) = unsafe { numeric(array, true) } {
        return status;
    }
    unsafe { crate::ffi::terlan_ndarray_v1_copy(array, output) }
}

/// Copies one finite array into an explicit dtype and memory order.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_asarray_chkfinite_as(
    array: Array,
    requested_dtype: i32,
    order: i32,
    output: *mut Array,
) -> i32 {
    if let Err(status) = unsafe { numeric(array, true) } {
        return status;
    }
    unsafe {
        crate::ffi::terlan_ndarray_v1_astype_ordered(array, requested_dtype, order, 1, 4, output)
    }
}

/// Converts one numeric array to Float64 by retaining real components.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_asfarray(array: Array, output: *mut Array) -> i32 {
    let dimensions = match unsafe { shape(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = match unsafe { numeric(array, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let reals = values
        .values
        .iter()
        .map(|value| value.re)
        .collect::<Vec<_>>();
    unsafe { write_result(owned_real(&reals, &dimensions, 3), output) }
}

/// Computes an unweighted arithmetic average over selected axes.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_average(
    array: Array,
    axes: *const i64,
    axis_count: i64,
    keep_dims: i32,
    output: *mut Array,
) -> i32 {
    unsafe { crate::ffi::terlan_ndarray_v1_mean(array, axes, axis_count, keep_dims, output) }
}

/// Builds one standard spectral window.
unsafe fn window(length: i64, kind: i32, parameter: f64, output: *mut Array) -> i32 {
    let length = usize::try_from(length.max(0)).expect("nonnegative i64 fits usize");
    if length <= 1 {
        let values = if length == 0 { Vec::new() } else { vec![1.0] };
        return unsafe { write_result(owned_real(&values, &[length as i64], 3), output) };
    }
    let denominator = length.saturating_sub(1).max(1) as f64;
    let values = (0..length)
        .map(|index| {
            let phase = 2.0 * std::f64::consts::PI * index as f64 / denominator;
            match kind {
                0 => 1.0 - ((index as f64 - denominator / 2.0) / (denominator / 2.0)).abs(),
                1 => 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos(),
                2 => 0.54 - 0.46 * phase.cos(),
                3 => 0.5 - 0.5 * phase.cos(),
                4 => {
                    let ratio = (2.0 * index as f64 / denominator - 1.0).clamp(-1.0, 1.0);
                    bessel_i0(parameter * (1.0 - ratio * ratio).sqrt()) / bessel_i0(parameter)
                }
                _ => 0.0,
            }
        })
        .collect::<Vec<_>>();
    unsafe { write_result(owned_real(&values, &[length as i64], 3), output) }
}

/// Computes the modified Bessel function I0 by its convergent power series.
fn bessel_i0(value: f64) -> f64 {
    let scaled = value * value / 4.0;
    let mut sum = 1.0;
    let mut term = 1.0;
    for order in 1..=100 {
        term *= scaled / (order * order) as f64;
        sum += term;
        if term.abs() <= f64::EPSILON * sum.abs() {
            break;
        }
    }
    sum
}

macro_rules! window_function {
    ($name:ident, $kind:expr) => {
        #[doc = "Returns a finite standard spectral window."]
        #[no_mangle]
        pub unsafe extern "C" fn $name(length: i64, output: *mut Array) -> i32 {
            unsafe { window(length, $kind, 0.0, output) }
        }
    };
}

window_function!(terlan_ndarray_v1_bartlett, 0);
window_function!(terlan_ndarray_v1_blackman, 1);
window_function!(terlan_ndarray_v1_hamming, 2);
window_function!(terlan_ndarray_v1_hanning, 3);

/// Returns a Kaiser window with the supplied beta parameter.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_kaiser(
    length: i64,
    beta: f64,
    output: *mut Array,
) -> i32 {
    unsafe { window(length, 4, beta, output) }
}

/// Computes one-dimensional discrete convolution.
unsafe fn convolution(
    lhs: Array,
    rhs: Array,
    mode: i64,
    correlate: bool,
    output: *mut Array,
) -> i32 {
    if !matches!(mode, 0..=2) {
        return INVALID_ARGUMENT;
    }
    match unsafe { shape(lhs) } {
        Ok(value) if value.len() == 1 => {}
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    match unsafe { shape(rhs) } {
        Ok(value) if value.len() == 1 => {}
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let lhs = match unsafe { numeric(lhs, false) } {
        Ok(value) if !value.values.is_empty() => value,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let rhs = match unsafe { numeric(rhs, false) } {
        Ok(value) if !value.values.is_empty() => value,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let mut full = vec![Complex64::new(0.0, 0.0); lhs.values.len() + rhs.values.len() - 1];
    for (left_index, left) in lhs.values.iter().enumerate() {
        for (right_index, right) in rhs.values.iter().enumerate() {
            let right_index = if correlate {
                rhs.values.len() - 1 - right_index
            } else {
                right_index
            };
            let right = if correlate { right.conj() } else { *right };
            full[left_index + right_index] += left * right;
        }
    }
    let (start, length) = match mode {
        0 => (
            lhs.values.len().min(rhs.values.len()) - 1,
            lhs.values.len().abs_diff(rhs.values.len()) + 1,
        ),
        1 => (
            (full.len() - lhs.values.len().max(rhs.values.len())) / 2,
            lhs.values.len().max(rhs.values.len()),
        ),
        2 => (0, full.len()),
        _ => unreachable!(),
    };
    let values = &full[start..start + length];
    let dtype = match unsafe { promoted_dtype(lhs.dtype, rhs.dtype) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe { write_result(owned_numeric_as(values, &[length as i64], dtype), output) }
}

/// Computes one-dimensional discrete convolution.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_convolve(
    lhs: Array,
    rhs: Array,
    mode: i64,
    output: *mut Array,
) -> i32 {
    unsafe { convolution(lhs, rhs, mode, false, output) }
}

/// Computes one-dimensional cross-correlation.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_correlate(
    lhs: Array,
    rhs: Array,
    mode: i64,
    output: *mut Array,
) -> i32 {
    unsafe { convolution(lhs, rhs, mode, true, output) }
}

/// Returns floor-division quotient and remainder arrays.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_divmod(
    lhs: Array,
    rhs: Array,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let mut quotient = std::ptr::null_mut();
    let status = unsafe { crate::ffi::terlan_ndarray_v1_floor_divide(lhs, rhs, &mut quotient) };
    if status != OK {
        return status;
    }
    let mut remainder = std::ptr::null_mut();
    let status = unsafe { crate::ffi::terlan_ndarray_v1_remainder(lhs, rhs, &mut remainder) };
    if status != OK {
        unsafe { super::native_support::delete_arrays(&[quotient]) };
        return status;
    }
    unsafe { transfer_arrays(vec![quotient, remainder], output, output_length) }
}

/// Truncates real values toward zero.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_fix(array: Array, output: *mut Array) -> i32 {
    let dimensions = match unsafe { shape(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = match unsafe { super::native_support::floats(array, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = values.into_iter().map(f64::trunc).collect::<Vec<_>>();
    unsafe { write_result(owned_real(&values, &dimensions, 3), output) }
}

/// Raises real values to real powers with Float64 output.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_float_power(
    lhs: Array,
    rhs: Array,
    output: *mut Array,
) -> i32 {
    unsafe { binary_numeric(lhs, rhs, |lhs, rhs| lhs.powc(rhs), false, output) }
}

/// Computes C-style floating remainders.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_fmod(lhs: Array, rhs: Array, output: *mut Array) -> i32 {
    unsafe {
        binary_numeric(
            lhs,
            rhs,
            |lhs, rhs| Complex64::new(lhs.re % rhs.re, 0.0),
            true,
            output,
        )
    }
}

/// Computes the NumPy heaviside function.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_heaviside(
    lhs: Array,
    zero_value: Array,
    output: *mut Array,
) -> i32 {
    unsafe {
        binary_numeric(
            lhs,
            zero_value,
            |lhs, zero| {
                Complex64::new(
                    if lhs.re.is_nan() {
                        f64::NAN
                    } else if lhs.re < 0.0 {
                        0.0
                    } else if lhs.re > 0.0 {
                        1.0
                    } else {
                        zero.re
                    },
                    0.0,
                )
            },
            true,
            output,
        )
    }
}

/// Evaluates the modified Bessel function I0 elementwise.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_i0(array: Array, output: *mut Array) -> i32 {
    let dimensions = match unsafe { shape(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = match unsafe { super::native_support::floats(array, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = values.into_iter().map(bessel_i0).collect::<Vec<_>>();
    unsafe { write_result(owned_real(&values, &dimensions, 3), output) }
}

/// Returns real components as an owned array.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_real(array: Array, output: *mut Array) -> i32 {
    let dimensions = match unsafe { shape(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = match unsafe { numeric(array, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let reals = values
        .values
        .iter()
        .map(|value| Complex64::new(value.re, 0.0))
        .collect::<Vec<_>>();
    let output_dtype = match values.dtype {
        12 => 11,
        13 => 3,
        dtype => dtype,
    };
    unsafe { write_result(owned_numeric_as(&reals, &dimensions, output_dtype), output) }
}

/// Returns imaginary components as an owned array.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_imag(array: Array, output: *mut Array) -> i32 {
    let dimensions = match unsafe { shape(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = match unsafe { numeric(array, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let imags = values
        .values
        .iter()
        .map(|value| Complex64::new(value.im, 0.0))
        .collect::<Vec<_>>();
    let output_dtype = match values.dtype {
        12 => 11,
        13 => 3,
        dtype => dtype,
    };
    unsafe { write_result(owned_numeric_as(&imags, &dimensions, output_dtype), output) }
}

/// Returns whether each element has a nonzero imaginary component.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_iscomplex(array: Array, output: *mut Array) -> i32 {
    let dimensions = match unsafe { shape(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = match unsafe { numeric(array, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mask = values
        .values
        .iter()
        .map(|value| u8::from(value.im != 0.0))
        .collect::<Vec<_>>();
    unsafe { write_result(owned_bool(&mask, &dimensions), output) }
}

/// Returns whether each element has no imaginary component.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_isreal(array: Array, output: *mut Array) -> i32 {
    let dimensions = match unsafe { shape(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = match unsafe { numeric(array, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mask = values
        .values
        .iter()
        .map(|value| u8::from(value.im == 0.0))
        .collect::<Vec<_>>();
    unsafe { write_result(owned_bool(&mask, &dimensions), output) }
}

/// Returns whether the array stores complex scalars.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_iscomplexobj(array: Array, output: *mut bool) -> i32 {
    match unsafe { dtype(array) } {
        Ok(value) => unsafe { write_bool(matches!(value, 12 | 13), output) },
        Err(status) => status,
    }
}

/// Returns whether the array stores non-complex scalars.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_isrealobj(array: Array, output: *mut bool) -> i32 {
    match unsafe { dtype(array) } {
        Ok(value) => unsafe { write_bool(!matches!(value, 12 | 13), output) },
        Err(status) => status,
    }
}

/// Returns a mask selecting negative infinity.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_isneginf(array: Array, output: *mut Array) -> i32 {
    unsafe { infinity_mask(array, false, output) }
}

/// Returns a mask selecting positive infinity.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_isposinf(array: Array, output: *mut Array) -> i32 {
    unsafe { infinity_mask(array, true, output) }
}

/// Creates one signed infinity mask.
unsafe fn infinity_mask(array: Array, positive: bool, output: *mut Array) -> i32 {
    let dimensions = match unsafe { shape(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = match unsafe { super::native_support::floats(array, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mask = values
        .iter()
        .map(|value| u8::from(value.is_infinite() && value.is_sign_positive() == positive))
        .collect::<Vec<_>>();
    unsafe { write_result(owned_bool(&mask, &dimensions), output) }
}

/// Multiplies each value by an integral power of two.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_ldexp(
    values: Array,
    exponents: Array,
    output: *mut Array,
) -> i32 {
    unsafe {
        binary_numeric(
            values,
            exponents,
            |value, exponent| {
                let scaled = if value.re == 0.0 {
                    value.re
                } else if exponent.re > f64::from(i32::MAX) {
                    value.re.signum() * f64::INFINITY
                } else if exponent.re < f64::from(i32::MIN) {
                    value.re.signum() * 0.0
                } else {
                    value.re * 2.0f64.powi(exponent.re as i32)
                };
                Complex64::new(scaled, 0.0)
            },
            true,
            output,
        )
    }
}

#[no_mangle]
/// Replaces NaN and infinities with explicit finite values.
pub unsafe extern "C" fn terlan_ndarray_v1_nan_to_num(
    array: Array,
    nan: f64,
    positive_infinity: f64,
    negative_infinity: f64,
    output: *mut Array,
) -> i32 {
    let dimensions = match unsafe { shape(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = match unsafe { numeric(array, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let replace = |value: f64| {
        if value.is_nan() {
            nan
        } else if value == f64::INFINITY {
            positive_infinity
        } else if value == f64::NEG_INFINITY {
            negative_infinity
        } else {
            value
        }
    };
    let result = values
        .values
        .iter()
        .map(|value| Complex64::new(replace(value.re), replace(value.im)))
        .collect::<Vec<_>>();
    unsafe { write_result(owned_numeric_as(&result, &dimensions, values.dtype), output) }
}

/// Returns a real array when all imaginary parts are below tolerance.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_real_if_close(
    array: Array,
    tolerance: f64,
    output: *mut Array,
) -> i32 {
    if tolerance < 0.0 || !tolerance.is_finite() {
        return INVALID_ARGUMENT;
    }
    let dimensions = match unsafe { shape(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = match unsafe { numeric(array, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let effective_tolerance = if tolerance > 1.0 {
        tolerance
            * if narrow(&values) {
                f64::from(f32::EPSILON)
            } else {
                f64::EPSILON
            }
    } else {
        tolerance
    };
    let real = values
        .values
        .iter()
        .all(|value| value.im.abs() < effective_tolerance);
    unsafe {
        write_result(
            owned_numeric(
                &values.values,
                &dimensions,
                values.complex && !real,
                narrow(&values),
            ),
            output,
        )
    }
}

/// Evaluates normalized sinc, `sin(pi*x)/(pi*x)`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_sinc(array: Array, output: *mut Array) -> i32 {
    let dimensions = match unsafe { shape(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = match unsafe { super::native_support::floats(array, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let result = values
        .into_iter()
        .map(|value| {
            if value == 0.0 {
                1.0
            } else {
                let scaled = std::f64::consts::PI * value;
                scaled.sin() / scaled
            }
        })
        .collect::<Vec<_>>();
    unsafe { write_result(owned_real(&result, &dimensions, 3), output) }
}

/// Returns the distance to the adjacent representable number toward positive infinity.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_spacing(array: Array, output: *mut Array) -> i32 {
    let dimensions = match unsafe { shape(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = match unsafe { super::native_support::floats(array, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let source_dtype = match unsafe { dtype(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let (result, output_dtype) =
        super::float_spacing_support::spacing_values(&values, source_dtype);
    unsafe { write_result(owned_real(&result, &dimensions, output_dtype), output) }
}

/// Returns a one-dimensional complex array in lexicographic numeric order.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_sort_complex(array: Array, output: *mut Array) -> i32 {
    let dimensions = match unsafe { shape(array) } {
        Ok(value) if value.len() == 1 => value,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let values = match unsafe { numeric(array, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut result = values.values.clone();
    result.sort_by(|lhs, rhs| {
        lhs.re
            .total_cmp(&rhs.re)
            .then_with(|| lhs.im.total_cmp(&rhs.im))
    });
    let narrow_result = matches!(values.dtype, 1 | 4 | 5 | 7 | 8 | 11 | 12);
    unsafe {
        write_result(
            owned_numeric(&result, &dimensions, true, narrow_result),
            output,
        )
    }
}

/// Removes leading, trailing, or both runs of zero from a rank-one array.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_trim_zeros(
    array: Array,
    mode: i64,
    output: *mut Array,
) -> i32 {
    if !matches!(mode, 0..=2) {
        return INVALID_ARGUMENT;
    }
    match unsafe { shape(array) } {
        Ok(value) if value.len() == 1 => {}
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    }
    let values = match unsafe { numeric(array, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let first = if mode != 1 {
        values
            .values
            .iter()
            .position(|value| *value != Complex64::new(0.0, 0.0))
            .unwrap_or(values.values.len())
    } else {
        0
    };
    let end = if mode != 0 {
        values
            .values
            .iter()
            .rposition(|value| *value != Complex64::new(0.0, 0.0))
            .map(|index| index + 1)
            .unwrap_or(first)
    } else {
        values.values.len()
    };
    unsafe {
        crate::ffi::terlan_ndarray_v1_slice_axis(
            array,
            0,
            first.min(end) as i64,
            end as i64,
            1,
            output,
        )
    }
}

/// Returns sorted unique flattened values.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_unique(array: Array, output: *mut Array) -> i32 {
    let mut arrays = std::ptr::null_mut();
    let mut length = 0usize;
    let status =
        unsafe { crate::ffi::terlan_ndarray_v1_unique_all(array, &mut arrays, &mut length) };
    if status != OK {
        return status;
    }
    if arrays.is_null() || length != 4 {
        unsafe { crate::ffi::terlan_ndarray_v1_delete_handle_values(arrays) };
        return INVALID_ARGUMENT;
    }
    let values = unsafe { std::slice::from_raw_parts(arrays, length) };
    let result = values[0] as usize as Array;
    for value in &values[1..] {
        let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(*value as usize as Array) };
    }
    unsafe { crate::ffi::terlan_ndarray_v1_delete_handle_values(arrays) };
    unsafe { write_result(Ok(result), output) }
}

/// Alias for integer bitwise inversion.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_invert(array: Array, output: *mut Array) -> i32 {
    unsafe { crate::ffi::terlan_ndarray_v1_bitwise_not(array, output) }
}

/// Alias for truncation toward zero.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_trunc(array: Array, output: *mut Array) -> i32 {
    unsafe { terlan_ndarray_v1_fix(array, output) }
}

#[cfg(test)]
#[path = "ndarray_numpy_numeric_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "ndarray_numpy_misc_tests.rs"]
mod misc_tests;
