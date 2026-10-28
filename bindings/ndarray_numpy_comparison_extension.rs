//! NumPy-compatible elementwise and NaN-aware comparison operations.

use super::native_support::{owned_bool, shape, write_result, Array, INVALID_ARGUMENT};
use super::numpy_numeric_extension::{binary_predicate, close_values, write_bool};

/// Returns elementwise closeness using NumPy's asymmetric tolerance rule.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_isclose(
    lhs: Array,
    rhs: Array,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    equal_nan: i32,
    output: *mut Array,
) -> i32 {
    if relative_tolerance < 0.0 || absolute_tolerance < 0.0 {
        return INVALID_ARGUMENT;
    }
    match unsafe {
        binary_predicate(lhs, rhs, |lhs, rhs| {
            close_values(
                lhs,
                rhs,
                relative_tolerance,
                absolute_tolerance,
                equal_nan != 0,
            )
        })
    } {
        Ok((values, dimensions)) => unsafe {
            write_result(owned_bool(&values, &dimensions), output)
        },
        Err(status) => status,
    }
}

/// Returns strict shape-and-value equality while treating paired NaNs as equal.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_array_equal_nan(
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
    let result = unsafe {
        binary_predicate(lhs, rhs, |lhs, rhs| {
            lhs == rhs
                || ((lhs.re.is_nan() || lhs.im.is_nan()) && (rhs.re.is_nan() || rhs.im.is_nan()))
        })
    };
    match result {
        Ok((values, _)) => unsafe { write_bool(values.iter().all(|value| *value != 0), output) },
        Err(status) => status,
    }
}
