//! NumPy-compatible type classification and non-finite replacement extensions.

use super::native_support::{
    delete_arrays, dtype, shape, Array, DTYPE_MISMATCH, OK, SHAPE_MISMATCH,
};

/// Returns the finite replacement magnitude for one canonical numeric dtype.
fn finite_maximum(dtype: i32) -> f64 {
    match dtype {
        11 | 12 => f64::from(f32::MAX),
        14 => 65_504.0,
        _ => f64::MAX,
    }
}

/// Replaces non-finite values using NumPy's dtype-specific defaults.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_nan_to_num_default(
    array: Array,
    output: *mut Array,
) -> i32 {
    let dtype = match unsafe { dtype(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let maximum = finite_maximum(dtype);
    unsafe {
        super::numpy_numeric_extension::terlan_ndarray_v1_nan_to_num(
            array, 0.0, maximum, -maximum, output,
        )
    }
}

/// Writes non-finite replacements into an exact-shape, exact-dtype destination.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_nan_to_num_into(
    destination: Array,
    array: Array,
    nan: f64,
    positive_infinity: f64,
    negative_infinity: f64,
) -> i32 {
    let destination_shape = match unsafe { shape(destination) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let source_shape = match unsafe { shape(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if destination_shape != source_shape {
        return SHAPE_MISMATCH;
    }
    let destination_dtype = match unsafe { dtype(destination) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let source_dtype = match unsafe { dtype(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if destination_dtype != source_dtype {
        return DTYPE_MISMATCH;
    }
    let mut staged = std::ptr::null_mut();
    let status = unsafe {
        super::numpy_numeric_extension::terlan_ndarray_v1_nan_to_num(
            array,
            nan,
            positive_infinity,
            negative_infinity,
            &mut staged,
        )
    };
    if status != OK {
        return status;
    }
    let status = unsafe { crate::ffi::terlan_ndarray_v1_copyto_casting(destination, staged, 0) };
    unsafe { delete_arrays(&[staged]) };
    status
}

#[cfg(test)]
#[path = "ndarray_type_check_tests.rs"]
mod tests;
