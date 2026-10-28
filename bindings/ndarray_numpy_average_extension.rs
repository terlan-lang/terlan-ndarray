//! NumPy-compatible returned-average operations.

use super::native_support::{delete_arrays, transfer_arrays, Array, OK};

/// Computes an average and its broadcast weight sum, then transfers both atomically.
unsafe fn average_with_sum(
    array: Array,
    weights: Option<Array>,
    axes: *const i64,
    axis_count: i64,
    keep_dims: i32,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let mut average = std::ptr::null_mut();
    let status = match weights {
        Some(weights) => unsafe {
            crate::ffi::terlan_ndarray_v1_weighted_average_axes(
                array,
                weights,
                axes,
                axis_count,
                keep_dims,
                &mut average,
            )
        },
        None => unsafe {
            crate::ffi::terlan_ndarray_v1_average(array, axes, axis_count, keep_dims, &mut average)
        },
    };
    if status != OK {
        return status;
    }
    let mut weight_sum = std::ptr::null_mut();
    let status = match weights {
        Some(weights) => unsafe {
            crate::ffi::terlan_ndarray_v1_weighted_average_weight_sum(
                array,
                weights,
                axes,
                axis_count,
                keep_dims,
                &mut weight_sum,
            )
        },
        None => unsafe {
            crate::ffi::terlan_ndarray_v1_average_weight_sum(
                array,
                axes,
                axis_count,
                keep_dims,
                &mut weight_sum,
            )
        },
    };
    if status != OK {
        unsafe { delete_arrays(&[average]) };
        return status;
    }
    unsafe { transfer_arrays(vec![average, weight_sum], output, output_length) }
}

/// Returns an unweighted average followed by its reduction cardinality array.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_average_with_sum(
    array: Array,
    axes: *const i64,
    axis_count: i64,
    keep_dims: i32,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    unsafe {
        average_with_sum(
            array,
            None,
            axes,
            axis_count,
            keep_dims,
            output,
            output_length,
        )
    }
}

/// Returns a weighted average followed by the matching weight-sum array.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_weighted_average_with_sum(
    array: Array,
    weights: Array,
    axes: *const i64,
    axis_count: i64,
    keep_dims: i32,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    unsafe {
        average_with_sum(
            array,
            Some(weights),
            axes,
            axis_count,
            keep_dims,
            output,
            output_length,
        )
    }
}

#[cfg(test)]
#[path = "ndarray_numpy_average_tests.rs"]
mod tests;
