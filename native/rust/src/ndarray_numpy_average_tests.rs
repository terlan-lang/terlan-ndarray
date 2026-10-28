//! Tests for NumPy-compatible returned-average operations.

use super::*;
use crate::ffi;

/// Creates one Float64 array for returned-average tests.
unsafe fn array(values: &[f64], shape: &[i64]) -> Array {
    let mut output = std::ptr::null_mut();
    let status = unsafe {
        ffi::terlan_ndarray_v1_from_floats(
            values.as_ptr(),
            values.len() as i64,
            shape.as_ptr(),
            shape.len() as i64,
            &mut output,
        )
    };
    assert_eq!(status, OK);
    output
}

/// Reads one real array as owned Float64 values.
unsafe fn values(array: Array) -> Vec<f64> {
    let mut output = std::ptr::null_mut();
    let mut length = 0usize;
    let status = unsafe { ffi::terlan_ndarray_v1_to_floats_cast(array, &mut output, &mut length) };
    assert_eq!(status, OK);
    let result = unsafe { std::slice::from_raw_parts(output, length) }.to_vec();
    unsafe { ffi::terlan_ndarray_v1_delete_float_values(output) };
    result
}

/// Deletes every array identity and the transferred identity buffer.
unsafe fn delete_returned(output: *mut u64, length: usize) {
    let arrays = unsafe { std::slice::from_raw_parts(output, length) };
    for array in arrays {
        assert_eq!(
            unsafe { ffi::terlan_ndarray_v1_delete(*array as usize as Array) },
            OK
        );
    }
    unsafe { ffi::terlan_ndarray_v1_delete_handle_values(output) };
}

#[test]
fn returned_average_publishes_values_and_weight_sums_in_order() {
    unsafe {
        let source = array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let weights = array(&[1.0, 2.0, 3.0], &[3]);
        let axis = 1i64;
        let mut output = std::ptr::null_mut();
        let mut length = 0usize;
        assert_eq!(
            terlan_ndarray_v1_weighted_average_with_sum(
                source,
                weights,
                &axis,
                1,
                0,
                &mut output,
                &mut length,
            ),
            OK
        );
        assert_eq!(length, 2);
        let arrays = std::slice::from_raw_parts(output, length);
        assert_eq!(
            values(arrays[0] as usize as Array),
            vec![14.0 / 6.0, 32.0 / 6.0]
        );
        assert_eq!(values(arrays[1] as usize as Array), vec![6.0, 6.0]);
        delete_returned(output, length);
        assert_eq!(ffi::terlan_ndarray_v1_delete(weights), OK);
        assert_eq!(ffi::terlan_ndarray_v1_delete(source), OK);
    }
}

#[test]
fn returned_average_rejects_null_outputs_without_leaking_arrays() {
    unsafe {
        let source = array(&[1.0, 2.0], &[2]);
        assert_ne!(
            terlan_ndarray_v1_average_with_sum(
                source,
                std::ptr::null(),
                0,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ),
            OK
        );
        assert_eq!(ffi::terlan_ndarray_v1_delete(source), OK);
    }
}
