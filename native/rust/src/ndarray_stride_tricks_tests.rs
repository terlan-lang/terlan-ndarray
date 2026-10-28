//! NumPy stride-trick parity tests over package-owned shared views.

use super::native_support::{delete_arrays, ints, owned_int, shape, Array, OK, SHAPE_MISMATCH};

const INVALID_ARGUMENT: i32 = 100;
const NEGATIVE_DIMENSION: i32 = 101;
const AXIS_OUT_OF_RANGE: i32 = 110;
const READ_ONLY: i32 = 121;

/// Returns whether two live arrays share package-owned storage.
unsafe fn shares(lhs: Array, rhs: Array) -> bool {
    let mut result = false;
    assert_eq!(
        unsafe { crate::ffi::terlan_ndarray_v1_shares_memory(lhs, rhs, &mut result) },
        OK
    );
    result
}

#[test]
fn as_strided_builds_checked_read_only_shared_views() {
    unsafe {
        let source = owned_int(&[1, 2, 3, 4], &[4]).unwrap();
        let mut every_other = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_as_strided(
                source,
                [2_i64].as_ptr(),
                [2_i64].as_ptr(),
                1,
                1,
                &mut every_other,
            ),
            OK
        );
        assert_eq!(shape(every_other).unwrap(), [2]);
        assert_eq!(ints(every_other).unwrap(), [1, 3]);
        assert!(shares(source, every_other));
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_set_int(every_other, [0_i64].as_ptr(), 1, 9),
            READ_ONLY
        );

        let mut repeated = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_as_strided(
                source,
                [3_i64, 4].as_ptr(),
                [0_i64, 1].as_ptr(),
                2,
                2,
                &mut repeated,
            ),
            OK
        );
        assert_eq!(
            ints(repeated).unwrap(),
            [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        );
        assert!(shares(source, repeated));
        delete_arrays(&[repeated, every_other, source]);
    }
}

#[test]
fn sliding_windows_cover_one_two_selected_and_repeated_axes() {
    unsafe {
        let line = owned_int(&[0, 1, 2, 3, 4], &[5]).unwrap();
        let mut line_windows = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_sliding_window_view(
                line,
                [2_i64].as_ptr(),
                1,
                std::ptr::null(),
                0,
                &mut line_windows,
            ),
            OK
        );
        assert_eq!(shape(line_windows).unwrap(), [4, 2]);
        assert_eq!(ints(line_windows).unwrap(), [0, 1, 1, 2, 2, 3, 3, 4]);
        assert!(shares(line, line_windows));

        let matrix = owned_int(&[0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23], &[3, 4]).unwrap();
        let mut all_axes = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_sliding_window_view(
                matrix,
                [2_i64, 2].as_ptr(),
                2,
                std::ptr::null(),
                0,
                &mut all_axes,
            ),
            OK
        );
        assert_eq!(shape(all_axes).unwrap(), [2, 3, 2, 2]);
        assert_eq!(
            ints(all_axes).unwrap(),
            [
                0, 1, 10, 11, 1, 2, 11, 12, 2, 3, 12, 13, 10, 11, 20, 21, 11, 12, 21, 22, 12, 13,
                22, 23,
            ]
        );

        let mut selected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_sliding_window_view(
                matrix,
                [3_i64].as_ptr(),
                1,
                [0_i64].as_ptr(),
                1,
                &mut selected,
            ),
            OK
        );
        assert_eq!(shape(selected).unwrap(), [1, 4, 3]);
        assert_eq!(
            ints(selected).unwrap(),
            [0, 10, 20, 1, 11, 21, 2, 12, 22, 3, 13, 23]
        );

        let mut repeated = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_sliding_window_view(
                matrix,
                [2_i64, 3].as_ptr(),
                2,
                [1_i64, -1].as_ptr(),
                2,
                &mut repeated,
            ),
            OK
        );
        assert_eq!(shape(repeated).unwrap(), [3, 1, 2, 3]);
        assert_eq!(
            ints(repeated).unwrap(),
            [0, 1, 2, 1, 2, 3, 10, 11, 12, 11, 12, 13, 20, 21, 22, 21, 22, 23,]
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_set_int(repeated, [0_i64; 4].as_ptr(), 4, 99),
            READ_ONLY
        );

        delete_arrays(&[repeated, selected, all_axes, matrix, line_windows, line]);
    }
}

#[test]
fn stride_tricks_reject_invalid_shapes_axes_and_storage_escape() {
    unsafe {
        let source = owned_int(&[1, 2, 3, 4], &[2, 2]).unwrap();
        let mut output = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_sliding_window_view(
                source,
                [-1_i64, 1].as_ptr(),
                2,
                std::ptr::null(),
                0,
                &mut output,
            ),
            NEGATIVE_DIMENSION
        );
        assert!(output.is_null());
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_sliding_window_view(
                source,
                [3_i64, 1].as_ptr(),
                2,
                std::ptr::null(),
                0,
                &mut output,
            ),
            SHAPE_MISMATCH
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_sliding_window_view(
                source,
                [1_i64].as_ptr(),
                1,
                std::ptr::null(),
                0,
                &mut output,
            ),
            SHAPE_MISMATCH
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_sliding_window_view(
                source,
                [1_i64].as_ptr(),
                1,
                [2_i64].as_ptr(),
                1,
                &mut output,
            ),
            AXIS_OUT_OF_RANGE
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_sliding_window_view(
                source,
                [1_i64, 1].as_ptr(),
                2,
                [0_i64].as_ptr(),
                1,
                &mut output,
            ),
            SHAPE_MISMATCH
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_as_strided(
                source,
                [3_i64].as_ptr(),
                [2_i64].as_ptr(),
                1,
                1,
                &mut output,
            ),
            INVALID_ARGUMENT
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_as_strided(
                source,
                std::ptr::null(),
                std::ptr::null(),
                1,
                1,
                &mut output,
            ),
            INVALID_ARGUMENT
        );
        delete_arrays(&[source]);
    }
}

#[test]
fn broadcast_stride_cases_cover_scalars_zero_dimensions_and_many_inputs() {
    unsafe {
        let scalar = owned_int(&[0], &[]).unwrap();
        let row = owned_int(&[0, 1, 2], &[1, 3]).unwrap();
        let column = owned_int(&[1, 2, 3], &[3, 1]).unwrap();
        let mut scalar_view = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_broadcast_to(
                scalar,
                [0_i64, 2].as_ptr(),
                2,
                &mut scalar_view,
            ),
            OK
        );
        assert_eq!(shape(scalar_view).unwrap(), [0, 2]);

        let identities = [column as usize as u64];
        let mut values = std::ptr::null_mut();
        let mut length = 0_usize;
        assert_eq!(
            super::numpy_shape_extension::terlan_ndarray_v1_broadcast_arrays(
                row,
                identities.as_ptr(),
                1,
                &mut values,
                &mut length,
            ),
            OK
        );
        assert_eq!(length, 2);
        let results = std::slice::from_raw_parts(values, length)
            .iter()
            .map(|identity| *identity as usize as Array)
            .collect::<Vec<_>>();
        assert_eq!(shape(results[0]).unwrap(), [3, 3]);
        assert_eq!(ints(results[0]).unwrap(), [0, 1, 2, 0, 1, 2, 0, 1, 2]);
        assert_eq!(ints(results[1]).unwrap(), [1, 1, 1, 2, 2, 2, 3, 3, 3]);
        delete_arrays(&results);
        crate::ffi::terlan_ndarray_v1_delete_handle_values(values);

        let mut common = std::ptr::null_mut();
        assert_eq!(
            super::numpy_shape_extension::terlan_ndarray_v1_broadcast_shapes(
                [1_i64, 0].as_ptr(),
                2,
                [0_i64, 1].as_ptr(),
                2,
                &mut common,
            ),
            OK
        );
        assert_eq!(ints(common).unwrap(), [0, 0]);
        let mut incompatible = std::ptr::null_mut();
        assert_eq!(
            super::numpy_shape_extension::terlan_ndarray_v1_broadcast_shapes(
                [2_i64, 3].as_ptr(),
                2,
                [2_i64].as_ptr(),
                1,
                &mut incompatible,
            ),
            SHAPE_MISMATCH
        );
        assert!(incompatible.is_null());
        delete_arrays(&[common, scalar_view, column, row, scalar]);
    }
}
