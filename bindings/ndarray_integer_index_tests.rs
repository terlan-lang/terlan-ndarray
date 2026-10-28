//! Signed and unsigned integer indexing tests.

use super::super::native_support::{
    delete_arrays, ints, owned_int, owned_int_as, owned_real, owned_uint, Array, OK,
};

const DTYPE_MISMATCH: i32 = 106;
const INDEX_OUT_OF_RANGE: i32 = 120;

/// Reads logical integer values.
unsafe fn values(array: Array) -> Vec<i64> {
    unsafe { ints(array).unwrap() }
}

/// Deletes all supplied arrays.
unsafe fn dispose(arrays: &[Array]) {
    unsafe { delete_arrays(arrays) }
}

#[test]
fn every_integer_dtype_indexes_and_assigns_with_same_kind_semantics() {
    unsafe {
        for dtype in [2, 4, 5, 6, 7, 8, 9, 10] {
            let source = owned_int(&[10, 20, 30], &[3]).unwrap();
            let indexes = owned_int_as(&[2, 0, 1], &[3], dtype).unwrap();
            let mut selected = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_take_along_axis(source, indexes, 0, &mut selected,),
                OK
            );
            assert_eq!(values(selected), [30, 10, 20]);

            let replacement = owned_int(&[3, 1, 2], &[3]).unwrap();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_put_along_axis(source, indexes, replacement, 0,),
                OK
            );
            assert_eq!(values(source), [1, 2, 3]);
            dispose(&[replacement, selected, indexes, source]);
        }
    }
}

#[test]
fn singleton_axis_indexes_preserve_dimensions_for_selection_and_assignment() {
    unsafe {
        let matrix = owned_int(&[1], &[1, 1]).unwrap();
        let matrix_index = owned_int(&[0], &[1, 1]).unwrap();
        let mut matrix_selected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_take_along_axis(
                matrix,
                matrix_index,
                1,
                &mut matrix_selected,
            ),
            OK
        );
        assert_eq!(values(matrix_selected), [1]);
        let two = owned_int(&[2], &[]).unwrap();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put_along_axis(matrix, matrix_index, two, 1),
            OK
        );
        assert_eq!(values(matrix), [2]);

        let cube = owned_int(&[1], &[1, 1, 1]).unwrap();
        let cube_index = owned_int(&[0], &[1, 1, 1]).unwrap();
        let mut cube_selected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_take_along_axis(cube, cube_index, 2, &mut cube_selected,),
            OK
        );
        assert_eq!(values(cube_selected), [1]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put_along_axis(cube, cube_index, two, 2),
            OK
        );
        assert_eq!(values(cube), [2]);
        dispose(&[
            cube_selected,
            cube_index,
            cube,
            two,
            matrix_selected,
            matrix_index,
            matrix,
        ]);
    }
}

#[test]
fn unsigned_coordinate_grids_select_and_assign() {
    unsafe {
        let matrix = owned_int(&[1, 2, 3, 4], &[2, 2]).unwrap();
        let coordinates = owned_int_as(&[1, 1, 0, 0], &[2, 2], 7).unwrap();
        let mut selected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_gather_coordinates(matrix, coordinates, &mut selected,),
            OK
        );
        assert_eq!(values(selected), [4, 1]);
        let replacement = owned_int(&[8, 9], &[2]).unwrap();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put_coordinates(matrix, coordinates, replacement,),
            OK
        );
        assert_eq!(values(matrix), [9, 2, 3, 8]);
        dispose(&[replacement, selected, coordinates, matrix]);
    }
}

#[test]
fn non_integer_and_unrepresentable_unsigned_indexes_fail_explicitly() {
    unsafe {
        let source = owned_int(&[1, 2, 3], &[3]).unwrap();
        let float_indexes = owned_real(&[0.0, 1.0], &[2], 3).unwrap();
        let huge = owned_uint(&[u64::MAX], &[1]).unwrap();
        for (indexes, expected) in [(float_indexes, DTYPE_MISMATCH), (huge, INDEX_OUT_OF_RANGE)] {
            let mut selected = 1_usize as Array;
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_take_along_axis(source, indexes, 0, &mut selected,),
                expected
            );
            assert!(selected.is_null());
        }
        dispose(&[huge, float_indexes, source]);
    }
}
