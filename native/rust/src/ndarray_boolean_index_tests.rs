//! Boolean prefix-index selection and assignment tests.

use super::super::native_support::{
    delete_arrays, ints, owned_bool, owned_int, shape, Array, OK, SHAPE_MISMATCH,
};

/// Creates one owned integer array.
unsafe fn integer_array(values: &[i64], dimensions: &[i64]) -> Array {
    unsafe { owned_int(values, dimensions).unwrap() }
}

/// Creates one owned boolean mask.
unsafe fn boolean_array(values: &[u8], dimensions: &[i64]) -> Array {
    unsafe { owned_bool(values, dimensions).unwrap() }
}

/// Reads logical integer values.
unsafe fn values(array: Array) -> Vec<i64> {
    unsafe { ints(array).unwrap() }
}

/// Deletes all supplied arrays.
unsafe fn dispose(arrays: &[Array]) {
    unsafe { delete_arrays(arrays) }
}

#[test]
fn prefix_masks_preserve_unmasked_suffix_dimensions() {
    unsafe {
        let vector = integer_array(&[1, 2, 3, 4], &[4]);
        let vector_mask = boolean_array(&[0, 1, 0, 0], &[4]);
        let mut vector_selected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_mask_select(vector, vector_mask, &mut vector_selected),
            OK
        );
        assert_eq!(values(vector_selected), [2]);
        let five = integer_array(&[5], &[]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_mask_assign(vector, vector_mask, five),
            OK
        );
        assert_eq!(values(vector), [1, 5, 3, 4]);

        let matrix = integer_array(&[1, 2, 3, 4, 5, 6, 7, 8, 9], &[3, 3]);
        let rows = boolean_array(&[0, 1, 0], &[3]);
        let mut selected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_mask_select(matrix, rows, &mut selected),
            OK
        );
        assert_eq!(shape(selected).unwrap(), [1, 3]);
        assert_eq!(values(selected), [4, 5, 6]);

        let replacement = integer_array(&[10, 11, 12], &[1, 3]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_mask_assign(matrix, rows, replacement),
            OK
        );
        assert_eq!(values(matrix), [1, 2, 3, 10, 11, 12, 7, 8, 9]);

        let element_mask = boolean_array(&[1, 0, 1, 0, 1, 0, 1, 0, 1], &[3, 3]);
        let mut elements = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_mask_select(matrix, element_mask, &mut elements),
            OK
        );
        assert_eq!(shape(elements).unwrap(), [5]);
        assert_eq!(values(elements), [1, 3, 11, 7, 9]);
        dispose(&[
            elements,
            element_mask,
            replacement,
            selected,
            rows,
            matrix,
            five,
            vector_selected,
            vector_mask,
            vector,
        ]);
    }
}

#[test]
fn scalar_boolean_masks_add_a_selection_axis() {
    unsafe {
        let matrix = integer_array(&[1, 2, 3, 4, 5, 6], &[2, 3]);
        let selected_mask = boolean_array(&[1], &[]);
        let rejected_mask = boolean_array(&[0], &[]);
        let mut selected = std::ptr::null_mut();
        let mut rejected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_mask_select(matrix, selected_mask, &mut selected,),
            OK
        );
        assert_eq!(shape(selected).unwrap(), [1, 2, 3]);
        assert_eq!(values(selected), values(matrix));
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_mask_select(matrix, rejected_mask, &mut rejected,),
            OK
        );
        assert_eq!(shape(rejected).unwrap(), [0, 2, 3]);
        assert!(values(rejected).is_empty());
        dispose(&[rejected, selected, rejected_mask, selected_mask, matrix]);
    }
}

#[test]
fn boolean_assignment_rejects_incompatible_values_atomically() {
    unsafe {
        let vector = integer_array(&[0, 1, 2, 3], &[4]);
        let all = boolean_array(&[1, 1, 1, 1], &[4]);
        let empty = integer_array(&[], &[0]);
        let short = integer_array(&[9, 8, 7], &[3]);
        for replacement in [empty, short] {
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_mask_assign(vector, all, replacement),
                SHAPE_MISMATCH
            );
            assert_eq!(values(vector), [0, 1, 2, 3]);
        }

        let wrong_prefix = boolean_array(&[1], &[1]);
        let mut output = 1_usize as Array;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_mask_select(vector, wrong_prefix, &mut output),
            SHAPE_MISMATCH
        );
        assert!(output.is_null());
        dispose(&[wrong_prefix, short, empty, all, vector]);
    }
}

#[test]
fn scalar_full_and_prefix_masks_cover_automated_boolean_indexing() {
    unsafe {
        let scalar = integer_array(&[5], &[]);
        let scalar_mask = boolean_array(&[1], &[]);
        let mut scalar_selected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_mask_select(scalar, scalar_mask, &mut scalar_selected,),
            OK
        );
        assert_eq!(shape(scalar_selected).unwrap(), [1]);
        assert_eq!(values(scalar_selected), [5]);
        let one = integer_array(&[1], &[]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_mask_assign(scalar, scalar_mask, one),
            OK
        );
        assert_eq!(values(scalar), [1]);

        let source = integer_array(&(0..90).collect::<Vec<_>>(), &[3, 1, 5, 6]);
        let full_mask = boolean_array(&[0; 90], &[3, 1, 5, 6]);
        let prefix_mask = boolean_array(&[0; 15], &[3, 1, 5]);
        for (mask, expected_shape) in [(full_mask, &[0_i64][..]), (prefix_mask, &[0_i64, 6][..])] {
            let mut selected = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_mask_select(source, mask, &mut selected),
                OK
            );
            assert_eq!(shape(selected).unwrap(), expected_shape);
            assert!(values(selected).is_empty());
            dispose(&[selected]);
        }

        let extra_axis = boolean_array(&[0; 90], &[1, 3, 1, 5, 6]);
        let mut rejected = 1_usize as Array;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_mask_select(source, extra_axis, &mut rejected),
            SHAPE_MISMATCH
        );
        assert!(rejected.is_null());
        dispose(&[
            extra_axis,
            prefix_mask,
            full_mask,
            source,
            one,
            scalar_selected,
            scalar_mask,
            scalar,
        ]);
    }
}
