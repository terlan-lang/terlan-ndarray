//! NumPy `test_function_base` conformance tests for admitted numerical contracts.

use super::native_support::{
    delete_arrays, dtype, floats, ints, owned_bool, owned_int, owned_real, owned_uint, shape,
    uints, Array, DOMAIN_ERROR, INVALID_ARGUMENT, OK,
};
use rustfft::num_complex::Complex64;

/// Returns one array's C- and Fortran-contiguity flags.
unsafe fn contiguous_flags(array: Array) -> (bool, bool) {
    let mut c = 0;
    let mut fortran = 0;
    assert_eq!(
        unsafe { crate::ffi::terlan_ndarray_v1_is_contiguous(array, &mut c) },
        OK
    );
    assert_eq!(
        unsafe { crate::ffi::terlan_ndarray_v1_is_fortran_contiguous(array, &mut fortran) },
        OK
    );
    (c == 1, fortran == 1)
}

/// Converts handles into their stable ABI integer representation.
fn handles(arrays: &[Array]) -> Vec<u64> {
    arrays.iter().map(|array| *array as usize as u64).collect()
}

#[test]
fn function_base_reductions_match_numpy_axes_values_and_accumulator_rules() {
    unsafe {
        let values = [3.0, 6.0, 9.0, 4.0, 10.0, 5.0, 8.0, 3.0, 2.0];
        let matrix = owned_real(&values, &[3, 3], 3).unwrap();
        let axis_zero = [0_i64];
        let axis_one = [1_i64];
        let both_axes = [0_i64, 1_i64];
        let mut maximum = std::ptr::null_mut();
        let mut minimum = std::ptr::null_mut();
        let mut peak = std::ptr::null_mut();
        let mut kept_peak = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_maximum(matrix, axis_zero.as_ptr(), 1, 0, &mut maximum,),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_minimum(matrix, axis_one.as_ptr(), 1, 0, &mut minimum,),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_peak_to_peak(matrix, axis_zero.as_ptr(), 1, 0, &mut peak,),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_peak_to_peak(
                matrix,
                both_axes.as_ptr(),
                2,
                1,
                &mut kept_peak,
            ),
            OK
        );
        assert_eq!(floats(maximum, false).unwrap(), [8.0, 10.0, 9.0]);
        assert_eq!(floats(minimum, false).unwrap(), [3.0, 4.0, 2.0]);
        assert_eq!(floats(peak, false).unwrap(), [5.0, 7.0, 7.0]);
        assert_eq!(shape(kept_peak).unwrap(), [1, 1]);
        assert_eq!(floats(kept_peak, false).unwrap(), [8.0]);

        let integers = owned_int(&[1, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let mut product = std::ptr::null_mut();
        let mut cumulative_sum = std::ptr::null_mut();
        let mut cumulative_product = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_product(integers, axis_one.as_ptr(), 1, 0, &mut product,),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_cumulative_sum(integers, 0, &mut cumulative_sum),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_cumulative_product(integers, -1, &mut cumulative_product,),
            OK
        );
        assert_eq!(ints(product).unwrap(), [6, 120]);
        assert_eq!(ints(cumulative_sum).unwrap(), [1, 2, 3, 5, 7, 9]);
        assert_eq!(ints(cumulative_product).unwrap(), [1, 2, 6, 4, 20, 120]);
        assert_eq!(dtype(cumulative_sum).unwrap(), 2);

        delete_arrays(&[
            cumulative_product,
            cumulative_sum,
            product,
            integers,
            kept_peak,
            peak,
            minimum,
            maximum,
            matrix,
        ]);
    }
}

#[test]
fn function_base_copy_and_chkfinite_preserve_layout_and_reject_nonfinite_values() {
    unsafe {
        let source = owned_real(&[1.0, 2.0, 3.0, 4.0], &[2, 2], 3).unwrap();
        let mut fortran = std::ptr::null_mut();
        let mut copied = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copy_ordered(source, 1, &mut fortran),
            OK
        );
        assert_eq!(contiguous_flags(fortran), (false, true));
        assert_eq!(crate::ffi::terlan_ndarray_v1_copy(fortran, &mut copied), OK);
        assert_eq!(contiguous_flags(copied), (false, true));
        assert_eq!(floats(copied, false).unwrap(), [1.0, 2.0, 3.0, 4.0]);
        let replacement = owned_real(&[10.0], &[], 3).unwrap();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put(copied, [0_i64].as_ptr(), 1, replacement),
            OK
        );
        assert_eq!(floats(source, false).unwrap(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(floats(copied, false).unwrap()[0], 10.0);

        let finite = owned_real(&[1.0, 2.0, 3.0], &[3], 11).unwrap();
        let infinity = owned_real(&[1.0, f64::INFINITY], &[2], 3).unwrap();
        let not_a_number = owned_real(&[1.0, f64::NAN], &[2], 3).unwrap();
        let mut checked = std::ptr::null_mut();
        assert_eq!(
            super::numpy_numeric_extension::terlan_ndarray_v1_asarray_chkfinite(
                finite,
                &mut checked,
            ),
            OK
        );
        assert_eq!(dtype(checked).unwrap(), 11);
        assert_eq!(floats(checked, false).unwrap(), [1.0, 2.0, 3.0]);
        let mut ordered = std::ptr::null_mut();
        assert_eq!(
            super::numpy_numeric_extension::terlan_ndarray_v1_asarray_chkfinite_as(
                source,
                3,
                1,
                &mut ordered,
            ),
            OK
        );
        assert_eq!(dtype(ordered).unwrap(), 3);
        assert_eq!(contiguous_flags(ordered), (false, true));
        assert_eq!(
            super::numpy_numeric_extension::terlan_ndarray_v1_asarray_chkfinite(
                infinity,
                &mut std::ptr::null_mut(),
            ),
            DOMAIN_ERROR
        );
        assert_eq!(
            super::numpy_numeric_extension::terlan_ndarray_v1_asarray_chkfinite(
                not_a_number,
                &mut std::ptr::null_mut(),
            ),
            DOMAIN_ERROR
        );

        delete_arrays(&[
            checked,
            ordered,
            not_a_number,
            infinity,
            finite,
            replacement,
            copied,
            fortran,
            source,
        ]);
    }
}

#[test]
fn function_base_select_broadcasts_promotes_and_accepts_many_conditions() {
    unsafe {
        let false_condition = owned_bool(&[0, 0, 0], &[3]).unwrap();
        let middle_condition = owned_bool(&[0, 1, 0], &[3]).unwrap();
        let final_condition = owned_bool(&[0, 0, 1], &[3]).unwrap();
        let first = owned_int(&[1, 2, 3], &[3]).unwrap();
        let second = owned_int(&[4, 5, 6], &[3]).unwrap();
        let third = owned_int(&[7, 8, 9], &[3]).unwrap();
        let default = owned_real(&[15.0], &[], 3).unwrap();
        let condition_handles = handles(&[false_condition, middle_condition, final_condition]);
        let choice_handles = handles(&[first, second, third]);
        let mut selected = std::ptr::null_mut();
        assert_eq!(
            super::numpy_index_extension::terlan_ndarray_v1_select(
                condition_handles.as_ptr(),
                3,
                choice_handles.as_ptr(),
                3,
                default,
                &mut selected,
            ),
            OK
        );
        assert_eq!(ints(selected).unwrap(), [15, 5, 9]);
        assert_eq!(dtype(selected).unwrap(), 2);

        let int8_first = super::native_support::owned_int_as(&[1, 2, 3], &[3], 4).unwrap();
        let int8_second = super::native_support::owned_int_as(&[4, 5, 6], &[3], 4).unwrap();
        let int8_third = super::native_support::owned_int_as(&[7, 8, 9], &[3], 4).unwrap();
        let zero_default = owned_int(&[0], &[]).unwrap();
        let int8_choices = handles(&[int8_first, int8_second, int8_third]);
        let mut selected_int8 = std::ptr::null_mut();
        assert_eq!(
            super::numpy_index_extension::terlan_ndarray_v1_select(
                condition_handles.as_ptr(),
                3,
                int8_choices.as_ptr(),
                3,
                zero_default,
                &mut selected_int8,
            ),
            OK
        );
        assert_eq!(dtype(selected_int8).unwrap(), 4);
        assert_eq!(ints(selected_int8).unwrap(), [0, 5, 9]);

        let imaginary_default =
            super::native_support::owned_complex(&[Complex64::new(0.0, 1.0)], &[], 13).unwrap();
        let mut selected_complex = std::ptr::null_mut();
        assert_eq!(
            super::numpy_index_extension::terlan_ndarray_v1_select(
                condition_handles.as_ptr(),
                3,
                choice_handles.as_ptr(),
                3,
                imaginary_default,
                &mut selected_complex,
            ),
            OK
        );
        assert_eq!(dtype(selected_complex).unwrap(), 13);
        assert_eq!(
            super::native_support::numeric(selected_complex, false)
                .unwrap()
                .values,
            [
                Complex64::new(0.0, 1.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(9.0, 0.0),
            ]
        );

        let scalar_true = owned_bool(&[1], &[]).unwrap();
        let scalar_choice = owned_int(&[1], &[]).unwrap();
        let matrix_choice = owned_int(&(0..12).collect::<Vec<_>>(), &[4, 3]).unwrap();
        let scalar_default = owned_int(&[0], &[]).unwrap();
        let broadcast_conditions = handles(&[scalar_true, middle_condition]);
        let broadcast_choices = handles(&[scalar_choice, matrix_choice]);
        let mut broadcast = std::ptr::null_mut();
        assert_eq!(
            super::numpy_index_extension::terlan_ndarray_v1_select(
                broadcast_conditions.as_ptr(),
                2,
                broadcast_choices.as_ptr(),
                2,
                scalar_default,
                &mut broadcast,
            ),
            OK
        );
        assert_eq!(shape(broadcast).unwrap(), [4, 3]);
        assert_eq!(ints(broadcast).unwrap(), [1; 12]);

        let hundred_conditions = vec![false_condition; 100];
        let hundred_choices = vec![first; 100];
        let condition_handles = handles(&hundred_conditions);
        let choice_handles = handles(&hundred_choices);
        let mut many = std::ptr::null_mut();
        assert_eq!(
            super::numpy_index_extension::terlan_ndarray_v1_select(
                condition_handles.as_ptr(),
                100,
                choice_handles.as_ptr(),
                100,
                scalar_default,
                &mut many,
            ),
            OK
        );
        assert_eq!(ints(many).unwrap(), [0, 0, 0]);
        let non_bool_conditions = handles(&[first]);
        let one_choice = handles(&[second]);
        let mut rejected = std::ptr::null_mut();
        assert_eq!(
            super::numpy_index_extension::terlan_ndarray_v1_select(
                non_bool_conditions.as_ptr(),
                1,
                one_choice.as_ptr(),
                1,
                scalar_default,
                &mut rejected,
            ),
            super::native_support::DTYPE_MISMATCH
        );
        assert!(rejected.is_null());
        assert_eq!(
            super::numpy_index_extension::terlan_ndarray_v1_select(
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                scalar_default,
                &mut std::ptr::null_mut(),
            ),
            INVALID_ARGUMENT
        );

        delete_arrays(&[
            many,
            selected_complex,
            imaginary_default,
            selected_int8,
            zero_default,
            int8_third,
            int8_second,
            int8_first,
            broadcast,
            scalar_default,
            matrix_choice,
            scalar_choice,
            scalar_true,
            selected,
            default,
            third,
            second,
            first,
            final_condition,
            middle_condition,
            false_condition,
        ]);
    }
}

#[test]
fn function_base_extract_and_place_round_trip_selected_values() {
    unsafe {
        let original = owned_int(&[1, 4, 3, 2, 5, 8, 7], &[7]).unwrap();
        let working = owned_int(&[1, 4, 3, 2, 5, 8, 7], &[7]).unwrap();
        let mask = owned_bool(&[0, 1, 0, 1, 0, 1, 0], &[7]).unwrap();
        let mut extracted = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_mask_select(working, mask, &mut extracted),
            OK
        );
        assert_eq!(ints(extracted).unwrap(), [4, 2, 8]);

        let zero = owned_int(&[0], &[]).unwrap();
        assert_eq!(crate::ffi::terlan_ndarray_v1_place(working, mask, zero), OK);
        assert_eq!(ints(working).unwrap(), [1, 0, 3, 0, 5, 0, 7]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_place(working, mask, extracted),
            OK
        );
        assert_eq!(ints(working).unwrap(), ints(original).unwrap());

        let alternating = owned_int(&[8, 9], &[2]).unwrap();
        let alternating_mask = owned_bool(&[1, 0, 1, 0, 1, 0, 1], &[7]).unwrap();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_place(working, alternating_mask, alternating),
            OK
        );
        assert_eq!(ints(working).unwrap(), [8, 4, 9, 2, 8, 8, 9]);

        delete_arrays(&[
            alternating_mask,
            alternating,
            zero,
            extracted,
            mask,
            working,
            original,
        ]);
    }
}

#[test]
fn function_base_trim_zeros_handles_full_width_unsigned_values_exactly() {
    unsafe {
        for value in [1_u64 << 62, 1_u64 << 63, u64::MAX] {
            let source = owned_uint(&[0, value, 0], &[3]).unwrap();
            let mut trimmed = std::ptr::null_mut();
            assert_eq!(
                super::numpy_numeric_extension::terlan_ndarray_v1_trim_zeros(
                    source,
                    2,
                    &mut trimmed,
                ),
                OK
            );
            assert_eq!(shape(trimmed).unwrap(), [1]);
            assert_eq!(dtype(trimmed).unwrap(), 10);
            assert_eq!(uints(trimmed).unwrap(), [value]);
            delete_arrays(&[trimmed, source]);
        }
    }
}

#[test]
fn function_base_delete_supports_axes_masks_duplicates_and_order_preservation() {
    unsafe {
        let vector = owned_int(&[0, 1, 2, 3, 4], &[5]).unwrap();
        let mut indexed = std::ptr::null_mut();
        assert_eq!(
            super::numpy_shape_extension::terlan_ndarray_v1_delete_indexes_axis(
                vector,
                [0_i64, -4, 2, 2].as_ptr(),
                4,
                0,
                &mut indexed,
            ),
            OK
        );
        assert_eq!(ints(indexed).unwrap(), [3, 4]);

        let remove = owned_bool(&[1, 0, 0, 1, 0], &[5]).unwrap();
        let mut masked = std::ptr::null_mut();
        assert_eq!(
            super::numpy_shape_extension::terlan_ndarray_v1_delete_mask_axis(
                vector,
                remove,
                0,
                &mut masked,
            ),
            OK
        );
        assert_eq!(ints(masked).unwrap(), [1, 2, 4]);

        let cube = owned_int(&(0..10).collect::<Vec<_>>(), &[1, 5, 2]).unwrap();
        let mut axis_deleted = std::ptr::null_mut();
        assert_eq!(
            super::numpy_shape_extension::terlan_ndarray_v1_delete_indexes_axis(
                cube,
                [0_i64, 2].as_ptr(),
                2,
                1,
                &mut axis_deleted,
            ),
            OK
        );
        assert_eq!(shape(axis_deleted).unwrap(), [1, 3, 2]);
        assert_eq!(ints(axis_deleted).unwrap(), [2, 3, 6, 7, 8, 9]);

        let matrix = owned_int(&(0..10).collect::<Vec<_>>(), &[2, 5]).unwrap();
        let mut fortran = std::ptr::null_mut();
        let mut unchanged = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copy_ordered(matrix, 1, &mut fortran),
            OK
        );
        assert_eq!(
            super::numpy_shape_extension::terlan_ndarray_v1_delete_indexes_axis(
                fortran,
                std::ptr::null(),
                0,
                1,
                &mut unchanged,
            ),
            OK
        );
        assert_eq!(contiguous_flags(unchanged), (false, true));

        let scalar = owned_int(&[1], &[]).unwrap();
        let short_mask = owned_bool(&[0, 1], &[2]).unwrap();
        let scalar_mask = owned_bool(&[1], &[]).unwrap();
        let mut rejected = std::ptr::null_mut();
        assert_eq!(
            super::numpy_shape_extension::terlan_ndarray_v1_delete_indexes_axis(
                scalar,
                std::ptr::null(),
                0,
                0,
                &mut rejected,
            ),
            super::native_support::AXIS_OUT_OF_RANGE
        );
        assert_eq!(
            super::numpy_shape_extension::terlan_ndarray_v1_delete_mask_axis(
                vector,
                short_mask,
                0,
                &mut rejected,
            ),
            super::native_support::SHAPE_MISMATCH
        );
        assert_eq!(
            super::numpy_shape_extension::terlan_ndarray_v1_delete_mask_axis(
                vector,
                scalar_mask,
                0,
                &mut rejected,
            ),
            super::native_support::SHAPE_MISMATCH
        );
        assert!(rejected.is_null());

        delete_arrays(&[
            scalar_mask,
            short_mask,
            scalar,
            unchanged,
            fortran,
            matrix,
            axis_deleted,
            cube,
            masked,
            remove,
            indexed,
            vector,
        ]);
    }
}

#[test]
fn function_base_insert_supports_scalar_list_axis_and_stable_index_semantics() {
    unsafe {
        let vector = owned_int(&[1, 2, 3], &[3]).unwrap();
        let one = owned_int(&[1], &[]).unwrap();
        let sequence = owned_int(&[1, 2, 3], &[3]).unwrap();
        let mut front = std::ptr::null_mut();
        let mut end = std::ptr::null_mut();
        let mut repeated = std::ptr::null_mut();
        let mut scalar_sequence = std::ptr::null_mut();
        assert_eq!(
            super::numpy_index_extension::terlan_ndarray_v1_insert(vector, 0, one, &mut front,),
            OK
        );
        assert_eq!(ints(front).unwrap(), [1, 1, 2, 3]);
        assert_eq!(
            super::numpy_index_extension::terlan_ndarray_v1_insert(vector, 3, one, &mut end,),
            OK
        );
        assert_eq!(ints(end).unwrap(), [1, 2, 3, 1]);
        assert_eq!(
            super::numpy_index_extension::terlan_ndarray_v1_insert_indexes(
                vector,
                [1_i64, 1, 1].as_ptr(),
                3,
                sequence,
                &mut repeated,
            ),
            OK
        );
        assert_eq!(ints(repeated).unwrap(), [1, 1, 2, 3, 2, 3]);
        assert_eq!(
            super::numpy_index_extension::terlan_ndarray_v1_insert(
                vector,
                1,
                sequence,
                &mut scalar_sequence,
            ),
            OK
        );
        assert_eq!(ints(scalar_sequence).unwrap(), [1, 1, 2, 3, 2, 3]);

        let nine = owned_int(&[9], &[]).unwrap();
        let indexes = [1_i64, -1, 3];
        let mut scalar_broadcast = std::ptr::null_mut();
        assert_eq!(
            super::numpy_index_extension::terlan_ndarray_v1_insert_indexes(
                vector,
                indexes.as_ptr(),
                indexes.len() as i64,
                nine,
                &mut scalar_broadcast,
            ),
            OK
        );
        assert_eq!(ints(scalar_broadcast).unwrap(), [1, 9, 2, 9, 3, 9]);
        assert_eq!(indexes, [1, -1, 3]);

        let replacement = owned_int(&[7, 8, 9], &[3]).unwrap();
        let mut reordered = std::ptr::null_mut();
        assert_eq!(
            super::numpy_index_extension::terlan_ndarray_v1_insert_indexes(
                vector,
                [-1_i64, 1, 3].as_ptr(),
                3,
                replacement,
                &mut reordered,
            ),
            OK
        );
        assert_eq!(ints(reordered).unwrap(), [1, 8, 2, 7, 3, 9]);

        let matrix = owned_int(&[1, 1, 2, 2, 3, 3], &[3, 2]).unwrap();
        let row_values = owned_int(&[1, 2, 3], &[3]).unwrap();
        let column_values = owned_int(&[1, 2, 3], &[3, 1]).unwrap();
        let mut scalar_row = std::ptr::null_mut();
        let mut list_column = std::ptr::null_mut();
        let mut scalar_columns = std::ptr::null_mut();
        let mut list_columns = std::ptr::null_mut();
        assert_eq!(
            super::numpy_index_extension::terlan_ndarray_v1_insert_scalar_axis(
                matrix,
                1,
                row_values,
                1,
                &mut scalar_row,
            ),
            OK
        );
        assert_eq!(shape(scalar_row).unwrap(), [3, 3]);
        assert_eq!(ints(scalar_row).unwrap(), [1, 1, 1, 2, 2, 2, 3, 3, 3]);
        assert_eq!(
            super::numpy_index_extension::terlan_ndarray_v1_insert_indexes_axis(
                matrix,
                [1_i64].as_ptr(),
                1,
                column_values,
                1,
                &mut list_column,
            ),
            OK
        );
        assert_eq!(ints(list_column).unwrap(), ints(scalar_row).unwrap());
        assert_eq!(
            super::numpy_index_extension::terlan_ndarray_v1_insert_scalar_axis(
                matrix,
                1,
                column_values,
                1,
                &mut scalar_columns,
            ),
            OK
        );
        assert_eq!(shape(scalar_columns).unwrap(), [3, 5]);
        assert_eq!(
            super::numpy_index_extension::terlan_ndarray_v1_insert_indexes_axis(
                matrix,
                [1_i64].as_ptr(),
                1,
                row_values,
                1,
                &mut list_columns,
            ),
            OK
        );
        assert_eq!(ints(list_columns).unwrap(), ints(scalar_columns).unwrap());

        let scalar = owned_int(&[1], &[]).unwrap();
        let mut rejected = std::ptr::null_mut();
        assert_eq!(
            super::numpy_index_extension::terlan_ndarray_v1_insert_indexes_axis(
                scalar,
                std::ptr::null(),
                0,
                one,
                0,
                &mut rejected,
            ),
            super::native_support::AXIS_OUT_OF_RANGE
        );
        assert!(rejected.is_null());

        delete_arrays(&[
            scalar,
            list_columns,
            scalar_columns,
            list_column,
            scalar_row,
            column_values,
            row_values,
            matrix,
            reordered,
            replacement,
            scalar_broadcast,
            nine,
            scalar_sequence,
            repeated,
            end,
            front,
            sequence,
            one,
            vector,
        ]);
    }
}
