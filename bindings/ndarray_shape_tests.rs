//! Tests for compositional shape operations.

use super::*;

const INDEX_OUT_OF_RANGE: i32 = 120;
const DTYPE_MISMATCH: i32 = 106;

/// Creates one owned fixture with an explicit real dtype.
unsafe fn fixture(values: &[f64], shape: &[i64], dtype: i32) -> Array {
    unsafe { super::super::native_support::owned_real(values, shape, dtype).unwrap() }
}

/// Returns borrowed resource identities.
fn identities(arrays: &[Array]) -> Vec<u64> {
    arrays.iter().map(|array| *array as usize as u64).collect()
}

/// Reads logical values from one fixture.
unsafe fn values(array: Array) -> Vec<f64> {
    unsafe { super::super::native_support::floats(array, false).unwrap() }
}

/// Reads one fixture shape.
unsafe fn dimensions(array: Array) -> Vec<i64> {
    unsafe { super::super::native_support::shape(array).unwrap() }
}

/// Deletes all fixture arrays.
unsafe fn dispose(arrays: &[Array]) {
    unsafe { super::super::native_support::delete_arrays(arrays) }
}

/// Deletes an ABI-owned list and every array identity it contains.
unsafe fn dispose_split_list(raw: *mut u64, length: usize) {
    let identities = unsafe { std::slice::from_raw_parts(raw, length) };
    for &identity in identities {
        let array = identity as usize as Array;
        assert_eq!(unsafe { crate::ffi::terlan_ndarray_v1_delete(array) }, OK);
    }
    unsafe { crate::ffi::terlan_ndarray_v1_delete_handle_values(raw) };
}

#[test]
fn arbitrary_concatenate_and_stack_promote_and_preserve_order() {
    unsafe {
        let first = fixture(&[1.0, 2.0], &[1, 2], 2);
        let second = fixture(&[3.5, 4.5], &[1, 2], 11);
        let third = fixture(&[5.0, 6.0], &[1, 2], 3);
        let handles = identities(&[first, second, third]);
        let mut concatenated = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_concatenate(handles.as_ptr(), 3, 0, &mut concatenated),
            OK
        );
        assert_eq!(dimensions(concatenated), [3, 2]);
        assert_eq!(values(concatenated), [1.0, 2.0, 3.5, 4.5, 5.0, 6.0]);
        assert_eq!(super::super::native_support::dtype(concatenated), Ok(3));

        let mut stacked = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_stack(handles.as_ptr(), 3, 1, &mut stacked),
            OK
        );
        assert_eq!(dimensions(stacked), [1, 3, 2]);
        assert_eq!(values(stacked), [1.0, 2.0, 3.5, 4.5, 5.0, 6.0]);
        dispose(&[stacked, concatenated, third, second, first]);
    }
}

#[test]
fn basic_slice_composes_reverse_ellipsis_squeeze_and_new_axis() {
    unsafe {
        let source = fixture(&(0..24).map(f64::from).collect::<Vec<_>>(), &[2, 3, 4], 3);
        let omitted = i64::MIN;
        let starts = [1, omitted];
        let stops = [2, omitted];
        let steps = [1, -1];
        let squeeze = [0];
        let new_axes = [0];
        let mut output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_basic_slice(
                source,
                starts.as_ptr(),
                2,
                stops.as_ptr(),
                2,
                steps.as_ptr(),
                2,
                1,
                squeeze.as_ptr(),
                1,
                new_axes.as_ptr(),
                1,
                &mut output,
            ),
            OK
        );
        assert_eq!(dimensions(output), [1, 3, 4]);
        assert_eq!(
            values(output),
            [15.0, 14.0, 13.0, 12.0, 19.0, 18.0, 17.0, 16.0, 23.0, 22.0, 21.0, 20.0]
        );

        let mut full_ellipsis = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_basic_slice(
                source,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                &mut full_ellipsis,
            ),
            OK
        );
        assert_ne!(full_ellipsis, source);
        assert_eq!(dimensions(full_ellipsis), [2, 3, 4]);

        let mut empty_tuple = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_basic_slice(
                source,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                -1,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                &mut empty_tuple,
            ),
            OK
        );
        assert_ne!(empty_tuple, source);
        assert_eq!(dimensions(empty_tuple), [2, 3, 4]);

        let mut expanded = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_expand_dims(source, 0, &mut expanded),
            OK
        );
        assert_eq!(dimensions(expanded), [1, 2, 3, 4]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_set_float(expanded, [0_i64, 0, 0, 0].as_ptr(), 4, 99.0,),
            OK
        );
        assert_eq!(values(source)[0], 99.0);

        let scalar = fixture(&[5.0], &[], 3);
        let mut scalar_view = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_basic_slice(
                scalar,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                &mut scalar_view,
            ),
            OK
        );
        assert_ne!(scalar_view, scalar);
        assert_eq!(dimensions(scalar_view), []);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_set_float(scalar_view, std::ptr::null(), 0, 8.0,),
            OK
        );
        assert_eq!(values(scalar), [8.0]);
        dispose(&[
            scalar_view,
            scalar,
            expanded,
            empty_tuple,
            full_ellipsis,
            output,
            source,
        ]);
    }
}

#[test]
fn shape_composition_rejects_empty_handles_and_malformed_specs() {
    unsafe {
        let source = fixture(&[1.0, 2.0], &[2], 3);
        let mut output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_concatenate(std::ptr::null(), 0, 0, &mut output),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_stack([0_u64].as_ptr(), 1, 0, &mut output),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_basic_slice(
                source,
                [0_i64].as_ptr(),
                1,
                [1_i64, 2].as_ptr(),
                2,
                [1_i64].as_ptr(),
                1,
                -1,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                &mut output,
            ),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        dispose(&[source]);
    }
}

#[test]
fn concatenate_and_stack_match_numpy_copy_axis_shape_and_empty_rules() {
    unsafe {
        let vector = fixture(&[1.0, 2.0], &[2], 3);
        let vector_two = fixture(&[3.0, 4.0], &[2], 3);
        let scalar = fixture(&[7.0], &[], 3);
        let matrix = fixture(&[5.0, 6.0], &[1, 2], 3);
        let vector_identity = identities(&[vector]);
        let mut output = std::ptr::null_mut();

        assert_eq!(
            terlan_ndarray_v1_concatenate(vector_identity.as_ptr(), 1, 0, &mut output),
            OK
        );
        assert_ne!(output, vector);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_set_float(output, [0_i64].as_ptr(), 1, 9.0),
            OK
        );
        assert_eq!(values(vector), [1.0, 2.0]);
        assert_eq!(values(output), [9.0, 2.0]);
        dispose(&[output]);

        for axis in [1, -2] {
            output = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_concatenate(vector_identity.as_ptr(), 1, axis, &mut output,),
                AXIS_OUT_OF_RANGE
            );
            assert!(output.is_null());
        }

        let scalar_identity = identities(&[scalar]);
        assert_eq!(
            terlan_ndarray_v1_concatenate(scalar_identity.as_ptr(), 1, 0, &mut output),
            AXIS_OUT_OF_RANGE
        );
        let mismatched = identities(&[vector, matrix]);
        assert_eq!(
            terlan_ndarray_v1_concatenate(mismatched.as_ptr(), 2, 0, &mut output),
            super::super::native_support::SHAPE_MISMATCH
        );

        let vectors = identities(&[vector, vector_two]);
        for (axis, expected) in [
            (0, vec![1.0, 2.0, 3.0, 4.0]),
            (1, vec![1.0, 3.0, 2.0, 4.0]),
            (-1, vec![1.0, 3.0, 2.0, 4.0]),
            (-2, vec![1.0, 2.0, 3.0, 4.0]),
        ] {
            assert_eq!(
                terlan_ndarray_v1_stack(vectors.as_ptr(), 2, axis, &mut output),
                OK
            );
            assert_eq!(dimensions(output), [2, 2]);
            assert_eq!(values(output), expected);
            dispose(&[output]);
        }

        let matrix_two = fixture(&[7.0, 8.0], &[1, 2], 3);
        let matrices = identities(&[matrix, matrix_two]);
        for (axis, expected_shape) in [
            (0, vec![2, 1, 2]),
            (1, vec![1, 2, 2]),
            (2, vec![1, 2, 2]),
            (-1, vec![1, 2, 2]),
            (-2, vec![1, 2, 2]),
            (-3, vec![2, 1, 2]),
        ] {
            assert_eq!(
                terlan_ndarray_v1_stack(matrices.as_ptr(), 2, axis, &mut output),
                OK
            );
            assert_eq!(dimensions(output), expected_shape);
            dispose(&[output]);
        }

        let empty = fixture(&[], &[0], 3);
        let empties = identities(&[empty, empty, empty]);
        for (axis, expected_shape) in [(0, vec![3, 0]), (1, vec![0, 3])] {
            assert_eq!(
                terlan_ndarray_v1_stack(empties.as_ptr(), 3, axis, &mut output),
                OK
            );
            assert_eq!(dimensions(output), expected_shape);
            dispose(&[output]);
        }

        let scalars = identities(&[scalar, scalar]);
        assert_eq!(
            terlan_ndarray_v1_stack(scalars.as_ptr(), 2, 0, &mut output),
            OK
        );
        assert_eq!(dimensions(output), [2]);
        assert_eq!(values(output), [7.0, 7.0]);
        dispose(&[output]);
        assert_eq!(
            terlan_ndarray_v1_stack(vectors.as_ptr(), 2, 2, &mut output),
            AXIS_OUT_OF_RANGE
        );
        assert!(output.is_null());
        let mixed_ranks = identities(&[vector, scalar]);
        assert_eq!(
            terlan_ndarray_v1_stack(mixed_ranks.as_ptr(), 2, 0, &mut output),
            super::super::native_support::SHAPE_MISMATCH
        );
        assert!(output.is_null());
        assert_eq!(
            terlan_ndarray_v1_concatenate(std::ptr::null(), 0, 0, &mut output),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        assert_eq!(
            terlan_ndarray_v1_stack(std::ptr::null(), 0, 0, &mut output),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        let columns = identities(&[matrix, matrix_two]);
        assert_eq!(
            terlan_ndarray_v1_concatenate(columns.as_ptr(), 2, -1, &mut output),
            OK
        );
        assert_eq!(dimensions(output), [1, 4]);
        assert_eq!(values(output), [5.0, 6.0, 7.0, 8.0]);
        dispose(&[
            output, empty, matrix_two, matrix, scalar, vector_two, vector,
        ]);
    }
}

#[test]
fn concatenate_flat_matches_numpy_axis_none_and_large_input_rules() {
    unsafe {
        let matrix = fixture(&[1.0, 2.0, 3.0, 4.0], &[2, 2], 6);
        let vector = fixture(&[5.0, 6.0], &[2], 6);
        let inputs = identities(&[matrix, vector]);
        let mut output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_concatenate_flat(inputs.as_ptr(), 2, &mut output),
            OK
        );
        assert_eq!(dimensions(output), [6]);
        assert_eq!(values(output), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        dispose(&[output]);

        let scalar = fixture(&[7.0], &[], 6);
        let many = vec![scalar; 99];
        let many = identities(&many);
        assert_eq!(
            terlan_ndarray_v1_concatenate_flat(many.as_ptr(), 99, &mut output),
            OK
        );
        assert_eq!(dimensions(output), [99]);
        assert_eq!(values(output), vec![7.0; 99]);
        dispose(&[output]);

        assert_eq!(
            terlan_ndarray_v1_concatenate_flat(std::ptr::null(), 0, &mut output),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        assert_eq!(
            terlan_ndarray_v1_concatenate_flat(inputs.as_ptr(), 2, std::ptr::null_mut()),
            INVALID_ARGUMENT
        );
        dispose(&[scalar, vector, matrix]);
    }
}

#[test]
fn concatenate_dtype_and_destination_paths_are_exact_atomic_and_alias_safe() {
    unsafe {
        let first = fixture(&[1.0, 2.0], &[2], 6);
        let second = fixture(&[3.0, 4.0], &[2], 6);
        let inputs = identities(&[first, second]);
        let mut output = std::ptr::null_mut();

        assert_eq!(
            terlan_ndarray_v1_concatenate_with_dtype(inputs.as_ptr(), 2, 0, 3, 2, &mut output,),
            OK
        );
        assert_eq!(super::super::native_support::dtype(output), Ok(3));
        assert_eq!(values(output), [1.0, 2.0, 3.0, 4.0]);
        dispose(&[output]);

        assert_eq!(
            terlan_ndarray_v1_concatenate_flat_with_dtype(inputs.as_ptr(), 2, 11, 3, &mut output,),
            OK
        );
        assert_eq!(super::super::native_support::dtype(output), Ok(11));
        dispose(&[output]);

        let destination = fixture(&[0.0; 4], &[4], 2);
        assert_eq!(
            terlan_ndarray_v1_concatenate_into(inputs.as_ptr(), 2, 0, destination, 2),
            OK
        );
        assert_eq!(values(destination), [1.0, 2.0, 3.0, 4.0]);

        let flat_destination = fixture(&[0.0; 4], &[4], 2);
        assert_eq!(
            terlan_ndarray_v1_concatenate_flat_into(inputs.as_ptr(), 2, flat_destination, 2,),
            OK
        );
        assert_eq!(values(flat_destination), [1.0, 2.0, 3.0, 4.0]);

        let wrong_shape = fixture(&[9.0; 4], &[2, 2], 2);
        assert_eq!(
            terlan_ndarray_v1_concatenate_into(inputs.as_ptr(), 2, 0, wrong_shape, 2),
            super::super::native_support::SHAPE_MISMATCH
        );
        assert_eq!(values(wrong_shape), [9.0; 4]);

        let rejected = fixture(&[8.0; 4], &[4], 2);
        assert_eq!(
            terlan_ndarray_v1_concatenate_into(inputs.as_ptr(), 2, 0, rejected, 9),
            INVALID_ARGUMENT
        );
        assert_eq!(values(rejected), [8.0; 4]);

        let alias_inputs = identities(&[first, first]);
        assert_eq!(
            terlan_ndarray_v1_concatenate_into(alias_inputs.as_ptr(), 2, 0, destination, 2),
            OK
        );
        assert_eq!(values(destination), [1.0, 2.0, 1.0, 2.0]);

        assert_eq!(
            terlan_ndarray_v1_concatenate_with_dtype(inputs.as_ptr(), 2, 0, 6, 9, &mut output,),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        dispose(&[
            rejected,
            wrong_shape,
            flat_destination,
            destination,
            second,
            first,
        ]);
    }
}

#[test]
fn integer_split_families_match_numpy_uneven_and_boundary_rules() {
    unsafe {
        let source = fixture(&(0..10).map(f64::from).collect::<Vec<_>>(), &[10], 3);
        for sections in 1_i64..=11 {
            let mut raw = std::ptr::null_mut();
            let mut length = 0usize;
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_array_split(
                    source,
                    sections,
                    0,
                    &mut raw,
                    &mut length,
                ),
                OK
            );
            assert_eq!(length, sections as usize);
            let parts = std::slice::from_raw_parts(raw, length);
            let quotient = 10 / sections;
            let remainder = 10 % sections;
            let mut start = 0_i64;
            for (index, &identity) in parts.iter().enumerate() {
                let expected_length = quotient + i64::from((index as i64) < remainder);
                let part = identity as usize as Array;
                assert_eq!(dimensions(part), [expected_length]);
                assert_eq!(
                    values(part),
                    (start..start + expected_length)
                        .map(|value| value as f64)
                        .collect::<Vec<_>>()
                );
                start += expected_length;
            }
            dispose_split_list(raw, length);
        }

        let mut raw = std::ptr::null_mut();
        let mut length = 0usize;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_array_split(source, 0, 0, &mut raw, &mut length),
            INVALID_ARGUMENT
        );
        assert!(raw.is_null());
        assert_eq!(length, 0);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_split(source, 2, -1, &mut raw, &mut length),
            OK
        );
        dispose_split_list(raw, length);
        raw = std::ptr::null_mut();
        length = 0;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_split(source, 3, 0, &mut raw, &mut length),
            super::super::native_support::SHAPE_MISMATCH
        );
        assert!(raw.is_null());
        assert_eq!(length, 0);

        let boundary_cases: &[(&[i64], &[i64])] = &[
            (&[1, 5, 7], &[1, 4, 2, 3]),
            (&[0, 5, 7], &[0, 5, 2, 3]),
            (&[0, 5, 7, 10, 12], &[0, 5, 2, 3, 0, 0]),
        ];
        for &(boundaries, expected_lengths) in boundary_cases {
            raw = std::ptr::null_mut();
            length = 0;
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_split_at(
                    source,
                    boundaries.as_ptr(),
                    boundaries.len() as i64,
                    -1,
                    &mut raw,
                    &mut length,
                ),
                OK
            );
            assert_eq!(length, expected_lengths.len());
            for (&identity, &expected) in std::slice::from_raw_parts(raw, length)
                .iter()
                .zip(expected_lengths.iter())
            {
                assert_eq!(dimensions(identity as usize as Array), [expected]);
            }
            dispose_split_list(raw, length);
        }

        let matrix = fixture(&(0..20).map(f64::from).collect::<Vec<_>>(), &[2, 10], 3);
        for (axis, expected_shapes) in [
            (0, vec![vec![1, 10], vec![1, 10], vec![0, 10]]),
            (-1, vec![vec![2, 4], vec![2, 3], vec![2, 3]]),
        ] {
            raw = std::ptr::null_mut();
            length = 0;
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_array_split(matrix, 3, axis, &mut raw, &mut length,),
                OK
            );
            for (&identity, expected_shape) in std::slice::from_raw_parts(raw, length)
                .iter()
                .zip(expected_shapes.iter())
            {
                assert_eq!(dimensions(identity as usize as Array), *expected_shape);
            }
            dispose_split_list(raw, length);
        }
        dispose(&[matrix]);
        dispose(&[source]);
    }
}

#[test]
fn along_axis_operations_broadcast_alias_and_fail_atomically() {
    unsafe {
        let source_values = (0..3)
            .flat_map(|row| (0..4).map(move |column| f64::from(row * 10 + column)))
            .collect::<Vec<_>>();
        let source = fixture(&source_values, &[3, 4, 1], 3);
        let index_values = [0.0, 1.0, 2.0, 3.0, 0.0, 3.0, 2.0, 1.0, 0.0, 3.0];
        let indices = fixture(&index_values, &[1, 2, 5], 2);
        let mut taken = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_take_along_axis(source, indices, 1, &mut taken),
            OK
        );
        assert_eq!(dimensions(taken), [3, 2, 5]);
        let expected = (0..3)
            .flat_map(|row| {
                index_values
                    .iter()
                    .map(move |index| f64::from(row * 10) + index)
            })
            .collect::<Vec<_>>();
        assert_eq!(values(taken), expected);

        let empty_indices = fixture(&[], &[3, 0, 1], 2);
        let mut empty_output = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_take_along_axis(
                source,
                empty_indices,
                1,
                &mut empty_output,
            ),
            OK
        );
        assert_eq!(dimensions(empty_output), [3, 0, 1]);

        let scalar_index = fixture(&[0.0], &[], 2);
        let float_indices = fixture(&index_values, &[1, 2, 5], 3);
        let bool_indices = fixture(&[0.0; 10], &[1, 2, 5], 1);
        for (candidate, axis, expected_status) in [
            (
                scalar_index,
                1,
                super::super::native_support::SHAPE_MISMATCH,
            ),
            (float_indices, 1, DTYPE_MISMATCH),
            (bool_indices, 1, DTYPE_MISMATCH),
            (indices, 10, AXIS_OUT_OF_RANGE),
        ] {
            let mut rejected = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_take_along_axis(
                    source,
                    candidate,
                    axis,
                    &mut rejected,
                ),
                expected_status
            );
            assert!(rejected.is_null());
        }

        let target = fixture(&[0.0; 12], &[3, 4, 1], 3);
        let replacement = fixture(&[20.0], &[], 3);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put_along_axis(target, indices, replacement, 1),
            OK
        );
        assert_eq!(values(target), [20.0; 12]);

        let bad_indices = fixture(
            &[0.0, 1.0, 2.0, 3.0, 0.0, 3.0, 2.0, 1.0, 0.0, 4.0],
            &[1, 2, 5],
            2,
        );
        let unchanged = fixture(&source_values, &[3, 4, 1], 3);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put_along_axis(unchanged, bad_indices, replacement, 1,),
            INDEX_OUT_OF_RANGE
        );
        assert_eq!(values(unchanged), source_values);

        let alias = fixture(&[1.0, 2.0, 3.0, 4.0], &[4], 3);
        let reverse = fixture(&[3.0, 2.0, 1.0, 0.0], &[4], 2);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put_along_axis(alias, reverse, alias, 0),
            OK
        );
        assert_eq!(values(alias), [4.0, 3.0, 2.0, 1.0]);

        let maxima = fixture(&[10.0, 30.0, 20.0, 60.0, 40.0, 50.0], &[2, 3], 3);
        let maximum_indices = fixture(&[1.0, 0.0], &[2, 1], 2);
        let minus_ninety_nine = fixture(&[-99.0], &[], 3);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put_along_axis(
                maxima,
                maximum_indices,
                minus_ninety_nine,
                1,
            ),
            OK
        );
        assert_eq!(values(maxima), [10.0, -99.0, 20.0, -99.0, 40.0, 50.0]);
        let mut selected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_take_along_axis(
                maxima,
                maximum_indices,
                1,
                &mut selected,
            ),
            OK
        );
        assert_eq!(values(selected), [-99.0, -99.0]);

        dispose(&[
            selected,
            minus_ninety_nine,
            maximum_indices,
            maxima,
            reverse,
            alias,
            unchanged,
            bad_indices,
            replacement,
            target,
            bool_indices,
            float_indices,
            scalar_index,
            empty_output,
            empty_indices,
            taken,
            indices,
            source,
        ]);
    }
}

#[test]
fn expand_dims_accepts_numpy_final_position_axis_lists() {
    unsafe {
        let source = fixture(&(0..27).map(f64::from).collect::<Vec<_>>(), &[3, 3, 3], 3);
        let cases: &[(&[i64], &[i64])] = &[
            (&[0, 1, 2], &[1, 1, 1, 3, 3, 3]),
            (&[0, -1, -2], &[1, 3, 3, 3, 1, 1]),
            (&[0, 3, 5], &[1, 3, 3, 1, 3, 1]),
            (&[0, -3, -5], &[1, 1, 3, 1, 3, 3]),
            (&[], &[3, 3, 3]),
        ];
        for &(axes, expected_shape) in cases {
            let mut output = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_expand_dims_axes(
                    source,
                    axes.as_ptr(),
                    axes.len() as i64,
                    &mut output,
                ),
                OK
            );
            assert_eq!(dimensions(output), expected_shape);
            assert_eq!(values(output), values(source));
            dispose(&[output]);
        }

        let mut output = std::ptr::null_mut();
        for axes in [&[0_i64, -6][..], &[0_i64, 6][..]] {
            assert_eq!(
                terlan_ndarray_v1_expand_dims_axes(
                    source,
                    axes.as_ptr(),
                    axes.len() as i64,
                    &mut output,
                ),
                AXIS_OUT_OF_RANGE
            );
            assert!(output.is_null());
        }
        assert_eq!(
            terlan_ndarray_v1_expand_dims_axes(source, [1_i64, 1].as_ptr(), 2, &mut output),
            DUPLICATE_AXIS
        );
        assert!(output.is_null());
        dispose(&[source]);

        let rank_four = fixture(&[0.0; 120], &[2, 3, 4, 5], 3);
        for axis in -5_i64..=4 {
            let mut expanded = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_expand_dims(rank_four, axis, &mut expanded),
                OK
            );
            let normalized = if axis < 0 { axis + 5 } else { axis } as usize;
            let mut expected = vec![2, 3, 4, 5];
            expected.insert(normalized, 1);
            assert_eq!(dimensions(expanded), expected);
            dispose(&[expanded]);
        }
        for axis in [-6_i64, 5] {
            let mut rejected = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_expand_dims(rank_four, axis, &mut rejected),
                AXIS_OUT_OF_RANGE
            );
            assert!(rejected.is_null());
        }
        dispose(&[rank_four]);
    }
}

#[test]
fn squeeze_removes_selected_or_all_singleton_dimensions() {
    unsafe {
        for shape in [vec![2, 2, 2, 1, 1], vec![2, 1, 2, 1, 2], vec![1, 1, 2, 2]] {
            let count = shape.iter().product::<i64>() as usize;
            let source = fixture(&vec![1.5; count], &shape, 3);
            let mut squeezed = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_squeeze(source, std::ptr::null(), 0, &mut squeezed,),
                OK
            );
            assert!(dimensions(squeezed).iter().all(|dimension| *dimension != 1));
            assert_eq!(values(squeezed), vec![1.5; count]);
            dispose(&[squeezed, source]);
        }

        let singleton = fixture(&[1.5], &[1, 1, 1], 3);
        let mut scalar = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_squeeze(singleton, std::ptr::null(), 0, &mut scalar,),
            OK
        );
        assert!(dimensions(scalar).is_empty());
        assert_eq!(values(scalar), [1.5]);
        dispose(&[scalar, singleton]);
    }
}

#[test]
fn tile_matches_numpy_rank_promotion_empty_and_copy_rules() {
    unsafe {
        let vector = fixture(&[0.0, 1.0, 2.0], &[3], 3);
        let matrix = fixture(&[1.0, 2.0, 3.0, 4.0], &[2, 2], 3);
        let cases: &[(Array, &[i64], &[i64], &[f64])] = &[
            (vector, &[2], &[6], &[0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            (
                vector,
                &[2, 2],
                &[2, 6],
                &[0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            ),
            (
                matrix,
                &[2],
                &[2, 4],
                &[1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0],
            ),
            (
                matrix,
                &[2, 1],
                &[4, 2],
                &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            ),
            (
                matrix,
                &[2, 2],
                &[4, 4],
                &[
                    1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0,
                ],
            ),
        ];
        for &(source, repetitions, expected_shape, expected_values) in cases {
            let mut output = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_tile(
                    source,
                    repetitions.as_ptr(),
                    repetitions.len() as i64,
                    &mut output,
                ),
                OK
            );
            assert_eq!(dimensions(output), expected_shape);
            assert_eq!(values(output), expected_values);
            dispose(&[output]);
        }

        let empty_matrix = fixture(&[], &[2, 0], 3);
        let empty_cube = fixture(&[], &[1, 1, 0], 3);
        for (source, repetitions, expected_shape) in [
            (empty_matrix, vec![2], vec![2, 0]),
            (empty_cube, vec![3, 2, 5], vec![3, 2, 0]),
        ] {
            let mut output = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_tile(
                    source,
                    repetitions.as_ptr(),
                    repetitions.len() as i64,
                    &mut output,
                ),
                OK
            );
            assert_eq!(dimensions(output), expected_shape);
            assert!(values(output).is_empty());
            dispose(&[output]);
        }

        let mut copy = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_tile(vector, [1_i64].as_ptr(), 1, &mut copy),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_set_float(copy, [0_i64].as_ptr(), 1, 9.0),
            OK
        );
        assert_eq!(values(vector), [0.0, 1.0, 2.0]);
        assert_eq!(values(copy), [9.0, 1.0, 2.0]);
        dispose(&[copy, empty_cube, empty_matrix, matrix, vector]);
    }
}
