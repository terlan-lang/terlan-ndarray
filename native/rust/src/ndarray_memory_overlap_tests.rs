//! NumPy memory-overlap query and work-budget regression tests.

use super::native_support::{
    Array, INVALID_ARGUMENT, OK, SHAPE_MISMATCH, delete_arrays, dtype, floats, ints, owned_bool,
    owned_int, owned_int_as, owned_real, shape, uints,
};

const TOO_HARD: i32 = 130;
const AXIS_OUT_OF_RANGE: i32 = 110;
const INDEX_OUT_OF_RANGE: i32 = 120;
const READ_ONLY: i32 = 121;

/// Calls the conservative overlap query with an explicit work policy.
unsafe fn may_share_with_work(lhs: Array, rhs: Array, max_work: i64) -> Result<bool, i32> {
    let mut output = false;
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_may_share_memory_with_work(lhs, rhs, max_work, &mut output)
    };
    if status == OK {
        Ok(output)
    } else {
        Err(status)
    }
}

/// Calls the exact overlap query with an explicit work policy.
unsafe fn shares_with_work(lhs: Array, rhs: Array, max_work: i64) -> Result<bool, i32> {
    let mut output = false;
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_shares_memory_with_work(lhs, rhs, max_work, &mut output)
    };
    if status == OK {
        Ok(output)
    } else {
        Err(status)
    }
}

/// Returns whether one array aliases itself through distinct logical indexes.
unsafe fn internal_overlap(array: Array) -> Result<bool, i32> {
    let mut output = false;
    let status = unsafe { crate::ffi::terlan_ndarray_v1_internal_overlap(array, &mut output) };
    if status == OK {
        Ok(output)
    } else {
        Err(status)
    }
}

#[test]
fn bounded_queries_distinguish_bounds_exactness_and_exhaustion() {
    unsafe {
        let source = owned_int(&(0..12).collect::<Vec<_>>(), &[12]).unwrap();
        let mut even = std::ptr::null_mut();
        let mut odd = std::ptr::null_mut();
        let mut overlap = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(source, 0, 0, 12, 2, &mut even),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(source, 0, 1, 12, 2, &mut odd),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(source, 0, 2, 10, 1, &mut overlap),
            OK
        );

        assert_eq!(may_share_with_work(even, odd, 0), Ok(true));
        assert_eq!(may_share_with_work(even, odd, -1), Ok(false));
        assert_eq!(may_share_with_work(even, odd, 1), Ok(true));
        assert_eq!(shares_with_work(even, odd, 1), Err(TOO_HARD));
        assert_eq!(shares_with_work(even, odd, -1), Ok(false));
        assert_eq!(shares_with_work(even, overlap, -1), Ok(true));

        delete_arrays(&[overlap, odd, even, source]);
    }
}

#[test]
fn internal_overlap_handles_scalar_empty_zero_and_nonzero_strides() {
    unsafe {
        let source = owned_int(&(0..25).collect::<Vec<_>>(), &[25]).unwrap();
        let scalar = owned_int(&[1], &[]).unwrap();
        let empty = owned_int(&[], &[0, 3]).unwrap();
        let mut repeated = std::ptr::null_mut();
        let mut unique = std::ptr::null_mut();
        let mut aliased = std::ptr::null_mut();
        let mut sliced = std::ptr::null_mut();
        let mut transposed = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_as_strided(
                source,
                [4_i64].as_ptr(),
                [0_i64].as_ptr(),
                1,
                1,
                &mut repeated,
            ),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(source, 0, 0, 25, 2, &mut sliced),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_reshape(sliced, [13_i64, 1].as_ptr(), 2, &mut transposed),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_as_strided(
                source,
                [4_i64, 4].as_ptr(),
                [3_i64, 4].as_ptr(),
                2,
                2,
                &mut unique,
            ),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_as_strided(
                source,
                [5_i64, 4].as_ptr(),
                [3_i64, 4].as_ptr(),
                2,
                2,
                &mut aliased,
            ),
            OK
        );

        assert_eq!(internal_overlap(source), Ok(false));
        assert_eq!(internal_overlap(scalar), Ok(false));
        assert_eq!(internal_overlap(empty), Ok(false));
        assert_eq!(internal_overlap(repeated), Ok(true));
        assert_eq!(internal_overlap(unique), Ok(false));
        assert_eq!(internal_overlap(aliased), Ok(true));
        assert_eq!(internal_overlap(sliced), Ok(false));
        assert_eq!(internal_overlap(transposed), Ok(false));

        delete_arrays(&[
            transposed, sliced, aliased, unique, repeated, empty, scalar, source,
        ]);
    }
}

#[test]
fn overlap_queries_reject_invalid_arguments_and_work_policies() {
    unsafe {
        let source = owned_int(&[1, 2], &[2]).unwrap();
        let mut output = true;
        assert_eq!(
            may_share_with_work(source, source, -2),
            Err(INVALID_ARGUMENT)
        );
        assert_eq!(shares_with_work(source, source, -2), Err(INVALID_ARGUMENT));
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_may_share_memory_with_work(
                source,
                source,
                -1,
                std::ptr::null_mut(),
            ),
            INVALID_ARGUMENT
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_shares_memory_with_work(
                std::ptr::null_mut(),
                source,
                -1,
                &mut output,
            ),
            INVALID_ARGUMENT
        );
        assert!(!output);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_internal_overlap(source, std::ptr::null_mut()),
            INVALID_ARGUMENT
        );
        delete_arrays(&[source]);
    }
}

#[test]
fn unary_binary_and_masked_destinations_stage_every_alias() {
    unsafe {
        let inverted_base = owned_int_as(&[1, 2, 3, 4, 5], &[5], 5).unwrap();
        let mut inverted_source = std::ptr::null_mut();
        let mut inverted_destination = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(
                inverted_base,
                0,
                0,
                4,
                1,
                &mut inverted_source,
            ),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(
                inverted_base,
                0,
                1,
                5,
                1,
                &mut inverted_destination,
            ),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_invert_into(inverted_destination, inverted_source,),
            OK
        );
        assert_eq!(ints(inverted_base).unwrap(), [1, -2, -3, -4, -5]);

        let added_base = owned_int(&[1, 2, 3, 4, 5, 6], &[6]).unwrap();
        let mut add_source = std::ptr::null_mut();
        let mut add_destination = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(added_base, 0, 0, 5, 1, &mut add_source,),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(added_base, 0, 1, 6, 1, &mut add_destination,),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_add_into(add_destination, add_source, add_source),
            OK
        );
        assert_eq!(ints(added_base).unwrap(), [1, 2, 4, 6, 8, 10]);

        let reciprocal_base = owned_real(&[1.0, 2.0, 4.0, 8.0], &[4], 3).unwrap();
        let mut reciprocal_source = std::ptr::null_mut();
        let mut reciprocal_destination = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(
                reciprocal_base,
                0,
                0,
                3,
                1,
                &mut reciprocal_source,
            ),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(
                reciprocal_base,
                0,
                1,
                4,
                1,
                &mut reciprocal_destination,
            ),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_reciprocal_into(
                reciprocal_destination,
                reciprocal_source,
            ),
            OK
        );
        assert_eq!(
            floats(reciprocal_base, false).unwrap(),
            [1.0, 1.0, 0.5, 0.25]
        );

        let booleans = owned_bool(&[0, 1, 0, 1], &[4]).unwrap();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_invert_where_into(booleans, booleans, booleans),
            OK
        );
        assert_eq!(ints(booleans).unwrap(), [0, 0, 0, 0]);

        delete_arrays(&[
            booleans,
            reciprocal_destination,
            reciprocal_source,
            reciprocal_base,
            add_destination,
            add_source,
            added_base,
            inverted_destination,
            inverted_source,
            inverted_base,
        ]);
    }
}

#[test]
fn ufunc_destinations_expand_broadcasts_and_masks_preserve_unselected_values() {
    unsafe {
        let lhs = owned_int(&[0, 1, 2], &[1, 3]).unwrap();
        let rhs = owned_int(&[0, 1, 2], &[3]).unwrap();
        let expanded = owned_real(&[-1.0; 24], &[2, 4, 3], 3).unwrap();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_add_into(expanded, lhs, rhs),
            OK
        );
        assert_eq!(floats(expanded, false).unwrap(), [0.0, 2.0, 4.0].repeat(8));

        let values = owned_int(&[0, 1, 2, 3], &[2, 2]).unwrap();
        let one = owned_int(&[1], &[]).unwrap();
        let mask = owned_bool(&[0, 1], &[2]).unwrap();
        let selected = owned_real(&[1.5; 4], &[2, 2], 3).unwrap();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_add_where_into(selected, values, one, mask),
            OK
        );
        assert_eq!(floats(selected, false).unwrap(), [1.5, 2.0, 1.5, 4.0]);
        delete_arrays(&[selected, mask, one, values, expanded, rhs, lhs]);
    }
}

#[test]
fn bitwise_inversion_preserves_every_integer_storage_width() {
    unsafe {
        for dtype_code in [2, 4, 5, 6, 7, 8, 9, 10] {
            let source = owned_int_as(&[0, 1, 3], &[3], dtype_code).unwrap();
            let mut inverted = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_invert(source, &mut inverted),
                OK
            );
            assert_eq!(dtype(inverted), Ok(dtype_code));
            match dtype_code {
                2 | 4 | 5 | 6 => assert_eq!(ints(inverted).unwrap(), [-1, -2, -4]),
                7 => assert_eq!(uints(inverted).unwrap(), [255, 254, 252]),
                8 => assert_eq!(uints(inverted).unwrap(), [65535, 65534, 65532]),
                9 => assert_eq!(
                    uints(inverted).unwrap(),
                    [4_294_967_295, 4_294_967_294, 4_294_967_292]
                ),
                10 => assert_eq!(
                    uints(inverted).unwrap(),
                    [u64::MAX, u64::MAX - 1, u64::MAX - 3]
                ),
                _ => unreachable!(),
            }
            delete_arrays(&[inverted, source]);
        }
    }
}

#[test]
fn indexed_ufunc_updates_stabilize_indexes_values_and_repetitions() {
    unsafe {
        let inverted = owned_int_as(&[0, 1, 2, 3, 4], &[5], 5).unwrap();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_invert_at(inverted, inverted),
            OK
        );
        assert_eq!(ints(inverted).unwrap(), [-1, -2, -3, -4, -5]);

        let added = owned_int(&[0, 1, 2, 3, 4, 5], &[6]).unwrap();
        let indexes = owned_int(&[0, 2, 4], &[3]).unwrap();
        let mut aliased_values = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(added, 0, 1, 4, 1, &mut aliased_values,),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_add_at(added, indexes, aliased_values),
            OK
        );
        assert_eq!(ints(added).unwrap(), [1, 1, 4, 3, 7, 5]);

        let repeated = owned_int(&[0, 0, 0], &[3]).unwrap();
        let repeated_indexes = owned_int(&[1, 1, 1], &[3]).unwrap();
        let scalar = owned_int(&[2], &[]).unwrap();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_add_at(repeated, repeated_indexes, scalar),
            OK
        );
        assert_eq!(ints(repeated).unwrap(), [0, 6, 0]);

        delete_arrays(&[
            scalar,
            repeated_indexes,
            repeated,
            aliased_values,
            indexes,
            added,
            inverted,
        ]);
    }
}

#[test]
fn reduceat_preserves_segments_axes_and_overlapping_outputs() {
    unsafe {
        let source = owned_int(&[0, 1, 2, 3, 4, 5, 6, 7], &[8]).unwrap();
        let indexes = owned_int(&[0, 3, 2, 7], &[4]).unwrap();
        let mut reduced = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_add_reduceat(source, indexes, 0, &mut reduced),
            OK
        );
        assert_eq!(ints(reduced).unwrap(), [3, 3, 20, 7]);

        let matrix = owned_int(&[0, 1, 2, 3, 4, 5, 6, 7], &[2, 4]).unwrap();
        let pair = owned_int(&[0, 2], &[2]).unwrap();
        let mut matrix_reduced = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_add_reduceat(matrix, pair, -1, &mut matrix_reduced),
            OK
        );
        assert_eq!(shape(matrix_reduced).unwrap(), [2, 2]);
        assert_eq!(ints(matrix_reduced).unwrap(), [1, 5, 9, 13]);

        let aliased = owned_int(&[0, 2, 4, 6, 4, 5, 6, 7], &[8]).unwrap();
        let mut aliased_indexes = std::ptr::null_mut();
        let mut destination = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(aliased, 0, 0, 4, 1, &mut aliased_indexes,),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(aliased, 0, 2, 6, 1, &mut destination,),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_add_reduceat_into(
                destination,
                aliased,
                aliased_indexes,
                0,
            ),
            OK
        );
        assert_eq!(ints(aliased).unwrap(), [0, 2, 2, 10, 9, 13, 6, 7]);

        delete_arrays(&[
            destination,
            aliased_indexes,
            aliased,
            matrix_reduced,
            pair,
            matrix,
            reduced,
            indexes,
            source,
        ]);
    }
}

#[test]
fn ufunc_output_operations_reject_bad_shapes_indexes_axes_and_mutability_atomically() {
    unsafe {
        let source = owned_int(&[1, 2, 3], &[3]).unwrap();
        let unchanged = owned_int(&[9, 9], &[2]).unwrap();
        let bad_indexes = owned_int(&[0, 3], &[2]).unwrap();
        let matrix_indexes = owned_int(&[0, 1], &[1, 2]).unwrap();
        let values = owned_int(&[0, 1, 2], &[3]).unwrap();
        let mut output = 1_usize as Array;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_add_reduceat(source, bad_indexes, 0, &mut output,),
            INDEX_OUT_OF_RANGE
        );
        assert!(output.is_null());
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_add_reduceat(source, matrix_indexes, 0, &mut output),
            INVALID_ARGUMENT
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_add_reduceat(source, bad_indexes, 2, &mut output),
            AXIS_OUT_OF_RANGE
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_add_into(unchanged, source, source),
            SHAPE_MISMATCH
        );
        assert_eq!(ints(unchanged).unwrap(), [9, 9]);

        let scalar = owned_int(&[1], &[]).unwrap();
        let mut read_only = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_broadcast_to(scalar, [3_i64].as_ptr(), 1, &mut read_only),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_invert_at(read_only, values),
            READ_ONLY
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_add_at(source, values, unchanged),
            SHAPE_MISMATCH
        );
        assert_eq!(ints(source).unwrap(), [1, 2, 3]);

        delete_arrays(&[
            read_only,
            scalar,
            values,
            matrix_indexes,
            bad_indexes,
            unchanged,
            source,
        ]);
    }
}
