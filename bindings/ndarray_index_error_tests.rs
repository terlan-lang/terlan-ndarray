//! Adversarial indexing and mutation error-contract tests.

use super::*;

const INDEX_OUT_OF_RANGE: i32 = 120;
const DTYPE_MISMATCH: i32 = 106;
const SHAPE_MISMATCH: i32 = 108;
const AXIS_OUT_OF_RANGE: i32 = 110;

/// Creates an owned Int64 fixture.
unsafe fn fixture(values: &[i64], shape: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_int_as(values, shape, 2).unwrap() }
}

/// Creates an owned Bool fixture.
unsafe fn bool_fixture(values: &[u8], shape: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_bool(values, shape).unwrap() }
}

/// Reads all logical Int64 values from one fixture.
unsafe fn values(array: Array) -> Vec<i64> {
    unsafe { super::super::native_support::ints(array).unwrap() }
}

/// Deletes all owned fixtures.
unsafe fn dispose(arrays: &[Array]) {
    unsafe { super::super::native_support::delete_arrays(arrays) }
}

#[test]
/// Rejects every nonempty take from a zero-length axis in every indexing mode.
fn take_rejects_nonempty_indexes_on_zero_length_axes_for_every_mode() {
    unsafe {
        let source = fixture(&[], &[2, 3, 0, 4]);
        for mode in 0..=2 {
            for index in [0_i64, 1, -1] {
                let mut output = 1_usize as Array;
                assert_eq!(
                    crate::ffi::terlan_ndarray_v1_take_mode(
                        source,
                        &index,
                        1,
                        2,
                        mode,
                        &mut output,
                    ),
                    INDEX_OUT_OF_RANGE
                );
                assert!(output.is_null());
            }
        }

        let mut output = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_take_mode(source, std::ptr::null(), 0, 2, 1, &mut output,),
            OK
        );
        assert_eq!(shape(output).unwrap(), [2, 3, 0, 4]);
        dispose(&[output, source]);
    }
}

#[test]
/// Rejects out-of-range scalar, flattened, and coordinate mutations atomically.
fn indexing_failures_leave_sources_unchanged_and_outputs_null() {
    unsafe {
        let source = fixture(&[1, 2, 3, 4, 5, 6], &[2, 3]);
        let replacement = fixture(&[9], &[]);
        let original = values(source);
        let bad_flat = [6_i64];
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put(source, bad_flat.as_ptr(), 1, replacement,),
            INDEX_OUT_OF_RANGE
        );
        assert_eq!(values(source), original);

        let mut scalar = 0_i64;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_get_int(source, [2_i64, 0].as_ptr(), 2, &mut scalar),
            INDEX_OUT_OF_RANGE
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_set_int(source, [0_i64, 3].as_ptr(), 2, 9),
            INDEX_OUT_OF_RANGE
        );
        assert_eq!(values(source), original);

        let bad_coordinates = fixture(&[0, 0, 2, 0], &[2, 2]);
        let mut output = 1_usize as Array;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_gather_coordinates(source, bad_coordinates, &mut output,),
            INDEX_OUT_OF_RANGE
        );
        assert!(output.is_null());
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put_coordinates(source, bad_coordinates, replacement,),
            INDEX_OUT_OF_RANGE
        );
        assert_eq!(values(source), original);
        dispose(&[bad_coordinates, replacement, source]);
    }
}

#[test]
/// Distinguishes malformed coordinate rank from valid but out-of-range indexes.
fn indexing_rejects_excess_missing_and_empty_dimension_coordinates() {
    unsafe {
        let source = fixture(&[1, 2, 3, 4, 5, 6], &[2, 3]);
        let mut scalar = 0_i64;
        for coordinates in [&[0_i64][..], &[0_i64, 0, 0][..]] {
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_get_int(
                    source,
                    coordinates.as_ptr(),
                    coordinates.len() as i64,
                    &mut scalar,
                ),
                SHAPE_MISMATCH
            );
        }

        let empty = fixture(&[], &[3, 0]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_get_int(empty, [1_i64, 0].as_ptr(), 2, &mut scalar),
            INDEX_OUT_OF_RANGE
        );
        let mut output = 1_usize as Array;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_index_axis(empty, 1, 0, &mut output),
            INDEX_OUT_OF_RANGE
        );
        assert!(output.is_null());
        dispose(&[empty, source]);
    }
}

#[test]
/// Rejects boolean masks whose rank or any dimension differs from the source.
fn boolean_indexing_rejects_every_rank_and_dimension_mismatch() {
    unsafe {
        let matrix = fixture(&[0; 9], &[3, 3]);
        let wrong_rows = bool_fixture(&[0; 9], &[1, 9]);
        let wrong_columns = bool_fixture(&[0; 8], &[1, 8]);
        let wrong_rank = bool_fixture(&[0; 9], &[1, 3, 3]);
        for mask in [wrong_rows, wrong_columns, wrong_rank] {
            let mut output = 1_usize as Array;
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_mask_select(matrix, mask, &mut output),
                SHAPE_MISMATCH
            );
            assert!(output.is_null());
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_mask_assign(matrix, mask, matrix),
                SHAPE_MISMATCH
            );
            assert_eq!(values(matrix), [0; 9]);
        }

        let cube = fixture(&[0; 2], &[1, 1, 2]);
        let mismatched_cube = bool_fixture(&[1, 0], &[1, 2, 1]);
        let mut output = 1_usize as Array;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_mask_select(cube, mismatched_cube, &mut output),
            SHAPE_MISMATCH
        );
        assert!(output.is_null());
        dispose(&[
            mismatched_cube,
            cube,
            wrong_rank,
            wrong_columns,
            wrong_rows,
            matrix,
        ]);
    }
}

#[test]
/// Requires explicit unsafe casting for masked complex-to-real assignment.
fn boolean_index_assignment_uses_explicit_casting_and_is_atomic() {
    unsafe {
        let destination = super::super::native_support::owned_real(&[0.0; 6], &[2, 3], 3).unwrap();
        let mask = bool_fixture(&[0, 1, 0, 0, 0, 0], &[2, 3]);
        let complex = super::super::native_support::owned_complex(
            &[rustfft::num_complex::Complex64::new(2.0, 1.0)],
            &[],
            13,
        )
        .unwrap();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copyto_where(destination, complex, mask, 3),
            DTYPE_MISMATCH
        );
        assert_eq!(
            super::super::native_support::floats(destination, false).unwrap(),
            [0.0; 6]
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copyto_where(destination, complex, mask, 4),
            OK
        );
        assert_eq!(
            super::super::native_support::floats(destination, false).unwrap(),
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
        );

        let imaginary = super::super::native_support::owned_complex(
            &[rustfft::num_complex::Complex64::new(0.0, 1.0)],
            &[],
            13,
        )
        .unwrap();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copyto_where(destination, imaginary, mask, 4),
            OK
        );
        assert_eq!(
            super::super::native_support::floats(destination, false).unwrap(),
            [0.0; 6]
        );
        dispose(&[imaginary, complex, mask, destination]);
    }
}

#[test]
/// Preserves rank-zero indexing, projection, contiguity, and output contracts.
fn zero_rank_arrays_reject_nonempty_indexes_and_preserve_scalar_outputs() {
    unsafe {
        let scalar = fixture(&[2], &[]);
        let vector = fixture(&[1], &[1]);
        let empty_indexes = fixture(&[], &[0]);
        let mut value = 0_i64;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_get_int(scalar, [0_i64].as_ptr(), 1, &mut value),
            SHAPE_MISMATCH
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_set_int(scalar, [0_i64].as_ptr(), 1, 9),
            SHAPE_MISMATCH
        );
        assert_eq!(values(scalar), [2]);

        let mut output = 1_usize as Array;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_take_indices(scalar, empty_indexes, 0, &mut output),
            AXIS_OUT_OF_RANGE
        );
        assert!(output.is_null());

        let mut scalar_view = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_index_axis(vector, 0, 0, &mut scalar_view),
            OK
        );
        assert_eq!(
            super::super::native_support::shape(scalar_view).unwrap(),
            []
        );
        assert_eq!(values(scalar_view), [1]);

        assert_eq!(
            crate::ffi::terlan_ndarray_v1_add_into(scalar, scalar, vector),
            SHAPE_MISMATCH
        );
        assert_eq!(values(scalar), [2]);

        let complex = super::super::native_support::owned_complex(
            &[rustfft::num_complex::Complex64::new(0.0, 1.0)],
            &[],
            13,
        )
        .unwrap();
        let mut real = std::ptr::null_mut();
        let mut imag = std::ptr::null_mut();
        assert_eq!(crate::terlan_ndarray_v1_real(complex, &mut real), OK);
        assert_eq!(crate::terlan_ndarray_v1_imag(complex, &mut imag), OK);
        for (projection, expected) in [(real, 0.0), (imag, 1.0)] {
            assert_eq!(super::super::native_support::shape(projection).unwrap(), []);
            assert_eq!(
                super::super::native_support::floats(projection, false).unwrap(),
                [expected]
            );
            let mut c_contiguous = 0_i32;
            let mut f_contiguous = 0_i32;
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_is_contiguous(projection, &mut c_contiguous),
                OK
            );
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_is_fortran_contiguous(projection, &mut f_contiguous,),
                OK
            );
            assert_eq!((c_contiguous, f_contiguous), (1, 1));
        }
        dispose(&[
            imag,
            real,
            complex,
            scalar_view,
            empty_indexes,
            vector,
            scalar,
        ]);
    }
}

#[test]
/// Stages overlapping forward, backward, and reversed basic-slice assignments.
fn overlapping_basic_slice_assignments_stage_every_source_value() {
    unsafe {
        let forward = fixture(&[0, 1, 2, 3], &[4]);
        let backward = fixture(&[0, 1, 2, 3], &[4]);
        let reversed = fixture(&[0, 1, 2, 3], &[4]);
        let mut forward_source = std::ptr::null_mut();
        let mut forward_destination = std::ptr::null_mut();
        let mut backward_source = std::ptr::null_mut();
        let mut backward_destination = std::ptr::null_mut();
        let mut reversed_source = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(forward, 0, 1, 4, 1, &mut forward_source,),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(forward, 0, 0, 3, 1, &mut forward_destination,),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copyto(forward_destination, forward_source),
            OK
        );
        assert_eq!(values(forward), [1, 2, 3, 3]);

        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(backward, 0, 0, 3, 1, &mut backward_source,),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(
                backward,
                0,
                1,
                4,
                1,
                &mut backward_destination,
            ),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copyto(backward_destination, backward_source),
            OK
        );
        assert_eq!(values(backward), [0, 0, 1, 2]);

        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(
                reversed,
                0,
                i64::MIN,
                i64::MIN,
                -1,
                &mut reversed_source,
            ),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copyto(reversed, reversed_source),
            OK
        );
        assert_eq!(values(reversed), [3, 2, 1, 0]);
        dispose(&[
            reversed_source,
            backward_destination,
            backward_source,
            forward_destination,
            forward_source,
            reversed,
            backward,
            forward,
        ]);
    }
}
