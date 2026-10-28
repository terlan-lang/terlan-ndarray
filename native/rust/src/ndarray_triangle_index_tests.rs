use super::*;

unsafe fn integer_values(array: Array) -> Vec<i64> {
    unsafe { super::super::native_support::ints(array).unwrap() }
}

#[test]
fn triangle_indices_match_numpy_square_rectangular_offsets_and_empty_shapes() {
    unsafe {
        let mut lower = std::ptr::null_mut();
        let mut lower_offset = std::ptr::null_mut();
        let mut lower_rectangular = std::ptr::null_mut();
        let mut upper = std::ptr::null_mut();
        let mut upper_offset = std::ptr::null_mut();
        let mut upper_rectangular = std::ptr::null_mut();
        let mut empty = std::ptr::null_mut();

        assert_eq!(terlan_ndarray_v1_tril_indices(4, 0, -1, &mut lower), OK);
        assert_eq!(shape(lower).unwrap(), [2, 10]);
        assert_eq!(
            integer_values(lower),
            [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 0, 0, 1, 0, 1, 2, 0, 1, 2, 3]
        );
        assert_eq!(
            terlan_ndarray_v1_tril_indices(4, 2, -1, &mut lower_offset),
            OK
        );
        assert_eq!(shape(lower_offset).unwrap(), [2, 15]);
        assert_eq!(
            terlan_ndarray_v1_tril_indices(4, 0, 5, &mut lower_rectangular),
            OK
        );
        assert_eq!(shape(lower_rectangular).unwrap(), [2, 10]);

        assert_eq!(terlan_ndarray_v1_triu_indices(4, 0, -1, &mut upper), OK);
        assert_eq!(shape(upper).unwrap(), [2, 10]);
        assert_eq!(
            integer_values(upper),
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 0, 1, 2, 3, 1, 2, 3, 2, 3, 3]
        );
        assert_eq!(
            terlan_ndarray_v1_triu_indices(4, 2, -1, &mut upper_offset),
            OK
        );
        assert_eq!(shape(upper_offset).unwrap(), [2, 3]);
        assert_eq!(integer_values(upper_offset), [0, 0, 1, 2, 3, 3]);
        assert_eq!(
            terlan_ndarray_v1_triu_indices(4, 0, 5, &mut upper_rectangular),
            OK
        );
        assert_eq!(shape(upper_rectangular).unwrap(), [2, 14]);
        assert_eq!(terlan_ndarray_v1_tril_indices(0, 0, -1, &mut empty), OK);
        assert_eq!(shape(empty).unwrap(), [2, 0]);

        super::super::native_support::delete_arrays(&[
            lower,
            lower_offset,
            lower_rectangular,
            upper,
            upper_offset,
            upper_rectangular,
            empty,
        ]);
    }
}

#[test]
fn triangle_indices_from_requires_rank_two_and_preserves_rectangular_shape() {
    unsafe {
        let matrix = super::super::native_support::owned_int(&[0; 20], &[4, 5]).unwrap();
        let vector = super::super::native_support::owned_int(&[0; 2], &[2]).unwrap();
        let cube = super::super::native_support::owned_int(&[0; 8], &[2, 2, 2]).unwrap();
        let mut lower = std::ptr::null_mut();
        let mut upper = std::ptr::null_mut();
        let mut rejected = 1usize as Array;

        assert_eq!(
            terlan_ndarray_v1_tril_indices_from(matrix, 0, &mut lower),
            OK
        );
        assert_eq!(shape(lower).unwrap(), [2, 10]);
        assert_eq!(
            terlan_ndarray_v1_triu_indices_from(matrix, 0, &mut upper),
            OK
        );
        assert_eq!(shape(upper).unwrap(), [2, 14]);
        assert_eq!(
            terlan_ndarray_v1_tril_indices_from(vector, 0, &mut rejected),
            SHAPE_MISMATCH
        );
        assert!(rejected.is_null());
        rejected = 1usize as Array;
        assert_eq!(
            terlan_ndarray_v1_triu_indices_from(cube, 0, &mut rejected),
            SHAPE_MISMATCH
        );
        assert!(rejected.is_null());

        super::super::native_support::delete_arrays(&[matrix, vector, cube, lower, upper]);
    }
}

#[test]
fn triangle_indices_reject_invalid_sizes_outputs_and_overflow_without_leaks() {
    unsafe {
        let mut output = 1usize as Array;
        assert_eq!(
            terlan_ndarray_v1_tril_indices(-1, 0, -1, &mut output),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        output = 1usize as Array;
        assert_eq!(
            terlan_ndarray_v1_triu_indices(2, 0, -2, &mut output),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        assert_eq!(
            terlan_ndarray_v1_tril_indices(i64::MAX, 0, i64::MAX, &mut output),
            SIZE_OVERFLOW
        );
        assert!(output.is_null());
        assert_eq!(
            terlan_ndarray_v1_tril_indices(2, 0, 2, std::ptr::null_mut()),
            INVALID_ARGUMENT
        );
    }
}
