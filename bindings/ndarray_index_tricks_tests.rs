//! NumPy index-tricks parity and hostile-input tests.

use super::index_tricks_extension::*;
use super::native_support::{
    delete_arrays, dtype, ints, owned_bool, owned_int, owned_int_as, owned_real, shape, Array,
    AXIS_OUT_OF_RANGE, INVALID_ARGUMENT, OK, SHAPE_MISMATCH, SIZE_OVERFLOW, UNSUPPORTED_DTYPE,
};
use super::numpy_index_extension::terlan_ndarray_v1_ix;
use super::numpy_shape_extension::{
    terlan_ndarray_v1_diag_indices, terlan_ndarray_v1_diag_indices_from,
};

/// Takes ownership of a generated ABI array list.
unsafe fn take_arrays(values: *mut u64, length: usize) -> Vec<Array> {
    let arrays = if length == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(values, length) }
            .iter()
            .map(|value| *value as usize as Array)
            .collect()
    };
    unsafe { crate::ffi::terlan_ndarray_v1_delete_handle_values(values) };
    arrays
}

/// Calls the configured ravel primitive and returns its owned result.
unsafe fn ravel(
    coordinates: Array,
    dimensions: &[i64],
    modes: &[i64],
    order: i64,
) -> Result<Array, i32> {
    let mut output = std::ptr::null_mut();
    let status = unsafe {
        terlan_ndarray_v1_ravel_multi_index_with(
            coordinates,
            dimensions.as_ptr(),
            dimensions.len() as i64,
            modes.as_ptr(),
            modes.len() as i64,
            order,
            &mut output,
        )
    };
    if status == OK {
        Ok(output)
    } else {
        assert!(output.is_null());
        Err(status)
    }
}

/// Calls the ordered unravel primitive and returns all owned axes.
unsafe fn unravel(indexes: Array, dimensions: &[i64], order: i64) -> Result<Vec<Array>, i32> {
    let mut output = std::ptr::null_mut();
    let mut length = 0usize;
    let status = unsafe {
        terlan_ndarray_v1_unravel_index_ordered(
            indexes,
            dimensions.as_ptr(),
            dimensions.len() as i64,
            order,
            &mut output,
            &mut length,
        )
    };
    if status == OK {
        Ok(unsafe { take_arrays(output, length) })
    } else {
        assert!(output.is_null());
        assert_eq!(length, 0);
        Err(status)
    }
}

#[test]
fn ravel_supports_c_fortran_scalar_and_per_axis_modes() {
    unsafe {
        let coordinates = owned_int(&[3, 6, 6, 4, 5, 1], &[2, 3]).unwrap();
        let c_order = ravel(coordinates, &[7, 6], &[0], 0).unwrap();
        let f_order = ravel(coordinates, &[7, 6], &[0], 1).unwrap();
        let mixed_modes = ravel(coordinates, &[4, 4], &[2, 1], 0).unwrap();
        assert_eq!(ints(c_order).unwrap(), [22, 41, 37]);
        assert_eq!(ints(f_order).unwrap(), [31, 41, 13]);
        assert_eq!(ints(mixed_modes).unwrap(), [12, 13, 13]);

        let scalar_coordinates = owned_int(&[3, 1, 4, 1], &[4]).unwrap();
        let scalar = ravel(scalar_coordinates, &[6, 7, 8, 9], &[0], 0).unwrap();
        assert_eq!(shape(scalar).unwrap(), []);
        assert_eq!(ints(scalar).unwrap(), [1621]);

        delete_arrays(&[
            scalar,
            scalar_coordinates,
            mixed_modes,
            f_order,
            c_order,
            coordinates,
        ]);
    }
}

#[test]
fn unravel_preserves_index_shape_order_empty_axes_and_zero_rank() {
    unsafe {
        let c_indexes = owned_int(&[22, 41, 37], &[3]).unwrap();
        let c_axes = unravel(c_indexes, &[7, 6], 0).unwrap();
        assert_eq!(ints(c_axes[0]).unwrap(), [3, 6, 6]);
        assert_eq!(ints(c_axes[1]).unwrap(), [4, 5, 1]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_set_int(c_axes[0], [0_i64].as_ptr(), 1, 1),
            OK
        );
        assert_eq!(ints(c_axes[0]).unwrap(), [1, 6, 6]);

        let f_indexes = owned_int(&[31, 41, 13], &[3]).unwrap();
        let f_axes = unravel(f_indexes, &[7, 6], 1).unwrap();
        assert_eq!(ints(f_axes[0]).unwrap(), [3, 6, 6]);
        assert_eq!(ints(f_axes[1]).unwrap(), [4, 5, 1]);

        let matrix_indexes = owned_int(&[0, 1, 2, 3], &[2, 2]).unwrap();
        let matrix_axes = unravel(matrix_indexes, &[2, 2], 0).unwrap();
        assert_eq!(shape(matrix_axes[0]).unwrap(), [2, 2]);
        assert_eq!(shape(matrix_axes[1]).unwrap(), [2, 2]);

        let empty_indexes = owned_int(&[], &[0]).unwrap();
        let empty_axes = unravel(empty_indexes, &[2, 1, 0], 0).unwrap();
        assert_eq!(empty_axes.len(), 3);
        assert!(empty_axes.iter().all(|axis| shape(*axis).unwrap() == [0]));

        let scalar_zero = owned_int(&[0], &[]).unwrap();
        assert!(unravel(scalar_zero, &[], 0).unwrap().is_empty());

        let mut owned = vec![
            scalar_zero,
            empty_indexes,
            matrix_indexes,
            f_indexes,
            c_indexes,
        ];
        owned.extend(c_axes);
        owned.extend(f_axes);
        owned.extend(matrix_axes);
        owned.extend(empty_axes);
        delete_arrays(&owned);
    }
}

#[test]
fn ravel_round_trips_every_admitted_signed_and_unsigned_index_dtype() {
    unsafe {
        let coordinate_values = [1, 0, 1, 2, 3, 4, 1, 6, 1, 3, 2, 0];
        for dtype_code in [2, 4, 5, 6, 7, 8, 9, 10] {
            let coordinates = owned_int_as(&coordinate_values, &[2, 6], dtype_code).unwrap();
            let c_indexes = ravel(coordinates, &[5, 8], &[0], 0).unwrap();
            let f_indexes = ravel(coordinates, &[5, 8], &[0], 1).unwrap();
            assert_eq!(ints(c_indexes).unwrap(), [9, 6, 9, 19, 26, 32]);
            assert_eq!(ints(f_indexes).unwrap(), [6, 30, 6, 17, 13, 4]);
            let c_axes = unravel(c_indexes, &[5, 8], 0).unwrap();
            let f_axes = unravel(f_indexes, &[5, 8], 1).unwrap();
            assert_eq!(ints(c_axes[0]).unwrap(), coordinate_values[..6]);
            assert_eq!(ints(c_axes[1]).unwrap(), coordinate_values[6..]);
            assert_eq!(ints(f_axes[0]).unwrap(), coordinate_values[..6]);
            assert_eq!(ints(f_axes[1]).unwrap(), coordinate_values[6..]);
            assert!(c_axes
                .iter()
                .chain(&f_axes)
                .all(|axis| dtype(*axis) == Ok(2)));
            let mut owned = vec![f_indexes, c_indexes, coordinates];
            owned.extend(c_axes);
            owned.extend(f_axes);
            delete_arrays(&owned);
        }
    }
}

#[test]
fn boolean_ix_and_repeated_inputs_preserve_numpy_open_mesh_shapes() {
    unsafe {
        let mask = owned_bool(&[1, 0, 1, 1], &[4]).unwrap();
        let mut raw = std::ptr::null_mut();
        let mut length = 0usize;
        assert_eq!(
            terlan_ndarray_v1_ix(mask, std::ptr::null(), 0, &mut raw, &mut length),
            OK
        );
        let boolean_mesh = take_arrays(raw, length);
        assert_eq!(shape(boolean_mesh[0]).unwrap(), [3]);
        assert_eq!(ints(boolean_mesh[0]).unwrap(), [0, 2, 3]);
        assert_eq!(dtype(boolean_mesh[0]).unwrap(), 2);

        let source = owned_int(&[0, 1, 2, 3, 4], &[5]).unwrap();
        let peers = [source as usize as u64];
        raw = std::ptr::null_mut();
        length = 0;
        assert_eq!(
            terlan_ndarray_v1_ix(source, peers.as_ptr(), 1, &mut raw, &mut length),
            OK
        );
        let repeated = take_arrays(raw, length);
        assert_eq!(shape(repeated[0]).unwrap(), [5, 1]);
        assert_eq!(shape(repeated[1]).unwrap(), [1, 5]);
        assert_eq!(shape(source).unwrap(), [5]);

        let empty_integer = owned_int(&[], &[0]).unwrap();
        raw = std::ptr::null_mut();
        length = 0;
        assert_eq!(
            terlan_ndarray_v1_ix(empty_integer, std::ptr::null(), 0, &mut raw, &mut length,),
            OK
        );
        let integer_empty_mesh = take_arrays(raw, length);
        assert_eq!(shape(integer_empty_mesh[0]).unwrap(), [0]);
        assert_eq!(dtype(integer_empty_mesh[0]).unwrap(), 2);

        let empty_float = owned_real(&[], &[0], 11).unwrap();
        raw = std::ptr::null_mut();
        length = 0;
        assert_eq!(
            terlan_ndarray_v1_ix(empty_float, std::ptr::null(), 0, &mut raw, &mut length,),
            OK
        );
        let float_empty_mesh = take_arrays(raw, length);
        assert_eq!(shape(float_empty_mesh[0]).unwrap(), [0]);
        assert_eq!(dtype(float_empty_mesh[0]).unwrap(), 11);

        let mut owned = vec![empty_float, empty_integer, source, mask];
        owned.extend(boolean_mesh);
        owned.extend(repeated);
        owned.extend(integer_empty_mesh);
        owned.extend(float_empty_mesh);
        delete_arrays(&owned);
    }
}

#[test]
fn diagonal_helpers_cover_wrapping_high_rank_and_shape_validation() {
    unsafe {
        let tall = owned_int(&[0; 30], &[10, 3]).unwrap();
        assert_eq!(
            terlan_ndarray_v1_fill_diagonal_with_wrap(tall, 5.0, true),
            OK
        );
        assert_eq!(
            ints(tall).unwrap(),
            [
                5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0,
                5, 0,
            ]
        );

        let cube = owned_int(&[0; 27], &[3, 3, 3]).unwrap();
        assert_eq!(
            terlan_ndarray_v1_fill_diagonal_with_wrap(cube, 4.0, true),
            OK
        );
        assert_eq!(
            ints(cube)
                .unwrap()
                .iter()
                .filter(|value| **value == 4)
                .count(),
            3
        );

        let mismatched = owned_int(&[0; 54], &[3, 3, 2, 3]).unwrap();
        assert_eq!(
            terlan_ndarray_v1_fill_diagonal_with_wrap(mismatched, 2.0, false),
            SHAPE_MISMATCH
        );
        let vector = owned_int(&[0; 3], &[3]).unwrap();
        assert_eq!(
            terlan_ndarray_v1_fill_diagonal_with_wrap(vector, 2.0, false),
            SHAPE_MISMATCH
        );

        let mut indexes = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_diag_indices(2, 3, &mut indexes), OK);
        assert_eq!(shape(indexes).unwrap(), [3, 2]);
        assert_eq!(ints(indexes).unwrap(), [0, 1, 0, 1, 0, 1]);
        let mut from_cube = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_diag_indices_from(cube, &mut from_cube),
            OK
        );
        assert_eq!(ints(from_cube).unwrap(), [0, 1, 2, 0, 1, 2, 0, 1, 2]);
        let mut rejected = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_diag_indices_from(mismatched, &mut rejected),
            SHAPE_MISMATCH
        );
        assert!(rejected.is_null());

        delete_arrays(&[from_cube, indexes, vector, mismatched, cube, tall]);
    }
}

#[test]
fn index_tricks_reject_invalid_modes_dtypes_bounds_and_overflow() {
    unsafe {
        let coordinates = owned_int(&[-1, 0], &[2]).unwrap();
        assert_eq!(ravel(coordinates, &[2, 2], &[0], 0), Err(AXIS_OUT_OF_RANGE));
        assert_eq!(ravel(coordinates, &[2, 2], &[3], 0), Err(INVALID_ARGUMENT));
        assert_eq!(ravel(coordinates, &[2, 2], &[0], 2), Err(INVALID_ARGUMENT));
        assert_eq!(
            ravel(coordinates, &[2, 2], &[0, 0, 0], 0),
            Err(INVALID_ARGUMENT)
        );

        let floating = owned_real(&[0.0, 0.0], &[2], 3).unwrap();
        assert_eq!(ravel(floating, &[2, 2], &[0], 0), Err(UNSUPPORTED_DTYPE));
        assert_eq!(unravel(floating, &[2, 2], 0), Err(UNSUPPORTED_DTYPE));

        let empty_coordinates = owned_int(&[], &[3, 0]).unwrap();
        let empty = ravel(empty_coordinates, &[2, 1, 0], &[0], 0).unwrap();
        assert_eq!(shape(empty).unwrap(), [0]);
        let one_coordinate = owned_int(&[0, 0, 0], &[3]).unwrap();
        assert_eq!(
            ravel(one_coordinate, &[2, 1, 0], &[0], 0),
            Err(AXIS_OUT_OF_RANGE)
        );

        let zero = owned_int(&[0], &[]).unwrap();
        assert_eq!(unravel(zero, &[2, 2], 2), Err(INVALID_ARGUMENT));
        let out_of_bounds = owned_int(&[4], &[]).unwrap();
        assert_eq!(unravel(out_of_bounds, &[2, 2], 0), Err(AXIS_OUT_OF_RANGE));
        let half_max = i64::MAX / 2;
        assert_eq!(
            ravel(coordinates, &[half_max + 1, 2], &[2], 0),
            Err(SIZE_OVERFLOW)
        );

        delete_arrays(&[
            out_of_bounds,
            zero,
            one_coordinate,
            empty,
            empty_coordinates,
            floating,
            coordinates,
        ]);
    }
}
