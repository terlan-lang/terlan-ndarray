//! Broadcasted assignment tests based on upstream indexing behavior.

use super::super::native_support::{
    delete_arrays, floats, owned_int, owned_real, shape, Array, OK, SHAPE_MISMATCH,
};

/// Creates one owned real-valued test array.
unsafe fn real(values: &[f64], dimensions: &[i64]) -> Array {
    unsafe { owned_real(values, dimensions, 3).unwrap() }
}

/// Creates one owned integer coordinate or index array.
unsafe fn integers(values: &[i64], dimensions: &[i64]) -> Array {
    unsafe { owned_int(values, dimensions).unwrap() }
}

/// Reads logical real values from one array.
unsafe fn values(array: Array) -> Vec<f64> {
    unsafe { floats(array, false).unwrap() }
}

/// Deletes all supplied arrays.
unsafe fn dispose(arrays: &[Array]) {
    unsafe { delete_arrays(arrays) }
}

#[test]
fn assignment_accepts_only_extra_leading_singleton_dimensions() {
    unsafe {
        let matrix = real(&[0.0; 6], &[2, 3]);
        let row = real(&[0.0, 1.0, 2.0], &[3]);
        assert_eq!(crate::ffi::terlan_ndarray_v1_copyto(matrix, row), OK);
        assert_eq!(values(matrix), [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
        let column = real(&[0.0, 1.0], &[2, 1]);
        assert_eq!(crate::ffi::terlan_ndarray_v1_copyto(matrix, column), OK);
        assert_eq!(values(matrix), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let destination = real(&[0.0; 6], &[3, 2]);
        let leading_ones = real(&[1.0; 6], &[1, 1, 3, 2]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copyto(destination, leading_ones),
            OK
        );
        assert_eq!(values(destination), [1.0; 6]);

        let narrow_destination = real(&[0.0; 5], &[5, 1]);
        let wide = real(&[0.0; 10], &[5, 2]);
        let empty_width = real(&[], &[5, 0]);
        for source in [wide, empty_width] {
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_copyto(narrow_destination, source),
                SHAPE_MISMATCH
            );
            assert_eq!(values(narrow_destination), [0.0; 5]);
        }

        let invalid = real(&[2.0; 12], &[2, 1, 3, 2]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copyto(destination, invalid),
            SHAPE_MISMATCH
        );
        assert_eq!(values(destination), [1.0; 6]);
        dispose(&[
            invalid,
            empty_width,
            wide,
            narrow_destination,
            leading_ones,
            destination,
            column,
            row,
            matrix,
        ]);
    }
}

#[test]
fn coordinate_grids_preserve_selection_shape_and_broadcast_values() {
    unsafe {
        let matrix = real(&[0.0; 25], &[5, 5]);
        let coordinate_values = [0, 0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 2, 2, 0, 2, 1, 2, 2];
        let coordinates = integers(&coordinate_values, &[3, 3, 2]);
        let replacements = real(&[2.0, 3.0, 4.0], &[3]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put_coordinates(matrix, coordinates, replacements,),
            OK
        );
        assert_eq!(
            values(matrix),
            [
                2.0, 3.0, 4.0, 0.0, 0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );

        let mut gathered = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_gather_coordinates(matrix, coordinates, &mut gathered,),
            OK
        );
        assert_eq!(shape(gathered).unwrap(), [3, 3]);
        assert_eq!(
            values(gathered),
            [2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0]
        );

        let invalid = real(&[9.0; 6], &[2, 1, 3]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put_coordinates(matrix, coordinates, invalid),
            SHAPE_MISMATCH
        );
        assert_eq!(
            values(matrix)[..15],
            [2.0, 3.0, 4.0, 0.0, 0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 2.0, 3.0, 4.0, 0.0, 0.0]
        );
        dispose(&[invalid, gathered, replacements, coordinates, matrix]);
    }
}

#[test]
fn along_axis_assignment_broadcasts_subspaces_and_leading_ones() {
    unsafe {
        let matrix = real(&[0.0; 6], &[3, 2]);
        let rows = integers(&[0, 1, 2], &[3, 1]);
        let row_values = real(&[1.0; 6], &[1, 3, 2]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put_along_axis(matrix, rows, row_values, 0),
            OK
        );
        assert_eq!(values(matrix), [1.0; 6]);

        let columns = integers(&[0, 1], &[1, 2]);
        let column_values = real(&[2.0; 6], &[1, 3, 2]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put_along_axis(matrix, columns, column_values, 1,),
            OK
        );
        assert_eq!(values(matrix), [2.0; 6]);

        let coordinate_values = [0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1];
        let coordinates = integers(&coordinate_values, &[3, 2, 2]);
        let coordinate_replacements = real(&[3.0; 6], &[1, 3, 2]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put_coordinates(
                matrix,
                coordinates,
                coordinate_replacements,
            ),
            OK
        );
        assert_eq!(values(matrix), [3.0; 6]);

        let reversed = real(&(0..100).map(f64::from).collect::<Vec<_>>(), &[100, 1]);
        let square = real(&[0.0; 10_000], &[100, 100]);
        let reverse_rows = integers(&(0..100).rev().collect::<Vec<_>>(), &[100, 1]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put_along_axis(square, reverse_rows, reversed, 0,),
            OK
        );
        let square_values = values(square);
        for row in 0..100 {
            assert!(square_values[row * 100..(row + 1) * 100]
                .iter()
                .all(|value| *value == f64::from(99 - row as i32)));
        }
        dispose(&[
            reverse_rows,
            square,
            reversed,
            coordinate_replacements,
            coordinates,
            column_values,
            columns,
            row_values,
            rows,
            matrix,
        ]);
    }
}
