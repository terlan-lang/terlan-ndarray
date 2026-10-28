//! Strided advanced-index selection and assignment regressions.

use super::super::native_support::{
    delete_arrays, ints, owned_bool, owned_int, owned_real, shape, Array, OK,
};

const DTYPE_MISMATCH: i32 = 106;
const SHAPE_MISMATCH: i32 = 108;
const INDEX_OUT_OF_RANGE: i32 = 120;

/// Creates one owned integer array.
unsafe fn array(values: &[i64], dimensions: &[i64]) -> Array {
    unsafe { owned_int(values, dimensions).unwrap() }
}

/// Reads logical integer values.
unsafe fn values(source: Array) -> Vec<i64> {
    unsafe { ints(source).unwrap() }
}

/// Deletes all supplied arrays.
unsafe fn dispose(arrays: &[Array]) {
    unsafe { delete_arrays(arrays) }
}

/// Creates a reversed view over one axis.
unsafe fn reversed(source: Array, axis: i64) -> Array {
    let mut output = std::ptr::null_mut();
    assert_eq!(
        unsafe { crate::ffi::terlan_ndarray_v1_flip(source, &axis, 1, &mut output) },
        OK
    );
    output
}

/// Gathers a one-dimensional source through an arbitrarily shaped index array.
unsafe fn gather_one_dimensional(source: Array, indexes: Array) -> Result<Array, i32> {
    let rank = unsafe { shape(indexes) }?.len() as i64;
    let mut coordinates = std::ptr::null_mut();
    let status =
        unsafe { crate::ffi::terlan_ndarray_v1_expand_dims(indexes, rank, &mut coordinates) };
    if status != OK {
        return Err(status);
    }
    let mut output = std::ptr::null_mut();
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_gather_coordinates(source, coordinates, &mut output)
    };
    let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(coordinates) };
    if status == OK {
        Ok(output)
    } else {
        Err(status)
    }
}

#[test]
fn empty_and_out_of_range_fancy_indexes_are_atomic() {
    unsafe {
        let source = array(&[0, 1, 2, 3, 4], &[5]);
        let empty = array(&[], &[0]);
        let mut selected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_take_along_axis(source, empty, 0, &mut selected,),
            OK
        );
        assert_eq!(shape(selected).unwrap(), [0]);
        assert!(values(selected).is_empty());

        for bad_position in [0_usize, 19] {
            let mut indexes = vec![1_i64; 20];
            indexes[bad_position] = 10;
            let indexes = array(&indexes, &[20]);
            let replacement = array(&[9], &[]);
            let mut rejected = 1_usize as Array;
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_take_along_axis(source, indexes, 0, &mut rejected,),
                INDEX_OUT_OF_RANGE
            );
            assert!(rejected.is_null());
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_put_along_axis(source, indexes, replacement, 0,),
                INDEX_OUT_OF_RANGE
            );
            assert_eq!(values(source), [0, 1, 2, 3, 4]);
            dispose(&[replacement, indexes]);
        }
        dispose(&[selected, empty, source]);
    }
}

#[test]
fn noncontiguous_indexes_preserve_their_logical_order() {
    unsafe {
        let source = array(&[0, 1, 2, 3, 4, 5], &[6]);
        let backing = array(&[0, 99, 3, 99], &[4]);
        let mut indexes = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(backing, 0, 0, 4, 2, &mut indexes,),
            OK
        );
        let mut selected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_take_along_axis(source, indexes, 0, &mut selected,),
            OK
        );
        assert_eq!(values(selected), [0, 3]);
        let replacement = array(&[-1], &[]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put_along_axis(source, indexes, replacement, 0,),
            OK
        );
        assert_eq!(values(source), [-1, 1, 2, -1, 4, 5]);
        dispose(&[replacement, selected, indexes, backing, source]);
    }
}

#[test]
fn strided_coordinate_grids_select_in_logical_order() {
    unsafe {
        let source = array(&[10, 20, 30], &[3]);
        let coordinate_values = (0..24).map(|index| index % 3).collect::<Vec<_>>();
        let backing = array(&coordinate_values, &[3, 4, 2, 1]);
        let mut coordinates = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_transpose(backing, 0, 1, &mut coordinates),
            OK
        );
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(backing), OK);
        let logical_coordinates = values(coordinates);
        let mut selected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_gather_coordinates(source, coordinates, &mut selected,),
            OK
        );
        assert_eq!(shape(selected).unwrap(), [4, 3, 2]);
        assert_eq!(
            values(selected),
            logical_coordinates
                .iter()
                .map(|index| [10, 20, 30][*index as usize])
                .collect::<Vec<_>>()
        );
        dispose(&[selected, coordinates, source]);
    }
}

#[test]
fn reversed_indexes_values_and_destinations_assign_deterministically() {
    unsafe {
        let target = array(&[1; 10], &[5, 2]);
        let index_backing = array(&[0; 5], &[5, 1]);
        let indexes = reversed(index_backing, 0);
        let value_backing = array(&(0..10).collect::<Vec<_>>(), &[5, 2]);
        let replacements = reversed(value_backing, 0);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put_along_axis(target, indexes, replacements, 0,),
            OK
        );
        assert_eq!(values(target), [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]);

        let matrix = array(&[0; 16], &[4, 4]);
        let reversed_rows = reversed(matrix, 0);
        let reversed_matrix = reversed(reversed_rows, 1);
        let columns = array(&[0, 1, 2, 3], &[1, 4]);
        let ten = array(&[10], &[]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put_along_axis(reversed_matrix, columns, ten, 1,),
            OK
        );
        assert_eq!(values(matrix), [10; 16]);
        dispose(&[
            ten,
            columns,
            reversed_matrix,
            reversed_rows,
            matrix,
            replacements,
            value_backing,
            indexes,
            index_backing,
            target,
        ]);
    }
}

#[test]
fn noncontiguous_subspace_values_assign_without_physical_order_leaks() {
    unsafe {
        let target = array(&[-1; 24], &[3, 4, 2]);
        let indexes = array(&[0, 1], &[2, 1, 1]);
        let backing = array(&(0..16).collect::<Vec<_>>(), &[2, 4, 2]);
        let mut replacements = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_transpose(backing, 0, 2, &mut replacements),
            OK
        );
        assert_eq!(shape(replacements).unwrap(), [2, 4, 2]);
        let expected = values(replacements);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put_along_axis(target, indexes, replacements, 0,),
            OK
        );
        assert_eq!(values(target)[..16], expected);
        assert_eq!(values(target)[16..], [-1; 8]);
        dispose(&[replacements, backing, indexes, target]);
    }
}

#[test]
fn fortran_index_grids_preserve_result_memory_order() {
    unsafe {
        let index_backing = array(&(0..10).collect::<Vec<_>>(), &[5, 2]);
        let mut indexes = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_transpose(index_backing, 0, 1, &mut indexes),
            OK
        );
        let mut fortran = 0;
        let mut contiguous = 1;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_is_fortran_contiguous(indexes, &mut fortran),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_is_contiguous(indexes, &mut contiguous),
            OK
        );
        assert_eq!((fortran, contiguous), (1, 0));

        let column = array(&(0..10).collect::<Vec<_>>(), &[10, 1]);
        let mut selected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_take_along_axis(column, indexes, 0, &mut selected,),
            OK
        );
        assert_eq!(shape(selected).unwrap(), [2, 5]);
        assert_eq!(values(selected), values(indexes));
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_is_fortran_contiguous(selected, &mut fortran),
            OK
        );
        assert_eq!(fortran, 1);

        let vector = array(&(0..10).collect::<Vec<_>>(), &[10]);
        let mut coordinates = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_expand_dims(indexes, 2, &mut coordinates),
            OK
        );
        let mut gathered = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_gather_coordinates(vector, coordinates, &mut gathered,),
            OK
        );
        assert_eq!(values(gathered), values(indexes));
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_is_fortran_contiguous(gathered, &mut fortran),
            OK
        );
        assert_eq!(fortran, 1);
        dispose(&[
            gathered,
            coordinates,
            vector,
            selected,
            column,
            indexes,
            index_backing,
        ]);
    }
}

#[test]
fn one_dimensional_index_matrix_covers_shapes_slices_and_rejections() {
    unsafe {
        let source = array(&(0..10).collect::<Vec<_>>(), &[10]);
        let mut full = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_basic_slice(
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
                &mut full,
            ),
            OK
        );
        assert_eq!(values(full), values(source));

        let mut scalar = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_index_axis(source, 0, 0, &mut scalar),
            OK
        );
        assert_eq!(shape(scalar).unwrap(), []);
        assert_eq!(values(scalar), [0]);

        for (start, stop, step, expected) in [
            (-5, 5, 2, &[][..]),
            (1, 1, 100, &[][..]),
            (4, -1, -2, &[][..]),
            (i64::MIN, i64::MIN, -3, &[9, 6, 3, 0][..]),
        ] {
            let mut sliced = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_slice_axis(source, 0, start, stop, step, &mut sliced,),
                OK
            );
            assert_eq!(values(sliced), expected);
            dispose(&[sliced]);
        }

        let index_cases: &[(&[i64], &[i64], &[i64])] = &[
            (&[], &[0, 1, 1], &[]),
            (&[0, 1, -2], &[3], &[0, 1, 8]),
            (&[2, 0, 1], &[3, 1], &[2, 0, 1]),
            (&[2, -1], &[2], &[2, 9]),
        ];
        for &(index_values, index_shape, expected) in index_cases {
            let indexes = array(index_values, index_shape);
            let selected = gather_one_dimensional(source, indexes).unwrap();
            assert_eq!(shape(selected).unwrap(), index_shape);
            assert_eq!(values(selected), expected);
            dispose(&[selected, indexes]);
        }

        let high_rank_shape = [1_i64; 32];
        let high_rank_indexes = array(&[0], &high_rank_shape[..31]);
        let high_rank = gather_one_dimensional(source, high_rank_indexes).unwrap();
        assert_eq!(shape(high_rank).unwrap(), high_rank_shape[..31]);
        assert_eq!(values(high_rank), [0]);

        for mask_shape in [&[3_i64][..], &[2_i64, 2][..], &[1_i64, 2, 2][..]] {
            let mask = owned_bool(
                &vec![0; mask_shape.iter().product::<i64>() as usize],
                mask_shape,
            )
            .unwrap();
            let mut rejected = 1_usize as Array;
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_mask_select(source, mask, &mut rejected),
                SHAPE_MISMATCH
            );
            assert!(rejected.is_null());
            dispose(&[mask]);
        }

        let float_indexes = owned_real(&[0.0, 1.0], &[2], 3).unwrap();
        assert_eq!(
            gather_one_dimensional(source, float_indexes),
            Err(DTYPE_MISMATCH)
        );
        dispose(&[
            float_indexes,
            high_rank,
            high_rank_indexes,
            scalar,
            full,
            source,
        ]);
    }
}
