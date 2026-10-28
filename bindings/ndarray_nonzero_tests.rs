//! NumPy `nonzero`, `argwhere`, and axis-aware count conformance.

use super::native_support::{
    delete_arrays, dtype, ints, owned_bool, owned_numeric_as, shape, take_arrays, Array,
    AXIS_OUT_OF_RANGE, INVALID_ARGUMENT, OK,
};
use rustfft::num_complex::Complex64;

/// Counts every truthy element in one array.
unsafe fn count_nonzero(array: Array) -> i64 {
    let mut count = -1_i64;
    assert_eq!(
        unsafe { crate::ffi::terlan_ndarray_v1_count_nonzero(array, &mut count) },
        OK
    );
    count
}

/// Reduces truth values over an explicit axis tuple.
unsafe fn count_axes(array: Array, axes: &[i64], keep_dims: bool) -> Array {
    let mut output = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            crate::ffi::terlan_ndarray_v1_count_nonzero_axes(
                array,
                axes.as_ptr(),
                axes.len() as i64,
                i32::from(keep_dims),
                &mut output,
            )
        },
        OK
    );
    assert!(!output.is_null());
    output
}

/// Collects and owns every coordinate array returned by `nonzero`.
unsafe fn nonzero(array: Array) -> Vec<Array> {
    let mut values = std::ptr::null_mut();
    let mut length = 0_usize;
    assert_eq!(
        unsafe { crate::ffi::terlan_ndarray_v1_nonzero(array, &mut values, &mut length) },
        OK
    );
    unsafe { take_arrays(values, length) }
}

/// Creates one admitted numeric array from canonical complex values.
unsafe fn numeric(values: &[(f64, f64)], shape: &[i64], dtype: i32) -> Array {
    let values = values
        .iter()
        .map(|&(real, imag)| Complex64::new(real, imag))
        .collect::<Vec<_>>();
    unsafe { owned_numeric_as(&values, shape, dtype) }.unwrap()
}

#[test]
fn nonzero_trivial_and_scalar_results_match_numpy_1_21() {
    unsafe {
        for dtype in [1, 3] {
            for (values, expected_count, expected_coordinates) in [
                (&[][..], 0_i64, &[][..]),
                (&[(0.0, 0.0)][..], 0, &[][..]),
                (&[(1.0, 0.0)][..], 1, &[0_i64][..]),
            ] {
                let array = numeric(values, &[values.len() as i64], dtype);
                assert_eq!(count_nonzero(array), expected_count);
                let coordinates = nonzero(array);
                assert_eq!(coordinates.len(), 1);
                assert_eq!(ints(coordinates[0]).unwrap(), expected_coordinates);
                delete_arrays(&coordinates);
                delete_arrays(&[array]);
            }
        }

        for (value, expected) in [(0_u8, &[][..]), (1_u8, &[0_i64][..])] {
            let scalar = owned_bool(&[value], &[]).unwrap();
            assert_eq!(count_nonzero(scalar), value as i64);
            let coordinates = nonzero(scalar);
            assert_eq!(coordinates.len(), 1);
            assert_eq!(shape(coordinates[0]).unwrap(), [expected.len() as i64]);
            assert_eq!(ints(coordinates[0]).unwrap(), expected);
            delete_arrays(&coordinates);
            delete_arrays(&[scalar]);
        }
    }
}

#[test]
fn nonzero_coordinates_match_one_two_dimensional_and_sparse_numpy_results() {
    unsafe {
        let vector = numeric(
            &[
                (1.0, 0.0),
                (0.0, 0.0),
                (2.0, 0.0),
                (-1.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (8.0, 0.0),
            ],
            &[7],
            3,
        );
        assert_eq!(count_nonzero(vector), 4);
        let coordinates = nonzero(vector);
        assert_eq!(ints(coordinates[0]).unwrap(), [0, 2, 3, 6]);
        delete_arrays(&coordinates);

        let high_rank_shape = [10_i64, 2, 3, 4, 5, 6];
        let high_rank_values = vec![(1.0, 0.0); 7_200];
        let high_rank = numeric(&high_rank_values, &high_rank_shape, 3);
        let first = nonzero(high_rank);
        let second = nonzero(high_rank);
        assert_eq!(first.len(), 6);
        assert_eq!(second.len(), 6);
        for axis in 0..6 {
            assert_eq!(ints(first[axis]).unwrap(), ints(second[axis]).unwrap());
            assert_eq!(shape(first[axis]).unwrap(), [7_200]);
        }
        assert_eq!(ints(first[0]).unwrap()[..720], [0; 720]);
        assert_eq!(ints(first[5]).unwrap()[..6], [0, 1, 2, 3, 4, 5]);
        delete_arrays(&first);
        delete_arrays(&second);
        delete_arrays(&[high_rank]);

        let matrix = numeric(
            &[
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 0.0),
                (2.0, 0.0),
                (0.0, 0.0),
                (3.0, 0.0),
            ],
            &[2, 3],
            3,
        );
        let coordinates = nonzero(matrix);
        assert_eq!(coordinates.len(), 2);
        assert_eq!(ints(coordinates[0]).unwrap(), [0, 1, 1]);
        assert_eq!(ints(coordinates[1]).unwrap(), [1, 0, 2]);
        delete_arrays(&coordinates);

        for offset in 0..20_usize {
            let mut values = vec![0_u8; 200];
            for index in (offset..200).step_by(20) {
                values[index] = 1;
            }
            let sparse = owned_bool(&values, &[200]).unwrap();
            let coordinates = nonzero(sparse);
            assert_eq!(
                ints(coordinates[0]).unwrap(),
                (offset..200)
                    .step_by(20)
                    .map(|value| value as i64)
                    .collect::<Vec<_>>()
            );
            delete_arrays(&coordinates);
            delete_arrays(&[sparse]);
        }
        delete_arrays(&[matrix, vector]);
    }
}

#[test]
fn count_nonzero_axes_matches_all_admitted_dtypes_and_empty_axis_identity() {
    unsafe {
        let source = [
            (1.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        ];
        for source_dtype in 1..=14 {
            let array = numeric(&source, &[3, 3], source_dtype);
            let rows = count_axes(array, &[1], false);
            let columns = count_axes(array, &[0], false);
            let all = count_axes(array, &[0, 1], false);
            let identity = count_axes(array, &[], false);
            assert_eq!(dtype(rows).unwrap(), 2);
            assert_eq!(ints(rows).unwrap(), [1, 1, 0]);
            assert_eq!(ints(columns).unwrap(), [2, 0, 0]);
            assert_eq!(shape(all).unwrap(), []);
            assert_eq!(ints(all).unwrap(), [2]);
            assert_eq!(shape(identity).unwrap(), [3, 3]);
            assert_eq!(ints(identity).unwrap(), [1, 0, 0, 1, 0, 0, 0, 0, 0]);
            assert_eq!(count_nonzero(array), 2);
            delete_arrays(&[identity, all, columns, rows, array]);
        }

        for source_dtype in [12, 13] {
            let array = numeric(&[(0.0, 1.0), (0.0, 0.0)], &[2], source_dtype);
            assert_eq!(count_nonzero(array), 1);
            let coordinates = nonzero(array);
            assert_eq!(ints(coordinates[0]).unwrap(), [0]);
            delete_arrays(&coordinates);
            delete_arrays(&[array]);
        }
    }
}

#[test]
fn count_nonzero_axis_order_empty_dimensions_and_keepdims_match_numpy() {
    unsafe {
        let values = (0..16)
            .map(|index| {
                if index % 3 == 0 {
                    (1.0, 0.0)
                } else {
                    (0.0, 0.0)
                }
            })
            .collect::<Vec<_>>();
        let cube = numeric(&values, &[2, 2, 2, 2], 3);
        for axes in [&[0_i64, 2_i64][..], &[2_i64, 0_i64][..]] {
            let result = count_axes(cube, axes, false);
            assert_eq!(shape(result).unwrap(), [2, 2]);
            assert_eq!(ints(result).unwrap(), [1, 2, 2, 1]);
            delete_arrays(&[result]);
        }

        let matrix = numeric(
            &[
                (0.0, 0.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 0.0),
                (0.0, 0.0),
                (3.0, 0.0),
                (5.0, 0.0),
                (0.0, 0.0),
                (7.0, 0.0),
                (9.0, 0.0),
                (2.0, 0.0),
                (0.0, 0.0),
            ],
            &[3, 4],
            3,
        );
        let columns = count_axes(matrix, &[0], true);
        let rows = count_axes(matrix, &[1], true);
        let all = count_axes(matrix, &[0, 1], true);
        assert_eq!(shape(columns).unwrap(), [1, 4]);
        assert_eq!(ints(columns).unwrap(), [1, 2, 3, 0]);
        assert_eq!(shape(rows).unwrap(), [3, 1]);
        assert_eq!(ints(rows).unwrap(), [1, 2, 3]);
        assert_eq!(shape(all).unwrap(), [1, 1]);
        assert_eq!(ints(all).unwrap(), [6]);

        let empty = numeric(&[], &[2, 0, 3], 11);
        let over_zero = count_axes(empty, &[1], false);
        let keep_zero = count_axes(empty, &[1], true);
        assert_eq!(shape(over_zero).unwrap(), [2, 3]);
        assert_eq!(ints(over_zero).unwrap(), [0; 6]);
        assert_eq!(shape(keep_zero).unwrap(), [2, 1, 3]);
        assert_eq!(ints(keep_zero).unwrap(), [0; 6]);
        delete_arrays(&[
            keep_zero, over_zero, empty, all, rows, columns, matrix, cube,
        ]);
    }
}

#[test]
fn argwhere_matches_rank_zero_one_two_and_exact_coordinate_order() {
    unsafe {
        for rank in 0..=2_usize {
            let dimensions = vec![2_i64; rank];
            let size = 1_usize << rank;
            for (values, expected_rows) in [
                (vec![0_u8; size], 0_usize),
                (
                    {
                        let mut values = vec![0_u8; size];
                        values[0] = 1;
                        values
                    },
                    1,
                ),
                (
                    {
                        let mut values = vec![1_u8; size];
                        values[0] = 0;
                        values
                    },
                    size - 1,
                ),
                (vec![1_u8; size], size),
            ] {
                let array = owned_bool(&values, &dimensions).unwrap();
                let mut output = std::ptr::null_mut();
                assert_eq!(
                    crate::ffi::terlan_ndarray_v1_argwhere(array, &mut output),
                    OK
                );
                assert_eq!(shape(output).unwrap(), [expected_rows as i64, rank as i64]);
                delete_arrays(&[output, array]);
            }
        }

        let matrix = numeric(
            &[
                (0.0, 0.0),
                (1.0, 0.0),
                (2.0, 0.0),
                (3.0, 0.0),
                (4.0, 0.0),
                (5.0, 0.0),
            ],
            &[2, 3],
            3,
        );
        let mut output = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_argwhere(matrix, &mut output),
            OK
        );
        assert_eq!(ints(output).unwrap(), [0, 1, 0, 2, 1, 0, 1, 1, 1, 2]);
        delete_arrays(&[output, matrix]);
    }
}

#[test]
fn nonzero_family_rejects_malformed_calls_without_partial_outputs() {
    unsafe {
        let array = numeric(&[(0.0, 0.0), (1.0, 0.0)], &[2], 3);
        let mut output = 1_usize as Array;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_count_nonzero_axes(
                array,
                [0_i64, 0_i64].as_ptr(),
                2,
                0,
                &mut output,
            ),
            111
        );
        assert!(output.is_null());
        output = 1_usize as Array;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_count_nonzero_axes(
                array,
                [1_i64].as_ptr(),
                1,
                0,
                &mut output,
            ),
            AXIS_OUT_OF_RANGE
        );
        assert!(output.is_null());
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_count_nonzero_axes(
                array,
                std::ptr::null(),
                1,
                0,
                &mut output,
            ),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());

        let mut values = 1_usize as *mut u64;
        let mut length = usize::MAX;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_nonzero(std::ptr::null_mut(), &mut values, &mut length,),
            INVALID_ARGUMENT
        );
        assert!(values.is_null());
        assert_eq!(length, 0);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_nonzero(array, std::ptr::null_mut(), &mut length),
            INVALID_ARGUMENT
        );
        delete_arrays(&[array]);
    }
}
