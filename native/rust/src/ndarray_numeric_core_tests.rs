//! NumPy `test_numeric` conformance for admitted core numerical operations.

use super::native_support::{
    delete_arrays, dtype, ints, numeric, owned_int, shape, take_arrays, Array, INVALID_ARGUMENT, OK,
};

/// Runs one roll call and returns its owned result.
unsafe fn roll(operation: impl FnOnce(*mut Array) -> i32) -> Array {
    let mut output = std::ptr::null_mut();
    assert_eq!(operation(&mut output), OK);
    assert!(!output.is_null());
    output
}

/// Returns C-order logical values after applying one axis permutation.
fn permuted_values(values: &[i64], source_shape: &[i64], order: &[usize]) -> Vec<i64> {
    let output_shape = order
        .iter()
        .map(|axis| source_shape[*axis] as usize)
        .collect::<Vec<_>>();
    (0..values.len())
        .map(|mut output_index| {
            let mut source_coordinates = vec![0_usize; source_shape.len()];
            for output_axis in (0..output_shape.len()).rev() {
                let dimension = output_shape[output_axis];
                source_coordinates[order[output_axis]] = output_index % dimension;
                output_index /= dimension;
            }
            let source_index = source_coordinates
                .iter()
                .zip(source_shape)
                .fold(0_usize, |flat, (coordinate, dimension)| {
                    flat * *dimension as usize + coordinate
                });
            values[source_index]
        })
        .collect()
}

/// Normalizes and applies the permutation represented by NumPy `rollaxis` arguments.
fn rollaxis_order(axis: i64, start: i64, rank: usize) -> Vec<usize> {
    let rank = rank as i64;
    let axis = if axis < 0 { axis + rank } else { axis };
    let mut start = if start < 0 { start + rank } else { start };
    if axis < start {
        start -= 1;
    }
    let mut order = (0..rank as usize).collect::<Vec<_>>();
    let moved = order.remove(axis as usize);
    order.insert(start as usize, moved);
    order
}

/// Executes one rollaxis call and validates its shape and logical permutation.
unsafe fn assert_rollaxis(
    source: Array,
    values: &[i64],
    source_shape: &[i64],
    axis: i64,
    start: i64,
    expected_shape: &[i64],
) {
    let mut output = std::ptr::null_mut();
    assert_eq!(
        unsafe { crate::ffi::terlan_ndarray_v1_rollaxis(source, axis, start, &mut output) },
        OK
    );
    assert_eq!(unsafe { shape(output) }.unwrap(), expected_shape);
    let order = rollaxis_order(axis, start, source_shape.len());
    assert_eq!(
        unsafe { ints(output) }.unwrap(),
        permuted_values(values, source_shape, &order),
        "rollaxis axis={axis} start={start} order={order:?}"
    );
    unsafe { delete_arrays(&[output]) };
}

#[test]
fn numeric_rollaxis_exhaustively_matches_numpy_shapes_values_and_negative_axes() {
    const TARGETS: [((i64, i64), [i64; 4]); 20] = [
        ((0, 0), [1, 2, 3, 4]),
        ((0, 1), [1, 2, 3, 4]),
        ((0, 2), [2, 1, 3, 4]),
        ((0, 3), [2, 3, 1, 4]),
        ((0, 4), [2, 3, 4, 1]),
        ((1, 0), [2, 1, 3, 4]),
        ((1, 1), [1, 2, 3, 4]),
        ((1, 2), [1, 2, 3, 4]),
        ((1, 3), [1, 3, 2, 4]),
        ((1, 4), [1, 3, 4, 2]),
        ((2, 0), [3, 1, 2, 4]),
        ((2, 1), [1, 3, 2, 4]),
        ((2, 2), [1, 2, 3, 4]),
        ((2, 3), [1, 2, 3, 4]),
        ((2, 4), [1, 2, 4, 3]),
        ((3, 0), [4, 1, 2, 3]),
        ((3, 1), [1, 4, 2, 3]),
        ((3, 2), [1, 2, 4, 3]),
        ((3, 3), [1, 2, 3, 4]),
        ((3, 4), [1, 2, 3, 4]),
    ];
    let expected_shape = |axis: i64, start: i64| {
        TARGETS
            .iter()
            .find(|((candidate_axis, candidate_start), _)| {
                *candidate_axis == axis && *candidate_start == start
            })
            .unwrap()
            .1
    };
    unsafe {
        let values = (0..24).collect::<Vec<_>>();
        let source_shape = [1_i64, 2_i64, 3_i64, 4_i64];
        let source = owned_int(&values, &source_shape).unwrap();
        for ((axis, start), target) in TARGETS {
            assert_rollaxis(source, &values, &source_shape, axis, start, &target);
            assert_rollaxis(
                source,
                &values,
                &source_shape,
                -(axis + 1),
                start,
                &expected_shape(3 - axis, start),
            );
            let negative_start = -(if start < 4 { start + 1 } else { start });
            let expected_start = 4 - if start < 4 { start + 1 } else { start };
            assert_rollaxis(
                source,
                &values,
                &source_shape,
                axis,
                negative_start,
                &expected_shape(axis, expected_start),
            );
            assert_rollaxis(
                source,
                &values,
                &source_shape,
                -(axis + 1),
                negative_start,
                &expected_shape(3 - axis, expected_start),
            );
        }
        delete_arrays(&[source]);
    }
}

#[test]
fn numeric_moveaxis_matches_single_multiple_empty_and_order_preserving_numpy_rules() {
    const CASES: [(&[i64], &[i64], &[i64]); 12] = [
        (&[0], &[-1], &[6, 7, 5]),
        (&[1], &[-1], &[5, 7, 6]),
        (&[2], &[-1], &[5, 6, 7]),
        (&[-1], &[-1], &[5, 6, 7]),
        (&[0], &[1], &[2, 1, 3, 4]),
        (&[1], &[2], &[1, 3, 2, 4]),
        (&[1], &[-1], &[1, 3, 4, 2]),
        (&[0, 1], &[2, 3], &[2, 3, 0, 1]),
        (&[2, 3], &[0, 1], &[2, 3, 0, 1]),
        (&[0, 1, 2], &[2, 3, 0], &[2, 3, 0, 1]),
        (&[3, 0], &[1, 0], &[0, 3, 1, 2]),
        (&[0, 3], &[0, 1], &[0, 3, 1, 2]),
    ];
    unsafe {
        let cube = owned_int(&(0..210).collect::<Vec<_>>(), &[5, 6, 7]).unwrap();
        for (sources, destinations, expected) in &CASES[..4] {
            let mut output = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_moveaxis(
                    cube,
                    sources.as_ptr(),
                    sources.len() as i64,
                    destinations.as_ptr(),
                    destinations.len() as i64,
                    &mut output,
                ),
                OK
            );
            assert_eq!(shape(output).unwrap(), *expected);
            delete_arrays(&[output]);
        }

        let hypercube = owned_int(&(0..24).collect::<Vec<_>>(), &[1, 2, 3, 4]).unwrap();
        for (sources, destinations, expected) in &CASES[4..7] {
            let mut output = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_moveaxis(
                    hypercube,
                    sources.as_ptr(),
                    sources.len() as i64,
                    destinations.as_ptr(),
                    destinations.len() as i64,
                    &mut output,
                ),
                OK
            );
            assert_eq!(shape(output).unwrap(), *expected);
            delete_arrays(&[output]);
        }

        let empty = owned_int(&[], &[0, 1, 2, 3]).unwrap();
        for (sources, destinations, expected) in &CASES[7..] {
            let mut output = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_moveaxis(
                    empty,
                    sources.as_ptr(),
                    sources.len() as i64,
                    destinations.as_ptr(),
                    destinations.len() as i64,
                    &mut output,
                ),
                OK
            );
            assert_eq!(shape(output).unwrap(), *expected);
            assert!(ints(output).unwrap().is_empty());
            delete_arrays(&[output]);
        }
        delete_arrays(&[empty, hypercube, cube]);
    }
}

#[test]
fn numeric_axis_movement_rejects_invalid_axes_counts_and_pointers_atomically() {
    unsafe {
        let source = owned_int(&(0..24).collect::<Vec<_>>(), &[1, 2, 3, 4]).unwrap();
        for (axis, start) in [(-5, 0), (4, 0), (0, -5), (0, 5)] {
            let mut output = 1_usize as Array;
            assert_ne!(
                crate::ffi::terlan_ndarray_v1_rollaxis(source, axis, start, &mut output),
                OK
            );
            assert!(output.is_null());
        }
        for (sources, destinations) in [
            (&[0_i64, 0_i64][..], &[0_i64, 1_i64][..]),
            (&[0_i64, 1_i64][..], &[1_i64, 1_i64][..]),
        ] {
            let mut output = 1_usize as Array;
            assert_ne!(
                crate::ffi::terlan_ndarray_v1_moveaxis(
                    source,
                    sources.as_ptr(),
                    2,
                    destinations.as_ptr(),
                    2,
                    &mut output,
                ),
                OK
            );
            assert!(output.is_null());
        }
        let mut output = 1_usize as Array;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_moveaxis(
                source,
                [0_i64].as_ptr(),
                1,
                [0_i64, 1_i64].as_ptr(),
                2,
                &mut output,
            ),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_moveaxis(
                source,
                std::ptr::null(),
                1,
                [0_i64].as_ptr(),
                1,
                &mut output,
            ),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        delete_arrays(&[source]);
    }
}

#[test]
fn numeric_indices_match_dense_sparse_scalar_empty_and_dtype_numpy_rules() {
    unsafe {
        let grid_shape = [2_i64, 3_i64];
        let expected = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        for requested_dtype in 2..=14 {
            let mut dense = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_indices(
                    grid_shape.as_ptr(),
                    2,
                    requested_dtype,
                    &mut dense,
                ),
                OK
            );
            assert_eq!(dtype(dense).unwrap(), requested_dtype);
            assert_eq!(shape(dense).unwrap(), [2, 2, 3]);
            let values = numeric(dense, false).unwrap().values;
            assert_eq!(
                values.iter().map(|value| value.re).collect::<Vec<_>>(),
                expected
            );
            assert!(values.iter().all(|value| value.im == 0.0));

            let mut raw = std::ptr::null_mut();
            let mut length = 0;
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_indices_sparse(
                    grid_shape.as_ptr(),
                    2,
                    requested_dtype,
                    &mut raw,
                    &mut length,
                ),
                OK
            );
            let sparse = take_arrays(raw, length);
            assert_eq!(sparse.len(), 2);
            assert_eq!(shape(sparse[0]).unwrap(), [2, 1]);
            assert_eq!(shape(sparse[1]).unwrap(), [1, 3]);
            assert_eq!(dtype(sparse[0]).unwrap(), requested_dtype);
            assert_eq!(dtype(sparse[1]).unwrap(), requested_dtype);
            assert_eq!(
                numeric(sparse[0], false)
                    .unwrap()
                    .values
                    .iter()
                    .map(|value| value.re)
                    .collect::<Vec<_>>(),
                [0.0, 1.0]
            );
            assert_eq!(
                numeric(sparse[1], false)
                    .unwrap()
                    .values
                    .iter()
                    .map(|value| value.re)
                    .collect::<Vec<_>>(),
                [0.0, 1.0, 2.0]
            );
            delete_arrays(&sparse);
            delete_arrays(&[dense]);
        }

        let mut dense = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_indices(std::ptr::null(), 0, 11, &mut dense),
            OK
        );
        assert_eq!(shape(dense).unwrap(), [0]);
        assert!(numeric(dense, false).unwrap().values.is_empty());
        delete_arrays(&[dense]);

        let mut raw = 1_usize as *mut u64;
        let mut length = usize::MAX;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_indices_sparse(
                std::ptr::null(),
                0,
                3,
                &mut raw,
                &mut length,
            ),
            OK
        );
        assert!(raw.is_null());
        assert_eq!(length, 0);

        let empty_shape = [0_i64];
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_indices_sparse(
                empty_shape.as_ptr(),
                1,
                6,
                &mut raw,
                &mut length,
            ),
            OK
        );
        let empty = take_arrays(raw, length);
        assert_eq!(empty.len(), 1);
        assert_eq!(shape(empty[0]).unwrap(), [0]);
        assert!(numeric(empty[0], false).unwrap().values.is_empty());
        delete_arrays(&empty);
    }
}

#[test]
fn numeric_indices_reject_invalid_contracts_without_partial_outputs() {
    unsafe {
        let shape = [2_i64, 3_i64];
        for requested_dtype in [1, 0, 15, i32::MIN, i32::MAX] {
            let mut dense = 1_usize as Array;
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_indices(
                    shape.as_ptr(),
                    2,
                    requested_dtype,
                    &mut dense,
                ),
                INVALID_ARGUMENT
            );
            assert!(dense.is_null());

            let mut raw = 1_usize as *mut u64;
            let mut length = usize::MAX;
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_indices_sparse(
                    shape.as_ptr(),
                    2,
                    requested_dtype,
                    &mut raw,
                    &mut length,
                ),
                INVALID_ARGUMENT
            );
            assert!(raw.is_null());
            assert_eq!(length, 0);
        }

        let invalid_shapes = [[-1_i64, 2_i64], [2_i64, -1_i64]];
        for invalid_shape in invalid_shapes {
            let mut raw = 1_usize as *mut u64;
            let mut length = usize::MAX;
            assert_ne!(
                crate::ffi::terlan_ndarray_v1_indices_sparse(
                    invalid_shape.as_ptr(),
                    2,
                    3,
                    &mut raw,
                    &mut length,
                ),
                OK
            );
            assert!(raw.is_null());
            assert_eq!(length, 0);
        }

        let mut raw = 1_usize as *mut u64;
        let mut length = usize::MAX;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_indices_sparse(
                std::ptr::null(),
                1,
                3,
                &mut raw,
                &mut length,
            ),
            INVALID_ARGUMENT
        );
        assert!(raw.is_null());
        assert_eq!(length, 0);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_indices_sparse(
                shape.as_ptr(),
                -1,
                3,
                &mut raw,
                &mut length,
            ),
            INVALID_ARGUMENT
        );
        assert!(raw.is_null());
        assert_eq!(length, 0);
    }
}

#[test]
fn numeric_roll_matches_flattened_multi_axis_repeated_and_empty_numpy_rules() {
    unsafe {
        let matrix = owned_int(&(0..10).collect::<Vec<_>>(), &[2, 5]).unwrap();
        let flat = roll(|output| crate::ffi::terlan_ndarray_v1_roll_flat(matrix, 1, output));
        assert_eq!(shape(flat).unwrap(), [2, 5]);
        assert_eq!(ints(flat).unwrap(), [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]);

        let axes = [0_i64, 1_i64];
        let both = roll(|output| {
            crate::ffi::terlan_ndarray_v1_roll_axes_by(
                matrix,
                1,
                axes.as_ptr(),
                axes.len() as i64,
                output,
            )
        });
        assert_eq!(ints(both).unwrap(), [9, 5, 6, 7, 8, 4, 0, 1, 2, 3]);

        let shifts = [1_i64, 0_i64];
        let rows = roll(|output| {
            crate::ffi::terlan_ndarray_v1_roll_axes(
                matrix,
                shifts.as_ptr(),
                shifts.len() as i64,
                axes.as_ptr(),
                axes.len() as i64,
                output,
            )
        });
        assert_eq!(ints(rows).unwrap(), [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]);

        let reverse_shifts = [-1_i64, -1_i64];
        let reversed = roll(|output| {
            crate::ffi::terlan_ndarray_v1_roll_axes(
                matrix,
                reverse_shifts.as_ptr(),
                2,
                axes.as_ptr(),
                2,
                output,
            )
        });
        assert_eq!(ints(reversed).unwrap(), [6, 7, 8, 9, 5, 1, 2, 3, 4, 0]);

        let repeated_axes = [1_i64, 1_i64];
        let repeated = roll(|output| {
            crate::ffi::terlan_ndarray_v1_roll_axes_by(matrix, 1, repeated_axes.as_ptr(), 2, output)
        });
        assert_eq!(ints(repeated).unwrap(), [3, 4, 0, 1, 2, 8, 9, 5, 6, 7]);

        let large = roll(|output| crate::ffi::terlan_ndarray_v1_roll(matrix, 6, 1, output));
        assert_eq!(ints(large).unwrap(), [4, 0, 1, 2, 3, 9, 5, 6, 7, 8]);

        let empty = owned_int(&[], &[0]).unwrap();
        let empty_rolled =
            roll(|output| crate::ffi::terlan_ndarray_v1_roll_flat(empty, i64::MAX, output));
        assert_eq!(shape(empty_rolled).unwrap(), [0]);
        assert!(ints(empty_rolled).unwrap().is_empty());

        let scalar = owned_int(&[7], &[]).unwrap();
        let scalar_rolled =
            roll(|output| crate::ffi::terlan_ndarray_v1_roll_flat(scalar, i64::MIN, output));
        assert_eq!(shape(scalar_rolled).unwrap(), []);
        assert_eq!(ints(scalar_rolled).unwrap(), [7]);

        delete_arrays(&[
            scalar_rolled,
            scalar,
            empty_rolled,
            empty,
            large,
            repeated,
            reversed,
            rows,
            both,
            flat,
            matrix,
        ]);
    }
}

#[test]
fn numeric_roll_rejects_malformed_axis_shift_contracts_atomically() {
    unsafe {
        let matrix = owned_int(&(0..6).collect::<Vec<_>>(), &[2, 3]).unwrap();
        let shifts = [1_i64, 2_i64];
        let axes = [0_i64];
        let mut output = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_roll_axes(
                matrix,
                shifts.as_ptr(),
                2,
                axes.as_ptr(),
                1,
                &mut output,
            ),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_roll_axes(
                matrix,
                std::ptr::null(),
                1,
                axes.as_ptr(),
                1,
                &mut output,
            ),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        assert_ne!(
            crate::ffi::terlan_ndarray_v1_roll_axes_by(matrix, 1, [2_i64].as_ptr(), 1, &mut output,),
            OK
        );
        assert!(output.is_null());
        assert_eq!(ints(matrix).unwrap(), [0, 1, 2, 3, 4, 5]);
        delete_arrays(&[matrix]);
    }
}
