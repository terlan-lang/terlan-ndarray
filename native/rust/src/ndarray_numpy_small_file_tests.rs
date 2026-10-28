//! NumPy small-file parity tests for generic shape and selection behavior.

use rustfft::num_complex::Complex64;

use super::super::native_support::{
    delete_arrays, dtype, ints, numeric, owned_complex, owned_int, owned_int_as, owned_real, shape,
    Array, OK,
};

/// Deletes an ABI-owned array list and every array in that list.
unsafe fn delete_array_list(values: *mut u64, length: usize) {
    let arrays = unsafe { std::slice::from_raw_parts(values, length) }
        .iter()
        .map(|value| *value as usize as Array)
        .collect::<Vec<_>>();
    unsafe {
        delete_arrays(&arrays);
        crate::ffi::terlan_ndarray_v1_delete_handle_values(values);
    }
}

/// Produces row-major values for a tiled source by applying NumPy rank promotion.
fn expected_tile(source: &[i64], source_shape: &[i64], repetitions: &[i64]) -> Vec<i64> {
    let rank = source_shape.len().max(repetitions.len());
    let mut promoted_shape = vec![1_i64; rank - source_shape.len()];
    promoted_shape.extend_from_slice(source_shape);
    let mut promoted_repetitions = vec![1_i64; rank - repetitions.len()];
    promoted_repetitions.extend_from_slice(repetitions);
    let output_shape = promoted_shape
        .iter()
        .zip(&promoted_repetitions)
        .map(|(dimension, repetition)| dimension * repetition)
        .collect::<Vec<_>>();
    let output_count = output_shape.iter().product::<i64>() as usize;
    (0..output_count)
        .map(|logical| {
            let mut remaining = logical as i64;
            let mut source_logical = 0_i64;
            let mut source_stride = 1_i64;
            for axis in (0..rank).rev() {
                let coordinate = remaining % output_shape[axis];
                remaining /= output_shape[axis];
                source_logical += (coordinate % promoted_shape[axis]) * source_stride;
                source_stride *= promoted_shape[axis];
            }
            source[source_logical as usize]
        })
        .collect()
}

/// Exercises shape arithmetic above 32-bit limits without allocating the virtual elements.
#[test]
fn virtual_large_array_split_uses_int64_dimensions() {
    unsafe {
        let scalar = owned_int(&[0], &[]).unwrap();
        let dimensions = [1_i64 << 32, 2];
        let mut virtual_array = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_broadcast_to(
                scalar,
                dimensions.as_ptr(),
                dimensions.len() as i64,
                &mut virtual_array,
            ),
            OK
        );

        let mut values = std::ptr::null_mut();
        let mut length = 0_usize;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_array_split(
                virtual_array,
                4,
                0,
                &mut values,
                &mut length,
            ),
            OK
        );
        assert_eq!(length, 4);
        for identity in std::slice::from_raw_parts(values, length) {
            assert_eq!(
                shape(*identity as usize as Array).unwrap(),
                [1_i64 << 30, 2]
            );
        }

        delete_array_list(values, length);
        delete_arrays(&[virtual_array, scalar]);
    }
}

/// Verifies minimum-rank views for real and complex scalar and nested inputs.
#[test]
fn atleast_rank_views_cover_scalar_complex_and_nested_arrays() {
    unsafe {
        let complex = owned_complex(&[Complex64::new(3.0, 2.0)], &[], 13).unwrap();
        let cube = owned_int(&(0..8).collect::<Vec<_>>(), &[2, 2, 2]).unwrap();
        let mut one = std::ptr::null_mut();
        let mut two = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_atleast_1d(complex, &mut one),
            OK
        );
        assert_eq!(crate::ffi::terlan_ndarray_v1_atleast_2d(cube, &mut two), OK);
        assert_eq!(shape(one).unwrap(), [1]);
        assert_eq!(shape(two).unwrap(), [2, 2, 2]);
        assert_eq!(dtype(one), Ok(13));
        delete_arrays(&[two, one, cube, complex]);
    }
}

/// Exercises gather and replacement along every axis and through a flattened view.
#[test]
fn along_axis_selection_and_replacement_cover_axes_and_flattening() {
    unsafe {
        let source = owned_int(&[10, 30, 20, 60, 40, 50], &[2, 3]).unwrap();
        let axis_zero = owned_int(&[1, 1, 1], &[1, 3]).unwrap();
        let axis_one = owned_int(&[1, 0], &[2, 1]).unwrap();
        let replacement = owned_int(&[-99], &[]).unwrap();
        let mut selected_zero = std::ptr::null_mut();
        let mut selected_one = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_take_along_axis(source, axis_zero, 0, &mut selected_zero,),
            OK
        );
        assert_eq!(ints(selected_zero).unwrap(), [60, 40, 50]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_take_along_axis(source, axis_one, 1, &mut selected_one,),
            OK
        );
        assert_eq!(ints(selected_one).unwrap(), [30, 60]);

        let target = owned_int(&[10, 30, 20, 60, 40, 50], &[2, 3]).unwrap();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_put_along_axis(target, axis_zero, replacement, 0),
            OK
        );
        assert_eq!(ints(target).unwrap(), [10, 30, 20, -99, -99, -99]);

        let flat_shape = [6_i64];
        let flat_index = owned_int(&[3], &[1]).unwrap();
        let mut flat = std::ptr::null_mut();
        let mut selected_flat = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_reshape(source, flat_shape.as_ptr(), 1, &mut flat,),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_take_along_axis(flat, flat_index, 0, &mut selected_flat,),
            OK
        );
        assert_eq!(ints(selected_flat).unwrap(), [60]);

        delete_arrays(&[
            selected_flat,
            flat,
            flat_index,
            target,
            selected_one,
            selected_zero,
            replacement,
            axis_one,
            axis_zero,
            source,
        ]);
    }
}

/// Verifies tiling across promoted ranks beyond the common matrix cases.
#[test]
fn tile_matches_generic_kron_equivalence_shapes() {
    unsafe {
        let source_values = [1_i64, 2, 3, 4];
        let source_shape = [2_i64, 2];
        let source = owned_int(&source_values, &source_shape).unwrap();
        for repetitions in [&[2_i64, 3, 2][..], &[2_i64, 1, 3, 1, 2][..]] {
            let mut tiled = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_tile(
                    source,
                    repetitions.as_ptr(),
                    repetitions.len() as i64,
                    &mut tiled,
                ),
                OK
            );
            assert_eq!(
                ints(tiled).unwrap(),
                expected_tile(&source_values, &source_shape, repetitions)
            );
            delete_arrays(&[tiled]);
        }
        delete_arrays(&[source]);
    }
}

/// Verifies NumPy's default columns, zero-width output, powers, and dtype promotion.
#[test]
fn vandermonde_columns_and_promoted_dtypes_match_numpy() {
    unsafe {
        let coordinates = owned_int(&[0, 1, -2, 3], &[4]).unwrap();
        let powers = [
            [0_i64, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [16, -8, 4, -2, 1],
            [81, 27, 9, 3, 1],
        ];
        for columns in 0_i64..=5 {
            let mut output = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_vander(coordinates, columns, 0, &mut output),
                OK
            );
            assert_eq!(shape(output).unwrap(), [4, columns]);
            let start = 5 - columns as usize;
            let expected = powers
                .iter()
                .flat_map(|row| row[start..].iter().copied())
                .collect::<Vec<_>>();
            assert_eq!(ints(output).unwrap(), expected);
            delete_arrays(&[output]);
        }

        let narrow = owned_int_as(&[11, -12, 13], &[3], 4).unwrap();
        let mut promoted = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_vander(narrow, -1, 0, &mut promoted),
            OK
        );
        assert_eq!(dtype(promoted), Ok(2));
        assert_eq!(
            ints(promoted).unwrap(),
            [121, 11, 1, 144, -12, 1, 169, 13, 1]
        );

        let float = owned_real(&[2.0], &[1], 11).unwrap();
        let mut promoted_float = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_vander(float, 3, 0, &mut promoted_float),
            OK
        );
        assert_eq!(dtype(promoted_float), Ok(3));

        let complex = owned_complex(
            &[Complex64::new(1.0, 1.0), Complex64::new(1.0, -1.0)],
            &[2],
            12,
        )
        .unwrap();
        let mut promoted_complex = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_vander(complex, 3, 0, &mut promoted_complex),
            OK
        );
        assert_eq!(dtype(promoted_complex), Ok(13));
        assert_eq!(
            numeric(promoted_complex, true).unwrap().values,
            [
                Complex64::new(0.0, 2.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, -2.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(1.0, 0.0),
            ]
        );

        delete_arrays(&[
            promoted_complex,
            complex,
            promoted_float,
            float,
            promoted,
            narrow,
            coordinates,
        ]);
    }
}

/// Verifies truncation and signed-infinity masks for arrays and rank-zero values.
#[test]
fn ufunclike_fix_and_signed_infinity_masks_match_numpy() {
    unsafe {
        let values = owned_real(
            &[f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 0.0, 3.0, -3.0],
            &[6],
            3,
        )
        .unwrap();
        let mut positive = std::ptr::null_mut();
        let mut negative = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_isposinf(values, &mut positive),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_isneginf(values, &mut negative),
            OK
        );
        assert_eq!(ints(positive).unwrap(), [1, 0, 0, 0, 0, 0]);
        assert_eq!(ints(negative).unwrap(), [0, 1, 0, 0, 0, 0]);

        let decimals =
            owned_real(&[1.0, 1.1, 1.5, 1.8, -1.0, -1.1, -1.5, -1.8], &[2, 4], 3).unwrap();
        let scalar = owned_real(&[3.14], &[], 3).unwrap();
        let mut fixed = std::ptr::null_mut();
        let mut fixed_scalar = std::ptr::null_mut();
        assert_eq!(crate::ffi::terlan_ndarray_v1_fix(decimals, &mut fixed), OK);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_fix(scalar, &mut fixed_scalar),
            OK
        );
        assert_eq!(ints(fixed).unwrap(), [1, 1, 1, 1, -1, -1, -1, -1]);
        assert_eq!(shape(fixed_scalar).unwrap(), []);
        assert_eq!(ints(fixed_scalar).unwrap(), [3]);

        let complex = owned_complex(&[Complex64::new(f64::INFINITY, 1.0)], &[1], 13).unwrap();
        let mut rejected = std::ptr::null_mut();
        assert_ne!(
            crate::ffi::terlan_ndarray_v1_isposinf(complex, &mut rejected),
            OK
        );
        assert!(rejected.is_null());

        delete_arrays(&[
            complex,
            fixed_scalar,
            fixed,
            scalar,
            decimals,
            negative,
            positive,
            values,
        ]);
    }
}
