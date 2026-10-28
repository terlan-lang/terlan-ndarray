//! Focused NumPy regression tests for admitted homogeneous numerical behavior.

use rustfft::num_complex::Complex64;

use super::native_support::{
    delete_arrays, dtype, floats, ints, numeric, owned_bool, owned_complex, owned_int, owned_real,
    shape, Array, AXIS_OUT_OF_RANGE, DOMAIN_ERROR, OK, SHAPE_MISMATCH, SIZE_OVERFLOW,
    UNSUPPORTED_DTYPE,
};

/// Returns the C- and Fortran-contiguity flags for one array.
unsafe fn contiguous_flags(array: Array) -> (bool, bool) {
    let mut c_contiguous = 0_i32;
    let mut f_contiguous = 0_i32;
    assert_eq!(
        unsafe { crate::ffi::terlan_ndarray_v1_is_contiguous(array, &mut c_contiguous) },
        OK
    );
    assert_eq!(
        unsafe { crate::ffi::terlan_ndarray_v1_is_fortran_contiguous(array, &mut f_contiguous) },
        OK
    );
    (c_contiguous == 1, f_contiguous == 1)
}

/// Returns one array's element strides.
unsafe fn array_strides(array: Array) -> Vec<i64> {
    let rank = unsafe { shape(array).unwrap() }.len();
    let mut values = std::ptr::null_mut();
    assert_eq!(
        unsafe { crate::ffi::terlan_ndarray_v1_strides(array, &mut values) },
        OK
    );
    if rank == 0 {
        Vec::new()
    } else {
        assert!(!values.is_null());
        unsafe { std::slice::from_raw_parts(values, rank) }.to_vec()
    }
}

/// Deletes an ABI-owned array list and its list allocation.
unsafe fn delete_returned_arrays(values: *mut u64, length: usize) {
    let arrays = unsafe { std::slice::from_raw_parts(values, length) }
        .iter()
        .map(|identity| *identity as usize as Array)
        .collect::<Vec<_>>();
    unsafe {
        delete_arrays(&arrays);
        crate::ffi::terlan_ndarray_v1_delete_handle_values(values);
    }
}

#[test]
fn empty_unique_boolean_nansum_and_repeated_digitize_remain_stable() {
    unsafe {
        let empty = owned_real(&[], &[0], 3).unwrap();
        let mut unique = std::ptr::null_mut();
        assert_eq!(
            super::numpy_numeric_extension::terlan_ndarray_v1_unique(empty, &mut unique),
            OK
        );
        assert_eq!(shape(unique).unwrap(), [0]);
        assert!(floats(unique, false).unwrap().is_empty());

        let booleans = owned_bool(&[0, 0], &[2]).unwrap();
        let mut total = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_nan_sum(booleans, std::ptr::null(), 0, 0, &mut total,),
            OK
        );
        assert_eq!(dtype(total).unwrap(), 2);
        assert_eq!(ints(total).unwrap(), [0]);

        let probes = owned_int(&[0, 1, 2, 3, 4], &[5]).unwrap();
        let bins = owned_int(&[1, 3], &[2]).unwrap();
        for _ in 0..100 {
            let mut positions = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_digitize(probes, bins, 0, &mut positions),
                OK
            );
            assert_eq!(ints(positions).unwrap(), [0, 1, 1, 2, 2]);
            delete_arrays(&[positions]);
        }
        delete_arrays(&[bins, probes, total, booleans, unique, empty]);
    }
}

#[test]
fn empty_multiplication_and_polynomial_division_preserve_numpy_contracts() {
    unsafe {
        let empty = owned_real(&[], &[0], 3).unwrap();
        let one = owned_real(&[1.0], &[1], 3).unwrap();
        let mut product = std::ptr::null_mut();
        assert_eq!(
            super::polynomial_extension::terlan_ndarray_v1_polymul(empty, one, &mut product),
            OK
        );
        assert_eq!(shape(product).unwrap(), [0]);
        assert!(floats(product, false).unwrap().is_empty());

        let dividend = owned_int(&[3, 2, 1], &[3]).unwrap();
        let divisor = owned_int(&[5, 4, 3, 2, 1], &[5]).unwrap();
        let mut values = std::ptr::null_mut();
        let mut length = 0_usize;
        assert_eq!(
            super::polynomial_extension::terlan_ndarray_v1_polydiv_all(
                dividend,
                divisor,
                &mut values,
                &mut length,
            ),
            OK
        );
        assert_eq!(length, 2);
        let outputs = std::slice::from_raw_parts(values, length)
            .iter()
            .map(|identity| *identity as usize as Array)
            .collect::<Vec<_>>();
        assert_eq!(dtype(outputs[0]).unwrap(), 3);
        assert_eq!(dtype(outputs[1]).unwrap(), 3);
        assert_eq!(floats(outputs[0], false).unwrap(), [0.0]);
        assert_eq!(floats(outputs[1], false).unwrap(), [3.0, 2.0, 1.0]);
        delete_returned_arrays(values, length);

        let complex = owned_complex(
            &[
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.5),
            ],
            &[3],
            13,
        )
        .unwrap();
        let mut complex_values = std::ptr::null_mut();
        let mut complex_length = 0_usize;
        assert_eq!(
            super::polynomial_extension::terlan_ndarray_v1_polydiv_all(
                complex,
                complex,
                &mut complex_values,
                &mut complex_length,
            ),
            OK
        );
        let complex_outputs = std::slice::from_raw_parts(complex_values, complex_length);
        assert_eq!(dtype(complex_outputs[0] as usize as Array).unwrap(), 13);
        assert_eq!(dtype(complex_outputs[1] as usize as Array).unwrap(), 13);
        delete_returned_arrays(complex_values, complex_length);
        delete_arrays(&[complex, divisor, dividend, product, one, empty]);
    }
}

#[test]
fn impossible_histogram_and_fancy_index_shapes_fail_before_allocation() {
    unsafe {
        let samples = owned_real(&[1.0; 10], &[1, 10], 3).unwrap();
        let bins = owned_int(&[1024; 10], &[10]).unwrap();
        let mut histogram = std::ptr::null_mut();
        let mut histogram_length = 0_usize;
        assert_eq!(
            super::histogramdd_extension::terlan_ndarray_v1_numpy_histogramdd_bins_array(
                samples,
                bins,
                false,
                &mut histogram,
                &mut histogram_length,
            ),
            SIZE_OVERFLOW
        );
        assert!(histogram.is_null());
        assert_eq!(histogram_length, 0);

        let index = owned_int(&vec![0; 7000], &[7000]).unwrap();
        let peers = [index as usize as u64; 4];
        let mut open_mesh = std::ptr::null_mut();
        let mut open_mesh_length = 0_usize;
        assert_eq!(
            super::numpy_index_extension::terlan_ndarray_v1_ix(
                index,
                peers.as_ptr(),
                peers.len() as i64,
                &mut open_mesh,
                &mut open_mesh_length,
            ),
            OK
        );
        assert_eq!(open_mesh_length, 5);
        let mesh = std::slice::from_raw_parts(open_mesh, open_mesh_length);
        let mut broadcast = std::ptr::null_mut();
        let mut broadcast_length = 0_usize;
        assert_eq!(
            super::numpy_shape_extension::terlan_ndarray_v1_broadcast_arrays(
                mesh[0] as usize as Array,
                mesh[1..].as_ptr(),
                4,
                &mut broadcast,
                &mut broadcast_length,
            ),
            SIZE_OVERFLOW
        );
        assert!(broadcast.is_null());
        assert_eq!(broadcast_length, 0);
        delete_returned_arrays(open_mesh, open_mesh_length);
        delete_arrays(&[index, bins, samples]);
    }
}

/// Preserves integer-index inputs and every dimension introduced by those indexes.
#[test]
fn negative_and_singleton_fancy_indexes_are_immutable_and_shape_stable() {
    unsafe {
        let source = owned_int(&(0..125).collect::<Vec<_>>(), &[5, 5, 5]).unwrap();
        let indexes = owned_int(&[-1, 0, 1], &[3]).unwrap();
        let mut selected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_take_indices(source, indexes, 0, &mut selected),
            OK
        );
        assert_eq!(ints(indexes).unwrap(), [-1, 0, 1]);
        assert_eq!(shape(selected).unwrap(), [3, 5, 5]);
        assert_eq!(ints(selected).unwrap()[..3], [100, 101, 102]);

        let vector = owned_int(&[1, 2], &[2]).unwrap();
        let singleton = owned_int(&[0], &[1]).unwrap();
        let mut singleton_result = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_take_indices(vector, singleton, 0, &mut singleton_result,),
            OK
        );
        assert_eq!(shape(singleton_result).unwrap(), [1]);
        assert_eq!(ints(singleton_result).unwrap(), [1]);
        delete_arrays(&[
            singleton_result,
            singleton,
            vector,
            selected,
            indexes,
            source,
        ]);
    }
}

/// Accepts identity permutations at NumPy's historical high-rank regression boundary.
#[test]
fn identity_permutation_accepts_sixteen_dimensions() {
    unsafe {
        let dimensions = vec![2_i64; 16];
        let axes = (0_i64..16).collect::<Vec<_>>();
        let source = owned_int(&vec![1_i64; 1 << 16], &dimensions).unwrap();
        let mut permuted = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_permute(
                source,
                axes.as_ptr(),
                axes.len() as i64,
                &mut permuted,
            ),
            OK
        );
        assert_eq!(shape(permuted).unwrap(), dimensions);
        assert_eq!(ints(permuted).unwrap().len(), 1 << 16);
        delete_arrays(&[permuted, source]);
    }
}

/// Produces identical dot results from a negative-stride view and its owned copy.
#[test]
fn dot_accepts_negative_stride_views_without_materialization() {
    unsafe {
        let row = owned_real(&[1.0, 5.0, 25.0, 125.0, 625.0], &[1, 5], 3).unwrap();
        let column = owned_real(&[20.0, 160.0, 640.0, 1280.0, 1024.0], &[5, 1], 3).unwrap();
        let mut reversed = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_flip(column, [0_i64].as_ptr(), 1, &mut reversed),
            OK
        );
        let mut copied = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copy(reversed, &mut copied),
            OK
        );
        let mut from_view = std::ptr::null_mut();
        let mut from_copy = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_dot(row, reversed, &mut from_view),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_dot(row, copied, &mut from_copy),
            OK
        );
        assert_eq!(shape(from_view).unwrap(), [1, 1]);
        assert_eq!(floats(from_view, false).unwrap(), [55924.0]);
        assert_eq!(
            floats(from_view, false).unwrap(),
            floats(from_copy, false).unwrap()
        );
        delete_arrays(&[from_copy, from_view, copied, reversed, column, row]);
    }
}

/// Keeps copy, empty-slice mutation, and scalar contiguity regressions stable.
#[test]
fn copies_empty_slice_products_and_scalar_flags_match_numpy() {
    unsafe {
        let source = owned_real(&[1.0], &[1], 3).unwrap();
        let mut copied = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_astype(source, 3, &mut copied),
            OK
        );
        let mut shares = true;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_shares_memory(source, copied, &mut shares),
            OK
        );
        assert!(!shares);

        let mut empty = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_basic_slice(
                source,
                [1_i64].as_ptr(),
                1,
                [1_i64].as_ptr(),
                1,
                [1_i64].as_ptr(),
                1,
                -1,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                &mut empty,
            ),
            OK
        );
        let two = owned_real(&[2.0], &[], 3).unwrap();
        let mut product = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_multiply(empty, two, &mut product),
            OK
        );
        assert_eq!(shape(product).unwrap(), [0]);
        assert_eq!(crate::ffi::terlan_ndarray_v1_copyto(empty, product), OK);
        assert_eq!(floats(source, false).unwrap(), [1.0]);

        let scalar64 = owned_real(&[1.0], &[], 3).unwrap();
        let scalar32 = owned_real(&[1.0], &[], 11).unwrap();
        assert_eq!(contiguous_flags(scalar64), (true, true));
        assert_eq!(contiguous_flags(scalar32), (true, true));
        delete_arrays(&[scalar32, scalar64, product, two, empty, copied, source]);
    }
}

/// Preserves traversal order across views, reshape, squeeze, and empty dot products.
#[test]
fn ordered_views_reshape_squeeze_and_empty_dot_match_numpy() {
    unsafe {
        let vector = owned_int(&[1, 2], &[2]).unwrap();
        let mut fortran_view = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_ravel_ordered(vector, 1, &mut fortran_view),
            OK
        );
        let mut shares = false;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_shares_memory(vector, fortran_view, &mut shares),
            OK
        );
        assert!(shares);
        let mut as_fortran = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_astype_ordered(vector, 2, 1, 0, 4, &mut as_fortran,),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_shares_memory(vector, as_fortran, &mut shares),
            OK
        );
        assert!(shares);

        let matrix = owned_int(&(1..=8).collect::<Vec<_>>(), &[4, 2]).unwrap();
        let mut column = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_index_axis(matrix, 1, 1, &mut column),
            OK
        );
        let mut reshaped = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_reshape_ordered(
                column,
                [2_i64, 2].as_ptr(),
                2,
                1,
                &mut reshaped,
            ),
            OK
        );
        assert_eq!(ints(reshaped).unwrap(), [2, 6, 4, 8]);
        assert_eq!(contiguous_flags(reshaped), (false, false));

        let zero_size = owned_real(&[], &[0, 2], 3).unwrap();
        let mut zero_reshaped = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_reshape(
                zero_size,
                [0_i64, 2].as_ptr(),
                2,
                &mut zero_reshaped,
            ),
            OK
        );
        let ten = owned_real(&[1.0; 10], &[10], 3).unwrap();
        let overflowing_shape = [2_i64, 13, 419, 691, 823, 2_977_518_503];
        let mut rejected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_reshape(
                ten,
                overflowing_shape.as_ptr(),
                overflowing_shape.len() as i64,
                &mut rejected,
            ),
            SIZE_OVERFLOW
        );
        assert!(rejected.is_null());

        let empty_lhs = owned_real(&[], &[2, 0], 3).unwrap();
        let empty_rhs = owned_real(&[], &[0, 3], 3).unwrap();
        let mut empty_product = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_dot(empty_lhs, empty_rhs, &mut empty_product),
            OK
        );
        assert_eq!(shape(empty_product).unwrap(), [2, 3]);
        assert_eq!(floats(empty_product, false).unwrap(), [0.0; 6]);

        let cube = owned_real(&[0.0; 8], &[2, 2, 2], 3).unwrap();
        let mut cube_fortran = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copy_ordered(cube, 1, &mut cube_fortran),
            OK
        );
        let mut singleton = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(cube_fortran, 2, 0, 1, 1, &mut singleton),
            OK
        );
        let mut squeezed = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_squeeze(singleton, [2_i64].as_ptr(), 1, &mut squeezed),
            OK
        );
        assert_eq!(contiguous_flags(squeezed), (false, true));
        let mut ordered_zeros = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_zeros_ordered(
                [2_i64, 3].as_ptr(),
                2,
                6,
                1,
                &mut ordered_zeros,
            ),
            OK
        );
        assert_eq!(contiguous_flags(ordered_zeros), (false, true));
        assert_eq!(ints(ordered_zeros).unwrap(), [0; 6]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_zeros_ordered(
                [2_i64, 3].as_ptr(),
                2,
                6,
                9,
                &mut rejected,
            ),
            super::native_support::INVALID_ARGUMENT
        );
        assert!(rejected.is_null());
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_zeros([i64::MAX, 2].as_ptr(), 2, 3, &mut rejected,),
            SIZE_OVERFLOW
        );
        assert!(rejected.is_null());

        let mismatched_matrix = owned_int(&(0..9).collect::<Vec<_>>(), &[3, 3]).unwrap();
        let mismatched_vector = owned_int(&[0, 0, 0], &[3]).unwrap();
        let mismatched = [
            mismatched_matrix as usize as u64,
            mismatched_vector as usize as u64,
        ];
        assert_eq!(
            super::numpy_shape_extension::terlan_ndarray_v1_hstack(
                mismatched.as_ptr(),
                2,
                &mut rejected,
            ),
            SHAPE_MISMATCH
        );
        assert!(rejected.is_null());

        let stride_source = super::native_support::owned_int_as(&[0; 12], &[12], 6).unwrap();
        let mut every_second = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(stride_source, 0, 0, 12, 2, &mut every_second,),
            OK
        );
        let mut trailing_c = std::ptr::null_mut();
        let mut trailing_f = std::ptr::null_mut();
        let trailing_shape = [3_i64, 2, 1, 1];
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_reshape_ordered(
                every_second,
                trailing_shape.as_ptr(),
                4,
                0,
                &mut trailing_c,
            ),
            OK
        );
        assert_eq!(array_strides(trailing_c), [4, 2, 2, 2]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_reshape_ordered(
                every_second,
                trailing_shape.as_ptr(),
                4,
                1,
                &mut trailing_f,
            ),
            OK
        );
        assert_eq!(array_strides(trailing_f), [2, 6, 12, 12]);

        let repeated_source = owned_real(&[1.0], &[1], 3).unwrap();
        let mut repeated = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_as_strided(
                repeated_source,
                [5_i64].as_ptr(),
                [0_i64].as_ptr(),
                1,
                1,
                &mut repeated,
            ),
            OK
        );
        let mut repeated_column = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_reshape(
                repeated,
                [5_i64, 1].as_ptr(),
                2,
                &mut repeated_column,
            ),
            OK
        );
        assert_eq!(array_strides(repeated_column)[0], 0);

        for lhs_dtype in 2..=14 {
            for rhs_dtype in 2..=14 {
                let mut lhs = std::ptr::null_mut();
                let mut rhs = std::ptr::null_mut();
                let mut product = std::ptr::null_mut();
                assert_eq!(
                    crate::ffi::terlan_ndarray_v1_zeros([0_i64].as_ptr(), 1, lhs_dtype, &mut lhs,),
                    OK
                );
                assert_eq!(
                    crate::ffi::terlan_ndarray_v1_zeros([0_i64].as_ptr(), 1, rhs_dtype, &mut rhs,),
                    OK
                );
                assert_eq!(
                    crate::ffi::terlan_ndarray_v1_dot(lhs, rhs, &mut product),
                    OK
                );
                assert_eq!(shape(product).unwrap(), []);
                delete_arrays(&[product, rhs, lhs]);
            }
        }
        delete_arrays(&[
            repeated_column,
            repeated,
            repeated_source,
            trailing_f,
            trailing_c,
            every_second,
            stride_source,
            mismatched_vector,
            mismatched_matrix,
            ordered_zeros,
            squeezed,
            singleton,
            cube_fortran,
            cube,
            empty_product,
            empty_rhs,
            empty_lhs,
            ten,
            zero_reshaped,
            zero_size,
            reshaped,
            column,
            matrix,
            as_fortran,
            fortran_view,
            vector,
        ]);
    }
}

/// Proves that noncommutative ufunc folds preserve axis order and reject invalid domains.
#[test]
fn noncommutative_reduce_and_accumulate_preserve_numpy_order() {
    unsafe {
        let integers = owned_int(&[10, 1, 2, 20, 4, 5], &[2, 3]).unwrap();
        let mut subtract_reduced = std::ptr::null_mut();
        let mut subtract_accumulated = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_subtract_reduce(
                integers,
                1,
                &mut subtract_reduced,
            ),
            OK
        );
        assert_eq!(shape(subtract_reduced).unwrap(), [2]);
        assert_eq!(ints(subtract_reduced).unwrap(), [7, 11]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_subtract_accumulate(
                integers,
                -1,
                &mut subtract_accumulated,
            ),
            OK
        );
        assert_eq!(shape(subtract_accumulated).unwrap(), [2, 3]);
        assert_eq!(
            ints(subtract_accumulated).unwrap(),
            [10, 9, 7, 20, 16, 11]
        );

        let divisors = owned_real(&[16.0, 2.0, 2.0, 81.0, 3.0, 3.0], &[2, 3], 3).unwrap();
        let mut divide_reduced = std::ptr::null_mut();
        let mut divide_accumulated = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_divide_reduce(divisors, 1, &mut divide_reduced),
            OK
        );
        assert_eq!(floats(divide_reduced, false).unwrap(), [4.0, 9.0]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_divide_accumulate(
                divisors,
                1,
                &mut divide_accumulated,
            ),
            OK
        );
        assert_eq!(
            floats(divide_accumulated, false).unwrap(),
            [16.0, 8.0, 4.0, 81.0, 27.0, 9.0]
        );

        let empty = owned_int(&[], &[0]).unwrap();
        let booleans = owned_bool(&[1, 0], &[2]).unwrap();
        let mut rejected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_subtract_reduce(empty, 0, &mut rejected),
            DOMAIN_ERROR
        );
        assert!(rejected.is_null());
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_subtract_accumulate(booleans, 0, &mut rejected),
            UNSUPPORTED_DTYPE
        );
        assert!(rejected.is_null());
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_divide_reduce(divisors, 2, &mut rejected),
            AXIS_OUT_OF_RANGE
        );
        assert!(rejected.is_null());

        delete_arrays(&[
            booleans,
            empty,
            divide_accumulated,
            divide_reduced,
            divisors,
            subtract_accumulated,
            subtract_reduced,
            integers,
        ]);
    }
}

/// Keeps numerical edge cases deterministic across casting, reductions, and sets.
#[test]
fn numerical_cast_reduction_output_and_set_regressions_match_numpy() {
    unsafe {
        let complex = owned_complex(
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.5),
                Complex64::new(0.5, 0.0),
            ],
            &[3],
            13,
        )
        .unwrap();
        let mut booleans = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_astype(complex, 1, &mut booleans),
            OK
        );
        assert_eq!(ints(booleans).unwrap(), [0, 1, 1]);

        let tiny = owned_real(&[2.0_f64.powi(-53)], &[1], 3).unwrap();
        let mut logged = std::ptr::null_mut();
        assert_eq!(crate::ffi::terlan_ndarray_v1_log1p(tiny, &mut logged), OK);
        assert!(floats(logged, false).unwrap()[0].is_finite());

        let real = owned_complex(&[Complex64::new(1.0, 0.0)], &[1], 13).unwrap();
        let complex_nan = owned_complex(&[Complex64::new(0.0, f64::NAN)], &[1], 13).unwrap();
        let mut maximum = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_elementwise_maximum(real, complex_nan, &mut maximum),
            OK
        );
        let maximum_value = numeric(maximum, false).unwrap().values[0];
        assert_eq!(maximum_value.re, 0.0);
        assert!(maximum_value.im.is_nan());

        let sign_source = owned_complex(
            &[
                Complex64::new(f64::NEG_INFINITY, 0.0),
                Complex64::new(-2.0, 1.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(f64::INFINITY, 0.0),
                Complex64::new(f64::NAN, 0.0),
            ],
            &[6],
            13,
        )
        .unwrap();
        let mut signs = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_sign(sign_source, &mut signs),
            OK
        );
        let sign_values = numeric(signs, false).unwrap().values;
        assert_eq!(
            &sign_values[..5],
            &[
                Complex64::new(-1.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
            ]
        );
        assert!(sign_values[5].re.is_nan());

        let values = (0..10_000).map(|value| value as f64).collect::<Vec<_>>();
        let narrow = owned_real(&values, &[10_000], 11).unwrap();
        let mut wide = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_astype(narrow, 3, &mut wide),
            OK
        );
        let mut total = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_sum(wide, std::ptr::null(), 0, 0, &mut total),
            OK
        );
        assert_eq!(floats(total, false).unwrap(), [49_995_000.0]);

        let destination = owned_real(&[0.0], &[], 11).unwrap();
        let lhs = owned_real(&[1.0], &[], 11).unwrap();
        let rhs = owned_real(&[1.0], &[], 3).unwrap();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_add_into(destination, lhs, rhs),
            OK
        );
        assert_eq!(floats(destination, false).unwrap(), [2.0]);

        let empty = owned_int(&[], &[0]).unwrap();
        let mut accumulated = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_cumulative_sum(empty, -1, &mut accumulated),
            OK
        );
        assert_eq!(shape(accumulated).unwrap(), [0]);
        assert!(ints(accumulated).unwrap().is_empty());

        let repeated = owned_int(
            &([0_i64; 5]
                .into_iter()
                .chain([1; 6])
                .chain([2; 6])
                .cycle()
                .take(68)
                .collect::<Vec<_>>()),
            &[68],
        )
        .unwrap();
        let mut outputs = std::ptr::null_mut();
        let mut output_count = 0_usize;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_unique_all(repeated, &mut outputs, &mut output_count),
            OK
        );
        assert_eq!(output_count, 4);
        let handles = std::slice::from_raw_parts(outputs, output_count);
        assert_eq!(ints(handles[0] as usize as Array).unwrap(), [0, 1, 2]);
        assert_eq!(ints(handles[1] as usize as Array).unwrap(), [0, 5, 11]);
        delete_returned_arrays(outputs, output_count);

        for (dtype, minimum) in [
            (4_i32, i8::MIN as i64),
            (5, i16::MIN as i64),
            (6, i32::MIN as i64),
            (2, i64::MIN),
        ] {
            let value = super::native_support::owned_int_as(&[minimum], &[1], dtype).unwrap();
            let negative_one = super::native_support::owned_int_as(&[-1], &[1], dtype).unwrap();
            let mut quotient = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_floor_divide(value, negative_one, &mut quotient),
                OK
            );
            assert_eq!(ints(quotient).unwrap(), [minimum]);
            delete_arrays(&[quotient, negative_one, value]);
        }
        delete_arrays(&[
            repeated,
            accumulated,
            empty,
            rhs,
            lhs,
            destination,
            total,
            wide,
            narrow,
            logged,
            tiny,
            signs,
            sign_source,
            maximum,
            complex_nan,
            real,
            booleans,
            complex,
        ]);
    }
}
