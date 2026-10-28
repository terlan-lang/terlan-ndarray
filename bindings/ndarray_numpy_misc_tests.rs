use super::*;

const DTYPE_MISMATCH: i32 = 106;

unsafe fn real_array(values: &[f64], dimensions: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_real(values, dimensions, 3).unwrap() }
}

unsafe fn real_values(array: Array) -> Vec<f64> {
    unsafe { super::super::native_support::floats(array, false).unwrap() }
}

unsafe fn complex_values(array: Array) -> Vec<Complex64> {
    unsafe { super::super::native_support::numeric(array, false) }
        .unwrap()
        .values
}

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-11,
            "{actual} != {expected}"
        );
    }
}

unsafe fn contiguous_flags(array: Array) -> (bool, bool) {
    let mut c = 0;
    let mut f = 0;
    assert_eq!(
        unsafe { crate::ffi::terlan_ndarray_v1_is_contiguous(array, &mut c) },
        0
    );
    assert_eq!(
        unsafe { crate::ffi::terlan_ndarray_v1_is_fortran_contiguous(array, &mut f) },
        0
    );
    (c == 1, f == 1)
}

unsafe fn array_strides(array: Array) -> Vec<i64> {
    let rank = unsafe { super::super::native_support::shape(array).unwrap() }.len();
    let mut values = std::ptr::null_mut();
    assert_eq!(
        unsafe { crate::ffi::terlan_ndarray_v1_strides(array, &mut values) },
        0
    );
    if rank == 0 {
        Vec::new()
    } else {
        assert!(!values.is_null());
        unsafe { std::slice::from_raw_parts(values, rank) }.to_vec()
    }
}

#[test]
fn copy_after_transpose_matches_numpy_for_scalar_vector_and_matrix_inputs() {
    unsafe {
        let scalar = super::super::native_support::owned_int_as(&[2], &[], 2).unwrap();
        let vector = super::super::native_support::owned_int_as(&[3, 2, 7, 0], &[4], 2).unwrap();
        let matrix =
            super::super::native_support::owned_int_as(&[0, 1, 2, 3, 4, 5], &[2, 3], 2).unwrap();
        let mut scalar_transposed = std::ptr::null_mut();
        let mut vector_transposed = std::ptr::null_mut();
        let mut matrix_transposed = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_transpose(scalar, 0, 0, &mut scalar_transposed),
            0
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_transpose(vector, 0, 0, &mut vector_transposed),
            0
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_transpose(matrix, 0, 1, &mut matrix_transposed),
            0
        );
        let mut scalar_copy = std::ptr::null_mut();
        let mut vector_copy = std::ptr::null_mut();
        let mut matrix_copy = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copy(scalar_transposed, &mut scalar_copy),
            0
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copy(vector_transposed, &mut vector_copy),
            0
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copy(matrix_transposed, &mut matrix_copy),
            0
        );
        assert_eq!(
            super::super::native_support::ints(scalar_copy).unwrap(),
            [2]
        );
        assert_eq!(
            super::super::native_support::ints(vector_copy).unwrap(),
            [3, 2, 7, 0]
        );
        assert_eq!(
            super::super::native_support::shape(matrix_copy).unwrap(),
            [3, 2]
        );
        assert_eq!(
            super::super::native_support::ints(matrix_copy).unwrap(),
            [0, 3, 1, 4, 2, 5]
        );
        let mut shares = true;
        for (source, copied) in [
            (scalar, scalar_copy),
            (vector, vector_copy),
            (matrix, matrix_copy),
        ] {
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_shares_memory(source, copied, &mut shares),
                0
            );
            assert!(!shares);
        }
        super::super::native_support::delete_arrays(&[
            scalar,
            vector,
            matrix,
            scalar_transposed,
            vector_transposed,
            matrix_transposed,
            scalar_copy,
            vector_copy,
            matrix_copy,
        ]);
    }
}

#[test]
fn astype_preserves_requested_order_copy_policy_and_casting_rules() {
    unsafe {
        let source =
            super::super::native_support::owned_real(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3], 11)
                .unwrap();
        let mut transposed = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_transpose(source, 0, 1, &mut transposed),
            0
        );
        let mut converted = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_astype_ordered(transposed, 6, 2, 1, 4, &mut converted,),
            0
        );
        assert_eq!(
            super::super::native_support::shape(converted).unwrap(),
            [3, 2]
        );
        assert_eq!(array_strides(converted), [1, 3]);
        assert_eq!(
            super::super::native_support::ints(converted).unwrap(),
            [0, 3, 1, 4, 2, 5]
        );
        assert_eq!(super::super::native_support::dtype(converted).unwrap(), 6);
        assert_eq!(contiguous_flags(converted), (false, true));

        let mut shared = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_astype_ordered(source, 11, 0, 0, 4, &mut shared),
            0
        );
        let mut shares = false;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_shares_memory(source, shared, &mut shares),
            0
        );
        assert!(shares);

        let mut fortran = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_astype_ordered(source, 11, 1, 0, 4, &mut fortran),
            0
        );
        assert_eq!(contiguous_flags(fortran), (false, true));
        assert_eq!(array_strides(fortran), [1, 2]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_shares_memory(source, fortran, &mut shares),
            0
        );
        assert!(!shares);

        let mut complex = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_astype_ordered(source, 12, 2, 1, 2, &mut complex),
            0
        );
        assert_eq!(super::super::native_support::dtype(complex).unwrap(), 12);
        let mut rejected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_astype_ordered(source, 6, 2, 1, 2, &mut rejected),
            DTYPE_MISMATCH
        );
        assert!(rejected.is_null());

        let fractional = real_array(&[1.5, -2.75], &[2]);
        let mut truncated = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_astype(fractional, 6, &mut truncated),
            0
        );
        assert_eq!(
            super::super::native_support::ints(truncated).unwrap(),
            [1, -2]
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_astype_ordered(source, 11, 9, 1, 4, &mut rejected),
            super::super::native_support::INVALID_ARGUMENT
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_astype_ordered(source, 11, 0, 1, 9, &mut rejected),
            super::super::native_support::INVALID_ARGUMENT
        );

        super::super::native_support::delete_arrays(&[
            source, transposed, converted, shared, fortran, complex, fractional, truncated,
        ]);
    }
}

#[test]
fn copy_orders_and_relaxed_contiguity_flags_match_numpy() {
    unsafe {
        let values = (0..24).collect::<Vec<_>>();
        let c_source =
            super::super::native_support::owned_int_as(&values, &[2, 1, 3, 4], 2).unwrap();
        let permute_source =
            super::super::native_support::owned_int_as(&values, &[2, 1, 4, 3], 2).unwrap();
        let mut neither = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_transpose(permute_source, 2, 3, &mut neither),
            0
        );
        assert_eq!(contiguous_flags(c_source), (true, false));
        assert_eq!(contiguous_flags(neither), (false, false));

        let mut c_copy = std::ptr::null_mut();
        let mut f_copy = std::ptr::null_mut();
        let mut k_copy = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copy_ordered(neither, 0, &mut c_copy),
            0
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copy_ordered(neither, 1, &mut f_copy),
            0
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copy_ordered(neither, 2, &mut k_copy),
            0
        );
        assert_eq!(contiguous_flags(c_copy), (true, false));
        assert_eq!(contiguous_flags(f_copy), (false, true));
        assert_eq!(contiguous_flags(k_copy), (false, false));
        let expected = super::super::native_support::ints(neither).unwrap();
        assert_eq!(
            super::super::native_support::ints(c_copy).unwrap(),
            expected
        );
        assert_eq!(
            super::super::native_support::ints(f_copy).unwrap(),
            expected
        );
        assert_eq!(
            super::super::native_support::ints(k_copy).unwrap(),
            expected
        );

        let empty = super::super::native_support::owned_int_as(&[], &[2, 2, 0, 2, 2], 2).unwrap();
        assert_eq!(contiguous_flags(empty), (true, true));
        let singleton = super::super::native_support::owned_int_as(&[1, 2], &[1, 2, 1], 2).unwrap();
        let mut singleton_f = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copy_ordered(singleton, 1, &mut singleton_f),
            0
        );
        assert_eq!(contiguous_flags(singleton_f), (true, true));

        let sliced_base =
            super::super::native_support::owned_int_as(&(0..16).collect::<Vec<_>>(), &[4, 4, 1], 2)
                .unwrap();
        let mut sliced = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(sliced_base, 0, 0, 4, 2, &mut sliced),
            0
        );
        assert_eq!(contiguous_flags(sliced), (false, false));
        let mut sliced_row = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_index_axis(sliced, 0, 0, &mut sliced_row),
            0
        );
        assert_eq!(contiguous_flags(sliced_row), (true, true));
        let mut raveled = std::ptr::null_mut();
        assert_eq!(crate::ffi::terlan_ndarray_v1_ravel(sliced, &mut raveled), 0);
        assert_eq!(contiguous_flags(raveled), (true, true));

        let swapped_base = super::super::native_support::owned_int_as(
            &(0..16).collect::<Vec<_>>(),
            &[2, 2, 1, 2, 2],
            2,
        )
        .unwrap();
        let mut swapped = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_transpose(swapped_base, 3, 4, &mut swapped),
            0
        );
        assert_eq!(contiguous_flags(swapped), (false, false));
        let mut first = std::ptr::null_mut();
        let mut first_two = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_index_axis(swapped, 0, 0, &mut first),
            0
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_index_axis(first, 0, 0, &mut first_two),
            0
        );
        assert_eq!(contiguous_flags(first_two), (false, true));
        let mut empty_slice = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(swapped, 2, 0, 0, 1, &mut empty_slice),
            0
        );
        assert_eq!(contiguous_flags(empty_slice), (true, true));

        let squeeze_source =
            super::super::native_support::owned_int_as(&[1, 2, 3], &[1, 3, 1], 2).unwrap();
        let mut squeezed = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_squeeze(
                squeeze_source,
                [0i64, 2].as_ptr(),
                2,
                &mut squeezed,
            ),
            0
        );
        assert_eq!(contiguous_flags(squeezed), (true, true));
        let mut invalid = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copy_ordered(c_source, 7, &mut invalid),
            super::super::native_support::INVALID_ARGUMENT
        );
        assert!(invalid.is_null());

        super::super::native_support::delete_arrays(&[
            c_source,
            permute_source,
            neither,
            c_copy,
            f_copy,
            k_copy,
            empty,
            singleton,
            singleton_f,
            sliced_base,
            sliced,
            sliced_row,
            raveled,
            swapped_base,
            swapped,
            first,
            first_two,
            empty_slice,
            squeeze_source,
            squeezed,
        ]);
    }
}

#[test]
fn angle_sort_complex_and_real_if_close_match_numpy_dtype_and_value_rules() {
    unsafe {
        let complex_input = [
            Complex64::new(1.0, 3.0),
            Complex64::new(2.0f64.sqrt() / 2.0, 2.0f64.sqrt() / 2.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(-1.0, 3.0),
        ];
        let angles = [
            3.0f64.atan(),
            1.0f64.atan(),
            0.0,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::PI,
            -std::f64::consts::FRAC_PI_2,
            -3.0f64.atan(),
            std::f64::consts::PI - 3.0f64.atan(),
        ];
        let complex =
            super::super::native_support::owned_complex(&complex_input, &[8], 13).unwrap();
        let mut radians = std::ptr::null_mut();
        let mut degrees = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_angle(complex, 0, &mut radians), 0);
        assert_eq!(terlan_ndarray_v1_angle(complex, 1, &mut degrees), 0);
        close(&real_values(radians), &angles);
        close(
            &real_values(degrees),
            &angles
                .iter()
                .map(|value| value * 180.0 / std::f64::consts::PI)
                .collect::<Vec<_>>(),
        );

        for (dtype, expected_dtype) in [(2, 13), (5, 12), (8, 12), (4, 12), (7, 12)] {
            let input =
                super::super::native_support::owned_int_as(&[5, 3, 6, 2, 1], &[5], dtype).unwrap();
            let mut sorted = std::ptr::null_mut();
            assert_eq!(terlan_ndarray_v1_sort_complex(input, &mut sorted), 0);
            assert_eq!(
                super::super::native_support::dtype(sorted).unwrap(),
                expected_dtype
            );
            assert_eq!(
                complex_values(sorted),
                [1.0, 2.0, 3.0, 5.0, 6.0].map(|value| Complex64::new(value, 0.0))
            );
            super::super::native_support::delete_arrays(&[input, sorted]);
        }
        let unsorted = super::super::native_support::owned_complex(
            &[
                Complex64::new(2.0, 3.0),
                Complex64::new(1.0, -2.0),
                Complex64::new(1.0, -3.0),
                Complex64::new(2.0, 1.0),
            ],
            &[4],
            13,
        )
        .unwrap();
        let mut sorted = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_sort_complex(unsorted, &mut sorted), 0);
        assert_eq!(
            complex_values(sorted),
            [
                Complex64::new(1.0, -3.0),
                Complex64::new(1.0, -2.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(2.0, 3.0),
            ]
        );

        let reals = [0.1, 0.25, 0.5, 0.75, 0.9];
        let tiny = reals.map(|value| Complex64::new(value, 1.0e-15));
        let visible = reals.map(|value| Complex64::new(value, 1.0e-7));
        let tiny_array = super::super::native_support::owned_complex(&tiny, &[5], 13).unwrap();
        let visible_array =
            super::super::native_support::owned_complex(&visible, &[5], 13).unwrap();
        let mut default_close = std::ptr::null_mut();
        let mut default_far = std::ptr::null_mut();
        let mut absolute_close = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_real_if_close(tiny_array, 100.0, &mut default_close),
            0
        );
        assert_eq!(
            terlan_ndarray_v1_real_if_close(visible_array, 100.0, &mut default_far),
            0
        );
        assert_eq!(
            terlan_ndarray_v1_real_if_close(visible_array, 1.0e-6, &mut absolute_close),
            0
        );
        assert_eq!(
            super::super::native_support::dtype(default_close).unwrap(),
            3
        );
        assert_eq!(
            super::super::native_support::dtype(default_far).unwrap(),
            13
        );
        assert_eq!(
            super::super::native_support::dtype(absolute_close).unwrap(),
            3
        );
        close(&real_values(default_close), &reals);
        close(&real_values(absolute_close), &reals);

        super::super::native_support::delete_arrays(&[
            complex,
            radians,
            degrees,
            unsorted,
            sorted,
            tiny_array,
            visible_array,
            default_close,
            default_far,
            absolute_close,
        ]);
    }
}

#[test]
fn type_checks_infinity_masks_and_nan_replacement_match_numpy() {
    unsafe {
        let real = real_array(&[-1.0, 0.0, 1.0], &[3]);
        let mixed = super::super::native_support::owned_complex(
            &[
                Complex64::new(0.0, -1.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            &[3],
            13,
        )
        .unwrap();
        let mut complex_mask = std::ptr::null_mut();
        let mut real_mask = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_iscomplex(real, &mut complex_mask), 0);
        assert_eq!(
            super::super::native_support::ints(complex_mask).unwrap(),
            [0, 0, 0]
        );
        super::super::native_support::delete_arrays(&[complex_mask]);
        assert_eq!(terlan_ndarray_v1_iscomplex(mixed, &mut complex_mask), 0);
        assert_eq!(
            super::super::native_support::ints(complex_mask).unwrap(),
            [1, 0, 0]
        );
        assert_eq!(terlan_ndarray_v1_isreal(real, &mut real_mask), 0);
        assert_eq!(
            super::super::native_support::ints(real_mask).unwrap(),
            [1, 1, 1]
        );
        super::super::native_support::delete_arrays(&[real_mask]);
        assert_eq!(terlan_ndarray_v1_isreal(mixed, &mut real_mask), 0);
        assert_eq!(
            super::super::native_support::ints(real_mask).unwrap(),
            [0, 1, 1]
        );
        let mut object_kind = false;
        assert_eq!(terlan_ndarray_v1_iscomplexobj(real, &mut object_kind), 0);
        assert!(!object_kind);
        assert_eq!(terlan_ndarray_v1_iscomplexobj(mixed, &mut object_kind), 0);
        assert!(object_kind);
        assert_eq!(terlan_ndarray_v1_isrealobj(real, &mut object_kind), 0);
        assert!(object_kind);
        assert_eq!(terlan_ndarray_v1_isrealobj(mixed, &mut object_kind), 0);
        assert!(!object_kind);

        let nonfinite = real_array(&[f64::NEG_INFINITY, f64::NAN, f64::INFINITY], &[3]);
        let mut negative = std::ptr::null_mut();
        let mut positive = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_isneginf(nonfinite, &mut negative), 0);
        assert_eq!(terlan_ndarray_v1_isposinf(nonfinite, &mut positive), 0);
        assert_eq!(
            super::super::native_support::ints(negative).unwrap(),
            [1, 0, 0]
        );
        assert_eq!(
            super::super::native_support::ints(positive).unwrap(),
            [0, 0, 1]
        );
        let mut replaced = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_nan_to_num(nonfinite, 10.0, 20.0, 30.0, &mut replaced),
            0
        );
        assert_eq!(real_values(replaced), [30.0, 10.0, 20.0]);
        assert_eq!(super::super::native_support::dtype(replaced).unwrap(), 3);

        let integer = super::super::native_support::owned_int_as(&[1], &[], 2).unwrap();
        let finite_float = real_array(&[1.1], &[]);
        let finite_complex =
            super::super::native_support::owned_complex(&[Complex64::new(1.0, 1.0)], &[], 13)
                .unwrap();
        assert_eq!(
            terlan_ndarray_v1_iscomplexobj(finite_float, &mut object_kind),
            0
        );
        assert!(!object_kind);
        assert_eq!(
            terlan_ndarray_v1_iscomplexobj(finite_complex, &mut object_kind),
            0
        );
        assert!(object_kind);
        for (source, expected_dtype) in [(integer, 2), (finite_float, 3), (finite_complex, 13)] {
            let mut output = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_nan_to_num(source, 10.0, 20.0, 30.0, &mut output),
                0
            );
            assert_eq!(
                super::super::native_support::dtype(output).unwrap(),
                expected_dtype
            );
            assert_eq!(complex_values(output), complex_values(source));
            super::super::native_support::delete_arrays(&[output]);
        }
        let mut one_pass = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_nan_to_num(nonfinite, f64::INFINITY, 999.0, -f64::MAX, &mut one_pass,),
            0
        );
        let one_pass_values = real_values(one_pass);
        assert_eq!(one_pass_values[0], -f64::MAX);
        assert!(one_pass_values[1].is_infinite() && one_pass_values[1].is_sign_positive());
        assert_eq!(one_pass_values[2], 999.0);

        super::super::native_support::delete_arrays(&[
            real,
            mixed,
            complex_mask,
            real_mask,
            nonfinite,
            negative,
            positive,
            replaced,
            integer,
            finite_float,
            finite_complex,
            one_pass,
        ]);
    }
}
