use super::*;

unsafe fn array(values: &[f64], dimensions: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_real(values, dimensions, 3).unwrap() }
}

unsafe fn values(array: Array) -> Vec<f64> {
    unsafe { super::super::native_support::floats(array, false).unwrap() }
}

unsafe fn complex_values(array: Array) -> Vec<Complex64> {
    unsafe { super::super::native_support::numeric(array, false) }
        .unwrap()
        .values
}

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-8, "{actual} != {expected}");
    }
}

fn close_complex(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).norm() < 1.0e-8,
            "{actual} != {expected}"
        );
    }
}

fn symmetric_sum(values: &[f64], expected_sum: f64) {
    for (left, right) in values.iter().zip(values.iter().rev()) {
        assert!((left - right).abs() < 1.0e-12, "window is not symmetric");
    }
    assert!((values.iter().sum::<f64>() - expected_sum).abs() < 5.0e-5);
}

unsafe fn correlate_close(lhs: Array, rhs: Array, expected: &[f64]) {
    let mut output = std::ptr::null_mut();
    assert_eq!(
        unsafe { terlan_ndarray_v1_correlate(lhs, rhs, 2, &mut output) },
        0
    );
    close(&unsafe { values(output) }, expected);
    unsafe { super::super::native_support::delete_arrays(&[output]) };
}

/// Executes one `allclose` case with NumPy's default tolerances.
unsafe fn assert_allclose_case(
    lhs_values: &[f64],
    lhs_shape: &[i64],
    rhs_values: &[f64],
    rhs_shape: &[i64],
    expected: bool,
) {
    let lhs = unsafe { array(lhs_values, lhs_shape) };
    let rhs = unsafe { array(rhs_values, rhs_shape) };
    let mut actual = !expected;
    assert_eq!(
        unsafe { terlan_ndarray_v1_allclose(lhs, rhs, 1.0e-5, 1.0e-8, 0, &mut actual) },
        0
    );
    assert_eq!(actual, expected);
    unsafe { super::super::native_support::delete_arrays(&[rhs, lhs]) };
}

#[test]
fn comparisons_complex_projections_and_finite_checks_are_exact() {
    unsafe {
        let lhs = array(&[1.0, 2.0], &[2, 1]);
        let rhs = array(&[1.0, 2.0], &[1, 2]);
        let mut close_value = false;
        let mut equal = false;
        assert_eq!(
            terlan_ndarray_v1_allclose(lhs, rhs, 0.0, 0.0, 0, &mut close_value),
            0
        );
        assert!(!close_value);
        assert_eq!(terlan_ndarray_v1_array_equiv(lhs, lhs, &mut equal), 0);
        assert!(equal);

        let complex = super::super::native_support::owned_complex(
            &[Complex64::new(3.0, 4.0), Complex64::new(-2.0, 0.0)],
            &[2],
            13,
        )
        .unwrap();
        let mut angle = std::ptr::null_mut();
        let mut real = std::ptr::null_mut();
        let mut imag = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_angle(complex, 0, &mut angle), 0);
        assert_eq!(terlan_ndarray_v1_real(complex, &mut real), 0);
        assert_eq!(terlan_ndarray_v1_imag(complex, &mut imag), 0);
        close(&values(real), &[3.0, -2.0]);
        close(&values(imag), &[4.0, 0.0]);
        assert!((values(angle)[0] - 4.0f64.atan2(3.0)).abs() < 1.0e-9);
        super::super::native_support::delete_arrays(&[lhs, rhs, complex, angle, real, imag]);
    }
}

#[test]
fn numpy_comparisons_cover_nan_infinity_broadcasting_and_immutability() {
    unsafe {
        let pair = array(&[1.0, 2.0], &[2]);
        let same = array(&[1.0, 2.0], &[2]);
        let longer = array(&[1.0, 2.0, 3.0], &[3]);
        let different = array(&[3.0, 4.0], &[2]);
        let partly_different = array(&[1.0, 3.0], &[2]);
        let one = array(&[1.0], &[1]);
        let vector_ones = array(&[1.0, 1.0], &[2]);
        let ones_column = array(&[1.0, 1.0], &[2, 1]);
        let two = array(&[2.0], &[1]);
        let pair_column = array(&[1.0, 2.0], &[2, 1]);
        let matrix = array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let mut result = false;
        for (rhs, expected) in [
            (same, true),
            (longer, false),
            (different, false),
            (partly_different, false),
        ] {
            assert_eq!(terlan_ndarray_v1_array_equal(pair, rhs, &mut result), 0);
            assert_eq!(result, expected);
        }
        for (lhs, rhs, expected) in [
            (pair, same, true),
            (pair, longer, false),
            (pair, different, false),
            (pair, partly_different, false),
            (vector_ones, one, true),
            (vector_ones, ones_column, true),
            (pair, two, false),
            (pair, pair_column, false),
            (pair, matrix, false),
        ] {
            assert_eq!(terlan_ndarray_v1_array_equiv(lhs, rhs, &mut result), 0);
            assert_eq!(result, expected);
        }

        let nan_same = array(&[1.0, 2.0, f64::NAN], &[3]);
        let nan_moved = array(&[1.0, f64::NAN, 2.0], &[3]);
        let infinity = array(&[1.0, 2.0, f64::INFINITY], &[3]);
        assert_eq!(
            terlan_ndarray_v1_array_equal(nan_same, nan_same, &mut result),
            0
        );
        assert!(!result);
        assert_eq!(
            terlan_ndarray_v1_array_equal_nan(nan_same, nan_same, &mut result),
            0
        );
        assert!(result);
        for rhs in [nan_moved, infinity] {
            assert_eq!(
                terlan_ndarray_v1_array_equal_nan(nan_same, rhs, &mut result),
                0
            );
            assert!(!result);
        }
        let complex_lhs =
            super::super::native_support::owned_complex(&[Complex64::new(f64::NAN, 1.0)], &[1], 13)
                .unwrap();
        let complex_rhs =
            super::super::native_support::owned_complex(&[Complex64::new(1.0, f64::NAN)], &[1], 13)
                .unwrap();
        assert_eq!(
            terlan_ndarray_v1_array_equal_nan(complex_lhs, complex_rhs, &mut result),
            0
        );
        assert!(result);

        let nonfinite_lhs = array(&[f64::INFINITY, 1.0], &[2]);
        let nonfinite_rhs = array(&[f64::INFINITY, f64::INFINITY], &[2]);
        let before_lhs = values(nonfinite_lhs);
        let before_rhs = values(nonfinite_rhs);
        assert_eq!(
            terlan_ndarray_v1_allclose(nonfinite_lhs, nonfinite_rhs, 1e-5, 1e-8, 0, &mut result),
            0
        );
        assert!(!result);
        assert_eq!(values(nonfinite_lhs), before_lhs);
        assert_eq!(values(nonfinite_rhs), before_rhs);
        let minimum = super::super::native_support::owned_int_as(&[i64::MIN], &[1], 2).unwrap();
        assert_eq!(
            terlan_ndarray_v1_allclose(minimum, minimum, 1e-5, 1e-8, 0, &mut result),
            0
        );
        assert!(result);
        assert_eq!(
            terlan_ndarray_v1_allclose(nan_same, nan_same, 1e-5, 1e-8, 1, &mut result),
            0
        );
        assert!(result);

        let close_lhs = array(&[f64::INFINITY, 0.0, f64::NAN], &[3]);
        let close_rhs = array(&[f64::INFINITY, 2e-8, f64::NAN], &[3]);
        let close_lhs_before = values(close_lhs);
        let close_rhs_before = values(close_rhs);
        let mut close = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_isclose(close_lhs, close_rhs, 1e-5, 1e-8, 0, &mut close),
            0
        );
        assert_eq!(
            super::super::native_support::ints(close).unwrap(),
            [1, 0, 0]
        );
        let mut close_nan = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_isclose(close_lhs, close_rhs, 1e-5, 1e-8, 1, &mut close_nan),
            0
        );
        assert_eq!(
            super::super::native_support::ints(close_nan).unwrap(),
            [1, 0, 1]
        );
        assert!(
            values(close_lhs)
                .iter()
                .zip(&close_lhs_before)
                .all(|(actual, expected)| actual == expected
                    || (actual.is_nan() && expected.is_nan()))
        );
        assert!(
            values(close_rhs)
                .iter()
                .zip(&close_rhs_before)
                .all(|(actual, expected)| actual == expected
                    || (actual.is_nan() && expected.is_nan()))
        );
        let varied_lhs = array(&[1e-8, 1.0, 1_000_020.00000001], &[3]);
        let varied_rhs = array(&[0.0, f64::NAN, 1_000_000.0], &[3]);
        let mut varied = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_isclose(varied_lhs, varied_rhs, 1e-5, 1e-8, 0, &mut varied),
            0
        );
        assert_eq!(
            super::super::native_support::ints(varied).unwrap(),
            [1, 0, 0]
        );
        let sequence_lhs = array(&[0.0, 1.0, 2.0], &[3]);
        let sequence_rhs = array(&[0.0, 1.0, 2.1], &[3]);
        let mut sequence = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_isclose(sequence_lhs, sequence_rhs, 1e-5, 1e-8, 0, &mut sequence,),
            0
        );
        assert_eq!(
            super::super::native_support::ints(sequence).unwrap(),
            [1, 1, 0]
        );
        let nan_scalar = array(&[f64::NAN], &[]);
        let nan_vector = array(&[f64::NAN, f64::NAN, f64::NAN], &[3]);
        let mut nan_broadcast = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_isclose(nan_scalar, nan_vector, 1e-5, 1e-8, 0, &mut nan_broadcast,),
            0
        );
        assert_eq!(
            super::super::native_support::ints(nan_broadcast).unwrap(),
            [0, 0, 0]
        );
        let zero = array(&[0.0], &[]);
        let mixed = array(&[1e-8, f64::INFINITY, f64::NEG_INFINITY, f64::NAN], &[4]);
        let mut broadcast = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_isclose(zero, mixed, 1e-5, 1e-8, 0, &mut broadcast),
            0
        );
        assert_eq!(
            super::super::native_support::ints(broadcast).unwrap(),
            [1, 0, 0, 0]
        );
        let positive = array(&[1.0], &[]);
        let mut scalar = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_isclose(positive, positive, 1e-5, 1e-8, 0, &mut scalar),
            0
        );
        assert_eq!(
            super::super::native_support::shape(scalar).unwrap(),
            Vec::<i64>::new()
        );
        assert_eq!(super::super::native_support::ints(scalar).unwrap(), [1]);
        let negative_infinity = array(&[f64::NEG_INFINITY], &[]);
        let positive_infinity = array(&[f64::INFINITY], &[]);
        let mut nonfinite_scalar = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_isclose(
                positive_infinity,
                negative_infinity,
                1e-5,
                1e-8,
                0,
                &mut nonfinite_scalar,
            ),
            0
        );
        assert_eq!(
            super::super::native_support::ints(nonfinite_scalar).unwrap(),
            [0]
        );

        super::super::native_support::delete_arrays(&[
            pair,
            same,
            longer,
            different,
            partly_different,
            one,
            vector_ones,
            ones_column,
            two,
            pair_column,
            matrix,
            nan_same,
            nan_moved,
            infinity,
            complex_lhs,
            complex_rhs,
            nonfinite_lhs,
            nonfinite_rhs,
            minimum,
            close_lhs,
            close_rhs,
            close,
            close_nan,
            varied_lhs,
            varied_rhs,
            varied,
            sequence_lhs,
            sequence_rhs,
            sequence,
            nan_scalar,
            nan_vector,
            nan_broadcast,
            zero,
            mixed,
            broadcast,
            positive,
            scalar,
            negative_infinity,
            positive_infinity,
            nonfinite_scalar,
        ]);
    }
}

#[test]
fn numpy_allclose_parameter_factories_match_tolerance_and_nonfinite_rules() {
    unsafe {
        let rtol = 1.0e-5;
        let atol = 1.0e-8;
        assert_allclose_case(&[1.0, 0.0], &[2], &[1.0, 0.0], &[2], true);
        assert_allclose_case(&[atol], &[1], &[0.0], &[1], true);
        assert_allclose_case(&[1.0], &[1], &[1.0 + rtol + atol], &[1], true);
        assert_allclose_case(&[100.0, 1000.0], &[2], &[100.001, 1000.01], &[2], true);
        assert_allclose_case(
            &[100.0, 1000.0],
            &[2],
            &[100.00100002, 1000.01000002],
            &[2],
            true,
        );

        let sequence = (0..125).map(f64::from).collect::<Vec<_>>();
        let relative = sequence
            .iter()
            .map(|value| value + value * rtol)
            .collect::<Vec<_>>();
        assert_allclose_case(&sequence, &[5, 5, 5], &relative, &[5, 5, 5], true);
        assert_allclose_case(&[f64::INFINITY], &[], &[f64::INFINITY], &[], true);
        assert_allclose_case(&[f64::INFINITY], &[], &[f64::INFINITY], &[1], true);

        for (lhs, rhs) in [
            (&[f64::INFINITY, 0.0][..], &[1.0, f64::INFINITY][..]),
            (&[f64::INFINITY, 0.0][..], &[1.0, 0.0][..]),
            (
                &[f64::INFINITY, f64::INFINITY][..],
                &[1.0, f64::INFINITY][..],
            ),
            (&[f64::INFINITY, f64::INFINITY][..], &[1.0, 0.0][..]),
            (&[f64::NEG_INFINITY, 0.0][..], &[f64::INFINITY, 0.0][..]),
            (&[f64::NAN, 0.0][..], &[f64::NAN, 0.0][..]),
            (&[atol * 2.0][..], &[0.0][..]),
            (&[1.0][..], &[1.0 + rtol + atol * 2.0][..]),
            (&[f64::INFINITY, 1.0][..], &[0.0, f64::INFINITY][..]),
        ] {
            assert_allclose_case(lhs, &[lhs.len() as i64], rhs, &[rhs.len() as i64], false);
        }
        let near_zero_failure = sequence
            .iter()
            .map(|value| value + value * atol + atol * 2.0)
            .collect::<Vec<_>>();
        assert_allclose_case(&sequence, &[5, 5, 5], &near_zero_failure, &[5, 5, 5], false);
    }
}

#[test]
fn windows_convolution_and_binary_numeric_operations_match_references() {
    unsafe {
        let mut empty = std::ptr::null_mut();
        let mut singleton = std::ptr::null_mut();
        let mut bartlett = std::ptr::null_mut();
        let mut blackman = std::ptr::null_mut();
        let mut hamming = std::ptr::null_mut();
        let mut hanning = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_bartlett(-3, &mut empty), 0);
        assert_eq!(super::super::native_support::shape(empty).unwrap(), vec![0]);
        assert_eq!(terlan_ndarray_v1_blackman(1, &mut singleton), 0);
        close(&values(singleton), &[1.0]);
        assert_eq!(terlan_ndarray_v1_bartlett(10, &mut bartlett), 0);
        assert_eq!(terlan_ndarray_v1_blackman(10, &mut blackman), 0);
        assert_eq!(terlan_ndarray_v1_hamming(10, &mut hamming), 0);
        assert_eq!(terlan_ndarray_v1_hanning(10, &mut hanning), 0);
        symmetric_sum(&values(bartlett), 4.4444);
        symmetric_sum(&values(blackman), 3.7800);
        symmetric_sum(&values(hamming), 4.9400);
        symmetric_sum(&values(hanning), 4.5000);

        let lhs = array(&[1.0, 2.0, 3.0], &[3]);
        let rhs = array(&[1.0, 1.0], &[2]);
        let exponent = array(&[1.0], &[1]);
        let mut convolved = std::ptr::null_mut();
        let mut powered = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_convolve(lhs, rhs, 2, &mut convolved), 0);
        assert_eq!(
            terlan_ndarray_v1_float_power(lhs, exponent, &mut powered),
            0
        );
        close(&values(convolved), &[1.0, 3.0, 5.0, 3.0]);
        close(&values(powered), &[1.0, 2.0, 3.0]);
        super::super::native_support::delete_arrays(&[
            empty, singleton, bartlett, blackman, hamming, hanning, lhs, rhs, exponent, convolved,
            powered,
        ]);
    }
}

#[test]
fn convolution_and_correlation_match_numpy_modes_dtypes_and_complex_values() {
    unsafe {
        let lhs = array(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
        let rhs = array(&[-1.0, -2.0, -3.0], &[3]);
        correlate_close(lhs, rhs, &[-3.0, -8.0, -14.0, -20.0, -26.0, -14.0, -5.0]);
        let mut rhs_prefix = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(rhs, 0, i64::MIN, -1, 1, &mut rhs_prefix),
            0
        );
        correlate_close(lhs, rhs_prefix, &[-2.0, -5.0, -8.0, -11.0, -14.0, -5.0]);
        correlate_close(rhs, lhs, &[-5.0, -14.0, -26.0, -20.0, -14.0, -8.0, -3.0]);
        let mut reversed_lhs = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(
                lhs,
                0,
                i64::MIN,
                i64::MIN,
                -1,
                &mut reversed_lhs,
            ),
            0
        );
        correlate_close(
            reversed_lhs,
            rhs,
            &[-15.0, -22.0, -22.0, -16.0, -10.0, -4.0, -1.0],
        );
        correlate_close(
            rhs,
            reversed_lhs,
            &[-1.0, -4.0, -10.0, -16.0, -22.0, -22.0, -15.0],
        );
        let sequence_values = (1..=19).map(f64::from).collect::<Vec<_>>();
        let sequence = array(&sequence_values, &[19]);
        let mut stepped = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(
                sequence,
                0,
                i64::MIN,
                i64::MIN,
                3,
                &mut stepped,
            ),
            0
        );
        correlate_close(
            stepped,
            rhs,
            &[
                -3.0, -14.0, -30.0, -48.0, -66.0, -84.0, -102.0, -54.0, -19.0,
            ],
        );
        close(&values(lhs), &[1.0, 2.0, 3.0, 4.0, 5.0]);
        close(&values(rhs), &[-1.0, -2.0, -3.0]);

        let integer_lhs = super::super::native_support::owned_int_as(&[1, 2, 3], &[3], 4).unwrap();
        let integer_rhs = super::super::native_support::owned_int_as(&[1, 1], &[2], 4).unwrap();
        let mut valid = std::ptr::null_mut();
        let mut same = std::ptr::null_mut();
        let mut full = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_convolve(integer_lhs, integer_rhs, 0, &mut valid),
            0
        );
        assert_eq!(
            terlan_ndarray_v1_convolve(integer_lhs, integer_rhs, 1, &mut same),
            0
        );
        assert_eq!(
            terlan_ndarray_v1_convolve(integer_lhs, integer_rhs, 2, &mut full),
            0
        );
        assert_eq!(super::super::native_support::dtype(full).unwrap(), 4);
        close(&values(valid), &[3.0, 5.0]);
        close(&values(same), &[1.0, 3.0, 5.0]);
        close(&values(full), &[1.0, 3.0, 5.0, 3.0]);
        close(&values(integer_lhs), &[1.0, 2.0, 3.0]);
        close(&values(integer_rhs), &[1.0, 1.0]);

        let complex_lhs = super::super::native_support::owned_complex(
            &[
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.0, -2.0),
                Complex64::new(3.0, 1.0),
            ],
            &[3],
            13,
        )
        .unwrap();
        let complex_rhs = super::super::native_support::owned_complex(
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 1.0),
            ],
            &[4],
            13,
        )
        .unwrap();
        let mut complex_result = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_correlate(complex_lhs, complex_rhs, 2, &mut complex_result),
            0
        );
        close_complex(
            &complex_values(complex_result),
            &[
                Complex64::new(-4.0, 1.0),
                Complex64::new(-5.0, -8.0),
                Complex64::new(11.0, -5.0),
                Complex64::new(8.0, -1.0),
                Complex64::new(6.0, 0.0),
                Complex64::new(3.0, 1.0),
            ],
        );

        let hundred_ones = array(&vec![1.0; 100], &[100]);
        let three_ones = array(&[1.0, 1.0, 1.0], &[3]);
        let mut ones_convolution = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_convolve(hundred_ones, three_ones, 2, &mut ones_convolution),
            0
        );
        let convolved_ones = values(ones_convolution);
        assert!(convolved_ones[2..100].iter().all(|value| *value == 3.0));
        close(&values(hundred_ones), &vec![1.0; 100]);
        close(&values(three_ones), &[1.0, 1.0, 1.0]);

        let empty = array(&[], &[0]);
        let mut rejected = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_convolve(empty, integer_rhs, 2, &mut rejected),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_correlate(integer_rhs, empty, 2, &mut rejected),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_convolve(integer_lhs, integer_rhs, -1, &mut rejected),
            INVALID_ARGUMENT
        );
        assert!(rejected.is_null());
        super::super::native_support::delete_arrays(&[
            lhs,
            rhs,
            rhs_prefix,
            reversed_lhs,
            sequence,
            stepped,
            integer_lhs,
            integer_rhs,
            valid,
            same,
            full,
            complex_lhs,
            complex_rhs,
            complex_result,
            hundred_ones,
            three_ones,
            ones_convolution,
            empty,
        ]);
    }
}

#[test]
fn kaiser_accepts_numpy_beta_domain_and_preserves_singleton_identity() {
    unsafe {
        let mut empty = std::ptr::null_mut();
        let mut negative = std::ptr::null_mut();
        let mut singleton = std::ptr::null_mut();
        let mut two = std::ptr::null_mut();
        let mut five = std::ptr::null_mut();
        let mut five_fractional = std::ptr::null_mut();
        let mut not_a_number = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_kaiser(0, 1.0, &mut empty), 0);
        assert!(values(empty).is_empty());
        assert_eq!(terlan_ndarray_v1_kaiser(3, -1.0, &mut negative), 0);
        close(
            &values(negative),
            &[0.7898483148251121, 1.0, 0.7898483148251121],
        );
        assert_eq!(
            terlan_ndarray_v1_kaiser(1, f64::INFINITY, &mut singleton),
            0
        );
        close(&values(singleton), &[1.0]);
        assert_eq!(terlan_ndarray_v1_kaiser(2, 1.0, &mut two), 0);
        close(&values(two), &[0.78984831, 0.78984831]);
        assert_eq!(terlan_ndarray_v1_kaiser(5, 1.0, &mut five), 0);
        close(
            &values(five),
            &[0.78984831, 0.94503323, 1.0, 0.94503323, 0.78984831],
        );
        assert_eq!(
            terlan_ndarray_v1_kaiser(5, 1.56789, &mut five_fractional),
            0
        );
        close(
            &values(five_fractional),
            &[0.58285404, 0.88409679, 1.0, 0.88409679, 0.58285404],
        );
        assert_eq!(terlan_ndarray_v1_kaiser(2, f64::NAN, &mut not_a_number), 0);
        assert!(values(not_a_number).iter().all(|value| value.is_nan()));
        super::super::native_support::delete_arrays(&[
            empty,
            negative,
            singleton,
            two,
            five,
            five_fractional,
            not_a_number,
        ]);
    }
}

#[test]
fn trim_zeros_preserves_views_dtypes_modes_and_empty_arrays() {
    unsafe {
        let source =
            super::super::native_support::owned_int_as(&[0, 0, 1, 0, 2, 3, 4, 0], &[8], 7).unwrap();
        let mut both = std::ptr::null_mut();
        let mut back = std::ptr::null_mut();
        let mut front = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_trim_zeros(source, 2, &mut both), 0);
        assert_eq!(terlan_ndarray_v1_trim_zeros(source, 1, &mut back), 0);
        assert_eq!(terlan_ndarray_v1_trim_zeros(source, 0, &mut front), 0);
        close(&values(both), &[1.0, 0.0, 2.0, 3.0, 4.0]);
        close(&values(back), &[0.0, 0.0, 1.0, 0.0, 2.0, 3.0, 4.0]);
        close(&values(front), &[1.0, 0.0, 2.0, 3.0, 4.0, 0.0]);
        assert_eq!(super::super::native_support::dtype(both).unwrap(), 7);

        let float_source = array(&[0.0, 0.0, 1.0, 0.0], &[4]);
        let complex_source = super::super::native_support::owned_complex(
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 2.0),
                Complex64::new(0.0, 0.0),
            ],
            &[3],
            13,
        )
        .unwrap();
        let mut float_trimmed = std::ptr::null_mut();
        let mut complex_trimmed = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_trim_zeros(float_source, 2, &mut float_trimmed),
            0
        );
        assert_eq!(
            terlan_ndarray_v1_trim_zeros(complex_source, 2, &mut complex_trimmed),
            0
        );
        close(&values(float_trimmed), &[1.0]);
        close_complex(
            &complex_values(complex_trimmed),
            &[Complex64::new(1.0, 2.0)],
        );
        assert_eq!(
            super::super::native_support::dtype(complex_trimmed).unwrap(),
            13
        );

        let zeros = array(&[0.0, 0.0, 0.0], &[3]);
        let integer_zeros = super::super::native_support::owned_int_as(&[0, 0], &[2], 6).unwrap();
        let complex_zeros =
            super::super::native_support::owned_complex(&[Complex64::new(0.0, 0.0); 2], &[2], 12)
                .unwrap();
        let empty = array(&[], &[0]);
        let mut trimmed_zeros = std::ptr::null_mut();
        let mut trimmed_integer_zeros = std::ptr::null_mut();
        let mut trimmed_complex_zeros = std::ptr::null_mut();
        let mut trimmed_empty = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_trim_zeros(zeros, 2, &mut trimmed_zeros),
            0
        );
        assert_eq!(
            terlan_ndarray_v1_trim_zeros(integer_zeros, 0, &mut trimmed_integer_zeros),
            0
        );
        assert_eq!(
            terlan_ndarray_v1_trim_zeros(complex_zeros, 1, &mut trimmed_complex_zeros),
            0
        );
        assert_eq!(
            terlan_ndarray_v1_trim_zeros(empty, 2, &mut trimmed_empty),
            0
        );
        assert!(values(trimmed_zeros).is_empty());
        assert!(values(trimmed_integer_zeros).is_empty());
        assert!(complex_values(trimmed_complex_zeros).is_empty());
        assert!(values(trimmed_empty).is_empty());
        super::super::native_support::delete_arrays(&[
            both,
            back,
            front,
            source,
            float_trimmed,
            complex_trimmed,
            float_source,
            complex_source,
            trimmed_zeros,
            trimmed_integer_zeros,
            trimmed_complex_zeros,
            trimmed_empty,
            zeros,
            integer_zeros,
            complex_zeros,
            empty,
        ]);
    }
}

#[test]
fn sinc_i0_unique_and_msort_match_numpy_reference_vectors() {
    unsafe {
        let sinc_input_values = (0..100)
            .map(|index| -1.0 + 2.0 * index as f64 / 99.0)
            .collect::<Vec<_>>();
        let sinc_input = array(&sinc_input_values, &[100]);
        let mut sinc_output = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_sinc(sinc_input, &mut sinc_output), 0);
        let sinc_values = values(sinc_output);
        for (left, right) in sinc_values.iter().zip(sinc_values.iter().rev()) {
            assert!((left - right).abs() < 1.0e-7);
        }
        let scalar_zero = array(&[0.0], &[]);
        let mut scalar_sinc = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_sinc(scalar_zero, &mut scalar_sinc), 0);
        close(&values(scalar_sinc), &[1.0]);

        let i0_input = array(&[0.49842636, 0.6969809, 0.22011976, 0.0155549, 10.0], &[5]);
        let mut i0_output = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_i0(i0_input, &mut i0_output), 0);
        close(
            &values(i0_output),
            &[
                1.06307822,
                1.12518299,
                1.01214991,
                1.00006049,
                2815.71662847,
            ],
        );
        let i0_matrix = array(
            &[
                0.827002, 0.99959078, 0.89694769, 0.39298162, 0.37954418, 0.05206293, 0.36465447,
                0.72446427, 0.48164949, 0.50324519,
            ],
            &[5, 2],
        );
        let mut i0_matrix_output = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_i0(i0_matrix, &mut i0_matrix_output), 0);
        close(
            &values(i0_matrix_output),
            &[
                1.17843223, 1.26583466, 1.21147086, 1.03898290, 1.03633899, 1.00067775, 1.03352052,
                1.13557954, 1.05884290, 1.06432317,
            ],
        );
        let negative_i0 = array(&[-0.5], &[1]);
        let positive_i0 = array(&[0.5], &[1]);
        let mut negative_result = std::ptr::null_mut();
        let mut positive_result = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_i0(negative_i0, &mut negative_result), 0);
        assert_eq!(terlan_ndarray_v1_i0(positive_i0, &mut positive_result), 0);
        close(&values(negative_result), &values(positive_result));
        let complex_i0 =
            super::super::native_support::owned_complex(&[Complex64::new(1.0, 2.0)], &[1], 13)
                .unwrap();
        let mut rejected = std::ptr::null_mut();
        assert_ne!(terlan_ndarray_v1_i0(complex_i0, &mut rejected), 0);
        assert!(rejected.is_null());

        let repeated =
            super::super::native_support::owned_int_as(&[4, 3, 2, 1, 1, 2, 3, 4, 0], &[9], 5)
                .unwrap();
        let mut unique = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_unique(repeated, &mut unique), 0);
        close(&values(unique), &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(super::super::native_support::dtype(unique).unwrap(), 5);
        let constant =
            super::super::native_support::owned_int_as(&[1, 1, 1, 1, 1], &[5], 4).unwrap();
        let mut constant_unique = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_unique(constant, &mut constant_unique), 0);
        close(&values(constant_unique), &[1.0]);
        let complex_unique_source = super::super::native_support::owned_complex(
            &[
                Complex64::new(5.0, 6.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(1.0, 10.0),
                Complex64::new(10.0, 0.0),
                Complex64::new(5.0, 6.0),
            ],
            &[5],
            13,
        )
        .unwrap();
        let mut complex_unique = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_unique(complex_unique_source, &mut complex_unique),
            0
        );
        close_complex(
            &complex_values(complex_unique),
            &[
                Complex64::new(1.0, 1.0),
                Complex64::new(1.0, 10.0),
                Complex64::new(5.0, 6.0),
                Complex64::new(10.0, 0.0),
            ],
        );

        let matrix = array(
            &[
                0.44567325, 0.79115165, 0.54900530, 0.36844147, 0.37325583, 0.96098397, 0.64864341,
                0.52929049, 0.39172155,
            ],
            &[3, 3],
        );
        let mut sorted = std::ptr::null_mut();
        assert_eq!(
            super::super::terlan_ndarray_v1_msort(matrix, &mut sorted),
            0
        );
        close(
            &values(sorted),
            &[
                0.36844147, 0.37325583, 0.39172155, 0.44567325, 0.52929049, 0.54900530, 0.64864341,
                0.79115165, 0.96098397,
            ],
        );
        super::super::native_support::delete_arrays(&[
            sinc_input,
            sinc_output,
            scalar_zero,
            scalar_sinc,
            i0_input,
            i0_output,
            i0_matrix,
            i0_matrix_output,
            negative_i0,
            positive_i0,
            negative_result,
            positive_result,
            complex_i0,
            repeated,
            unique,
            constant,
            constant_unique,
            complex_unique_source,
            complex_unique,
            matrix,
            sorted,
        ]);
    }
}

#[test]
fn nan_normalization_sinc_spacing_and_unique_cover_boundary_values() {
    unsafe {
        let source = array(&[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0], &[4]);
        let mut normalized = std::ptr::null_mut();
        let mut sinc = std::ptr::null_mut();
        let mut spacing = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_nan_to_num(source, 7.0, 8.0, -9.0, &mut normalized),
            0
        );
        close(&values(normalized), &[7.0, 8.0, -9.0, 0.0]);
        assert_eq!(terlan_ndarray_v1_sinc(normalized, &mut sinc), 0);
        assert_eq!(values(sinc)[3], 1.0);
        assert_eq!(terlan_ndarray_v1_spacing(normalized, &mut spacing), 0);
        assert!(values(spacing).iter().all(|value| *value > 0.0));

        let repeated = array(&[2.0, 1.0, 2.0, 1.0], &[4]);
        let mut unique = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_unique(repeated, &mut unique), 0);
        close(&values(unique), &[1.0, 2.0]);
        super::super::native_support::delete_arrays(&[
            source, normalized, sinc, spacing, repeated, unique,
        ]);
    }
}

#[test]
fn numerical_completion_rejects_bad_shapes_modes_domains_and_outputs() {
    unsafe {
        let vector = array(&[1.0, 2.0], &[2]);
        let matrix = array(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let mut output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_convolve(matrix, vector, 0, &mut output),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_convolve(vector, vector, 9, &mut output),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_trim_zeros(vector, 9, &mut output),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_allclose(vector, vector, -1.0, 0.0, 0, std::ptr::null_mut()),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        super::super::native_support::delete_arrays(&[vector, matrix]);
    }
}
