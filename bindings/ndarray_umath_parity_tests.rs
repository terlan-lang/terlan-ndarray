//! Frozen NumPy umath conformance for admitted homogeneous numerical arrays.

use super::native_support::{
    Array, OK, UNSUPPORTED_DTYPE, delete_arrays, dtype, floats, ints, numeric, owned_bool,
    owned_complex, owned_int, owned_int_as, owned_real, shape,
};
use rustfft::num_complex::Complex64;

type UnaryOperation = unsafe extern "C" fn(Array, *mut Array) -> i32;
type BinaryOperation = unsafe extern "C" fn(Array, Array, *mut Array) -> i32;

/// Executes one generated unary ABI operation and returns its owned output.
unsafe fn unary(operation: UnaryOperation, input: Array) -> Array {
    let mut output = std::ptr::null_mut();
    assert_eq!(unsafe { operation(input, &mut output) }, OK);
    assert!(!output.is_null());
    output
}

/// Executes one generated binary ABI operation and returns its owned output.
unsafe fn binary(operation: BinaryOperation, lhs: Array, rhs: Array) -> Array {
    let mut output = std::ptr::null_mut();
    assert_eq!(unsafe { operation(lhs, rhs, &mut output) }, OK);
    assert!(!output.is_null());
    output
}

/// Compares floating values while preserving NaN, infinity, and signed-zero behavior.
fn assert_floats(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if expected.is_nan() {
            assert!(actual.is_nan(), "index {index}: expected NaN, got {actual}");
        } else if expected.is_infinite() {
            assert_eq!(actual, expected, "index {index}");
        } else if expected == 0.0 {
            assert_eq!(actual, expected, "index {index}");
            assert_eq!(
                actual.is_sign_negative(),
                expected.is_sign_negative(),
                "index {index}"
            );
        } else {
            let scale = 1.0_f64.max(expected.abs());
            assert!(
                (actual - expected).abs() <= tolerance * scale,
                "index {index}: expected {expected}, got {actual}"
            );
        }
    }
}

/// Exercises Float32 transcendental loops through non-contiguous logical storage.
#[test]
fn float32_transcendentals_preserve_dtype_and_strided_values() {
    unsafe {
        let source = owned_real(&[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0], &[8], 11).unwrap();
        let mut strided = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(source, 0, 7, -1, -2, &mut strided),
            OK
        );
        let values = floats(strided, false).unwrap();
        let operations: [(UnaryOperation, fn(f64) -> f64); 4] = [
            (crate::ffi::terlan_ndarray_v1_exp, f64::exp),
            (crate::ffi::terlan_ndarray_v1_log, f64::ln),
            (crate::ffi::terlan_ndarray_v1_sin, f64::sin),
            (crate::ffi::terlan_ndarray_v1_cos, f64::cos),
        ];
        let mut outputs = Vec::new();
        for (operation, reference) in operations {
            let output = unary(operation, strided);
            assert_eq!(dtype(output).unwrap(), 11);
            let expected = values.iter().copied().map(reference).collect::<Vec<_>>();
            assert_floats(&floats(output, false).unwrap(), &expected, 2.0e-6);
            outputs.push(output);
        }
        delete_arrays(&outputs);
        delete_arrays(&[strided, source]);
    }
}

/// Covers scalar transcendental values, boundaries, and precision-sensitive identities.
#[test]
fn scalar_transcendental_families_match_numpy_special_values() {
    unsafe {
        let cubes = owned_real(
            &[
                -15.625,
                1.0,
                8.0,
                -27.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
            ],
            &[7],
            3,
        )
        .unwrap();
        let roots = unary(crate::ffi::terlan_ndarray_v1_cbrt, cubes);
        assert_floats(
            &floats(roots, false).unwrap(),
            &[
                -2.5,
                1.0,
                2.0,
                -3.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
            ],
            1.0e-12,
        );

        let powers = owned_real(&[0.0, 1.0, 2.0, 10.0], &[4], 3).unwrap();
        let exp2 = unary(crate::ffi::terlan_ndarray_v1_exp2, powers);
        assert_floats(
            &floats(exp2, false).unwrap(),
            &[1.0, 2.0, 4.0, 1024.0],
            1.0e-12,
        );
        let log2 = unary(crate::ffi::terlan_ndarray_v1_log2, exp2);
        assert_floats(
            &floats(log2, false).unwrap(),
            &[0.0, 1.0, 2.0, 10.0],
            1.0e-12,
        );

        let angles = owned_real(
            &[std::f64::consts::PI, -0.5 * std::f64::consts::PI],
            &[2],
            3,
        )
        .unwrap();
        let degrees = unary(crate::ffi::terlan_ndarray_v1_degrees, angles);
        assert_floats(&floats(degrees, false).unwrap(), &[180.0, -90.0], 1.0e-12);
        let radians = unary(crate::ffi::terlan_ndarray_v1_radians, degrees);
        assert_floats(
            &floats(radians, false).unwrap(),
            &[std::f64::consts::PI, -0.5 * std::f64::consts::PI],
            1.0e-12,
        );

        let near_zero = owned_real(
            &[0.2, 1.0e-12, 0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY],
            &[6],
            3,
        )
        .unwrap();
        let expm1 = unary(crate::ffi::terlan_ndarray_v1_expm1, near_zero);
        assert_floats(
            &floats(expm1, false).unwrap(),
            &[
                0.2_f64.exp_m1(),
                1.0e-12_f64.exp_m1(),
                0.0,
                -0.0,
                f64::INFINITY,
                -1.0,
            ],
            1.0e-12,
        );
        let log_inputs = owned_real(
            &[0.2, 1.0e-12, -1.0, -2.0, f64::INFINITY, f64::NAN],
            &[6],
            3,
        )
        .unwrap();
        let log1p = unary(crate::ffi::terlan_ndarray_v1_log1p, log_inputs);
        assert_floats(
            &floats(log1p, false).unwrap(),
            &[
                0.2_f64.ln_1p(),
                1.0e-12_f64.ln_1p(),
                f64::NEG_INFINITY,
                f64::NAN,
                f64::INFINITY,
                f64::NAN,
            ],
            1.0e-12,
        );
        delete_arrays(&[
            log1p, log_inputs, expm1, near_zero, radians, degrees, angles, log2, exp2, powers,
            roots, cubes,
        ]);
    }
}

/// Covers every signed-zero and infinity quadrant admitted by `arctan2`.
#[test]
fn arctan2_hypot_and_logaddexp_cover_ieee_quadrants() {
    unsafe {
        let y_values = [
            1.0,
            -1.0,
            0.0,
            -0.0,
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let x_values = [
            1.0,
            1.0,
            -0.0,
            -0.0,
            1.0,
            1.0,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            1.0,
            1.0,
            f64::INFINITY,
            f64::INFINITY,
        ];
        let y = owned_real(&y_values, &[12], 3).unwrap();
        let x = owned_real(&x_values, &[12], 3).unwrap();
        let angles = binary(crate::ffi::terlan_ndarray_v1_arctan2, y, x);
        assert_floats(
            &floats(angles, false).unwrap(),
            &[
                0.25 * std::f64::consts::PI,
                -0.25 * std::f64::consts::PI,
                std::f64::consts::PI,
                -std::f64::consts::PI,
                0.0,
                -0.0,
                std::f64::consts::PI,
                -std::f64::consts::PI,
                0.5 * std::f64::consts::PI,
                -0.5 * std::f64::consts::PI,
                0.25 * std::f64::consts::PI,
                -0.25 * std::f64::consts::PI,
            ],
            1.0e-12,
        );

        let hypot_lhs =
            owned_real(&[3.0, 0.0, f64::NAN, f64::NAN, f64::INFINITY], &[5], 3).unwrap();
        let hypot_rhs = owned_real(&[4.0, 0.0, 1.0, f64::INFINITY, f64::NAN], &[5], 3).unwrap();
        let hypot = binary(crate::ffi::terlan_ndarray_v1_hypot, hypot_lhs, hypot_rhs);
        assert_floats(
            &floats(hypot, false).unwrap(),
            &[5.0, 0.0, f64::NAN, f64::INFINITY, f64::INFINITY],
            1.0e-12,
        );

        let logs_lhs = owned_real(
            &[
                1.0_f64.ln(),
                2.0_f64.ln(),
                1_000_000.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
            ],
            &[6],
            3,
        )
        .unwrap();
        let logs_rhs = owned_real(
            &[
                5.0_f64.ln(),
                4.0_f64.ln(),
                1_000_200.0,
                1.0,
                f64::NEG_INFINITY,
                f64::INFINITY,
            ],
            &[6],
            3,
        )
        .unwrap();
        let natural = binary(crate::ffi::terlan_ndarray_v1_logaddexp, logs_lhs, logs_rhs);
        assert_floats(
            &floats(natural, false).unwrap(),
            &[
                6.0_f64.ln(),
                6.0_f64.ln(),
                1_000_200.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
            ],
            1.0e-12,
        );
        let base_two = binary(crate::ffi::terlan_ndarray_v1_logaddexp2, logs_lhs, logs_rhs);
        assert!(floats(base_two, false).unwrap()[5].is_nan());

        let log2_lhs = owned_real(
            &[
                1.0f64.log2(),
                2.0f64.log2(),
                3.0f64.log2(),
                4.0f64.log2(),
                5.0f64.log2(),
                1_000_000.0,
                -1_000_000.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ],
            &[9],
            3,
        )
        .unwrap();
        let log2_rhs = owned_real(
            &[
                5.0f64.log2(),
                4.0f64.log2(),
                3.0f64.log2(),
                2.0f64.log2(),
                1.0f64.log2(),
                1_000_200.0,
                -1_000_200.0,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ],
            &[9],
            3,
        )
        .unwrap();
        let log2_sum = binary(crate::ffi::terlan_ndarray_v1_logaddexp2, log2_lhs, log2_rhs);
        assert_floats(
            &floats(log2_sum, false).unwrap(),
            &[
                6.0f64.log2(),
                6.0f64.log2(),
                6.0f64.log2(),
                6.0f64.log2(),
                6.0f64.log2(),
                1_000_200.0,
                -1_000_000.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ],
            1.0e-12,
        );
        delete_arrays(&[
            log2_sum, log2_rhs, log2_lhs, base_two, natural, logs_rhs, logs_lhs, hypot, hypot_rhs,
            hypot_lhs, angles, x, y,
        ]);
    }
}

/// Covers power, floor-division, and remainder ordering across signed values.
#[test]
fn power_division_and_remainder_match_numpy_numeric_rules() {
    unsafe {
        let bases = owned_int(&[-3, -2, -1, 0, 1, 2, 3], &[7]).unwrap();
        let exponents = owned_int(&[0, 1, 2, 3, 4, 5, 6], &[7]).unwrap();
        let powers = binary(crate::ffi::terlan_ndarray_v1_power, bases, exponents);
        assert_eq!(ints(powers).unwrap(), [1, -2, 1, 0, 1, 32, 729]);

        let lhs = owned_real(&[71.0, -71.0, 71.0, -71.0, 78.0 * 6.0e-8], &[5], 3).unwrap();
        let rhs = owned_real(&[19.0, 19.0, -19.0, -19.0, 6.0e-8], &[5], 3).unwrap();
        let quotient = binary(crate::ffi::terlan_ndarray_v1_floor_divide, lhs, rhs);
        let remainder = binary(crate::ffi::terlan_ndarray_v1_remainder, lhs, rhs);
        let quotient_values = floats(quotient, false).unwrap();
        let remainder_values = floats(remainder, false).unwrap();
        assert_floats(&quotient_values[..4], &[3.0, -4.0, -4.0, 3.0], 1.0e-12);
        assert_floats(&remainder_values[..4], &[14.0, 5.0, -5.0, -14.0], 1.0e-12);
        let roundoff_lhs = 78.0 * 6.0e-8;
        assert_eq!(
            quotient_values[4] * 6.0e-8 + remainder_values[4],
            roundoff_lhs
        );
        assert!(remainder_values[4] >= 0.0 && remainder_values[4] < 6.0e-8);
        delete_arrays(&[remainder, quotient, rhs, lhs, powers, exponents, bases]);
    }
}

/// Covers floating classification, signs, extrema, and adjacent-value operations.
#[test]
fn classification_extrema_and_adjacent_values_match_numpy() {
    unsafe {
        let source = owned_real(
            &[
                f64::NAN,
                -f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                -1.0,
                1.0,
                -0.0,
                0.0,
            ],
            &[8],
            3,
        )
        .unwrap();
        let nan = unary(crate::ffi::terlan_ndarray_v1_is_nan, source);
        let infinite = unary(crate::ffi::terlan_ndarray_v1_is_infinite, source);
        let finite = unary(crate::ffi::terlan_ndarray_v1_is_finite, source);
        let signbit = unary(crate::ffi::terlan_ndarray_v1_sign_bit, source);
        let sign = unary(crate::ffi::terlan_ndarray_v1_sign, source);
        let positive = unary(crate::ffi::terlan_ndarray_v1_positive, source);
        assert_eq!(ints(nan).unwrap(), [1, 1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(ints(infinite).unwrap(), [0, 0, 1, 1, 0, 0, 0, 0]);
        assert_eq!(ints(finite).unwrap(), [0, 0, 0, 0, 1, 1, 1, 1]);
        assert_eq!(ints(signbit).unwrap(), [0, 1, 0, 1, 1, 0, 1, 0]);
        assert_floats(
            &floats(sign, false).unwrap(),
            &[f64::NAN, f64::NAN, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0],
            0.0,
        );
        assert_floats(
            &floats(positive, false).unwrap(),
            &[
                f64::NAN,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                -1.0,
                1.0,
                -0.0,
                0.0,
            ],
            0.0,
        );

        let lhs = owned_real(&[f64::NAN, 1.0, f64::NAN, -0.0], &[4], 3).unwrap();
        let rhs = owned_real(&[2.0, f64::NAN, f64::NAN, 0.0], &[4], 3).unwrap();
        let minimum = binary(crate::ffi::terlan_ndarray_v1_elementwise_minimum, lhs, rhs);
        let maximum = binary(crate::ffi::terlan_ndarray_v1_elementwise_maximum, lhs, rhs);
        let fmin = binary(crate::ffi::terlan_ndarray_v1_fmin, lhs, rhs);
        let fmax = binary(crate::ffi::terlan_ndarray_v1_fmax, lhs, rhs);
        assert_floats(
            &floats(minimum, false).unwrap(),
            &[f64::NAN, f64::NAN, f64::NAN, 0.0],
            0.0,
        );
        assert_floats(
            &floats(maximum, false).unwrap(),
            &[f64::NAN, f64::NAN, f64::NAN, 0.0],
            0.0,
        );
        assert_floats(
            &floats(fmin, false).unwrap(),
            &[2.0, 1.0, f64::NAN, 0.0],
            0.0,
        );
        assert_floats(
            &floats(fmax, false).unwrap(),
            &[2.0, 1.0, f64::NAN, 0.0],
            0.0,
        );

        let toward = owned_real(&[1.0, -1.0, 1.0], &[3], 3).unwrap();
        let from = owned_real(&[0.0, 0.0, 1.0], &[3], 3).unwrap();
        let signs = owned_real(&[0.0, -0.0, -1.0], &[3], 3).unwrap();
        let adjacent = binary(crate::ffi::terlan_ndarray_v1_nextafter, from, toward);
        let copied_sign = binary(crate::ffi::terlan_ndarray_v1_copysign, toward, signs);
        let adjacent_values = floats(adjacent, false).unwrap();
        assert!(adjacent_values[0] > 0.0 && adjacent_values[1] < 0.0);
        assert_eq!(adjacent_values[2], 1.0);
        assert!(floats(copied_sign, false).unwrap()[1].is_sign_negative());
        delete_arrays(&[
            copied_sign,
            adjacent,
            from,
            toward,
            signs,
            fmax,
            fmin,
            maximum,
            minimum,
            rhs,
            lhs,
            positive,
            sign,
            signbit,
            finite,
            infinite,
            nan,
            source,
        ]);
    }
}

/// Covers Rust-owned numerical completion kernels used by the public NumPy module.
#[test]
fn heaviside_ldexp_and_frexp_match_numpy_boundaries() {
    unsafe {
        let values = owned_real(
            &[
                -30.0,
                -0.1,
                0.0,
                0.2,
                7.5,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ],
            &[2, 4],
            3,
        )
        .unwrap();
        let zero = owned_real(&[0.5], &[], 3).unwrap();
        let mut heaviside = std::ptr::null_mut();
        assert_eq!(
            super::numpy_numeric_extension::terlan_ndarray_v1_heaviside(
                values,
                zero,
                &mut heaviside
            ),
            OK
        );
        assert_floats(
            &floats(heaviside, false).unwrap(),
            &[0.0, 0.0, 0.5, 1.0, 1.0, f64::NAN, 1.0, 0.0],
            0.0,
        );

        let mantissas = owned_real(&[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0], &[7], 11).unwrap();
        let exponents = owned_int(&[3, 2, 1, 0, -1, -2, -3], &[7]).unwrap();
        let mut expanded = std::ptr::null_mut();
        assert_eq!(
            super::numpy_numeric_extension::terlan_ndarray_v1_ldexp(
                mantissas,
                exponents,
                &mut expanded
            ),
            OK
        );
        assert_eq!(shape(expanded).unwrap(), [7]);
        assert_floats(&floats(expanded, false).unwrap(), &[1.0; 7], 0.0);

        let decomposed = owned_real(
            &[
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                0.0,
                -0.0,
                1.0,
                -1.0,
            ],
            &[7],
            3,
        )
        .unwrap();
        let mut outputs = std::ptr::null_mut();
        let mut output_count = 0_usize;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_frexp_all(decomposed, &mut outputs, &mut output_count),
            OK
        );
        assert_eq!(output_count, 2);
        let handles = std::slice::from_raw_parts(outputs, output_count);
        let fractions = handles[0] as usize as Array;
        let powers = handles[1] as usize as Array;
        assert_floats(
            &floats(fractions, false).unwrap(),
            &[
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                0.0,
                -0.0,
                0.5,
                -0.5,
            ],
            0.0,
        );
        assert_eq!(ints(powers).unwrap(), [0, 0, 0, 0, 0, 1, 1]);
        delete_arrays(&[fractions, powers]);
        crate::ffi::terlan_ndarray_v1_delete_handle_values(outputs);
        delete_arrays(&[
            decomposed, expanded, exponents, mantissas, heaviside, zero, values,
        ]);
    }
}

/// Covers complex division and power without reducing through real intermediates.
#[test]
fn complex_power_and_division_preserve_components() {
    unsafe {
        let values = owned_complex(
            &[
                Complex64::new(1.0, 2.0),
                Complex64::new(2.0, 3.0),
                Complex64::new(3.0, 4.0),
            ],
            &[3],
            13,
        )
        .unwrap();
        let two = owned_complex(&[Complex64::new(2.0, 0.0)], &[], 13).unwrap();
        let squared = binary(crate::ffi::terlan_ndarray_v1_power, values, two);
        let quotient = binary(crate::ffi::terlan_ndarray_v1_true_divide, squared, values);
        let actual = numeric(quotient, false).unwrap().values;
        let expected = numeric(values, false).unwrap().values;
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((*actual - expected).norm() <= 1.0e-12);
        }
        delete_arrays(&[quotient, squared, two, values]);
    }
}

/// Covers IEEE special values across the admitted real unary loops.
#[test]
fn special_float_unary_operations_match_numpy_classification_and_signs() {
    unsafe {
        for dtype_code in [14, 11, 3] {
            let input = owned_real(
                &[
                    f64::NAN,
                    -f64::NAN,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    0.0,
                    -0.0,
                    -1.0,
                    1.0,
                ],
                &[8],
                dtype_code,
            )
            .unwrap();
            let absolute = unary(crate::ffi::terlan_ndarray_v1_absolute, input);
            let square = unary(crate::ffi::terlan_ndarray_v1_square, input);
            let reciprocal = unary(crate::ffi::terlan_ndarray_v1_reciprocal, input);
            let exponential = unary(crate::ffi::terlan_ndarray_v1_exp, input);
            let logarithm = unary(crate::ffi::terlan_ndarray_v1_log, input);
            let sine = unary(crate::ffi::terlan_ndarray_v1_sin, input);
            let cosine = unary(crate::ffi::terlan_ndarray_v1_cos, input);
            let square_root = unary(crate::ffi::terlan_ndarray_v1_sqrt, input);
            assert_floats(
                &floats(absolute, false).unwrap(),
                &[
                    f64::NAN,
                    f64::NAN,
                    f64::INFINITY,
                    f64::INFINITY,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                ],
                0.0,
            );
            assert_floats(
                &floats(square, false).unwrap(),
                &[
                    f64::NAN,
                    f64::NAN,
                    f64::INFINITY,
                    f64::INFINITY,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                ],
                0.0,
            );
            assert_floats(
                &floats(reciprocal, false).unwrap(),
                &[
                    f64::NAN,
                    f64::NAN,
                    0.0,
                    -0.0,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    -1.0,
                    1.0,
                ],
                0.0,
            );
            assert_floats(
                &floats(exponential, false).unwrap(),
                &[
                    f64::NAN,
                    f64::NAN,
                    f64::INFINITY,
                    0.0,
                    1.0,
                    1.0,
                    (-1.0f64).exp(),
                    1.0f64.exp(),
                ],
                if dtype_code == 14 { 1.0e-3 } else { 2.0e-6 },
            );
            assert_floats(
                &floats(logarithm, false).unwrap(),
                &[
                    f64::NAN,
                    f64::NAN,
                    f64::INFINITY,
                    f64::NAN,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                    f64::NAN,
                    0.0,
                ],
                0.0,
            );
            assert_floats(
                &floats(sine, false).unwrap(),
                &[
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                    0.0,
                    -0.0,
                    (-1.0f64).sin(),
                    1.0f64.sin(),
                ],
                if dtype_code == 14 { 1.0e-3 } else { 2.0e-6 },
            );
            assert_floats(
                &floats(cosine, false).unwrap(),
                &[
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                    1.0,
                    1.0,
                    1.0f64.cos(),
                    1.0f64.cos(),
                ],
                if dtype_code == 14 { 1.0e-3 } else { 2.0e-6 },
            );
            assert_floats(
                &floats(square_root, false).unwrap(),
                &[
                    f64::NAN,
                    f64::NAN,
                    f64::INFINITY,
                    f64::NAN,
                    0.0,
                    -0.0,
                    f64::NAN,
                    1.0,
                ],
                0.0,
            );
            delete_arrays(&[
                square_root,
                cosine,
                sine,
                logarithm,
                exponential,
                reciprocal,
                square,
                absolute,
                input,
            ]);
        }
    }
}

/// Verifies NumPy's wide `float_power` result policy for every admitted input kind.
#[test]
fn float_power_promotes_real_and_complex_inputs_to_wide_results() {
    unsafe {
        let bools = owned_bool(&[1], &[1]).unwrap();
        let int8 = owned_int_as(&[2], &[1], 4).unwrap();
        let float16 = owned_real(&[3.0], &[1], 14).unwrap();
        let float32 = owned_real(&[4.0], &[1], 11).unwrap();
        let float64 = owned_real(&[5.0], &[1], 3).unwrap();
        let complex64 = owned_complex(&[Complex64::new(1.0, 1.0)], &[1], 12).unwrap();
        let complex128 = owned_complex(&[Complex64::new(1.0, -1.0)], &[1], 13).unwrap();
        let exponent = owned_int(&[2], &[]).unwrap();
        let mut outputs = Vec::new();
        for input in [bools, int8, float16, float32, float64] {
            let mut output = std::ptr::null_mut();
            assert_eq!(
                super::numpy_numeric_extension::terlan_ndarray_v1_float_power(
                    input,
                    exponent,
                    &mut output,
                ),
                OK
            );
            assert_eq!(dtype(output).unwrap(), 3);
            outputs.push(output);
        }
        for input in [complex64, complex128] {
            let mut output = std::ptr::null_mut();
            assert_eq!(
                super::numpy_numeric_extension::terlan_ndarray_v1_float_power(
                    input,
                    exponent,
                    &mut output,
                ),
                OK
            );
            assert_eq!(dtype(output).unwrap(), 13);
            outputs.push(output);
        }
        let first_complex = numeric(outputs[5], false).unwrap().values[0];
        let second_complex = numeric(outputs[6], false).unwrap().values[0];
        assert!((first_complex - Complex64::new(0.0, 2.0)).norm() <= 1.0e-12);
        assert!((second_complex - Complex64::new(0.0, -2.0)).norm() <= 1.0e-12);
        delete_arrays(&outputs);
        delete_arrays(&[
            exponent, complex128, complex64, float64, float32, float16, int8, bools,
        ]);
    }
}

/// Covers stable complex `expm1`, NaN comparisons, and finite ordering.
#[test]
fn complex_expm1_comparisons_and_nan_extrema_match_numpy() {
    unsafe {
        let tiny = owned_complex(
            &[Complex64::new(1.0e-12, 0.0), Complex64::new(0.0, 1.0e-12)],
            &[2],
            13,
        )
        .unwrap();
        let expm1 = unary(crate::ffi::terlan_ndarray_v1_expm1, tiny);
        let values = numeric(expm1, false).unwrap().values;
        assert!((values[0] - Complex64::new(1.0e-12, 0.0)).norm() <= 1.0e-24);
        assert!((values[1] - Complex64::new(-5.0e-25, 1.0e-12)).norm() <= 1.0e-24);

        let lhs = owned_complex(
            &[
                Complex64::new(f64::NAN, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, -1.0),
            ],
            &[3],
            13,
        )
        .unwrap();
        let rhs = owned_complex(
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(f64::NAN, 0.0),
                Complex64::new(1.0, 1.0),
            ],
            &[3],
            13,
        )
        .unwrap();
        let less = binary(crate::ffi::terlan_ndarray_v1_less, lhs, rhs);
        let less_equal = binary(crate::ffi::terlan_ndarray_v1_less_equal, lhs, rhs);
        let greater = binary(crate::ffi::terlan_ndarray_v1_greater, lhs, rhs);
        let greater_equal = binary(crate::ffi::terlan_ndarray_v1_greater_equal, lhs, rhs);
        let equal = binary(crate::ffi::terlan_ndarray_v1_equal, lhs, rhs);
        assert_eq!(ints(less).unwrap(), [0, 0, 1]);
        assert_eq!(ints(less_equal).unwrap(), [0, 0, 1]);
        assert_eq!(ints(greater).unwrap(), [0, 0, 0]);
        assert_eq!(ints(greater_equal).unwrap(), [0, 0, 0]);
        assert_eq!(ints(equal).unwrap(), [0, 0, 0]);

        let fmin = binary(crate::ffi::terlan_ndarray_v1_fmin, lhs, rhs);
        let fmax = binary(crate::ffi::terlan_ndarray_v1_fmax, lhs, rhs);
        let minimum = binary(crate::ffi::terlan_ndarray_v1_elementwise_minimum, lhs, rhs);
        let maximum = binary(crate::ffi::terlan_ndarray_v1_elementwise_maximum, lhs, rhs);
        let min_values = numeric(fmin, false).unwrap().values;
        let max_values = numeric(fmax, false).unwrap().values;
        let minimum_values = numeric(minimum, false).unwrap().values;
        let maximum_values = numeric(maximum, false).unwrap().values;
        assert_eq!(min_values[0], Complex64::new(0.0, 0.0));
        assert_eq!(min_values[1], Complex64::new(1.0, 0.0));
        assert_eq!(max_values[0], Complex64::new(0.0, 0.0));
        assert_eq!(max_values[1], Complex64::new(1.0, 0.0));
        for value in [
            minimum_values[0],
            minimum_values[1],
            maximum_values[0],
            maximum_values[1],
        ] {
            assert!(value.re.is_nan() || value.im.is_nan());
        }
        delete_arrays(&[
            maximum,
            minimum,
            fmax,
            fmin,
            equal,
            greater_equal,
            greater,
            less_equal,
            less,
            rhs,
            lhs,
            expm1,
            tiny,
        ]);
    }
}

/// Covers exponent saturation and overflow-safe integer rational functions.
#[test]
fn ldexp_gcd_and_lcm_handle_machine_integer_boundaries() {
    unsafe {
        let values = owned_real(&[2.0, -2.0, 0.0, -0.0], &[4], 3).unwrap();
        let exponents = owned_int(&[i64::MAX, i64::MIN, i64::MAX, i64::MIN], &[4]).unwrap();
        let mut expanded = std::ptr::null_mut();
        assert_eq!(
            super::numpy_numeric_extension::terlan_ndarray_v1_ldexp(
                values,
                exponents,
                &mut expanded,
            ),
            OK
        );
        assert_floats(
            &floats(expanded, false).unwrap(),
            &[f64::INFINITY, -0.0, 0.0, -0.0],
            0.0,
        );

        let big = i64::from(i32::MAX / 11);
        let lcm_lhs = owned_int_as(&[2 * big], &[1], 6).unwrap();
        let lcm_rhs = owned_int_as(&[5 * big], &[1], 6).unwrap();
        let lcm = binary(crate::ffi::terlan_ndarray_v1_lcm, lcm_lhs, lcm_rhs);
        assert_eq!(ints(lcm).unwrap(), [10 * big]);

        let minimum = owned_int(&[i64::MIN, i64::MIN], &[2]).unwrap();
        let quarter = -(i64::MIN / 4);
        let multiples = owned_int(&[quarter * 3, -quarter * 3], &[2]).unwrap();
        let gcd = binary(crate::ffi::terlan_ndarray_v1_gcd, minimum, multiples);
        assert_eq!(ints(gcd).unwrap(), [quarter, quarter]);

        let truth = owned_bool(&[1], &[]).unwrap();
        let mut positive = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_positive(truth, &mut positive),
            UNSUPPORTED_DTYPE
        );
        assert!(positive.is_null());
        delete_arrays(&[
            truth, gcd, multiples, minimum, lcm, lcm_rhs, lcm_lhs, expanded, exponents, values,
        ]);
    }
}
