//! Tests for precision-aware NumPy `spacing` behavior.

use super::spacing_values;

/// Asserts equality while admitting matching NaN results.
fn assert_float(actual: f64, expected: f64) {
    assert!(
        actual == expected || (actual.is_nan() && expected.is_nan()),
        "expected {expected:?}, received {actual:?}",
    );
}

#[test]
fn float16_spacing_covers_every_finite_encoding() {
    let values = (0u16..=u16::MAX)
        .map(half::f16::from_bits)
        .map(half::f16::to_f64)
        .collect::<Vec<_>>();
    let (actual, dtype) = spacing_values(&values, 14);
    assert_eq!(dtype, 14);
    for (bits, (&value, &spacing)) in values.iter().zip(&actual).enumerate() {
        let bits = bits as u16;
        let expected = if value.is_nan() || value.is_infinite() {
            f64::NAN
        } else if value == 0.0 {
            half::f16::from_bits(1).to_f64()
        } else if value.is_sign_positive() {
            half::f16::from_bits(bits + 1).to_f64() - value
        } else {
            half::f16::from_bits(bits - 1).to_f64() - value
        };
        assert_float(spacing, expected);
    }
}

#[test]
fn floating_spacing_preserves_precision_and_special_values() {
    let values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let (half, half_dtype) = spacing_values(&values, 14);
    let (single, single_dtype) = spacing_values(&values, 11);
    let (double, double_dtype) = spacing_values(&values, 3);
    assert_eq!((half_dtype, single_dtype, double_dtype), (14, 11, 3));
    assert_eq!(half[0], 2f64.powi(-24));
    assert_eq!(half[1], 2f64.powi(-24));
    assert_eq!(half[2], 2f64.powi(-10));
    assert_eq!(half[3], 2f64.powi(-11));
    assert_eq!(single[0], f32::from_bits(1) as f64);
    assert_eq!(single[2], f32::EPSILON as f64);
    assert_eq!(double[0], f64::from_bits(1));
    assert_eq!(double[2], f64::EPSILON);
    for result in [&half, &single, &double] {
        assert!(result[4].is_nan());
        assert!(result[5].is_nan());
        assert!(result[6].is_nan());
    }
}

#[test]
fn integer_inputs_promote_to_float64_spacing() {
    let (actual, dtype) = spacing_values(&[1.0, -1.0], 9);
    assert_eq!(dtype, 3);
    assert_eq!(actual, vec![f64::EPSILON, f64::EPSILON / 2.0]);
}

#[test]
fn nextafter_distance_equals_spacing_for_positive_finite_values() {
    for value in [1.0_f32, 1.0e-5, 1_000.0] {
        let (spacing, dtype) = spacing_values(&[f64::from(value)], 11);
        assert_eq!(dtype, 11);
        assert_eq!(f64::from(value.next_up() - value), spacing[0]);
    }
    for value in [1.0_f64, 1.0e-5, 1_000.0] {
        let (spacing, dtype) = spacing_values(&[value], 3);
        assert_eq!(dtype, 3);
        assert_eq!(value.next_up() - value, spacing[0]);
    }
}
