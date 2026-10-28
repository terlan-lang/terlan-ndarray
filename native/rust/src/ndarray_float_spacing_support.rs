//! Precision-aware NumPy `spacing` support for real arrays.

use half::f16;

/// Returns NumPy's positive spacing for one binary16 value.
fn spacing_f16(value: f64) -> f64 {
    let value = f16::from_f64(value);
    if value.is_nan() || value.is_infinite() {
        return f64::NAN;
    }
    if value == f16::ZERO {
        return f16::from_bits(1).to_f64();
    }
    let bits = value.to_bits();
    let adjacent = if value.is_sign_positive() {
        f16::from_bits(bits + 1)
    } else {
        f16::from_bits(bits - 1)
    };
    adjacent.to_f64() - value.to_f64()
}

/// Returns NumPy's positive spacing for one binary32 value.
fn spacing_f32(value: f64) -> f64 {
    let value = value as f32;
    if value.is_nan() || value.is_infinite() {
        return f64::NAN;
    }
    if value == 0.0 {
        return f32::from_bits(1) as f64;
    }
    let bits = value.to_bits();
    let adjacent = if value.is_sign_positive() {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    };
    (adjacent - value) as f64
}

/// Returns NumPy's positive spacing for one binary64 value.
fn spacing_f64(value: f64) -> f64 {
    if value.is_nan() || value.is_infinite() {
        return f64::NAN;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    let adjacent = if value.is_sign_positive() {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    };
    adjacent - value
}

/// Computes spacing values and the NumPy-compatible result dtype.
pub(crate) fn spacing_values(values: &[f64], dtype: i32) -> (Vec<f64>, i32) {
    let operation: fn(f64) -> f64 = match dtype {
        14 => spacing_f16,
        11 => spacing_f32,
        _ => spacing_f64,
    };
    let output_dtype = match dtype {
        14 => 14,
        11 => 11,
        _ => 3,
    };
    (
        values.iter().copied().map(operation).collect(),
        output_dtype,
    )
}

#[cfg(test)]
#[path = "ndarray_float_spacing_tests.rs"]
mod tests;
