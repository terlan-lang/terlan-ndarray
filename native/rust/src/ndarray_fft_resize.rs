//! Axis-local FFT input padding and truncation.

use rustfft::num_complex::Complex64;

/// Returns row-major complex data resized along one axis with zero padding.
pub(super) fn resize_complex_axis(
    data: Vec<Complex64>,
    shape: &mut [i64],
    axis: usize,
    output_length: i64,
) -> Result<Vec<Complex64>, i32> {
    let output_length = usize::try_from(output_length).map_err(|_| super::SIZE_OVERFLOW)?;
    if output_length == 0 {
        return Err(super::DOMAIN_ERROR);
    }
    let input_length = usize::try_from(shape[axis]).map_err(|_| super::SHAPE_MISMATCH)?;
    let outer = shape[..axis].iter().try_fold(1usize, |value, &dimension| {
        value
            .checked_mul(usize::try_from(dimension).map_err(|_| super::SHAPE_MISMATCH)?)
            .ok_or(super::SIZE_OVERFLOW)
    })?;
    let inner = shape[axis + 1..]
        .iter()
        .try_fold(1usize, |value, &dimension| {
            value
                .checked_mul(usize::try_from(dimension).map_err(|_| super::SHAPE_MISMATCH)?)
                .ok_or(super::SIZE_OVERFLOW)
        })?;
    let output_count = outer
        .checked_mul(output_length)
        .and_then(|value| value.checked_mul(inner))
        .ok_or(super::SIZE_OVERFLOW)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_count)
        .map_err(|_| super::SIZE_OVERFLOW)?;
    output.resize(output_count, Complex64::new(0.0, 0.0));
    let copied_length = input_length.min(output_length);
    for outer_index in 0..outer {
        for axis_index in 0..copied_length {
            let source = (outer_index * input_length + axis_index) * inner;
            let destination = (outer_index * output_length + axis_index) * inner;
            output[destination..destination + inner].copy_from_slice(&data[source..source + inner]);
        }
    }
    shape[axis] = i64::try_from(output_length).map_err(|_| super::SIZE_OVERFLOW)?;
    Ok(output)
}
