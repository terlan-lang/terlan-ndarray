//! NumPy-compatible two-dimensional histogram construction.

use super::native_support::{
    delete_arrays, floats, owned_real, transfer_arrays, Array, INVALID_ARGUMENT, SHAPE_MISMATCH,
    SIZE_OVERFLOW,
};

/// One validated axis-bin definition.
enum AxisBins {
    /// A positive count with inclusive outer bounds.
    Uniform {
        count: usize,
        lower: f64,
        upper: f64,
    },
    /// A strictly increasing finite edge vector.
    Edges(Vec<f64>),
}

impl AxisBins {
    /// Materializes the complete edge vector for this axis.
    fn edges(self) -> Vec<f64> {
        match self {
            Self::Edges(edges) => edges,
            Self::Uniform {
                count,
                lower,
                upper,
            } => (0..=count)
                .map(|index| lower + (upper - lower) * index as f64 / count as f64)
                .collect(),
        }
    }
}

/// Validates and clears one transferred handle-array result pair.
unsafe fn initialize_outputs(output: *mut *mut u64, output_length: *mut usize) -> Result<(), i32> {
    if output.is_null() || output_length.is_null() {
        return Err(INVALID_ARGUMENT);
    }
    unsafe {
        *output = std::ptr::null_mut();
        *output_length = 0;
    }
    Ok(())
}

/// Converts one positive ABI bin count to a host index.
fn bin_count(value: i64) -> Result<usize, i32> {
    match usize::try_from(value) {
        Ok(value) if value > 0 => Ok(value),
        _ => Err(INVALID_ARGUMENT),
    }
}

/// Implements NumPy's inferred empty and constant outer-edge rules.
fn inferred_bounds(values: &[f64]) -> Result<(f64, f64), i32> {
    if values.is_empty() {
        return Ok((0.0, 1.0));
    }
    let mut lower = f64::INFINITY;
    let mut upper = f64::NEG_INFINITY;
    for &value in values {
        if !value.is_finite() {
            return Err(INVALID_ARGUMENT);
        }
        lower = lower.min(value);
        upper = upper.max(value);
    }
    if lower == upper {
        lower -= 0.5;
        upper += 0.5;
    }
    Ok((lower, upper))
}

/// Validates one explicit finite range and bin count.
fn uniform(count: i64, lower: f64, upper: f64) -> Result<AxisBins, i32> {
    let count = bin_count(count)?;
    if !lower.is_finite() || !upper.is_finite() || lower >= upper {
        return Err(INVALID_ARGUMENT);
    }
    Ok(AxisBins::Uniform {
        count,
        lower,
        upper,
    })
}

/// Validates one explicit edge vector.
fn explicit_edges(values: Vec<f64>) -> Result<AxisBins, i32> {
    if values.len() < 2
        || values.iter().any(|value| !value.is_finite())
        || values.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(INVALID_ARGUMENT);
    }
    Ok(AxisBins::Edges(values))
}

/// Finds the NumPy bin containing one scalar, including the final right edge.
fn bin(value: f64, edges: &[f64]) -> Option<usize> {
    if !value.is_finite() || value < edges[0] || value > edges[edges.len() - 1] {
        return None;
    }
    if value == edges[edges.len() - 1] {
        return Some(edges.len() - 2);
    }
    let insertion = edges.partition_point(|edge| *edge <= value);
    insertion
        .checked_sub(1)
        .filter(|index| *index + 1 < edges.len())
}

/// Computes and transfers a histogram with two owned edge vectors.
unsafe fn histogram(
    x: Array,
    y: Array,
    x_bins: AxisBins,
    y_bins: AxisBins,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    if let Err(status) = unsafe { initialize_outputs(output, output_length) } {
        return status;
    }
    let x_values = match unsafe { floats(x, false) } {
        Ok(values) => values,
        Err(status) => return status,
    };
    let y_values = match unsafe { floats(y, false) } {
        Ok(values) if values.len() == x_values.len() => values,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let x_edges = x_bins.edges();
    let y_edges = y_bins.edges();
    let x_count = x_edges.len() - 1;
    let y_count = y_edges.len() - 1;
    let cells = match x_count.checked_mul(y_count) {
        Some(value) => value,
        None => return SIZE_OVERFLOW,
    };
    let mut values = vec![0.0; cells];
    for (&x_value, &y_value) in x_values.iter().zip(&y_values) {
        if let (Some(x_index), Some(y_index)) = (bin(x_value, &x_edges), bin(y_value, &y_edges)) {
            values[x_index * y_count + y_index] += 1.0;
        }
    }
    if density {
        let total = values.iter().sum::<f64>();
        if total > 0.0 {
            for x_index in 0..x_count {
                let x_width = x_edges[x_index + 1] - x_edges[x_index];
                for y_index in 0..y_count {
                    let y_width = y_edges[y_index + 1] - y_edges[y_index];
                    values[x_index * y_count + y_index] /= total * x_width * y_width;
                }
            }
        }
    }
    let x_count = match i64::try_from(x_count) {
        Ok(value) => value,
        Err(_) => return SIZE_OVERFLOW,
    };
    let y_count = match i64::try_from(y_count) {
        Ok(value) => value,
        Err(_) => return SIZE_OVERFLOW,
    };
    let histogram = match unsafe { owned_real(&values, &[x_count, y_count], 3) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let x_output = match unsafe { owned_real(&x_edges, &[x_count + 1], 3) } {
        Ok(value) => value,
        Err(status) => {
            unsafe { delete_arrays(&[histogram]) };
            return status;
        }
    };
    let y_output = match unsafe { owned_real(&y_edges, &[y_count + 1], 3) } {
        Ok(value) => value,
        Err(status) => {
            unsafe { delete_arrays(&[histogram, x_output]) };
            return status;
        }
    };
    unsafe { transfer_arrays(vec![histogram, x_output, y_output], output, output_length) }
}

/// Computes a histogram with inferred axis ranges and equal bin counts.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_histogram2d(
    x: Array,
    y: Array,
    bins: i64,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    if let Err(status) = unsafe { initialize_outputs(output, output_length) } {
        return status;
    }
    let x_values = match unsafe { floats(x, false) } {
        Ok(values) => values,
        Err(status) => return status,
    };
    let y_values = match unsafe { floats(y, false) } {
        Ok(values) if values.len() == x_values.len() => values,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let bins = match bin_count(bins) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let (x_lower, x_upper) = match inferred_bounds(&x_values) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let (y_lower, y_upper) = match inferred_bounds(&y_values) {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe {
        histogram(
            x,
            y,
            AxisBins::Uniform {
                count: bins,
                lower: x_lower,
                upper: x_upper,
            },
            AxisBins::Uniform {
                count: bins,
                lower: y_lower,
                upper: y_upper,
            },
            density,
            output,
            output_length,
        )
    }
}

/// Computes a histogram with explicit per-axis counts and ranges.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_histogram2d_range(
    x: Array,
    y: Array,
    x_bins: i64,
    y_bins: i64,
    minima: Array,
    maxima: Array,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    if let Err(status) = unsafe { initialize_outputs(output, output_length) } {
        return status;
    }
    let minima = match unsafe { floats(minima, true) } {
        Ok(values) if values.len() == 2 => values,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let maxima = match unsafe { floats(maxima, true) } {
        Ok(values) if values.len() == 2 => values,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let x_bins = match uniform(x_bins, minima[0], maxima[0]) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let y_bins = match uniform(y_bins, minima[1], maxima[1]) {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe { histogram(x, y, x_bins, y_bins, density, output, output_length) }
}

/// Computes a histogram from explicit edge vectors for both axes.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_histogram2d_edges(
    x: Array,
    y: Array,
    x_edges: Array,
    y_edges: Array,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    if let Err(status) = unsafe { initialize_outputs(output, output_length) } {
        return status;
    }
    let x_edges = match unsafe { floats(x_edges, true) }.and_then(explicit_edges) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let y_edges = match unsafe { floats(y_edges, true) }.and_then(explicit_edges) {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe { histogram(x, y, x_edges, y_edges, density, output, output_length) }
}

/// Computes a histogram with explicit x edges and uniform y bins.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_histogram2d_x_edges(
    x: Array,
    y: Array,
    x_edges: Array,
    y_bins: i64,
    y_lower: f64,
    y_upper: f64,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    if let Err(status) = unsafe { initialize_outputs(output, output_length) } {
        return status;
    }
    let x_edges = match unsafe { floats(x_edges, true) }.and_then(explicit_edges) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let y_bins = match uniform(y_bins, y_lower, y_upper) {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe { histogram(x, y, x_edges, y_bins, density, output, output_length) }
}

/// Computes a histogram with uniform x bins and explicit y edges.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_histogram2d_y_edges(
    x: Array,
    y: Array,
    x_bins: i64,
    x_lower: f64,
    x_upper: f64,
    y_edges: Array,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    if let Err(status) = unsafe { initialize_outputs(output, output_length) } {
        return status;
    }
    let x_bins = match uniform(x_bins, x_lower, x_upper) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let y_edges = match unsafe { floats(y_edges, true) }.and_then(explicit_edges) {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe { histogram(x, y, x_bins, y_edges, density, output, output_length) }
}

#[cfg(test)]
#[path = "ndarray_histogram2d_tests.rs"]
mod tests;
