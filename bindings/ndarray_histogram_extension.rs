//! NumPy-compatible one-dimensional histogram construction.

use half::f16;
use rustfft::num_complex::Complex64;

use super::native_support::{
    delete_arrays, dtype, floats, numeric, owned_int, owned_numeric_as, owned_real, shape,
    transfer_arrays, Array, DOMAIN_ERROR, INVALID_ARGUMENT, SHAPE_MISMATCH, SIZE_OVERFLOW,
    UNSUPPORTED_DTYPE,
};

/// Returns NumPy's edge dtype for equal-width bins derived from one array.
unsafe fn inferred_edge_dtype(array: Array) -> Result<i32, i32> {
    let dtype = unsafe { dtype(array)? };
    Ok(if matches!(dtype, 3 | 11 | 14) {
        dtype
    } else {
        3
    })
}

/// Rounds one explicit range endpoint through the source floating dtype.
fn cast_range_endpoint(value: f64, dtype: i32) -> f64 {
    match dtype {
        14 => f16::from_f64(value).to_f64(),
        11 => f64::from(value as f32),
        _ => value,
    }
}

/// Square-root bin-count estimator.
const ESTIMATOR_SQRT: i64 = 0;
/// Sturges bin-count estimator.
const ESTIMATOR_STURGES: i64 = 1;
/// Rice bin-count estimator.
const ESTIMATOR_RICE: i64 = 2;
/// Scott bin-width estimator.
const ESTIMATOR_SCOTT: i64 = 3;
/// Stone cross-validation estimator.
const ESTIMATOR_STONE: i64 = 4;
/// Doane skewness-aware estimator.
const ESTIMATOR_DOANE: i64 = 5;
/// Freedman-Diaconis estimator.
const ESTIMATOR_FD: i64 = 6;
/// NumPy automatic estimator.
const ESTIMATOR_AUTO: i64 = 7;

/// Clears one generated owned-array-list output pair.
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

/// Returns NumPy's inferred finite outer bounds.
fn inferred_bounds(values: &[f64]) -> Result<(f64, f64), i32> {
    if values.is_empty() {
        return Ok((0.0, 1.0));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(DOMAIN_ERROR);
    }
    let mut lower = values[0];
    let mut upper = values[0];
    for &value in &values[1..] {
        lower = lower.min(value);
        upper = upper.max(value);
    }
    if lower == upper {
        lower -= 0.5;
        upper += 0.5;
    }
    Ok((lower, upper))
}

/// Materializes equal-width edges after validating their range and count.
fn uniform_edges(bins: i64, lower: f64, upper: f64) -> Result<Vec<f64>, i32> {
    let bins = usize::try_from(bins).map_err(|_| INVALID_ARGUMENT)?;
    if bins == 0 || !lower.is_finite() || !upper.is_finite() || lower >= upper {
        return Err(INVALID_ARGUMENT);
    }
    let capacity = bins.checked_add(1).ok_or(SIZE_OVERFLOW)?;
    let mut edges = Vec::with_capacity(capacity);
    for index in 0..=bins {
        edges.push(lower + (upper - lower) * index as f64 / bins as f64);
    }
    Ok(edges)
}

/// Reads and validates a one-dimensional explicit edge array.
unsafe fn explicit_edges(array: Array) -> Result<Vec<f64>, i32> {
    let dimensions = unsafe { shape(array)? };
    if dimensions.len() != 1 {
        return Err(SHAPE_MISMATCH);
    }
    let edges = unsafe { floats(array, false)? };
    if edges.len() < 2
        || edges.iter().any(|value| value.is_nan())
        || edges.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(INVALID_ARGUMENT);
    }
    Ok(edges)
}

/// Finds the bin containing one scalar, including the final right edge.
fn bin(value: f64, edges: &[f64]) -> Option<usize> {
    if value.is_nan() || value < edges[0] || value > edges[edges.len() - 1] {
        return None;
    }
    if value == edges[edges.len() - 1] {
        return Some(edges.len() - 2);
    }
    edges
        .partition_point(|edge| *edge <= value)
        .checked_sub(1)
        .filter(|index| *index + 1 < edges.len())
}

/// Computes one weighted or unweighted histogram and transfers its arrays.
unsafe fn histogram_result(
    array: Array,
    weights: Option<Array>,
    edges: Vec<f64>,
    edge_dtype: i32,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    if let Err(status) = unsafe { initialize_outputs(output, output_length) } {
        return status;
    }
    let values = match unsafe { numeric(array, false) } {
        Ok(values) if !values.complex => values,
        Ok(_) => return UNSUPPORTED_DTYPE,
        Err(status) => return status,
    };
    let weight_values = match weights {
        Some(weights) => {
            let weight_shape = match unsafe { shape(weights) } {
                Ok(value) => value,
                Err(status) => return status,
            };
            let value_shape = match unsafe { shape(array) } {
                Ok(value) => value,
                Err(status) => return status,
            };
            if weight_shape != value_shape {
                return SHAPE_MISMATCH;
            }
            match unsafe { numeric(weights, false) } {
                Ok(values) => Some(values),
                Err(status) => return status,
            }
        }
        None => None,
    };
    let mut counts = vec![Complex64::new(0.0, 0.0); edges.len() - 1];
    for (index, value) in values.values.iter().enumerate() {
        if let Some(bin) = bin(value.re, &edges) {
            counts[bin] += weight_values
                .as_ref()
                .map_or(Complex64::new(1.0, 0.0), |weights| weights.values[index]);
        }
    }
    if density {
        let total: Complex64 = counts.iter().copied().sum();
        for (index, count) in counts.iter_mut().enumerate() {
            *count /= total;
            *count /= edges[index + 1] - edges[index];
        }
    }
    let count_shape = [match i64::try_from(counts.len()) {
        Ok(value) => value,
        Err(_) => return SIZE_OVERFLOW,
    }];
    let count_array = match (&weight_values, density) {
        (None, false) => {
            let integers = counts
                .iter()
                .map(|value| value.re as i64)
                .collect::<Vec<_>>();
            unsafe { owned_int(&integers, &count_shape) }
        }
        (Some(weights), false) => unsafe { owned_numeric_as(&counts, &count_shape, weights.dtype) },
        (Some(weights), true) if weights.complex => unsafe {
            owned_numeric_as(&counts, &count_shape, 13)
        },
        _ => {
            let reals = counts.iter().map(|value| value.re).collect::<Vec<_>>();
            unsafe { owned_real(&reals, &count_shape, 3) }
        }
    };
    let count_array = match count_array {
        Ok(value) => value,
        Err(status) => return status,
    };
    let edge_shape = [match i64::try_from(edges.len()) {
        Ok(value) => value,
        Err(_) => {
            unsafe { delete_arrays(&[count_array]) };
            return SIZE_OVERFLOW;
        }
    }];
    let edge_array = match unsafe { owned_real(&edges, &edge_shape, edge_dtype) } {
        Ok(value) => value,
        Err(status) => {
            unsafe { delete_arrays(&[count_array]) };
            return status;
        }
    };
    unsafe { transfer_arrays(vec![count_array, edge_array], output, output_length) }
}

/// Returns a linear percentile from sorted finite values.
fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let position = fraction * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
}

/// Returns NumPy's population standard deviation.
fn standard_deviation(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

/// Computes Stone's cross-validation bin count.
fn stone_bins(values: &[f64], lower: f64, upper: f64) -> usize {
    let sample_lower = values.iter().copied().fold(f64::INFINITY, f64::min);
    let sample_upper = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sample_spread = sample_upper - sample_lower;
    if values.len() <= 1 || sample_spread == 0.0 {
        return 1;
    }
    let limit = 100usize.max((values.len() as f64).sqrt() as usize);
    let mut selected = 1;
    let mut selected_score = f64::INFINITY;
    for bins in 1..=limit {
        let mut counts = vec![0.0; bins];
        for &value in values {
            let index = if value == upper {
                Some(bins - 1)
            } else if value >= lower && value < upper {
                let index = ((value - lower) * bins as f64 / (upper - lower)).floor() as usize;
                (index < bins).then_some(index)
            } else {
                None
            };
            if let Some(index) = index {
                counts[index] += 1.0;
            }
        }
        let probability_square = counts
            .iter()
            .map(|count| (count / values.len() as f64).powi(2))
            .sum::<f64>();
        let width = sample_spread / bins as f64;
        let score = (2.0 - (values.len() as f64 + 1.0) * probability_square) / width;
        if score < selected_score {
            selected = bins;
            selected_score = score;
        }
    }
    ((upper - lower) / (sample_spread / selected as f64))
        .ceil()
        .max(1.0) as usize
}

/// Computes one NumPy-compatible automatic bin count.
fn estimator_bins(source: &[f64], estimator: i64, lower: f64, upper: f64) -> Result<usize, i32> {
    if !matches!(estimator, 0..=7) {
        return Err(INVALID_ARGUMENT);
    }
    let mut values = source
        .iter()
        .copied()
        .filter(|value| *value >= lower && *value <= upper)
        .collect::<Vec<_>>();
    if values.is_empty() || lower == upper {
        return Ok(1);
    }
    values.sort_by(f64::total_cmp);
    let count = values.len() as f64;
    let spread = upper - lower;
    let sample_spread = values[values.len() - 1] - values[0];
    let width = match estimator {
        ESTIMATOR_SQRT => sample_spread / count.sqrt(),
        ESTIMATOR_STURGES => sample_spread / (count.log2() + 1.0),
        ESTIMATOR_RICE => sample_spread / (2.0 * count.cbrt()),
        ESTIMATOR_SCOTT => {
            (24.0 * std::f64::consts::PI.sqrt() / count).cbrt() * standard_deviation(&values)
        }
        ESTIMATOR_STONE => return Ok(stone_bins(&values, lower, upper)),
        ESTIMATOR_DOANE => {
            if values.len() <= 2 {
                0.0
            } else {
                let deviation = standard_deviation(&values);
                if deviation == 0.0 {
                    0.0
                } else {
                    let mean = values.iter().sum::<f64>() / count;
                    let skew = values
                        .iter()
                        .map(|value| ((value - mean) / deviation).powi(3))
                        .sum::<f64>()
                        / count;
                    let sigma = (6.0 * (count - 2.0) / ((count + 1.0) * (count + 3.0))).sqrt();
                    sample_spread / (1.0 + count.log2() + (1.0 + skew.abs() / sigma).log2())
                }
            }
        }
        ESTIMATOR_FD | ESTIMATOR_AUTO => {
            let iqr = percentile(&values, 0.75) - percentile(&values, 0.25);
            let fd = 2.0 * iqr * count.powf(-1.0 / 3.0);
            if estimator == ESTIMATOR_FD {
                fd
            } else {
                let sturges = sample_spread / (count.log2() + 1.0);
                if fd == 0.0 {
                    sturges
                } else {
                    fd.min(sturges)
                }
            }
        }
        _ => unreachable!(),
    };
    Ok(if width > 0.0 {
        (spread / width).ceil().max(1.0) as usize
    } else {
        1
    })
}

/// Builds inferred fixed-count edges for one real array.
unsafe fn inferred_edges(array: Array, bins: i64) -> Result<Vec<f64>, i32> {
    let values = unsafe { floats(array, false)? };
    let (lower, upper) = inferred_bounds(&values)?;
    uniform_edges(bins, lower, upper)
}

/// Builds estimator-selected edges, optionally under an explicit range.
unsafe fn estimated_edges(
    array: Array,
    estimator: i64,
    range: Option<(f64, f64)>,
) -> Result<Vec<f64>, i32> {
    let values = unsafe { floats(array, false)? };
    let (lower, upper) = match range {
        Some((lower, upper)) if lower.is_finite() && upper.is_finite() && lower < upper => {
            (lower, upper)
        }
        Some(_) => return Err(INVALID_ARGUMENT),
        None => inferred_bounds(&values)?,
    };
    let bins = estimator_bins(&values, estimator, lower, upper)?;
    uniform_edges(bins as i64, lower, upper)
}

/// Builds an inferred-range one-dimensional histogram.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_numpy_histogram(
    array: Array,
    bins: i64,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let edge_dtype = match unsafe { inferred_edge_dtype(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let edges = match unsafe { inferred_edges(array, bins) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe {
        histogram_result(
            array,
            None,
            edges,
            edge_dtype,
            density,
            output,
            output_length,
        )
    }
}

/// Builds an explicit-range one-dimensional histogram.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_numpy_histogram_range(
    array: Array,
    bins: i64,
    lower: f64,
    upper: f64,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let edge_dtype = match unsafe { inferred_edge_dtype(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let edges = match uniform_edges(
        bins,
        cast_range_endpoint(lower, edge_dtype),
        cast_range_endpoint(upper, edge_dtype),
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe {
        histogram_result(
            array,
            None,
            edges,
            edge_dtype,
            density,
            output,
            output_length,
        )
    }
}

/// Builds an explicit-edge one-dimensional histogram.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_numpy_histogram_edges(
    array: Array,
    edges: Array,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let edge_dtype = match unsafe { dtype(edges) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let edges = match unsafe { explicit_edges(edges) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe {
        histogram_result(
            array,
            None,
            edges,
            edge_dtype,
            density,
            output,
            output_length,
        )
    }
}

/// Builds an inferred-range weighted histogram.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_numpy_histogram_weighted(
    array: Array,
    weights: Array,
    bins: i64,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let edge_dtype = match unsafe { inferred_edge_dtype(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let edges = match unsafe { inferred_edges(array, bins) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe {
        histogram_result(
            array,
            Some(weights),
            edges,
            edge_dtype,
            density,
            output,
            output_length,
        )
    }
}

/// Builds an explicit-range weighted histogram.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_numpy_histogram_weighted_range(
    array: Array,
    weights: Array,
    bins: i64,
    lower: f64,
    upper: f64,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let edge_dtype = match unsafe { inferred_edge_dtype(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let edges = match uniform_edges(
        bins,
        cast_range_endpoint(lower, edge_dtype),
        cast_range_endpoint(upper, edge_dtype),
    ) {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe {
        histogram_result(
            array,
            Some(weights),
            edges,
            edge_dtype,
            density,
            output,
            output_length,
        )
    }
}

/// Builds an explicit-edge weighted histogram.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_numpy_histogram_weighted_edges(
    array: Array,
    weights: Array,
    edges: Array,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let edge_dtype = match unsafe { dtype(edges) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let edges = match unsafe { explicit_edges(edges) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe {
        histogram_result(
            array,
            Some(weights),
            edges,
            edge_dtype,
            density,
            output,
            output_length,
        )
    }
}

/// Builds an inferred-range estimator-selected histogram.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_numpy_histogram_estimator(
    array: Array,
    estimator: i64,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let edge_dtype = match unsafe { inferred_edge_dtype(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let edges = match unsafe { estimated_edges(array, estimator, None) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe {
        histogram_result(
            array,
            None,
            edges,
            edge_dtype,
            density,
            output,
            output_length,
        )
    }
}

/// Builds an explicit-range estimator-selected histogram.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_numpy_histogram_estimator_range(
    array: Array,
    estimator: i64,
    lower: f64,
    upper: f64,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let edge_dtype = match unsafe { inferred_edge_dtype(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let edges = match unsafe {
        estimated_edges(
            array,
            estimator,
            Some((
                cast_range_endpoint(lower, edge_dtype),
                cast_range_endpoint(upper, edge_dtype),
            )),
        )
    } {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe {
        histogram_result(
            array,
            None,
            edges,
            edge_dtype,
            density,
            output,
            output_length,
        )
    }
}

/// Returns inferred fixed-count histogram edges without counting samples.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_numpy_histogram_bin_edges(
    array: Array,
    bins: i64,
    output: *mut Array,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    let edge_dtype = match unsafe { inferred_edge_dtype(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let edges = match unsafe { inferred_edges(array, bins) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let shape = [match i64::try_from(edges.len()) {
        Ok(value) => value,
        Err(_) => return SIZE_OVERFLOW,
    }];
    match unsafe { owned_real(&edges, &shape, edge_dtype) } {
        Ok(value) => {
            unsafe { *output = value };
            0
        }
        Err(status) => status,
    }
}

/// Returns estimator-selected histogram edges without counting samples.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_numpy_histogram_bin_edges_estimator(
    array: Array,
    estimator: i64,
    output: *mut Array,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    let edge_dtype = match unsafe { inferred_edge_dtype(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let edges = match unsafe { estimated_edges(array, estimator, None) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let shape = [match i64::try_from(edges.len()) {
        Ok(value) => value,
        Err(_) => return SIZE_OVERFLOW,
    }];
    match unsafe { owned_real(&edges, &shape, edge_dtype) } {
        Ok(value) => {
            unsafe { *output = value };
            0
        }
        Err(status) => status,
    }
}

#[cfg(test)]
#[path = "ndarray_histogram_tests.rs"]
mod tests;
