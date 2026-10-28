//! NumPy-compatible multidimensional histogram construction.

use rustfft::num_complex::Complex64;

use super::native_support::{
    delete_arrays, dtype, floats, ints, numeric, owned_int_as, owned_numeric_as, owned_real, shape,
    transfer_arrays, Array, INVALID_ARGUMENT, SHAPE_MISMATCH, SIZE_OVERFLOW, UNSUPPORTED_DTYPE,
};

/// Exact or floating values used for samples and explicit edges.
enum ScalarValues {
    /// Signed integer values retained without a Float64 round trip.
    Integers(Vec<i64>),
    /// Real values represented by the package computation type.
    Reals(Vec<f64>),
}

impl ScalarValues {
    /// Returns the number of stored scalar values.
    fn len(&self) -> usize {
        match self {
            Self::Integers(values) => values.len(),
            Self::Reals(values) => values.len(),
        }
    }

    /// Returns one value as Float64 for inferred ranges and mixed comparisons.
    fn float(&self, index: usize) -> f64 {
        match self {
            Self::Integers(values) => values[index] as f64,
            Self::Reals(values) => values[index],
        }
    }
}

/// One validated histogram axis.
struct HistogramAxis {
    /// Bin edges in source-preserving representation.
    values: ScalarValues,
    /// Dtype used for the returned edge array.
    dtype: i32,
}

impl HistogramAxis {
    /// Returns the number of bins on this axis.
    fn bins(&self) -> usize {
        self.values.len() - 1
    }

    /// Returns one bin width as Float64 for density normalization.
    fn width(&self, index: usize) -> f64 {
        self.values.float(index + 1) - self.values.float(index)
    }
}

/// Clears a generated transferred-array output pair.
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

/// Returns whether a dtype has an exact signed integer readback path.
fn is_signed_integer(dtype: i32) -> bool {
    matches!(dtype, 2 | 4 | 5 | 6)
}

/// Returns the edge dtype NumPy derives from equal-width source data.
fn inferred_edge_dtype(dtype: i32) -> i32 {
    if matches!(dtype, 3 | 11 | 14) {
        dtype
    } else {
        3
    }
}

/// Reads real sample or edge values while retaining signed integers exactly.
unsafe fn scalar_values(array: Array) -> Result<(ScalarValues, i32), i32> {
    let dtype = unsafe { dtype(array)? };
    if matches!(dtype, 12 | 13) {
        return Err(UNSUPPORTED_DTYPE);
    }
    if is_signed_integer(dtype) {
        return unsafe { ints(array).map(|values| (ScalarValues::Integers(values), dtype)) };
    }
    unsafe { floats(array, false).map(|values| (ScalarValues::Reals(values), dtype)) }
}

/// Validates the `[sample, dimension]` source matrix.
unsafe fn sample_matrix(array: Array) -> Result<(ScalarValues, i32, usize, usize), i32> {
    let dimensions = unsafe { shape(array)? };
    if dimensions.len() != 2 || dimensions[1] <= 0 {
        return Err(SHAPE_MISMATCH);
    }
    let sample_count = usize::try_from(dimensions[0]).map_err(|_| SHAPE_MISMATCH)?;
    let dimension_count = usize::try_from(dimensions[1]).map_err(|_| SHAPE_MISMATCH)?;
    let (values, dtype) = unsafe { scalar_values(array)? };
    let expected = sample_count
        .checked_mul(dimension_count)
        .ok_or(SIZE_OVERFLOW)?;
    if values.len() != expected {
        return Err(SHAPE_MISMATCH);
    }
    Ok((values, dtype, sample_count, dimension_count))
}

/// Builds equal-width edges for one axis.
fn uniform_axis(count: i64, lower: f64, upper: f64, dtype: i32) -> Result<HistogramAxis, i32> {
    let count = usize::try_from(count).map_err(|_| INVALID_ARGUMENT)?;
    if count == 0 || !lower.is_finite() || !upper.is_finite() || lower >= upper {
        return Err(INVALID_ARGUMENT);
    }
    let length = count.checked_add(1).ok_or(SIZE_OVERFLOW)?;
    let values = (0..length)
        .map(|index| lower + (upper - lower) * index as f64 / count as f64)
        .collect();
    Ok(HistogramAxis {
        values: ScalarValues::Reals(values),
        dtype,
    })
}

/// Computes NumPy's empty, constant, and finite inferred bounds for one axis.
fn inferred_axis(
    values: &ScalarValues,
    sample_count: usize,
    dimension_count: usize,
    dimension: usize,
    bins: i64,
    edge_dtype: i32,
) -> Result<HistogramAxis, i32> {
    if sample_count == 0 {
        return uniform_axis(bins, 0.0, 1.0, edge_dtype);
    }
    let mut lower = f64::INFINITY;
    let mut upper = f64::NEG_INFINITY;
    for sample in 0..sample_count {
        let value = values.float(sample * dimension_count + dimension);
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
    uniform_axis(bins, lower, upper, edge_dtype)
}

/// Reads one positive per-axis bin-count vector.
unsafe fn bin_counts(array: Array, dimension_count: usize) -> Result<Vec<i64>, i32> {
    let dimensions = unsafe { shape(array)? };
    if dimensions != [i64::try_from(dimension_count).map_err(|_| SIZE_OVERFLOW)?] {
        return Err(SHAPE_MISMATCH);
    }
    let values = unsafe { ints(array)? };
    if values.iter().any(|value| *value <= 0) {
        return Err(INVALID_ARGUMENT);
    }
    Ok(values)
}

/// Reads one finite per-axis range endpoint vector.
unsafe fn range_values(array: Array, dimension_count: usize) -> Result<Vec<f64>, i32> {
    let dimensions = unsafe { shape(array)? };
    if dimensions != [i64::try_from(dimension_count).map_err(|_| SIZE_OVERFLOW)?] {
        return Err(SHAPE_MISMATCH);
    }
    let values = unsafe { floats(array, true)? };
    if values.len() != dimension_count {
        return Err(SHAPE_MISMATCH);
    }
    Ok(values)
}

/// Builds inferred equal-width axes from one scalar or per-axis bin contract.
fn inferred_axes(
    values: &ScalarValues,
    sample_dtype: i32,
    sample_count: usize,
    dimension_count: usize,
    bins: &[i64],
) -> Result<Vec<HistogramAxis>, i32> {
    if bins.len() != dimension_count {
        return Err(SHAPE_MISMATCH);
    }
    let edge_dtype = inferred_edge_dtype(sample_dtype);
    (0..dimension_count)
        .map(|dimension| {
            inferred_axis(
                values,
                sample_count,
                dimension_count,
                dimension,
                bins[dimension],
                edge_dtype,
            )
        })
        .collect()
}

/// Builds explicit-range equal-width axes.
fn range_axes(
    bins: &[i64],
    minima: &[f64],
    maxima: &[f64],
    edge_dtype: i32,
) -> Result<Vec<HistogramAxis>, i32> {
    if bins.len() != minima.len() || bins.len() != maxima.len() {
        return Err(SHAPE_MISMATCH);
    }
    bins.iter()
        .zip(minima)
        .zip(maxima)
        .map(|((&count, &lower), &upper)| uniform_axis(count, lower, upper, edge_dtype))
        .collect()
}

/// Reads packed explicit axes using offsets and output dtype metadata.
unsafe fn explicit_axes(
    packed: Array,
    offsets: Array,
    dtypes: Array,
    dimension_count: usize,
) -> Result<Vec<HistogramAxis>, i32> {
    if unsafe { shape(packed)? }.len() != 1 {
        return Err(SHAPE_MISMATCH);
    }
    let offsets = unsafe { ints(offsets)? };
    let dtypes = unsafe { ints(dtypes)? };
    if offsets.len() != dimension_count + 1
        || dtypes.len() != dimension_count
        || offsets.first() != Some(&0)
    {
        return Err(SHAPE_MISMATCH);
    }
    let (values, _) = unsafe { scalar_values(packed)? };
    if usize::try_from(*offsets.last().ok_or(SHAPE_MISMATCH)?).map_err(|_| SHAPE_MISMATCH)?
        != values.len()
    {
        return Err(SHAPE_MISMATCH);
    }
    let mut axes = Vec::with_capacity(dimension_count);
    for dimension in 0..dimension_count {
        let start = usize::try_from(offsets[dimension]).map_err(|_| SHAPE_MISMATCH)?;
        let end = usize::try_from(offsets[dimension + 1]).map_err(|_| SHAPE_MISMATCH)?;
        if end < start + 2 || end > values.len() {
            return Err(INVALID_ARGUMENT);
        }
        let axis_values = match &values {
            ScalarValues::Integers(values) => ScalarValues::Integers(values[start..end].to_vec()),
            ScalarValues::Reals(values) => ScalarValues::Reals(values[start..end].to_vec()),
        };
        let invalid = match &axis_values {
            ScalarValues::Integers(values) => values.windows(2).any(|pair| pair[0] > pair[1]),
            ScalarValues::Reals(values) => {
                values.iter().any(|value| value.is_nan())
                    || values.windows(2).any(|pair| pair[0] > pair[1])
            }
        };
        let dtype = i32::try_from(dtypes[dimension]).map_err(|_| UNSUPPORTED_DTYPE)?;
        if invalid || !matches!(dtype, 2..=11 | 14) {
            return Err(INVALID_ARGUMENT);
        }
        axes.push(HistogramAxis {
            values: axis_values,
            dtype,
        });
    }
    Ok(axes)
}

/// Finds the bin for one sample value on one axis.
fn axis_bin(samples: &ScalarValues, sample_index: usize, axis: &HistogramAxis) -> Option<usize> {
    match (samples, &axis.values) {
        (ScalarValues::Integers(samples), ScalarValues::Integers(edges)) => {
            let value = samples[sample_index];
            if value < edges[0] || value > edges[edges.len() - 1] {
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
        _ => {
            let value = samples.float(sample_index);
            let last = axis.values.float(axis.values.len() - 1);
            if value.is_nan() || value < axis.values.float(0) || value > last {
                return None;
            }
            if value == last {
                return Some(axis.bins() - 1);
            }
            (0..axis.bins()).find(|index| {
                value >= axis.values.float(*index) && value < axis.values.float(*index + 1)
            })
        }
    }
}

/// Allocates one edge array while retaining its reviewed dtype.
unsafe fn edge_array(axis: &HistogramAxis) -> Result<Array, i32> {
    let shape = [i64::try_from(axis.values.len()).map_err(|_| SIZE_OVERFLOW)?];
    match &axis.values {
        ScalarValues::Integers(values) => unsafe { owned_int_as(values, &shape, axis.dtype) },
        ScalarValues::Reals(values) if is_signed_integer(axis.dtype) => {
            let values = values.iter().map(|value| *value as i64).collect::<Vec<_>>();
            unsafe { owned_int_as(&values, &shape, axis.dtype) }
        }
        ScalarValues::Reals(values) => unsafe { owned_real(values, &shape, axis.dtype) },
    }
}

/// Counts samples and transfers the histogram followed by every edge vector.
unsafe fn histogramdd_result(
    samples: &ScalarValues,
    sample_count: usize,
    dimension_count: usize,
    axes: Vec<HistogramAxis>,
    weights: Option<Array>,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    if let Err(status) = unsafe { initialize_outputs(output, output_length) } {
        return status;
    }
    let weights = match weights {
        Some(array) => {
            let weight_shape = match unsafe { shape(array) } {
                Ok(value) => value,
                Err(status) => return status,
            };
            if weight_shape != [sample_count as i64] {
                return SHAPE_MISMATCH;
            }
            match unsafe { numeric(array, false) } {
                Ok(values) => Some(values),
                Err(status) => return status,
            }
        }
        None => None,
    };
    let shape = axes
        .iter()
        .map(|axis| i64::try_from(axis.bins()).map_err(|_| SIZE_OVERFLOW))
        .collect::<Result<Vec<_>, _>>();
    let shape = match shape {
        Ok(value) => value,
        Err(status) => return status,
    };
    let cells = match shape.iter().try_fold(1usize, |total, value| {
        usize::try_from(*value)
            .ok()
            .and_then(|value| total.checked_mul(value))
    }) {
        Some(value) => value,
        None => return SIZE_OVERFLOW,
    };
    let mut counts = vec![Complex64::new(0.0, 0.0); cells];
    for sample in 0..sample_count {
        let mut coordinate = 0usize;
        let mut admitted = true;
        for (dimension, axis) in axes.iter().enumerate() {
            let index = sample * dimension_count + dimension;
            let Some(bin) = axis_bin(samples, index, axis) else {
                admitted = false;
                break;
            };
            coordinate = coordinate * axis.bins() + bin;
        }
        if admitted {
            counts[coordinate] += weights
                .as_ref()
                .map_or(Complex64::new(1.0, 0.0), |weights| weights.values[sample]);
        }
    }
    if density {
        let total: Complex64 = counts.iter().copied().sum();
        for (coordinate, count) in counts.iter_mut().enumerate() {
            *count /= total;
            let mut remainder = coordinate;
            for axis in axes.iter().rev() {
                let index = remainder % axis.bins();
                remainder /= axis.bins();
                *count /= axis.width(index);
            }
        }
    }
    let complex = weights.as_ref().is_some_and(|values| values.complex);
    let histogram = if complex {
        unsafe { owned_numeric_as(&counts, &shape, 13) }
    } else {
        let values = counts.iter().map(|value| value.re).collect::<Vec<_>>();
        unsafe { owned_real(&values, &shape, 3) }
    };
    let histogram = match histogram {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut arrays = vec![histogram];
    for axis in &axes {
        match unsafe { edge_array(axis) } {
            Ok(array) => arrays.push(array),
            Err(status) => {
                unsafe { delete_arrays(&arrays) };
                return status;
            }
        }
    }
    unsafe { transfer_arrays(arrays, output, output_length) }
}

/// Executes one histogram from prevalidated axes.
unsafe fn run(
    samples_array: Array,
    axes: Vec<HistogramAxis>,
    weights: Option<Array>,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let (values, _, sample_count, dimension_count) = match unsafe { sample_matrix(samples_array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if axes.len() != dimension_count {
        return SHAPE_MISMATCH;
    }
    unsafe {
        histogramdd_result(
            &values,
            sample_count,
            dimension_count,
            axes,
            weights,
            density,
            output,
            output_length,
        )
    }
}

/// Builds inferred axes for one scalar bin count.
unsafe fn scalar_bin_axes(samples_array: Array, bins: i64) -> Result<Vec<HistogramAxis>, i32> {
    if bins <= 0 {
        return Err(INVALID_ARGUMENT);
    }
    let (values, sample_dtype, sample_count, dimension_count) =
        unsafe { sample_matrix(samples_array)? };
    inferred_axes(
        &values,
        sample_dtype,
        sample_count,
        dimension_count,
        &vec![bins; dimension_count],
    )
}

/// Builds inferred axes for explicit per-axis counts.
unsafe fn array_bin_axes(samples_array: Array, bins: Array) -> Result<Vec<HistogramAxis>, i32> {
    let (values, sample_dtype, sample_count, dimension_count) =
        unsafe { sample_matrix(samples_array)? };
    let bins = unsafe { bin_counts(bins, dimension_count)? };
    inferred_axes(&values, sample_dtype, sample_count, dimension_count, &bins)
}

/// Builds an inferred-range multidimensional histogram.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_numpy_histogramdd(
    samples: Array,
    bins: i64,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let axes = match unsafe { scalar_bin_axes(samples, bins) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe { run(samples, axes, None, density, output, output_length) }
}

/// Builds an inferred-range histogram with per-axis counts.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_numpy_histogramdd_bins_array(
    samples: Array,
    bins: Array,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let axes = match unsafe { array_bin_axes(samples, bins) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe { run(samples, axes, None, density, output, output_length) }
}

/// Builds an explicit-range histogram with per-axis counts.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_numpy_histogramdd_range_arrays(
    samples: Array,
    bins: Array,
    minima: Array,
    maxima: Array,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let (_, sample_dtype, _, dimension_count) = match unsafe { sample_matrix(samples) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let bins = match unsafe { bin_counts(bins, dimension_count) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let minima = match unsafe { range_values(minima, dimension_count) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let maxima = match unsafe { range_values(maxima, dimension_count) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let axes = match range_axes(&bins, &minima, &maxima, inferred_edge_dtype(sample_dtype)) {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe { run(samples, axes, None, density, output, output_length) }
}

/// Builds a histogram from packed explicit edge vectors.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_numpy_histogramdd_edges_packed(
    samples: Array,
    edges: Array,
    offsets: Array,
    dtypes: Array,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let (_, _, _, dimension_count) = match unsafe { sample_matrix(samples) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let axes = match unsafe { explicit_axes(edges, offsets, dtypes, dimension_count) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe { run(samples, axes, None, density, output, output_length) }
}

/// Builds a weighted inferred-range histogram with one shared bin count.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_numpy_histogramdd_weighted(
    samples: Array,
    weights: Array,
    bins: i64,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let axes = match unsafe { scalar_bin_axes(samples, bins) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe { run(samples, axes, Some(weights), density, output, output_length) }
}

/// Builds a weighted inferred-range histogram with per-axis counts.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_numpy_histogramdd_weighted_bins_array(
    samples: Array,
    weights: Array,
    bins: Array,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let axes = match unsafe { array_bin_axes(samples, bins) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe { run(samples, axes, Some(weights), density, output, output_length) }
}

/// Builds a weighted explicit-range histogram with per-axis counts.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_numpy_histogramdd_weighted_range_arrays(
    samples: Array,
    weights: Array,
    bins: Array,
    minima: Array,
    maxima: Array,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let (_, sample_dtype, _, dimension_count) = match unsafe { sample_matrix(samples) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let bins = match unsafe { bin_counts(bins, dimension_count) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let minima = match unsafe { range_values(minima, dimension_count) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let maxima = match unsafe { range_values(maxima, dimension_count) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let axes = match range_axes(&bins, &minima, &maxima, inferred_edge_dtype(sample_dtype)) {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe { run(samples, axes, Some(weights), density, output, output_length) }
}

/// Builds a weighted histogram from packed explicit edge vectors.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_numpy_histogramdd_weighted_edges_packed(
    samples: Array,
    weights: Array,
    edges: Array,
    offsets: Array,
    dtypes: Array,
    density: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let (_, _, _, dimension_count) = match unsafe { sample_matrix(samples) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let axes = match unsafe { explicit_axes(edges, offsets, dtypes, dimension_count) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe { run(samples, axes, Some(weights), density, output, output_length) }
}

#[cfg(test)]
#[path = "ndarray_histogramdd_tests.rs"]
mod tests;
