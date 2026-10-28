//! Structured NumPy-compatible random distributions for owned streams.

use faer::{Mat, Side};
use rand::Rng;
use rand_distr::weighted::WeightedAliasIndex;
use rand_distr::{Binomial, Distribution, Gamma, Hypergeometric, StandardNormal};

use super::native_support::{
    dtype, element_count, floats, normalize_axis, shape, write_result, Array,
};
use super::{
    random_stream_mut, ALLOCATION_FAILED, INVALID_ARGUMENT, OK, SIZE_OVERFLOW, UNSUPPORTED_DTYPE,
};

type Stream = *mut super::super::ffi::TerlanNdRandomStream;

/// Returns NumPy's probability-sum tolerance for one floating storage dtype.
fn probability_tolerance(dtype: i32) -> f64 {
    let epsilon = match dtype {
        14 => 2.0_f64.powi(-10),
        11 => f64::from(f32::EPSILON),
        _ => f64::EPSILON,
    };
    epsilon.sqrt()
}

unsafe fn sample_shape(shape_ptr: *const i64, rank: i64) -> Result<(Vec<i64>, usize), i32> {
    if rank < 0 || (rank > 0 && shape_ptr.is_null()) {
        return Err(INVALID_ARGUMENT);
    }
    let rank = usize::try_from(rank).map_err(|_| SIZE_OVERFLOW)?;
    let dimensions = if rank == 0 {
        Vec::new()
    } else {
        // SAFETY: the caller provides exactly `rank` readable dimensions.
        unsafe { std::slice::from_raw_parts(shape_ptr, rank) }.to_vec()
    };
    let count = element_count(&dimensions)?;
    Ok((dimensions, count))
}

fn event_shape(mut sample_shape: Vec<i64>, event_length: usize) -> Result<Vec<i64>, i32> {
    sample_shape.push(i64::try_from(event_length).map_err(|_| SIZE_OVERFLOW)?);
    Ok(sample_shape)
}

unsafe fn owned_integer(values: &[i64], shape: &[i64], dtype: i32) -> Result<Array, i32> {
    if !matches!(dtype, 2 | 4 | 5 | 6 | 7 | 8 | 9 | 10) {
        return Err(UNSUPPORTED_DTYPE);
    }
    let mut output = std::ptr::null_mut();
    let value_count = i64::try_from(values.len()).map_err(|_| SIZE_OVERFLOW)?;
    let rank = i64::try_from(shape.len()).map_err(|_| SIZE_OVERFLOW)?;
    // SAFETY: both slices remain live while the constructor copies them.
    let status = unsafe {
        super::super::ffi::terlan_ndarray_v1_from_ints_as(
            values.as_ptr(),
            value_count,
            shape.as_ptr(),
            rank,
            dtype,
            &mut output,
        )
    };
    if status == OK {
        Ok(output)
    } else {
        Err(status)
    }
}

unsafe fn owned_float(values: &[f64], shape: &[i64], dtype: i32) -> Result<Array, i32> {
    if !matches!(dtype, 3 | 11 | 14) {
        return Err(UNSUPPORTED_DTYPE);
    }
    let mut output = std::ptr::null_mut();
    let value_count = i64::try_from(values.len()).map_err(|_| SIZE_OVERFLOW)?;
    let rank = i64::try_from(shape.len()).map_err(|_| SIZE_OVERFLOW)?;
    // SAFETY: both slices remain live while the constructor copies them.
    let status = unsafe {
        super::super::ffi::terlan_ndarray_v1_from_floats_as(
            values.as_ptr(),
            value_count,
            shape.as_ptr(),
            rank,
            dtype,
            &mut output,
        )
    };
    if status == OK {
        Ok(output)
    } else {
        Err(status)
    }
}

unsafe fn choice_result(
    source: Array,
    indices: &[i64],
    axis: usize,
    output_shape: &[i64],
    output: *mut Array,
) -> i32 {
    let mut selected = std::ptr::null_mut();
    let count = match i64::try_from(indices.len()) {
        Ok(count) => count,
        Err(_) => return SIZE_OVERFLOW,
    };
    // SAFETY: the source and index slice remain live while `take` copies values.
    let status = unsafe {
        super::super::ffi::terlan_ndarray_v1_take(
            source,
            indices.as_ptr(),
            count,
            axis as i64,
            &mut selected,
        )
    };
    if status != OK {
        return status;
    }
    let rank = match i64::try_from(output_shape.len()) {
        Ok(rank) => rank,
        Err(_) => {
            // SAFETY: `selected` was transferred by the successful `take` call.
            let _ = unsafe { super::super::ffi::terlan_ndarray_v1_delete(selected) };
            return SIZE_OVERFLOW;
        }
    };
    // SAFETY: the temporary array and output shape remain live for the copying reshape.
    let status = unsafe {
        super::super::ffi::terlan_ndarray_v1_reshape(selected, output_shape.as_ptr(), rank, output)
    };
    // SAFETY: reshape does not consume the selected temporary.
    let _ = unsafe { super::super::ffi::terlan_ndarray_v1_delete(selected) };
    status
}

/// Replaces one source axis with the requested sample dimensions.
fn choice_output_shape(source: &[i64], sample: &[i64], axis: usize) -> Vec<i64> {
    let mut output = Vec::with_capacity(source.len() - 1 + sample.len());
    output.extend_from_slice(&source[..axis]);
    output.extend_from_slice(sample);
    output.extend_from_slice(&source[axis + 1..]);
    output
}

/// Samples source indices under one normalized axis and probability policy.
unsafe fn sample_choice(
    stream: Stream,
    source: Array,
    output_shape: *const i64,
    output_rank: i64,
    probabilities: Array,
    replacement: i32,
    axis: i64,
    shuffle: i32,
    output: *mut Array,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    // SAFETY: the checked output slot is exclusively owned by this call.
    unsafe { *output = std::ptr::null_mut() };
    if !matches!(replacement, 0 | 1) || !matches!(shuffle, 0 | 1) {
        return INVALID_ARGUMENT;
    }
    let source_shape = match unsafe { shape(source) } {
        Ok(dimensions) if !dimensions.is_empty() => dimensions,
        Ok(_) => return INVALID_ARGUMENT,
        Err(status) => return status,
    };
    let axis = match normalize_axis(axis, source_shape.len()) {
        Ok(axis) => axis,
        Err(status) => return status,
    };
    let source_length = match usize::try_from(source_shape[axis]) {
        Ok(length) => length,
        Err(_) => return SIZE_OVERFLOW,
    };
    let (sample_shape, output_count) = match unsafe { sample_shape(output_shape, output_rank) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if output_count > 0 && source_length == 0 {
        return INVALID_ARGUMENT;
    }
    let weights = if probabilities.is_null() {
        None
    } else {
        let probability_dtype = match unsafe { dtype(probabilities) } {
            Ok(dtype) => dtype,
            Err(status) => return status,
        };
        match unsafe { shape(probabilities) } {
            Ok(dimensions) if dimensions == [source_shape[axis]] => {}
            Ok(_) => return INVALID_ARGUMENT,
            Err(status) => return status,
        }
        match unsafe { floats(probabilities, true) } {
            Ok(values) => {
                let sum: f64 = values.iter().sum();
                let tolerance = probability_tolerance(probability_dtype);
                if values.iter().any(|value| *value < 0.0)
                    || (output_count > 0 && !values.iter().any(|value| *value > 0.0))
                    || (sum - 1.0).abs() > tolerance
                {
                    return INVALID_ARGUMENT;
                }
                Some(values)
            }
            Err(status) => return status,
        }
    };
    if replacement == 0 {
        let selectable = weights.as_ref().map_or(source_length, |values| {
            values.iter().filter(|value| **value > 0.0).count()
        });
        if output_count > selectable {
            return INVALID_ARGUMENT;
        }
    }
    // Sampling stays transactional until the selected array is constructed.
    let stream = match unsafe { random_stream_mut(stream) } {
        Ok(stream) => stream,
        Err(status) => return status,
    };
    let mut generator = stream.generator.clone();
    let mut indices = Vec::new();
    if indices.try_reserve_exact(output_count).is_err() {
        return ALLOCATION_FAILED;
    }
    if replacement == 1 {
        if let Some(weights) = weights {
            let distribution = match WeightedAliasIndex::new(weights) {
                Ok(distribution) => distribution,
                Err(_) => return INVALID_ARGUMENT,
            };
            indices.extend((0..output_count).map(|_| distribution.sample(&mut generator) as i64));
        } else {
            indices
                .extend((0..output_count).map(|_| generator.random_range(0..source_length) as i64));
        }
    } else {
        let mut candidates: Vec<(usize, f64)> = weights.map_or_else(
            || (0..source_length).map(|index| (index, 1.0)).collect(),
            |values| {
                values
                    .into_iter()
                    .enumerate()
                    .filter(|(_, weight)| *weight > 0.0)
                    .collect()
            },
        );
        for _ in 0..output_count {
            let distribution = match WeightedAliasIndex::new(
                candidates.iter().map(|(_, weight)| *weight).collect(),
            ) {
                Ok(distribution) => distribution,
                Err(_) => return INVALID_ARGUMENT,
            };
            let selected = distribution.sample(&mut generator);
            indices.push(candidates.swap_remove(selected).0 as i64);
        }
        if shuffle == 0 {
            indices.sort_unstable();
        }
    }
    let result_shape = choice_output_shape(&source_shape, &sample_shape, axis);
    // SAFETY: all borrowed resources and output storage remain live for the call.
    let status = unsafe { choice_result(source, &indices, axis, &result_shape, output) };
    if status == OK {
        stream.generator = generator;
    }
    status
}

/// Samples values from a rank-one array with optional weighted probabilities.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_choice(
    stream: Stream,
    source: Array,
    output_shape: *const i64,
    output_rank: i64,
    probabilities: Array,
    replacement: i32,
    output: *mut Array,
) -> i32 {
    unsafe {
        sample_choice(
            stream,
            source,
            output_shape,
            output_rank,
            probabilities,
            replacement,
            0,
            1,
            output,
        )
    }
}

/// Samples values along one source axis with explicit output-order policy.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_choice_axis(
    stream: Stream,
    source: Array,
    output_shape: *const i64,
    output_rank: i64,
    probabilities: Array,
    replacement: i32,
    axis: i64,
    shuffle: i32,
    output: *mut Array,
) -> i32 {
    unsafe {
        sample_choice(
            stream,
            source,
            output_shape,
            output_rank,
            probabilities,
            replacement,
            axis,
            shuffle,
            output,
        )
    }
}

/// Samples uniformly along one source axis without a probability array.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_choice_uniform_axis(
    stream: Stream,
    source: Array,
    output_shape: *const i64,
    output_rank: i64,
    replacement: i32,
    axis: i64,
    shuffle: i32,
    output: *mut Array,
) -> i32 {
    unsafe {
        sample_choice(
            stream,
            source,
            output_shape,
            output_rank,
            std::ptr::null_mut(),
            replacement,
            axis,
            shuffle,
            output,
        )
    }
}

/// Samples vectors from a Dirichlet distribution using maintained Gamma samplers.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_dirichlet(
    stream: Stream,
    alpha: Array,
    sample_shape_ptr: *const i64,
    sample_rank: i64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    match unsafe { shape(alpha) } {
        Ok(dimensions) if dimensions.len() == 1 && dimensions[0] > 0 => {}
        Ok(_) => return INVALID_ARGUMENT,
        Err(status) => return status,
    }
    let alphas = match unsafe { floats(alpha, true) } {
        Ok(values) if values.iter().all(|value| *value > 0.0) => values,
        Ok(_) => return INVALID_ARGUMENT,
        Err(status) => return status,
    };
    let samplers: Vec<Gamma<f64>> = match alphas
        .iter()
        .map(|alpha| Gamma::new(*alpha, 1.0))
        .collect::<Result<_, _>>()
    {
        Ok(samplers) => samplers,
        Err(_) => return INVALID_ARGUMENT,
    };
    let (sample_shape, sample_count) = match unsafe { sample_shape(sample_shape_ptr, sample_rank) }
    {
        Ok(value) => value,
        Err(status) => return status,
    };
    let output_shape = match event_shape(sample_shape, alphas.len()) {
        Ok(shape) => shape,
        Err(status) => return status,
    };
    let total = match sample_count.checked_mul(alphas.len()) {
        Some(total) => total,
        None => return SIZE_OVERFLOW,
    };
    let stream = match unsafe { random_stream_mut(stream) } {
        Ok(stream) => stream,
        Err(status) => return status,
    };
    let mut values = Vec::new();
    if values.try_reserve_exact(total).is_err() {
        return ALLOCATION_FAILED;
    }
    for _ in 0..sample_count {
        let start = values.len();
        values.extend(
            samplers
                .iter()
                .map(|sampler| sampler.sample(&mut stream.generator)),
        );
        let sum: f64 = values[start..].iter().sum();
        for value in &mut values[start..] {
            *value /= sum;
        }
    }
    // SAFETY: construction copies both local slices before returning.
    unsafe { write_result(owned_float(&values, &output_shape, dtype), output) }
}

/// Samples category counts from a multinomial distribution.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_multinomial(
    stream: Stream,
    trials: i64,
    probabilities: Array,
    sample_shape_ptr: *const i64,
    sample_rank: i64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    if output.is_null() || trials < 0 {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    let probability_shape = match unsafe { shape(probabilities) } {
        Ok(dimensions) if dimensions.len() == 1 && dimensions[0] > 0 => dimensions,
        Ok(_) => return INVALID_ARGUMENT,
        Err(status) => return status,
    };
    let probabilities = match unsafe { floats(probabilities, true) } {
        Ok(values) if values.iter().all(|value| (0.0..=1.0).contains(value)) => values,
        Ok(_) => return INVALID_ARGUMENT,
        Err(status) => return status,
    };
    let probability_sum: f64 = probabilities.iter().sum();
    if (probability_sum - 1.0).abs() > f64::EPSILON * probabilities.len() as f64 * 8.0 {
        return INVALID_ARGUMENT;
    }
    let (sample_shape, sample_count) = match unsafe { sample_shape(sample_shape_ptr, sample_rank) }
    {
        Ok(value) => value,
        Err(status) => return status,
    };
    let categories = probability_shape[0] as usize;
    let output_shape = match event_shape(sample_shape, categories) {
        Ok(shape) => shape,
        Err(status) => return status,
    };
    let total = match sample_count.checked_mul(categories) {
        Some(total) => total,
        None => return SIZE_OVERFLOW,
    };
    let stream = match unsafe { random_stream_mut(stream) } {
        Ok(stream) => stream,
        Err(status) => return status,
    };
    let mut values = Vec::new();
    if values.try_reserve_exact(total).is_err() {
        return ALLOCATION_FAILED;
    }
    for _ in 0..sample_count {
        let mut remaining_trials = trials as u64;
        let mut remaining_probability = 1.0;
        for &probability in probabilities.iter().take(categories.saturating_sub(1)) {
            let conditional = if remaining_probability <= 0.0 {
                0.0
            } else {
                (probability / remaining_probability).clamp(0.0, 1.0)
            };
            let sampled = Binomial::new(remaining_trials, conditional)
                .expect("validated conditional probability creates binomial")
                .sample(&mut stream.generator);
            values.push(sampled as i64);
            remaining_trials -= sampled;
            remaining_probability -= probability;
        }
        values.push(remaining_trials as i64);
    }
    // SAFETY: construction copies both local slices before returning.
    unsafe { write_result(owned_integer(&values, &output_shape, dtype), output) }
}

/// Samples category counts without replacement from a multivariate population.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_multivariate_hypergeometric(
    stream: Stream,
    colors: Array,
    samples: i64,
    sample_shape_ptr: *const i64,
    sample_rank: i64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    if output.is_null() || samples < 0 {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    let color_shape = match unsafe { shape(colors) } {
        Ok(dimensions) if dimensions.len() == 1 && dimensions[0] > 0 => dimensions,
        Ok(_) => return INVALID_ARGUMENT,
        Err(status) => return status,
    };
    let colors: Vec<u64> = match unsafe { floats(colors, true) } {
        Ok(values)
            if values.iter().all(|value| {
                *value >= 0.0 && value.fract() == 0.0 && *value <= u64::MAX as f64
            }) =>
        {
            values.into_iter().map(|value| value as u64).collect()
        }
        Ok(_) => return INVALID_ARGUMENT,
        Err(status) => return status,
    };
    let population = match colors
        .iter()
        .try_fold(0u64, |sum, color| sum.checked_add(*color))
    {
        Some(population) if samples as u64 <= population => population,
        _ => return INVALID_ARGUMENT,
    };
    let (sample_shape, sample_count) = match unsafe { sample_shape(sample_shape_ptr, sample_rank) }
    {
        Ok(value) => value,
        Err(status) => return status,
    };
    let categories = color_shape[0] as usize;
    let output_shape = match event_shape(sample_shape, categories) {
        Ok(shape) => shape,
        Err(status) => return status,
    };
    let total = match sample_count.checked_mul(categories) {
        Some(total) => total,
        None => return SIZE_OVERFLOW,
    };
    let stream = match unsafe { random_stream_mut(stream) } {
        Ok(stream) => stream,
        Err(status) => return status,
    };
    let mut values = Vec::new();
    if values.try_reserve_exact(total).is_err() {
        return ALLOCATION_FAILED;
    }
    for _ in 0..sample_count {
        let mut remaining_population = population;
        let mut remaining_samples = samples as u64;
        for &color in colors.iter().take(categories.saturating_sub(1)) {
            let sampled = Hypergeometric::new(remaining_population, color, remaining_samples)
                .expect("validated population creates hypergeometric")
                .sample(&mut stream.generator);
            values.push(sampled as i64);
            remaining_samples -= sampled;
            remaining_population -= color;
        }
        values.push(remaining_samples as i64);
    }
    // SAFETY: construction copies both local slices before returning.
    unsafe { write_result(owned_integer(&values, &output_shape, dtype), output) }
}

/// Samples vectors from a multivariate normal distribution using `faer` Cholesky.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_multivariate_normal(
    stream: Stream,
    mean: Array,
    covariance: Array,
    sample_shape_ptr: *const i64,
    sample_rank: i64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    let mean_shape = match unsafe { shape(mean) } {
        Ok(dimensions) if dimensions.len() == 1 && dimensions[0] > 0 => dimensions,
        Ok(_) => return INVALID_ARGUMENT,
        Err(status) => return status,
    };
    let dimensions = mean_shape[0] as usize;
    let covariance_shape = match unsafe { shape(covariance) } {
        Ok(shape) if shape == [mean_shape[0], mean_shape[0]] => shape,
        Ok(_) => return INVALID_ARGUMENT,
        Err(status) => return status,
    };
    let _ = covariance_shape;
    let mean = match unsafe { floats(mean, true) } {
        Ok(values) => values,
        Err(status) => return status,
    };
    let covariance = match unsafe { floats(covariance, true) } {
        Ok(values) => values,
        Err(status) => return status,
    };
    for row in 0..dimensions {
        for column in 0..dimensions {
            let lhs = covariance[row * dimensions + column];
            let rhs = covariance[column * dimensions + row];
            if (lhs - rhs).abs() > 1.0e-12 * lhs.abs().max(rhs.abs()).max(1.0) {
                return INVALID_ARGUMENT;
            }
        }
    }
    let covariance = Mat::from_fn(dimensions, dimensions, |row, column| {
        covariance[row * dimensions + column]
    });
    let factor = match covariance.llt(Side::Lower) {
        Ok(factor) => factor,
        Err(_) => return INVALID_ARGUMENT,
    };
    let lower = factor.L();
    let (sample_shape, sample_count) = match unsafe { sample_shape(sample_shape_ptr, sample_rank) }
    {
        Ok(value) => value,
        Err(status) => return status,
    };
    let output_shape = match event_shape(sample_shape, dimensions) {
        Ok(shape) => shape,
        Err(status) => return status,
    };
    let total = match sample_count.checked_mul(dimensions) {
        Some(total) => total,
        None => return SIZE_OVERFLOW,
    };
    let stream = match unsafe { random_stream_mut(stream) } {
        Ok(stream) => stream,
        Err(status) => return status,
    };
    let mut values = Vec::new();
    if values.try_reserve_exact(total).is_err() {
        return ALLOCATION_FAILED;
    }
    let mut standard = vec![0.0; dimensions];
    for _ in 0..sample_count {
        for value in &mut standard {
            *value = StandardNormal.sample(&mut stream.generator);
        }
        for row in 0..dimensions {
            let transformed = (0..=row)
                .map(|column| lower[(row, column)] * standard[column])
                .sum::<f64>();
            values.push(mean[row] + transformed);
        }
    }
    // SAFETY: construction copies both local slices before returning.
    unsafe { write_result(owned_float(&values, &output_shape, dtype), output) }
}
