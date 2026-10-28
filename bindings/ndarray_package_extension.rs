//! Maintained random-distribution support for `terlan-ndarray`.

#[path = "ndarray_native_support.rs"]
mod native_support;

#[path = "ndarray_linear_algebra_extension.rs"]
mod linear_algebra_extension;
pub use linear_algebra_extension::*;

#[path = "ndarray_linear_algebra_completion.rs"]
mod linear_algebra_completion;
pub use linear_algebra_completion::*;

#[path = "ndarray_linalg_decomposition.rs"]
mod linalg_decomposition;

#[path = "ndarray_linalg_support.rs"]
mod linalg_support;

#[path = "ndarray_linalg_system.rs"]
mod linalg_system;

#[path = "ndarray_linalg_analysis.rs"]
mod linalg_analysis;

#[path = "ndarray_norm_support.rs"]
mod norm_support;

#[path = "ndarray_serialization_extension.rs"]
mod serialization_extension;
pub use serialization_extension::*;

#[path = "ndarray_arrow_extension.rs"]
mod arrow_extension;
pub use arrow_extension::*;

#[path = "ndarray_fft_extension.rs"]
mod fft_extension;
pub use fft_extension::*;

#[path = "ndarray_polynomial_extension.rs"]
mod polynomial_extension;
pub use polynomial_extension::*;

#[path = "ndarray_polynomial_power_extension.rs"]
mod polynomial_power_extension;
pub use polynomial_power_extension::*;

#[path = "ndarray_polynomial_numpy_completion.rs"]
mod polynomial_numpy_completion;
pub use polynomial_numpy_completion::*;

#[path = "ndarray_polynomial_basis_support.rs"]
mod polynomial_basis_support;

#[path = "ndarray_polynomial_basis_numpy_completion.rs"]
mod polynomial_basis_numpy_completion;
pub use polynomial_basis_numpy_completion::*;

#[path = "ndarray_polynomial_chebyshev_extension.rs"]
mod polynomial_chebyshev_extension;
pub use polynomial_chebyshev_extension::*;

#[path = "ndarray_polynomial_legendre_extension.rs"]
mod polynomial_legendre_extension;
pub use polynomial_legendre_extension::*;

#[path = "ndarray_numpy_numeric_extension.rs"]
mod numpy_numeric_extension;
pub use numpy_numeric_extension::*;

#[path = "ndarray_numpy_average_extension.rs"]
mod numpy_average_extension;
pub use numpy_average_extension::*;

#[path = "ndarray_float_spacing_support.rs"]
mod float_spacing_support;

#[path = "ndarray_numpy_comparison_extension.rs"]
mod numpy_comparison_extension;
pub use numpy_comparison_extension::*;

#[path = "ndarray_numpy_data_extension.rs"]
mod numpy_data_extension;
pub use numpy_data_extension::*;

#[path = "ndarray_type_check_extension.rs"]
mod type_check_extension;
pub use type_check_extension::*;

#[path = "ndarray_numpy_shape_extension.rs"]
mod numpy_shape_extension;
pub use numpy_shape_extension::*;

#[path = "ndarray_creation_sparse_extension.rs"]
mod creation_sparse_extension;
pub use creation_sparse_extension::*;

#[cfg(test)]
#[path = "ndarray_stride_tricks_tests.rs"]
mod stride_tricks_tests;

#[cfg(test)]
#[path = "ndarray_regression_tests.rs"]
mod regression_tests;

#[cfg(test)]
#[path = "ndarray_reduction_option_tests.rs"]
mod reduction_option_tests;

#[cfg(test)]
#[path = "ndarray_index_tricks_tests.rs"]
mod index_tricks_tests;

#[cfg(test)]
#[path = "ndarray_memory_overlap_tests.rs"]
mod memory_overlap_tests;

#[cfg(test)]
#[path = "ndarray_function_base_tests.rs"]
mod function_base_tests;

#[cfg(test)]
#[path = "ndarray_scalar_math_tests.rs"]
mod scalar_math_tests;

#[cfg(test)]
#[path = "ndarray_numeric_core_tests.rs"]
mod numeric_core_tests;

#[cfg(test)]
#[path = "ndarray_nonzero_tests.rs"]
mod nonzero_tests;

#[cfg(test)]
#[path = "ndarray_bool_numeric_tests.rs"]
mod bool_numeric_tests;

#[cfg(test)]
#[path = "ndarray_umath_parity_tests.rs"]
mod umath_parity_tests;

#[cfg(test)]
#[path = "ndarray_umath_boundary_tests.rs"]
mod umath_boundary_tests;

#[path = "ndarray_numpy_block_extension.rs"]
mod numpy_block_extension;
pub use numpy_block_extension::*;

#[path = "ndarray_numpy_index_extension.rs"]
mod numpy_index_extension;
pub use numpy_index_extension::*;

#[path = "ndarray_index_tricks_extension.rs"]
mod index_tricks_extension;
pub use index_tricks_extension::*;

#[path = "ndarray_histogram2d_extension.rs"]
mod histogram2d_extension;
pub use histogram2d_extension::*;

#[path = "ndarray_histogram_extension.rs"]
mod histogram_extension;
pub use histogram_extension::*;

#[path = "ndarray_histogramdd_extension.rs"]
mod histogramdd_extension;
pub use histogramdd_extension::*;

#[path = "ndarray_tensor_algebra_extension.rs"]
mod tensor_algebra_extension;
pub use tensor_algebra_extension::*;

#[path = "ndarray_shape_extension.rs"]
mod shape_extension;
pub use shape_extension::*;

#[path = "ndarray_random_distribution_extension.rs"]
mod random_distribution_extension;
pub use random_distribution_extension::*;

#[path = "ndarray_random_broadcast_extension.rs"]
mod random_broadcast_extension;
pub use random_broadcast_extension::*;

#[path = "ndarray_random_structured_extension.rs"]
mod random_structured_extension;
pub use random_structured_extension::*;

#[path = "ndarray_random_output_extension.rs"]
mod random_output_extension;
pub use random_output_extension::*;

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

const OK: i32 = 0;
const INVALID_ARGUMENT: i32 = 100;
const NEGATIVE_DIMENSION: i32 = 101;
const SIZE_OVERFLOW: i32 = 103;
const ALLOCATION_FAILED: i32 = 104;
const UNSUPPORTED_DTYPE: i32 = 109;

/// Stateful deterministic random stream owned by the ndarray package.
pub(crate) struct RandomStream {
    pub(crate) generator: ChaCha8Rng,
}

pub(crate) unsafe fn random_stream_mut<'a>(
    stream: *mut super::ffi::TerlanNdRandomStream,
) -> Result<&'a mut RandomStream, i32> {
    if stream.is_null() {
        return Err(INVALID_ARGUMENT);
    }
    // SAFETY: generated resource ownership admits only live handles created below.
    Ok(unsafe { &mut *stream.cast::<RandomStream>() })
}

fn checked_element_count(shape: &[i64]) -> Result<usize, i32> {
    let mut count = 1usize;
    for &dimension in shape {
        let dimension = usize::try_from(dimension).map_err(|_| NEGATIVE_DIMENSION)?;
        count = count.checked_mul(dimension).ok_or(SIZE_OVERFLOW)?;
        if count > i64::MAX as usize {
            return Err(SIZE_OVERFLOW);
        }
    }
    Ok(count)
}

pub(crate) fn random_output(
    shape: *const i64,
    rank: i64,
    dtype: i32,
    out_array: *mut *mut super::ffi::TerlanNdArray,
    sample: impl FnMut() -> f64,
) -> i32 {
    if out_array.is_null() || rank < 0 || (rank > 0 && shape.is_null()) {
        return INVALID_ARGUMENT;
    }
    // SAFETY: `out_array` was checked and is exclusively owned for this call.
    unsafe { *out_array = std::ptr::null_mut() };
    if !matches!(dtype, 3 | 11 | 14) {
        return UNSUPPORTED_DTYPE;
    }
    let rank = match usize::try_from(rank) {
        Ok(rank) => rank,
        Err(_) => return SIZE_OVERFLOW,
    };
    let shape_slice = if rank == 0 {
        &[]
    } else {
        // SAFETY: positive-rank calls require `shape` to reference `rank` dimensions.
        unsafe { std::slice::from_raw_parts(shape, rank) }
    };
    let count = match checked_element_count(shape_slice) {
        Ok(count) => count,
        Err(status) => return status,
    };
    let mut values = Vec::new();
    if values.try_reserve_exact(count).is_err() {
        return ALLOCATION_FAILED;
    }
    values.extend(std::iter::repeat_with(sample).take(count));
    let value_count = match i64::try_from(count) {
        Ok(count) => count,
        Err(_) => return SIZE_OVERFLOW,
    };
    // SAFETY: all slices remain live for the call and the C constructor copies them.
    unsafe {
        super::ffi::terlan_ndarray_v1_from_floats_as(
            values.as_ptr(),
            value_count,
            shape,
            rank as i64,
            dtype,
            out_array,
        )
    }
}

pub(crate) fn random_checked_integer_output(
    shape: *const i64,
    rank: i64,
    dtype: i32,
    out_array: *mut *mut super::ffi::TerlanNdArray,
    mut sample: impl FnMut() -> Result<i64, i32>,
) -> i32 {
    if out_array.is_null() || rank < 0 || (rank > 0 && shape.is_null()) {
        return INVALID_ARGUMENT;
    }
    // SAFETY: `out_array` was checked and is exclusively owned for this call.
    unsafe { *out_array = std::ptr::null_mut() };
    if !matches!(dtype, 2 | 4 | 5 | 6 | 7 | 8 | 9 | 10) {
        return UNSUPPORTED_DTYPE;
    }
    let rank = match usize::try_from(rank) {
        Ok(rank) => rank,
        Err(_) => return SIZE_OVERFLOW,
    };
    let shape_slice = if rank == 0 {
        &[]
    } else {
        // SAFETY: positive-rank calls require `shape` to reference `rank` dimensions.
        unsafe { std::slice::from_raw_parts(shape, rank) }
    };
    let count = match checked_element_count(shape_slice) {
        Ok(count) => count,
        Err(status) => return status,
    };
    let mut values = Vec::new();
    if values.try_reserve_exact(count).is_err() {
        return ALLOCATION_FAILED;
    }
    for _ in 0..count {
        match sample() {
            Ok(value) => values.push(value),
            Err(status) => return status,
        }
    }
    let value_count = match i64::try_from(count) {
        Ok(count) => count,
        Err(_) => return SIZE_OVERFLOW,
    };
    // SAFETY: all slices remain live for the call and the C constructor copies them.
    unsafe {
        super::ffi::terlan_ndarray_v1_from_ints_as(
            values.as_ptr(),
            value_count,
            shape,
            rank as i64,
            dtype,
            out_array,
        )
    }
}

pub(crate) fn random_integer_output(
    shape: *const i64,
    rank: i64,
    dtype: i32,
    out_array: *mut *mut super::ffi::TerlanNdArray,
    mut sample: impl FnMut() -> i64,
) -> i32 {
    random_checked_integer_output(shape, rank, dtype, out_array, || Ok(sample()))
}

fn random_bool_output(
    shape: *const i64,
    rank: i64,
    out_array: *mut *mut super::ffi::TerlanNdArray,
    sample: impl FnMut() -> bool,
) -> i32 {
    if out_array.is_null() || rank < 0 || (rank > 0 && shape.is_null()) {
        return INVALID_ARGUMENT;
    }
    // SAFETY: `out_array` was checked and is exclusively owned for this call.
    unsafe { *out_array = std::ptr::null_mut() };
    let rank = match usize::try_from(rank) {
        Ok(rank) => rank,
        Err(_) => return SIZE_OVERFLOW,
    };
    // SAFETY: the ABI requires `shape` to reference `rank` readable dimensions.
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let count = match checked_element_count(shape_slice) {
        Ok(count) => count,
        Err(status) => return status,
    };
    let mut values = Vec::new();
    if values.try_reserve_exact(count).is_err() {
        return ALLOCATION_FAILED;
    }
    values.extend(std::iter::repeat_with(sample).take(count).map(u8::from));
    let value_count = match i64::try_from(count) {
        Ok(count) => count,
        Err(_) => return SIZE_OVERFLOW,
    };
    // SAFETY: all slices remain live for the call and the C constructor copies them.
    unsafe {
        super::ffi::terlan_ndarray_v1_from_bools(
            values.as_ptr(),
            value_count,
            shape,
            rank as i64,
            out_array,
        )
    }
}

/// Creates reproducible uniform samples in the half-open interval `[low, high)`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_uniform(
    shape: *const i64,
    rank: i64,
    low: f64,
    high: f64,
    seed: i64,
    dtype: i32,
    out_array: *mut *mut super::ffi::TerlanNdArray,
) -> i32 {
    if !(high - low).is_finite() || low > high {
        return INVALID_ARGUMENT;
    }
    let mut generator = ChaCha8Rng::seed_from_u64(seed as u64);
    random_output(shape, rank, dtype, out_array, || {
        low + (high - low) * generator.random::<f64>()
    })
}

/// Creates reproducible samples from a normal distribution.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_normal(
    shape: *const i64,
    rank: i64,
    mean: f64,
    standard_deviation: f64,
    seed: i64,
    dtype: i32,
    out_array: *mut *mut super::ffi::TerlanNdArray,
) -> i32 {
    if standard_deviation.is_nan() || standard_deviation < 0.0 {
        return INVALID_ARGUMENT;
    }
    let mut generator = ChaCha8Rng::seed_from_u64(seed as u64);
    random_output(shape, rank, dtype, out_array, || {
        let sample: f64 = StandardNormal.sample(&mut generator);
        mean + standard_deviation * sample
    })
}

/// Creates reproducible integer samples in the half-open interval `[low, high)`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_integers(
    shape: *const i64,
    rank: i64,
    low: i64,
    high: i64,
    seed: i64,
    dtype: i32,
    out_array: *mut *mut super::ffi::TerlanNdArray,
) -> i32 {
    if low >= high {
        if !out_array.is_null() {
            // SAFETY: the non-null output slot is exclusively owned by this call.
            unsafe { *out_array = std::ptr::null_mut() };
        }
        return INVALID_ARGUMENT;
    }
    let mut generator = ChaCha8Rng::seed_from_u64(seed as u64);
    random_integer_output(shape, rank, dtype, out_array, || {
        generator.random_range(low..high)
    })
}

/// Creates reproducible Boolean samples with probability `probability` of true.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_bernoulli(
    shape: *const i64,
    rank: i64,
    probability: f64,
    seed: i64,
    out_array: *mut *mut super::ffi::TerlanNdArray,
) -> i32 {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        if !out_array.is_null() {
            // SAFETY: the non-null output slot is exclusively owned by this call.
            unsafe { *out_array = std::ptr::null_mut() };
        }
        return INVALID_ARGUMENT;
    }
    let mut generator = ChaCha8Rng::seed_from_u64(seed as u64);
    random_bool_output(shape, rank, out_array, || {
        generator.random_bool(probability)
    })
}

/// Creates a reproducible permutation of integers from zero to `length - 1`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_permutation(
    length: i64,
    seed: i64,
    out_array: *mut *mut super::ffi::TerlanNdArray,
) -> i32 {
    if out_array.is_null() || length < 0 {
        return INVALID_ARGUMENT;
    }
    // SAFETY: `out_array` was checked and is exclusively owned for this call.
    unsafe { *out_array = std::ptr::null_mut() };
    let count = match usize::try_from(length) {
        Ok(count) => count,
        Err(_) => return SIZE_OVERFLOW,
    };
    let mut values = Vec::new();
    if values.try_reserve_exact(count).is_err() {
        return ALLOCATION_FAILED;
    }
    values.extend(0..length);
    let mut generator = ChaCha8Rng::seed_from_u64(seed as u64);
    values.shuffle(&mut generator);
    // SAFETY: `values` and the one-dimensional shape remain live for the copying call.
    unsafe {
        super::ffi::terlan_ndarray_v1_from_ints(values.as_ptr(), length, &length, 1, out_array)
    }
}

/// Returns an independently owned array shuffled along one normalized axis.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_shuffle(
    array: *mut super::ffi::TerlanNdArray,
    axis: i64,
    seed: i64,
    out_array: *mut *mut super::ffi::TerlanNdArray,
) -> i32 {
    if array.is_null() || out_array.is_null() {
        return INVALID_ARGUMENT;
    }
    // SAFETY: `out_array` was checked and is exclusively owned for this call.
    unsafe { *out_array = std::ptr::null_mut() };
    let mut rank = 0i64;
    // SAFETY: `array` is a borrowed live handle and `rank` is a writable output slot.
    let status = unsafe { super::ffi::terlan_ndarray_v1_rank(array, &mut rank) };
    if status != OK || rank == 0 || axis < -rank || axis >= rank {
        return if status != OK {
            status
        } else {
            INVALID_ARGUMENT
        };
    }
    let normalized = if axis < 0 { axis + rank } else { axis };
    let mut shape = std::ptr::null_mut();
    // SAFETY: `array` is live and owns the borrowed shape for this entire call.
    let status = unsafe { super::ffi::terlan_ndarray_v1_shape(array, &mut shape) };
    if status != OK {
        return status;
    }
    // SAFETY: successful shape access yields `rank` readable dimensions.
    let length = unsafe { *shape.add(normalized as usize) };
    let count = match usize::try_from(length) {
        Ok(count) => count,
        Err(_) => return SIZE_OVERFLOW,
    };
    let mut indices = Vec::new();
    if indices.try_reserve_exact(count).is_err() {
        return ALLOCATION_FAILED;
    }
    indices.extend(0..length);
    let mut generator = ChaCha8Rng::seed_from_u64(seed as u64);
    indices.shuffle(&mut generator);
    // SAFETY: the borrowed array and index slice remain live for the copying call.
    unsafe {
        super::ffi::terlan_ndarray_v1_take(array, indices.as_ptr(), length, normalized, out_array)
    }
}

/// Creates an owned deterministic ChaCha8 stream at the supplied seed.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_new(
    seed: i64,
    out_stream: *mut *mut super::ffi::TerlanNdRandomStream,
) -> i32 {
    if out_stream.is_null() {
        return INVALID_ARGUMENT;
    }
    let stream = Box::new(RandomStream {
        generator: ChaCha8Rng::seed_from_u64(seed as u64),
    });
    // SAFETY: the checked output slot receives ownership of one boxed stream.
    unsafe { *out_stream = Box::into_raw(stream).cast() };
    OK
}

/// Disposes one owned deterministic random stream.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_delete(
    stream: *mut super::ffi::TerlanNdRandomStream,
) -> i32 {
    if stream.is_null() {
        return INVALID_ARGUMENT;
    }
    // SAFETY: generated ownership transfers each stream to this destructor once.
    drop(unsafe { Box::from_raw(stream.cast::<RandomStream>()) });
    OK
}

/// Advances a stream while creating uniform floating samples.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_uniform(
    stream: *mut super::ffi::TerlanNdRandomStream,
    shape: *const i64,
    rank: i64,
    low: f64,
    high: f64,
    dtype: i32,
    out_array: *mut *mut super::ffi::TerlanNdArray,
) -> i32 {
    if !low.is_finite() || !high.is_finite() || low >= high {
        return INVALID_ARGUMENT;
    }
    let stream = match unsafe { random_stream_mut(stream) } {
        Ok(stream) => stream,
        Err(status) => return status,
    };
    random_output(shape, rank, dtype, out_array, || {
        low + (high - low) * stream.generator.random::<f64>()
    })
}

/// Advances a stream while creating normally distributed floating samples.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_normal(
    stream: *mut super::ffi::TerlanNdRandomStream,
    shape: *const i64,
    rank: i64,
    mean: f64,
    standard_deviation: f64,
    dtype: i32,
    out_array: *mut *mut super::ffi::TerlanNdArray,
) -> i32 {
    if !mean.is_finite() || !standard_deviation.is_finite() || standard_deviation < 0.0 {
        return INVALID_ARGUMENT;
    }
    let stream = match unsafe { random_stream_mut(stream) } {
        Ok(stream) => stream,
        Err(status) => return status,
    };
    random_output(shape, rank, dtype, out_array, || {
        let sample: f64 = StandardNormal.sample(&mut stream.generator);
        mean + standard_deviation * sample
    })
}

/// Advances a stream while creating bounded integer samples.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_integers(
    stream: *mut super::ffi::TerlanNdRandomStream,
    shape: *const i64,
    rank: i64,
    low: i64,
    high: i64,
    dtype: i32,
    out_array: *mut *mut super::ffi::TerlanNdArray,
) -> i32 {
    if low >= high {
        return INVALID_ARGUMENT;
    }
    let stream = match unsafe { random_stream_mut(stream) } {
        Ok(stream) => stream,
        Err(status) => return status,
    };
    random_integer_output(shape, rank, dtype, out_array, || {
        stream.generator.random_range(low..high)
    })
}

/// Advances a stream while creating Bernoulli samples.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_bernoulli(
    stream: *mut super::ffi::TerlanNdRandomStream,
    shape: *const i64,
    rank: i64,
    probability: f64,
    out_array: *mut *mut super::ffi::TerlanNdArray,
) -> i32 {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return INVALID_ARGUMENT;
    }
    let stream = match unsafe { random_stream_mut(stream) } {
        Ok(stream) => stream,
        Err(status) => return status,
    };
    random_bool_output(shape, rank, out_array, || {
        stream.generator.random_bool(probability)
    })
}

/// Advances a stream while creating a permutation of consecutive integers.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_permutation(
    stream: *mut super::ffi::TerlanNdRandomStream,
    length: i64,
    out_array: *mut *mut super::ffi::TerlanNdArray,
) -> i32 {
    if out_array.is_null() || length < 0 {
        return INVALID_ARGUMENT;
    }
    // SAFETY: the checked output slot is exclusively owned by this call.
    unsafe { *out_array = std::ptr::null_mut() };
    let stream = match unsafe { random_stream_mut(stream) } {
        Ok(stream) => stream,
        Err(status) => return status,
    };
    let count = match usize::try_from(length) {
        Ok(count) => count,
        Err(_) => return SIZE_OVERFLOW,
    };
    let mut values = Vec::new();
    if values.try_reserve_exact(count).is_err() {
        return ALLOCATION_FAILED;
    }
    values.extend(0..length);
    values.shuffle(&mut stream.generator);
    // SAFETY: values and shape remain live for the copying constructor.
    unsafe {
        super::ffi::terlan_ndarray_v1_from_ints(values.as_ptr(), length, &length, 1, out_array)
    }
}

/// Advances a stream while returning a shuffled independent array.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_shuffle(
    stream: *mut super::ffi::TerlanNdRandomStream,
    array: *mut super::ffi::TerlanNdArray,
    axis: i64,
    out_array: *mut *mut super::ffi::TerlanNdArray,
) -> i32 {
    if array.is_null() || out_array.is_null() {
        return INVALID_ARGUMENT;
    }
    // SAFETY: the checked output slot is exclusively owned by this call.
    unsafe { *out_array = std::ptr::null_mut() };
    let stream = match unsafe { random_stream_mut(stream) } {
        Ok(stream) => stream,
        Err(status) => return status,
    };
    let mut rank = 0i64;
    // SAFETY: array is live and rank is writable.
    let status = unsafe { super::ffi::terlan_ndarray_v1_rank(array, &mut rank) };
    if status != OK || rank == 0 || axis < -rank || axis >= rank {
        return if status != OK {
            status
        } else {
            INVALID_ARGUMENT
        };
    }
    let normalized = if axis < 0 { axis + rank } else { axis };
    let mut shape = std::ptr::null_mut();
    // SAFETY: array owns the borrowed shape for this call.
    let status = unsafe { super::ffi::terlan_ndarray_v1_shape(array, &mut shape) };
    if status != OK {
        return status;
    }
    // SAFETY: shape has rank readable dimensions.
    let length = unsafe { *shape.add(normalized as usize) };
    let count = match usize::try_from(length) {
        Ok(count) => count,
        Err(_) => return SIZE_OVERFLOW,
    };
    let mut indices = Vec::new();
    if indices.try_reserve_exact(count).is_err() {
        return ALLOCATION_FAILED;
    }
    indices.extend(0..length);
    indices.shuffle(&mut stream.generator);
    // SAFETY: source and indices remain live for the copying call.
    unsafe {
        super::ffi::terlan_ndarray_v1_take(array, indices.as_ptr(), length, normalized, out_array)
    }
}

#[cfg(test)]
#[path = "ndarray_random_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "ndarray_random_distribution_tests.rs"]
mod distribution_tests;

#[cfg(test)]
#[path = "ndarray_random_broadcast_tests.rs"]
mod broadcast_distribution_tests;

#[cfg(test)]
#[path = "ndarray_random_unsigned_tests.rs"]
mod unsigned_distribution_tests;

#[cfg(test)]
#[path = "ndarray_random_power_tests.rs"]
mod power_distribution_tests;

#[cfg(test)]
#[path = "ndarray_random_structured_tests.rs"]
mod structured_distribution_tests;

#[cfg(test)]
#[path = "ndarray_random_regression_tests.rs"]
mod random_regression_tests;
