//! Maintained `rustfft` transforms for `terlan-ndarray` CPU arrays.

use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

#[path = "ndarray_fft_resize.rs"]
mod resize;
use resize::resize_complex_axis;

const OK: i32 = 0;
const INVALID_ARGUMENT: i32 = 100;
const SIZE_OVERFLOW: i32 = 103;
const SHAPE_MISMATCH: i32 = 108;
const UNSUPPORTED_DTYPE: i32 = 109;
const AXIS_OUT_OF_RANGE: i32 = 110;
const DOMAIN_ERROR: i32 = 126;

type Array = *mut super::super::ffi::TerlanNdArray;

unsafe fn array_shape(array: Array) -> Result<Vec<i64>, i32> {
    if array.is_null() {
        return Err(INVALID_ARGUMENT);
    }
    let mut rank = 0i64;
    // SAFETY: the array is borrowed and the rank slot is writable.
    let status = unsafe { super::super::ffi::terlan_ndarray_v1_rank(array, &mut rank) };
    if status != OK {
        return Err(status);
    }
    let rank = usize::try_from(rank).map_err(|_| SHAPE_MISMATCH)?;
    let mut shape = std::ptr::null_mut();
    // SAFETY: a live array exposes a borrowed shape with exactly `rank` entries.
    let status = unsafe { super::super::ffi::terlan_ndarray_v1_shape(array, &mut shape) };
    if status != OK {
        return Err(status);
    }
    if rank == 0 {
        return Ok(Vec::new());
    }
    if shape.is_null() {
        return Err(INVALID_ARGUMENT);
    }
    // SAFETY: successful shape lookup guarantees `rank` readable dimensions.
    Ok(unsafe { std::slice::from_raw_parts(shape, rank) }.to_vec())
}

unsafe fn array_dtype(array: Array) -> Result<i32, i32> {
    let mut dtype = 0;
    // SAFETY: the array is borrowed and the dtype slot is writable.
    let status = unsafe { super::super::ffi::terlan_ndarray_v1_dtype(array, &mut dtype) };
    if status == OK {
        Ok(dtype)
    } else {
        Err(status)
    }
}

unsafe fn copied_floats(array: Array) -> Result<Vec<f64>, i32> {
    let mut values = std::ptr::null_mut();
    let mut length = 0usize;
    // SAFETY: the C ABI initializes an owned f64 allocation on success.
    let status = unsafe {
        super::super::ffi::terlan_ndarray_v1_to_floats_cast(array, &mut values, &mut length)
    };
    if status != OK {
        return Err(status);
    }
    let output = if length == 0 {
        Vec::new()
    } else if values.is_null() {
        return Err(INVALID_ARGUMENT);
    } else {
        // SAFETY: the allocation contains exactly `length` initialized values.
        unsafe { std::slice::from_raw_parts(values, length) }.to_vec()
    };
    // SAFETY: this is the matching destructor for the returned allocation.
    unsafe { super::super::ffi::terlan_ndarray_v1_delete_float_values(values) };
    Ok(output)
}

unsafe fn copied_complex(array: Array) -> Result<Vec<Complex64>, i32> {
    let dtype = unsafe { array_dtype(array)? };
    if matches!(dtype, 12 | 13) {
        let mut reals = std::ptr::null_mut();
        let mut imags = std::ptr::null_mut();
        let mut real_length = 0usize;
        let mut imag_length = 0usize;
        // SAFETY: both calls initialize owned f64 arrays on success.
        let real_status = unsafe {
            super::super::ffi::terlan_ndarray_v1_complex_reals(array, &mut reals, &mut real_length)
        };
        let imag_status = if real_status == OK {
            unsafe {
                super::super::ffi::terlan_ndarray_v1_complex_imags(
                    array,
                    &mut imags,
                    &mut imag_length,
                )
            }
        } else {
            real_status
        };
        if real_status != OK || imag_status != OK || real_length != imag_length {
            unsafe {
                super::super::ffi::terlan_ndarray_v1_delete_float_values(reals);
                super::super::ffi::terlan_ndarray_v1_delete_float_values(imags);
            }
            return Err(if real_status != OK {
                real_status
            } else {
                imag_status
            });
        }
        let real_slice = if real_length == 0 {
            &[][..]
        } else {
            // SAFETY: successful extraction returned `real_length` initialized values.
            unsafe { std::slice::from_raw_parts(reals, real_length) }
        };
        let imag_slice = if imag_length == 0 {
            &[][..]
        } else {
            // SAFETY: successful extraction returned `imag_length` initialized values.
            unsafe { std::slice::from_raw_parts(imags, imag_length) }
        };
        let output = real_slice
            .iter()
            .zip(imag_slice)
            .map(|(&re, &im)| Complex64::new(re, im))
            .collect();
        unsafe {
            super::super::ffi::terlan_ndarray_v1_delete_float_values(reals);
            super::super::ffi::terlan_ndarray_v1_delete_float_values(imags);
        }
        Ok(output)
    } else {
        Ok(unsafe { copied_floats(array)? }
            .into_iter()
            .map(|value| Complex64::new(value, 0.0))
            .collect())
    }
}

fn normalize_axis(axis: i64, rank: usize) -> Result<usize, i32> {
    let rank = i64::try_from(rank).map_err(|_| AXIS_OUT_OF_RANGE)?;
    let axis = if axis < 0 { axis + rank } else { axis };
    if axis < 0 || axis >= rank {
        Err(AXIS_OUT_OF_RANGE)
    } else {
        usize::try_from(axis).map_err(|_| AXIS_OUT_OF_RANGE)
    }
}

fn normalized_axes(axes: *const i64, count: i64, rank: usize) -> Result<Vec<usize>, i32> {
    if count < 0 || (count > 0 && axes.is_null()) {
        return Err(INVALID_ARGUMENT);
    }
    let input = if count == 0 {
        return Ok((0..rank).collect());
    } else {
        let count = usize::try_from(count).map_err(|_| SIZE_OVERFLOW)?;
        // SAFETY: the ABI requires `axes` to expose exactly `count` values.
        unsafe { std::slice::from_raw_parts(axes, count) }
    };
    let mut output = Vec::with_capacity(input.len());
    for &axis in input {
        let axis = normalize_axis(axis, rank)?;
        if output.contains(&axis) {
            return Err(INVALID_ARGUMENT);
        }
        output.push(axis);
    }
    Ok(output)
}

fn normalization_scale(length: usize, inverse: bool, normalization: i32) -> Result<f64, i32> {
    if length == 0 || !(0..=2).contains(&normalization) {
        return Err(if length == 0 {
            DOMAIN_ERROR
        } else {
            INVALID_ARGUMENT
        });
    }
    let length = length as f64;
    Ok(match normalization {
        0 if inverse => 1.0 / length,
        1 => 1.0 / length.sqrt(),
        2 if !inverse => 1.0 / length,
        _ => 1.0,
    })
}

fn transform_axis(
    data: &mut [Complex64],
    shape: &[i64],
    axis: usize,
    inverse: bool,
    normalization: i32,
) -> Result<(), i32> {
    let length = usize::try_from(shape[axis]).map_err(|_| SHAPE_MISMATCH)?;
    let scale = normalization_scale(length, inverse, normalization)?;
    let outer = shape[..axis].iter().try_fold(1usize, |value, &dimension| {
        value
            .checked_mul(usize::try_from(dimension).map_err(|_| SHAPE_MISMATCH)?)
            .ok_or(SIZE_OVERFLOW)
    })?;
    let inner = shape[axis + 1..]
        .iter()
        .try_fold(1usize, |value, &dimension| {
            value
                .checked_mul(usize::try_from(dimension).map_err(|_| SHAPE_MISMATCH)?)
                .ok_or(SIZE_OVERFLOW)
        })?;
    let mut planner = FftPlanner::<f64>::new();
    let transform = if inverse {
        planner.plan_fft_inverse(length)
    } else {
        planner.plan_fft_forward(length)
    };
    let mut lane = vec![Complex64::new(0.0, 0.0); length];
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            for (axis_index, value) in lane.iter_mut().enumerate() {
                *value = data[(outer_index * length + axis_index) * inner + inner_index];
            }
            transform.process(&mut lane);
            for (axis_index, value) in lane.iter().enumerate() {
                data[(outer_index * length + axis_index) * inner + inner_index] = *value * scale;
            }
        }
    }
    Ok(())
}

unsafe fn owned_complex(data: &[Complex64], shape: &[i64], dtype: i32) -> Result<Array, i32> {
    let reals: Vec<f64> = data.iter().map(|value| value.re).collect();
    let imags: Vec<f64> = data.iter().map(|value| value.im).collect();
    let mut output = std::ptr::null_mut();
    let count = i64::try_from(data.len()).map_err(|_| SIZE_OVERFLOW)?;
    let rank = i64::try_from(shape.len()).map_err(|_| SIZE_OVERFLOW)?;
    // SAFETY: the C constructor copies all supplied slices before returning.
    let status = unsafe {
        super::super::ffi::terlan_ndarray_v1_from_complex(
            reals.as_ptr(),
            count,
            imags.as_ptr(),
            count,
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

unsafe fn owned_real(data: &[f64], shape: &[i64], dtype: i32) -> Result<Array, i32> {
    let mut output = std::ptr::null_mut();
    let count = i64::try_from(data.len()).map_err(|_| SIZE_OVERFLOW)?;
    let rank = i64::try_from(shape.len()).map_err(|_| SIZE_OVERFLOW)?;
    // SAFETY: the C constructor copies all supplied slices before returning.
    let status = unsafe {
        super::super::ffi::terlan_ndarray_v1_from_floats_as(
            data.as_ptr(),
            count,
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

unsafe fn complex_transform(
    array: Array,
    axes: *const i64,
    axis_count: i64,
    length: Option<i64>,
    inverse: bool,
    normalization: i32,
    out_array: *mut Array,
) -> i32 {
    if out_array.is_null() {
        return INVALID_ARGUMENT;
    }
    // SAFETY: the checked output slot is exclusively owned by this call.
    unsafe { *out_array = std::ptr::null_mut() };
    let mut shape = match unsafe { array_shape(array) } {
        Ok(shape) => shape,
        Err(status) => return status,
    };
    let axes = match normalized_axes(axes, axis_count, shape.len()) {
        Ok(axes) => axes,
        Err(status) => return status,
    };
    let mut data = match unsafe { copied_complex(array) } {
        Ok(data) => data,
        Err(status) => return status,
    };
    if let Some(length) = length {
        if axes.len() != 1 {
            return INVALID_ARGUMENT;
        }
        data = match resize_complex_axis(data, &mut shape, axes[0], length) {
            Ok(data) => data,
            Err(status) => return status,
        };
    }
    for axis in axes {
        if let Err(status) = transform_axis(&mut data, &shape, axis, inverse, normalization) {
            return status;
        }
    }
    if let Err(status) = unsafe { array_dtype(array) } {
        return status;
    }
    match unsafe { owned_complex(&data, &shape, 13) } {
        Ok(output) => {
            unsafe { *out_array = output };
            OK
        }
        Err(status) => status,
    }
}

/// Applies a one-dimensional complex FFT along one normalized axis.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_fft(
    array: Array,
    axis: i64,
    normalization: i32,
    out_array: *mut Array,
) -> i32 {
    unsafe { complex_transform(array, &axis, 1, None, false, normalization, out_array) }
}

/// Applies a length-adjusted one-dimensional complex FFT.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_fft_sized(
    array: Array,
    length: i64,
    axis: i64,
    normalization: i32,
    out_array: *mut Array,
) -> i32 {
    unsafe {
        complex_transform(
            array,
            &axis,
            1,
            Some(length),
            false,
            normalization,
            out_array,
        )
    }
}

/// Applies a one-dimensional inverse complex FFT along one normalized axis.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_ifft(
    array: Array,
    axis: i64,
    normalization: i32,
    out_array: *mut Array,
) -> i32 {
    unsafe { complex_transform(array, &axis, 1, None, true, normalization, out_array) }
}

/// Applies a length-adjusted one-dimensional inverse complex FFT.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_ifft_sized(
    array: Array,
    length: i64,
    axis: i64,
    normalization: i32,
    out_array: *mut Array,
) -> i32 {
    unsafe {
        complex_transform(
            array,
            &axis,
            1,
            Some(length),
            true,
            normalization,
            out_array,
        )
    }
}

/// Applies complex FFTs across normalized distinct axes.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_fftn(
    array: Array,
    axes: *const i64,
    axis_count: i64,
    normalization: i32,
    out_array: *mut Array,
) -> i32 {
    unsafe {
        complex_transform(
            array,
            axes,
            axis_count,
            None,
            false,
            normalization,
            out_array,
        )
    }
}

/// Applies inverse complex FFTs across normalized distinct axes.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_ifftn(
    array: Array,
    axes: *const i64,
    axis_count: i64,
    normalization: i32,
    out_array: *mut Array,
) -> i32 {
    unsafe {
        complex_transform(
            array,
            axes,
            axis_count,
            None,
            true,
            normalization,
            out_array,
        )
    }
}

fn truncate_axis(
    data: &[Complex64],
    shape: &[i64],
    axis: usize,
    output_length: usize,
) -> Vec<Complex64> {
    let input_length = shape[axis] as usize;
    let outer = shape[..axis]
        .iter()
        .map(|&value| value as usize)
        .product::<usize>();
    let inner = shape[axis + 1..]
        .iter()
        .map(|&value| value as usize)
        .product::<usize>();
    let mut output = Vec::with_capacity(outer * output_length * inner);
    for outer_index in 0..outer {
        for axis_index in 0..output_length {
            let start = (outer_index * input_length + axis_index) * inner;
            output.extend_from_slice(&data[start..start + inner]);
        }
    }
    output
}

unsafe fn real_forward(
    array: Array,
    length: Option<i64>,
    axis: i64,
    normalization: i32,
    out_array: *mut Array,
) -> i32 {
    if out_array.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *out_array = std::ptr::null_mut() };
    match unsafe { array_dtype(array) } {
        Ok(dtype) if matches!(dtype, 12 | 13) => return UNSUPPORTED_DTYPE,
        Ok(_) => {}
        Err(status) => return status,
    }
    let mut shape = match unsafe { array_shape(array) } {
        Ok(shape) => shape,
        Err(status) => return status,
    };
    let axis = match normalize_axis(axis, shape.len()) {
        Ok(axis) => axis,
        Err(status) => return status,
    };
    let mut data = match unsafe { copied_complex(array) } {
        Ok(data) => data,
        Err(status) => return status,
    };
    if let Some(length) = length {
        data = match resize_complex_axis(data, &mut shape, axis, length) {
            Ok(data) => data,
            Err(status) => return status,
        };
    }
    if let Err(status) = transform_axis(&mut data, &shape, axis, false, normalization) {
        return status;
    }
    let output_length = shape[axis] as usize / 2 + 1;
    data = truncate_axis(&data, &shape, axis, output_length);
    shape[axis] = output_length as i64;
    match unsafe { owned_complex(&data, &shape, 13) } {
        Ok(output) => {
            unsafe { *out_array = output };
            OK
        }
        Err(status) => status,
    }
}

/// Applies a real-input FFT and retains the non-negative frequency bins.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_rfft(
    array: Array,
    axis: i64,
    normalization: i32,
    out_array: *mut Array,
) -> i32 {
    unsafe { real_forward(array, None, axis, normalization, out_array) }
}

/// Applies a length-adjusted real-input FFT.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_rfft_sized(
    array: Array,
    length: i64,
    axis: i64,
    normalization: i32,
    out_array: *mut Array,
) -> i32 {
    unsafe { real_forward(array, Some(length), axis, normalization, out_array) }
}

fn expand_hermitian(
    compact: &[Complex64],
    shape: &[i64],
    axis: usize,
    output_length: usize,
) -> Result<Vec<Complex64>, i32> {
    let compact_length = shape[axis] as usize;
    if compact_length != output_length / 2 + 1 || output_length == 0 {
        return Err(SHAPE_MISMATCH);
    }
    let outer = shape[..axis]
        .iter()
        .map(|&value| value as usize)
        .product::<usize>();
    let inner = shape[axis + 1..]
        .iter()
        .map(|&value| value as usize)
        .product::<usize>();
    let mut output = vec![Complex64::new(0.0, 0.0); outer * output_length * inner];
    for outer_index in 0..outer {
        for axis_index in 0..output_length {
            for inner_index in 0..inner {
                let destination = (outer_index * output_length + axis_index) * inner + inner_index;
                if axis_index < compact_length {
                    output[destination] =
                        compact[(outer_index * compact_length + axis_index) * inner + inner_index];
                } else {
                    output[destination] = compact[(outer_index * compact_length + output_length
                        - axis_index)
                        * inner
                        + inner_index]
                        .conj();
                }
            }
        }
    }
    Ok(output)
}

unsafe fn real_inverse(
    array: Array,
    length: i64,
    axis: i64,
    normalization: i32,
    hermitian_forward: bool,
    out_array: *mut Array,
) -> i32 {
    if out_array.is_null() || length <= 0 {
        return INVALID_ARGUMENT;
    }
    unsafe { *out_array = std::ptr::null_mut() };
    if let Err(status) = unsafe { array_dtype(array) } {
        return status;
    }
    let mut shape = match unsafe { array_shape(array) } {
        Ok(shape) => shape,
        Err(status) => return status,
    };
    let axis = match normalize_axis(axis, shape.len()) {
        Ok(axis) => axis,
        Err(status) => return status,
    };
    let compact = match unsafe { copied_complex(array) } {
        Ok(data) => data,
        Err(status) => return status,
    };
    let length = match usize::try_from(length) {
        Ok(length) => length,
        Err(_) => return SIZE_OVERFLOW,
    };
    let mut data = match expand_hermitian(&compact, &shape, axis, length) {
        Ok(data) => data,
        Err(status) => return status,
    };
    shape[axis] = length as i64;
    if let Err(status) = transform_axis(&mut data, &shape, axis, !hermitian_forward, normalization)
    {
        return status;
    }
    let reals: Vec<f64> = data.into_iter().map(|value| value.re).collect();
    match unsafe { owned_real(&reals, &shape, 3) } {
        Ok(output) => {
            unsafe { *out_array = output };
            OK
        }
        Err(status) => status,
    }
}

/// Reconstructs a real signal from non-negative frequency bins.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_irfft(
    array: Array,
    length: i64,
    axis: i64,
    normalization: i32,
    out_array: *mut Array,
) -> i32 {
    unsafe { real_inverse(array, length, axis, normalization, false, out_array) }
}

/// Returns compact real-input FFTs across normalized distinct axes.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_rfftn(
    array: Array,
    axes: *const i64,
    axis_count: i64,
    normalization: i32,
    out_array: *mut Array,
) -> i32 {
    if out_array.is_null() {
        return INVALID_ARGUMENT;
    }
    // SAFETY: the checked output slot is exclusively owned by this call.
    unsafe { *out_array = std::ptr::null_mut() };
    let shape = match unsafe { array_shape(array) } {
        Ok(shape) => shape,
        Err(status) => return status,
    };
    let axes = match normalized_axes(axes, axis_count, shape.len()) {
        Ok(axes) if !axes.is_empty() => axes,
        Ok(_) => return INVALID_ARGUMENT,
        Err(status) => return status,
    };
    let real_axis = axes[axes.len() - 1] as i64;
    let mut compact = std::ptr::null_mut();
    // SAFETY: the borrowed array and temporary output slot are live for the call.
    let status = unsafe { terlan_ndarray_v1_rfft(array, real_axis, normalization, &mut compact) };
    if status != OK {
        return status;
    }
    if axes.len() == 1 {
        // SAFETY: ownership of the successful temporary transfers to the caller.
        unsafe { *out_array = compact };
        return OK;
    }
    let complex_axes: Vec<i64> = axes[..axes.len() - 1]
        .iter()
        .map(|axis| *axis as i64)
        .collect();
    // SAFETY: the temporary and axis slice remain live while FFTN copies the result.
    let status = unsafe {
        terlan_ndarray_v1_fftn(
            compact,
            complex_axes.as_ptr(),
            complex_axes.len() as i64,
            normalization,
            out_array,
        )
    };
    // SAFETY: FFTN borrows rather than consumes the compact temporary.
    let _ = unsafe { super::super::ffi::terlan_ndarray_v1_delete(compact) };
    status
}

/// Reconstructs a real array from compact N-dimensional frequency bins.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_irfftn(
    array: Array,
    length: i64,
    axes: *const i64,
    axis_count: i64,
    normalization: i32,
    out_array: *mut Array,
) -> i32 {
    if out_array.is_null() {
        return INVALID_ARGUMENT;
    }
    // SAFETY: the checked output slot is exclusively owned by this call.
    unsafe { *out_array = std::ptr::null_mut() };
    let shape = match unsafe { array_shape(array) } {
        Ok(shape) => shape,
        Err(status) => return status,
    };
    let axes = match normalized_axes(axes, axis_count, shape.len()) {
        Ok(axes) if !axes.is_empty() => axes,
        Ok(_) => return INVALID_ARGUMENT,
        Err(status) => return status,
    };
    let real_axis = axes[axes.len() - 1] as i64;
    let complex_axes: Vec<i64> = axes[..axes.len() - 1]
        .iter()
        .map(|axis| *axis as i64)
        .collect();
    let mut transformed = std::ptr::null_mut();
    let input = if complex_axes.is_empty() {
        array
    } else {
        // SAFETY: the source, axis slice, and temporary output slot are live.
        let status = unsafe {
            terlan_ndarray_v1_ifftn(
                array,
                complex_axes.as_ptr(),
                complex_axes.len() as i64,
                normalization,
                &mut transformed,
            )
        };
        if status != OK {
            return status;
        }
        transformed
    };
    // SAFETY: the selected input remains live while IRFFT copies the result.
    let status =
        unsafe { terlan_ndarray_v1_irfft(input, length, real_axis, normalization, out_array) };
    if !transformed.is_null() {
        // SAFETY: IRFFT borrows rather than consumes the transformed temporary.
        let _ = unsafe { super::super::ffi::terlan_ndarray_v1_delete(transformed) };
    }
    status
}

/// Transforms a compact Hermitian signal into a real spectrum.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_hfft(
    array: Array,
    length: i64,
    axis: i64,
    normalization: i32,
    out_array: *mut Array,
) -> i32 {
    unsafe { real_inverse(array, length, axis, normalization, true, out_array) }
}

/// Applies an inverse Hermitian transform with optional axis resizing.
unsafe fn hermitian_inverse(
    array: Array,
    length: Option<i64>,
    axis: i64,
    normalization: i32,
    out_array: *mut Array,
) -> i32 {
    if out_array.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *out_array = std::ptr::null_mut() };
    match unsafe { array_dtype(array) } {
        Ok(dtype) if matches!(dtype, 12 | 13) => return UNSUPPORTED_DTYPE,
        Ok(_) => {}
        Err(status) => return status,
    }
    let mut shape = match unsafe { array_shape(array) } {
        Ok(shape) => shape,
        Err(status) => return status,
    };
    let axis = match normalize_axis(axis, shape.len()) {
        Ok(axis) => axis,
        Err(status) => return status,
    };
    let mut data = match unsafe { copied_complex(array) } {
        Ok(data) => data,
        Err(status) => return status,
    };
    if let Some(length) = length {
        data = match resize_complex_axis(data, &mut shape, axis, length) {
            Ok(data) => data,
            Err(status) => return status,
        };
    }
    if let Err(status) = transform_axis(&mut data, &shape, axis, true, normalization) {
        return status;
    }
    let output_length = shape[axis] as usize / 2 + 1;
    data = truncate_axis(&data, &shape, axis, output_length);
    shape[axis] = output_length as i64;
    match unsafe { owned_complex(&data, &shape, 13) } {
        Ok(output) => {
            unsafe { *out_array = output };
            OK
        }
        Err(status) => status,
    }
}

/// Transforms a real spectrum into its compact Hermitian signal.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_ihfft(
    array: Array,
    axis: i64,
    normalization: i32,
    out_array: *mut Array,
) -> i32 {
    unsafe { hermitian_inverse(array, None, axis, normalization, out_array) }
}

/// Applies a length-adjusted inverse Hermitian FFT.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_ihfft_sized(
    array: Array,
    length: i64,
    axis: i64,
    normalization: i32,
    out_array: *mut Array,
) -> i32 {
    unsafe { hermitian_inverse(array, Some(length), axis, normalization, out_array) }
}

unsafe fn frequencies(length: i64, spacing: f64, real: bool, out_array: *mut Array) -> i32 {
    if out_array.is_null() || length <= 0 || !spacing.is_finite() || spacing == 0.0 {
        return INVALID_ARGUMENT;
    }
    unsafe { *out_array = std::ptr::null_mut() };
    let count = if real { length / 2 + 1 } else { length };
    let count_usize = match usize::try_from(count) {
        Ok(count) => count,
        Err(_) => return SIZE_OVERFLOW,
    };
    let mut values = Vec::new();
    if values.try_reserve_exact(count_usize).is_err() {
        return SIZE_OVERFLOW;
    }
    for index in 0..count {
        let frequency = if real || index <= (length - 1) / 2 {
            index
        } else {
            index - length
        };
        values.push(frequency as f64 / (length as f64 * spacing));
    }
    match unsafe { owned_real(&values, &[count], 3) } {
        Ok(output) => {
            unsafe { *out_array = output };
            OK
        }
        Err(status) => status,
    }
}

/// Returns signed discrete frequency bins.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_fftfreq(
    length: i64,
    spacing: f64,
    out_array: *mut Array,
) -> i32 {
    unsafe { frequencies(length, spacing, false, out_array) }
}

/// Returns non-negative real-transform frequency bins.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_rfftfreq(
    length: i64,
    spacing: f64,
    out_array: *mut Array,
) -> i32 {
    unsafe { frequencies(length, spacing, true, out_array) }
}

unsafe fn shift_array(
    array: Array,
    axes: *const i64,
    axis_count: i64,
    inverse: bool,
    out_array: *mut Array,
) -> i32 {
    if out_array.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *out_array = std::ptr::null_mut() };
    let shape = match unsafe { array_shape(array) } {
        Ok(shape) => shape,
        Err(status) => return status,
    };
    let axes = match normalized_axes(axes, axis_count, shape.len()) {
        Ok(axes) => axes,
        Err(status) => return status,
    };
    let mut current = array;
    let mut owns_current = false;
    for axis in axes {
        let length = shape[axis];
        let shift = if inverse { -(length / 2) } else { length / 2 };
        let mut next = std::ptr::null_mut();
        // SAFETY: current is live and the output slot is writable.
        let status = unsafe {
            super::super::ffi::terlan_ndarray_v1_roll(current, shift, axis as i64, &mut next)
        };
        if owns_current {
            let _ = unsafe { super::super::ffi::terlan_ndarray_v1_delete(current) };
        }
        if status != OK {
            return status;
        }
        current = next;
        owns_current = true;
    }
    if !owns_current {
        let mut copied = std::ptr::null_mut();
        let status = unsafe { super::super::ffi::terlan_ndarray_v1_copy(array, &mut copied) };
        if status != OK {
            return status;
        }
        current = copied;
    }
    unsafe { *out_array = current };
    OK
}

/// Moves zero-frequency bins to centered axis positions.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_fftshift(
    array: Array,
    axes: *const i64,
    axis_count: i64,
    out_array: *mut Array,
) -> i32 {
    unsafe { shift_array(array, axes, axis_count, false, out_array) }
}

/// Reverses centered frequency-bin shifts.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_ifftshift(
    array: Array,
    axes: *const i64,
    axis_count: i64,
    out_array: *mut Array,
) -> i32 {
    unsafe { shift_array(array, axes, axis_count, true, out_array) }
}

#[cfg(test)]
#[path = "ndarray_fft_tests.rs"]
mod tests;
