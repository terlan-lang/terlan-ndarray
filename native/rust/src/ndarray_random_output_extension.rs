//! Transactional destination writes for owned random streams.

use super::native_support::{dtype, shape, Array};
use super::{random_stream_mut, RandomStream, INVALID_ARGUMENT, OK, UNSUPPORTED_DTYPE};

type Stream = *mut super::super::ffi::TerlanNdRandomStream;

const UNSUPPORTED_LAYOUT: i32 = 123;

/// Validates the shape, dtype, and C-contiguous layout required by NumPy outputs.
unsafe fn destination_contract(destination: Array) -> Result<(Vec<i64>, i32), i32> {
    let dimensions = unsafe { shape(destination)? };
    let destination_dtype = unsafe { dtype(destination)? };
    if !matches!(destination_dtype, 3 | 11) {
        return Err(UNSUPPORTED_DTYPE);
    }
    let mut contiguous = 0_i32;
    let status =
        unsafe { super::super::ffi::terlan_ndarray_v1_is_contiguous(destination, &mut contiguous) };
    if status != OK {
        return Err(status);
    }
    if contiguous != 1 {
        return Err(UNSUPPORTED_LAYOUT);
    }
    Ok((dimensions, destination_dtype))
}

/// Samples through a cloned stream and commits both output and state atomically.
unsafe fn fill_transactionally(
    stream: Stream,
    destination: Array,
    generate: impl FnOnce(Stream, &[i64], i32, *mut Array) -> i32,
) -> i32 {
    let (dimensions, destination_dtype) = match unsafe { destination_contract(destination) } {
        Ok(contract) => contract,
        Err(status) => return status,
    };
    let stream = match unsafe { random_stream_mut(stream) } {
        Ok(stream) => stream,
        Err(status) => return status,
    };
    let mut staged = RandomStream {
        generator: stream.generator.clone(),
    };
    let staged_handle = (&mut staged as *mut RandomStream).cast();
    let mut sampled = std::ptr::null_mut();
    let status = generate(staged_handle, &dimensions, destination_dtype, &mut sampled);
    if status != OK {
        if !sampled.is_null() {
            let _ = unsafe { super::super::ffi::terlan_ndarray_v1_delete(sampled) };
        }
        return status;
    }
    let status = unsafe { super::super::ffi::terlan_ndarray_v1_copyto(destination, sampled) };
    let delete_status = unsafe { super::super::ffi::terlan_ndarray_v1_delete(sampled) };
    if status == OK && delete_status == OK {
        stream.generator = staged.generator;
        OK
    } else if status != OK {
        status
    } else {
        delete_status
    }
}

/// Fills a C-contiguous destination with uniform values in `[0, 1)`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_uniform_into(
    stream: Stream,
    destination: Array,
) -> i32 {
    unsafe {
        fill_transactionally(stream, destination, |stream, shape, dtype, output| {
            super::terlan_ndarray_v1_random_stream_uniform(
                stream,
                shape.as_ptr(),
                shape.len() as i64,
                0.0,
                1.0,
                dtype,
                output,
            )
        })
    }
}

/// Fills a C-contiguous destination with standard-normal values.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_standard_normal_into(
    stream: Stream,
    destination: Array,
) -> i32 {
    unsafe {
        fill_transactionally(stream, destination, |stream, shape, dtype, output| {
            super::terlan_ndarray_v1_random_stream_standard_normal(
                stream,
                shape.as_ptr(),
                shape.len() as i64,
                dtype,
                output,
            )
        })
    }
}

/// Fills a C-contiguous destination with standard-exponential values.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_standard_exponential_into(
    stream: Stream,
    destination: Array,
) -> i32 {
    unsafe {
        fill_transactionally(stream, destination, |stream, shape, dtype, output| {
            super::terlan_ndarray_v1_random_stream_standard_exponential(
                stream,
                shape.as_ptr(),
                shape.len() as i64,
                dtype,
                output,
            )
        })
    }
}

/// Fills a C-contiguous destination from broadcast unit-scale gamma shapes.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_standard_gamma_into(
    stream: Stream,
    destination: Array,
    distribution_shapes: Array,
) -> i32 {
    if distribution_shapes.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe {
        fill_transactionally(stream, destination, |stream, shape, dtype, output| {
            let mut broadcast = std::ptr::null_mut();
            let status = super::super::ffi::terlan_ndarray_v1_broadcast_to(
                distribution_shapes,
                shape.as_ptr(),
                shape.len() as i64,
                &mut broadcast,
            );
            if status != OK {
                return status;
            }
            let parameters = [broadcast as usize as u64];
            let status = super::terlan_ndarray_v1_random_stream_sample_broadcast(
                stream,
                parameters.as_ptr(),
                1,
                12,
                dtype,
                output,
            );
            let delete_status = super::super::ffi::terlan_ndarray_v1_delete(broadcast);
            if status != OK {
                status
            } else {
                delete_status
            }
        })
    }
}
