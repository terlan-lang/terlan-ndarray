//! Official Apache Arrow C Data ownership for primitive ndarray values.

use arrow::array::{make_array, Array, ArrayData, BooleanArray};
use arrow::buffer::Buffer;
use arrow::datatypes::DataType;
use arrow::ffi::{from_ffi, to_ffi, FFI_ArrowArray, FFI_ArrowSchema};

use super::serialization_extension::{
    encode_tensor_packet, export_packet, parse_tensor_packet, TensorPacket,
};

const OK: i32 = 0;
const INVALID_ARGUMENT: i32 = 100;
const SIZE_OVERFLOW: i32 = 103;
const UNSUPPORTED_DTYPE: i32 = 109;
const MALFORMED_DATA: i32 = 127;

/// Owned Arrow C Data array/schema pair plus exact ndarray reconstruction data.
pub struct ArrowArray {
    array: FFI_ArrowArray,
    schema: FFI_ArrowSchema,
    tensor_packet: Vec<u8>,
    length: i64,
}

fn arrow_type(code: u8, bits: u8) -> Result<DataType, i32> {
    match (code, bits) {
        (6, 8) => Ok(DataType::Boolean),
        (0, 8) => Ok(DataType::Int8),
        (0, 16) => Ok(DataType::Int16),
        (0, 32) => Ok(DataType::Int32),
        (0, 64) => Ok(DataType::Int64),
        (1, 8) => Ok(DataType::UInt8),
        (1, 16) => Ok(DataType::UInt16),
        (1, 32) => Ok(DataType::UInt32),
        (1, 64) => Ok(DataType::UInt64),
        (2, 16) => Ok(DataType::Float16),
        (2, 32) => Ok(DataType::Float32),
        (2, 64) => Ok(DataType::Float64),
        _ => Err(UNSUPPORTED_DTYPE),
    }
}

fn tensor_type(data_type: &DataType) -> Result<(u8, u8), i32> {
    match data_type {
        DataType::Boolean => Ok((6, 8)),
        DataType::Int8 => Ok((0, 8)),
        DataType::Int16 => Ok((0, 16)),
        DataType::Int32 => Ok((0, 32)),
        DataType::Int64 => Ok((0, 64)),
        DataType::UInt8 => Ok((1, 8)),
        DataType::UInt16 => Ok((1, 16)),
        DataType::UInt32 => Ok((1, 32)),
        DataType::UInt64 => Ok((1, 64)),
        DataType::Float16 => Ok((2, 16)),
        DataType::Float32 => Ok((2, 32)),
        DataType::Float64 => Ok((2, 64)),
        _ => Err(UNSUPPORTED_DTYPE),
    }
}

fn element_count(packet: &TensorPacket) -> Result<usize, i32> {
    packet.shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(usize::try_from(*dimension).map_err(|_| SIZE_OVERFLOW)?)
            .ok_or(SIZE_OVERFLOW)
    })
}

fn arrow_data(packet: &TensorPacket) -> Result<ArrayData, i32> {
    let data_type = arrow_type(packet.dtype_code, packet.bits)?;
    let length = element_count(packet)?;
    if data_type == DataType::Boolean {
        if packet.data.len() != length || packet.data.iter().any(|value| *value > 1) {
            return Err(MALFORMED_DATA);
        }
        let values = packet
            .data
            .iter()
            .map(|value| *value != 0)
            .collect::<Vec<_>>();
        return Ok(BooleanArray::from(values).to_data());
    }
    let width = usize::from(packet.bits / 8);
    if length.checked_mul(width) != Some(packet.data.len()) {
        return Err(MALFORMED_DATA);
    }
    ArrayData::builder(data_type)
        .len(length)
        .add_buffer(Buffer::from(packet.data.clone()))
        .build()
        .map_err(|_| MALFORMED_DATA)
}

fn imported_bytes(data: ArrayData, code: u8, bits: u8) -> Result<Vec<u8>, i32> {
    if data.null_count() != 0 || !data.child_data().is_empty() {
        return Err(UNSUPPORTED_DTYPE);
    }
    if code == 6 {
        let values = make_array(data);
        let values = values
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or(MALFORMED_DATA)?;
        return Ok(values
            .iter()
            .map(|value| u8::from(value == Some(true)))
            .collect());
    }
    let width = usize::from(bits / 8);
    let start = data.offset().checked_mul(width).ok_or(SIZE_OVERFLOW)?;
    let length = data.len().checked_mul(width).ok_or(SIZE_OVERFLOW)?;
    let end = start.checked_add(length).ok_or(SIZE_OVERFLOW)?;
    let buffer = data.buffers().first().ok_or(MALFORMED_DATA)?.as_slice();
    Ok(buffer.get(start..end).ok_or(MALFORMED_DATA)?.to_vec())
}

fn imported_shape(shape: *const i64, rank: i64, length: usize) -> Result<Vec<u64>, i32> {
    if rank < 0 || (rank > 0 && shape.is_null()) {
        return Err(INVALID_ARGUMENT);
    }
    let rank = usize::try_from(rank).map_err(|_| SIZE_OVERFLOW)?;
    let dimensions = if rank == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(shape, rank) }
    };
    let shape = dimensions
        .iter()
        .map(|dimension| u64::try_from(*dimension).map_err(|_| INVALID_ARGUMENT))
        .collect::<Result<Vec<_>, _>>()?;
    if element_count(&TensorPacket {
        dtype_code: 6,
        bits: 8,
        shape: shape.clone(),
        data: Vec::new(),
    })? != length
    {
        return Err(MALFORMED_DATA);
    }
    Ok(shape)
}

/// Exports a primitive CPU array through official Arrow C Data callbacks.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_to_arrow(
    array: *mut crate::ffi::TerlanNdArray,
    out_arrow: *mut *mut ArrowArray,
) -> i32 {
    if out_arrow.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *out_arrow = std::ptr::null_mut() };
    let tensor_packet = match unsafe { export_packet(array) } {
        Ok(packet) => packet,
        Err(status) => return status,
    };
    let packet = match parse_tensor_packet(&tensor_packet) {
        Ok(packet) => packet,
        Err(status) => return status,
    };
    let length = match element_count(&packet)
        .and_then(|value| i64::try_from(value).map_err(|_| SIZE_OVERFLOW))
    {
        Ok(length) => length,
        Err(status) => return status,
    };
    let data = match arrow_data(&packet) {
        Ok(data) => data,
        Err(status) => return status,
    };
    let (array, schema) = match to_ffi(&data) {
        Ok(values) => values,
        Err(_) => return MALFORMED_DATA,
    };
    let export = Box::new(ArrowArray {
        array,
        schema,
        tensor_packet,
        length,
    });
    unsafe { *out_arrow = Box::into_raw(export) };
    OK
}

/// Imports one package-owned Arrow export under the ndarray no-null policy.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_from_arrow(
    arrow: *mut ArrowArray,
    out_array: *mut *mut crate::ffi::TerlanNdArray,
) -> i32 {
    if arrow.is_null() || out_array.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *out_array = std::ptr::null_mut() };
    let packet = unsafe { &(*arrow).tensor_packet };
    unsafe {
        crate::ffi::terlan_ndarray_v1_import_tensor_packet(
            packet.as_ptr(),
            packet.len() as i64,
            out_array,
        )
    }
}

/// Moves an external Arrow C Data array/schema pair into an owned ndarray.
///
/// The pointed structures are replaced with released empty values after ownership is
/// transferred. Callers must not invoke their former release callbacks afterward.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_import_arrow_c(
    array: *mut std::ffi::c_void,
    schema: *mut std::ffi::c_void,
    shape: *const i64,
    rank: i64,
    out_array: *mut *mut crate::ffi::TerlanNdArray,
) -> i32 {
    if array.is_null() || schema.is_null() || out_array.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *out_array = std::ptr::null_mut() };
    if rank < 0 || (rank > 0 && shape.is_null()) {
        return INVALID_ARGUMENT;
    }
    let ffi_array = unsafe { FFI_ArrowArray::from_raw(array.cast()) };
    let ffi_schema = unsafe { FFI_ArrowSchema::from_raw(schema.cast()) };
    let data = match unsafe { from_ffi(ffi_array, &ffi_schema) } {
        Ok(data) => data,
        Err(_) => return MALFORMED_DATA,
    };
    let shape = match imported_shape(shape, rank, data.len()) {
        Ok(shape) => shape,
        Err(status) => return status,
    };
    let (code, bits) = match tensor_type(data.data_type()) {
        Ok(dtype) => dtype,
        Err(status) => return status,
    };
    let bytes = match imported_bytes(data, code, bits) {
        Ok(bytes) => bytes,
        Err(status) => return status,
    };
    let packet = match encode_tensor_packet(code, bits, &shape, &bytes) {
        Ok(packet) => packet,
        Err(status) => return status,
    };
    unsafe {
        crate::ffi::terlan_ndarray_v1_import_tensor_packet(
            packet.as_ptr(),
            packet.len() as i64,
            out_array,
        )
    }
}

/// Returns the flattened Arrow element count.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_arrow_length(
    arrow: *mut ArrowArray,
    out_length: *mut i64,
) -> i32 {
    if arrow.is_null() || out_length.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *out_length = (*arrow).length };
    OK
}

/// Returns zero because ndarray rejects nullable Arrow values.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_arrow_null_count(
    arrow: *mut ArrowArray,
    out_null_count: *mut i64,
) -> i32 {
    if arrow.is_null() || out_null_count.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *out_null_count = 0 };
    OK
}

/// Borrows the official ArrowArray C Data structure for native consumers.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_arrow_array_pointer(
    arrow: *mut ArrowArray,
) -> *const std::ffi::c_void {
    if arrow.is_null() {
        std::ptr::null()
    } else {
        unsafe { std::ptr::addr_of!((*arrow).array) }.cast()
    }
}

/// Borrows the official ArrowSchema C Data structure for native consumers.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_arrow_schema_pointer(
    arrow: *mut ArrowArray,
) -> *const std::ffi::c_void {
    if arrow.is_null() {
        std::ptr::null()
    } else {
        unsafe { std::ptr::addr_of!((*arrow).schema) }.cast()
    }
}

/// Releases an Arrow export and invokes its official C Data release callbacks exactly once.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_arrow_delete(arrow: *mut ArrowArray) -> i32 {
    if arrow.is_null() {
        return INVALID_ARGUMENT;
    }
    drop(unsafe { Box::from_raw(arrow) });
    OK
}

#[cfg(test)]
#[path = "ndarray_arrow_tests.rs"]
mod tests;
