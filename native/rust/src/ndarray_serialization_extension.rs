//! Bounded deterministic NumPy and package-native serialization.

use npyz::{DType, Endianness, NpyFile, Order, TypeChar};
use std::io::Cursor;

const OK: i32 = 0;
const INVALID_ARGUMENT: i32 = 100;
const SIZE_OVERFLOW: i32 = 103;
const ALLOCATION_FAILED: i32 = 104;
const UNSUPPORTED_DTYPE: i32 = 109;
const UNSUPPORTED_LAYOUT: i32 = 123;
const MALFORMED_DATA: i32 = 127;
const INTEGRITY_ERROR: i32 = 128;
const IO_ERROR: i32 = 129;
const TNXP_HEADER: usize = 16;
const RAW_HEADER: usize = 20;

pub(super) struct TensorPacket {
    pub(super) dtype_code: u8,
    pub(super) bits: u8,
    pub(super) shape: Vec<u64>,
    pub(super) data: Vec<u8>,
}

fn read_u32(bytes: &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?))
}

fn read_u64(bytes: &[u8]) -> Option<u64> {
    Some(u64::from_le_bytes(bytes.get(..8)?.try_into().ok()?))
}

pub(super) fn parse_tensor_packet(packet: &[u8]) -> Result<TensorPacket, i32> {
    if packet.len() < TNXP_HEADER || &packet[..4] != b"TNXP" || packet[4] != 1 {
        return Err(MALFORMED_DATA);
    }
    let rank = read_u32(&packet[12..]).ok_or(MALFORMED_DATA)? as usize;
    let metadata = TNXP_HEADER
        .checked_add(rank.checked_mul(16).ok_or(SIZE_OVERFLOW)?)
        .ok_or(SIZE_OVERFLOW)?;
    if metadata > packet.len() {
        return Err(MALFORMED_DATA);
    }
    let mut shape = Vec::new();
    shape
        .try_reserve_exact(rank)
        .map_err(|_| ALLOCATION_FAILED)?;
    for index in 0..rank {
        shape.push(read_u64(&packet[TNXP_HEADER + index * 8..]).ok_or(MALFORMED_DATA)?);
    }
    Ok(TensorPacket {
        dtype_code: packet[6],
        bits: packet[7],
        shape,
        data: packet[metadata..].to_vec(),
    })
}

fn numpy_descriptor(code: u8, bits: u8) -> Result<&'static str, i32> {
    match (code, bits) {
        (6, 8) => Ok("|b1"),
        (0, 8) => Ok("|i1"),
        (0, 16) => Ok("<i2"),
        (0, 32) => Ok("<i4"),
        (0, 64) => Ok("<i8"),
        (1, 8) => Ok("|u1"),
        (1, 16) => Ok("<u2"),
        (1, 32) => Ok("<u4"),
        (1, 64) => Ok("<u8"),
        (2, 16) => Ok("<f2"),
        (2, 32) => Ok("<f4"),
        (2, 64) => Ok("<f8"),
        (5, 64) => Ok("<c8"),
        (5, 128) => Ok("<c16"),
        _ => Err(UNSUPPORTED_DTYPE),
    }
}

fn dlpack_dtype(dtype: DType) -> Result<(u8, u8), i32> {
    let DType::Plain(dtype) = dtype else {
        return Err(UNSUPPORTED_DTYPE);
    };
    let bytes = u8::try_from(dtype.size_field()).map_err(|_| UNSUPPORTED_DTYPE)?;
    let bits = bytes.checked_mul(8).ok_or(UNSUPPORTED_DTYPE)?;
    if bytes > 1 && dtype.endianness() != Endianness::of_machine() {
        return Err(UNSUPPORTED_LAYOUT);
    }
    match (dtype.type_char(), bytes) {
        (TypeChar::Bool, 1) => Ok((6, bits)),
        (TypeChar::Int, 1 | 2 | 4 | 8) => Ok((0, bits)),
        (TypeChar::Uint, 1 | 2 | 4 | 8) => Ok((1, bits)),
        (TypeChar::Float, 2 | 4 | 8) => Ok((2, bits)),
        (TypeChar::Complex, 8 | 16) => Ok((5, bits)),
        _ => Err(UNSUPPORTED_DTYPE),
    }
}

fn ndarray_dtype(code: u8, bits: u8) -> Result<i32, i32> {
    match (code, bits) {
        (6, 8) => Ok(1),
        (0, 64) => Ok(2),
        (2, 64) => Ok(3),
        (0, 8) => Ok(4),
        (0, 16) => Ok(5),
        (0, 32) => Ok(6),
        (1, 8) => Ok(7),
        (1, 16) => Ok(8),
        (1, 32) => Ok(9),
        (1, 64) => Ok(10),
        (2, 32) => Ok(11),
        (5, 64) => Ok(12),
        (5, 128) => Ok(13),
        (2, 16) => Ok(14),
        _ => Err(UNSUPPORTED_DTYPE),
    }
}

fn shape_text(shape: &[u64]) -> String {
    match shape {
        [] => "()".to_string(),
        [value] => format!("({value},)"),
        values => format!(
            "({})",
            values
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn linear_coordinates(mut index: u64, shape: &[u64], fortran_order: bool) -> Vec<u64> {
    let mut coordinates = vec![0; shape.len()];
    if fortran_order {
        for axis in 0..shape.len() {
            let extent = shape[axis];
            if extent > 0 {
                coordinates[axis] = index % extent;
                index /= extent;
            }
        }
    } else {
        for axis in (0..shape.len()).rev() {
            let extent = shape[axis];
            if extent > 0 {
                coordinates[axis] = index % extent;
                index /= extent;
            }
        }
    }
    coordinates
}

fn linear_index(coordinates: &[u64], shape: &[u64], fortran_order: bool) -> u64 {
    if fortran_order {
        let mut stride = 1u64;
        coordinates
            .iter()
            .zip(shape)
            .map(|(coordinate, extent)| {
                let index = coordinate.saturating_mul(stride);
                stride = stride.saturating_mul(*extent);
                index
            })
            .sum()
    } else {
        coordinates
            .iter()
            .zip(shape)
            .fold(0u64, |index, (coordinate, extent)| {
                index.saturating_mul(*extent).saturating_add(*coordinate)
            })
    }
}

fn reorder_data(
    data: &[u8],
    shape: &[u64],
    element_size: usize,
    source_fortran: bool,
    destination_fortran: bool,
) -> Result<Vec<u8>, i32> {
    let elements = shape
        .iter()
        .try_fold(1u64, |count, extent| count.checked_mul(*extent))
        .ok_or(SIZE_OVERFLOW)?;
    let expected = usize::try_from(elements)
        .ok()
        .and_then(|count| count.checked_mul(element_size))
        .ok_or(SIZE_OVERFLOW)?;
    if data.len() != expected {
        return Err(MALFORMED_DATA);
    }
    if source_fortran == destination_fortran || elements <= 1 {
        return Ok(data.to_vec());
    }
    let mut reordered = vec![0; expected];
    for destination in 0..elements {
        let coordinates = linear_coordinates(destination, shape, destination_fortran);
        let source = linear_index(&coordinates, shape, source_fortran);
        let source = usize::try_from(source)
            .ok()
            .and_then(|index| index.checked_mul(element_size))
            .ok_or(SIZE_OVERFLOW)?;
        let destination = usize::try_from(destination)
            .ok()
            .and_then(|index| index.checked_mul(element_size))
            .ok_or(SIZE_OVERFLOW)?;
        reordered[destination..destination + element_size]
            .copy_from_slice(&data[source..source + element_size]);
    }
    Ok(reordered)
}

unsafe fn array_uses_fortran_order(array: *mut crate::ffi::TerlanNdArray) -> Result<bool, i32> {
    let mut c_contiguous = 0;
    let mut fortran_contiguous = 0;
    let status = unsafe { crate::ffi::terlan_ndarray_v1_is_contiguous(array, &mut c_contiguous) };
    if status != OK {
        return Err(status);
    }
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_is_fortran_contiguous(array, &mut fortran_contiguous)
    };
    if status != OK {
        return Err(status);
    }
    Ok(fortran_contiguous != 0 && c_contiguous == 0)
}

fn encode_npy(mut packet: TensorPacket, fortran_order: bool) -> Result<Vec<u8>, i32> {
    let descriptor = numpy_descriptor(packet.dtype_code, packet.bits)?;
    let element_size = usize::from(packet.bits / 8);
    packet.data = reorder_data(
        &packet.data,
        &packet.shape,
        element_size,
        false,
        fortran_order,
    )?;
    let dictionary = format!(
        "{{'descr': '{descriptor}', 'fortran_order': {fortran_order_text}, 'shape': {shape}, }}",
        fortran_order_text = if fortran_order { "True" } else { "False" },
        shape = shape_text(&packet.shape),
    );
    let prefix = 10usize;
    let padding = (16 - ((prefix + dictionary.len() + 1) % 16)) % 16;
    let header_length = dictionary
        .len()
        .checked_add(padding)
        .and_then(|value| value.checked_add(1))
        .ok_or(SIZE_OVERFLOW)?;
    let header_length = u16::try_from(header_length).map_err(|_| SIZE_OVERFLOW)?;
    let capacity = prefix
        .checked_add(header_length as usize)
        .and_then(|value| value.checked_add(packet.data.len()))
        .ok_or(SIZE_OVERFLOW)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| ALLOCATION_FAILED)?;
    output.extend_from_slice(b"\x93NUMPY\x01\x00");
    output.extend_from_slice(&header_length.to_le_bytes());
    output.extend_from_slice(dictionary.as_bytes());
    output.resize(prefix + header_length as usize - 1, b' ');
    output.push(b'\n');
    output.extend_from_slice(&packet.data);
    Ok(output)
}

fn npy_payload_offset(bytes: &[u8]) -> Result<usize, i32> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(MALFORMED_DATA);
    }
    let (prefix, header) = match (bytes[6], bytes[7]) {
        (1, 0) => (10usize, u16::from_le_bytes([bytes[8], bytes[9]]) as usize),
        (2 | 3, 0) if bytes.len() >= 12 => (
            12usize,
            u32::from_le_bytes(bytes[8..12].try_into().map_err(|_| MALFORMED_DATA)?) as usize,
        ),
        _ => return Err(MALFORMED_DATA),
    };
    prefix
        .checked_add(header)
        .filter(|offset| *offset <= bytes.len())
        .ok_or(MALFORMED_DATA)
}

pub(super) fn encode_tensor_packet(
    code: u8,
    bits: u8,
    shape: &[u64],
    data: &[u8],
) -> Result<Vec<u8>, i32> {
    let rank = u32::try_from(shape.len()).map_err(|_| SIZE_OVERFLOW)?;
    let metadata = TNXP_HEADER
        .checked_add(shape.len().checked_mul(16).ok_or(SIZE_OVERFLOW)?)
        .ok_or(SIZE_OVERFLOW)?;
    let total = metadata.checked_add(data.len()).ok_or(SIZE_OVERFLOW)?;
    let mut packet = Vec::new();
    packet
        .try_reserve_exact(total)
        .map_err(|_| ALLOCATION_FAILED)?;
    packet.resize(total, 0);
    packet[..4].copy_from_slice(b"TNXP");
    packet[4] = 1;
    packet[5] = if cfg!(target_endian = "little") { 1 } else { 2 };
    packet[6] = code;
    packet[7] = bits;
    packet[8..10].copy_from_slice(&1u16.to_le_bytes());
    packet[10] = 1;
    packet[12..16].copy_from_slice(&rank.to_le_bytes());
    let mut stride = 1u64;
    for index in (0..shape.len()).rev() {
        let shape_offset = TNXP_HEADER + index * 8;
        packet[shape_offset..shape_offset + 8].copy_from_slice(&shape[index].to_le_bytes());
        let stride_offset = TNXP_HEADER + shape.len() * 8 + index * 8;
        packet[stride_offset..stride_offset + 8].copy_from_slice(&stride.to_le_bytes());
        stride = stride.checked_mul(shape[index]).ok_or(SIZE_OVERFLOW)?;
    }
    packet[metadata..].copy_from_slice(data);
    Ok(packet)
}

pub(super) unsafe fn export_packet(array: *mut crate::ffi::TerlanNdArray) -> Result<Vec<u8>, i32> {
    let consumer = c"terlan-ndarray-serialization";
    let mut packet = std::ptr::null_mut();
    let mut length = 0usize;
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_export_tensor_packet(
            array,
            consumer.as_ptr(),
            &mut packet,
            &mut length,
        )
    };
    if status != OK {
        return Err(status);
    }
    let bytes = if length == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(packet, length) }.to_vec()
    };
    unsafe { crate::ffi::terlan_ndarray_v1_delete_tensor_packet(packet) };
    Ok(bytes)
}

unsafe fn transfer_bytes(bytes: Vec<u8>, out: *mut *mut u8, length: *mut usize) -> i32 {
    if out.is_null() || length.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe {
        *out = std::ptr::null_mut();
        *length = 0;
    }
    if bytes.is_empty() {
        return OK;
    }
    let output = unsafe { libc::malloc(bytes.len()) }.cast::<u8>();
    if output.is_null() {
        return ALLOCATION_FAILED;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output, bytes.len());
        *out = output;
        *length = bytes.len();
    }
    OK
}

fn bounded_input<'a>(bytes: *const u8, length: i64, maximum: i64) -> Result<&'a [u8], i32> {
    if length < 0 || maximum < 0 || length > maximum || (length > 0 && bytes.is_null()) {
        return Err(INVALID_ARGUMENT);
    }
    Ok(if length == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(bytes, length as usize) }
    })
}

/// Encodes an array as deterministic NumPy `.npy` bytes.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_to_npy(
    array: *mut crate::ffi::TerlanNdArray,
    out_bytes: *mut *mut u8,
    out_length: *mut usize,
) -> i32 {
    let fortran_order = match unsafe { array_uses_fortran_order(array) } {
        Ok(order) => order,
        Err(status) => return status,
    };
    let packet = match unsafe { export_packet(array) }.and_then(|bytes| parse_tensor_packet(&bytes))
    {
        Ok(packet) => packet,
        Err(status) => return status,
    };
    match encode_npy(packet, fortran_order) {
        Ok(bytes) => unsafe { transfer_bytes(bytes, out_bytes, out_length) },
        Err(status) => status,
    }
}

struct DecodedNpy {
    packet: Vec<u8>,
    fortran_order: bool,
}

fn decode_npy(bytes: &[u8]) -> Result<DecodedNpy, i32> {
    let file = match NpyFile::new(Cursor::new(bytes)) {
        Ok(file) => file,
        Err(_) => return Err(MALFORMED_DATA),
    };
    let fortran_order = file.order() == Order::Fortran;
    let (code, bits) = match dlpack_dtype(file.dtype()) {
        Ok(dtype) => dtype,
        Err(status) => return Err(status),
    };
    let shape = file.shape().to_vec();
    let expected = match shape
        .iter()
        .try_fold((bits / 8) as u64, |count, value| count.checked_mul(*value))
    {
        Some(value) => value,
        None => return Err(SIZE_OVERFLOW),
    };
    let offset = match npy_payload_offset(bytes) {
        Ok(offset) => offset,
        Err(status) => return Err(status),
    };
    let data = &bytes[offset..];
    if u64::try_from(data.len()).ok() != Some(expected) {
        return Err(MALFORMED_DATA);
    }
    let data = reorder_data(data, &shape, usize::from(bits / 8), fortran_order, false)?;
    let packet = match encode_tensor_packet(code, bits, &shape, &data) {
        Ok(packet) => packet,
        Err(status) => return Err(status),
    };
    Ok(DecodedNpy {
        packet,
        fortran_order,
    })
}

unsafe fn import_decoded_npy(
    decoded: DecodedNpy,
    out_array: *mut *mut crate::ffi::TerlanNdArray,
) -> i32 {
    let mut c_ordered = std::ptr::null_mut();
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_import_tensor_packet(
            decoded.packet.as_ptr(),
            decoded.packet.len() as i64,
            &mut c_ordered,
        )
    };
    if status != OK {
        return status;
    }
    if !decoded.fortran_order {
        unsafe { *out_array = c_ordered };
        return OK;
    }
    let status = unsafe { crate::ffi::terlan_ndarray_v1_copy_ordered(c_ordered, 1, out_array) };
    let delete_status = unsafe { crate::ffi::terlan_ndarray_v1_delete(c_ordered) };
    if status == OK && delete_status != OK {
        return delete_status;
    }
    status
}

/// Decodes bounded NumPy `.npy` bytes into an owned CPU array.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_from_npy(
    bytes: *const u8,
    length: i64,
    maximum: i64,
    out_array: *mut *mut crate::ffi::TerlanNdArray,
) -> i32 {
    if out_array.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *out_array = std::ptr::null_mut() };
    let bytes = match bounded_input(bytes, length, maximum) {
        Ok(bytes) => bytes,
        Err(status) => return status,
    };
    match decode_npy(bytes) {
        Ok(decoded) => unsafe { import_decoded_npy(decoded, out_array) },
        Err(status) => status,
    }
}

unsafe fn path_input<'a>(path: *const std::ffi::c_char) -> Result<&'a std::path::Path, i32> {
    if path.is_null() {
        return Err(INVALID_ARGUMENT);
    }
    let text = unsafe { std::ffi::CStr::from_ptr(path) }
        .to_str()
        .map_err(|_| INVALID_ARGUMENT)?;
    if text.is_empty() {
        return Err(INVALID_ARGUMENT);
    }
    Ok(std::path::Path::new(text))
}

/// Saves an array to an exact filesystem path in NumPy `.npy` format.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_save_npy(
    array: *mut crate::ffi::TerlanNdArray,
    path: *const std::ffi::c_char,
) -> i32 {
    let path = match unsafe { path_input(path) } {
        Ok(path) => path,
        Err(status) => return status,
    };
    let fortran_order = match unsafe { array_uses_fortran_order(array) } {
        Ok(order) => order,
        Err(status) => return status,
    };
    let packet = match unsafe { export_packet(array) }.and_then(|bytes| parse_tensor_packet(&bytes))
    {
        Ok(packet) => packet,
        Err(status) => return status,
    };
    let bytes = match encode_npy(packet, fortran_order) {
        Ok(bytes) => bytes,
        Err(status) => return status,
    };
    std::fs::write(path, bytes).map_or(IO_ERROR, |_| OK)
}

/// Loads a bounded NumPy `.npy` file from an exact filesystem path.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_load_npy(
    path: *const std::ffi::c_char,
    maximum: i64,
    out_array: *mut *mut crate::ffi::TerlanNdArray,
) -> i32 {
    if out_array.is_null() || maximum < 0 {
        return INVALID_ARGUMENT;
    }
    unsafe { *out_array = std::ptr::null_mut() };
    let path = match unsafe { path_input(path) } {
        Ok(path) => path,
        Err(status) => return status,
    };
    let length = match std::fs::metadata(path) {
        Ok(metadata) => metadata.len(),
        Err(_) => return IO_ERROR,
    };
    if length > maximum as u64 {
        return INVALID_ARGUMENT;
    }
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(_) => return IO_ERROR,
    };
    match decode_npy(&bytes) {
        Ok(decoded) => unsafe { import_decoded_npy(decoded, out_array) },
        Err(status) => status,
    }
}

/// Encodes an array as a versioned checksummed package-native packet.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_serialize(
    array: *mut crate::ffi::TerlanNdArray,
    out_bytes: *mut *mut u8,
    out_length: *mut usize,
) -> i32 {
    let packet = match unsafe { export_packet(array) } {
        Ok(packet) => packet,
        Err(status) => return status,
    };
    let length = match u64::try_from(packet.len()) {
        Ok(length) => length,
        Err(_) => return SIZE_OVERFLOW,
    };
    let mut output = Vec::with_capacity(RAW_HEADER + packet.len());
    output.extend_from_slice(b"TNAR");
    output.extend_from_slice(&1u32.to_le_bytes());
    output.extend_from_slice(&length.to_le_bytes());
    output.extend_from_slice(&crc32fast::hash(&packet).to_le_bytes());
    output.extend_from_slice(&packet);
    unsafe { transfer_bytes(output, out_bytes, out_length) }
}

/// Decodes a bounded checksummed package-native packet into an owned CPU array.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_deserialize(
    bytes: *const u8,
    length: i64,
    maximum: i64,
    out_array: *mut *mut crate::ffi::TerlanNdArray,
) -> i32 {
    if out_array.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *out_array = std::ptr::null_mut() };
    let bytes = match bounded_input(bytes, length, maximum) {
        Ok(bytes) => bytes,
        Err(status) => return status,
    };
    if bytes.len() < RAW_HEADER || &bytes[..4] != b"TNAR" || read_u32(&bytes[4..]) != Some(1) {
        return MALFORMED_DATA;
    }
    let packet_length = match read_u64(&bytes[8..]).and_then(|value| usize::try_from(value).ok()) {
        Some(value) => value,
        None => return SIZE_OVERFLOW,
    };
    if packet_length != bytes.len() - RAW_HEADER {
        return MALFORMED_DATA;
    }
    let packet = &bytes[RAW_HEADER..];
    if read_u32(&bytes[16..]) != Some(crc32fast::hash(packet)) {
        return INTEGRITY_ERROR;
    }
    unsafe {
        crate::ffi::terlan_ndarray_v1_import_tensor_packet(
            packet.as_ptr(),
            packet.len() as i64,
            out_array,
        )
    }
}

enum Mapping {
    Read(memmap2::Mmap),
    Write(memmap2::MmapMut),
}

impl Mapping {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Read(mapping) => mapping,
            Self::Write(mapping) => mapping,
        }
    }

    fn data_pointer(&mut self, offset: usize) -> *mut std::ffi::c_void {
        match self {
            Self::Read(mapping) => unsafe { mapping.as_ptr().add(offset) }.cast_mut().cast(),
            Self::Write(mapping) => unsafe { mapping.as_mut_ptr().add(offset) }.cast(),
        }
    }

    fn flush(&self) -> i32 {
        match self {
            Self::Read(_) => OK,
            Self::Write(mapping) => mapping.flush().map_or(IO_ERROR, |_| OK),
        }
    }
}

unsafe extern "C" fn release_mapping(context: *mut std::ffi::c_void) {
    if !context.is_null() {
        drop(unsafe { Box::from_raw(context.cast::<Mapping>()) });
    }
}

unsafe extern "C" fn flush_mapping(context: *mut std::ffi::c_void) -> i32 {
    if context.is_null() {
        INVALID_ARGUMENT
    } else {
        unsafe { &*context.cast::<Mapping>() }.flush()
    }
}

unsafe extern "C" {
    fn terlan_ndarray_create_external_ordered(
        data: *mut std::ffi::c_void,
        byte_count: i64,
        shape: *const i64,
        rank: i64,
        dtype: i32,
        writable: i32,
        order: i32,
        context: *mut std::ffi::c_void,
        release: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,
        flush: Option<unsafe extern "C" fn(*mut std::ffi::c_void) -> i32>,
        out_array: *mut *mut crate::ffi::TerlanNdArray,
    ) -> i32;
}

/// Opens a bounded NumPy file as a read-only or read-write memory-backed array.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_mmap_npy(
    path: *const std::ffi::c_char,
    writable: i32,
    maximum: i64,
    out_array: *mut *mut crate::ffi::TerlanNdArray,
) -> i32 {
    if path.is_null() || out_array.is_null() || maximum < 0 || !matches!(writable, 0 | 1) {
        return INVALID_ARGUMENT;
    }
    unsafe { *out_array = std::ptr::null_mut() };
    let path = match unsafe { std::ffi::CStr::from_ptr(path) }.to_str() {
        Ok(path) if !path.is_empty() => path,
        _ => return INVALID_ARGUMENT,
    };
    let file = match std::fs::OpenOptions::new()
        .read(true)
        .write(writable == 1)
        .open(path)
    {
        Ok(file) => file,
        Err(_) => return IO_ERROR,
    };
    let file_length = match file
        .metadata()
        .and_then(|metadata| i64::try_from(metadata.len()).map_err(std::io::Error::other))
    {
        Ok(length) if length <= maximum => length,
        Ok(_) => return INVALID_ARGUMENT,
        Err(_) => return IO_ERROR,
    };
    if file_length == 0 {
        return MALFORMED_DATA;
    }
    let mapping = if writable == 1 {
        match unsafe { memmap2::MmapOptions::new().map_mut(&file) } {
            Ok(mapping) => Mapping::Write(mapping),
            Err(_) => return IO_ERROR,
        }
    } else {
        match unsafe { memmap2::MmapOptions::new().map(&file) } {
            Ok(mapping) => Mapping::Read(mapping),
            Err(_) => return IO_ERROR,
        }
    };
    let file = match NpyFile::new(Cursor::new(mapping.bytes())) {
        Ok(file) => file,
        Err(_) => return MALFORMED_DATA,
    };
    let order = if file.order() == Order::Fortran { 1 } else { 0 };
    let (code, bits) = match dlpack_dtype(file.dtype()) {
        Ok(dtype) => dtype,
        Err(status) => return status,
    };
    let dtype = match ndarray_dtype(code, bits) {
        Ok(dtype) => dtype,
        Err(status) => return status,
    };
    let shape = match file
        .shape()
        .iter()
        .map(|value| i64::try_from(*value).map_err(|_| SIZE_OVERFLOW))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(shape) => shape,
        Err(status) => return status,
    };
    let offset = match npy_payload_offset(mapping.bytes()) {
        Ok(offset) => offset,
        Err(status) => return status,
    };
    let byte_count = mapping.bytes().len() - offset;
    let byte_count = match i64::try_from(byte_count) {
        Ok(length) => length,
        Err(_) => return SIZE_OVERFLOW,
    };
    let mut owner = Box::new(mapping);
    let data = owner.data_pointer(offset);
    let context = Box::into_raw(owner).cast();
    let status = unsafe {
        terlan_ndarray_create_external_ordered(
            data,
            byte_count,
            shape.as_ptr(),
            shape.len() as i64,
            dtype,
            writable,
            order,
            context,
            Some(release_mapping),
            Some(flush_mapping),
            out_array,
        )
    };
    if status != OK {
        unsafe { release_mapping(context) };
    }
    status
}

#[cfg(test)]
#[path = "ndarray_serialization_tests.rs"]
mod tests;
