//! NumPy-compatible NaN, bit-packing, and byte-ingress operations.

use rustfft::num_complex::Complex64;

use super::native_support::{
    delete_arrays, dtype, element_count, ints, normalize_axis, numeric, owned_int_as,
    owned_numeric_as, shape, write_result, Array, DOMAIN_ERROR, INVALID_ARGUMENT, OK,
    SIZE_OVERFLOW,
};

/// Decodes raw bytes as an owned UInt8 vector.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_frombuffer(
    bytes: *const u8,
    length: i64,
    output: *mut Array,
) -> i32 {
    let length = match usize::try_from(length) {
        Ok(value) => value,
        Err(_) => return INVALID_ARGUMENT,
    };
    if length > 0 && bytes.is_null() {
        return INVALID_ARGUMENT;
    }
    let values = if length == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(bytes, length) }
            .iter()
            .map(|value| i64::from(*value))
            .collect::<Vec<_>>()
    };
    let mut array = std::ptr::null_mut();
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_from_ints_as(
            values.as_ptr(),
            length as i64,
            &[length as i64] as *const i64,
            1,
            7,
            &mut array,
        )
    };
    if status != OK {
        return status;
    }
    unsafe { write_result(Ok(array), output) }
}

/// Returns the flattened index of the maximum non-NaN value.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_nanargmax(array: Array, output: *mut i64) -> i32 {
    unsafe { nanarg(array, true, output) }
}

/// Returns the flattened index of the minimum non-NaN value.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_nanargmin(array: Array, output: *mut i64) -> i32 {
    unsafe { nanarg(array, false, output) }
}

/// Computes one flattened non-NaN extremum index.
unsafe fn nanarg(array: Array, maximum: bool, output: *mut i64) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    let values = match unsafe { super::native_support::floats(array, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let result = values
        .iter()
        .enumerate()
        .filter(|(_, value)| !value.is_nan())
        .reduce(|lhs, rhs| {
            if (maximum && rhs.1 > lhs.1) || (!maximum && rhs.1 < lhs.1) {
                rhs
            } else {
                lhs
            }
        });
    match result {
        Some((index, _)) => match i64::try_from(index) {
            Ok(index) => {
                unsafe { *output = index };
                OK
            }
            Err(_) => SIZE_OVERFLOW,
        },
        None => DOMAIN_ERROR,
    }
}

/// Returns a cumulative sum after replacing NaNs with zero.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_nancumsum(
    array: Array,
    axis: i64,
    output: *mut Array,
) -> i32 {
    unsafe { nan_cumulative(array, axis, false, 0, output) }
}

/// Returns a cumulative product after replacing NaNs with one.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_nancumprod(
    array: Array,
    axis: i64,
    output: *mut Array,
) -> i32 {
    unsafe { nan_cumulative(array, axis, true, 0, output) }
}

/// Runs one existing cumulative kernel over NaN-normalized values.
unsafe fn nan_cumulative(
    array: Array,
    axis: i64,
    product: bool,
    requested_dtype: i32,
    output: *mut Array,
) -> i32 {
    let input_dtype = match unsafe { dtype(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if matches!(input_dtype, 1 | 2 | 4..=10) {
        return unsafe { nan_cumulative_dispatch(array, axis, product, requested_dtype, output) };
    }
    let dimensions = match unsafe { shape(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = match unsafe { numeric(array, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let identity = if product { 1.0 } else { 0.0 };
    let normalized = values
        .values
        .iter()
        .map(|value| {
            Complex64::new(
                if value.re.is_nan() {
                    identity
                } else {
                    value.re
                },
                if value.im.is_nan() {
                    identity
                } else {
                    value.im
                },
            )
        })
        .collect::<Vec<_>>();
    let temporary = match unsafe { owned_numeric_as(&normalized, &dimensions, input_dtype) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let status =
        unsafe { nan_cumulative_dispatch(temporary, axis, product, requested_dtype, output) };
    unsafe { delete_arrays(&[temporary]) };
    status
}

/// Dispatches one cumulative operation with an optional explicit output dtype.
unsafe fn nan_cumulative_dispatch(
    array: Array,
    axis: i64,
    product: bool,
    requested_dtype: i32,
    output: *mut Array,
) -> i32 {
    if product && requested_dtype == 0 {
        unsafe { crate::ffi::terlan_ndarray_v1_cumulative_product(array, axis, output) }
    } else if product {
        unsafe {
            crate::ffi::terlan_ndarray_v1_cumulative_product_as(
                array,
                axis,
                requested_dtype,
                output,
            )
        }
    } else if requested_dtype == 0 {
        unsafe { crate::ffi::terlan_ndarray_v1_cumulative_sum(array, axis, output) }
    } else {
        unsafe {
            crate::ffi::terlan_ndarray_v1_cumulative_sum_as(array, axis, requested_dtype, output)
        }
    }
}

/// Runs one cumulative operation after flattening in logical C order.
unsafe fn nan_cumulative_flat(
    array: Array,
    product: bool,
    requested_dtype: i32,
    output: *mut Array,
) -> i32 {
    let mut flattened = std::ptr::null_mut();
    let status = unsafe { crate::ffi::terlan_ndarray_v1_flatten(array, &mut flattened) };
    if status != OK {
        return status;
    }
    let status = unsafe { nan_cumulative(flattened, 0, product, requested_dtype, output) };
    unsafe { delete_arrays(&[flattened]) };
    status
}

/// Writes one cumulative result into an existing exact-shape destination.
unsafe fn nan_cumulative_into(
    destination: Array,
    array: Array,
    axis: Option<i64>,
    product: bool,
) -> i32 {
    let mut result = std::ptr::null_mut();
    let status = match axis {
        Some(axis) => unsafe { nan_cumulative(array, axis, product, 0, &mut result) },
        None => unsafe { nan_cumulative_flat(array, product, 0, &mut result) },
    };
    if status != OK {
        return status;
    }
    let status = unsafe { crate::ffi::terlan_ndarray_v1_copyto_casting(destination, result, 4) };
    unsafe { delete_arrays(&[result]) };
    status
}

/// Returns a flattened cumulative sum after replacing NaNs with zero.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_nancumsum_flat(array: Array, output: *mut Array) -> i32 {
    unsafe { nan_cumulative_flat(array, false, 0, output) }
}

/// Returns a flattened cumulative product after replacing NaNs with one.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_nancumprod_flat(
    array: Array,
    output: *mut Array,
) -> i32 {
    unsafe { nan_cumulative_flat(array, true, 0, output) }
}

/// Returns a cumulative sum in an explicit output dtype.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_nancumsum_as(
    array: Array,
    axis: i64,
    requested_dtype: i32,
    output: *mut Array,
) -> i32 {
    unsafe { nan_cumulative(array, axis, false, requested_dtype, output) }
}

/// Returns a cumulative product in an explicit output dtype.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_nancumprod_as(
    array: Array,
    axis: i64,
    requested_dtype: i32,
    output: *mut Array,
) -> i32 {
    unsafe { nan_cumulative(array, axis, true, requested_dtype, output) }
}

/// Writes a cumulative sum into an existing exact-shape destination.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_nancumsum_into(
    destination: Array,
    array: Array,
    axis: i64,
) -> i32 {
    unsafe { nan_cumulative_into(destination, array, Some(axis), false) }
}

/// Writes a cumulative product into an existing exact-shape destination.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_nancumprod_into(
    destination: Array,
    array: Array,
    axis: i64,
) -> i32 {
    unsafe { nan_cumulative_into(destination, array, Some(axis), true) }
}

/// Writes one flattened cumulative sum into an existing vector destination.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_nancumsum_flat_into(
    destination: Array,
    array: Array,
) -> i32 {
    unsafe { nan_cumulative_into(destination, array, None, false) }
}

/// Writes one flattened cumulative product into an existing vector destination.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_nancumprod_flat_into(
    destination: Array,
    array: Array,
) -> i32 {
    unsafe { nan_cumulative_into(destination, array, None, true) }
}

/// Returns the median of all non-NaN flattened values.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_nanmedian(array: Array, output: *mut Array) -> i32 {
    unsafe { crate::ffi::terlan_ndarray_v1_nanmedian_axes(array, std::ptr::null(), 0, 0, output) }
}

/// Returns a percentile of all non-NaN flattened values.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_nanpercentile(
    array: Array,
    percentile: f64,
    output: *mut Array,
) -> i32 {
    unsafe {
        crate::ffi::terlan_ndarray_v1_nanpercentile_axes(
            array,
            percentile,
            std::ptr::null(),
            0,
            0,
            0,
            output,
        )
    }
}

/// Returns a quantile of all non-NaN flattened values.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_nanquantile(
    array: Array,
    quantile: f64,
    output: *mut Array,
) -> i32 {
    unsafe {
        crate::ffi::terlan_ndarray_v1_nanquantile_axes(
            array,
            quantile,
            std::ptr::null(),
            0,
            0,
            0,
            output,
        )
    }
}

/// Packs truth values into UInt8 bytes along one optional axis.
unsafe fn packbits(array: Array, axis: Option<i64>, bit_order: i64, output: *mut Array) -> i32 {
    if !matches!(bit_order, 0 | 1) {
        return INVALID_ARGUMENT;
    }
    match unsafe { dtype(array) } {
        Ok(1 | 2 | 4..=10) => {}
        Ok(_) => return DOMAIN_ERROR,
        Err(status) => return status,
    }
    let values = match unsafe { ints(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let dimensions = match axis {
        Some(_) => match unsafe { shape(array) } {
            Ok(value) => value,
            Err(status) => return status,
        },
        None => vec![values.len() as i64],
    };
    let axis = match normalize_axis(axis.unwrap_or(0), dimensions.len()) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let axis_length = dimensions[axis] as usize;
    let packed_axis = axis_length.div_ceil(8);
    let outer = match element_count(&dimensions[..axis]) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let inner = match element_count(&dimensions[axis + 1..]) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let output_count = match outer
        .checked_mul(packed_axis)
        .and_then(|value| value.checked_mul(inner))
    {
        Some(value) => value,
        None => return SIZE_OVERFLOW,
    };
    let mut packed = vec![0i64; output_count];
    for outer_index in 0..outer {
        for byte_index in 0..packed_axis {
            for inner_index in 0..inner {
                let output_index = (outer_index * packed_axis + byte_index) * inner + inner_index;
                for offset in 0..8 {
                    let axis_index = byte_index * 8 + offset;
                    if axis_index >= axis_length {
                        break;
                    }
                    let input_index =
                        (outer_index * axis_length + axis_index) * inner + inner_index;
                    if values[input_index] != 0 {
                        let bit = if bit_order == 0 { 7 - offset } else { offset };
                        packed[output_index] |= 1 << bit;
                    }
                }
            }
        }
    }
    let mut output_shape = dimensions;
    output_shape[axis] = packed_axis as i64;
    unsafe { write_result(owned_int_as(&packed, &output_shape, 7), output) }
}

/// Packs flattened truth values into UInt8 bytes.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_packbits(
    array: Array,
    bit_order: i64,
    output: *mut Array,
) -> i32 {
    unsafe { packbits(array, None, bit_order, output) }
}

/// Packs truth values along one normalized axis.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_packbits_axis(
    array: Array,
    axis: i64,
    bit_order: i64,
    output: *mut Array,
) -> i32 {
    unsafe { packbits(array, Some(axis), bit_order, output) }
}

/// Unpacks UInt8 values along one optional axis and optional output count.
unsafe fn unpackbits(
    array: Array,
    axis: Option<i64>,
    count: Option<i64>,
    bit_order: i64,
    output: *mut Array,
) -> i32 {
    if !matches!(bit_order, 0 | 1) {
        return INVALID_ARGUMENT;
    }
    match unsafe { dtype(array) } {
        Ok(7) => {}
        Ok(_) => return DOMAIN_ERROR,
        Err(status) => return status,
    }
    let values = match unsafe { ints(array) } {
        Ok(value) if value.iter().all(|value| (0..=255).contains(value)) => value,
        Ok(_) => return DOMAIN_ERROR,
        Err(status) => return status,
    };
    let dimensions = match axis {
        Some(_) => match unsafe { shape(array) } {
            Ok(value) => value,
            Err(status) => return status,
        },
        None => vec![values.len() as i64],
    };
    let axis = match normalize_axis(axis.unwrap_or(0), dimensions.len()) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let axis_length = dimensions[axis] as usize;
    let available = match axis_length.checked_mul(8) {
        Some(value) => value,
        None => return SIZE_OVERFLOW,
    };
    let requested = match count {
        None => available,
        Some(value) if value >= 0 => match usize::try_from(value) {
            Ok(value) => value,
            Err(_) => return SIZE_OVERFLOW,
        },
        Some(value) => match usize::try_from(value.unsigned_abs()) {
            Ok(remove) if remove <= available => available - remove,
            _ => return INVALID_ARGUMENT,
        },
    };
    let outer = match element_count(&dimensions[..axis]) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let inner = match element_count(&dimensions[axis + 1..]) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let output_count = match outer
        .checked_mul(requested)
        .and_then(|value| value.checked_mul(inner))
    {
        Some(value) => value,
        None => return SIZE_OVERFLOW,
    };
    let mut unpacked = vec![0i64; output_count];
    for outer_index in 0..outer {
        for bit_index in 0..requested {
            let byte_index = bit_index / 8;
            let bit = if bit_order == 0 {
                7 - bit_index % 8
            } else {
                bit_index % 8
            };
            for inner_index in 0..inner {
                let output_index = (outer_index * requested + bit_index) * inner + inner_index;
                if bit_index < available {
                    let input_index =
                        (outer_index * axis_length + byte_index) * inner + inner_index;
                    unpacked[output_index] = (values[input_index] >> bit) & 1;
                }
            }
        }
    }
    let requested_dimension = match i64::try_from(requested) {
        Ok(value) => value,
        Err(_) => return SIZE_OVERFLOW,
    };
    let mut output_shape = dimensions;
    output_shape[axis] = requested_dimension;
    unsafe { write_result(owned_int_as(&unpacked, &output_shape, 7), output) }
}

/// Unpacks a counted number of flattened UInt8 bits.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_unpackbits(
    array: Array,
    count: i64,
    bit_order: i64,
    output: *mut Array,
) -> i32 {
    unsafe { unpackbits(array, None, Some(count), bit_order, output) }
}

/// Unpacks every available flattened UInt8 bit.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_unpackbits_all(
    array: Array,
    bit_order: i64,
    output: *mut Array,
) -> i32 {
    unsafe { unpackbits(array, None, None, bit_order, output) }
}

/// Unpacks a counted number of bits along one normalized axis.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_unpackbits_axis(
    array: Array,
    axis: i64,
    count: i64,
    bit_order: i64,
    output: *mut Array,
) -> i32 {
    unsafe { unpackbits(array, Some(axis), Some(count), bit_order, output) }
}

/// Unpacks every available bit along one normalized axis.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_unpackbits_axis_all(
    array: Array,
    axis: i64,
    bit_order: i64,
    output: *mut Array,
) -> i32 {
    unsafe { unpackbits(array, Some(axis), None, bit_order, output) }
}

#[cfg(test)]
#[path = "ndarray_numpy_data_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "ndarray_nan_reduction_tests.rs"]
mod nan_reduction_tests;
