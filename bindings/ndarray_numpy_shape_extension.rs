//! NumPy-compatible shape, indexing, stacking, and mutation completion.

use super::native_support::{
    delete_arrays, dtype, ints, normalize_axis, owned_bool, owned_int, shape, transfer_arrays,
    write_result, Array, AXIS_OUT_OF_RANGE, DTYPE_MISMATCH, INVALID_ARGUMENT, OK, SHAPE_MISMATCH,
};
use super::polynomial_power_extension::broadcast_shape;

/// Reads one C handle array without assuming handle layout beyond the ABI identity.
pub(crate) unsafe fn handles(values: *const u64, count: i64) -> Result<Vec<Array>, i32> {
    let count = usize::try_from(count).map_err(|_| INVALID_ARGUMENT)?;
    if count > 0 && values.is_null() {
        return Err(INVALID_ARGUMENT);
    }
    let values = if count == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(values, count) }
    };
    let arrays = values
        .iter()
        .map(|value| *value as usize as Array)
        .collect::<Vec<_>>();
    if arrays.iter().any(|array| array.is_null()) {
        Err(INVALID_ARGUMENT)
    } else {
        Ok(arrays)
    }
}

/// Returns the first-axis length, or one for a scalar.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_alen(array: Array, output: *mut i64) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    match unsafe { shape(array) } {
        Ok(dimensions) if !dimensions.is_empty() => {
            unsafe { *output = dimensions[0] };
            OK
        }
        Ok(_) => {
            unsafe { *output = 1 };
            OK
        }
        Err(status) => status,
    }
}

/// Appends values along one axis, with `-1` selecting flattened append.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_append(
    array: Array,
    values: Array,
    axis: i64,
    output: *mut Array,
) -> i32 {
    let mut flattened = [array, values];
    if axis == -1 {
        for (index, source) in [array, values].into_iter().enumerate() {
            let status =
                unsafe { crate::ffi::terlan_ndarray_v1_ravel(source, &mut flattened[index]) };
            if status != OK {
                unsafe { delete_arrays(&flattened[..index]) };
                return status;
            }
        }
    }
    let identities = flattened.map(|value| value as usize as u64);
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_concatenate(
            identities.as_ptr(),
            2,
            if axis == -1 { 0 } else { axis },
            output,
        )
    };
    if axis == -1 {
        unsafe { delete_arrays(&flattened) };
    }
    status
}

/// Broadcasts one first array and a list of peers to a shared shape.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_broadcast_arrays(
    first: Array,
    remaining: *const u64,
    remaining_count: i64,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let mut arrays = vec![first];
    match unsafe { handles(remaining, remaining_count) } {
        Ok(mut value) => arrays.append(&mut value),
        Err(status) => return status,
    };
    let shapes = match arrays
        .iter()
        .map(|array| unsafe { shape(*array) })
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(value) => value,
        Err(status) => return status,
    };
    let borrowed = shapes.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let output_shape = match broadcast_shape(&borrowed) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut results = Vec::with_capacity(arrays.len());
    for array in arrays {
        let mut result = std::ptr::null_mut();
        let status = unsafe {
            crate::ffi::terlan_ndarray_v1_broadcast_to(
                array,
                output_shape.as_ptr(),
                output_shape.len() as i64,
                &mut result,
            )
        };
        if status != OK {
            unsafe { delete_arrays(&results) };
            return status;
        }
        results.push(result);
    }
    unsafe { transfer_arrays(results, output, output_length) }
}

/// Returns the broadcast result for two explicit shape vectors.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_broadcast_shapes(
    lhs: *const i64,
    lhs_count: i64,
    rhs: *const i64,
    rhs_count: i64,
    output: *mut Array,
) -> i32 {
    let (lhs_count, rhs_count) = match (usize::try_from(lhs_count), usize::try_from(rhs_count)) {
        (Ok(lhs), Ok(rhs)) => (lhs, rhs),
        _ => return INVALID_ARGUMENT,
    };
    if (lhs_count > 0 && lhs.is_null()) || (rhs_count > 0 && rhs.is_null()) {
        return INVALID_ARGUMENT;
    }
    let lhs = if lhs_count == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(lhs, lhs_count) }
    };
    let rhs = if rhs_count == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(rhs, rhs_count) }
    };
    let result = match broadcast_shape(&[lhs, rhs]) {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe { write_result(owned_int(&result, &[result.len() as i64]), output) }
}

/// Stacks columns after promoting rank-one inputs to column vectors.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_column_stack(
    arrays: *const u64,
    count: i64,
    output: *mut Array,
) -> i32 {
    unsafe { promoted_concatenate(arrays, count, 1, 1, output) }
}

/// Stacks arrays depth-wise after rank-three promotion.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_dstack(
    arrays: *const u64,
    count: i64,
    output: *mut Array,
) -> i32 {
    unsafe { promoted_concatenate(arrays, count, 3, 2, output) }
}

/// Stacks arrays horizontally using NumPy rank-one axis behavior.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_hstack(
    arrays: *const u64,
    count: i64,
    output: *mut Array,
) -> i32 {
    let arrays = match unsafe { handles(arrays, count) } {
        Ok(value) if !value.is_empty() => value,
        Ok(_) => return INVALID_ARGUMENT,
        Err(status) => return status,
    };
    let mut promoted = Vec::with_capacity(arrays.len());
    for array in arrays {
        let mut value = std::ptr::null_mut();
        let status = unsafe { crate::ffi::terlan_ndarray_v1_atleast_1d(array, &mut value) };
        if status != OK {
            unsafe { delete_arrays(&promoted) };
            return status;
        }
        promoted.push(value);
    }
    let rank = match unsafe { shape(promoted[0]) } {
        Ok(value) => value.len(),
        Err(status) => {
            unsafe { delete_arrays(&promoted) };
            return status;
        }
    };
    let identities = promoted
        .iter()
        .map(|value| *value as usize as u64)
        .collect::<Vec<_>>();
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_concatenate(
            identities.as_ptr(),
            identities.len() as i64,
            if rank == 1 { 0 } else { 1 },
            output,
        )
    };
    unsafe { delete_arrays(&promoted) };
    status
}

/// Stacks arrays vertically after rank-two promotion.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_vstack(
    arrays: *const u64,
    count: i64,
    output: *mut Array,
) -> i32 {
    unsafe { promoted_concatenate(arrays, count, 2, 0, output) }
}

/// Alias for vertical row stacking.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_row_stack(
    arrays: *const u64,
    count: i64,
    output: *mut Array,
) -> i32 {
    unsafe { terlan_ndarray_v1_vstack(arrays, count, output) }
}

/// Promotes inputs and concatenates them on one axis.
unsafe fn promoted_concatenate(
    arrays: *const u64,
    count: i64,
    rank: i32,
    axis: i64,
    output: *mut Array,
) -> i32 {
    let arrays = match unsafe { handles(arrays, count) } {
        Ok(value) if !value.is_empty() => value,
        Ok(_) => return INVALID_ARGUMENT,
        Err(status) => return status,
    };
    let mut promoted = Vec::with_capacity(arrays.len());
    for array in arrays {
        let mut value = std::ptr::null_mut();
        let status = match rank {
            1 => {
                let dimensions = match unsafe { shape(array) } {
                    Ok(value) => value,
                    Err(status) => return status,
                };
                if dimensions.len() == 1 {
                    unsafe { crate::ffi::terlan_ndarray_v1_expand_dims(array, 1, &mut value) }
                } else {
                    unsafe { crate::ffi::terlan_ndarray_v1_copy(array, &mut value) }
                }
            }
            2 => unsafe { crate::ffi::terlan_ndarray_v1_atleast_2d(array, &mut value) },
            3 => unsafe { crate::ffi::terlan_ndarray_v1_atleast_3d(array, &mut value) },
            _ => return INVALID_ARGUMENT,
        };
        if status != OK {
            unsafe { delete_arrays(&promoted) };
            return status;
        }
        promoted.push(value);
    }
    let identities = promoted
        .iter()
        .map(|value| *value as usize as u64)
        .collect::<Vec<_>>();
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_concatenate(
            identities.as_ptr(),
            promoted.len() as i64,
            axis,
            output,
        )
    };
    unsafe { delete_arrays(&promoted) };
    status
}

/// Deletes flattened indexes from a rank-one array.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_delete_indexes(
    array: Array,
    indexes: *const i64,
    index_count: i64,
    output: *mut Array,
) -> i32 {
    unsafe { terlan_ndarray_v1_delete_indexes_axis(array, indexes, index_count, 0, output) }
}

/// Deletes normalized integer indexes along one axis without changing scalar storage.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_delete_indexes_axis(
    array: Array,
    indexes: *const i64,
    index_count: i64,
    axis: i64,
    output: *mut Array,
) -> i32 {
    let dimensions = match unsafe { shape(array) } {
        Ok(value) if !value.is_empty() => value,
        Ok(_) => return AXIS_OUT_OF_RANGE,
        Err(status) => return status,
    };
    let axis = match normalize_axis(axis, dimensions.len()) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let count = match usize::try_from(index_count) {
        Ok(value) => value,
        Err(_) => return INVALID_ARGUMENT,
    };
    if count > 0 && indexes.is_null() {
        return INVALID_ARGUMENT;
    }
    let indexes = if count == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(indexes, count) }
    };
    if count == 0 {
        return unsafe { crate::ffi::terlan_ndarray_v1_copy(array, output) };
    }
    let mut removed = vec![false; dimensions[axis] as usize];
    for &index in indexes {
        let index = if index < 0 {
            dimensions[axis] + index
        } else {
            index
        };
        let index = match usize::try_from(index) {
            Ok(value) if value < removed.len() => value,
            _ => return AXIS_OUT_OF_RANGE,
        };
        removed[index] = true;
    }
    let keep = removed
        .iter()
        .map(|removed| u8::from(!removed))
        .collect::<Vec<_>>();
    let mask = match unsafe { owned_bool(&keep, &[keep.len() as i64]) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let status =
        unsafe { crate::ffi::terlan_ndarray_v1_compress(array, mask, axis as i64, output) };
    unsafe { delete_arrays(&[mask]) };
    status
}

/// Deletes positions selected by an exact-length Bool mask along one axis.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_delete_mask_axis(
    array: Array,
    mask: Array,
    axis: i64,
    output: *mut Array,
) -> i32 {
    let dimensions = match unsafe { shape(array) } {
        Ok(value) if !value.is_empty() => value,
        Ok(_) => return AXIS_OUT_OF_RANGE,
        Err(status) => return status,
    };
    let axis = match normalize_axis(axis, dimensions.len()) {
        Ok(value) => value,
        Err(status) => return status,
    };
    if unsafe { dtype(mask) } != Ok(1) {
        return DTYPE_MISMATCH;
    }
    let mask_shape = match unsafe { shape(mask) } {
        Ok(value) if value.len() == 1 && value[0] == dimensions[axis] => value,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let removed = match unsafe { ints(mask) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if removed.iter().all(|value| *value == 0) {
        return unsafe { crate::ffi::terlan_ndarray_v1_copy(array, output) };
    }
    let keep = removed
        .iter()
        .map(|value| u8::from(*value == 0))
        .collect::<Vec<_>>();
    let keep_mask = match unsafe { owned_bool(&keep, &mask_shape) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let status =
        unsafe { crate::ffi::terlan_ndarray_v1_compress(array, keep_mask, axis as i64, output) };
    unsafe { delete_arrays(&[keep_mask]) };
    status
}

/// Extracts a diagonal or creates a diagonal matrix from rank-one values.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_diag(
    array: Array,
    offset: i64,
    output: *mut Array,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    let dimensions = match unsafe { shape(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if dimensions.len() == 2 {
        return unsafe { crate::ffi::terlan_ndarray_v1_diagonal(array, offset, 0, 1, output) };
    }
    if dimensions.len() == 1 {
        return unsafe { crate::ffi::terlan_ndarray_v1_diagflat(array, offset, output) };
    }
    SHAPE_MISMATCH
}

/// Returns repeated diagonal indexes for a square rank.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_diag_indices(
    size: i64,
    rank: i64,
    output: *mut Array,
) -> i32 {
    let (size, rank) = match (usize::try_from(size), usize::try_from(rank)) {
        (Ok(size), Ok(rank)) if rank >= 2 => (size, rank),
        _ => return INVALID_ARGUMENT,
    };
    let values = (0..rank).flat_map(|_| 0..size as i64).collect::<Vec<_>>();
    unsafe { write_result(owned_int(&values, &[rank as i64, size as i64]), output) }
}

/// Returns diagonal indexes matching a square array's rank.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_diag_indices_from(
    array: Array,
    output: *mut Array,
) -> i32 {
    let dimensions = match unsafe { shape(array) } {
        Ok(value) if value.len() >= 2 && value.iter().all(|dimension| *dimension == value[0]) => {
            value
        }
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    unsafe { terlan_ndarray_v1_diag_indices(dimensions[0], dimensions.len() as i64, output) }
}

unsafe fn split_designated_axis(
    array: Array,
    sections: i64,
    minimum_rank: usize,
    axis: i64,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    match unsafe { shape(array) } {
        Ok(dimensions) if dimensions.len() >= minimum_rank => unsafe {
            crate::ffi::terlan_ndarray_v1_split(array, sections, axis, output, output_length)
        },
        Ok(_) => SHAPE_MISMATCH,
        Err(status) => status,
    }
}

macro_rules! split_axis {
    ($name:ident, $axis:expr, $minimum_rank:expr) => {
        #[doc = "Splits an array evenly along the designated axis."]
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            array: Array,
            sections: i64,
            output: *mut *mut u64,
            output_length: *mut usize,
        ) -> i32 {
            unsafe {
                split_designated_axis(array, sections, $minimum_rank, $axis, output, output_length)
            }
        }
    };
}

split_axis!(terlan_ndarray_v1_dsplit, 2, 3);
split_axis!(terlan_ndarray_v1_vsplit, 0, 2);

/// Splits vectors on axis zero and higher-rank arrays on axis one.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_hsplit(
    array: Array,
    sections: i64,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let rank = match unsafe { shape(array) } {
        Ok(dimensions) if !dimensions.is_empty() => dimensions.len(),
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    unsafe {
        split_designated_axis(
            array,
            sections,
            1,
            if rank == 1 { 0 } else { 1 },
            output,
            output_length,
        )
    }
}

/// Writes a scalar across every diagonal of a mutable array.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_fill_diagonal(array: Array, value: f64) -> i32 {
    unsafe {
        super::index_tricks_extension::terlan_ndarray_v1_fill_diagonal_with_wrap(
            array, value, false,
        )
    }
}

/// Builds a Float64 array filled with one scalar.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_full(
    shape_values: *const i64,
    rank: i64,
    value: f64,
    output: *mut Array,
) -> i32 {
    unsafe { crate::ffi::terlan_ndarray_v1_full_float(shape_values, rank, value, 3, output) }
}

/// Builds a Float64 array matching another array's shape.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_full_like(
    array: Array,
    value: f64,
    output: *mut Array,
) -> i32 {
    unsafe { crate::ffi::terlan_ndarray_v1_full_like_float(array, value, output) }
}
