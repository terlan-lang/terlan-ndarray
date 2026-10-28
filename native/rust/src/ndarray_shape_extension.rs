//! Compositional shape operations over package-owned arrays.

use super::native_support::{
    normalize_axis, shape, Array, AXIS_OUT_OF_RANGE, INVALID_ARGUMENT, OK, SIZE_OVERFLOW,
};

const DUPLICATE_AXIS: i32 = 111;

unsafe extern "C" {
    /// Concatenates two arrays through the canonical C promotion kernel.
    fn terlan_ndarray_concatenate_pair(
        lhs: Array,
        rhs: Array,
        axis: i64,
        output: *mut Array,
    ) -> i32;
}

/// Reads a checked caller-owned slice.
unsafe fn values<'a>(pointer: *const i64, count: i64) -> Result<&'a [i64], i32> {
    let count = usize::try_from(count).map_err(|_| INVALID_ARGUMENT)?;
    if count == 0 {
        return Ok(&[]);
    }
    if pointer.is_null() {
        return Err(INVALID_ARGUMENT);
    }
    Ok(unsafe { std::slice::from_raw_parts(pointer, count) })
}

/// Decodes a nonempty list of borrowed package resource identities.
unsafe fn arrays<'a>(identities: *const u64, count: i64) -> Result<&'a [u64], i32> {
    let count = usize::try_from(count).map_err(|_| INVALID_ARGUMENT)?;
    if count == 0 || identities.is_null() {
        return Err(INVALID_ARGUMENT);
    }
    let identities = unsafe { std::slice::from_raw_parts(identities, count) };
    if identities.iter().any(|identity| *identity == 0) {
        return Err(INVALID_ARGUMENT);
    }
    Ok(identities)
}

/// Flattens every borrowed input into independently owned logical C-order arrays.
unsafe fn flattened_arrays(identities: *const u64, count: i64) -> Result<Vec<Array>, i32> {
    let identities = unsafe { arrays(identities, count) }?;
    let mut flattened = Vec::with_capacity(identities.len());
    for &identity in identities {
        let source = match array(identity) {
            Ok(value) => value,
            Err(status) => {
                unsafe { super::native_support::delete_arrays(&flattened) };
                return Err(status);
            }
        };
        let mut result = std::ptr::null_mut();
        let status = unsafe { crate::ffi::terlan_ndarray_v1_ravel(source, &mut result) };
        if status != OK {
            if !result.is_null() {
                let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(result) };
            }
            unsafe { super::native_support::delete_arrays(&flattened) };
            return Err(status);
        }
        flattened.push(result);
    }
    Ok(flattened)
}

/// Returns stable borrowed identities for package-owned temporary arrays.
fn array_identities(arrays: &[Array]) -> Vec<u64> {
    arrays.iter().map(|array| *array as usize as u64).collect()
}

/// Returns whether a numerical casting policy code belongs to the public ABI.
fn valid_casting(casting: i32) -> bool {
    (0..=4).contains(&casting)
}

/// Converts and consumes one owned temporary under an explicit casting rule.
unsafe fn cast_owned(source: Array, dtype: i32, casting: i32) -> Result<Array, i32> {
    let mut converted = std::ptr::null_mut();
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_astype_ordered(source, dtype, 2, 1, casting, &mut converted)
    };
    let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(source) };
    if status == OK {
        Ok(converted)
    } else {
        if !converted.is_null() {
            let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(converted) };
        }
        Err(status)
    }
}

/// Copies one owned temporary into an exact-shape destination and consumes it.
unsafe fn copy_owned_into(destination: Array, source: Array, casting: i32) -> i32 {
    let destination_shape = unsafe { shape(destination) };
    let source_shape = unsafe { shape(source) };
    let status = match (destination_shape, source_shape) {
        (Ok(destination_shape), Ok(source_shape)) if destination_shape == source_shape => unsafe {
            crate::ffi::terlan_ndarray_v1_copyto_casting(destination, source, casting)
        },
        (Ok(_), Ok(_)) => super::native_support::SHAPE_MISMATCH,
        (Err(status), _) | (_, Err(status)) => status,
    };
    let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(source) };
    status
}

/// Converts one checked identity into its borrowed resource pointer.
fn array(identity: u64) -> Result<Array, i32> {
    let identity = usize::try_from(identity).map_err(|_| INVALID_ARGUMENT)?;
    let array = identity as Array;
    if array.is_null() {
        Err(INVALID_ARGUMENT)
    } else {
        Ok(array)
    }
}

/// Deletes an intermediate and replaces it with the result of one view operation.
unsafe fn replace(current: &mut Array, next: Array, status: i32) -> Result<(), i32> {
    if status != OK {
        if !next.is_null() {
            let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(next) };
        }
        return Err(status);
    }
    let previous = std::mem::replace(current, next);
    let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(previous) };
    Ok(())
}

/// Writes an owned result or cleans it when the output slot is invalid.
unsafe fn produced(result: Result<Array, i32>, output: *mut Array) -> i32 {
    unsafe { super::native_support::write_result(result, output) }
}

/// Builds a shared view from slice, ellipsis, squeeze, and new-axis components.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_basic_slice(
    source: Array,
    starts: *const i64,
    start_count: i64,
    stops: *const i64,
    stop_count: i64,
    steps: *const i64,
    step_count: i64,
    ellipsis_index: i64,
    squeeze_axes: *const i64,
    squeeze_count: i64,
    new_axes: *const i64,
    new_axis_count: i64,
    output: *mut Array,
) -> i32 {
    if output.is_null() || source.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    let starts = match unsafe { values(starts, start_count) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let stops = match unsafe { values(stops, stop_count) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let steps = match unsafe { values(steps, step_count) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let squeeze_axes = match unsafe { values(squeeze_axes, squeeze_count) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let new_axes = match unsafe { values(new_axes, new_axis_count) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if starts.len() != stops.len() || starts.len() != steps.len() {
        return INVALID_ARGUMENT;
    }
    let shape = match unsafe { super::native_support::shape(source) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if starts.len() > shape.len() {
        return AXIS_OUT_OF_RANGE;
    }
    let ellipsis = if ellipsis_index == -1 {
        starts.len()
    } else {
        match usize::try_from(ellipsis_index) {
            Ok(value) if value <= starts.len() => value,
            _ => return INVALID_ARGUMENT,
        }
    };
    let omitted = shape.len() - starts.len();
    let mut current: Array = std::ptr::null_mut();
    let rank = match i64::try_from(shape.len()) {
        Ok(value) => value,
        Err(_) => return SIZE_OVERFLOW,
    };
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_reshape(source, shape.as_ptr(), rank, &mut current)
    };
    if status != OK {
        return status;
    }
    for index in 0..starts.len() {
        let axis = if index < ellipsis {
            index
        } else {
            index + omitted
        };
        let axis = match i64::try_from(axis) {
            Ok(value) => value,
            Err(_) => {
                let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(current) };
                return SIZE_OVERFLOW;
            }
        };
        let mut next = std::ptr::null_mut();
        let status = unsafe {
            crate::ffi::terlan_ndarray_v1_slice_axis(
                current,
                axis,
                starts[index],
                stops[index],
                steps[index],
                &mut next,
            )
        };
        if let Err(status) = unsafe { replace(&mut current, next, status) } {
            let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(current) };
            return status;
        }
    }
    if !squeeze_axes.is_empty() {
        let mut next = std::ptr::null_mut();
        let count = match i64::try_from(squeeze_axes.len()) {
            Ok(value) => value,
            Err(_) => {
                let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(current) };
                return SIZE_OVERFLOW;
            }
        };
        let status = unsafe {
            crate::ffi::terlan_ndarray_v1_squeeze(current, squeeze_axes.as_ptr(), count, &mut next)
        };
        if let Err(status) = unsafe { replace(&mut current, next, status) } {
            let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(current) };
            return status;
        }
    }
    for &axis in new_axes {
        let mut next = std::ptr::null_mut();
        let status = unsafe { crate::ffi::terlan_ndarray_v1_expand_dims(current, axis, &mut next) };
        if let Err(status) = unsafe { replace(&mut current, next, status) } {
            let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(current) };
            return status;
        }
    }
    unsafe { *output = current };
    OK
}

/// Inserts multiple singleton axes at their final output positions.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_expand_dims_axes(
    source: Array,
    axes: *const i64,
    axis_count: i64,
    output: *mut Array,
) -> i32 {
    if output.is_null() || source.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    let axes = match unsafe { values(axes, axis_count) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let source_shape = match unsafe { shape(source) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let output_rank = match source_shape.len().checked_add(axes.len()) {
        Some(value) => value,
        None => return SIZE_OVERFLOW,
    };
    let output_rank_i64 = match i64::try_from(output_rank) {
        Ok(value) => value,
        Err(_) => return SIZE_OVERFLOW,
    };
    let mut normalized = Vec::with_capacity(axes.len());
    for &axis in axes {
        let axis = if axis < 0 {
            match axis.checked_add(output_rank_i64) {
                Some(value) => value,
                None => return AXIS_OUT_OF_RANGE,
            }
        } else {
            axis
        };
        if axis < 0 || axis >= output_rank_i64 {
            return AXIS_OUT_OF_RANGE;
        }
        if normalized.contains(&axis) {
            return DUPLICATE_AXIS;
        }
        normalized.push(axis);
    }
    normalized.sort_unstable();

    if normalized.is_empty() {
        let source_rank = match i64::try_from(source_shape.len()) {
            Ok(value) => value,
            Err(_) => return SIZE_OVERFLOW,
        };
        let mut expanded = std::ptr::null_mut();
        let status = unsafe {
            crate::ffi::terlan_ndarray_v1_expand_dims(source, source_rank, &mut expanded)
        };
        if status != OK {
            return status;
        }
        let mut result = std::ptr::null_mut();
        let status = unsafe {
            crate::ffi::terlan_ndarray_v1_squeeze(expanded, &source_rank, 1, &mut result)
        };
        let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(expanded) };
        return unsafe { produced((status == OK).then_some(result).ok_or(status), output) };
    }

    let mut current: Array = std::ptr::null_mut();
    for axis in normalized {
        let input = if current.is_null() { source } else { current };
        let mut next = std::ptr::null_mut();
        let status = unsafe { crate::ffi::terlan_ndarray_v1_expand_dims(input, axis, &mut next) };
        if !current.is_null() {
            let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(current) };
        }
        if status != OK {
            if !next.is_null() {
                let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(next) };
            }
            return status;
        }
        current = next;
    }
    unsafe { produced(Ok(current), output) }
}

/// Concatenates an arbitrary nonempty input list with canonical dtype promotion.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_concatenate(
    identities: *const u64,
    count: i64,
    axis: i64,
    output: *mut Array,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    let identities = match unsafe { arrays(identities, count) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let first = match array(identities[0]) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let first_shape = match unsafe { shape(first) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if let Err(status) = normalize_axis(axis, first_shape.len()) {
        return status;
    }
    let mut current = std::ptr::null_mut();
    let status = unsafe { crate::ffi::terlan_ndarray_v1_copy(first, &mut current) };
    if status != OK {
        return status;
    }
    for &identity in &identities[1..] {
        let rhs = match array(identity) {
            Ok(value) => value,
            Err(status) => {
                let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(current) };
                return status;
            }
        };
        let mut next = std::ptr::null_mut();
        let status = unsafe { terlan_ndarray_concatenate_pair(current, rhs, axis, &mut next) };
        let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(current) };
        if status != OK {
            if !next.is_null() {
                let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(next) };
            }
            return status;
        }
        current = next;
    }
    unsafe { produced(Ok(current), output) }
}

/// Concatenates logical C-order flattened inputs, matching NumPy `axis=None`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_concatenate_flat(
    identities: *const u64,
    count: i64,
    output: *mut Array,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    let flattened = match unsafe { flattened_arrays(identities, count) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let flattened_identities = array_identities(&flattened);
    let status =
        unsafe { terlan_ndarray_v1_concatenate(flattened_identities.as_ptr(), count, 0, output) };
    unsafe { super::native_support::delete_arrays(&flattened) };
    status
}

/// Concatenates inputs and converts the result under an explicit dtype policy.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_concatenate_with_dtype(
    identities: *const u64,
    count: i64,
    axis: i64,
    dtype: i32,
    casting: i32,
    output: *mut Array,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    if !valid_casting(casting) {
        return INVALID_ARGUMENT;
    }
    let mut concatenated = std::ptr::null_mut();
    let status =
        unsafe { terlan_ndarray_v1_concatenate(identities, count, axis, &mut concatenated) };
    if status != OK {
        return status;
    }
    unsafe { produced(cast_owned(concatenated, dtype, casting), output) }
}

/// Flattens, concatenates, and converts inputs under an explicit dtype policy.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_concatenate_flat_with_dtype(
    identities: *const u64,
    count: i64,
    dtype: i32,
    casting: i32,
    output: *mut Array,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    if !valid_casting(casting) {
        return INVALID_ARGUMENT;
    }
    let mut concatenated = std::ptr::null_mut();
    let status =
        unsafe { terlan_ndarray_v1_concatenate_flat(identities, count, &mut concatenated) };
    if status != OK {
        return status;
    }
    unsafe { produced(cast_owned(concatenated, dtype, casting), output) }
}

/// Concatenates inputs and atomically writes them into an exact-shape destination.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_concatenate_into(
    identities: *const u64,
    count: i64,
    axis: i64,
    destination: Array,
    casting: i32,
) -> i32 {
    if !valid_casting(casting) {
        return INVALID_ARGUMENT;
    }
    let mut concatenated = std::ptr::null_mut();
    let status =
        unsafe { terlan_ndarray_v1_concatenate(identities, count, axis, &mut concatenated) };
    if status != OK {
        return status;
    }
    unsafe { copy_owned_into(destination, concatenated, casting) }
}

/// Flattens and concatenates inputs into an exact-shape destination.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_concatenate_flat_into(
    identities: *const u64,
    count: i64,
    destination: Array,
    casting: i32,
) -> i32 {
    if !valid_casting(casting) {
        return INVALID_ARGUMENT;
    }
    let mut concatenated = std::ptr::null_mut();
    let status =
        unsafe { terlan_ndarray_v1_concatenate_flat(identities, count, &mut concatenated) };
    if status != OK {
        return status;
    }
    unsafe { copy_owned_into(destination, concatenated, casting) }
}

/// Stacks an arbitrary nonempty input list along one new axis.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_stack(
    identities: *const u64,
    count: i64,
    axis: i64,
    output: *mut Array,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    let identities = match unsafe { arrays(identities, count) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut expanded = Vec::with_capacity(identities.len());
    for &identity in identities {
        let source = match array(identity) {
            Ok(value) => value,
            Err(status) => {
                unsafe { super::native_support::delete_arrays(&expanded) };
                return status;
            }
        };
        let mut view = std::ptr::null_mut();
        let status = unsafe { crate::ffi::terlan_ndarray_v1_expand_dims(source, axis, &mut view) };
        if status != OK {
            unsafe { super::native_support::delete_arrays(&expanded) };
            return status;
        }
        expanded.push(view);
    }
    let expanded_identities: Vec<u64> = expanded
        .iter()
        .map(|value| *value as usize as u64)
        .collect();
    let status =
        unsafe { terlan_ndarray_v1_concatenate(expanded_identities.as_ptr(), count, axis, output) };
    unsafe { super::native_support::delete_arrays(&expanded) };
    status
}

#[cfg(test)]
#[path = "ndarray_shape_tests.rs"]
mod tests;
