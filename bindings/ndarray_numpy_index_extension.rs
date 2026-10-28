//! NumPy-compatible compound indexing, mesh, and histogram operations.

use super::native_support::{
    delete_arrays, dtype, normalize_axis, numeric, owned_int, promoted_dtype, shape,
    transfer_arrays, write_result, Array, ALLOCATION_FAILED, AXIS_OUT_OF_RANGE, DTYPE_MISMATCH,
    INVALID_ARGUMENT, OK, SHAPE_MISMATCH, SIZE_OVERFLOW,
};
use super::numpy_shape_extension::handles;

/// Inserts values before one flattened rank-one scalar index.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_insert(
    array: Array,
    index: i64,
    values: Array,
    output: *mut Array,
) -> i32 {
    unsafe { terlan_ndarray_v1_insert_scalar_axis(array, index, values, 0, output) }
}

/// Inserts values before flattened rank-one integer indexes.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_insert_indexes(
    array: Array,
    indexes: *const i64,
    index_count: i64,
    values: Array,
    output: *mut Array,
) -> i32 {
    unsafe { terlan_ndarray_v1_insert_indexes_axis(array, indexes, index_count, values, 0, output) }
}

/// Inserts values before one scalar index along a normalized axis.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_insert_scalar_axis(
    array: Array,
    index: i64,
    values: Array,
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
    let index = match normalize_insert_index(index, dimensions[axis]) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let prepared = match unsafe { prepare_insert_values(array, values, axis, true, 0) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let status = unsafe { concatenate_insertion(array, prepared, axis, index, output) };
    unsafe { delete_arrays(&[prepared]) };
    status
}

/// Inserts values before stable integer-index positions along a normalized axis.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_insert_indexes_axis(
    array: Array,
    indexes: *const i64,
    index_count: i64,
    values: Array,
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
    if count == 0 {
        return unsafe { crate::ffi::terlan_ndarray_v1_copy(array, output) };
    }
    let indexes = unsafe { std::slice::from_raw_parts(indexes, count) };
    if count == 1 {
        let index = match normalize_insert_index(indexes[0], dimensions[axis]) {
            Ok(value) => value,
            Err(status) => return status,
        };
        let prepared = match unsafe { prepare_insert_values(array, values, axis, false, 0) } {
            Ok(value) => value,
            Err(status) => return status,
        };
        let status = unsafe { concatenate_insertion(array, prepared, axis, index, output) };
        unsafe { delete_arrays(&[prepared]) };
        return status;
    }

    let mut events = Vec::with_capacity(count);
    for (ordinal, &index) in indexes.iter().enumerate() {
        let index = match normalize_insert_index(index, dimensions[axis]) {
            Ok(value) => value,
            Err(status) => return status,
        };
        events.push((index, ordinal));
    }
    events.sort_by_key(|event| event.0);
    for (offset, event) in events.iter_mut().enumerate() {
        event.0 += offset as i64;
    }
    let prepared = match unsafe { prepare_insert_values(array, values, axis, false, count as i64) }
    {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut pieces = Vec::with_capacity(dimensions[axis] as usize + count);
    let output_length = dimensions[axis] as usize + count;
    let mut source_index = 0_i64;
    for output_index in 0..output_length {
        let insertion = events
            .iter()
            .find(|event| event.0 == output_index as i64)
            .map(|event| event.1 as i64);
        let (source, selected) = match insertion {
            Some(ordinal) => (prepared, ordinal),
            None => {
                let selected = source_index;
                source_index += 1;
                (array, selected)
            }
        };
        let mut piece = std::ptr::null_mut();
        let status = unsafe {
            crate::ffi::terlan_ndarray_v1_slice_axis(
                source,
                axis as i64,
                selected,
                selected + 1,
                1,
                &mut piece,
            )
        };
        if status != OK {
            unsafe { delete_arrays(&pieces) };
            unsafe { delete_arrays(&[prepared]) };
            return status;
        }
        pieces.push(piece);
    }
    let identities = pieces
        .iter()
        .map(|piece| *piece as usize as u64)
        .collect::<Vec<_>>();
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_concatenate(
            identities.as_ptr(),
            identities.len() as i64,
            axis as i64,
            output,
        )
    };
    unsafe { delete_arrays(&pieces) };
    unsafe { delete_arrays(&[prepared]) };
    status
}

/// Normalizes one insertion index while accepting the position after the axis.
fn normalize_insert_index(index: i64, length: i64) -> Result<i64, i32> {
    let normalized = if index < 0 { length + index } else { index };
    if normalized < 0 || normalized > length {
        Err(AXIS_OUT_OF_RANGE)
    } else {
        Ok(normalized)
    }
}

/// Casts, rank-promotes, reorients, and broadcasts insertion values.
unsafe fn prepare_insert_values(
    array: Array,
    values: Array,
    axis: usize,
    scalar_index: bool,
    explicit_count: i64,
) -> Result<Array, i32> {
    let dimensions = unsafe { shape(array)? };
    let value_dimensions = unsafe { shape(values)? };
    let array_dtype = unsafe { dtype(array)? };
    let mut current = std::ptr::null_mut();
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_astype_ordered(values, array_dtype, 2, 1, 4, &mut current)
    };
    if status != OK {
        return Err(status);
    }
    let mut promoted_shape = vec![1_i64; dimensions.len().saturating_sub(value_dimensions.len())];
    promoted_shape.extend_from_slice(&value_dimensions);
    if promoted_shape.len() != dimensions.len() {
        unsafe { delete_arrays(&[current]) };
        return Err(SHAPE_MISMATCH);
    }
    let mut promoted = std::ptr::null_mut();
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_reshape(
            current,
            promoted_shape.as_ptr(),
            promoted_shape.len() as i64,
            &mut promoted,
        )
    };
    unsafe { delete_arrays(&[current]) };
    if status != OK {
        return Err(status);
    }
    current = promoted;
    if scalar_index && axis != 0 {
        let mut moved = std::ptr::null_mut();
        let status = unsafe {
            crate::ffi::terlan_ndarray_v1_moveaxis(
                current,
                [0_i64].as_ptr(),
                1,
                [axis as i64].as_ptr(),
                1,
                &mut moved,
            )
        };
        unsafe { delete_arrays(&[current]) };
        if status != OK {
            return Err(status);
        }
        current = moved;
    }
    let current_shape = unsafe { shape(current)? };
    let mut target = dimensions;
    target[axis] = if explicit_count > 0 {
        explicit_count
    } else {
        current_shape[axis]
    };
    let mut broadcast = std::ptr::null_mut();
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_broadcast_to(
            current,
            target.as_ptr(),
            target.len() as i64,
            &mut broadcast,
        )
    };
    unsafe { delete_arrays(&[current]) };
    if status == OK {
        Ok(broadcast)
    } else {
        Err(status)
    }
}

/// Concatenates source prefixes and suffixes around prepared insertion values.
unsafe fn concatenate_insertion(
    array: Array,
    values: Array,
    axis: usize,
    index: i64,
    output: *mut Array,
) -> i32 {
    let mut before = std::ptr::null_mut();
    let mut after = std::ptr::null_mut();
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_slice_axis(array, axis as i64, 0, index, 1, &mut before)
    };
    if status != OK {
        return status;
    }
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_slice_axis(array, axis as i64, index, i64::MAX, 1, &mut after)
    };
    if status != OK {
        unsafe { delete_arrays(&[before]) };
        return status;
    }
    let identities = [
        before as usize as u64,
        values as usize as u64,
        after as usize as u64,
    ];
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_concatenate(
            identities.as_ptr(),
            identities.len() as i64,
            axis as i64,
            output,
        )
    };
    unsafe { delete_arrays(&[after, before]) };
    status
}

/// Returns open-mesh index arrays for one first coordinate and its peers.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_ix(
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
    let rank = arrays.len();
    let mut results = Vec::with_capacity(rank);
    for (axis, array) in arrays.into_iter().enumerate() {
        match unsafe { shape(array) } {
            Ok(value) if value.len() == 1 => {}
            Ok(_) => {
                unsafe { delete_arrays(&results) };
                return SHAPE_MISMATCH;
            }
            Err(status) => {
                unsafe { delete_arrays(&results) };
                return status;
            }
        }
        let mut target = vec![1i64; rank];
        let reshaped_source = match unsafe { super::native_support::dtype(array) } {
            Ok(1) => {
                let values = match unsafe { super::native_support::ints(array) } {
                    Ok(value) => value,
                    Err(status) => {
                        unsafe { delete_arrays(&results) };
                        return status;
                    }
                };
                let selected = values
                    .iter()
                    .enumerate()
                    .filter_map(|(index, value)| (*value != 0).then_some(index as i64))
                    .collect::<Vec<_>>();
                match unsafe { owned_int(&selected, &[selected.len() as i64]) } {
                    Ok(value) => value,
                    Err(status) => {
                        unsafe { delete_arrays(&results) };
                        return status;
                    }
                }
            }
            Ok(_) => array,
            Err(status) => {
                unsafe { delete_arrays(&results) };
                return status;
            }
        };
        target[axis] = match unsafe { shape(reshaped_source) } {
            Ok(value) => value[0],
            Err(status) => {
                if reshaped_source != array {
                    unsafe { delete_arrays(&[reshaped_source]) };
                }
                unsafe { delete_arrays(&results) };
                return status;
            }
        };
        let mut result = std::ptr::null_mut();
        let status = unsafe {
            crate::ffi::terlan_ndarray_v1_reshape(
                reshaped_source,
                target.as_ptr(),
                target.len() as i64,
                &mut result,
            )
        };
        if reshaped_source != array {
            unsafe { delete_arrays(&[reshaped_source]) };
        }
        if status != OK {
            unsafe { delete_arrays(&results) };
            return status;
        }
        results.push(result);
    }
    unsafe { transfer_arrays(results, output, output_length) }
}

/// Returns NumPy-compatible coordinate grids with explicit indexing, sparsity, and ownership.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_meshgrid(
    first: Array,
    remaining: *const u64,
    remaining_count: i64,
    indexing: i64,
    sparse: bool,
    copy: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    if output.is_null() || output_length.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe {
        *output = std::ptr::null_mut();
        *output_length = 0;
    }
    if !matches!(indexing, 0 | 1) {
        return INVALID_ARGUMENT;
    }
    let mut arrays = vec![first];
    match unsafe { handles(remaining, remaining_count) } {
        Ok(mut value) => arrays.append(&mut value),
        Err(status) => return status,
    };
    let dimensions = match arrays
        .iter()
        .map(|array| unsafe { shape(*array) })
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(value) if value.iter().all(|shape| shape.len() == 1) => value,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let mut target = dimensions.iter().map(|shape| shape[0]).collect::<Vec<_>>();
    if indexing == 0 && target.len() > 1 {
        target.swap(0, 1);
    }
    let mut results = Vec::with_capacity(arrays.len());
    for (axis, array) in arrays.into_iter().enumerate() {
        let output_axis = if indexing == 0 && target.len() > 1 {
            match axis {
                0 => 1,
                1 => 0,
                _ => axis,
            }
        } else {
            axis
        };
        let mut reshaped_shape = vec![1i64; target.len()];
        reshaped_shape[output_axis] = dimensions[axis][0];
        let mut reshaped = std::ptr::null_mut();
        let status = unsafe {
            crate::ffi::terlan_ndarray_v1_reshape(
                array,
                reshaped_shape.as_ptr(),
                reshaped_shape.len() as i64,
                &mut reshaped,
            )
        };
        if status != OK {
            unsafe { delete_arrays(&results) };
            return status;
        }
        let mut expanded = reshaped;
        let status = if sparse {
            OK
        } else {
            expanded = std::ptr::null_mut();
            unsafe {
                crate::ffi::terlan_ndarray_v1_broadcast_to(
                    reshaped,
                    target.as_ptr(),
                    target.len() as i64,
                    &mut expanded,
                )
            }
        };
        if !sparse {
            unsafe { delete_arrays(&[reshaped]) };
        }
        if status != OK {
            unsafe { delete_arrays(&results) };
            return status;
        }
        if copy {
            let mut owned = std::ptr::null_mut();
            let status = unsafe { crate::ffi::terlan_ndarray_v1_copy(expanded, &mut owned) };
            unsafe { delete_arrays(&[expanded]) };
            if status != OK {
                unsafe { delete_arrays(&results) };
                return status;
            }
            results.push(owned);
        } else {
            results.push(expanded);
        }
    }
    unsafe { transfer_arrays(results, output, output_length) }
}

/// Sorts values along axis zero.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_msort(array: Array, output: *mut Array) -> i32 {
    unsafe { crate::ffi::terlan_ndarray_v1_sort(array, 0, output) }
}

/// Applies uniform constant padding per axis.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_pad(
    array: Array,
    leading: *const i64,
    leading_count: i64,
    trailing: *const i64,
    trailing_count: i64,
    value: f64,
    output: *mut Array,
) -> i32 {
    let rank = match unsafe { shape(array) } {
        Ok(value) => value.len(),
        Err(status) => return status,
    };
    let values = vec![value; rank];
    unsafe {
        crate::ffi::terlan_ndarray_v1_pad_constant(
            array,
            leading,
            leading_count,
            trailing,
            trailing_count,
            values.as_ptr(),
            rank as i64,
            values.as_ptr(),
            rank as i64,
            output,
        )
    }
}

/// Converts coordinate tuples into flattened row-major indexes.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_ravel_multi_index(
    coordinates: Array,
    dimensions: *const i64,
    rank: i64,
    output: *mut Array,
) -> i32 {
    let raise = [0i64];
    unsafe {
        super::index_tricks_extension::terlan_ndarray_v1_ravel_multi_index_with(
            coordinates,
            dimensions,
            rank,
            raise.as_ptr(),
            1,
            0,
            output,
        )
    }
}

/// Selects among choices using ordered condition masks.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_select(
    conditions: *const u64,
    condition_count: i64,
    choices: *const u64,
    choice_count: i64,
    default_value: Array,
    output: *mut Array,
) -> i32 {
    let conditions = match unsafe { handles(conditions, condition_count) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let choices = match unsafe { handles(choices, choice_count) } {
        Ok(value) if value.len() == conditions.len() && !value.is_empty() => value,
        Ok(_) => return INVALID_ARGUMENT,
        Err(status) => return status,
    };
    for condition in &conditions {
        match unsafe { dtype(*condition) } {
            Ok(1) => {}
            Ok(_) => return DTYPE_MISMATCH,
            Err(status) => return status,
        }
    }
    let mut choice_dtype = match unsafe { dtype(choices[0]) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    for choice in choices.iter().skip(1) {
        let current = match unsafe { dtype(*choice) } {
            Ok(value) => value,
            Err(status) => return status,
        };
        choice_dtype = match unsafe { promoted_dtype(choice_dtype, current) } {
            Ok(value) => value,
            Err(status) => return status,
        };
    }
    let (starting_default, temporary_default) =
        match unsafe { weak_scalar_default(default_value, choice_dtype) } {
            Ok(value) => value,
            Err(status) => return status,
        };
    let mut current = std::ptr::null_mut();
    let status = unsafe { crate::ffi::terlan_ndarray_v1_copy(starting_default, &mut current) };
    if temporary_default {
        unsafe { delete_arrays(&[starting_default]) };
    }
    if status != OK {
        return status;
    }
    for index in (0..conditions.len()).rev() {
        let mut next = std::ptr::null_mut();
        let status = unsafe {
            crate::ffi::terlan_ndarray_v1_where(
                conditions[index],
                choices[index],
                current,
                &mut next,
            )
        };
        unsafe { delete_arrays(&[current]) };
        if status != OK {
            return status;
        }
        current = next;
    }
    unsafe { write_result(Ok(current), output) }
}

/// Returns whether one scalar value can use a choice dtype without widening it.
fn scalar_fits_dtype(value: rustfft::num_complex::Complex64, target: i32) -> bool {
    if !value.re.is_finite() || !value.im.is_finite() {
        return matches!(target, 3 | 11 | 12 | 13 | 14);
    }
    match target {
        1 => value.im == 0.0 && matches!(value.re, 0.0 | 1.0),
        2 => value.im == 0.0 && value.re.fract() == 0.0,
        4 => integer_in_range(value, i8::MIN as f64, i8::MAX as f64),
        5 => integer_in_range(value, i16::MIN as f64, i16::MAX as f64),
        6 => integer_in_range(value, i32::MIN as f64, i32::MAX as f64),
        7 => integer_in_range(value, u8::MIN as f64, u8::MAX as f64),
        8 => integer_in_range(value, u16::MIN as f64, u16::MAX as f64),
        9 => integer_in_range(value, u32::MIN as f64, u32::MAX as f64),
        10 => value.im == 0.0 && value.re.fract() == 0.0 && value.re >= 0.0,
        3 | 11 | 14 => value.im == 0.0,
        12 | 13 => true,
        _ => false,
    }
}

/// Returns whether one complex computation value is an in-range integer.
fn integer_in_range(value: rustfft::num_complex::Complex64, minimum: f64, maximum: f64) -> bool {
    value.im == 0.0 && value.re.fract() == 0.0 && value.re >= minimum && value.re <= maximum
}

/// Narrows one fitting rank-zero default to the already-promoted choice dtype.
unsafe fn weak_scalar_default(
    default_value: Array,
    choice_dtype: i32,
) -> Result<(Array, bool), i32> {
    let dimensions = unsafe { shape(default_value)? };
    let default_dtype = unsafe { dtype(default_value)? };
    if !dimensions.is_empty() || default_dtype == choice_dtype {
        return Ok((default_value, false));
    }
    let values = unsafe { numeric(default_value, false)? };
    if values.values.len() != 1 || !scalar_fits_dtype(values.values[0], choice_dtype) {
        return Ok((default_value, false));
    }
    let mut converted = std::ptr::null_mut();
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_astype_ordered(
            default_value,
            choice_dtype,
            2,
            1,
            4,
            &mut converted,
        )
    };
    if status == OK {
        Ok((converted, true))
    } else {
        Err(status)
    }
}

/// Builds lower-triangle index pairs for a rectangular matrix.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_tril_indices(
    rows: i64,
    offset: i64,
    columns: i64,
    output: *mut Array,
) -> i32 {
    unsafe { triangle_indices(rows, columns, offset, false, output) }
}

/// Builds upper-triangle index pairs for a rectangular matrix.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_triu_indices(
    rows: i64,
    offset: i64,
    columns: i64,
    output: *mut Array,
) -> i32 {
    unsafe { triangle_indices(rows, columns, offset, true, output) }
}

/// Builds triangle indexes matching one rank-two array.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_tril_indices_from(
    array: Array,
    offset: i64,
    output: *mut Array,
) -> i32 {
    if !output.is_null() {
        unsafe { *output = std::ptr::null_mut() };
    }
    let dimensions = match unsafe { shape(array) } {
        Ok(value) if value.len() == 2 => value,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    unsafe { triangle_indices(dimensions[0], dimensions[1], offset, false, output) }
}

/// Builds upper-triangle indexes matching one rank-two array.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_triu_indices_from(
    array: Array,
    offset: i64,
    output: *mut Array,
) -> i32 {
    if !output.is_null() {
        unsafe { *output = std::ptr::null_mut() };
    }
    let dimensions = match unsafe { shape(array) } {
        Ok(value) if value.len() == 2 => value,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    unsafe { triangle_indices(dimensions[0], dimensions[1], offset, true, output) }
}

/// Materializes one triangle's coordinate pairs.
unsafe fn triangle_indices(
    rows: i64,
    columns: i64,
    offset: i64,
    upper: bool,
    output: *mut Array,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    let columns = if columns == -1 { rows } else { columns };
    let (rows, columns) = match (usize::try_from(rows), usize::try_from(columns)) {
        (Ok(rows), Ok(columns)) => (rows, columns),
        _ => return INVALID_ARGUMENT,
    };
    let capacity = match rows.checked_mul(columns) {
        Some(value) => value,
        None => return SIZE_OVERFLOW,
    };
    let mut row_values = Vec::new();
    let mut column_values = Vec::new();
    if row_values.try_reserve_exact(capacity).is_err()
        || column_values.try_reserve_exact(capacity).is_err()
    {
        return ALLOCATION_FAILED;
    }
    for row in 0..rows {
        for column in 0..columns {
            let selected = if upper {
                column as i64 - row as i64 >= offset
            } else {
                column as i64 - row as i64 <= offset
            };
            if selected {
                row_values.push(row as i64);
                column_values.push(column as i64);
            }
        }
    }
    let count = row_values.len();
    row_values.extend(column_values);
    unsafe { write_result(owned_int(&row_values, &[2, count as i64]), output) }
}

/// Converts flattened row-major indexes into one coordinate array per axis.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_unravel_index(
    indexes: Array,
    dimensions: *const i64,
    rank: i64,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    unsafe {
        super::index_tricks_extension::terlan_ndarray_v1_unravel_index_ordered(
            indexes,
            dimensions,
            rank,
            0,
            output,
            output_length,
        )
    }
}

#[cfg(test)]
#[path = "ndarray_numpy_shape_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "ndarray_triangle_index_tests.rs"]
mod triangle_tests;

#[cfg(test)]
#[path = "ndarray_index_error_tests.rs"]
mod index_error_tests;

#[cfg(test)]
#[path = "ndarray_broadcast_assignment_tests.rs"]
mod broadcast_assignment_tests;

#[cfg(test)]
#[path = "ndarray_boolean_index_tests.rs"]
mod boolean_index_tests;

#[cfg(test)]
#[path = "ndarray_integer_index_tests.rs"]
mod integer_index_tests;

#[cfg(test)]
#[path = "ndarray_strided_index_tests.rs"]
mod strided_index_tests;

#[cfg(test)]
#[path = "ndarray_numpy_small_file_tests.rs"]
mod small_file_tests;
