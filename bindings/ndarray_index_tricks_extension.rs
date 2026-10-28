//! NumPy-compatible multidimensional index conversion and diagonal mutation.

use super::native_support::{
    delete_arrays, dtype, element_count, ints, owned_int, shape, transfer_arrays, write_result,
    Array, AXIS_OUT_OF_RANGE, INVALID_ARGUMENT, OK, SHAPE_MISMATCH, SIZE_OVERFLOW,
    UNSUPPORTED_DTYPE,
};

/// Coordinate handling policy for one ravel axis.
#[derive(Clone, Copy)]
enum IndexMode {
    /// Reject coordinates outside the axis.
    Raise,
    /// Reduce coordinates modulo the axis length.
    Wrap,
    /// Clamp coordinates to the nearest axis endpoint.
    Clip,
}

/// Converts one ABI mode code into its checked policy.
fn index_mode(value: i64) -> Result<IndexMode, i32> {
    match value {
        0 => Ok(IndexMode::Raise),
        1 => Ok(IndexMode::Wrap),
        2 => Ok(IndexMode::Clip),
        _ => Err(INVALID_ARGUMENT),
    }
}

/// Reads a possibly empty borrowed `i64` slice.
unsafe fn borrowed_i64(values: *const i64, count: usize) -> Result<Vec<i64>, i32> {
    if count == 0 {
        Ok(Vec::new())
    } else if values.is_null() {
        Err(INVALID_ARGUMENT)
    } else {
        Ok(unsafe { std::slice::from_raw_parts(values, count) }.to_vec())
    }
}

/// Validates dimensions and returns their representable element count.
fn checked_dimensions(dimensions: &[i64]) -> Result<usize, i32> {
    if dimensions.iter().any(|dimension| *dimension < 0) {
        return Err(SHAPE_MISMATCH);
    }
    let count = element_count(dimensions)?;
    if count > i64::MAX as usize {
        Err(SIZE_OVERFLOW)
    } else {
        Ok(count)
    }
}

/// Computes checked C- or Fortran-order element strides.
fn ordered_strides(dimensions: &[i64], order: i64) -> Result<Vec<usize>, i32> {
    if !matches!(order, 0 | 1) {
        return Err(INVALID_ARGUMENT);
    }
    let mut strides = vec![1usize; dimensions.len()];
    if order == 0 {
        for axis in (0..dimensions.len().saturating_sub(1)).rev() {
            let dimension = usize::try_from(dimensions[axis + 1]).map_err(|_| SHAPE_MISMATCH)?;
            strides[axis] = strides[axis + 1]
                .checked_mul(dimension)
                .ok_or(SIZE_OVERFLOW)?;
        }
    } else {
        for axis in 1..dimensions.len() {
            let dimension = usize::try_from(dimensions[axis - 1]).map_err(|_| SHAPE_MISMATCH)?;
            strides[axis] = strides[axis - 1]
                .checked_mul(dimension)
                .ok_or(SIZE_OVERFLOW)?;
        }
    }
    Ok(strides)
}

/// Normalizes one coordinate under a checked axis mode.
fn normalized_coordinate(coordinate: i64, dimension: i64, mode: IndexMode) -> Result<usize, i32> {
    if dimension <= 0 {
        return Err(AXIS_OUT_OF_RANGE);
    }
    let coordinate = match mode {
        IndexMode::Raise if coordinate < 0 || coordinate >= dimension => {
            return Err(AXIS_OUT_OF_RANGE);
        }
        IndexMode::Raise => coordinate,
        IndexMode::Wrap => coordinate.rem_euclid(dimension),
        IndexMode::Clip => coordinate.clamp(0, dimension - 1),
    };
    usize::try_from(coordinate).map_err(|_| AXIS_OUT_OF_RANGE)
}

/// Returns the result shape and coordinate count for packed coordinate arrays.
fn coordinate_layout(coordinate_shape: &[i64], rank: usize) -> Result<(Vec<i64>, usize), i32> {
    if coordinate_shape.first().copied() != Some(rank as i64) {
        return Err(SHAPE_MISMATCH);
    }
    let result_shape = coordinate_shape[1..].to_vec();
    let count = element_count(&result_shape)?;
    Ok((result_shape, count))
}

/// Converts packed coordinate arrays into flattened indexes with explicit modes and order.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_ravel_multi_index_with(
    coordinates: Array,
    dimensions: *const i64,
    rank: i64,
    modes: *const i64,
    mode_count: i64,
    order: i64,
    output: *mut Array,
) -> i32 {
    let rank = match usize::try_from(rank) {
        Ok(value) if value > 0 => value,
        _ => return INVALID_ARGUMENT,
    };
    let mode_count = match usize::try_from(mode_count) {
        Ok(value) if value == 1 || value == rank => value,
        _ => return INVALID_ARGUMENT,
    };
    let dimensions = match unsafe { borrowed_i64(dimensions, rank) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let modes = match unsafe { borrowed_i64(modes, mode_count) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if let Err(status) = checked_dimensions(&dimensions) {
        return status;
    }
    let strides = match ordered_strides(&dimensions, order) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let coordinate_shape = match unsafe { shape(coordinates) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let (result_shape, count) = match coordinate_layout(&coordinate_shape, rank) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let coordinate_dtype = match unsafe { dtype(coordinates) } {
        Ok(value) if matches!(value, 2 | 4..=10) => value,
        Ok(_) => return UNSUPPORTED_DTYPE,
        Err(status) => return status,
    };
    let _ = coordinate_dtype;
    let coordinates = match unsafe { ints(coordinates) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if coordinates.len() != rank.checked_mul(count).unwrap_or(usize::MAX) {
        return SHAPE_MISMATCH;
    }
    let mut result = Vec::new();
    if result.try_reserve_exact(count).is_err() {
        return super::native_support::ALLOCATION_FAILED;
    }
    for item in 0..count {
        let mut flat = 0usize;
        for axis in 0..rank {
            let mode = match index_mode(modes[if mode_count == 1 { 0 } else { axis }]) {
                Ok(value) => value,
                Err(status) => return status,
            };
            let coordinate = match normalized_coordinate(
                coordinates[axis * count + item],
                dimensions[axis],
                mode,
            ) {
                Ok(value) => value,
                Err(status) => return status,
            };
            flat = match coordinate
                .checked_mul(strides[axis])
                .and_then(|value| flat.checked_add(value))
            {
                Some(value) => value,
                None => return SIZE_OVERFLOW,
            };
        }
        let flat = match i64::try_from(flat) {
            Ok(value) => value,
            Err(_) => return SIZE_OVERFLOW,
        };
        result.push(flat);
    }
    unsafe { write_result(owned_int(&result, &result_shape), output) }
}

/// Converts flattened indexes into one coordinate array per axis in explicit order.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_unravel_index_ordered(
    indexes: Array,
    dimensions: *const i64,
    rank: i64,
    order: i64,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let rank = match usize::try_from(rank) {
        Ok(value) => value,
        Err(_) => return INVALID_ARGUMENT,
    };
    let dimensions = match unsafe { borrowed_i64(dimensions, rank) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let maximum = match checked_dimensions(&dimensions) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let strides = match ordered_strides(&dimensions, order) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let index_shape = match unsafe { shape(indexes) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let index_dtype = match unsafe { dtype(indexes) } {
        Ok(value) if matches!(value, 2 | 4..=10) => value,
        Ok(_) => return UNSUPPORTED_DTYPE,
        Err(status) => return status,
    };
    let _ = index_dtype;
    let indexes = match unsafe { ints(indexes) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    if rank == 0 {
        if index_shape.is_empty() && indexes == [0] && !output.is_null() && !output_length.is_null()
        {
            unsafe {
                *output = std::ptr::null_mut();
                *output_length = 0;
            }
            return OK;
        }
        return AXIS_OUT_OF_RANGE;
    }
    let mut axes = vec![Vec::new(); rank];
    for axis in &mut axes {
        if axis.try_reserve_exact(indexes.len()).is_err() {
            return super::native_support::ALLOCATION_FAILED;
        }
    }
    for index in indexes {
        let index = match usize::try_from(index) {
            Ok(value) if value < maximum => value,
            _ => return AXIS_OUT_OF_RANGE,
        };
        for axis in 0..rank {
            axes[axis].push(((index / strides[axis]) % dimensions[axis] as usize) as i64);
        }
    }
    let mut results = Vec::with_capacity(rank);
    for axis in axes {
        match unsafe { owned_int(&axis, &index_shape) } {
            Ok(value) => results.push(value),
            Err(status) => {
                unsafe { delete_arrays(&results) };
                return status;
            }
        }
    }
    unsafe { transfer_arrays(results, output, output_length) }
}

/// Writes a scalar across a NumPy-compatible diagonal with optional tall-matrix wrapping.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_fill_diagonal_with_wrap(
    array: Array,
    value: f64,
    wrap: bool,
) -> i32 {
    let dimensions = match unsafe { shape(array) } {
        Ok(value) if value.len() >= 2 => value,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    if dimensions.len() > 2
        && dimensions
            .iter()
            .any(|dimension| *dimension != dimensions[0])
    {
        return SHAPE_MISMATCH;
    }
    let mut targets = Vec::new();
    if dimensions.len() == 2 && wrap {
        let (rows, columns) = (dimensions[0], dimensions[1]);
        if rows == 0 || columns == 0 {
            return OK;
        }
        let step = match usize::try_from(columns)
            .ok()
            .and_then(|value| value.checked_add(1))
        {
            Some(value) => value,
            None => return SIZE_OVERFLOW,
        };
        let count = match checked_dimensions(&dimensions) {
            Ok(value) => value,
            Err(status) => return status,
        };
        for flat in (0..count).step_by(step) {
            targets.push(flat as i64);
        }
    } else {
        let length = dimensions.iter().copied().min().unwrap_or(0);
        let strides = match ordered_strides(&dimensions, 0) {
            Ok(value) => value,
            Err(status) => return status,
        };
        for index in 0..length {
            let index = index as usize;
            let flat = match strides.iter().try_fold(0usize, |flat, stride| {
                index
                    .checked_mul(*stride)
                    .and_then(|offset| flat.checked_add(offset))
            }) {
                Some(value) => value,
                None => return SIZE_OVERFLOW,
            };
            let flat = match i64::try_from(flat) {
                Ok(value) => value,
                Err(_) => return SIZE_OVERFLOW,
            };
            targets.push(flat);
        }
    }
    let scalar_dtype = match unsafe { dtype(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut scalar = std::ptr::null_mut();
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_full_float(
            std::ptr::null(),
            0,
            value,
            scalar_dtype,
            &mut scalar,
        )
    };
    if status != OK {
        return status;
    }
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_put(array, targets.as_ptr(), targets.len() as i64, scalar)
    };
    unsafe { delete_arrays(&[scalar]) };
    status
}
