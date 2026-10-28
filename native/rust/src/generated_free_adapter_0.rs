/// Concatenates a nonempty list of shape-compatible arrays with canonical dtype promotion.
pub fn concatenate(arrays: &[&Array], axis: i64) -> Result<Array, CAbiError> {
    let arrays_handles = arrays.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
    let arrays_length = i64::try_from(arrays.len()).map_err(|_| CAbiError { operation: "ndarray.array.concatenate", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_concatenate(arrays_handles.as_ptr(), arrays_length, axis, &mut raw) };
    check_status("ndarray.array.concatenate", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.concatenate", status: -1 })?;
    Ok(Array { raw })
}
/// Stacks a nonempty list of equal-shape arrays with canonical dtype promotion.
pub fn stack(arrays: &[&Array], axis: i64) -> Result<Array, CAbiError> {
    let arrays_handles = arrays.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
    let arrays_length = i64::try_from(arrays.len()).map_err(|_| CAbiError { operation: "ndarray.array.stack", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_stack(arrays_handles.as_ptr(), arrays_length, axis, &mut raw) };
    check_status("ndarray.array.stack", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.stack", status: -1 })?;
    Ok(Array { raw })
}
/// Returns live native allocations for lifecycle auditing.
pub fn live_count() -> i64 {
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let value = unsafe { ffi::terlan_ndarray_v1_live_count() };
    value
}

/// Broadcasts numerical endpoint arrays and inserts evenly spaced samples at the requested axis.
pub fn linspace_arrays(start: &Array, stop: &Array, count: i64, endpoint: i64, axis: i64, dtype: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_linspace_arrays(start.raw.as_ptr(), stop.raw.as_ptr(), count, endpoint as i32, axis, dtype as i32, &mut raw) };
    check_status("ndarray.creation.linspace_arrays", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.creation.linspace_arrays", status: -1 })?;
    Ok(Array { raw })
}

/// Returns the broadcast-shaped spacing for array endpoints, including NaN for an undefined spacing.
pub fn linspace_steps(start: &Array, stop: &Array, count: i64, endpoint: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_linspace_steps(start.raw.as_ptr(), stop.raw.as_ptr(), count, endpoint as i32, &mut raw) };
    check_status("ndarray.creation.linspace_steps", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.creation.linspace_steps", status: -1 })?;
    Ok(Array { raw })
}

/// Creates a rectangular diagonal-unit matrix with explicit C or Fortran storage order.
pub fn eye_ordered(rows: i64, columns: i64, diagonal: i64, dtype: i64, order: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_eye(rows, columns, diagonal, dtype as i32, order as i32, &mut raw) };
    check_status("ndarray.creation.eye_ordered", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.creation.eye_ordered", status: -1 })?;
    Ok(Array { raw })
}

/// Allocates an uninitialized array with the source shape and dtype.
pub fn empty_like(array: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_empty_like(array.raw.as_ptr(), &mut raw) };
    check_status("ndarray.creation.empty_like", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.creation.empty_like", status: -1 })?;
    Ok(Array { raw })
}

/// Allocates a zero-filled array with the source shape and dtype.
pub fn zeros_like(array: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_zeros_like(array.raw.as_ptr(), &mut raw) };
    check_status("ndarray.creation.zeros_like", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.creation.zeros_like", status: -1 })?;
    Ok(Array { raw })
}

/// Allocates a one-filled array with the source shape and dtype.
pub fn ones_like(array: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_ones_like(array.raw.as_ptr(), &mut raw) };
    check_status("ndarray.creation.ones_like", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.creation.ones_like", status: -1 })?;
    Ok(Array { raw })
}

/// Fills a new source-shaped and source-typed array with an integer scalar.
pub fn full_like_int(array: &Array, value: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_full_like_int(array.raw.as_ptr(), value, &mut raw) };
    check_status("ndarray.creation.full_like_int", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.creation.full_like_int", status: -1 })?;
    Ok(Array { raw })
}

/// Fills a new source-shaped and source-typed array with a floating scalar.
pub fn full_like_float(array: &Array, value: f64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_full_like_float(array.raw.as_ptr(), value, &mut raw) };
    check_status("ndarray.creation.full_like_float", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.creation.full_like_float", status: -1 })?;
    Ok(Array { raw })
}

/// Returns powers of a positive base with evenly spaced exponents.
pub fn logspace(start: f64, stop: f64, count: i64, endpoint: i64, base: f64, dtype: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_logspace(start, stop, count, endpoint as i32, base, dtype as i32, &mut raw) };
    check_status("ndarray.creation.logspace", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.creation.logspace", status: -1 })?;
    Ok(Array { raw })
}

/// Returns evenly spaced values on a same-sign geometric progression.
pub fn geomspace(start: f64, stop: f64, count: i64, endpoint: i64, dtype: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_geomspace(start, stop, count, endpoint as i32, dtype as i32, &mut raw) };
    check_status("ndarray.creation.geomspace", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.creation.geomspace", status: -1 })?;
    Ok(Array { raw })
}

/// Broadcasts exponent arrays and inserts powers of a positive base at the requested axis.
pub fn logspace_arrays(start: &Array, stop: &Array, count: i64, endpoint: i64, axis: i64, base: f64, dtype: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_logspace_arrays(start.raw.as_ptr(), stop.raw.as_ptr(), count, endpoint as i32, axis, base, dtype as i32, &mut raw) };
    check_status("ndarray.creation.logspace_arrays", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.creation.logspace_arrays", status: -1 })?;
    Ok(Array { raw })
}

/// Broadcasts nonzero endpoints and inserts geometrically spaced samples at the requested axis.
pub fn geomspace_arrays(start: &Array, stop: &Array, count: i64, endpoint: i64, axis: i64, dtype: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_geomspace_arrays(start.raw.as_ptr(), stop.raw.as_ptr(), count, endpoint as i32, axis, dtype as i32, &mut raw) };
    check_status("ndarray.creation.geomspace_arrays", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.creation.geomspace_arrays", status: -1 })?;
    Ok(Array { raw })
}

/// Creates a matrix with ones at and below an indexed diagonal.
pub fn tri(rows: i64, columns: i64, diagonal: i64, dtype: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_tri(rows, columns, diagonal, dtype as i32, &mut raw) };
    check_status("ndarray.creation.tri", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.creation.tri", status: -1 })?;
    Ok(Array { raw })
}

/// Copies an array while zeroing entries above an indexed diagonal in every matrix plane.
pub fn tril(array: &Array, diagonal: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_tril(array.raw.as_ptr(), diagonal, &mut raw) };
    check_status("ndarray.creation.tril", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.creation.tril", status: -1 })?;
    Ok(Array { raw })
}

/// Copies an array while zeroing entries below an indexed diagonal in every matrix plane.
pub fn triu(array: &Array, diagonal: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_triu(array.raw.as_ptr(), diagonal, &mut raw) };
    check_status("ndarray.creation.triu", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.creation.triu", status: -1 })?;
    Ok(Array { raw })
}

/// Flattens an array and places its values on an indexed diagonal of a new square matrix.
pub fn diagflat(array: &Array, diagonal: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_diagflat(array.raw.as_ptr(), diagonal, &mut raw) };
    check_status("ndarray.creation.diagflat", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.creation.diagflat", status: -1 })?;
    Ok(Array { raw })
}

/// Returns one dense numeric coordinate grid for every dimension of a shape.
pub fn indices(shape: &[i64], dtype: i64) -> Result<Array, CAbiError> {
    let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.creation.indices", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_indices(shape.as_ptr(), shape_length, dtype as i32, &mut raw) };
    check_status("ndarray.creation.indices", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.creation.indices", status: -1 })?;
    Ok(Array { raw })
}

/// Returns one sparse numeric coordinate grid per dimension, with singleton dimensions elsewhere.
pub fn indices_sparse(shape: &[i64], dtype: i64) -> Result<Vec<Array>, CAbiError> {
    let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.creation.indices_sparse", status: -2 })?;
    let mut out_out_values: *mut u64 = std::ptr::null_mut();
    let mut out_out_values_length: usize = 0;
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_indices_sparse(shape.as_ptr(), shape_length, dtype as i32, &mut out_out_values, &mut out_out_values_length) };
    check_status("ndarray.creation.indices_sparse", status, 0)?;
    let raw_values = if out_out_values_length == 0 {
        Ok(Vec::new())
    } else {
        NonNull::new(out_out_values).map(|pointer| {
            // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
            unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
        }).ok_or(CAbiError { operation: "ndarray.creation.indices_sparse", status: -1 })
    };
    // SAFETY: only the temporary identity array is released; its resource elements were transferred.
    unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
    let raw_values = raw_values?;
    if raw_values.iter().any(|value| *value == 0) {
        for value in raw_values.into_iter().filter(|value| *value != 0) {
            // SAFETY: nonzero identities still belong to this failed transfer.
            let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
        }
        return Err(CAbiError { operation: "ndarray.creation.indices_sparse", status: -1 });
    }
    Ok(raw_values.into_iter().map(|value| Array {
        // SAFETY: zero identities were rejected above.
        raw: unsafe { NonNull::new_unchecked(value as _) },
    }).collect())
}

/// Builds a Vandermonde matrix with explicit column count and exponent order.
pub fn vander(array: &Array, columns: i64, increasing: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_vander(array.raw.as_ptr(), columns, increasing as i32, &mut raw) };
    check_status("ndarray.creation.vander", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.creation.vander", status: -1 })?;
    Ok(Array { raw })
}

/// Returns a view with at least one dimension.
pub fn atleast_1d(array: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_atleast_1d(array.raw.as_ptr(), &mut raw) };
    check_status("ndarray.shape.atleast_1d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.atleast_1d", status: -1 })?;
    Ok(Array { raw })
}

/// Returns a view with at least two dimensions, promoting vectors to rows.
pub fn atleast_2d(array: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_atleast_2d(array.raw.as_ptr(), &mut raw) };
    check_status("ndarray.shape.atleast_2d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.atleast_2d", status: -1 })?;
    Ok(Array { raw })
}
