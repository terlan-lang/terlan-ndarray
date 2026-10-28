/// Returns a view with at least three dimensions using NumPy rank-promotion rules.
pub fn atleast_3d(array: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_atleast_3d(array.raw.as_ptr(), &mut raw) };
    check_status("ndarray.shape.atleast_3d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.atleast_3d", status: -1 })?;
    Ok(Array { raw })
}
/// Returns a contiguous rank-one view or materialization in logical row-major order.
pub fn ravel(array: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_ravel(array.raw.as_ptr(), &mut raw) };
    check_status("ndarray.shape.ravel", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.ravel", status: -1 })?;
    Ok(Array { raw })
}
/// Ravels using explicit C, Fortran, input-compatible, or physical traversal order.
pub fn ravel_ordered(array: &Array, order: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_ravel_ordered(array.raw.as_ptr(), order as i32, &mut raw) };
    check_status("ndarray.shape.ravel_ordered", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.ravel_ordered", status: -1 })?;
    Ok(Array { raw })
}

/// Returns an independent rank-one copy in logical row-major order.
pub fn flatten(array: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_flatten(array.raw.as_ptr(), &mut raw) };
    check_status("ndarray.shape.flatten", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.flatten", status: -1 })?;
    Ok(Array { raw })
}

/// Flattens into an independent copy using explicit C, Fortran, input-compatible, or physical traversal order.
pub fn flatten_ordered(array: &Array, order: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_flatten_ordered(array.raw.as_ptr(), order as i32, &mut raw) };
    check_status("ndarray.shape.flatten_ordered", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.flatten_ordered", status: -1 })?;
    Ok(Array { raw })
}

/// Moves distinct source axes to distinct destinations while preserving all other axis order.
pub fn moveaxis(array: &Array, sources: &[i64], destinations: &[i64]) -> Result<Array, CAbiError> {
    let sources_length = i64::try_from(sources.len()).map_err(|_| CAbiError { operation: "ndarray.shape.moveaxis", status: -2 })?;
    let destinations_length = i64::try_from(destinations.len()).map_err(|_| CAbiError { operation: "ndarray.shape.moveaxis", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_moveaxis(array.raw.as_ptr(), sources.as_ptr(), sources_length, destinations.as_ptr(), destinations_length, &mut raw) };
    check_status("ndarray.shape.moveaxis", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.moveaxis", status: -1 })?;
    Ok(Array { raw })
}

/// Rolls one axis backward to an insertion position.
pub fn rollaxis(array: &Array, axis: i64, start: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_rollaxis(array.raw.as_ptr(), axis, start, &mut raw) };
    check_status("ndarray.shape.rollaxis", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.rollaxis", status: -1 })?;
    Ok(Array { raw })
}

/// Returns a shared negative-stride view with every axis reversed.
pub fn flip_all(array: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_flip_all(array.raw.as_ptr(), &mut raw) };
    check_status("ndarray.shape.flip_all", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.flip_all", status: -1 })?;
    Ok(Array { raw })
}

/// Returns a shared view reversed along axis zero.
pub fn flipud(array: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_flipud(array.raw.as_ptr(), &mut raw) };
    check_status("ndarray.shape.flipud", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.flipud", status: -1 })?;
    Ok(Array { raw })
}

/// Returns a shared view reversed along axis one.
pub fn fliplr(array: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_fliplr(array.raw.as_ptr(), &mut raw) };
    check_status("ndarray.shape.fliplr", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.fliplr", status: -1 })?;
    Ok(Array { raw })
}

/// Rotates matrix planes counterclockwise by a multiple of 90 degrees.
pub fn rot90(array: &Array, rotations: i64, axis_a: i64, axis_b: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_rot90(array.raw.as_ptr(), rotations, axis_a, axis_b, &mut raw) };
    check_status("ndarray.shape.rot90", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.rot90", status: -1 })?;
    Ok(Array { raw })
}

/// Flattens every input in logical C order and concatenates the resulting vectors.
pub fn concatenate_flat(arrays: &[&Array]) -> Result<Array, CAbiError> {
    let arrays_handles = arrays.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
    let arrays_length = i64::try_from(arrays.len()).map_err(|_| CAbiError { operation: "ndarray.shape.concatenate_flat", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_concatenate_flat(arrays_handles.as_ptr(), arrays_length, &mut raw) };
    check_status("ndarray.shape.concatenate_flat", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.concatenate_flat", status: -1 })?;
    Ok(Array { raw })
}

/// Concatenates arrays and converts the result under an explicit dtype and casting policy.
pub fn concatenate_with_dtype(arrays: &[&Array], axis: i64, dtype: i64, casting: i64) -> Result<Array, CAbiError> {
    let arrays_handles = arrays.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
    let arrays_length = i64::try_from(arrays.len()).map_err(|_| CAbiError { operation: "ndarray.shape.concatenate_with_dtype", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_concatenate_with_dtype(arrays_handles.as_ptr(), arrays_length, axis, dtype as i32, casting as i32, &mut raw) };
    check_status("ndarray.shape.concatenate_with_dtype", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.concatenate_with_dtype", status: -1 })?;
    Ok(Array { raw })
}

/// Flattens and concatenates arrays before explicit dtype conversion.
pub fn concatenate_flat_with_dtype(arrays: &[&Array], dtype: i64, casting: i64) -> Result<Array, CAbiError> {
    let arrays_handles = arrays.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
    let arrays_length = i64::try_from(arrays.len()).map_err(|_| CAbiError { operation: "ndarray.shape.concatenate_flat_with_dtype", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_concatenate_flat_with_dtype(arrays_handles.as_ptr(), arrays_length, dtype as i32, casting as i32, &mut raw) };
    check_status("ndarray.shape.concatenate_flat_with_dtype", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.concatenate_flat_with_dtype", status: -1 })?;
    Ok(Array { raw })
}

/// Concatenates arrays into an exact-shape destination under an explicit casting policy.
pub fn concatenate_into(arrays: &[&Array], axis: i64, destination: &Array, casting: i64) -> Result<(), CAbiError> {
    let arrays_handles = arrays.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
    let arrays_length = i64::try_from(arrays.len()).map_err(|_| CAbiError { operation: "ndarray.shape.concatenate_into", status: -2 })?;
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_concatenate_into(arrays_handles.as_ptr(), arrays_length, axis, destination.raw.as_ptr(), casting as i32) };
    check_status("ndarray.shape.concatenate_into", status, 0)?;
    Ok(())
}

/// Flattens and concatenates arrays into an exact-shape destination.
pub fn concatenate_flat_into(arrays: &[&Array], destination: &Array, casting: i64) -> Result<(), CAbiError> {
    let arrays_handles = arrays.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
    let arrays_length = i64::try_from(arrays.len()).map_err(|_| CAbiError { operation: "ndarray.shape.concatenate_flat_into", status: -2 })?;
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_concatenate_flat_into(arrays_handles.as_ptr(), arrays_length, destination.raw.as_ptr(), casting as i32) };
    check_status("ndarray.shape.concatenate_flat_into", status, 0)?;
    Ok(())
}

/// Builds one shared basic-indexing view; Int minimum denotes an omitted bound and ellipsis_index expands omitted source axes.
pub fn basic_slice(array: &Array, starts: &[i64], stops: &[i64], steps: &[i64], ellipsis_index: i64, squeeze_axes: &[i64], new_axes: &[i64]) -> Result<Array, CAbiError> {
    let starts_length = i64::try_from(starts.len()).map_err(|_| CAbiError { operation: "ndarray.shape.basic_slice", status: -2 })?;
    let stops_length = i64::try_from(stops.len()).map_err(|_| CAbiError { operation: "ndarray.shape.basic_slice", status: -2 })?;
    let steps_length = i64::try_from(steps.len()).map_err(|_| CAbiError { operation: "ndarray.shape.basic_slice", status: -2 })?;
    let squeeze_axes_length = i64::try_from(squeeze_axes.len()).map_err(|_| CAbiError { operation: "ndarray.shape.basic_slice", status: -2 })?;
    let new_axes_length = i64::try_from(new_axes.len()).map_err(|_| CAbiError { operation: "ndarray.shape.basic_slice", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_basic_slice(array.raw.as_ptr(), starts.as_ptr(), starts_length, stops.as_ptr(), stops_length, steps.as_ptr(), steps_length, ellipsis_index, squeeze_axes.as_ptr(), squeeze_axes_length, new_axes.as_ptr(), new_axes_length, &mut raw) };
    check_status("ndarray.shape.basic_slice", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.basic_slice", status: -1 })?;
    Ok(Array { raw })
}

/// Takes an integer index Array while preserving every index dimension in the output shape.
pub fn take_indices(array: &Array, indices: &Array, axis: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_take_indices(array.raw.as_ptr(), indices.raw.as_ptr(), axis, &mut raw) };
    check_status("ndarray.indexing.take_indices", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.indexing.take_indices", status: -1 })?;
    Ok(Array { raw })
}

/// Takes an integer index Array using raise, wrap, or clip normalization while preserving index dimensions.
pub fn take_indices_mode(array: &Array, indices: &Array, axis: i64, mode: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_take_indices_mode(array.raw.as_ptr(), indices.raw.as_ptr(), axis, mode as i32, &mut raw) };
    check_status("ndarray.indexing.take_indices_mode", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.indexing.take_indices_mode", status: -1 })?;
    Ok(Array { raw })
}

/// Pads each axis with explicit leading and trailing constants.
pub fn pad_constant(array: &Array, leading: &[i64], trailing: &[i64], leading_values: &[f64], trailing_values: &[f64]) -> Result<Array, CAbiError> {
    let leading_length = i64::try_from(leading.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_constant", status: -2 })?;
    let trailing_length = i64::try_from(trailing.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_constant", status: -2 })?;
    let leading_values_length = i64::try_from(leading_values.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_constant", status: -2 })?;
    let trailing_values_length = i64::try_from(trailing_values.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_constant", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_pad_constant(array.raw.as_ptr(), leading.as_ptr(), leading_length, trailing.as_ptr(), trailing_length, leading_values.as_ptr(), leading_values_length, trailing_values.as_ptr(), trailing_values_length, &mut raw) };
    check_status("ndarray.shape.pad_constant", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.pad_constant", status: -1 })?;
    Ok(Array { raw })
}

/// Pads each axis by repeating its nearest edge value.
pub fn pad_edge(array: &Array, leading: &[i64], trailing: &[i64]) -> Result<Array, CAbiError> {
    let leading_length = i64::try_from(leading.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_edge", status: -2 })?;
    let trailing_length = i64::try_from(trailing.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_edge", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_pad_edge(array.raw.as_ptr(), leading.as_ptr(), leading_length, trailing.as_ptr(), trailing_length, &mut raw) };
    check_status("ndarray.shape.pad_edge", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.pad_edge", status: -1 })?;
    Ok(Array { raw })
}

/// Pads each axis with a reflection that excludes the edge value.
pub fn pad_reflect(array: &Array, leading: &[i64], trailing: &[i64]) -> Result<Array, CAbiError> {
    let leading_length = i64::try_from(leading.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_reflect", status: -2 })?;
    let trailing_length = i64::try_from(trailing.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_reflect", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_pad_reflect(array.raw.as_ptr(), leading.as_ptr(), leading_length, trailing.as_ptr(), trailing_length, &mut raw) };
    check_status("ndarray.shape.pad_reflect", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.pad_reflect", status: -1 })?;
    Ok(Array { raw })
}

/// Pads each axis with a reflection that includes the edge value.
pub fn pad_symmetric(array: &Array, leading: &[i64], trailing: &[i64]) -> Result<Array, CAbiError> {
    let leading_length = i64::try_from(leading.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_symmetric", status: -2 })?;
    let trailing_length = i64::try_from(trailing.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_symmetric", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_pad_symmetric(array.raw.as_ptr(), leading.as_ptr(), leading_length, trailing.as_ptr(), trailing_length, &mut raw) };
    check_status("ndarray.shape.pad_symmetric", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.pad_symmetric", status: -1 })?;
    Ok(Array { raw })
}

/// Pads each axis by wrapping values from its opposite end.
pub fn pad_wrap(array: &Array, leading: &[i64], trailing: &[i64]) -> Result<Array, CAbiError> {
    let leading_length = i64::try_from(leading.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_wrap", status: -2 })?;
    let trailing_length = i64::try_from(trailing.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_wrap", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_pad_wrap(array.raw.as_ptr(), leading.as_ptr(), leading_length, trailing.as_ptr(), trailing_length, &mut raw) };
    check_status("ndarray.shape.pad_wrap", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.pad_wrap", status: -1 })?;
    Ok(Array { raw })
}
