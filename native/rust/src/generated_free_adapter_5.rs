/// Evaluates over a two-dimensional Cartesian grid.
pub fn leggrid2d(x: &Array, y: &Array, coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_leggrid2d(x.raw.as_ptr(), y.raw.as_ptr(), coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.leggrid2d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.leggrid2d", status: -1 })?;
    Ok(Array { raw })
}
/// Evaluates over a three-dimensional Cartesian grid.
pub fn leggrid3d(x: &Array, y: &Array, z: &Array, coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_leggrid3d(x.raw.as_ptr(), y.raw.as_ptr(), z.raw.as_ptr(), coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.leggrid3d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.leggrid3d", status: -1 })?;
    Ok(Array { raw })
}
/// Builds a Vandermonde matrix.
pub fn legvander(coordinates: &Array, degree: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legvander(coordinates.raw.as_ptr(), degree, &mut raw) };
    check_status("ndarray.array.legvander", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legvander", status: -1 })?;
    Ok(Array { raw })
}

/// Builds a two-dimensional Vandermonde matrix.
pub fn legvander2d(x: &Array, y: &Array, x_degree: i64, y_degree: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legvander2d(x.raw.as_ptr(), y.raw.as_ptr(), x_degree, y_degree, &mut raw) };
    check_status("ndarray.array.legvander2d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legvander2d", status: -1 })?;
    Ok(Array { raw })
}

/// Builds a three-dimensional Vandermonde matrix.
pub fn legvander3d(x: &Array, y: &Array, z: &Array, x_degree: i64, y_degree: i64, z_degree: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legvander3d(x.raw.as_ptr(), y.raw.as_ptr(), z.raw.as_ptr(), x_degree, y_degree, z_degree, &mut raw) };
    check_status("ndarray.array.legvander3d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legvander3d", status: -1 })?;
    Ok(Array { raw })
}

/// Returns Gaussian quadrature nodes and weights.
pub fn leggauss(degree: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_leggauss(degree, &mut raw) };
    check_status("ndarray.array.leggauss", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.leggauss", status: -1 })?;
    Ok(Array { raw })
}

/// Evaluates the orthogonality weight function.
pub fn legweight(coordinates: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legweight(coordinates.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.legweight", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legweight", status: -1 })?;
    Ok(Array { raw })
}

/// Differentiates Legendre coefficient lanes along an axis with derivative scaling.
pub fn legder_axis(coefficients: &Array, order: i64, scale: f64, axis: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legder_axis(coefficients.raw.as_ptr(), order, scale, axis, &mut raw) };
    check_status("ndarray.array.legder_axis", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legder_axis", status: -1 })?;
    Ok(Array { raw })
}

/// Integrates Legendre coefficient lanes along an axis with constants, a lower bound, and scaling.
pub fn legint_axis(coefficients: &Array, order: i64, constants: &Array, lower_bound: f64, scale: f64, axis: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legint_axis(coefficients.raw.as_ptr(), order, constants.raw.as_ptr(), lower_bound, scale, axis, &mut raw) };
    check_status("ndarray.array.legint_axis", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legint_axis", status: -1 })?;
    Ok(Array { raw })
}

/// Evaluates Legendre coefficient tensors with Cartesian or broadcast semantics.
pub fn legval_tensor(coordinates: &Array, coefficients: &Array, tensor: bool) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legval_tensor(coordinates.raw.as_ptr(), coefficients.raw.as_ptr(), tensor as i32, &mut raw) };
    check_status("ndarray.array.legval_tensor", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legval_tensor", status: -1 })?;
    Ok(Array { raw })
}

/// Fits selected Legendre degrees to scalar or matrix-valued samples.
pub fn legfit_degrees(coordinates: &Array, values: &Array, degrees: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legfit_degrees(coordinates.raw.as_ptr(), values.raw.as_ptr(), degrees.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.legfit_degrees", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legfit_degrees", status: -1 })?;
    Ok(Array { raw })
}

/// Fits selected Legendre degrees using sample weights.
pub fn legfit_weighted_degrees(coordinates: &Array, values: &Array, degrees: &Array, weights: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legfit_weighted_degrees(coordinates.raw.as_ptr(), values.raw.as_ptr(), degrees.raw.as_ptr(), weights.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.legfit_weighted_degrees", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legfit_weighted_degrees", status: -1 })?;
    Ok(Array { raw })
}

/// Provides NumPy-compatible bartlett behavior.
pub fn bartlett(length: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_bartlett(length, &mut raw) };
    check_status("ndarray.numpy.bartlett", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.bartlett", status: -1 })?;
    Ok(Array { raw })
}

/// Provides NumPy-compatible blackman behavior.
pub fn blackman(length: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_blackman(length, &mut raw) };
    check_status("ndarray.numpy.blackman", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.blackman", status: -1 })?;
    Ok(Array { raw })
}

/// Provides NumPy-compatible hamming behavior.
pub fn hamming(length: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_hamming(length, &mut raw) };
    check_status("ndarray.numpy.hamming", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.hamming", status: -1 })?;
    Ok(Array { raw })
}

/// Provides NumPy-compatible hanning behavior.
pub fn hanning(length: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_hanning(length, &mut raw) };
    check_status("ndarray.numpy.hanning", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.hanning", status: -1 })?;
    Ok(Array { raw })
}

/// Provides NumPy-compatible kaiser behavior.
pub fn kaiser(length: i64, beta: f64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_kaiser(length, beta, &mut raw) };
    check_status("ndarray.numpy.kaiser", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.kaiser", status: -1 })?;
    Ok(Array { raw })
}

/// Provides NumPy-compatible block behavior.
pub fn block_native(arrays: &[&Array], layout: &[i64]) -> Result<Array, CAbiError> {
    let arrays_handles = arrays.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
    let arrays_length = i64::try_from(arrays.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.block_native", status: -2 })?;
    let layout_length = i64::try_from(layout.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.block_native", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_block(arrays_handles.as_ptr(), arrays_length, layout.as_ptr(), layout_length, &mut raw) };
    check_status("ndarray.numpy.block_native", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.block_native", status: -1 })?;
    Ok(Array { raw })
}

/// Provides NumPy-compatible column stack behavior.
pub fn column_stack(arrays: &[&Array]) -> Result<Array, CAbiError> {
    let arrays_handles = arrays.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
    let arrays_length = i64::try_from(arrays.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.column_stack", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_column_stack(arrays_handles.as_ptr(), arrays_length, &mut raw) };
    check_status("ndarray.numpy.column_stack", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.column_stack", status: -1 })?;
    Ok(Array { raw })
}

/// Provides NumPy-compatible dstack behavior.
pub fn dstack(arrays: &[&Array]) -> Result<Array, CAbiError> {
    let arrays_handles = arrays.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
    let arrays_length = i64::try_from(arrays.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.dstack", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_dstack(arrays_handles.as_ptr(), arrays_length, &mut raw) };
    check_status("ndarray.numpy.dstack", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.dstack", status: -1 })?;
    Ok(Array { raw })
}

/// Provides NumPy-compatible hstack behavior.
pub fn hstack(arrays: &[&Array]) -> Result<Array, CAbiError> {
    let arrays_handles = arrays.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
    let arrays_length = i64::try_from(arrays.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.hstack", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_hstack(arrays_handles.as_ptr(), arrays_length, &mut raw) };
    check_status("ndarray.numpy.hstack", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.hstack", status: -1 })?;
    Ok(Array { raw })
}

/// Provides NumPy-compatible row stack behavior.
pub fn row_stack(arrays: &[&Array]) -> Result<Array, CAbiError> {
    let arrays_handles = arrays.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
    let arrays_length = i64::try_from(arrays.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.row_stack", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_row_stack(arrays_handles.as_ptr(), arrays_length, &mut raw) };
    check_status("ndarray.numpy.row_stack", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.row_stack", status: -1 })?;
    Ok(Array { raw })
}

/// Provides NumPy-compatible select behavior.
pub fn select(conditions: &[&Array], choices: &[&Array], default_value: &Array) -> Result<Array, CAbiError> {
    let conditions_handles = conditions.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
    let conditions_length = i64::try_from(conditions.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.select", status: -2 })?;
    let choices_handles = choices.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
    let choices_length = i64::try_from(choices.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.select", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_select(conditions_handles.as_ptr(), conditions_length, choices_handles.as_ptr(), choices_length, default_value.raw.as_ptr(), &mut raw) };
    check_status("ndarray.numpy.select", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.select", status: -1 })?;
    Ok(Array { raw })
}

/// Provides NumPy-compatible vstack behavior.
pub fn vstack(arrays: &[&Array]) -> Result<Array, CAbiError> {
    let arrays_handles = arrays.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
    let arrays_length = i64::try_from(arrays.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.vstack", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_vstack(arrays_handles.as_ptr(), arrays_length, &mut raw) };
    check_status("ndarray.numpy.vstack", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.vstack", status: -1 })?;
    Ok(Array { raw })
}
