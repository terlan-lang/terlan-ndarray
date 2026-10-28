/// Pads each axis with odd reflection excluding the edge value.
pub fn pad_reflect_odd(array: &Array, leading: &[i64], trailing: &[i64]) -> Result<Array, CAbiError> {
    let leading_length = i64::try_from(leading.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_reflect_odd", status: -2 })?;
    let trailing_length = i64::try_from(trailing.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_reflect_odd", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_pad_reflect_odd(array.raw.as_ptr(), leading.as_ptr(), leading_length, trailing.as_ptr(), trailing_length, &mut raw) };
    check_status("ndarray.shape.pad_reflect_odd", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.pad_reflect_odd", status: -1 })?;
    Ok(Array { raw })
}
/// Pads each axis with odd reflection including the edge value.
pub fn pad_symmetric_odd(array: &Array, leading: &[i64], trailing: &[i64]) -> Result<Array, CAbiError> {
    let leading_length = i64::try_from(leading.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_symmetric_odd", status: -2 })?;
    let trailing_length = i64::try_from(trailing.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_symmetric_odd", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_pad_symmetric_odd(array.raw.as_ptr(), leading.as_ptr(), leading_length, trailing.as_ptr(), trailing_length, &mut raw) };
    check_status("ndarray.shape.pad_symmetric_odd", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.pad_symmetric_odd", status: -1 })?;
    Ok(Array { raw })
}
/// Pads each axis with linear ramps from explicit end values to source edges.
pub fn pad_linear_ramp(array: &Array, leading: &[i64], trailing: &[i64], leading_values: &[f64], trailing_values: &[f64]) -> Result<Array, CAbiError> {
    let leading_length = i64::try_from(leading.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_linear_ramp", status: -2 })?;
    let trailing_length = i64::try_from(trailing.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_linear_ramp", status: -2 })?;
    let leading_values_length = i64::try_from(leading_values.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_linear_ramp", status: -2 })?;
    let trailing_values_length = i64::try_from(trailing_values.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_linear_ramp", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_pad_linear_ramp(array.raw.as_ptr(), leading.as_ptr(), leading_length, trailing.as_ptr(), trailing_length, leading_values.as_ptr(), leading_values_length, trailing_values.as_ptr(), trailing_values_length, &mut raw) };
    check_status("ndarray.shape.pad_linear_ramp", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.pad_linear_ramp", status: -1 })?;
    Ok(Array { raw })
}

/// Allocates padded regions without exposing uninitialized memory and preserves source values.
pub fn pad_empty(array: &Array, leading: &[i64], trailing: &[i64]) -> Result<Array, CAbiError> {
    let leading_length = i64::try_from(leading.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_empty", status: -2 })?;
    let trailing_length = i64::try_from(trailing.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_empty", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_pad_empty(array.raw.as_ptr(), leading.as_ptr(), leading_length, trailing.as_ptr(), trailing_length, &mut raw) };
    check_status("ndarray.shape.pad_empty", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.pad_empty", status: -1 })?;
    Ok(Array { raw })
}

/// Pads with edge minimums over optional clipped statistic lengths.
pub fn pad_minimum(array: &Array, leading: &[i64], trailing: &[i64], leading_lengths: &[i64], trailing_lengths: &[i64]) -> Result<Array, CAbiError> {
    let leading_length = i64::try_from(leading.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_minimum", status: -2 })?;
    let trailing_length = i64::try_from(trailing.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_minimum", status: -2 })?;
    let leading_lengths_length = i64::try_from(leading_lengths.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_minimum", status: -2 })?;
    let trailing_lengths_length = i64::try_from(trailing_lengths.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_minimum", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_pad_minimum(array.raw.as_ptr(), leading.as_ptr(), leading_length, trailing.as_ptr(), trailing_length, leading_lengths.as_ptr(), leading_lengths_length, trailing_lengths.as_ptr(), trailing_lengths_length, &mut raw) };
    check_status("ndarray.shape.pad_minimum", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.pad_minimum", status: -1 })?;
    Ok(Array { raw })
}

/// Pads with edge maximums over optional clipped statistic lengths.
pub fn pad_maximum(array: &Array, leading: &[i64], trailing: &[i64], leading_lengths: &[i64], trailing_lengths: &[i64]) -> Result<Array, CAbiError> {
    let leading_length = i64::try_from(leading.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_maximum", status: -2 })?;
    let trailing_length = i64::try_from(trailing.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_maximum", status: -2 })?;
    let leading_lengths_length = i64::try_from(leading_lengths.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_maximum", status: -2 })?;
    let trailing_lengths_length = i64::try_from(trailing_lengths.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_maximum", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_pad_maximum(array.raw.as_ptr(), leading.as_ptr(), leading_length, trailing.as_ptr(), trailing_length, leading_lengths.as_ptr(), leading_lengths_length, trailing_lengths.as_ptr(), trailing_lengths_length, &mut raw) };
    check_status("ndarray.shape.pad_maximum", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.pad_maximum", status: -1 })?;
    Ok(Array { raw })
}

/// Pads with edge means over optional clipped statistic lengths.
pub fn pad_mean(array: &Array, leading: &[i64], trailing: &[i64], leading_lengths: &[i64], trailing_lengths: &[i64]) -> Result<Array, CAbiError> {
    let leading_length = i64::try_from(leading.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_mean", status: -2 })?;
    let trailing_length = i64::try_from(trailing.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_mean", status: -2 })?;
    let leading_lengths_length = i64::try_from(leading_lengths.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_mean", status: -2 })?;
    let trailing_lengths_length = i64::try_from(trailing_lengths.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_mean", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_pad_mean(array.raw.as_ptr(), leading.as_ptr(), leading_length, trailing.as_ptr(), trailing_length, leading_lengths.as_ptr(), leading_lengths_length, trailing_lengths.as_ptr(), trailing_lengths_length, &mut raw) };
    check_status("ndarray.shape.pad_mean", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.pad_mean", status: -1 })?;
    Ok(Array { raw })
}

/// Pads with edge medians over optional clipped statistic lengths.
pub fn pad_median(array: &Array, leading: &[i64], trailing: &[i64], leading_lengths: &[i64], trailing_lengths: &[i64]) -> Result<Array, CAbiError> {
    let leading_length = i64::try_from(leading.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_median", status: -2 })?;
    let trailing_length = i64::try_from(trailing.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_median", status: -2 })?;
    let leading_lengths_length = i64::try_from(leading_lengths.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_median", status: -2 })?;
    let trailing_lengths_length = i64::try_from(trailing_lengths.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_median", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_pad_median(array.raw.as_ptr(), leading.as_ptr(), leading_length, trailing.as_ptr(), trailing_length, leading_lengths.as_ptr(), leading_lengths_length, trailing_lengths.as_ptr(), trailing_lengths_length, &mut raw) };
    check_status("ndarray.shape.pad_median", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.pad_median", status: -1 })?;
    Ok(Array { raw })
}

/// Pads each axis with exact numerical values supplied by scalar or rank-sized arrays.
pub fn pad_constant_values(array: &Array, leading: &[i64], trailing: &[i64], leading_values: &Array, trailing_values: &Array) -> Result<Array, CAbiError> {
    let leading_length = i64::try_from(leading.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_constant_values", status: -2 })?;
    let trailing_length = i64::try_from(trailing.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_constant_values", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_pad_constant_values(array.raw.as_ptr(), leading.as_ptr(), leading_length, trailing.as_ptr(), trailing_length, leading_values.raw.as_ptr(), trailing_values.raw.as_ptr(), &mut raw) };
    check_status("ndarray.shape.pad_constant_values", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.pad_constant_values", status: -1 })?;
    Ok(Array { raw })
}

/// Pads each axis with linear ramps using exact numerical endpoints from scalar or rank-sized arrays.
pub fn pad_linear_ramp_values(array: &Array, leading: &[i64], trailing: &[i64], leading_values: &Array, trailing_values: &Array) -> Result<Array, CAbiError> {
    let leading_length = i64::try_from(leading.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_linear_ramp_values", status: -2 })?;
    let trailing_length = i64::try_from(trailing.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_linear_ramp_values", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_pad_linear_ramp_values(array.raw.as_ptr(), leading.as_ptr(), leading_length, trailing.as_ptr(), trailing_length, leading_values.raw.as_ptr(), trailing_values.raw.as_ptr(), &mut raw) };
    check_status("ndarray.shape.pad_linear_ramp_values", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.pad_linear_ramp_values", status: -1 })?;
    Ok(Array { raw })
}

/// Pads with minimum, maximum, mean, or median values selected by mode 0 through 3 over explicit edge lengths.
pub fn pad_statistical(array: &Array, leading: &[i64], trailing: &[i64], leading_lengths: &[i64], trailing_lengths: &[i64], mode: i64) -> Result<Array, CAbiError> {
    let leading_length = i64::try_from(leading.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_statistical", status: -2 })?;
    let trailing_length = i64::try_from(trailing.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_statistical", status: -2 })?;
    let leading_lengths_length = i64::try_from(leading_lengths.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_statistical", status: -2 })?;
    let trailing_lengths_length = i64::try_from(trailing_lengths.len()).map_err(|_| CAbiError { operation: "ndarray.shape.pad_statistical", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_pad_statistical(array.raw.as_ptr(), leading.as_ptr(), leading_length, trailing.as_ptr(), trailing_length, leading_lengths.as_ptr(), leading_lengths_length, trailing_lengths.as_ptr(), trailing_lengths_length, mode, &mut raw) };
    check_status("ndarray.shape.pad_statistical", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.shape.pad_statistical", status: -1 })?;
    Ok(Array { raw })
}

/// Multiplies a chain of arrays using package matmul semantics.
pub fn multi_dot(arrays: &[&Array]) -> Result<Array, CAbiError> {
    let arrays_handles = arrays.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
    let arrays_length = i64::try_from(arrays.len()).map_err(|_| CAbiError { operation: "ndarray.array.multi_dot", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_multi_dot(arrays_handles.as_ptr(), arrays_length, &mut raw) };
    check_status("ndarray.array.multi_dot", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.multi_dot", status: -1 })?;
    Ok(Array { raw })
}

/// Returns coefficients for offset plus scale times x.
pub fn polyline(offset: f64, scale: f64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_polyline(offset, scale, &mut raw) };
    check_status("ndarray.array.polyline", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyline", status: -1 })?;
    Ok(Array { raw })
}

/// Evaluates a two-dimensional coefficient grid over broadcast coordinates.
pub fn polyval2d(x: &Array, y: &Array, coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_polyval2d(x.raw.as_ptr(), y.raw.as_ptr(), coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.polyval2d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyval2d", status: -1 })?;
    Ok(Array { raw })
}

/// Evaluates a three-dimensional coefficient grid over broadcast coordinates.
pub fn polyval3d(x: &Array, y: &Array, z: &Array, coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_polyval3d(x.raw.as_ptr(), y.raw.as_ptr(), z.raw.as_ptr(), coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.polyval3d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyval3d", status: -1 })?;
    Ok(Array { raw })
}

/// Evaluates a two-dimensional coefficient grid over a Cartesian product.
pub fn polygrid2d(x: &Array, y: &Array, coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_polygrid2d(x.raw.as_ptr(), y.raw.as_ptr(), coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.polygrid2d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polygrid2d", status: -1 })?;
    Ok(Array { raw })
}

/// Evaluates a three-dimensional coefficient grid over a Cartesian product.
pub fn polygrid3d(x: &Array, y: &Array, z: &Array, coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_polygrid3d(x.raw.as_ptr(), y.raw.as_ptr(), z.raw.as_ptr(), coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.polygrid3d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polygrid3d", status: -1 })?;
    Ok(Array { raw })
}

/// Builds a two-dimensional broadcast power Vandermonde matrix.
pub fn polyvander2d(x: &Array, y: &Array, x_degree: i64, y_degree: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_polyvander2d(x.raw.as_ptr(), y.raw.as_ptr(), x_degree, y_degree, &mut raw) };
    check_status("ndarray.array.polyvander2d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyvander2d", status: -1 })?;
    Ok(Array { raw })
}

/// Builds a three-dimensional broadcast power Vandermonde matrix.
pub fn polyvander3d(x: &Array, y: &Array, z: &Array, x_degree: i64, y_degree: i64, z_degree: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_polyvander3d(x.raw.as_ptr(), y.raw.as_ptr(), z.raw.as_ptr(), x_degree, y_degree, z_degree, &mut raw) };
    check_status("ndarray.array.polyvander3d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyvander3d", status: -1 })?;
    Ok(Array { raw })
}

/// Executes an Einstein summation expression over one or more arrays.
pub fn einsum(operands: &[&Array], equation: &str) -> Result<Array, CAbiError> {
    let equation_c = std::ffi::CString::new(equation.as_bytes()).map_err(|_| CAbiError { operation: "ndarray.array.einsum", status: -3 })?;
    let operands_handles = operands.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
    let operands_length = i64::try_from(operands.len()).map_err(|_| CAbiError { operation: "ndarray.array.einsum", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_einsum(equation_c.as_ptr(), operands_handles.as_ptr(), operands_length, &mut raw) };
    check_status("ndarray.array.einsum", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.einsum", status: -1 })?;
    Ok(Array { raw })
}

/// Executes Einstein summation using an explicit C, Fortran, keep, or any output-order policy.
pub fn einsum_ordered(operands: &[&Array], equation: &str, order: i64) -> Result<Array, CAbiError> {
    let equation_c = std::ffi::CString::new(equation.as_bytes()).map_err(|_| CAbiError { operation: "ndarray.array.einsum_ordered", status: -3 })?;
    let operands_handles = operands.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
    let operands_length = i64::try_from(operands.len()).map_err(|_| CAbiError { operation: "ndarray.array.einsum_ordered", status: -2 })?;
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_einsum_ordered(equation_c.as_ptr(), operands_handles.as_ptr(), operands_length, order as i32, &mut raw) };
    check_status("ndarray.array.einsum_ordered", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.einsum_ordered", status: -1 })?;
    Ok(Array { raw })
}

/// Converts coefficients to the power basis.
pub fn cheb2poly(coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_cheb2poly(coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.cheb2poly", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.cheb2poly", status: -1 })?;
    Ok(Array { raw })
}

/// Converts power coefficients to the cheb basis.
pub fn poly2cheb(coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_poly2cheb(coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.poly2cheb", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.poly2cheb", status: -1 })?;
    Ok(Array { raw })
}

/// Adds coefficient vectors.
pub fn chebadd(lhs: &Array, rhs: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebadd(lhs.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.chebadd", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebadd", status: -1 })?;
    Ok(Array { raw })
}
