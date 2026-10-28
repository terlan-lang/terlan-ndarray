/// Interpolates sampled values in the Chebyshev basis.
pub fn chebinterpolate(values: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebinterpolate(values.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.chebinterpolate", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebinterpolate", status: -1 })?;
    Ok(Array { raw })
}
/// Differentiates Chebyshev coefficient lanes along an axis with derivative scaling.
pub fn chebder_axis(coefficients: &Array, order: i64, scale: f64, axis: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebder_axis(coefficients.raw.as_ptr(), order, scale, axis, &mut raw) };
    check_status("ndarray.array.chebder_axis", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebder_axis", status: -1 })?;
    Ok(Array { raw })
}
/// Integrates Chebyshev coefficient lanes along an axis with constants, a lower bound, and scaling.
pub fn chebint_axis(coefficients: &Array, order: i64, constants: &Array, lower_bound: f64, scale: f64, axis: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebint_axis(coefficients.raw.as_ptr(), order, constants.raw.as_ptr(), lower_bound, scale, axis, &mut raw) };
    check_status("ndarray.array.chebint_axis", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebint_axis", status: -1 })?;
    Ok(Array { raw })
}

/// Evaluates Chebyshev coefficient tensors with Cartesian or broadcast semantics.
pub fn chebval_tensor(coordinates: &Array, coefficients: &Array, tensor: bool) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebval_tensor(coordinates.raw.as_ptr(), coefficients.raw.as_ptr(), tensor as i32, &mut raw) };
    check_status("ndarray.array.chebval_tensor", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebval_tensor", status: -1 })?;
    Ok(Array { raw })
}

/// Fits selected Chebyshev degrees to scalar or matrix-valued samples.
pub fn chebfit_degrees(coordinates: &Array, values: &Array, degrees: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebfit_degrees(coordinates.raw.as_ptr(), values.raw.as_ptr(), degrees.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.chebfit_degrees", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebfit_degrees", status: -1 })?;
    Ok(Array { raw })
}

/// Fits selected Chebyshev degrees using sample weights.
pub fn chebfit_weighted_degrees(coordinates: &Array, values: &Array, degrees: &Array, weights: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebfit_weighted_degrees(coordinates.raw.as_ptr(), values.raw.as_ptr(), degrees.raw.as_ptr(), weights.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.chebfit_weighted_degrees", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebfit_weighted_degrees", status: -1 })?;
    Ok(Array { raw })
}

/// Converts coefficients to the power basis.
pub fn leg2poly(coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_leg2poly(coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.leg2poly", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.leg2poly", status: -1 })?;
    Ok(Array { raw })
}

/// Converts power coefficients to the leg basis.
pub fn poly2leg(coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_poly2leg(coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.poly2leg", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.poly2leg", status: -1 })?;
    Ok(Array { raw })
}

/// Adds coefficient vectors.
pub fn legadd(lhs: &Array, rhs: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legadd(lhs.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.legadd", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legadd", status: -1 })?;
    Ok(Array { raw })
}

/// Subtracts coefficient vectors.
pub fn legsub(lhs: &Array, rhs: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legsub(lhs.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.legsub", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legsub", status: -1 })?;
    Ok(Array { raw })
}

/// Multiplies coefficient vectors.
pub fn legmul(lhs: &Array, rhs: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legmul(lhs.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.legmul", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legmul", status: -1 })?;
    Ok(Array { raw })
}

/// Multiplies coefficients by x.
pub fn legmulx(coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legmulx(coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.legmulx", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legmulx", status: -1 })?;
    Ok(Array { raw })
}

/// Raises coefficients to a nonnegative power.
pub fn legpow(coefficients: &Array, exponent: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legpow(coefficients.raw.as_ptr(), exponent, &mut raw) };
    check_status("ndarray.array.legpow", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legpow", status: -1 })?;
    Ok(Array { raw })
}

/// Differentiates coefficients.
pub fn legder(coefficients: &Array, order: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legder(coefficients.raw.as_ptr(), order, &mut raw) };
    check_status("ndarray.array.legder", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legder", status: -1 })?;
    Ok(Array { raw })
}

/// Integrates coefficients.
pub fn legint(coefficients: &Array, order: i64, constant: f64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legint(coefficients.raw.as_ptr(), order, constant, &mut raw) };
    check_status("ndarray.array.legint", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legint", status: -1 })?;
    Ok(Array { raw })
}

/// Fits coefficients to samples.
pub fn legfit(coordinates: &Array, values: &Array, degree: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legfit(coordinates.raw.as_ptr(), values.raw.as_ptr(), degree, &mut raw) };
    check_status("ndarray.array.legfit", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legfit", status: -1 })?;
    Ok(Array { raw })
}

/// Builds coefficients from roots.
pub fn legfromroots(roots: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legfromroots(roots.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.legfromroots", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legfromroots", status: -1 })?;
    Ok(Array { raw })
}

/// Builds a companion matrix.
pub fn legcompanion(coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legcompanion(coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.legcompanion", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legcompanion", status: -1 })?;
    Ok(Array { raw })
}

/// Returns polynomial roots.
pub fn legroots(coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legroots(coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.legroots", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legroots", status: -1 })?;
    Ok(Array { raw })
}

/// Removes insignificant high-order coefficients.
pub fn legtrim(coefficients: &Array, tolerance: f64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legtrim(coefficients.raw.as_ptr(), tolerance, &mut raw) };
    check_status("ndarray.array.legtrim", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legtrim", status: -1 })?;
    Ok(Array { raw })
}

/// Builds coefficients for offset plus scale times x.
pub fn legline(offset: f64, scale: f64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legline(offset, scale, &mut raw) };
    check_status("ndarray.array.legline", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legline", status: -1 })?;
    Ok(Array { raw })
}

/// Evaluates coefficients.
pub fn legval(coordinates: &Array, coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legval(coordinates.raw.as_ptr(), coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.legval", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legval", status: -1 })?;
    Ok(Array { raw })
}

/// Evaluates a two-dimensional coefficient tensor.
pub fn legval2d(x: &Array, y: &Array, coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legval2d(x.raw.as_ptr(), y.raw.as_ptr(), coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.legval2d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legval2d", status: -1 })?;
    Ok(Array { raw })
}

/// Evaluates a three-dimensional coefficient tensor.
pub fn legval3d(x: &Array, y: &Array, z: &Array, coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_legval3d(x.raw.as_ptr(), y.raw.as_ptr(), z.raw.as_ptr(), coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.legval3d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.legval3d", status: -1 })?;
    Ok(Array { raw })
}
