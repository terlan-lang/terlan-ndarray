/// Subtracts coefficient vectors.
pub fn chebsub(lhs: &Array, rhs: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebsub(lhs.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.chebsub", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebsub", status: -1 })?;
    Ok(Array { raw })
}
/// Multiplies coefficient vectors.
pub fn chebmul(lhs: &Array, rhs: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebmul(lhs.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.chebmul", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebmul", status: -1 })?;
    Ok(Array { raw })
}
/// Multiplies coefficients by x.
pub fn chebmulx(coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebmulx(coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.chebmulx", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebmulx", status: -1 })?;
    Ok(Array { raw })
}

/// Raises coefficients to a nonnegative power.
pub fn chebpow(coefficients: &Array, exponent: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebpow(coefficients.raw.as_ptr(), exponent, &mut raw) };
    check_status("ndarray.array.chebpow", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebpow", status: -1 })?;
    Ok(Array { raw })
}

/// Differentiates coefficients.
pub fn chebder(coefficients: &Array, order: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebder(coefficients.raw.as_ptr(), order, &mut raw) };
    check_status("ndarray.array.chebder", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebder", status: -1 })?;
    Ok(Array { raw })
}

/// Integrates coefficients.
pub fn chebint(coefficients: &Array, order: i64, constant: f64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebint(coefficients.raw.as_ptr(), order, constant, &mut raw) };
    check_status("ndarray.array.chebint", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebint", status: -1 })?;
    Ok(Array { raw })
}

/// Fits coefficients to samples.
pub fn chebfit(coordinates: &Array, values: &Array, degree: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebfit(coordinates.raw.as_ptr(), values.raw.as_ptr(), degree, &mut raw) };
    check_status("ndarray.array.chebfit", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebfit", status: -1 })?;
    Ok(Array { raw })
}

/// Builds coefficients from roots.
pub fn chebfromroots(roots: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebfromroots(roots.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.chebfromroots", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebfromroots", status: -1 })?;
    Ok(Array { raw })
}

/// Builds a companion matrix.
pub fn chebcompanion(coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebcompanion(coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.chebcompanion", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebcompanion", status: -1 })?;
    Ok(Array { raw })
}

/// Returns polynomial roots.
pub fn chebroots(coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebroots(coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.chebroots", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebroots", status: -1 })?;
    Ok(Array { raw })
}

/// Removes insignificant high-order coefficients.
pub fn chebtrim(coefficients: &Array, tolerance: f64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebtrim(coefficients.raw.as_ptr(), tolerance, &mut raw) };
    check_status("ndarray.array.chebtrim", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebtrim", status: -1 })?;
    Ok(Array { raw })
}

/// Builds coefficients for offset plus scale times x.
pub fn chebline(offset: f64, scale: f64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebline(offset, scale, &mut raw) };
    check_status("ndarray.array.chebline", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebline", status: -1 })?;
    Ok(Array { raw })
}

/// Evaluates coefficients.
pub fn chebval(coordinates: &Array, coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebval(coordinates.raw.as_ptr(), coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.chebval", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebval", status: -1 })?;
    Ok(Array { raw })
}

/// Evaluates a two-dimensional coefficient tensor.
pub fn chebval2d(x: &Array, y: &Array, coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebval2d(x.raw.as_ptr(), y.raw.as_ptr(), coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.chebval2d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebval2d", status: -1 })?;
    Ok(Array { raw })
}

/// Evaluates a three-dimensional coefficient tensor.
pub fn chebval3d(x: &Array, y: &Array, z: &Array, coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebval3d(x.raw.as_ptr(), y.raw.as_ptr(), z.raw.as_ptr(), coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.chebval3d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebval3d", status: -1 })?;
    Ok(Array { raw })
}

/// Evaluates over a two-dimensional Cartesian grid.
pub fn chebgrid2d(x: &Array, y: &Array, coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebgrid2d(x.raw.as_ptr(), y.raw.as_ptr(), coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.chebgrid2d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebgrid2d", status: -1 })?;
    Ok(Array { raw })
}

/// Evaluates over a three-dimensional Cartesian grid.
pub fn chebgrid3d(x: &Array, y: &Array, z: &Array, coefficients: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebgrid3d(x.raw.as_ptr(), y.raw.as_ptr(), z.raw.as_ptr(), coefficients.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.chebgrid3d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebgrid3d", status: -1 })?;
    Ok(Array { raw })
}

/// Builds a Vandermonde matrix.
pub fn chebvander(coordinates: &Array, degree: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebvander(coordinates.raw.as_ptr(), degree, &mut raw) };
    check_status("ndarray.array.chebvander", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebvander", status: -1 })?;
    Ok(Array { raw })
}

/// Builds a two-dimensional Vandermonde matrix.
pub fn chebvander2d(x: &Array, y: &Array, x_degree: i64, y_degree: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebvander2d(x.raw.as_ptr(), y.raw.as_ptr(), x_degree, y_degree, &mut raw) };
    check_status("ndarray.array.chebvander2d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebvander2d", status: -1 })?;
    Ok(Array { raw })
}

/// Builds a three-dimensional Vandermonde matrix.
pub fn chebvander3d(x: &Array, y: &Array, z: &Array, x_degree: i64, y_degree: i64, z_degree: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebvander3d(x.raw.as_ptr(), y.raw.as_ptr(), z.raw.as_ptr(), x_degree, y_degree, z_degree, &mut raw) };
    check_status("ndarray.array.chebvander3d", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebvander3d", status: -1 })?;
    Ok(Array { raw })
}

/// Returns Gaussian quadrature nodes and weights.
pub fn chebgauss(degree: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebgauss(degree, &mut raw) };
    check_status("ndarray.array.chebgauss", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebgauss", status: -1 })?;
    Ok(Array { raw })
}

/// Evaluates the orthogonality weight function.
pub fn chebweight(coordinates: &Array) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebweight(coordinates.raw.as_ptr(), &mut raw) };
    check_status("ndarray.array.chebweight", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebweight", status: -1 })?;
    Ok(Array { raw })
}

/// Returns first-kind Chebyshev points.
pub fn chebpts1(count: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebpts1(count, &mut raw) };
    check_status("ndarray.array.chebpts1", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebpts1", status: -1 })?;
    Ok(Array { raw })
}

/// Returns second-kind Chebyshev points.
pub fn chebpts2(count: i64) -> Result<Array, CAbiError> {
    let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let status = unsafe { ffi::terlan_ndarray_v1_chebpts2(count, &mut raw) };
    check_status("ndarray.array.chebpts2", status, 0)?;
    let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.chebpts2", status: -1 })?;
    Ok(Array { raw })
}
