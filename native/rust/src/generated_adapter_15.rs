use super::*;

impl Array {
    /// Computes general complex eigenvalues without materializing eigenvectors.
    pub fn eigvals(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_eigvals(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.eigvals", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.eigvals", status: -1 })?;
        Ok(Self { raw })
    }

    /// Computes ascending eigenvalues of a finite Hermitian matrix.
    pub fn eigvalsh(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_eigvalsh(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.eigvalsh", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.eigvalsh", status: -1 })?;
        Ok(Self { raw })
    }

    /// Computes Hermitian eigenvalues from a selected triangle.
    pub fn eigvalsh_general(&self, upper: bool) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_eigvalsh_general(self.raw.as_ptr(), upper, &mut raw) };
        check_status("ndarray.array.eigvalsh_general", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.eigvalsh_general", status: -1 })?;
        Ok(Self { raw })
    }

    /// Raises a square matrix to an integer power by checked exponentiation.
    pub fn matrix_power(&self, exponent: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_matrix_power(self.raw.as_ptr(), exponent, &mut raw) };
        check_status("ndarray.array.matrix_power", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.matrix_power", status: -1 })?;
        Ok(Self { raw })
    }

    /// Multiplies a chain and atomically writes the result into a validated destination.
    pub fn multi_dot_into(&mut self, arrays: &[&Array]) -> Result<(), CAbiError> {
        let arrays_handles = arrays.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
        let arrays_length = i64::try_from(arrays.len()).map_err(|_| CAbiError { operation: "ndarray.array.multi_dot_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_multi_dot_into(self.raw.as_ptr(), arrays_handles.as_ptr(), arrays_length) };
        check_status("ndarray.array.multi_dot_into", status, 0)?;
        Ok(())
    }

    /// Computes the default Euclidean vector or Frobenius matrix norm.
    pub fn norm(&self) -> Result<f64, CAbiError> {
        let mut out_out_value: f64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_norm(self.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.array.norm", status, 0)?;
        Ok(out_out_value)
    }

    /// Computes a generalized vector or matrix norm over normalized axes.
    pub fn norm_general(&self, order_kind: i64, order_value: f64, axes: &[i64], keepdims: bool) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.norm_general", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_norm_general(self.raw.as_ptr(), order_kind, order_value, axes.as_ptr(), axes_length, keepdims, &mut raw) };
        check_status("ndarray.array.norm_general", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.norm_general", status: -1 })?;
        Ok(Self { raw })
    }

    /// Computes an SVD pseudoinverse with an explicit singular-value cutoff.
    pub fn pinv(&self, tolerance: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_pinv(self.raw.as_ptr(), tolerance, &mut raw) };
        check_status("ndarray.array.pinv", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.pinv", status: -1 })?;
        Ok(Self { raw })
    }

    /// Computes determinant sign and logarithmic absolute determinant as scalar arrays.
    pub fn slogdet(&self) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_slogdet(self.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.slogdet", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.slogdet", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.array.slogdet", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Computes the inverse of a tensor interpreted as a square matrix split at an index.
    pub fn tensorinv(&self, index: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_tensorinv(self.raw.as_ptr(), index, &mut raw) };
        check_status("ndarray.array.tensorinv", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.tensorinv", status: -1 })?;
        Ok(Self { raw })
    }

    /// Solves a tensor equation after optionally moving selected coefficient axes to the end.
    pub fn tensorsolve(&self, rhs: &Self, axes: &[i64]) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.tensorsolve", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_tensorsolve(self.raw.as_ptr(), rhs.raw.as_ptr(), axes.as_ptr(), axes_length, &mut raw) };
        check_status("ndarray.array.tensorsolve", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.tensorsolve", status: -1 })?;
        Ok(Self { raw })
    }

    /// Evaluates low-to-high power-basis coefficients at every coordinate.
    pub fn polyval(&self, coefficients: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polyval(self.raw.as_ptr(), coefficients.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.polyval", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyval", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns all complex roots of a finite nonconstant power-basis polynomial.
    pub fn polyroots(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polyroots(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.polyroots", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyroots", status: -1 })?;
        Ok(Self { raw })
    }

    /// Fits low-to-high power-basis coefficients by finite SVD least squares.
    pub fn polyfit(&self, values: &Self, degree: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polyfit(self.raw.as_ptr(), values.raw.as_ptr(), degree, &mut raw) };
        check_status("ndarray.array.polyfit", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyfit", status: -1 })?;
        Ok(Self { raw })
    }

    /// Adds two low-to-high polynomial coefficient vectors.
    pub fn polyadd(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polyadd(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.polyadd", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyadd", status: -1 })?;
        Ok(Self { raw })
    }

    /// Subtracts two low-to-high polynomial coefficient vectors.
    pub fn polysub(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polysub(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.polysub", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polysub", status: -1 })?;
        Ok(Self { raw })
    }

    /// Multiplies two low-to-high polynomial coefficient vectors.
    pub fn polymul(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polymul(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.polymul", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polymul", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns polynomial quotient and remainder as two owned arrays.
    pub fn polydiv_all(&self, rhs: &Self) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polydiv_all(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.polydiv_all", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.polydiv_all", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.array.polydiv_all", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Differentiates low-to-high polynomial coefficients by a nonnegative order.
    pub fn polyder(&self, order: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polyder(self.raw.as_ptr(), order, &mut raw) };
        check_status("ndarray.array.polyder", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyder", status: -1 })?;
        Ok(Self { raw })
    }

    /// Integrates low-to-high polynomial coefficients with a repeated constant.
    pub fn polyint(&self, order: i64, constant: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polyint(self.raw.as_ptr(), order, constant, &mut raw) };
        check_status("ndarray.array.polyint", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyint", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the power-basis companion matrix used for root finding.
    pub fn polycompanion(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polycompanion(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.polycompanion", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polycompanion", status: -1 })?;
        Ok(Self { raw })
    }

    /// Converts coefficients among power, Chebyshev, and Legendre bases over a finite domain and window.
    pub fn polyconvert(&self, source_basis: i64, target_basis: i64, domain_minimum: f64, domain_maximum: f64, window_minimum: f64, window_maximum: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polyconvert(self.raw.as_ptr(), source_basis as i32, target_basis as i32, domain_minimum, domain_maximum, window_minimum, window_maximum, &mut raw) };
        check_status("ndarray.array.polyconvert", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyconvert", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns low-to-high coefficients for a monic polynomial with the supplied roots.
    pub fn polyfromroots(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polyfromroots(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.polyfromroots", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyfromroots", status: -1 })?;
        Ok(Self { raw })
    }

    /// Multiplies a coefficient vector by x.
    pub fn polymulx(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polymulx(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.polymulx", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polymulx", status: -1 })?;
        Ok(Self { raw })
    }

}
