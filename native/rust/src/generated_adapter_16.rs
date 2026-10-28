use super::*;

impl Array {
    /// Raises a coefficient vector to a nonnegative integer power.
    pub fn polypow(&self, exponent: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polypow(self.raw.as_ptr(), exponent, &mut raw) };
        check_status("ndarray.array.polypow", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polypow", status: -1 })?;
        Ok(Self { raw })
    }

    /// Removes redundant leading coefficients under a tolerance.
    pub fn polytrim(&self, tolerance: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polytrim(self.raw.as_ptr(), tolerance, &mut raw) };
        check_status("ndarray.array.polytrim", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polytrim", status: -1 })?;
        Ok(Self { raw })
    }

    /// Evaluates a monic polynomial directly from roots.
    pub fn polyvalfromroots(&self, roots: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polyvalfromroots(self.raw.as_ptr(), roots.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.polyvalfromroots", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyvalfromroots", status: -1 })?;
        Ok(Self { raw })
    }

    /// Differentiates tensor coefficients along one axis with linear-variable scaling.
    pub fn polyder_axis(&self, order: i64, scale: f64, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polyder_axis(self.raw.as_ptr(), order, scale, axis, &mut raw) };
        check_status("ndarray.array.polyder_axis", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyder_axis", status: -1 })?;
        Ok(Self { raw })
    }

    /// Integrates tensor coefficients along one axis with per-order constants, a lower bound, and scaling.
    pub fn polyint_axis(&self, order: i64, constants: &Self, lower_bound: f64, scale: f64, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polyint_axis(self.raw.as_ptr(), order, constants.raw.as_ptr(), lower_bound, scale, axis, &mut raw) };
        check_status("ndarray.array.polyint_axis", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyint_axis", status: -1 })?;
        Ok(Self { raw })
    }

    /// Evaluates root-defined polynomial tensors in Cartesian or broadcast mode.
    pub fn polyvalfromroots_tensor(&self, roots: &Self, tensor: bool) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polyvalfromroots_tensor(self.raw.as_ptr(), roots.raw.as_ptr(), tensor as i32, &mut raw) };
        check_status("ndarray.array.polyvalfromroots_tensor", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyvalfromroots_tensor", status: -1 })?;
        Ok(Self { raw })
    }

    /// Fits selected power-basis degrees to vector or matrix observations.
    pub fn polyfit_degrees(&self, values: &Self, degrees: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polyfit_degrees(self.raw.as_ptr(), values.raw.as_ptr(), degrees.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.polyfit_degrees", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyfit_degrees", status: -1 })?;
        Ok(Self { raw })
    }

    /// Fits selected power-basis degrees with one weight per observation.
    pub fn polyfit_weighted_degrees(&self, values: &Self, degrees: &Self, weights: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polyfit_weighted_degrees(self.raw.as_ptr(), values.raw.as_ptr(), degrees.raw.as_ptr(), weights.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.polyfit_weighted_degrees", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyfit_weighted_degrees", status: -1 })?;
        Ok(Self { raw })
    }

    /// Fits contiguous power degrees and returns coefficients plus scaled or unscaled covariance.
    pub fn polyfit_covariance(&self, values: &Self, degree: i64, scale_covariance: bool) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polyfit_covariance(self.raw.as_ptr(), values.raw.as_ptr(), degree, scale_covariance as i32, &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.polyfit_covariance", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.polyfit_covariance", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.array.polyfit_covariance", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Fits weighted contiguous power degrees and returns coefficients plus scaled or unscaled covariance.
    pub fn polyfit_weighted_covariance(&self, values: &Self, degree: i64, weights: &Self, scale_covariance: bool) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polyfit_weighted_covariance(self.raw.as_ptr(), values.raw.as_ptr(), degree, weights.raw.as_ptr(), scale_covariance as i32, &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.polyfit_weighted_covariance", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.polyfit_weighted_covariance", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.array.polyfit_weighted_covariance", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Builds a power Vandermonde matrix with columns from degree zero upward.
    pub fn polyvander(&self, degree: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_polyvander(self.raw.as_ptr(), degree, &mut raw) };
        check_status("ndarray.array.polyvander", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.polyvander", status: -1 })?;
        Ok(Self { raw })
    }

    /// Contracts two tensors over paired normalized axis lists.
    pub fn tensordot(&self, rhs: &Self, lhs_axes: &[i64], rhs_axes: &[i64]) -> Result<Self, CAbiError> {
        let lhs_axes_length = i64::try_from(lhs_axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.tensordot", status: -2 })?;
        let rhs_axes_length = i64::try_from(rhs_axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.tensordot", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_tensordot(self.raw.as_ptr(), rhs.raw.as_ptr(), lhs_axes.as_ptr(), lhs_axes_length, rhs_axes.as_ptr(), rhs_axes_length, &mut raw) };
        check_status("ndarray.array.tensordot", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.tensordot", status: -1 })?;
        Ok(Self { raw })
    }

    /// Computes a NumPy-compatible rank-padded Kronecker product.
    pub fn kron(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_kron(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.kron", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.kron", status: -1 })?;
        Ok(Self { raw })
    }

    /// Computes a three-component vector cross product along one normalized axis.
    pub fn cross(&self, rhs: &Self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_cross(self.raw.as_ptr(), rhs.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.array.cross", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.cross", status: -1 })?;
        Ok(Self { raw })
    }

    /// Executes Einstein summation and atomically writes the result into an exactly shaped destination.
    pub fn einsum_into(&mut self, operands: &[&Array], equation: &str) -> Result<(), CAbiError> {
        let equation_c = std::ffi::CString::new(equation.as_bytes()).map_err(|_| CAbiError { operation: "ndarray.array.einsum_into", status: -3 })?;
        let operands_handles = operands.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
        let operands_length = i64::try_from(operands.len()).map_err(|_| CAbiError { operation: "ndarray.array.einsum_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_einsum_into(self.raw.as_ptr(), equation_c.as_ptr(), operands_handles.as_ptr(), operands_length) };
        check_status("ndarray.array.einsum_into", status, 0)?;
        Ok(())
    }

    /// Returns quotient and remainder.
    pub fn chebdiv(&self, rhs: &Self) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_chebdiv(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.chebdiv", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.chebdiv", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.array.chebdiv", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Returns quotient and remainder.
    pub fn legdiv(&self, rhs: &Self) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_legdiv(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.legdiv", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.legdiv", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.array.legdiv", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Returns stable lexicographic indexes for stacked keys along one normalized key axis, using the final key as primary.
    pub fn lexsort(&self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_lexsort_axis(self.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.numpy.lexsort", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.lexsort", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible allclose behavior.
    pub fn allclose(&self, rhs: &Self, relative_tolerance: f64, absolute_tolerance: f64, equal_nan: bool) -> Result<bool, CAbiError> {
        let mut out_out_value: bool = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_allclose(self.raw.as_ptr(), rhs.raw.as_ptr(), relative_tolerance, absolute_tolerance, equal_nan as i32, &mut out_out_value) };
        check_status("ndarray.numpy.allclose", status, 0)?;
        Ok(out_out_value)
    }

    /// Returns elementwise NumPy-compatible closeness.
    pub fn isclose(&self, rhs: &Self, relative_tolerance: f64, absolute_tolerance: f64, equal_nan: bool) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_isclose(self.raw.as_ptr(), rhs.raw.as_ptr(), relative_tolerance, absolute_tolerance, equal_nan as i32, &mut raw) };
        check_status("ndarray.numpy.isclose", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.isclose", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible angle behavior.
    pub fn angle(&self, degrees: bool) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_angle(self.raw.as_ptr(), degrees as i32, &mut raw) };
        check_status("ndarray.numpy.angle", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.angle", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible array equal behavior.
    pub fn array_equal(&self, rhs: &Self) -> Result<bool, CAbiError> {
        let mut out_out_value: bool = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_array_equal(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.numpy.array_equal", status, 0)?;
        Ok(out_out_value)
    }

    /// Returns strict equality while treating paired NaNs as equal.
    pub fn array_equal_nan(&self, rhs: &Self) -> Result<bool, CAbiError> {
        let mut out_out_value: bool = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_array_equal_nan(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.numpy.array_equal_nan", status, 0)?;
        Ok(out_out_value)
    }

    /// Provides NumPy-compatible array equiv behavior.
    pub fn array_equiv(&self, rhs: &Self) -> Result<bool, CAbiError> {
        let mut out_out_value: bool = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_array_equiv(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.numpy.array_equiv", status, 0)?;
        Ok(out_out_value)
    }

}
