use super::*;

impl Array {
    /// Returns a Bool mask identifying infinite components.
    pub fn is_infinite(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_is_infinite(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.is_infinite", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.is_infinite", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a Bool mask identifying values with finite components.
    pub fn is_finite(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_is_finite(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.is_finite", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.is_finite", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise complex conjugate.
    pub fn conjugate(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_conjugate(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.conjugate", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.conjugate", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns two raised to each element.
    pub fn exp2(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_exp2(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.exp2", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.exp2", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns real elementwise cube roots.
    pub fn cbrt(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_cbrt(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.cbrt", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.cbrt", status: -1 })?;
        Ok(Self { raw })
    }

    /// Converts real angles from radians to degrees.
    pub fn degrees(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_degrees(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.degrees", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.degrees", status: -1 })?;
        Ok(Self { raw })
    }

    /// Converts real angles from degrees to radians.
    pub fn radians(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_radians(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.radians", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.radians", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a Bool mask preserving negative-zero sign bits.
    pub fn sign_bit(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_sign_bit(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.sign_bit", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.sign_bit", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns mantissa and integer-exponent arrays.
    pub fn frexp_all(&self) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_frexp_all(self.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.frexp_all", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.frexp_all", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.array.frexp_all", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Returns fractional and integral-part arrays.
    pub fn modf_all(&self) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_modf_all(self.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.modf_all", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.modf_all", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.array.modf_all", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Reduces normalized axes; an empty axis list reduces all axes.
    pub fn sum(&self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.sum", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_sum(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.sum", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.sum", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reduces normalized axes using NumPy-compatible product semantics.
    pub fn product(&self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.product", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_product(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.product", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.product", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reduces normalized axes using NumPy-compatible mean semantics.
    pub fn mean(&self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.mean", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_mean(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.mean", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.mean", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns means over elements selected by a broadcast-compatible truth mask.
    pub fn mean_where(&self, mask: &Self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.mean_where", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_mean_where(self.raw.as_ptr(), mask.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.mean_where", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.mean_where", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns means using an explicit result dtype.
    pub fn mean_as(&self, axes: &[i64], keep_dims: i64, dtype: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.mean_as", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_mean_as(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, dtype as i32, &mut raw) };
        check_status("ndarray.array.mean_as", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.mean_as", status: -1 })?;
        Ok(Self { raw })
    }

    /// Writes means into an exactly shaped writable destination.
    pub fn mean_into(&mut self, array: &Self, axes: &[i64], keep_dims: i64) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.mean_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_mean_into(self.raw.as_ptr(), array.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32) };
        check_status("ndarray.array.mean_into", status, 0)?;
        Ok(())
    }

    /// Reduces normalized axes using NumPy-compatible minimum semantics.
    pub fn minimum(&self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.minimum", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_minimum(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.minimum", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.minimum", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reduces normalized axes using NumPy-compatible maximum semantics.
    pub fn maximum(&self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.maximum", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_maximum(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.maximum", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.maximum", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reduces normalized axes using NumPy-compatible any semantics.
    pub fn any(&self, axes: &[i64], keep_dims: bool) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.any", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_any(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.any", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.any", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reduces normalized axes using NumPy-compatible all semantics.
    pub fn all(&self, axes: &[i64], keep_dims: bool) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.all", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_all(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.all", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.all", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reduces normalized axes using NumPy-compatible argmin semantics.
    pub fn argmin(&self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.argmin", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_argmin(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.argmin", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.argmin", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reduces normalized axes using NumPy-compatible argmax semantics.
    pub fn argmax(&self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.argmax", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_argmax(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.argmax", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.argmax", status: -1 })?;
        Ok(Self { raw })
    }

    /// Writes argmin indexes into an exactly shaped writable integer or Bool destination.
    pub fn argmin_into(&mut self, array: &Self, axes: &[i64], keep_dims: i64) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.argmin_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_argmin_into(self.raw.as_ptr(), array.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32) };
        check_status("ndarray.array.argmin_into", status, 0)?;
        Ok(())
    }

    /// Writes argmax indexes into an exactly shaped writable integer or Bool destination.
    pub fn argmax_into(&mut self, array: &Self, axes: &[i64], keep_dims: i64) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.argmax_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_argmax_into(self.raw.as_ptr(), array.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32) };
        check_status("ndarray.array.argmax_into", status, 0)?;
        Ok(())
    }

}
