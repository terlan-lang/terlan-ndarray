use super::*;

impl Array {
    /// Provides NumPy-compatible asarray chkfinite behavior.
    pub fn asarray_chkfinite(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_asarray_chkfinite(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.asarray_chkfinite", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.asarray_chkfinite", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a finite-only copy in an explicit dtype and memory order.
    pub fn asarray_chkfinite_as(&self, requested_dtype: i64, order: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_asarray_chkfinite_as(self.raw.as_ptr(), requested_dtype as i32, order as i32, &mut raw) };
        check_status("ndarray.numpy.asarray_chkfinite_as", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.asarray_chkfinite_as", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible asfarray behavior.
    pub fn asfarray(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_asfarray(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.asfarray", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.asfarray", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns repeated differences after prepending compatible values along one axis.
    pub fn difference_prepend(&self, prepend: &Self, order: i64, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_difference_prepend(self.raw.as_ptr(), prepend.raw.as_ptr(), order, axis, &mut raw) };
        check_status("ndarray.numpy.difference_prepend", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.difference_prepend", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns repeated differences after appending compatible values along one axis.
    pub fn difference_append(&self, append: &Self, order: i64, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_difference_append(self.raw.as_ptr(), append.raw.as_ptr(), order, axis, &mut raw) };
        check_status("ndarray.numpy.difference_append", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.difference_append", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns repeated differences with compatible values on both ends of one axis.
    pub fn difference_surround(&self, prepend: &Self, append: &Self, order: i64, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_difference_surround(self.raw.as_ptr(), prepend.raw.as_ptr(), append.raw.as_ptr(), order, axis, &mut raw) };
        check_status("ndarray.numpy.difference_surround", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.difference_surround", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a uniform-spacing gradient with first- or second-order boundaries.
    pub fn gradient_edge(&self, spacing: f64, axis: i64, edge_order: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_gradient_edge(self.raw.as_ptr(), spacing, axis, edge_order as i32, &mut raw) };
        check_status("ndarray.numpy.gradient_edge", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.gradient_edge", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a gradient along one axis using rank-one sample coordinates.
    pub fn gradient_coordinates(&self, coordinates: &Self, axis: i64, edge_order: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_gradient_coordinates(self.raw.as_ptr(), coordinates.raw.as_ptr(), axis, edge_order as i32, &mut raw) };
        check_status("ndarray.numpy.gradient_coordinates", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.gradient_coordinates", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns one uniform-spacing gradient for every source axis.
    pub fn gradient_all(&self, spacing: f64, edge_order: i64) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_gradient_all(self.raw.as_ptr(), spacing, edge_order as i32, &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.numpy.gradient_all", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.numpy.gradient_all", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.numpy.gradient_all", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Returns gradients for selected axes using one scalar spacing per axis.
    pub fn gradient_uniform_axes(&self, spacings: &[f64], axes: &[i64], edge_order: i64) -> Result<Vec<Array>, CAbiError> {
        let spacings_length = i64::try_from(spacings.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.gradient_uniform_axes", status: -2 })?;
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.gradient_uniform_axes", status: -2 })?;
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_gradient_uniform_axes(self.raw.as_ptr(), spacings.as_ptr(), spacings_length, axes.as_ptr(), axes_length, edge_order as i32, &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.numpy.gradient_uniform_axes", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.numpy.gradient_uniform_axes", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.numpy.gradient_uniform_axes", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Returns gradients for selected axes using scalar or coordinate spacing arrays.
    pub fn gradient_axes(&self, spacings: &[&Array], axes: &[i64], edge_order: i64) -> Result<Vec<Array>, CAbiError> {
        let spacings_handles = spacings.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
        let spacings_length = i64::try_from(spacings.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.gradient_axes", status: -2 })?;
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.gradient_axes", status: -2 })?;
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_gradient_axes(self.raw.as_ptr(), spacings_handles.as_ptr(), spacings_length, axes.as_ptr(), axes_length, edge_order as i32, &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.numpy.gradient_axes", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.numpy.gradient_axes", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.numpy.gradient_axes", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Unwraps a real-valued signal using an explicit discontinuity threshold.
    pub fn unwrap_discontinuity(&self, period: f64, discontinuity: f64, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_unwrap_discontinuity(self.raw.as_ptr(), period, discontinuity, axis, &mut raw) };
        check_status("ndarray.numpy.unwrap_discontinuity", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.unwrap_discontinuity", status: -1 })?;
        Ok(Self { raw })
    }

    /// Unwraps using integral period arithmetic and preserves integral semantics.
    pub fn unwrap_integer_period(&self, period: i64, discontinuity: i64, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_unwrap_integer_period(self.raw.as_ptr(), period, discontinuity, axis, &mut raw) };
        check_status("ndarray.numpy.unwrap_integer_period", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.unwrap_integer_period", status: -1 })?;
        Ok(Self { raw })
    }

    /// Interpolates with the first and last samples as the default boundaries.
    pub fn interpolate_default(&self, coordinates: &Self, samples: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_interpolate_default(self.raw.as_ptr(), coordinates.raw.as_ptr(), samples.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.interpolate_default", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.interpolate_default", status: -1 })?;
        Ok(Self { raw })
    }

    /// Interpolates with explicit real or complex scalar boundary arrays.
    pub fn interpolate_boundaries(&self, coordinates: &Self, samples: &Self, left: &Self, right: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_interpolate_boundaries(self.raw.as_ptr(), coordinates.raw.as_ptr(), samples.raw.as_ptr(), left.raw.as_ptr(), right.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.interpolate_boundaries", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.interpolate_boundaries", status: -1 })?;
        Ok(Self { raw })
    }

    /// Interpolates over normalized and sorted periodic coordinates.
    pub fn interpolate_periodic(&self, coordinates: &Self, samples: &Self, period: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_interpolate_periodic(self.raw.as_ptr(), coordinates.raw.as_ptr(), samples.raw.as_ptr(), period, &mut raw) };
        check_status("ndarray.numpy.interpolate_periodic", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.interpolate_periodic", status: -1 })?;
        Ok(Self { raw })
    }

    /// Counts occurrences of each nonnegative integer value.
    pub fn numpy_bincount(&self, minimum_length: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_bincount_unweighted(self.raw.as_ptr(), minimum_length, &mut raw) };
        check_status("ndarray.numpy.bincount", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.bincount", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible average behavior.
    pub fn average(&self, axes: &[i64], keep_dims: bool) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.average", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_average(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.numpy.average", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.average", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible bitwise not behavior.
    pub fn bitwise_not(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_bitwise_not(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.bitwise_not", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.bitwise_not", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible convolve behavior.
    pub fn convolve(&self, rhs: &Self, mode: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_convolve(self.raw.as_ptr(), rhs.raw.as_ptr(), mode, &mut raw) };
        check_status("ndarray.numpy.convolve", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.convolve", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible correlate behavior.
    pub fn correlate(&self, rhs: &Self, mode: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_correlate(self.raw.as_ptr(), rhs.raw.as_ptr(), mode, &mut raw) };
        check_status("ndarray.numpy.correlate", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.correlate", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible divmod behavior.
    pub fn divmod(&self, rhs: &Self) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_divmod(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.numpy.divmod", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.numpy.divmod", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.numpy.divmod", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Provides NumPy-compatible fix behavior.
    pub fn fix(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_fix(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.fix", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.fix", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible float power behavior.
    pub fn float_power(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_float_power(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.float_power", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.float_power", status: -1 })?;
        Ok(Self { raw })
    }

}
