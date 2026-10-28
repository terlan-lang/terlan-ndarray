use super::*;

impl Array {
    /// Unpacks a counted number of UInt8 bits along one normalized axis.
    pub fn unpackbits_axis(&self, axis: i64, count: i64, bit_order: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_unpackbits_axis(self.raw.as_ptr(), axis, count, bit_order, &mut raw) };
        check_status("ndarray.numpy.unpackbits_axis", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.unpackbits_axis", status: -1 })?;
        Ok(Self { raw })
    }

    /// Unpacks every available UInt8 bit along one normalized axis.
    pub fn unpackbits_axis_all(&self, axis: i64, bit_order: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_unpackbits_axis_all(self.raw.as_ptr(), axis, bit_order, &mut raw) };
        check_status("ndarray.numpy.unpackbits_axis_all", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.unpackbits_axis_all", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible alen behavior.
    pub fn alen(&self) -> Result<i64, CAbiError> {
        let mut out_out_value: i64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_alen(self.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.numpy.alen", status, 0)?;
        Ok(out_out_value)
    }

    /// Provides NumPy-compatible append behavior.
    pub fn append(&self, values: &Self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_append(self.raw.as_ptr(), values.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.numpy.append", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.append", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible broadcast arrays behavior.
    pub fn broadcast_arrays(&self, remaining: &[&Array]) -> Result<Vec<Array>, CAbiError> {
        let remaining_handles = remaining.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
        let remaining_length = i64::try_from(remaining.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.broadcast_arrays", status: -2 })?;
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_broadcast_arrays(self.raw.as_ptr(), remaining_handles.as_ptr(), remaining_length, &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.numpy.broadcast_arrays", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.numpy.broadcast_arrays", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.numpy.broadcast_arrays", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Provides NumPy-compatible broadcast shapes behavior.
    pub fn broadcast_shapes(lhs: &[i64], rhs: &[i64]) -> Result<Self, CAbiError> {
        let lhs_length = i64::try_from(lhs.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.broadcast_shapes", status: -2 })?;
        let rhs_length = i64::try_from(rhs.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.broadcast_shapes", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_broadcast_shapes(lhs.as_ptr(), lhs_length, rhs.as_ptr(), rhs_length, &mut raw) };
        check_status("ndarray.numpy.broadcast_shapes", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.broadcast_shapes", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a read-only shared view with an explicit shape and element strides.
    pub fn as_strided(&self, shape: &[i64], strides: &[i64]) -> Result<Self, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.as_strided", status: -2 })?;
        let strides_length = i64::try_from(strides.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.as_strided", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_as_strided(self.raw.as_ptr(), shape.as_ptr(), strides.as_ptr(), shape_length, strides_length, &mut raw) };
        check_status("ndarray.numpy.as_strided", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.as_strided", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a read-only shared sliding-window view, using every axis when axes is empty.
    pub fn sliding_window_view(&self, window_shape: &[i64], axes: &[i64]) -> Result<Self, CAbiError> {
        let window_shape_length = i64::try_from(window_shape.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.sliding_window_view", status: -2 })?;
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.sliding_window_view", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_sliding_window_view(self.raw.as_ptr(), window_shape.as_ptr(), window_shape_length, axes.as_ptr(), axes_length, &mut raw) };
        check_status("ndarray.numpy.sliding_window_view", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.sliding_window_view", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible copyto behavior.
    pub fn copyto(&mut self, source: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_copyto(self.raw.as_ptr(), source.raw.as_ptr()) };
        check_status("ndarray.numpy.copyto", status, 0)?;
        Ok(())
    }

    /// Broadcasts and copies values under an explicit NumPy-compatible casting rule.
    pub fn copyto_casting(&mut self, source: &Self, casting: i64) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_copyto_casting(self.raw.as_ptr(), source.raw.as_ptr(), casting as i32) };
        check_status("ndarray.numpy.copyto_casting", status, 0)?;
        Ok(())
    }

    /// Broadcasts a source and truth mask before copying selected values under an explicit casting rule.
    pub fn copyto_where(&mut self, source: &Self, mask: &Self, casting: i64) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_copyto_where(self.raw.as_ptr(), source.raw.as_ptr(), mask.raw.as_ptr(), casting as i32) };
        check_status("ndarray.numpy.copyto_where", status, 0)?;
        Ok(())
    }

    /// Provides NumPy-compatible delete behavior.
    pub fn delete(&self, indexes: &[i64]) -> Result<Self, CAbiError> {
        let indexes_length = i64::try_from(indexes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.delete", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_delete_indexes(self.raw.as_ptr(), indexes.as_ptr(), indexes_length, &mut raw) };
        check_status("ndarray.numpy.delete", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.delete", status: -1 })?;
        Ok(Self { raw })
    }

    /// Deletes normalized integer indexes along one axis.
    pub fn delete_indexes_axis(&self, indexes: &[i64], axis: i64) -> Result<Self, CAbiError> {
        let indexes_length = i64::try_from(indexes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.delete_axis", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_delete_indexes_axis(self.raw.as_ptr(), indexes.as_ptr(), indexes_length, axis, &mut raw) };
        check_status("ndarray.numpy.delete_axis", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.delete_axis", status: -1 })?;
        Ok(Self { raw })
    }

    /// Deletes positions selected by an exact-length Bool mask along one axis.
    pub fn delete_mask_axis(&self, mask: &Self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_delete_mask_axis(self.raw.as_ptr(), mask.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.numpy.delete_mask_axis", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.delete_mask_axis", status: -1 })?;
        Ok(Self { raw })
    }

    /// Extracts a matrix diagonal view or constructs a dtype-preserving diagonal matrix from a vector.
    pub fn diag(&self, offset: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_diag(self.raw.as_ptr(), offset, &mut raw) };
        check_status("ndarray.numpy.diag", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.diag", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible diag indices behavior.
    pub fn diag_indices(size: i64, rank: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_diag_indices(size, rank, &mut raw) };
        check_status("ndarray.numpy.diag_indices", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.diag_indices", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible diag indices from behavior.
    pub fn diag_indices_from(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_diag_indices_from(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.diag_indices_from", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.diag_indices_from", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible dsplit behavior.
    pub fn dsplit(&self, sections: i64) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_dsplit(self.raw.as_ptr(), sections, &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.numpy.dsplit", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.numpy.dsplit", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.numpy.dsplit", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Provides NumPy-compatible ediff1d behavior.
    pub fn ediff1d(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_ediff1d(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.ediff1d", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.ediff1d", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible fill diagonal behavior.
    pub fn fill_diagonal_native(&mut self, value: f64, wrap: bool) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_fill_diagonal_with_wrap(self.raw.as_ptr(), value, wrap) };
        check_status("ndarray.numpy.fill_diagonal_native", status, 0)?;
        Ok(())
    }

    /// Provides NumPy-compatible full behavior.
    pub fn full(shape: &[i64], value: f64) -> Result<Self, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.full", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_full(shape.as_ptr(), shape_length, value, &mut raw) };
        check_status("ndarray.numpy.full", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.full", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible full like behavior.
    pub fn full_like(&self, value: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_full_like(self.raw.as_ptr(), value, &mut raw) };
        check_status("ndarray.numpy.full_like", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.full_like", status: -1 })?;
        Ok(Self { raw })
    }

    /// Builds a one-dimensional histogram with an inferred finite range.
    pub fn numpy_histogram(&self, bins: i64, density: bool) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_numpy_histogram(self.raw.as_ptr(), bins, density, &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.numpy.histogram", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.numpy.histogram", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.numpy.histogram", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Builds a one-dimensional histogram over an explicit finite range.
    pub fn numpy_histogram_range(&self, bins: i64, lower: f64, upper: f64, density: bool) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_numpy_histogram_range(self.raw.as_ptr(), bins, lower, upper, density, &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.numpy.histogram_range", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.numpy.histogram_range", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.numpy.histogram_range", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

}
