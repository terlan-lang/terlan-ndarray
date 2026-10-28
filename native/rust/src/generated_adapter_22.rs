use super::*;

impl Array {
    /// Inserts values before one scalar index along a normalized axis.
    pub fn insert_scalar_axis(&self, index: i64, values: &Self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_insert_scalar_axis(self.raw.as_ptr(), index, values.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.numpy.insert_scalar_axis", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.insert_scalar_axis", status: -1 })?;
        Ok(Self { raw })
    }

    /// Inserts values before stable integer-index positions along a normalized axis.
    pub fn insert_indexes_axis(&self, indexes: &[i64], values: &Self, axis: i64) -> Result<Self, CAbiError> {
        let indexes_length = i64::try_from(indexes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.insert_indexes_axis", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_insert_indexes_axis(self.raw.as_ptr(), indexes.as_ptr(), indexes_length, values.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.numpy.insert_indexes_axis", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.insert_indexes_axis", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible ix  behavior.
    pub fn ix_(&self, remaining: &[&Array]) -> Result<Vec<Array>, CAbiError> {
        let remaining_handles = remaining.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
        let remaining_length = i64::try_from(remaining.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.ix_", status: -2 })?;
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_ix(self.raw.as_ptr(), remaining_handles.as_ptr(), remaining_length, &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.numpy.ix_", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.numpy.ix_", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.numpy.ix_", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Provides NumPy-compatible may share memory behavior.
    pub fn may_share_memory(&self, rhs: &Self) -> Result<bool, CAbiError> {
        let mut out_out_value: bool = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_may_share_memory(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.numpy.may_share_memory", status, 0)?;
        Ok(out_out_value)
    }

    /// Returns an exact overlap answer within a work budget and conservatively returns true when the budget is exhausted; zero selects the bounds-only query and negative one selects unlimited exact work.
    pub fn may_share_memory_with_work(&self, rhs: &Self, max_work: i64) -> Result<bool, CAbiError> {
        let mut out_out_value: bool = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_may_share_memory_with_work(self.raw.as_ptr(), rhs.raw.as_ptr(), max_work, &mut out_out_value) };
        check_status("ndarray.numpy.may_share_memory_with_work", status, 0)?;
        Ok(out_out_value)
    }

    /// Returns whether two or more logical indexes in one strided array address the same element.
    pub fn internal_overlap(&self) -> Result<bool, CAbiError> {
        let mut out_out_value: bool = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_internal_overlap(self.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.numpy.internal_overlap", status, 0)?;
        Ok(out_out_value)
    }

    /// Implements checked mesh-grid indexing, sparsity, and copy options for the typed public wrapper.
    pub fn meshgrid_native(&self, remaining: &[&Array], indexing: i64, sparse: bool, copy: bool) -> Result<Vec<Array>, CAbiError> {
        let remaining_handles = remaining.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
        let remaining_length = i64::try_from(remaining.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.meshgrid_native", status: -2 })?;
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_meshgrid(self.raw.as_ptr(), remaining_handles.as_ptr(), remaining_length, indexing, sparse, copy, &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.numpy.meshgrid_native", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.numpy.meshgrid_native", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.numpy.meshgrid_native", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Provides NumPy-compatible msort behavior.
    pub fn msort(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_msort(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.msort", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.msort", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible pad behavior.
    pub fn pad(&self, leading: &[i64], trailing: &[i64], value: f64) -> Result<Self, CAbiError> {
        let leading_length = i64::try_from(leading.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.pad", status: -2 })?;
        let trailing_length = i64::try_from(trailing.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.pad", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_pad(self.raw.as_ptr(), leading.as_ptr(), leading_length, trailing.as_ptr(), trailing_length, value, &mut raw) };
        check_status("ndarray.numpy.pad", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.pad", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible place behavior.
    pub fn place(&mut self, mask: &Self, values: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_place(self.raw.as_ptr(), mask.raw.as_ptr(), values.raw.as_ptr()) };
        check_status("ndarray.numpy.place", status, 0)?;
        Ok(())
    }

    /// Provides NumPy-compatible putmask behavior.
    pub fn putmask(&mut self, mask: &Self, values: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_putmask(self.raw.as_ptr(), mask.raw.as_ptr(), values.raw.as_ptr()) };
        check_status("ndarray.numpy.putmask", status, 0)?;
        Ok(())
    }

    /// Provides NumPy-compatible ravel multi index behavior.
    pub fn ravel_multi_index_native(&self, dimensions: &[i64], modes: &[i64], order: i64) -> Result<Self, CAbiError> {
        let dimensions_length = i64::try_from(dimensions.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.ravel_multi_index_native", status: -2 })?;
        let modes_length = i64::try_from(modes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.ravel_multi_index_native", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_ravel_multi_index_with(self.raw.as_ptr(), dimensions.as_ptr(), dimensions_length, modes.as_ptr(), modes_length, order, &mut raw) };
        check_status("ndarray.numpy.ravel_multi_index_native", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.ravel_multi_index_native", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible shares memory behavior.
    pub fn shares_memory(&self, rhs: &Self) -> Result<bool, CAbiError> {
        let mut out_out_value: bool = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_shares_memory(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.numpy.shares_memory", status, 0)?;
        Ok(out_out_value)
    }

    /// Returns exact memory overlap within the supplied work budget, failing when finite work is exhausted; negative one selects unlimited exact work.
    pub fn shares_memory_with_work(&self, rhs: &Self, max_work: i64) -> Result<bool, CAbiError> {
        let mut out_out_value: bool = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_shares_memory_with_work(self.raw.as_ptr(), rhs.raw.as_ptr(), max_work, &mut out_out_value) };
        check_status("ndarray.numpy.shares_memory_with_work", status, 0)?;
        Ok(out_out_value)
    }

    /// Provides NumPy-compatible tril indices behavior.
    pub fn tril_indices(rows: i64, offset: i64, columns: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_tril_indices(rows, offset, columns, &mut raw) };
        check_status("ndarray.numpy.tril_indices", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.tril_indices", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible tril indices from behavior.
    pub fn tril_indices_from(&self, offset: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_tril_indices_from(self.raw.as_ptr(), offset, &mut raw) };
        check_status("ndarray.numpy.tril_indices_from", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.tril_indices_from", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible triu indices behavior.
    pub fn triu_indices(rows: i64, offset: i64, columns: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_triu_indices(rows, offset, columns, &mut raw) };
        check_status("ndarray.numpy.triu_indices", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.triu_indices", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible triu indices from behavior.
    pub fn triu_indices_from(&self, offset: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_triu_indices_from(self.raw.as_ptr(), offset, &mut raw) };
        check_status("ndarray.numpy.triu_indices_from", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.triu_indices_from", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible unravel index behavior.
    pub fn unravel_index_native(&self, dimensions: &[i64], order: i64) -> Result<Vec<Array>, CAbiError> {
        let dimensions_length = i64::try_from(dimensions.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.unravel_index_native", status: -2 })?;
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_unravel_index_ordered(self.raw.as_ptr(), dimensions.as_ptr(), dimensions_length, order, &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.numpy.unravel_index_native", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.numpy.unravel_index_native", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.numpy.unravel_index_native", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Provides NumPy-compatible vsplit behavior.
    pub fn vsplit(&self, sections: i64) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_vsplit(self.raw.as_ptr(), sections, &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.numpy.vsplit", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.numpy.vsplit", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.numpy.vsplit", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Returns an average weighted by a same-shape array or one-dimensional axis weights.
    pub fn average_weighted(&self, weights: &Self, axes: &[i64], keep_dims: bool) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.average_weighted", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_weighted_average_axes(self.raw.as_ptr(), weights.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.numpy.average_weighted", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.average_weighted", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns an unweighted average followed by its broadcast reduction cardinality.
    pub fn average_with_sum(&self, axes: &[i64], keep_dims: bool) -> Result<Vec<Array>, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.average_with_sum", status: -2 })?;
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_average_with_sum(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.numpy.average_with_sum", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.numpy.average_with_sum", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.numpy.average_with_sum", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Returns a weighted average followed by its broadcast weight sum.
    pub fn average_weighted_with_sum(&self, weights: &Self, axes: &[i64], keep_dims: bool) -> Result<Vec<Array>, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.average_weighted_with_sum", status: -2 })?;
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_weighted_average_with_sum(self.raw.as_ptr(), weights.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.numpy.average_weighted_with_sum", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.numpy.average_weighted_with_sum", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.numpy.average_weighted_with_sum", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Returns covariance in the requested floating-point or complex dtype.
    pub fn covariance_typed(&self, row_variables: i64, correction: i64, dtype: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_covariance_typed(self.raw.as_ptr(), row_variables as i32, correction, dtype as i32, &mut raw) };
        check_status("ndarray.numpy.covariance_typed", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.covariance_typed", status: -1 })?;
        Ok(Self { raw })
    }

}
