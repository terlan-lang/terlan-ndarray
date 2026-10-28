use super::*;

impl Array {
    /// Computes numerical matrix rank; zero tolerance selects the deterministic default.
    pub fn matrix_rank(&self, tolerance: f64) -> Result<i64, CAbiError> {
        let mut out_out_rank: i64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_matrix_rank(self.raw.as_ptr(), tolerance, &mut out_out_rank) };
        check_status("ndarray.array.matrix_rank", status, 0)?;
        Ok(out_out_rank)
    }

    /// Encodes an array as deterministic NumPy .npy bytes.
    pub fn to_npy(&self) -> Result<Vec<u8>, CAbiError> {
        let mut out_out_bytes: *mut u8 = std::ptr::null_mut();
        let mut out_out_bytes_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_to_npy(self.raw.as_ptr(), &mut out_out_bytes, &mut out_out_bytes_length) };
        check_status("ndarray.array.to_npy", status, 0)?;
        let values = if out_out_bytes_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_bytes).map(|pointer| {
                // SAFETY: the producer returned `out_out_bytes_length` initialized elements; they are copied before destruction.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_bytes_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.to_npy", status: -1 })
        };
        // SAFETY: ownership metadata transfers this array to the named destructor exactly once.
        unsafe { ffi::terlan_ndarray_v1_delete_tensor_packet(out_out_bytes) };
        Ok(values?)
    }

    /// Decodes bounded NumPy .npy bytes into an owned CPU array.
    pub fn from_npy(bytes: &[u8], maximum: i64) -> Result<Self, CAbiError> {
        let bytes_length = i64::try_from(bytes.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_npy", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_from_npy(bytes.as_ptr(), bytes_length, maximum, &mut raw) };
        check_status("ndarray.array.from_npy", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.from_npy", status: -1 })?;
        Ok(Self { raw })
    }

    /// Saves an array to an exact filesystem path in deterministic NumPy .npy format.
    pub fn save_npy(&self, path: &str) -> Result<(), CAbiError> {
        let path_c = std::ffi::CString::new(path.as_bytes()).map_err(|_| CAbiError { operation: "ndarray.array.save_npy", status: -3 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_save_npy(self.raw.as_ptr(), path_c.as_ptr()) };
        check_status("ndarray.array.save_npy", status, 0)?;
        Ok(())
    }

    /// Loads a bounded NumPy .npy file from an exact filesystem path.
    pub fn load_npy(path: &str, maximum: i64) -> Result<Self, CAbiError> {
        let path_c = std::ffi::CString::new(path.as_bytes()).map_err(|_| CAbiError { operation: "ndarray.array.load_npy", status: -3 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_load_npy(path_c.as_ptr(), maximum, &mut raw) };
        check_status("ndarray.array.load_npy", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.load_npy", status: -1 })?;
        Ok(Self { raw })
    }

    /// Encodes an array as a versioned checksummed package-native packet.
    pub fn serialize(&self) -> Result<Vec<u8>, CAbiError> {
        let mut out_out_bytes: *mut u8 = std::ptr::null_mut();
        let mut out_out_bytes_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_serialize(self.raw.as_ptr(), &mut out_out_bytes, &mut out_out_bytes_length) };
        check_status("ndarray.array.serialize", status, 0)?;
        let values = if out_out_bytes_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_bytes).map(|pointer| {
                // SAFETY: the producer returned `out_out_bytes_length` initialized elements; they are copied before destruction.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_bytes_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.serialize", status: -1 })
        };
        // SAFETY: ownership metadata transfers this array to the named destructor exactly once.
        unsafe { ffi::terlan_ndarray_v1_delete_tensor_packet(out_out_bytes) };
        Ok(values?)
    }

    /// Decodes a bounded checksummed package-native packet into an owned CPU array.
    pub fn deserialize(bytes: &[u8], maximum: i64) -> Result<Self, CAbiError> {
        let bytes_length = i64::try_from(bytes.len()).map_err(|_| CAbiError { operation: "ndarray.array.deserialize", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_deserialize(bytes.as_ptr(), bytes_length, maximum, &mut raw) };
        check_status("ndarray.array.deserialize", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.deserialize", status: -1 })?;
        Ok(Self { raw })
    }

    /// Opens a bounded NumPy file as a read-only or read-write memory-backed array.
    pub fn mmap_npy(path: &str, writable: bool, maximum: i64) -> Result<Self, CAbiError> {
        let path_c = std::ffi::CString::new(path.as_bytes()).map_err(|_| CAbiError { operation: "ndarray.array.mmap_npy", status: -3 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_mmap_npy(path_c.as_ptr(), writable as i32, maximum, &mut raw) };
        check_status("ndarray.array.mmap_npy", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.mmap_npy", status: -1 })?;
        Ok(Self { raw })
    }

    /// Flushes writable memory-backed storage and is a no-op for ordinary or read-only arrays.
    pub fn flush(&self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_flush(self.raw.as_ptr()) };
        check_status("ndarray.array.flush", status, 0)?;
        Ok(())
    }

    /// Exports a compatible primitive array through official Arrow C Data ownership callbacks.
    pub fn to_arrow(&self) -> Result<ArrowArray, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArrowArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_to_arrow(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.to_arrow", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.to_arrow", status: -1 })?;
        Ok(ArrowArray { raw })
    }

    /// Imports one package-owned non-null Arrow C Data export as an owned array.
    pub fn from_arrow(arrow: &ArrowArray) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_from_arrow(arrow.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.from_arrow", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.from_arrow", status: -1 })?;
        Ok(Self { raw })
    }

    /// Splits an axis into equally sized shared views.
    pub fn split(&self, sections: i64, axis: i64) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_split(self.raw.as_ptr(), sections, axis, &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.split", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.split", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.array.split", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Splits an axis into uneven shared views with larger leading sections.
    pub fn array_split(&self, sections: i64, axis: i64) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_array_split(self.raw.as_ptr(), sections, axis, &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.array_split", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.array_split", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.array.array_split", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Splits an axis before each normalized index and preserves empty sections.
    pub fn split_at(&self, indices: &[i64], axis: i64) -> Result<Vec<Array>, CAbiError> {
        let indices_length = i64::try_from(indices.len()).map_err(|_| CAbiError { operation: "ndarray.array.split_at", status: -2 })?;
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_split_at(self.raw.as_ptr(), indices.as_ptr(), indices_length, axis, &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.split_at", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.split_at", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.array.split_at", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Counts truthy values across the flattened array.
    pub fn count_nonzero(&self) -> Result<i64, CAbiError> {
        let mut out_out_count: i64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_count_nonzero(self.raw.as_ptr(), &mut out_out_count) };
        check_status("ndarray.array.count_nonzero", status, 0)?;
        Ok(out_out_count)
    }

    /// Counts nonzero elements over normalized axes with optional retained dimensions.
    pub fn count_nonzero_axes(&self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.count_nonzero_axes", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_count_nonzero_axes(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.count_nonzero_axes", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.count_nonzero_axes", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns flattened indexes of truthy values as Int64.
    pub fn flatnonzero(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_flatnonzero(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.flatnonzero", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.flatnonzero", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns one coordinate row for every truthy value.
    pub fn argwhere(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_argwhere(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.argwhere", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.argwhere", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns one Int64 coordinate array per source axis.
    pub fn nonzero(&self) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nonzero(self.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.nonzero", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.nonzero", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.array.nonzero", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Returns a shared view with one singleton axis inserted.
    pub fn expand_dims(&self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_expand_dims(self.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.array.expand_dims", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.expand_dims", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a shared view with singleton axes inserted at final output positions.
    pub fn expand_dims_axes(&self, axes: &[i64]) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.expand_dims_axes", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_expand_dims_axes(self.raw.as_ptr(), axes.as_ptr(), axes_length, &mut raw) };
        check_status("ndarray.array.expand_dims_axes", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.expand_dims_axes", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a shared view with selected singleton axes removed; an empty list removes all singleton axes.
    pub fn squeeze(&self, axes: &[i64]) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.squeeze", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_squeeze(self.raw.as_ptr(), axes.as_ptr(), axes_length, &mut raw) };
        check_status("ndarray.array.squeeze", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.squeeze", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a shared negative-stride view with selected axes reversed.
    pub fn flip(&self, axes: &[i64]) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.flip", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_flip(self.raw.as_ptr(), axes.as_ptr(), axes_length, &mut raw) };
        check_status("ndarray.array.flip", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.flip", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns an owned array cyclically shifted along one axis.
    pub fn roll(&self, shift: i64, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_roll(self.raw.as_ptr(), shift, axis, &mut raw) };
        check_status("ndarray.array.roll", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.roll", status: -1 })?;
        Ok(Self { raw })
    }

}
