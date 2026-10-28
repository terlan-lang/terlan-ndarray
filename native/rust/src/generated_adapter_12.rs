use super::*;

impl Array {
    /// Transforms an explicitly sized real spectrum to a compact Hermitian signal.
    pub fn ihfft_sized(&self, length: i64, axis: i64, normalization: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_ihfft_sized(self.raw.as_ptr(), length, axis, normalization as i32, &mut raw) };
        check_status("ndarray.array.ihfft_sized", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.ihfft_sized", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns signed discrete frequency bins.
    pub fn fftfreq(length: i64, spacing: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_fftfreq(length, spacing, &mut raw) };
        check_status("ndarray.array.fftfreq", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.fftfreq", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns non-negative real-transform frequency bins.
    pub fn rfftfreq(length: i64, spacing: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_rfftfreq(length, spacing, &mut raw) };
        check_status("ndarray.array.rfftfreq", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.rfftfreq", status: -1 })?;
        Ok(Self { raw })
    }

    /// Moves zero-frequency bins to centered positions on selected axes.
    pub fn fftshift(&self, axes: &[i64]) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.fftshift", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_fftshift(self.raw.as_ptr(), axes.as_ptr(), axes_length, &mut raw) };
        check_status("ndarray.array.fftshift", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.fftshift", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reverses centered frequency-bin shifts on selected axes.
    pub fn ifftshift(&self, axes: &[i64]) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.ifftshift", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_ifftshift(self.raw.as_ptr(), axes.as_ptr(), axes_length, &mut raw) };
        check_status("ndarray.array.ifftshift", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.ifftshift", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns stable left or right insertion indexes into one sorted rank-one array.
    pub fn searchsorted(&self, values: &Self, right: bool) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_searchsorted(self.raw.as_ptr(), values.raw.as_ptr(), right as i32, &mut raw) };
        check_status("ndarray.array.searchsorted", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.searchsorted", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns insertion indexes using a validated Int64 permutation of an unsorted rank-one array.
    pub fn searchsorted_with_sorter(&self, values: &Self, sorter: &Self, right: bool) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_searchsorted_with_sorter(self.raw.as_ptr(), values.raw.as_ptr(), sorter.raw.as_ptr(), right as i32, &mut raw) };
        check_status("ndarray.array.searchsorted_with_sorter", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.searchsorted_with_sorter", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns stable lexicographic indexes for a [key_count, item_count] matrix using the final key as primary.
    pub fn lexsort_keys(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_lexsort_keys(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.lexsort_keys", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.lexsort_keys", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns sorted unique values, first indexes, inverse indexes, and occurrence counts.
    pub fn unique_all(&self) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_unique_all(self.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.unique_all", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.unique_all", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.array.unique_all", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Returns the sorted flattened union of two equal-dtype arrays.
    pub fn union(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_union(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.union", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.union", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the sorted flattened intersection of two equal-dtype arrays.
    pub fn intersect(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_intersect(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.intersect", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.intersect", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns sorted flattened values present only in the first array.
    pub fn set_difference(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_set_difference(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.set_difference", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.set_difference", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns sorted flattened values present in exactly one input.
    pub fn set_xor(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_set_xor(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.set_xor", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.set_xor", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a same-shape Bool mask indicating membership in the flattened test array.
    pub fn is_in(&self, test_elements: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_is_in(self.raw.as_ptr(), test_elements.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.is_in", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.is_in", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a flattened Bool membership mask with optional inversion.
    pub fn in1d(&self, test_elements: &Self, invert: bool) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_in1d(self.raw.as_ptr(), test_elements.raw.as_ptr(), invert as i32, &mut raw) };
        check_status("ndarray.array.in1d", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.in1d", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns first-array values absent from the second while preserving order and duplicates.
    pub fn set_difference_unique(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_set_difference_unique(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.set_difference_unique", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.set_difference_unique", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns sorted intersections and their first flattened indexes in both inputs.
    pub fn intersect_all(&self, rhs: &Self) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_intersect_all(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.intersect_all", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.intersect_all", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.array.intersect_all", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Returns lexicographically unique axis slices, first indexes, inverse indexes, and counts.
    pub fn unique_axis_all(&self, axis: i64) -> Result<Vec<Array>, CAbiError> {
        let mut out_out_values: *mut u64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_unique_axis_all(self.raw.as_ptr(), axis, &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.unique_axis_all", status, 0)?;
        let raw_values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized transferred handle identities.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.unique_axis_all", status: -1 })
        };
        // SAFETY: only the temporary identity array is released; its resource elements were transferred.
        unsafe { ffi::terlan_ndarray_v1_delete_handle_values(out_out_values) };
        let raw_values = raw_values?;
        if raw_values.iter().any(|value| *value == 0) {
            for value in raw_values.into_iter().filter(|value| *value != 0) {
                // SAFETY: nonzero identities still belong to this failed transfer.
                let _ = unsafe { ffi::terlan_ndarray_v1_delete(value as _) };
            }
            return Err(CAbiError { operation: "ndarray.array.unique_axis_all", status: -1 });
        }
        Ok(raw_values.into_iter().map(|value| Array {
            // SAFETY: zero identities were rejected above.
            raw: unsafe { NonNull::new_unchecked(value as _) },
        }).collect())
    }

    /// Returns flattened consecutive differences with typed values prepended and appended.
    pub fn ediff1d_edges(&self, begin: &Self, end: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_ediff1d_edges(self.raw.as_ptr(), begin.raw.as_ptr(), end.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.ediff1d_edges", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.ediff1d_edges", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns inclusive sums along one normalized axis using reduction accumulator promotion.
    pub fn cumulative_sum(&self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_cumulative_sum(self.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.array.cumulative_sum", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.cumulative_sum", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns inclusive products along one normalized axis using reduction accumulator promotion.
    pub fn cumulative_product(&self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_cumulative_product(self.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.array.cumulative_product", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.cumulative_product", status: -1 })?;
        Ok(Self { raw })
    }

    /// Exports a pointer-free tensor packet through a package-native DLPack lifecycle.
    pub fn export_tensor_packet(&self, consumer: &str) -> Result<Vec<u8>, CAbiError> {
        let consumer_c = std::ffi::CString::new(consumer.as_bytes()).map_err(|_| CAbiError { operation: "ndarray.array.export_tensor_packet", status: -3 })?;
        let mut out_out_packet: *mut u8 = std::ptr::null_mut();
        let mut out_out_packet_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_export_tensor_packet(self.raw.as_ptr(), consumer_c.as_ptr(), &mut out_out_packet, &mut out_out_packet_length) };
        check_status("ndarray.array.export_tensor_packet", status, 0)?;
        let values = if out_out_packet_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_packet).map(|pointer| {
                // SAFETY: the producer returned `out_out_packet_length` initialized elements; they are copied before destruction.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_packet_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.export_tensor_packet", status: -1 })
        };
        // SAFETY: ownership metadata transfers this array to the named destructor exactly once.
        unsafe { ffi::terlan_ndarray_v1_delete_tensor_packet(out_out_packet) };
        Ok(values?)
    }

    /// Imports a validated pointer-free tensor packet as an owned CPU array.
    pub fn import_tensor_packet(packet: &[u8]) -> Result<Self, CAbiError> {
        let packet_length = i64::try_from(packet.len()).map_err(|_| CAbiError { operation: "ndarray.array.import_tensor_packet", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_import_tensor_packet(packet.as_ptr(), packet_length, &mut raw) };
        check_status("ndarray.array.import_tensor_packet", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.import_tensor_packet", status: -1 })?;
        Ok(Self { raw })
    }

    /// Selects broadcast choice values with raise, wrap, or clip index handling.
    pub fn choose(&self, choices: &[&Array], mode: i64) -> Result<Self, CAbiError> {
        let choices_handles = choices.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
        let choices_length = i64::try_from(choices.len()).map_err(|_| CAbiError { operation: "ndarray.array.choose", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_choose(self.raw.as_ptr(), choices_handles.as_ptr(), choices_length, mode as i32, &mut raw) };
        check_status("ndarray.array.choose", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.choose", status: -1 })?;
        Ok(Self { raw })
    }

}
