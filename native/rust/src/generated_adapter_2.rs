use super::*;

impl Array {
    /// Returns a shared reshape view when possible and otherwise a contiguous materialization.
    pub fn reshape(&self, shape: &[i64]) -> Result<Self, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.reshape", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_reshape(self.raw.as_ptr(), shape.as_ptr(), shape_length, &mut raw) };
        check_status("ndarray.array.reshape", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.reshape", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reshapes using explicit C, Fortran, or input-compatible traversal order.
    pub fn reshape_ordered(&self, shape: &[i64], order: i64) -> Result<Self, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.reshape_ordered", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_reshape_ordered(self.raw.as_ptr(), shape.as_ptr(), shape_length, order as i32, &mut raw) };
        check_status("ndarray.array.reshape_ordered", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.reshape_ordered", status: -1 })?;
        Ok(Self { raw })
    }

    /// Resizes a uniquely owned contiguous array in place, preserving its flattened prefix and zero-initializing growth.
    pub fn resize_in_place(&mut self, shape: &[i64]) -> Result<(), CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.resize_in_place", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_resize_in_place(self.raw.as_ptr(), shape.as_ptr(), shape_length) };
        check_status("ndarray.array.resize_in_place", status, 0)?;
        Ok(())
    }

    /// Returns a shared strided view with two normalized axes exchanged.
    pub fn transpose(&self, axis_a: i64, axis_b: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_transpose(self.raw.as_ptr(), axis_a, axis_b, &mut raw) };
        check_status("ndarray.array.transpose", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.transpose", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns an independently owned copy preserving source-compatible memory order.
    pub fn copy(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_copy(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.copy", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.copy", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns an independently owned copy using C, Fortran, or source-compatible memory order.
    pub fn copy_ordered(&self, order: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_copy_ordered(self.raw.as_ptr(), order as i32, &mut raw) };
        check_status("ndarray.array.copy_ordered", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.copy_ordered", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a checked C-contiguous conversion to the requested dtype code.
    pub fn astype(&self, dtype: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_astype(self.raw.as_ptr(), dtype as i32, &mut raw) };
        check_status("ndarray.array.astype", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.astype", status: -1 })?;
        Ok(Self { raw })
    }

    /// Converts dtype under explicit NumPy-compatible order, copy, and casting rules.
    pub fn astype_ordered(&self, dtype: i64, order: i64, copy: bool, casting: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_astype_ordered(self.raw.as_ptr(), dtype as i32, order as i32, copy as i32, casting as i32, &mut raw) };
        check_status("ndarray.array.astype_ordered", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.astype_ordered", status: -1 })?;
        Ok(Self { raw })
    }

    /// Materializes the logical array into independently owned C-contiguous storage.
    pub fn contiguous(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_contiguous(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.contiguous", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.contiguous", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a shared view with one axis selected and removed; negative axes and indexes are normalized.
    pub fn index_axis(&self, axis: i64, index: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_index_axis(self.raw.as_ptr(), axis, index, &mut raw) };
        check_status("ndarray.array.index_axis", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.index_axis", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a shared basic-slice view; the minimum Int value denotes an omitted start or stop.
    pub fn slice_axis(&self, axis: i64, start: i64, stop: i64, step: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_slice_axis(self.raw.as_ptr(), axis, start, stop, step, &mut raw) };
        check_status("ndarray.array.slice_axis", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.slice_axis", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a shared view using a complete normalized axis permutation.
    pub fn permute(&self, axes: &[i64]) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.permute", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_permute(self.raw.as_ptr(), axes.as_ptr(), axes_length, &mut raw) };
        check_status("ndarray.array.permute", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.permute", status: -1 })?;
        Ok(Self { raw })
    }

    /// Copies normalized integer selections along one axis, preserving duplicate indexes and order.
    pub fn take(&self, indices: &[i64], axis: i64) -> Result<Self, CAbiError> {
        let indices_length = i64::try_from(indices.len()).map_err(|_| CAbiError { operation: "ndarray.array.take", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_take(self.raw.as_ptr(), indices.as_ptr(), indices_length, axis, &mut raw) };
        check_status("ndarray.array.take", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.take", status: -1 })?;
        Ok(Self { raw })
    }

    /// Copies integer selections using raise, wrap, or clip index normalization.
    pub fn take_mode(&self, indices: &[i64], axis: i64, mode: i64) -> Result<Self, CAbiError> {
        let indices_length = i64::try_from(indices.len()).map_err(|_| CAbiError { operation: "ndarray.array.take_mode", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_take_mode(self.raw.as_ptr(), indices.as_ptr(), indices_length, axis, mode as i32, &mut raw) };
        check_status("ndarray.array.take_mode", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.take_mode", status: -1 })?;
        Ok(Self { raw })
    }

    /// Takes one Int64 index at every output coordinate along a normalized axis.
    pub fn take_along_axis(&self, indices: &Self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_take_along_axis(self.raw.as_ptr(), indices.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.array.take_along_axis", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.take_along_axis", status: -1 })?;
        Ok(Self { raw })
    }

    /// Assigns scalar or matching values at normalized flattened indexes in order.
    pub fn put(&mut self, indices: &[i64], values: &Self) -> Result<(), CAbiError> {
        let indices_length = i64::try_from(indices.len()).map_err(|_| CAbiError { operation: "ndarray.array.put", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_put(self.raw.as_ptr(), indices.as_ptr(), indices_length, values.raw.as_ptr()) };
        check_status("ndarray.array.put", status, 0)?;
        Ok(())
    }

    /// Assigns scalar or shape-matched values using one Int64 index at each coordinate.
    pub fn put_along_axis(&mut self, indices: &Self, values: &Self, axis: i64) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_put_along_axis(self.raw.as_ptr(), indices.raw.as_ptr(), values.raw.as_ptr(), axis) };
        check_status("ndarray.array.put_along_axis", status, 0)?;
        Ok(())
    }

    /// Selects axis entries using a rank-one Bool mask, truncating the axis to the mask length.
    pub fn compress(&self, mask: &Self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_compress(self.raw.as_ptr(), mask.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.array.compress", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.compress", status: -1 })?;
        Ok(Self { raw })
    }

    /// Selects flattened logical values using a rank-one Bool mask, truncating to the mask length.
    pub fn compress_flat(&self, mask: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_compress_flat(self.raw.as_ptr(), mask.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.compress_flat", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.compress_flat", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns flattened values selected by an exact-shape Bool mask.
    pub fn mask_select(&self, mask: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_mask_select(self.raw.as_ptr(), mask.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.mask_select", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.mask_select", status: -1 })?;
        Ok(Self { raw })
    }

    /// Assigns scalar or matching values where an exact-shape Bool mask is true.
    pub fn mask_assign(&mut self, mask: &Self, values: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_mask_assign(self.raw.as_ptr(), mask.raw.as_ptr(), values.raw.as_ptr()) };
        check_status("ndarray.array.mask_assign", status, 0)?;
        Ok(())
    }

    /// Gathers flattened values from an Int64 coordinate matrix shaped [count, rank].
    pub fn gather_coordinates(&self, coordinates: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_gather_coordinates(self.raw.as_ptr(), coordinates.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.gather_coordinates", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.gather_coordinates", status: -1 })?;
        Ok(Self { raw })
    }

    /// Atomically assigns scalar or matching values through an Int64 coordinate matrix shaped [count, rank].
    pub fn put_coordinates(&mut self, coordinates: &Self, values: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_put_coordinates(self.raw.as_ptr(), coordinates.raw.as_ptr(), values.raw.as_ptr()) };
        check_status("ndarray.array.put_coordinates", status, 0)?;
        Ok(())
    }

    /// Returns the truth value of an array containing exactly one element.
    pub fn item_bool(&self) -> Result<bool, CAbiError> {
        let mut out_out_value: bool = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_item_bool(self.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.array.item_bool", status, 0)?;
        Ok(out_out_value)
    }

}
