use super::*;

impl Array {
    /// Returns an owned array cyclically shifted after flattening in logical row-major order.
    pub fn roll_flat(&self, shift: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_roll_flat(self.raw.as_ptr(), shift, &mut raw) };
        check_status("ndarray.array.roll_flat", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.roll_flat", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns an owned array shifted by the same amount along every selected axis.
    pub fn roll_axes_by(&self, shift: i64, axes: &[i64]) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.roll_axes_by", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_roll_axes_by(self.raw.as_ptr(), shift, axes.as_ptr(), axes_length, &mut raw) };
        check_status("ndarray.array.roll_axes_by", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.roll_axes_by", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns an owned array shifted along selected axes using one broadcast shift or one shift per axis.
    pub fn roll_axes(&self, shifts: &[i64], axes: &[i64]) -> Result<Self, CAbiError> {
        let shifts_length = i64::try_from(shifts.len()).map_err(|_| CAbiError { operation: "ndarray.array.roll_axes", status: -2 })?;
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.roll_axes", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_roll_axes(self.raw.as_ptr(), shifts.as_ptr(), shifts_length, axes.as_ptr(), axes_length, &mut raw) };
        check_status("ndarray.array.roll_axes", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.roll_axes", status: -1 })?;
        Ok(Self { raw })
    }

    /// Repeats each element a non-negative number of times along one axis.
    pub fn repeat(&self, repeats: i64, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_repeat(self.raw.as_ptr(), repeats, axis, &mut raw) };
        check_status("ndarray.array.repeat", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.repeat", status: -1 })?;
        Ok(Self { raw })
    }

    /// Repeats axis entries using one broadcast count or one count per source entry.
    pub fn repeat_counts(&self, repeats: &[i64], axis: i64) -> Result<Self, CAbiError> {
        let repeats_length = i64::try_from(repeats.len()).map_err(|_| CAbiError { operation: "ndarray.array.repeat_counts", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_repeat_counts(self.raw.as_ptr(), repeats.as_ptr(), repeats_length, axis, &mut raw) };
        check_status("ndarray.array.repeat_counts", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.repeat_counts", status: -1 })?;
        Ok(Self { raw })
    }

    /// Flattens in logical C order and repeats entries using broadcast or per-entry counts.
    pub fn repeat_flat(&self, repeats: &[i64]) -> Result<Self, CAbiError> {
        let repeats_length = i64::try_from(repeats.len()).map_err(|_| CAbiError { operation: "ndarray.array.repeat_flat", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_repeat_flat(self.raw.as_ptr(), repeats.as_ptr(), repeats_length, &mut raw) };
        check_status("ndarray.array.repeat_flat", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.repeat_flat", status: -1 })?;
        Ok(Self { raw })
    }

    /// Tiles an array according to trailing-axis repetition counts.
    pub fn tile(&self, repetitions: &[i64]) -> Result<Self, CAbiError> {
        let repetitions_length = i64::try_from(repetitions.len()).map_err(|_| CAbiError { operation: "ndarray.array.tile", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_tile(self.raw.as_ptr(), repetitions.as_ptr(), repetitions_length, &mut raw) };
        check_status("ndarray.array.tile", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.tile", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns an axis-wise stable ascending copy using deterministic NaN and complex ordering.
    pub fn sort(&self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_sort(self.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.array.sort", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.sort", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns stable Int64 indices that order each axis slice ascending.
    pub fn argsort(&self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_argsort(self.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.array.argsort", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.argsort", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns an ascending axis-wise copy without preserving equal-value order.
    pub fn sort_unstable(&self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_sort_unstable(self.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.array.sort_unstable", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.sort_unstable", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns ascending Int64 indexes without preserving equal-value order.
    pub fn argsort_unstable(&self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_argsort_unstable(self.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.array.argsort_unstable", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.argsort_unstable", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a stable ascending one-dimensional copy.
    pub fn sort_flat(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_sort_flat(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.sort_flat", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.sort_flat", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns stable Int64 indices ordering the flattened array.
    pub fn argsort_flat(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_argsort_flat(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.argsort_flat", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.argsort_flat", status: -1 })?;
        Ok(Self { raw })
    }

    /// Sorts a writable array in place along one axis.
    pub fn sort_inplace(&mut self, axis: i64) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_sort_inplace(self.raw.as_ptr(), axis) };
        check_status("ndarray.array.sort_inplace", status, 0)?;
        Ok(())
    }

    /// Returns an axis-wise copy satisfying the selected kth partition postcondition.
    pub fn partition(&self, kth: i64, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_partition(self.raw.as_ptr(), kth, axis, &mut raw) };
        check_status("ndarray.array.partition", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.partition", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns Int64 indexes satisfying the selected kth partition postcondition.
    pub fn argpartition(&self, kth: i64, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_argpartition(self.raw.as_ptr(), kth, axis, &mut raw) };
        check_status("ndarray.array.argpartition", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.argpartition", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns an axis-wise partition for every integer position in kths without mutating either input.
    pub fn partition_kths(&self, kths: &Self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_partition_kths(self.raw.as_ptr(), kths.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.array.partition_kths", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.partition_kths", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns Int64 indexes satisfying every integer position in kths.
    pub fn argpartition_kths(&self, kths: &Self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_argpartition_kths(self.raw.as_ptr(), kths.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.array.argpartition_kths", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.argpartition_kths", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a flattened partitioned copy for one selected position.
    pub fn partition_flat(&self, kth: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_partition_flat(self.raw.as_ptr(), kth, &mut raw) };
        check_status("ndarray.array.partition_flat", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.partition_flat", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns flattened Int64 indexes satisfying one selected partition position.
    pub fn argpartition_flat(&self, kth: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_argpartition_flat(self.raw.as_ptr(), kth, &mut raw) };
        check_status("ndarray.array.argpartition_flat", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.argpartition_flat", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a flattened partitioned copy for every integer position in kths.
    pub fn partition_kths_flat(&self, kths: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_partition_kths_flat(self.raw.as_ptr(), kths.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.partition_kths_flat", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.partition_kths_flat", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns flattened Int64 indexes satisfying every integer position in kths.
    pub fn argpartition_kths_flat(&self, kths: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_argpartition_kths_flat(self.raw.as_ptr(), kths.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.argpartition_kths_flat", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.argpartition_kths_flat", status: -1 })?;
        Ok(Self { raw })
    }

    /// Partitions a writable array in place along one axis for one selected position.
    pub fn partition_inplace(&mut self, kth: i64, axis: i64) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_partition_inplace(self.raw.as_ptr(), kth, axis) };
        check_status("ndarray.array.partition_inplace", status, 0)?;
        Ok(())
    }

    /// Partitions a writable array in place along one axis for every integer position in kths.
    pub fn partition_kths_inplace(&mut self, kths: &Self, axis: i64) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_partition_kths_inplace(self.raw.as_ptr(), kths.raw.as_ptr(), axis) };
        check_status("ndarray.array.partition_kths_inplace", status, 0)?;
        Ok(())
    }

}
