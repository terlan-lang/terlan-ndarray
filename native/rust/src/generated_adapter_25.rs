use super::*;

impl Array {
    /// Writes reciprocals where a Bool mask is true while preserving masked destination elements.
    pub fn reciprocal_where_into(&mut self, array: &Self, mask: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_reciprocal_where_into(self.raw.as_ptr(), array.raw.as_ptr(), mask.raw.as_ptr()) };
        check_status("ndarray.numpy.reciprocal_where_into", status, 0)?;
        Ok(())
    }

    /// Applies unbuffered bitwise inversion at rank-one integer indexes after stabilizing aliased indexes.
    pub fn invert_at(&mut self, indexes: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_invert_at(self.raw.as_ptr(), indexes.raw.as_ptr()) };
        check_status("ndarray.numpy.invert_at", status, 0)?;
        Ok(())
    }

    /// Applies sequential unbuffered addition at rank-one integer indexes with broadcast values and stable aliases.
    pub fn add_at(&mut self, indexes: &Self, values: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_add_at(self.raw.as_ptr(), indexes.raw.as_ptr(), values.raw.as_ptr()) };
        check_status("ndarray.numpy.add_at", status, 0)?;
        Ok(())
    }

    /// Returns parity reductions of truthy values over selected axes.
    pub fn logical_xor_reduce(&self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.logical_xor_reduce", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_logical_xor_reduce(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.numpy.logical_xor_reduce", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.logical_xor_reduce", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns segmented addition reductions beginning at each rank-one integer index along an axis.
    pub fn add_reduceat(&self, indexes: &Self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_add_reduceat(self.raw.as_ptr(), indexes.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.numpy.add_reduceat", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.add_reduceat", status: -1 })?;
        Ok(Self { raw })
    }

    /// Writes segmented addition reductions into an exact-shape destination after staging aliased inputs.
    pub fn add_reduceat_into(&mut self, array: &Self, indexes: &Self, axis: i64) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_add_reduceat_into(self.raw.as_ptr(), array.raw.as_ptr(), indexes.raw.as_ptr(), axis) };
        check_status("ndarray.numpy.add_reduceat_into", status, 0)?;
        Ok(())
    }

}
