use super::*;

impl Array {
    /// Writes NaN-omitting minima into an exact-shape destination.
    pub fn nanmin_into(&mut self, array: &Self, axes: &[i64], keep_dims: bool) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanmin_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_minimum_into(self.raw.as_ptr(), array.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32) };
        check_status("ndarray.numpy.nanmin_into", status, 0)?;
        Ok(())
    }

    /// Writes NaN-omitting maxima into an exact-shape destination.
    pub fn nanmax_into(&mut self, array: &Self, axes: &[i64], keep_dims: bool) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanmax_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_maximum_into(self.raw.as_ptr(), array.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32) };
        check_status("ndarray.numpy.nanmax_into", status, 0)?;
        Ok(())
    }

    /// Writes NaN-omitting variances into an exact-shape inexact destination.
    pub fn nanvar_into(&mut self, array: &Self, axes: &[i64], keep_dims: bool, correction: i64) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanvar_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_variance_into(self.raw.as_ptr(), array.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, correction) };
        check_status("ndarray.numpy.nanvar_into", status, 0)?;
        Ok(())
    }

    /// Writes NaN-omitting standard deviations into an exact-shape inexact destination.
    pub fn nanstd_into(&mut self, array: &Self, axes: &[i64], keep_dims: bool, correction: i64) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanstd_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_standard_deviation_into(self.raw.as_ptr(), array.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, correction) };
        check_status("ndarray.numpy.nanstd_into", status, 0)?;
        Ok(())
    }

    /// Returns a flattened cumulative sum with NaNs replaced by zero.
    pub fn nancumsum_flat(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nancumsum_flat(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.nancumsum_flat", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nancumsum_flat", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a flattened cumulative product with NaNs replaced by one.
    pub fn nancumprod_flat(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nancumprod_flat(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.nancumprod_flat", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nancumprod_flat", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a NaN-normalized cumulative sum in an explicit dtype.
    pub fn nancumsum_as(&self, axis: i64, requested_dtype: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nancumsum_as(self.raw.as_ptr(), axis, requested_dtype as i32, &mut raw) };
        check_status("ndarray.numpy.nancumsum_as", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nancumsum_as", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a NaN-normalized cumulative product in an explicit dtype.
    pub fn nancumprod_as(&self, axis: i64, requested_dtype: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nancumprod_as(self.raw.as_ptr(), axis, requested_dtype as i32, &mut raw) };
        check_status("ndarray.numpy.nancumprod_as", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nancumprod_as", status: -1 })?;
        Ok(Self { raw })
    }

    /// Writes a NaN-normalized cumulative sum into an exact-shape destination.
    pub fn nancumsum_into(&mut self, array: &Self, axis: i64) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nancumsum_into(self.raw.as_ptr(), array.raw.as_ptr(), axis) };
        check_status("ndarray.numpy.nancumsum_into", status, 0)?;
        Ok(())
    }

    /// Writes a NaN-normalized cumulative product into an exact-shape destination.
    pub fn nancumprod_into(&mut self, array: &Self, axis: i64) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nancumprod_into(self.raw.as_ptr(), array.raw.as_ptr(), axis) };
        check_status("ndarray.numpy.nancumprod_into", status, 0)?;
        Ok(())
    }

    /// Writes a flattened NaN-normalized cumulative sum into a vector destination.
    pub fn nancumsum_flat_into(&mut self, array: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nancumsum_flat_into(self.raw.as_ptr(), array.raw.as_ptr()) };
        check_status("ndarray.numpy.nancumsum_flat_into", status, 0)?;
        Ok(())
    }

    /// Writes a flattened NaN-normalized cumulative product into a vector destination.
    pub fn nancumprod_flat_into(&mut self, array: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nancumprod_flat_into(self.raw.as_ptr(), array.raw.as_ptr()) };
        check_status("ndarray.numpy.nancumprod_flat_into", status, 0)?;
        Ok(())
    }

    /// Replaces NaN and infinity components with dtype-specific finite defaults.
    pub fn nan_to_num_default(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_to_num_default(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.nan_to_num_default", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nan_to_num_default", status: -1 })?;
        Ok(Self { raw })
    }

    /// Writes non-finite replacements into an exact-shape, exact-dtype destination.
    pub fn nan_to_num_into(&mut self, array: &Self, nan: f64, positive_infinity: f64, negative_infinity: f64) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_to_num_into(self.raw.as_ptr(), array.raw.as_ptr(), nan, positive_infinity, negative_infinity) };
        check_status("ndarray.numpy.nan_to_num_into", status, 0)?;
        Ok(())
    }

    /// Writes a broadcast addition into an exact-shape destination after staging all aliased inputs.
    pub fn add_into(&mut self, lhs: &Self, rhs: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_add_into(self.raw.as_ptr(), lhs.raw.as_ptr(), rhs.raw.as_ptr()) };
        check_status("ndarray.numpy.add_into", status, 0)?;
        Ok(())
    }

    /// Writes broadcast sums where a Bool mask is true while preserving masked destination elements and staging aliases.
    pub fn add_where_into(&mut self, lhs: &Self, rhs: &Self, mask: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_add_where_into(self.raw.as_ptr(), lhs.raw.as_ptr(), rhs.raw.as_ptr(), mask.raw.as_ptr()) };
        check_status("ndarray.numpy.add_where_into", status, 0)?;
        Ok(())
    }

    /// Writes magnitudes into an exact-shape destination after staging aliased input.
    pub fn absolute_into(&mut self, array: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_absolute_into(self.raw.as_ptr(), array.raw.as_ptr()) };
        check_status("ndarray.numpy.absolute_into", status, 0)?;
        Ok(())
    }

    /// Writes truth-value negation into an exact-shape Bool destination.
    pub fn logical_not_into(&mut self, array: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_logical_not_into(self.raw.as_ptr(), array.raw.as_ptr()) };
        check_status("ndarray.numpy.logical_not_into", status, 0)?;
        Ok(())
    }

    /// Writes broadcast truth-value conjunction into an exact-shape Bool destination.
    pub fn logical_and_into(&mut self, lhs: &Self, rhs: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_logical_and_into(self.raw.as_ptr(), lhs.raw.as_ptr(), rhs.raw.as_ptr()) };
        check_status("ndarray.numpy.logical_and_into", status, 0)?;
        Ok(())
    }

    /// Writes broadcast truth-value disjunction into an exact-shape Bool destination.
    pub fn logical_or_into(&mut self, lhs: &Self, rhs: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_logical_or_into(self.raw.as_ptr(), lhs.raw.as_ptr(), rhs.raw.as_ptr()) };
        check_status("ndarray.numpy.logical_or_into", status, 0)?;
        Ok(())
    }

    /// Writes broadcast truth-value exclusive disjunction into an exact-shape Bool destination.
    pub fn logical_xor_into(&mut self, lhs: &Self, rhs: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_logical_xor_into(self.raw.as_ptr(), lhs.raw.as_ptr(), rhs.raw.as_ptr()) };
        check_status("ndarray.numpy.logical_xor_into", status, 0)?;
        Ok(())
    }

    /// Writes bitwise inversion into an exact-shape destination after staging aliased input.
    pub fn invert_into(&mut self, array: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_invert_into(self.raw.as_ptr(), array.raw.as_ptr()) };
        check_status("ndarray.numpy.invert_into", status, 0)?;
        Ok(())
    }

    /// Writes bitwise inversion where a Bool mask is true and stages masks that alias the destination.
    pub fn invert_where_into(&mut self, array: &Self, mask: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_invert_where_into(self.raw.as_ptr(), array.raw.as_ptr(), mask.raw.as_ptr()) };
        check_status("ndarray.numpy.invert_where_into", status, 0)?;
        Ok(())
    }

    /// Writes reciprocals into an exact-shape destination after staging aliased input.
    pub fn reciprocal_into(&mut self, array: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_reciprocal_into(self.raw.as_ptr(), array.raw.as_ptr()) };
        check_status("ndarray.numpy.reciprocal_into", status, 0)?;
        Ok(())
    }

}
