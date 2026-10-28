use super::*;

impl Array {
    /// Returns a read-only zero-stride view broadcast to a compatible shape.
    pub fn broadcast_to(&self, shape: &[i64]) -> Result<Self, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.broadcast_to", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_broadcast_to(self.raw.as_ptr(), shape.as_ptr(), shape_length, &mut raw) };
        check_status("ndarray.array.broadcast_to", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.broadcast_to", status: -1 })?;
        Ok(Self { raw })
    }

    /// Divides broadcast-compatible arrays using NumPy dtype promotion.
    pub fn true_divide(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_true_divide(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.true_divide", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.true_divide", status: -1 })?;
        Ok(Self { raw })
    }

    /// Floor-divides broadcast-compatible real arrays.
    pub fn floor_divide(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_floor_divide(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.floor_divide", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.floor_divide", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the NumPy-sign remainder of broadcast-compatible real arrays.
    pub fn remainder(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_remainder(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.remainder", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.remainder", status: -1 })?;
        Ok(Self { raw })
    }

    /// Raises broadcast-compatible arrays elementwise.
    pub fn power(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_power(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.power", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.power", status: -1 })?;
        Ok(Self { raw })
    }

    /// Selects broadcasted minima while propagating NaN values.
    pub fn elementwise_minimum(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_elementwise_minimum(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.elementwise_minimum", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.elementwise_minimum", status: -1 })?;
        Ok(Self { raw })
    }

    /// Selects broadcasted maxima while propagating NaN values.
    pub fn elementwise_maximum(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_elementwise_maximum(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.elementwise_maximum", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.elementwise_maximum", status: -1 })?;
        Ok(Self { raw })
    }

    /// Selects broadcasted minima while preferring a non-NaN operand.
    pub fn fmin(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_fmin(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.fmin", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.fmin", status: -1 })?;
        Ok(Self { raw })
    }

    /// Selects broadcasted maxima while preferring a non-NaN operand.
    pub fn fmax(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_fmax(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.fmax", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.fmax", status: -1 })?;
        Ok(Self { raw })
    }

    /// Clips values through broadcast-compatible lower and upper arrays.
    pub fn clip(&self, minimum: &Self, maximum: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_clip(self.raw.as_ptr(), minimum.raw.as_ptr(), maximum.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.clip", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.clip", status: -1 })?;
        Ok(Self { raw })
    }

    /// Clips values through a broadcast-compatible lower bound.
    pub fn clip_lower(&self, minimum: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_clip_lower(self.raw.as_ptr(), minimum.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.clip_lower", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.clip_lower", status: -1 })?;
        Ok(Self { raw })
    }

    /// Clips values through a broadcast-compatible upper bound.
    pub fn clip_upper(&self, maximum: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_clip_upper(self.raw.as_ptr(), maximum.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.clip_upper", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.clip_upper", status: -1 })?;
        Ok(Self { raw })
    }

    /// Clips into a writable destination after complete shape and cast validation.
    pub fn clip_into(&mut self, array: &Self, minimum: &Self, maximum: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_clip_into(self.raw.as_ptr(), array.raw.as_ptr(), minimum.raw.as_ptr(), maximum.raw.as_ptr()) };
        check_status("ndarray.array.clip_into", status, 0)?;
        Ok(())
    }

    /// Clips a lower bound into a writable destination after complete validation.
    pub fn clip_lower_into(&mut self, array: &Self, minimum: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_clip_lower_into(self.raw.as_ptr(), array.raw.as_ptr(), minimum.raw.as_ptr()) };
        check_status("ndarray.array.clip_lower_into", status, 0)?;
        Ok(())
    }

    /// Clips an upper bound into a writable destination after complete validation.
    pub fn clip_upper_into(&mut self, array: &Self, maximum: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_clip_upper_into(self.raw.as_ptr(), array.raw.as_ptr(), maximum.raw.as_ptr()) };
        check_status("ndarray.array.clip_upper_into", status, 0)?;
        Ok(())
    }

    /// Clips an array in place after complete validation.
    pub fn clip_inplace(&mut self, minimum: &Self, maximum: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_clip_inplace(self.raw.as_ptr(), minimum.raw.as_ptr(), maximum.raw.as_ptr()) };
        check_status("ndarray.array.clip_inplace", status, 0)?;
        Ok(())
    }

    /// Applies a lower bound to an array in place after complete validation.
    pub fn clip_lower_inplace(&mut self, minimum: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_clip_lower_inplace(self.raw.as_ptr(), minimum.raw.as_ptr()) };
        check_status("ndarray.array.clip_lower_inplace", status, 0)?;
        Ok(())
    }

    /// Applies an upper bound to an array in place after complete validation.
    pub fn clip_upper_inplace(&mut self, maximum: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_clip_upper_inplace(self.raw.as_ptr(), maximum.raw.as_ptr()) };
        check_status("ndarray.array.clip_upper_inplace", status, 0)?;
        Ok(())
    }

    /// Compares broadcast-compatible arrays for equality.
    pub fn equal(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_equal(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.equal", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.equal", status: -1 })?;
        Ok(Self { raw })
    }

    /// Compares broadcast-compatible arrays for inequality.
    pub fn not_equal(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_not_equal(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.not_equal", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.not_equal", status: -1 })?;
        Ok(Self { raw })
    }

    /// Compares broadcast-compatible real arrays with less-than.
    pub fn less(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_less(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.less", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.less", status: -1 })?;
        Ok(Self { raw })
    }

    /// Compares broadcast-compatible real arrays with less-than-or-equal.
    pub fn less_equal(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_less_equal(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.less_equal", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.less_equal", status: -1 })?;
        Ok(Self { raw })
    }

    /// Compares broadcast-compatible real arrays with greater-than.
    pub fn greater(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_greater(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.greater", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.greater", status: -1 })?;
        Ok(Self { raw })
    }

    /// Compares broadcast-compatible real arrays with greater-than-or-equal.
    pub fn greater_equal(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_greater_equal(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.greater_equal", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.greater_equal", status: -1 })?;
        Ok(Self { raw })
    }

}
