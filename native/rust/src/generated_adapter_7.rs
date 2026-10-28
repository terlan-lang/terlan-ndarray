use super::*;

impl Array {
    /// Computes elementwise truth-value conjunction with broadcasting.
    pub fn logical_and(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_logical_and(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.logical_and", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.logical_and", status: -1 })?;
        Ok(Self { raw })
    }

    /// Computes elementwise truth-value disjunction with broadcasting.
    pub fn logical_or(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_logical_or(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.logical_or", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.logical_or", status: -1 })?;
        Ok(Self { raw })
    }

    /// Computes elementwise truth-value exclusive disjunction with broadcasting.
    pub fn logical_xor(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_logical_xor(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.logical_xor", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.logical_xor", status: -1 })?;
        Ok(Self { raw })
    }

    /// Computes broadcast-compatible integer bitwise conjunction.
    pub fn bitwise_and(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_bitwise_and(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.bitwise_and", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.bitwise_and", status: -1 })?;
        Ok(Self { raw })
    }

    /// Computes broadcast-compatible integer bitwise disjunction.
    pub fn bitwise_or(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_bitwise_or(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.bitwise_or", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.bitwise_or", status: -1 })?;
        Ok(Self { raw })
    }

    /// Computes broadcast-compatible integer bitwise exclusive disjunction.
    pub fn bitwise_xor(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_bitwise_xor(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.bitwise_xor", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.bitwise_xor", status: -1 })?;
        Ok(Self { raw })
    }

    /// Shifts integer values left by validated elementwise counts.
    pub fn left_shift(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_left_shift(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.left_shift", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.left_shift", status: -1 })?;
        Ok(Self { raw })
    }

    /// Shifts integer values right by validated elementwise counts.
    pub fn right_shift(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_right_shift(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.right_shift", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.right_shift", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns elementwise non-negative greatest common divisors.
    pub fn gcd(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_gcd(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.gcd", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.gcd", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns checked elementwise least common multiples.
    pub fn lcm(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_lcm(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.lcm", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.lcm", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns stable elementwise Euclidean magnitudes.
    pub fn hypot(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_hypot(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.hypot", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.hypot", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns signed elementwise angles from y and x components.
    pub fn arctan2(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_arctan2(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.arctan2", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.arctan2", status: -1 })?;
        Ok(Self { raw })
    }

    /// Computes stable elementwise logarithms of summed exponentials.
    pub fn logaddexp(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_logaddexp(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.logaddexp", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.logaddexp", status: -1 })?;
        Ok(Self { raw })
    }

    /// Computes stable base-two logarithms of summed powers of two.
    pub fn logaddexp2(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_logaddexp2(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.logaddexp2", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.logaddexp2", status: -1 })?;
        Ok(Self { raw })
    }

    /// Copies elementwise floating signs from the second operand.
    pub fn copysign(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_copysign(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.copysign", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.copysign", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the next representable values toward the second operand.
    pub fn nextafter(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nextafter(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.nextafter", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.nextafter", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a broadcast-compatible tolerance-comparison Bool mask.
    pub fn is_close(&self, rhs: &Self, relative_tolerance: f64, absolute_tolerance: f64, equal_nan: bool) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_is_close(self.raw.as_ptr(), rhs.raw.as_ptr(), relative_tolerance, absolute_tolerance, equal_nan as i32, &mut raw) };
        check_status("ndarray.array.is_close", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.is_close", status: -1 })?;
        Ok(Self { raw })
    }

    /// Selects promoted branch values through a broadcast-compatible truth-value condition.
    pub fn select_where(&self, selected: &Self, rejected: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_where(self.raw.as_ptr(), selected.raw.as_ptr(), rejected.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.select_where", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.select_where", status: -1 })?;
        Ok(Self { raw })
    }

    /// Negates every numerical element.
    pub fn negative(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_negative(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.negative", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.negative", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns an owned numerical copy while rejecting Bool input.
    pub fn positive(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_positive(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.positive", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.positive", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns elementwise magnitude, using real output for complex arrays.
    pub fn absolute(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_absolute(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.absolute", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.absolute", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise numerical sign.
    pub fn sign(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_sign(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.sign", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.sign", status: -1 })?;
        Ok(Self { raw })
    }

    /// Squares every numerical element.
    pub fn square(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_square(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.square", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.square", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise principal square root.
    pub fn sqrt(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_sqrt(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.sqrt", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.sqrt", status: -1 })?;
        Ok(Self { raw })
    }

}
