use super::*;

impl Array {
    /// Returns one divided by each nonzero element.
    pub fn reciprocal(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_reciprocal(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.reciprocal", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.reciprocal", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise exponential.
    pub fn exp(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_exp(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.exp", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.exp", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise exponential minus one.
    pub fn expm1(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_expm1(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.expm1", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.expm1", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise natural logarithm.
    pub fn log(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_log(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.log", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.log", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise natural logarithm of one plus each value.
    pub fn log1p(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_log1p(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.log1p", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.log1p", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise base-two logarithm.
    pub fn log2(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_log2(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.log2", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.log2", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise base-ten logarithm.
    pub fn log10(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_log10(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.log10", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.log10", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise hyperbolic sine.
    pub fn sinh(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_sinh(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.sinh", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.sinh", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise hyperbolic cosine.
    pub fn cosh(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_cosh(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.cosh", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.cosh", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise hyperbolic tangent.
    pub fn tanh(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_tanh(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.tanh", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.tanh", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise inverse sine.
    pub fn arcsin(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_arcsin(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.arcsin", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.arcsin", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise inverse cosine.
    pub fn arccos(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_arccos(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.arccos", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.arccos", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise inverse tangent.
    pub fn arctan(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_arctan(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.arctan", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.arctan", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise inverse hyperbolic sine.
    pub fn arcsinh(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_arcsinh(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.arcsinh", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.arcsinh", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise inverse hyperbolic cosine.
    pub fn arccosh(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_arccosh(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.arccosh", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.arccosh", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise inverse hyperbolic tangent.
    pub fn arctanh(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_arctanh(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.arctanh", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.arctanh", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise sine.
    pub fn sin(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_sin(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.sin", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.sin", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise cosine.
    pub fn cos(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_cos(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.cos", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.cos", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise tangent.
    pub fn tan(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_tan(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.tan", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.tan", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise floor for real arrays.
    pub fn floor(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_floor(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.floor", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.floor", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the elementwise ceiling for real arrays.
    pub fn ceil(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_ceil(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.ceil", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.ceil", status: -1 })?;
        Ok(Self { raw })
    }

    /// Rounds every element to the nearest even value.
    pub fn round(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_round(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.round", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.round", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns elementwise truth-value negation.
    pub fn logical_not(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_logical_not(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.logical_not", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.logical_not", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a Bool mask identifying NaN components.
    pub fn is_nan(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_is_nan(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.is_nan", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.is_nan", status: -1 })?;
        Ok(Self { raw })
    }

}
