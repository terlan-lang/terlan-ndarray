use super::*;

impl Array {
    /// Provides NumPy-compatible fmod behavior.
    pub fn fmod(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_fmod(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.fmod", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.fmod", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible heaviside behavior.
    pub fn heaviside(&self, zero_value: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_heaviside(self.raw.as_ptr(), zero_value.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.heaviside", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.heaviside", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible i0 behavior.
    pub fn i0(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_i0(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.i0", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.i0", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible real behavior.
    pub fn real(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_real(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.real", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.real", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible imag behavior.
    pub fn imag(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_imag(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.imag", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.imag", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible iscomplex behavior.
    pub fn iscomplex(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_iscomplex(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.iscomplex", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.iscomplex", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible isreal behavior.
    pub fn isreal(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_isreal(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.isreal", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.isreal", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible iscomplexobj behavior.
    pub fn iscomplexobj(&self) -> Result<bool, CAbiError> {
        let mut out_out_value: bool = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_iscomplexobj(self.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.numpy.iscomplexobj", status, 0)?;
        Ok(out_out_value)
    }

    /// Provides NumPy-compatible isrealobj behavior.
    pub fn isrealobj(&self) -> Result<bool, CAbiError> {
        let mut out_out_value: bool = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_isrealobj(self.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.numpy.isrealobj", status, 0)?;
        Ok(out_out_value)
    }

    /// Provides NumPy-compatible isneginf behavior.
    pub fn isneginf(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_isneginf(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.isneginf", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.isneginf", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible isposinf behavior.
    pub fn isposinf(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_isposinf(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.isposinf", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.isposinf", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible ldexp behavior.
    pub fn ldexp(&self, exponents: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_ldexp(self.raw.as_ptr(), exponents.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.ldexp", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.ldexp", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible nan to num behavior.
    pub fn nan_to_num(&self, nan: f64, positive_infinity: f64, negative_infinity: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_to_num(self.raw.as_ptr(), nan, positive_infinity, negative_infinity, &mut raw) };
        check_status("ndarray.numpy.nan_to_num", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nan_to_num", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible real if close behavior.
    pub fn real_if_close(&self, tolerance: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_real_if_close(self.raw.as_ptr(), tolerance, &mut raw) };
        check_status("ndarray.numpy.real_if_close", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.real_if_close", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible sinc behavior.
    pub fn sinc(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_sinc(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.sinc", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.sinc", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible spacing behavior.
    pub fn spacing(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_spacing(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.spacing", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.spacing", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible sort complex behavior.
    pub fn sort_complex(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_sort_complex(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.sort_complex", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.sort_complex", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible trim zeros behavior.
    pub fn trim_zeros(&self, mode: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_trim_zeros(self.raw.as_ptr(), mode, &mut raw) };
        check_status("ndarray.numpy.trim_zeros", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.trim_zeros", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible unique behavior.
    pub fn unique(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_unique(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.unique", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.unique", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible invert behavior.
    pub fn invert(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_invert(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.invert", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.invert", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible trunc behavior.
    pub fn trunc(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_trunc(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.trunc", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.trunc", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible frombuffer behavior.
    pub fn frombuffer(bytes: &[u8]) -> Result<Self, CAbiError> {
        let bytes_length = i64::try_from(bytes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.frombuffer", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_frombuffer(bytes.as_ptr(), bytes_length, &mut raw) };
        check_status("ndarray.numpy.frombuffer", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.frombuffer", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible nanargmax behavior.
    pub fn nanargmax(&self) -> Result<i64, CAbiError> {
        let mut out_out_value: i64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nanargmax(self.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.numpy.nanargmax", status, 0)?;
        Ok(out_out_value)
    }

    /// Provides NumPy-compatible nanargmin behavior.
    pub fn nanargmin(&self) -> Result<i64, CAbiError> {
        let mut out_out_value: i64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nanargmin(self.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.numpy.nanargmin", status, 0)?;
        Ok(out_out_value)
    }

}
