use super::*;

impl Array {
    /// Copies boolean values and shape into owned CPU storage.
    pub fn from_bools(values: &[bool], shape: &[i64]) -> Result<Self, CAbiError> {
        let values_bytes = values.iter().copied().map(u8::from).collect::<Vec<_>>();
        let values_length = i64::try_from(values.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_bools", status: -2 })?;
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_bools", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_from_bools(values_bytes.as_ptr(), values_length, shape.as_ptr(), shape_length, &mut raw) };
        check_status("ndarray.array.from_bools", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.from_bools", status: -1 })?;
        Ok(Self { raw })
    }

    /// Copies Int64 values and shape into owned CPU storage.
    pub fn from_ints(values: &[i64], shape: &[i64]) -> Result<Self, CAbiError> {
        let values_length = i64::try_from(values.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_ints", status: -2 })?;
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_ints", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_from_ints(values.as_ptr(), values_length, shape.as_ptr(), shape_length, &mut raw) };
        check_status("ndarray.array.from_ints", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.from_ints", status: -1 })?;
        Ok(Self { raw })
    }

    /// Copies Float64 values and shape into owned CPU storage.
    pub fn from_floats(values: &[f64], shape: &[i64]) -> Result<Self, CAbiError> {
        let values_length = i64::try_from(values.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_floats", status: -2 })?;
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_floats", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_from_floats(values.as_ptr(), values_length, shape.as_ptr(), shape_length, &mut raw) };
        check_status("ndarray.array.from_floats", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.from_floats", status: -1 })?;
        Ok(Self { raw })
    }

    /// Constructs an array in a checked integer-compatible dtype code.
    pub fn from_ints_as(values: &[i64], shape: &[i64], dtype: i64) -> Result<Self, CAbiError> {
        let values_length = i64::try_from(values.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_ints_as", status: -2 })?;
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_ints_as", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_from_ints_as(values.as_ptr(), values_length, shape.as_ptr(), shape_length, dtype as i32, &mut raw) };
        check_status("ndarray.array.from_ints_as", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.from_ints_as", status: -1 })?;
        Ok(Self { raw })
    }

    /// Constructs an array in a checked floating-compatible dtype code.
    pub fn from_floats_as(values: &[f64], shape: &[i64], dtype: i64) -> Result<Self, CAbiError> {
        let values_length = i64::try_from(values.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_floats_as", status: -2 })?;
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_floats_as", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_from_floats_as(values.as_ptr(), values_length, shape.as_ptr(), shape_length, dtype as i32, &mut raw) };
        check_status("ndarray.array.from_floats_as", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.from_floats_as", status: -1 })?;
        Ok(Self { raw })
    }

    /// Constructs Complex64 or Complex128 storage from parallel real and imaginary values.
    pub fn from_complex(real_values: &[f64], imag_values: &[f64], shape: &[i64], dtype: i64) -> Result<Self, CAbiError> {
        let real_values_length = i64::try_from(real_values.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_complex", status: -2 })?;
        let imag_values_length = i64::try_from(imag_values.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_complex", status: -2 })?;
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_complex", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_from_complex(real_values.as_ptr(), real_values_length, imag_values.as_ptr(), imag_values_length, shape.as_ptr(), shape_length, dtype as i32, &mut raw) };
        check_status("ndarray.array.from_complex", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.from_complex", status: -1 })?;
        Ok(Self { raw })
    }

    /// Allocates an initialized array; native-boundary safety guarantees zeroed storage rather than exposing uninitialized bytes.
    pub fn empty(shape: &[i64], dtype: i64) -> Result<Self, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.empty", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_empty(shape.as_ptr(), shape_length, dtype as i32, &mut raw) };
        check_status("ndarray.array.empty", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.empty", status: -1 })?;
        Ok(Self { raw })
    }

    /// Creates a zero-filled array in the requested dtype.
    pub fn zeros(shape: &[i64], dtype: i64) -> Result<Self, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.zeros", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_zeros(shape.as_ptr(), shape_length, dtype as i32, &mut raw) };
        check_status("ndarray.array.zeros", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.zeros", status: -1 })?;
        Ok(Self { raw })
    }

    /// Creates a zero-filled array with explicit C or Fortran storage order.
    pub fn zeros_ordered(shape: &[i64], dtype: i64, order: i64) -> Result<Self, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.zeros_ordered", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_zeros_ordered(shape.as_ptr(), shape_length, dtype as i32, order as i32, &mut raw) };
        check_status("ndarray.array.zeros_ordered", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.zeros_ordered", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reduces an axis by ordered subtraction without applying a commutative reassociation.
    pub fn subtract_reduce(&self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_subtract_reduce(self.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.array.subtract_reduce", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.subtract_reduce", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the ordered cumulative subtraction along an axis.
    pub fn subtract_accumulate(&self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_subtract_accumulate(self.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.array.subtract_accumulate", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.subtract_accumulate", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reduces an axis by ordered floating or complex division.
    pub fn divide_reduce(&self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_divide_reduce(self.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.array.divide_reduce", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.divide_reduce", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the ordered cumulative floating or complex division along an axis.
    pub fn divide_accumulate(&self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_divide_accumulate(self.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.array.divide_accumulate", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.divide_accumulate", status: -1 })?;
        Ok(Self { raw })
    }

    /// Creates a one-filled array in the requested dtype.
    pub fn ones(shape: &[i64], dtype: i64) -> Result<Self, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.ones", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_ones(shape.as_ptr(), shape_length, dtype as i32, &mut raw) };
        check_status("ndarray.array.ones", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.ones", status: -1 })?;
        Ok(Self { raw })
    }

    /// Creates an array filled from one checked integer value.
    pub fn full_int(shape: &[i64], value: i64, dtype: i64) -> Result<Self, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.full_int", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_full_int(shape.as_ptr(), shape_length, value, dtype as i32, &mut raw) };
        check_status("ndarray.array.full_int", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.full_int", status: -1 })?;
        Ok(Self { raw })
    }

    /// Creates an array filled from one checked floating value.
    pub fn full_float(shape: &[i64], value: f64, dtype: i64) -> Result<Self, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.full_float", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_full_float(shape.as_ptr(), shape_length, value, dtype as i32, &mut raw) };
        check_status("ndarray.array.full_float", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.full_float", status: -1 })?;
        Ok(Self { raw })
    }

    /// Creates consecutive Int64 values from zero to an exclusive integer stop.
    pub fn arange_int_stop(stop: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_arange_int_stop(stop, &mut raw) };
        check_status("ndarray.array.arange_int_stop", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.arange_int_stop", status: -1 })?;
        Ok(Self { raw })
    }

    /// Creates consecutive Float64 values from zero to an exclusive floating stop.
    pub fn arange_float_stop(stop: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_arange_float_stop(stop, &mut raw) };
        check_status("ndarray.array.arange_float_stop", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.arange_float_stop", status: -1 })?;
        Ok(Self { raw })
    }

    /// Creates exact consecutive Int64 values over a half-open integer interval.
    pub fn arange_int_interval(start: i64, stop: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_arange_int_interval(start, stop, &mut raw) };
        check_status("ndarray.array.arange_int_interval", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.arange_int_interval", status: -1 })?;
        Ok(Self { raw })
    }

    /// Creates consecutive Float64 values over a half-open floating interval.
    pub fn arange_float_interval(start: f64, stop: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_arange_float_interval(start, stop, &mut raw) };
        check_status("ndarray.array.arange_float_interval", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.arange_float_interval", status: -1 })?;
        Ok(Self { raw })
    }

    /// Creates exact Int64 values over a half-open integer interval.
    pub fn arange_int(start: i64, stop: i64, step: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_arange_int(start, stop, step, &mut raw) };
        check_status("ndarray.array.arange_int", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.arange_int", status: -1 })?;
        Ok(Self { raw })
    }

    /// Creates Float64 values over a half-open interval with an explicit step.
    pub fn arange_float_step(start: f64, stop: f64, step: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_arange_float_step(start, stop, step, &mut raw) };
        check_status("ndarray.array.arange_float_step", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.arange_float_step", status: -1 })?;
        Ok(Self { raw })
    }

    /// Creates evenly stepped values over a half-open interval.
    pub fn arange_float(start: f64, stop: f64, step: f64, dtype: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_arange(start, stop, step, dtype as i32, &mut raw) };
        check_status("ndarray.array.arange", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.arange", status: -1 })?;
        Ok(Self { raw })
    }

    /// Creates a requested number of evenly spaced values.
    pub fn linspace(start: f64, stop: f64, count: i64, endpoint: i64, dtype: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_linspace(start, stop, count, endpoint as i32, dtype as i32, &mut raw) };
        check_status("ndarray.array.linspace", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.linspace", status: -1 })?;
        Ok(Self { raw })
    }

}
