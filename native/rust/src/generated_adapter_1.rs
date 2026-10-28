use super::*;

impl Array {
    /// Creates a rectangular identity-style matrix at an integer diagonal offset.
    pub fn identity(rows: i64, columns: i64, diagonal: i64, dtype: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_identity(rows, columns, diagonal, dtype as i32, &mut raw) };
        check_status("ndarray.array.identity", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.identity", status: -1 })?;
        Ok(Self { raw })
    }

    /// Creates reproducible ChaCha8 uniform floating samples in a half-open interval.
    pub fn random_uniform(shape: &[i64], low: f64, high: f64, seed: i64, dtype: i64) -> Result<Self, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.random_uniform", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_uniform(shape.as_ptr(), shape_length, low, high, seed, dtype as i32, &mut raw) };
        check_status("ndarray.array.random_uniform", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.random_uniform", status: -1 })?;
        Ok(Self { raw })
    }

    /// Creates reproducible ChaCha8 normal floating samples with explicit distribution parameters.
    pub fn random_normal(shape: &[i64], mean: f64, standard_deviation: f64, seed: i64, dtype: i64) -> Result<Self, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.random_normal", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_normal(shape.as_ptr(), shape_length, mean, standard_deviation, seed, dtype as i32, &mut raw) };
        check_status("ndarray.array.random_normal", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.random_normal", status: -1 })?;
        Ok(Self { raw })
    }

    /// Creates reproducible integer samples in a half-open interval.
    pub fn random_integers(shape: &[i64], low: i64, high: i64, seed: i64, dtype: i64) -> Result<Self, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.random_integers", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_integers(shape.as_ptr(), shape_length, low, high, seed, dtype as i32, &mut raw) };
        check_status("ndarray.array.random_integers", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.random_integers", status: -1 })?;
        Ok(Self { raw })
    }

    /// Creates reproducible Boolean samples from a Bernoulli distribution.
    pub fn random_bernoulli(shape: &[i64], probability: f64, seed: i64) -> Result<Self, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.random_bernoulli", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_bernoulli(shape.as_ptr(), shape_length, probability, seed, &mut raw) };
        check_status("ndarray.array.random_bernoulli", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.random_bernoulli", status: -1 })?;
        Ok(Self { raw })
    }

    /// Creates a reproducibly shuffled permutation of consecutive integers.
    pub fn random_permutation(length: i64, seed: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_permutation(length, seed, &mut raw) };
        check_status("ndarray.array.random_permutation", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.random_permutation", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns an independently owned array reproducibly shuffled along one axis.
    pub fn random_shuffle(&self, axis: i64, seed: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_shuffle(self.raw.as_ptr(), axis, seed, &mut raw) };
        check_status("ndarray.array.random_shuffle", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.random_shuffle", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a shared strided diagonal view across two distinct axes.
    pub fn diagonal(&self, offset: i64, axis_a: i64, axis_b: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_diagonal(self.raw.as_ptr(), offset, axis_a, axis_b, &mut raw) };
        check_status("ndarray.array.diagonal", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.diagonal", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns array rank.
    pub fn rank(&self) -> Result<i64, CAbiError> {
        let mut out_out_rank: i64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_rank(self.raw.as_ptr(), &mut out_out_rank) };
        check_status("ndarray.array.rank", status, 0)?;
        Ok(out_out_rank)
    }

    /// Copies array shape into a Terlan list.
    pub fn shape(&self) -> Result<Vec<i64>, CAbiError> {
        let out_shape_length = self.rank()?;
        let mut out_out_shape: *mut i64 = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_shape(self.raw.as_ptr(), &mut out_out_shape) };
        check_status("ndarray.array.shape", status, 0)?;
        let length = usize::try_from(out_shape_length).map_err(|_| CAbiError { operation: "ndarray.array.shape", status: -2 })?;
        let values = if length == 0 {
            Vec::new()
        } else {
            let pointer = NonNull::new(out_out_shape).ok_or(CAbiError { operation: "ndarray.array.shape", status: -1 })?;
            // SAFETY: metadata ties this borrowed array to `self`; it is copied before the borrow ends.
            unsafe { std::slice::from_raw_parts(pointer.as_ptr(), length).to_vec() }
        };
        Ok(values)
    }

    /// Returns logical element count.
    pub fn numel(&self) -> Result<i64, CAbiError> {
        let mut out_out_numel: i64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_numel(self.raw.as_ptr(), &mut out_out_numel) };
        check_status("ndarray.array.numel", status, 0)?;
        Ok(out_out_numel)
    }

    /// Copies the array's logical element strides, including zero and negative view strides.
    pub fn strides(&self) -> Result<Vec<i64>, CAbiError> {
        let out_strides_length = self.rank()?;
        let mut out_out_strides: *mut i64 = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_strides(self.raw.as_ptr(), &mut out_out_strides) };
        check_status("ndarray.array.strides", status, 0)?;
        let length = usize::try_from(out_strides_length).map_err(|_| CAbiError { operation: "ndarray.array.strides", status: -2 })?;
        let values = if length == 0 {
            Vec::new()
        } else {
            let pointer = NonNull::new(out_out_strides).ok_or(CAbiError { operation: "ndarray.array.strides", status: -1 })?;
            // SAFETY: metadata ties this borrowed array to `self`; it is copied before the borrow ends.
            unsafe { std::slice::from_raw_parts(pointer.as_ptr(), length).to_vec() }
        };
        Ok(values)
    }

    /// Returns the checked byte count of the logical array elements.
    pub fn byte_count(&self) -> Result<i64, CAbiError> {
        let mut out_out_byte_count: i64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_byte_count(self.raw.as_ptr(), &mut out_out_byte_count) };
        check_status("ndarray.array.byte_count", status, 0)?;
        Ok(out_out_byte_count)
    }

    /// Returns the ABI-v1 dtype code.
    pub fn dtype_code(&self) -> Result<i64, CAbiError> {
        let mut out_out_dtype: i32 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_dtype(self.raw.as_ptr(), &mut out_out_dtype) };
        check_status("ndarray.array.dtype_code", status, 0)?;
        Ok(out_out_dtype as i64)
    }

    /// Returns one when the current view is canonical C-contiguous.
    pub fn contiguous_code(&self) -> Result<i64, CAbiError> {
        let mut out_out_contiguous: i32 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_is_contiguous(self.raw.as_ptr(), &mut out_out_contiguous) };
        check_status("ndarray.array.contiguous_code", status, 0)?;
        Ok(out_out_contiguous as i64)
    }

    /// Returns one when the current view is Fortran-contiguous.
    pub fn fortran_contiguous_code(&self) -> Result<i64, CAbiError> {
        let mut out_out_contiguous: i32 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_is_fortran_contiguous(self.raw.as_ptr(), &mut out_out_contiguous) };
        check_status("ndarray.array.fortran_contiguous_code", status, 0)?;
        Ok(out_out_contiguous as i64)
    }

    /// Returns the ABI-v1 CPU device code.
    pub fn device_type_code(&self) -> Result<i64, CAbiError> {
        let mut out_out_device_type: i32 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_device_type(self.raw.as_ptr(), &mut out_out_device_type) };
        check_status("ndarray.array.device_type_code", status, 0)?;
        Ok(out_out_device_type as i64)
    }

    /// Returns an owned copy of Bool values.
    pub fn to_bools(&self) -> Result<Vec<bool>, CAbiError> {
        let mut out_out_values: *mut u8 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_to_bools(self.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.to_bools", status, 0)?;
        let values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized elements; they are copied before destruction.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.to_bools", status: -1 })
        };
        // SAFETY: ownership metadata transfers this array to the named destructor exactly once.
        unsafe { ffi::terlan_ndarray_v1_delete_bool_values(out_out_values) };
        let values = values?;
        values.into_iter().map(|value| match value {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(CAbiError { operation: "ndarray.array.to_bools", status: -4 }),
        }).collect()
    }

    /// Returns an owned copy of Int64 values.
    pub fn to_ints(&self) -> Result<Vec<i64>, CAbiError> {
        let mut out_out_values: *mut i64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_to_ints(self.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.to_ints", status, 0)?;
        let values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized elements; they are copied before destruction.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.to_ints", status: -1 })
        };
        // SAFETY: ownership metadata transfers this array to the named destructor exactly once.
        unsafe { ffi::terlan_ndarray_v1_delete_int_values(out_out_values) };
        Ok(values?)
    }

    /// Returns an owned copy of Float64 values.
    pub fn to_floats(&self) -> Result<Vec<f64>, CAbiError> {
        let mut out_out_values: *mut f64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_to_floats(self.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.to_floats", status, 0)?;
        let values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized elements; they are copied before destruction.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.to_floats", status: -1 })
        };
        // SAFETY: ownership metadata transfers this array to the named destructor exactly once.
        unsafe { ffi::terlan_ndarray_v1_delete_float_values(out_out_values) };
        Ok(values?)
    }

    /// Returns checked integral values from any exactly representable non-complex dtype.
    pub fn to_ints_cast(&self) -> Result<Vec<i64>, CAbiError> {
        let mut out_out_values: *mut i64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_to_ints_cast(self.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.to_ints_cast", status, 0)?;
        let values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized elements; they are copied before destruction.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.to_ints_cast", status: -1 })
        };
        // SAFETY: ownership metadata transfers this array to the named destructor exactly once.
        unsafe { ffi::terlan_ndarray_v1_delete_int_values(out_out_values) };
        Ok(values?)
    }

    /// Returns real values converted to Terlan Float from any non-complex dtype.
    pub fn to_floats_cast(&self) -> Result<Vec<f64>, CAbiError> {
        let mut out_out_values: *mut f64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_to_floats_cast(self.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.to_floats_cast", status, 0)?;
        let values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized elements; they are copied before destruction.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.to_floats_cast", status: -1 })
        };
        // SAFETY: ownership metadata transfers this array to the named destructor exactly once.
        unsafe { ffi::terlan_ndarray_v1_delete_float_values(out_out_values) };
        Ok(values?)
    }

    /// Returns the real components of a complex array.
    pub fn complex_reals(&self) -> Result<Vec<f64>, CAbiError> {
        let mut out_out_values: *mut f64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_complex_reals(self.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.complex_reals", status, 0)?;
        let values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized elements; they are copied before destruction.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.complex_reals", status: -1 })
        };
        // SAFETY: ownership metadata transfers this array to the named destructor exactly once.
        unsafe { ffi::terlan_ndarray_v1_delete_float_values(out_out_values) };
        Ok(values?)
    }

    /// Returns the imaginary components of a complex array.
    pub fn complex_imags(&self) -> Result<Vec<f64>, CAbiError> {
        let mut out_out_values: *mut f64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_complex_imags(self.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.complex_imags", status, 0)?;
        let values = if out_out_values_length == 0 {
            Ok(Vec::new())
        } else {
            NonNull::new(out_out_values).map(|pointer| {
                // SAFETY: the producer returned `out_out_values_length` initialized elements; they are copied before destruction.
                unsafe { std::slice::from_raw_parts(pointer.as_ptr(), out_out_values_length).to_vec() }
            }).ok_or(CAbiError { operation: "ndarray.array.complex_imags", status: -1 })
        };
        // SAFETY: ownership metadata transfers this array to the named destructor exactly once.
        unsafe { ffi::terlan_ndarray_v1_delete_float_values(out_out_values) };
        Ok(values?)
    }

}
