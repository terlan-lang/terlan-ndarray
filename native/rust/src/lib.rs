#![deny(unsafe_op_in_unsafe_fn)]

pub mod ffi {
    #[repr(C)]
    pub struct TerlanNdArray {
        _private: [u8; 0],
    }

    unsafe extern "C" {
        pub fn terlan_ndarray_v1_add(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        pub fn terlan_ndarray_v1_byte_count(array: *mut TerlanNdArray, out_byte_count: *mut i64) -> i32;
        pub fn terlan_ndarray_v1_delete(array: *mut TerlanNdArray) -> i32;
        pub fn terlan_ndarray_v1_delete_bool_values(values: *mut u8);
        pub fn terlan_ndarray_v1_delete_float_values(values: *mut f64);
        pub fn terlan_ndarray_v1_delete_int_values(values: *mut i64);
        pub fn terlan_ndarray_v1_device_type(array: *mut TerlanNdArray, out_device_type: *mut i32) -> i32;
        pub fn terlan_ndarray_v1_dtype(array: *mut TerlanNdArray, out_dtype: *mut i32) -> i32;
        pub fn terlan_ndarray_v1_from_bools(values: *const u8, value_count: i64, shape: *const i64, rank: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        pub fn terlan_ndarray_v1_from_floats(values: *const f64, value_count: i64, shape: *const i64, rank: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        pub fn terlan_ndarray_v1_from_ints(values: *const i64, value_count: i64, shape: *const i64, rank: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        pub fn terlan_ndarray_v1_is_contiguous(array: *mut TerlanNdArray, out_contiguous: *mut i32) -> i32;
        pub fn terlan_ndarray_v1_live_count() -> i64;
        pub fn terlan_ndarray_v1_multiply(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        pub fn terlan_ndarray_v1_numel(array: *mut TerlanNdArray, out_numel: *mut i64) -> i32;
        pub fn terlan_ndarray_v1_rank(array: *mut TerlanNdArray, out_rank: *mut i64) -> i32;
        pub fn terlan_ndarray_v1_reshape(array: *mut TerlanNdArray, shape: *const i64, rank: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        pub fn terlan_ndarray_v1_shape(array: *mut TerlanNdArray, out_shape: *mut *mut i64) -> i32;
        pub fn terlan_ndarray_v1_strides(array: *mut TerlanNdArray, out_strides: *mut *mut i64) -> i32;
        pub fn terlan_ndarray_v1_subtract(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        pub fn terlan_ndarray_v1_sum(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        pub fn terlan_ndarray_v1_to_bools(array: *mut TerlanNdArray, out_values: *mut *mut u8, out_length: *mut usize) -> i32;
        pub fn terlan_ndarray_v1_to_floats(array: *mut TerlanNdArray, out_values: *mut *mut f64, out_length: *mut usize) -> i32;
        pub fn terlan_ndarray_v1_to_ints(array: *mut TerlanNdArray, out_values: *mut *mut i64, out_length: *mut usize) -> i32;
        pub fn terlan_ndarray_v1_transpose(array: *mut TerlanNdArray, axis_a: i64, axis_b: i64, out_array: *mut *mut TerlanNdArray) -> i32;
    }
}

use std::ptr::NonNull;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CAbiError {
    pub operation: &'static str,
    pub status: i32,
}

pub struct Array {
    raw: NonNull<ffi::TerlanNdArray>,
}

impl Array {
    pub fn from_bools(values: &[bool], shape: &[i64]) -> Result<Self, CAbiError> {
        let values_bytes = values.iter().copied().map(u8::from).collect::<Vec<_>>();
        let values_length = i64::try_from(values.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_bools", status: -2 })?;
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_bools", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_from_bools(values_bytes.as_ptr(), values_length, shape.as_ptr(), shape_length, &mut raw) };
        check_status("ndarray.array.from_bools", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.from_bools", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn from_ints(values: &[i64], shape: &[i64]) -> Result<Self, CAbiError> {
        let values_length = i64::try_from(values.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_ints", status: -2 })?;
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_ints", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_from_ints(values.as_ptr(), values_length, shape.as_ptr(), shape_length, &mut raw) };
        check_status("ndarray.array.from_ints", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.from_ints", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn from_floats(values: &[f64], shape: &[i64]) -> Result<Self, CAbiError> {
        let values_length = i64::try_from(values.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_floats", status: -2 })?;
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.from_floats", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_from_floats(values.as_ptr(), values_length, shape.as_ptr(), shape_length, &mut raw) };
        check_status("ndarray.array.from_floats", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.from_floats", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn rank(&self) -> Result<i64, CAbiError> {
        let mut out_out_rank: i64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_rank(self.raw.as_ptr(), &mut out_out_rank) };
        check_status("ndarray.array.rank", status as i32, 0)?;
        Ok(out_out_rank as i64)
    }

    pub fn shape(&self) -> Result<Vec<i64>, CAbiError> {
        let out_shape_length = self.rank()?;
        let mut out_out_shape: *mut i64 = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_shape(self.raw.as_ptr(), &mut out_out_shape) };
        check_status("ndarray.array.shape", status as i32, 0)?;
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

    pub fn numel(&self) -> Result<i64, CAbiError> {
        let mut out_out_numel: i64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_numel(self.raw.as_ptr(), &mut out_out_numel) };
        check_status("ndarray.array.numel", status as i32, 0)?;
        Ok(out_out_numel as i64)
    }

    pub fn strides(&self) -> Result<Vec<i64>, CAbiError> {
        let out_strides_length = self.rank()?;
        let mut out_out_strides: *mut i64 = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_strides(self.raw.as_ptr(), &mut out_out_strides) };
        check_status("ndarray.array.strides", status as i32, 0)?;
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

    pub fn byte_count(&self) -> Result<i64, CAbiError> {
        let mut out_out_byte_count: i64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_byte_count(self.raw.as_ptr(), &mut out_out_byte_count) };
        check_status("ndarray.array.byte_count", status as i32, 0)?;
        Ok(out_out_byte_count as i64)
    }

    pub fn dtype_code(&self) -> Result<i64, CAbiError> {
        let mut out_out_dtype: i32 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_dtype(self.raw.as_ptr(), &mut out_out_dtype) };
        check_status("ndarray.array.dtype_code", status as i32, 0)?;
        Ok(out_out_dtype as i64)
    }

    pub fn contiguous_code(&self) -> Result<i64, CAbiError> {
        let mut out_out_contiguous: i32 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_is_contiguous(self.raw.as_ptr(), &mut out_out_contiguous) };
        check_status("ndarray.array.contiguous_code", status as i32, 0)?;
        Ok(out_out_contiguous as i64)
    }

    pub fn device_type_code(&self) -> Result<i64, CAbiError> {
        let mut out_out_device_type: i32 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_device_type(self.raw.as_ptr(), &mut out_out_device_type) };
        check_status("ndarray.array.device_type_code", status as i32, 0)?;
        Ok(out_out_device_type as i64)
    }

    pub fn to_bools(&self) -> Result<Vec<bool>, CAbiError> {
        let mut out_out_values: *mut u8 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_to_bools(self.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.to_bools", status as i32, 0)?;
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

    pub fn to_ints(&self) -> Result<Vec<i64>, CAbiError> {
        let mut out_out_values: *mut i64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_to_ints(self.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.to_ints", status as i32, 0)?;
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

    pub fn to_floats(&self) -> Result<Vec<f64>, CAbiError> {
        let mut out_out_values: *mut f64 = std::ptr::null_mut();
        let mut out_out_values_length: usize = 0;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_to_floats(self.raw.as_ptr(), &mut out_out_values, &mut out_out_values_length) };
        check_status("ndarray.array.to_floats", status as i32, 0)?;
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

    pub fn reshape(&self, shape: &[i64]) -> Result<Self, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.reshape", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_reshape(self.raw.as_ptr(), shape.as_ptr(), shape_length, &mut raw) };
        check_status("ndarray.array.reshape", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.reshape", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn transpose(&self, axis_a: i64, axis_b: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_transpose(self.raw.as_ptr(), axis_a as i64, axis_b as i64, &mut raw) };
        check_status("ndarray.array.transpose", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.transpose", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_add(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.add", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.add", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn subtract(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_subtract(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.subtract", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.subtract", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn multiply(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_multiply(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.multiply", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.multiply", status: -1 })?;
        Ok(Self { raw })
    }

    pub fn sum(&self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.sum", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_sum(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.sum", status as i32, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.sum", status: -1 })?;
        Ok(Self { raw })
    }

}

impl Drop for Array {
    fn drop(&mut self) {
        // SAFETY: this adapter is the sole owner and invokes the destructor once.
        let status = unsafe { ffi::terlan_ndarray_v1_delete(self.raw.as_ptr()) };
        debug_assert_eq!(status, 0);
    }
}

pub fn live_count() -> i64 {
    // SAFETY: generated arguments follow the reviewed ownership metadata.
    let value = unsafe { ffi::terlan_ndarray_v1_live_count() };
    value as i64
}

fn check_status(operation: &'static str, status: i32, success: i32) -> Result<(), CAbiError> {
    if status == success {
        Ok(())
    } else {
        Err(CAbiError { operation, status })
    }
}

