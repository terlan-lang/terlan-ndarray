#![deny(unsafe_op_in_unsafe_fn)]

pub mod ffi {
    #[repr(C)]
    pub struct TerlanNdArray {
        _private: [u8; 0],
    }

    unsafe extern "C" {
        pub fn terlan_ndarray_v1_delete(array: *mut TerlanNdArray) -> i32;
        pub fn terlan_ndarray_v1_dtype(array: *mut TerlanNdArray, out_dtype: *mut i32) -> i32;
        pub fn terlan_ndarray_v1_from_bools(values: *const u8, value_count: i64, shape: *const i64, rank: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        pub fn terlan_ndarray_v1_from_floats(values: *const f64, value_count: i64, shape: *const i64, rank: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        pub fn terlan_ndarray_v1_from_ints(values: *const i64, value_count: i64, shape: *const i64, rank: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        pub fn terlan_ndarray_v1_is_contiguous(array: *mut TerlanNdArray, out_contiguous: *mut i32) -> i32;
        pub fn terlan_ndarray_v1_live_count() -> i64;
        pub fn terlan_ndarray_v1_numel(array: *mut TerlanNdArray, out_numel: *mut i64) -> i32;
        pub fn terlan_ndarray_v1_rank(array: *mut TerlanNdArray, out_rank: *mut i64) -> i32;
        pub fn terlan_ndarray_v1_shape(array: *mut TerlanNdArray, out_shape: *mut *mut i64) -> i32;
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

