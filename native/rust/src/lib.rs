//! Generated Rust adapter for the `ndarray` native package.

#![deny(unsafe_op_in_unsafe_fn)]

/// Raw C ABI declarations used by the safe generated adapter.
pub mod ffi {
    /// Opaque CPU array with package-owned shared storage, shape, offset, and element strides.
    #[repr(C)]
    pub struct TerlanNdArray {
        _private: [u8; 0],
    }

    /// Owned Arrow C Data array and schema with release callbacks and no nullable values.
    #[repr(C)]
    pub struct TerlanNdArrowArray {
        _private: [u8; 0],
    }

    /// Thread-confined ChaCha8 random state advanced explicitly by package operations.
    #[repr(C)]
    pub struct TerlanNdRandomStream {
        _private: [u8; 0],
    }

    include!("generated_ffi_0.rs");
    include!("generated_ffi_1.rs");
    include!("generated_ffi_2.rs");
    include!("generated_ffi_3.rs");
    include!("generated_ffi_4.rs");
    include!("generated_ffi_5.rs");
    include!("generated_ffi_6.rs");
    include!("generated_ffi_7.rs");
    include!("generated_ffi_8.rs");
    include!("generated_ffi_9.rs");
    include!("generated_ffi_10.rs");
    include!("generated_ffi_11.rs");
    include!("generated_ffi_12.rs");
    include!("generated_ffi_13.rs");
    include!("generated_ffi_14.rs");
    include!("generated_ffi_15.rs");
    include!("generated_ffi_16.rs");
}

use std::ptr::NonNull;

/// Error returned when a generated C ABI operation fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CAbiError {
    /// Stable operation identifier for the failed call.
    pub operation: &'static str,
    /// Status code returned by the native implementation.
    pub status: i32,
}

/// Opaque CPU array with package-owned shared storage, shape, offset, and element strides.
pub struct Array {
    raw: NonNull<ffi::TerlanNdArray>,
}

/// Owned Arrow C Data array and schema with release callbacks and no nullable values.
pub struct ArrowArray {
    raw: NonNull<ffi::TerlanNdArrowArray>,
}

/// Thread-confined ChaCha8 random state advanced explicitly by package operations.
pub struct RandomStream {
    raw: NonNull<ffi::TerlanNdRandomStream>,
}

mod generated_adapter_0;
mod generated_adapter_1;
mod generated_adapter_2;
mod generated_adapter_3;
mod generated_adapter_4;
mod generated_adapter_5;
mod generated_adapter_6;
mod generated_adapter_7;
mod generated_adapter_8;
mod generated_adapter_9;
mod generated_adapter_10;
mod generated_adapter_11;
mod generated_adapter_12;
mod generated_adapter_13;
mod generated_adapter_14;
mod generated_adapter_15;
mod generated_adapter_16;
mod generated_adapter_17;
mod generated_adapter_18;
mod generated_adapter_19;
mod generated_adapter_20;
mod generated_adapter_21;
mod generated_adapter_22;
mod generated_adapter_23;
mod generated_adapter_24;
mod generated_adapter_25;

impl Drop for Array {
    fn drop(&mut self) {
        // SAFETY: this adapter is the sole owner and invokes the destructor once.
        let status = unsafe { ffi::terlan_ndarray_v1_delete(self.raw.as_ptr()) };
        debug_assert_eq!(status, 0);
    }
}

impl ArrowArray {
    /// Returns the flattened Arrow element count.
    pub fn length(&self) -> Result<i64, CAbiError> {
        let mut out_out_length: i64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_arrow_length(self.raw.as_ptr(), &mut out_out_length) };
        check_status("ndarray.arrow_array.length", status, 0)?;
        Ok(out_out_length)
    }

    /// Returns zero under the ndarray no-null Arrow policy.
    pub fn null_count(&self) -> Result<i64, CAbiError> {
        let mut out_out_null_count: i64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_arrow_null_count(self.raw.as_ptr(), &mut out_out_null_count) };
        check_status("ndarray.arrow_array.null_count", status, 0)?;
        Ok(out_out_null_count)
    }

}

impl Drop for ArrowArray {
    fn drop(&mut self) {
        // SAFETY: this adapter is the sole owner and invokes the destructor once.
        let status = unsafe { ffi::terlan_ndarray_v1_arrow_delete(self.raw.as_ptr()) };
        debug_assert_eq!(status, 0);
    }
}

mod generated_adapter_26;
mod generated_adapter_27;
mod generated_adapter_28;

impl Drop for RandomStream {
    fn drop(&mut self) {
        // SAFETY: this adapter is the sole owner and invokes the destructor once.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_delete(self.raw.as_ptr()) };
        debug_assert_eq!(status, 0);
    }
}

include!("generated_free_adapter_0.rs");
include!("generated_free_adapter_1.rs");
include!("generated_free_adapter_2.rs");
include!("generated_free_adapter_3.rs");
include!("generated_free_adapter_4.rs");
include!("generated_free_adapter_5.rs");

fn check_status<T>(operation: &'static str, status: T, success: i32) -> Result<(), CAbiError>
where
    T: TryInto<i32>,
{
    let status = status.try_into().map_err(|_| CAbiError { operation, status: -2 })?;
    if status == success {
        Ok(())
    } else {
        Err(CAbiError { operation, status })
    }
}

mod package_extension;
pub use package_extension::*;
