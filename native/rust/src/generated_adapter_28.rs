use super::*;

impl RandomStream {
    /// Advances the stream while returning an independently owned shuffled array.
    pub fn shuffle(&mut self, array: &Array, axis: i64) -> Result<Array, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_shuffle(self.raw.as_ptr(), array.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.random_stream.shuffle", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.shuffle", status: -1 })?;
        Ok(Array { raw })
    }

}
