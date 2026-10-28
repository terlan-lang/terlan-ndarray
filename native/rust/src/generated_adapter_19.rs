use super::*;

impl Array {
    /// Provides NumPy-compatible nancumprod behavior.
    pub fn nancumprod(&self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nancumprod(self.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.numpy.nancumprod", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nancumprod", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible nancumsum behavior.
    pub fn nancumsum(&self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nancumsum(self.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.numpy.nancumsum", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nancumsum", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible nanmedian behavior.
    pub fn nanmedian(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nanmedian(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.numpy.nanmedian", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanmedian", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible nanpercentile behavior.
    pub fn nanpercentile(&self, percentile: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nanpercentile(self.raw.as_ptr(), percentile, &mut raw) };
        check_status("ndarray.numpy.nanpercentile", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanpercentile", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible nanquantile behavior.
    pub fn nanquantile(&self, quantile: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nanquantile(self.raw.as_ptr(), quantile, &mut raw) };
        check_status("ndarray.numpy.nanquantile", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanquantile", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns one NaN-skipping quantile across normalized axes.
    pub fn nanquantile_axes(&self, quantile: f64, axes: &[i64], keep_dims: i64, interpolation: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanquantile_axes", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nanquantile_axes(self.raw.as_ptr(), quantile, axes.as_ptr(), axes_length, keep_dims as i32, interpolation as i32, &mut raw) };
        check_status("ndarray.numpy.nanquantile_axes", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanquantile_axes", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns one NaN-skipping percentile across normalized axes.
    pub fn nanpercentile_axes(&self, percentile: f64, axes: &[i64], keep_dims: i64, interpolation: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanpercentile_axes", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nanpercentile_axes(self.raw.as_ptr(), percentile, axes.as_ptr(), axes_length, keep_dims as i32, interpolation as i32, &mut raw) };
        check_status("ndarray.numpy.nanpercentile_axes", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanpercentile_axes", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns multiple NaN-skipping quantiles with the quantile dimension first.
    pub fn nanquantiles(&self, quantiles: &[f64], axes: &[i64], keep_dims: i64, interpolation: i64) -> Result<Self, CAbiError> {
        let quantiles_length = i64::try_from(quantiles.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanquantiles", status: -2 })?;
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanquantiles", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nanquantiles(self.raw.as_ptr(), quantiles.as_ptr(), quantiles_length, axes.as_ptr(), axes_length, keep_dims as i32, interpolation as i32, &mut raw) };
        check_status("ndarray.numpy.nanquantiles", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanquantiles", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns multiple NaN-skipping percentiles with the percentile dimension first.
    pub fn nanpercentiles(&self, percentiles: &[f64], axes: &[i64], keep_dims: i64, interpolation: i64) -> Result<Self, CAbiError> {
        let percentiles_length = i64::try_from(percentiles.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanpercentiles", status: -2 })?;
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanpercentiles", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nanpercentiles(self.raw.as_ptr(), percentiles.as_ptr(), percentiles_length, axes.as_ptr(), axes_length, keep_dims as i32, interpolation as i32, &mut raw) };
        check_status("ndarray.numpy.nanpercentiles", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanpercentiles", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a NaN-skipping median across normalized axes.
    pub fn nanmedian_axes(&self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanmedian_axes", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nanmedian_axes(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.numpy.nanmedian_axes", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanmedian_axes", status: -1 })?;
        Ok(Self { raw })
    }

    /// Atomically writes a median into an exactly shaped writable destination.
    pub fn median_into(&mut self, array: &Self, axes: &[i64], keep_dims: i64) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.median_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_median_into(self.raw.as_ptr(), array.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32) };
        check_status("ndarray.numpy.median_into", status, 0)?;
        Ok(())
    }

    /// Atomically writes a NaN-skipping median into an exactly shaped writable destination.
    pub fn nanmedian_into(&mut self, array: &Self, axes: &[i64], keep_dims: i64) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanmedian_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nanmedian_into(self.raw.as_ptr(), array.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32) };
        check_status("ndarray.numpy.nanmedian_into", status, 0)?;
        Ok(())
    }

    /// Atomically writes one quantile into an exactly shaped writable destination.
    pub fn quantile_into(&mut self, array: &Self, quantile: f64, axes: &[i64], keep_dims: i64, interpolation: i64) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.quantile_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_quantile_into(self.raw.as_ptr(), array.raw.as_ptr(), quantile, axes.as_ptr(), axes_length, keep_dims as i32, interpolation as i32) };
        check_status("ndarray.numpy.quantile_into", status, 0)?;
        Ok(())
    }

    /// Atomically writes one percentile into an exactly shaped writable destination.
    pub fn percentile_into(&mut self, array: &Self, percentile: f64, axes: &[i64], keep_dims: i64, interpolation: i64) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.percentile_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_percentile_into(self.raw.as_ptr(), array.raw.as_ptr(), percentile, axes.as_ptr(), axes_length, keep_dims as i32, interpolation as i32) };
        check_status("ndarray.numpy.percentile_into", status, 0)?;
        Ok(())
    }

    /// Atomically writes multiple quantiles into an exactly shaped writable destination.
    pub fn quantiles_into(&mut self, array: &Self, quantiles: &[f64], axes: &[i64], keep_dims: i64, interpolation: i64) -> Result<(), CAbiError> {
        let quantiles_length = i64::try_from(quantiles.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.quantiles_into", status: -2 })?;
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.quantiles_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_quantiles_into(self.raw.as_ptr(), array.raw.as_ptr(), quantiles.as_ptr(), quantiles_length, axes.as_ptr(), axes_length, keep_dims as i32, interpolation as i32) };
        check_status("ndarray.numpy.quantiles_into", status, 0)?;
        Ok(())
    }

    /// Atomically writes multiple percentiles into an exactly shaped writable destination.
    pub fn percentiles_into(&mut self, array: &Self, percentiles: &[f64], axes: &[i64], keep_dims: i64, interpolation: i64) -> Result<(), CAbiError> {
        let percentiles_length = i64::try_from(percentiles.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.percentiles_into", status: -2 })?;
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.percentiles_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_percentiles_into(self.raw.as_ptr(), array.raw.as_ptr(), percentiles.as_ptr(), percentiles_length, axes.as_ptr(), axes_length, keep_dims as i32, interpolation as i32) };
        check_status("ndarray.numpy.percentiles_into", status, 0)?;
        Ok(())
    }

    /// Atomically writes one NaN-skipping quantile into an exactly shaped writable destination.
    pub fn nanquantile_into(&mut self, array: &Self, quantile: f64, axes: &[i64], keep_dims: i64, interpolation: i64) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanquantile_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nanquantile_into(self.raw.as_ptr(), array.raw.as_ptr(), quantile, axes.as_ptr(), axes_length, keep_dims as i32, interpolation as i32) };
        check_status("ndarray.numpy.nanquantile_into", status, 0)?;
        Ok(())
    }

    /// Atomically writes one NaN-skipping percentile into an exactly shaped writable destination.
    pub fn nanpercentile_into(&mut self, array: &Self, percentile: f64, axes: &[i64], keep_dims: i64, interpolation: i64) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanpercentile_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nanpercentile_into(self.raw.as_ptr(), array.raw.as_ptr(), percentile, axes.as_ptr(), axes_length, keep_dims as i32, interpolation as i32) };
        check_status("ndarray.numpy.nanpercentile_into", status, 0)?;
        Ok(())
    }

    /// Atomically writes multiple NaN-skipping quantiles into an exactly shaped writable destination.
    pub fn nanquantiles_into(&mut self, array: &Self, quantiles: &[f64], axes: &[i64], keep_dims: i64, interpolation: i64) -> Result<(), CAbiError> {
        let quantiles_length = i64::try_from(quantiles.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanquantiles_into", status: -2 })?;
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanquantiles_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nanquantiles_into(self.raw.as_ptr(), array.raw.as_ptr(), quantiles.as_ptr(), quantiles_length, axes.as_ptr(), axes_length, keep_dims as i32, interpolation as i32) };
        check_status("ndarray.numpy.nanquantiles_into", status, 0)?;
        Ok(())
    }

    /// Atomically writes multiple NaN-skipping percentiles into an exactly shaped writable destination.
    pub fn nanpercentiles_into(&mut self, array: &Self, percentiles: &[f64], axes: &[i64], keep_dims: i64, interpolation: i64) -> Result<(), CAbiError> {
        let percentiles_length = i64::try_from(percentiles.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanpercentiles_into", status: -2 })?;
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanpercentiles_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nanpercentiles_into(self.raw.as_ptr(), array.raw.as_ptr(), percentiles.as_ptr(), percentiles_length, axes.as_ptr(), axes_length, keep_dims as i32, interpolation as i32) };
        check_status("ndarray.numpy.nanpercentiles_into", status, 0)?;
        Ok(())
    }

    /// Provides NumPy-compatible packbits behavior.
    pub fn packbits(&self, bit_order: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_packbits(self.raw.as_ptr(), bit_order, &mut raw) };
        check_status("ndarray.numpy.packbits", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.packbits", status: -1 })?;
        Ok(Self { raw })
    }

    /// Provides NumPy-compatible unpackbits behavior.
    pub fn unpackbits(&self, count: i64, bit_order: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_unpackbits(self.raw.as_ptr(), count, bit_order, &mut raw) };
        check_status("ndarray.numpy.unpackbits", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.unpackbits", status: -1 })?;
        Ok(Self { raw })
    }

    /// Unpacks every available flattened UInt8 bit.
    pub fn unpackbits_all(&self, bit_order: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_unpackbits_all(self.raw.as_ptr(), bit_order, &mut raw) };
        check_status("ndarray.numpy.unpackbits_all", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.unpackbits_all", status: -1 })?;
        Ok(Self { raw })
    }

    /// Packs truth values into UInt8 bytes along one normalized axis.
    pub fn packbits_axis(&self, axis: i64, bit_order: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_packbits_axis(self.raw.as_ptr(), axis, bit_order, &mut raw) };
        check_status("ndarray.numpy.packbits_axis", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.packbits_axis", status: -1 })?;
        Ok(Self { raw })
    }

}
