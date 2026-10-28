use super::*;

impl Array {
    /// Reduces normalized axes using variance with an explicit non-negative correction.
    pub fn variance(&self, axes: &[i64], keep_dims: i64, correction: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.variance", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_variance(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, correction, &mut raw) };
        check_status("ndarray.array.variance", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.variance", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reduces normalized axes using standard deviation with an explicit non-negative correction.
    pub fn standard_deviation(&self, axes: &[i64], keep_dims: i64, correction: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.standard_deviation", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_standard_deviation(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, correction, &mut raw) };
        check_status("ndarray.array.standard_deviation", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.standard_deviation", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns variances over elements selected by a broadcast-compatible truth mask.
    pub fn variance_where(&self, mask: &Self, axes: &[i64], keep_dims: i64, correction: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.variance_where", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_variance_where(self.raw.as_ptr(), mask.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, correction, &mut raw) };
        check_status("ndarray.array.variance_where", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.variance_where", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns variances using an explicit result dtype.
    pub fn variance_as(&self, axes: &[i64], keep_dims: i64, correction: i64, dtype: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.variance_as", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_variance_as(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, correction, dtype as i32, &mut raw) };
        check_status("ndarray.array.variance_as", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.variance_as", status: -1 })?;
        Ok(Self { raw })
    }

    /// Writes variances into an exactly shaped writable destination.
    pub fn variance_into(&mut self, array: &Self, axes: &[i64], keep_dims: i64, correction: i64) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.variance_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_variance_into(self.raw.as_ptr(), array.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, correction) };
        check_status("ndarray.array.variance_into", status, 0)?;
        Ok(())
    }

    /// Returns standard deviations over elements selected by a broadcast-compatible truth mask.
    pub fn standard_deviation_where(&self, mask: &Self, axes: &[i64], keep_dims: i64, correction: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.standard_deviation_where", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_standard_deviation_where(self.raw.as_ptr(), mask.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, correction, &mut raw) };
        check_status("ndarray.array.standard_deviation_where", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.standard_deviation_where", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns standard deviations using an explicit result dtype.
    pub fn standard_deviation_as(&self, axes: &[i64], keep_dims: i64, correction: i64, dtype: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.standard_deviation_as", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_standard_deviation_as(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, correction, dtype as i32, &mut raw) };
        check_status("ndarray.array.standard_deviation_as", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.standard_deviation_as", status: -1 })?;
        Ok(Self { raw })
    }

    /// Writes standard deviations into an exactly shaped writable destination.
    pub fn standard_deviation_into(&mut self, array: &Self, axes: &[i64], keep_dims: i64, correction: i64) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.standard_deviation_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_standard_deviation_into(self.raw.as_ptr(), array.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, correction) };
        check_status("ndarray.array.standard_deviation_into", status, 0)?;
        Ok(())
    }

    /// Returns the linear-interpolated median across normalized axes.
    pub fn median(&self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.median", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_median(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.median", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.median", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns an axis reduction at a unit-interval quantile using an explicit interpolation policy.
    pub fn quantile(&self, quantile: f64, axes: &[i64], keep_dims: i64, interpolation: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.quantile", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_quantile(self.raw.as_ptr(), quantile, axes.as_ptr(), axes_length, keep_dims as i32, interpolation as i32, &mut raw) };
        check_status("ndarray.array.quantile", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.quantile", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns an axis reduction at a percentage quantile using an explicit interpolation policy.
    pub fn percentile(&self, percentile: f64, axes: &[i64], keep_dims: i64, interpolation: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.percentile", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_percentile(self.raw.as_ptr(), percentile, axes.as_ptr(), axes_length, keep_dims as i32, interpolation as i32, &mut raw) };
        check_status("ndarray.array.percentile", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.percentile", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns multiple unit-interval quantiles with the quantile dimension first.
    pub fn quantiles(&self, quantiles: &[f64], axes: &[i64], keep_dims: i64, interpolation: i64) -> Result<Self, CAbiError> {
        let quantiles_length = i64::try_from(quantiles.len()).map_err(|_| CAbiError { operation: "ndarray.array.quantiles", status: -2 })?;
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.quantiles", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_quantiles(self.raw.as_ptr(), quantiles.as_ptr(), quantiles_length, axes.as_ptr(), axes_length, keep_dims as i32, interpolation as i32, &mut raw) };
        check_status("ndarray.array.quantiles", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.quantiles", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns multiple percentage quantiles with the percentile dimension first.
    pub fn percentiles(&self, percentiles: &[f64], axes: &[i64], keep_dims: i64, interpolation: i64) -> Result<Self, CAbiError> {
        let percentiles_length = i64::try_from(percentiles.len()).map_err(|_| CAbiError { operation: "ndarray.array.percentiles", status: -2 })?;
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.percentiles", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_percentiles(self.raw.as_ptr(), percentiles.as_ptr(), percentiles_length, axes.as_ptr(), axes_length, keep_dims as i32, interpolation as i32, &mut raw) };
        check_status("ndarray.array.percentiles", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.percentiles", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the maximum minus minimum across normalized axes.
    pub fn peak_to_peak(&self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.peak_to_peak", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_peak_to_peak(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.peak_to_peak", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.peak_to_peak", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns an axis average weighted by a same-shape array or matching rank-one vector.
    pub fn weighted_average(&self, weights: &Self, axis: i64, keep_dims: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_weighted_average(self.raw.as_ptr(), weights.raw.as_ptr(), axis, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.weighted_average", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.weighted_average", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a covariance matrix with explicit variable orientation and degrees-of-freedom correction.
    pub fn covariance(&self, row_variables: i64, correction: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_covariance(self.raw.as_ptr(), row_variables as i32, correction, &mut raw) };
        check_status("ndarray.array.covariance", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.covariance", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns the normalized covariance matrix for non-constant variables.
    pub fn correlation(&self, row_variables: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_correlation(self.raw.as_ptr(), row_variables as i32, &mut raw) };
        check_status("ndarray.array.correlation", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.correlation", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns sums while omitting NaN values.
    pub fn nan_sum(&self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.nan_sum", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_sum(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.nan_sum", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.nan_sum", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns products while omitting NaN values.
    pub fn nan_product(&self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.nan_product", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_product(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.nan_product", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.nan_product", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns means while omitting NaN values.
    pub fn nan_mean(&self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.nan_mean", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_mean(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.nan_mean", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.nan_mean", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns minima while omitting NaN values.
    pub fn nan_minimum(&self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.nan_minimum", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_minimum(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.nan_minimum", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.nan_minimum", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns maxima while omitting NaN values.
    pub fn nan_maximum(&self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.nan_maximum", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_maximum(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.nan_maximum", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.nan_maximum", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns variance while omitting NaN values.
    pub fn nan_variance(&self, axes: &[i64], keep_dims: i64, correction: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.nan_variance", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_variance(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, correction, &mut raw) };
        check_status("ndarray.array.nan_variance", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.nan_variance", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns standard deviation while omitting NaN values.
    pub fn nan_standard_deviation(&self, axes: &[i64], keep_dims: i64, correction: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.nan_standard_deviation", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_standard_deviation(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, correction, &mut raw) };
        check_status("ndarray.array.nan_standard_deviation", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.nan_standard_deviation", status: -1 })?;
        Ok(Self { raw })
    }

}
