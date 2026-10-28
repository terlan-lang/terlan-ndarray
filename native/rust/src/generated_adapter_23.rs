use super::*;

impl Array {
    /// Returns covariance after appending variables from a second observation array.
    pub fn covariance_pair(&self, other: &Self, row_variables: i64, correction: i64, dtype: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_covariance_pair(self.raw.as_ptr(), other.raw.as_ptr(), row_variables as i32, correction, dtype as i32, &mut raw) };
        check_status("ndarray.numpy.covariance_pair", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.covariance_pair", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns covariance with nonnegative integral observation frequencies.
    pub fn covariance_frequency_weighted(&self, frequency_weights: &Self, row_variables: i64, correction: i64, dtype: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_covariance_frequency_weighted(self.raw.as_ptr(), frequency_weights.raw.as_ptr(), row_variables as i32, correction, dtype as i32, &mut raw) };
        check_status("ndarray.numpy.covariance_frequency_weighted", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.covariance_frequency_weighted", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns covariance with nonnegative relative observation weights.
    pub fn covariance_analytic_weighted(&self, analytic_weights: &Self, row_variables: i64, correction: i64, dtype: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_covariance_analytic_weighted(self.raw.as_ptr(), analytic_weights.raw.as_ptr(), row_variables as i32, correction, dtype as i32, &mut raw) };
        check_status("ndarray.numpy.covariance_analytic_weighted", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.covariance_analytic_weighted", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns covariance with combined frequency and analytic observation weights.
    pub fn covariance_weighted(&self, frequency_weights: &Self, analytic_weights: &Self, row_variables: i64, correction: i64, dtype: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_covariance_weighted(self.raw.as_ptr(), frequency_weights.raw.as_ptr(), analytic_weights.raw.as_ptr(), row_variables as i32, correction, dtype as i32, &mut raw) };
        check_status("ndarray.numpy.covariance_weighted", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.covariance_weighted", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns paired covariance with combined observation weights.
    pub fn covariance_pair_weighted(&self, other: &Self, frequency_weights: &Self, analytic_weights: &Self, row_variables: i64, correction: i64, dtype: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_covariance_pair_weighted(self.raw.as_ptr(), other.raw.as_ptr(), frequency_weights.raw.as_ptr(), analytic_weights.raw.as_ptr(), row_variables as i32, correction, dtype as i32, &mut raw) };
        check_status("ndarray.numpy.covariance_pair_weighted", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.covariance_pair_weighted", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns correlation coefficients in the requested floating-point or complex dtype.
    pub fn correlation_typed(&self, row_variables: i64, dtype: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_correlation_typed(self.raw.as_ptr(), row_variables as i32, dtype as i32, &mut raw) };
        check_status("ndarray.numpy.correlation_typed", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.correlation_typed", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns correlation coefficients after appending variables from a second observation array.
    pub fn correlation_pair(&self, other: &Self, row_variables: i64, dtype: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_correlation_pair(self.raw.as_ptr(), other.raw.as_ptr(), row_variables as i32, dtype as i32, &mut raw) };
        check_status("ndarray.numpy.correlation_pair", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.correlation_pair", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns sums while omitting NaN values.
    pub fn nansum(&self, axes: &[i64], keep_dims: bool) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nansum", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_sum(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.numpy.nansum", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nansum", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns products while omitting NaN values.
    pub fn nanprod(&self, axes: &[i64], keep_dims: bool) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanprod", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_product(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.numpy.nanprod", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanprod", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns means while omitting NaN values.
    pub fn nanmean(&self, axes: &[i64], keep_dims: bool) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanmean", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_mean(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.numpy.nanmean", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanmean", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns minima while omitting NaN values.
    pub fn nanmin(&self, axes: &[i64], keep_dims: bool) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanmin", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_minimum(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.numpy.nanmin", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanmin", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns maxima while omitting NaN values.
    pub fn nanmax(&self, axes: &[i64], keep_dims: bool) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanmax", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_maximum(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.numpy.nanmax", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanmax", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns variance while omitting NaN values.
    pub fn nanvar(&self, axes: &[i64], keep_dims: bool, correction: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanvar", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_variance(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, correction, &mut raw) };
        check_status("ndarray.numpy.nanvar", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanvar", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns standard deviation while omitting NaN values.
    pub fn nanstd(&self, axes: &[i64], keep_dims: bool, correction: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanstd", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_standard_deviation(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, correction, &mut raw) };
        check_status("ndarray.numpy.nanstd", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanstd", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns NaN-omitting sums in an explicit dtype.
    pub fn nansum_as(&self, axes: &[i64], keep_dims: bool, dtype: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nansum_as", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_sum_as(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, dtype as i32, &mut raw) };
        check_status("ndarray.numpy.nansum_as", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nansum_as", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns NaN-omitting products in an explicit dtype.
    pub fn nanprod_as(&self, axes: &[i64], keep_dims: bool, dtype: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanprod_as", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_product_as(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, dtype as i32, &mut raw) };
        check_status("ndarray.numpy.nanprod_as", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanprod_as", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns NaN-omitting means in an explicit inexact dtype.
    pub fn nanmean_as(&self, axes: &[i64], keep_dims: bool, dtype: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanmean_as", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_mean_as(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, dtype as i32, &mut raw) };
        check_status("ndarray.numpy.nanmean_as", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanmean_as", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns NaN-omitting variances in an explicit inexact dtype.
    pub fn nanvar_as(&self, axes: &[i64], keep_dims: bool, correction: i64, dtype: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanvar_as", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_variance_as(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, correction, dtype as i32, &mut raw) };
        check_status("ndarray.numpy.nanvar_as", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanvar_as", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns NaN-omitting standard deviations in an explicit inexact dtype.
    pub fn nanstd_as(&self, axes: &[i64], keep_dims: bool, correction: i64, dtype: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanstd_as", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_standard_deviation_as(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, correction, dtype as i32, &mut raw) };
        check_status("ndarray.numpy.nanstd_as", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanstd_as", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns minimum indexes over normalized axes while omitting NaNs.
    pub fn nanargmin_axes(&self, axes: &[i64], keep_dims: bool) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanargmin_axes", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_argmin(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.numpy.nanargmin_axes", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanargmin_axes", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns maximum indexes over normalized axes while omitting NaNs.
    pub fn nanargmax_axes(&self, axes: &[i64], keep_dims: bool) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanargmax_axes", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_argmax(self.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.numpy.nanargmax_axes", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.numpy.nanargmax_axes", status: -1 })?;
        Ok(Self { raw })
    }

    /// Writes NaN-omitting sums into an exact-shape destination.
    pub fn nansum_into(&mut self, array: &Self, axes: &[i64], keep_dims: bool) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nansum_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_sum_into(self.raw.as_ptr(), array.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32) };
        check_status("ndarray.numpy.nansum_into", status, 0)?;
        Ok(())
    }

    /// Writes NaN-omitting products into an exact-shape destination.
    pub fn nanprod_into(&mut self, array: &Self, axes: &[i64], keep_dims: bool) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanprod_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_product_into(self.raw.as_ptr(), array.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32) };
        check_status("ndarray.numpy.nanprod_into", status, 0)?;
        Ok(())
    }

    /// Writes NaN-omitting means into an exact-shape inexact destination.
    pub fn nanmean_into(&mut self, array: &Self, axes: &[i64], keep_dims: bool) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.numpy.nanmean_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nan_mean_into(self.raw.as_ptr(), array.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32) };
        check_status("ndarray.numpy.nanmean_into", status, 0)?;
        Ok(())
    }

}
