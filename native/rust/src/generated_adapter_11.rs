use super::*;

impl Array {
    /// Counts flattened values into equal-width bins over an explicit finite range.
    pub fn histogram(&self, bins: i64, minimum: f64, maximum: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_histogram(self.raw.as_ptr(), bins, minimum, maximum, &mut raw) };
        check_status("ndarray.array.histogram", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.histogram", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns equal-width Float64 histogram edges for an explicit finite range.
    pub fn histogram_bin_edges(&self, bins: i64, minimum: f64, maximum: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_histogram_bin_edges(self.raw.as_ptr(), bins, minimum, maximum, &mut raw) };
        check_status("ndarray.array.histogram_bin_edges", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.histogram_bin_edges", status: -1 })?;
        Ok(Self { raw })
    }

    /// Counts a [sample, dimension] matrix into an equal-width multidimensional histogram.
    pub fn histogramdd(&self, bins: i64, minima: &Self, maxima: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_histogramdd(self.raw.as_ptr(), bins, minima.raw.as_ptr(), maxima.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.histogramdd", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.histogramdd", status: -1 })?;
        Ok(Self { raw })
    }

    /// Accumulates finite weights by non-negative integer value.
    pub fn bincount(&self, weights: &Self, minimum_length: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_bincount(self.raw.as_ptr(), weights.raw.as_ptr(), minimum_length, &mut raw) };
        check_status("ndarray.array.bincount", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.bincount", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns insertion-bin indexes against strictly increasing rank-one edges.
    pub fn digitize(&self, bins: &Self, right: bool) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_digitize(self.raw.as_ptr(), bins.raw.as_ptr(), right as i32, &mut raw) };
        check_status("ndarray.array.digitize", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.digitize", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns repeated forward differences along one normalized axis.
    pub fn difference(&self, order: i64, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_difference(self.raw.as_ptr(), order, axis, &mut raw) };
        check_status("ndarray.array.difference", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.difference", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns central interior and first-order edge gradients for uniform spacing.
    pub fn gradient(&self, spacing: f64, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_gradient(self.raw.as_ptr(), spacing, axis, &mut raw) };
        check_status("ndarray.array.gradient", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.gradient", status: -1 })?;
        Ok(Self { raw })
    }

    /// Unwraps real periodic samples along one normalized axis.
    pub fn unwrap(&self, period: f64, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_unwrap(self.raw.as_ptr(), period, axis, &mut raw) };
        check_status("ndarray.array.unwrap", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.unwrap", status: -1 })?;
        Ok(Self { raw })
    }

    /// Linearly interpolates real values over strictly increasing rank-one coordinates.
    pub fn interpolate(&self, coordinates: &Self, samples: &Self, left: f64, right: f64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_interpolate(self.raw.as_ptr(), coordinates.raw.as_ptr(), samples.raw.as_ptr(), left, right, &mut raw) };
        check_status("ndarray.array.interpolate", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.interpolate", status: -1 })?;
        Ok(Self { raw })
    }

    /// Integrates along one axis using explicit rank-one coordinates and the trapezoidal rule.
    pub fn trapezoidal_integral(&self, coordinates: &Self, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_trapezoidal_integral(self.raw.as_ptr(), coordinates.raw.as_ptr(), axis, &mut raw) };
        check_status("ndarray.array.trapezoidal_integral", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.trapezoidal_integral", status: -1 })?;
        Ok(Self { raw })
    }

    /// Integrates along one axis using uniform finite spacing and the trapezoidal rule.
    pub fn trapezoidal_integral_uniform(&self, spacing: f64, axis: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_trapezoidal_integral_uniform(self.raw.as_ptr(), spacing, axis, &mut raw) };
        check_status("ndarray.array.trapezoidal_integral_uniform", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.trapezoidal_integral_uniform", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a complex FFT along one normalized axis.
    pub fn fft(&self, axis: i64, normalization: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_fft(self.raw.as_ptr(), axis, normalization as i32, &mut raw) };
        check_status("ndarray.array.fft", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.fft", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns a complex FFT after truncating or zero-padding one normalized axis to an explicit length.
    pub fn fft_sized(&self, length: i64, axis: i64, normalization: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_fft_sized(self.raw.as_ptr(), length, axis, normalization as i32, &mut raw) };
        check_status("ndarray.array.fft_sized", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.fft_sized", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns an inverse complex FFT along one normalized axis.
    pub fn ifft(&self, axis: i64, normalization: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_ifft(self.raw.as_ptr(), axis, normalization as i32, &mut raw) };
        check_status("ndarray.array.ifft", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.ifft", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns an inverse complex FFT after truncating or zero-padding one normalized axis to an explicit length.
    pub fn ifft_sized(&self, length: i64, axis: i64, normalization: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_ifft_sized(self.raw.as_ptr(), length, axis, normalization as i32, &mut raw) };
        check_status("ndarray.array.ifft_sized", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.ifft_sized", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns complex FFTs across normalized distinct axes.
    pub fn fftn(&self, axes: &[i64], normalization: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.fftn", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_fftn(self.raw.as_ptr(), axes.as_ptr(), axes_length, normalization as i32, &mut raw) };
        check_status("ndarray.array.fftn", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.fftn", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns inverse complex FFTs across normalized distinct axes.
    pub fn ifftn(&self, axes: &[i64], normalization: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.ifftn", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_ifftn(self.raw.as_ptr(), axes.as_ptr(), axes_length, normalization as i32, &mut raw) };
        check_status("ndarray.array.ifftn", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.ifftn", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns non-negative bins from a real-input FFT.
    pub fn rfft(&self, axis: i64, normalization: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_rfft(self.raw.as_ptr(), axis, normalization as i32, &mut raw) };
        check_status("ndarray.array.rfft", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.rfft", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns non-negative bins after truncating or zero-padding a real-input axis to an explicit length.
    pub fn rfft_sized(&self, length: i64, axis: i64, normalization: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_rfft_sized(self.raw.as_ptr(), length, axis, normalization as i32, &mut raw) };
        check_status("ndarray.array.rfft_sized", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.rfft_sized", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reconstructs a real signal of explicit length from compact frequency bins.
    pub fn irfft(&self, length: i64, axis: i64, normalization: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_irfft(self.raw.as_ptr(), length, axis, normalization as i32, &mut raw) };
        check_status("ndarray.array.irfft", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.irfft", status: -1 })?;
        Ok(Self { raw })
    }

    /// Returns compact real-input FFTs across normalized distinct axes.
    pub fn rfftn(&self, axes: &[i64], normalization: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.rfftn", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_rfftn(self.raw.as_ptr(), axes.as_ptr(), axes_length, normalization as i32, &mut raw) };
        check_status("ndarray.array.rfftn", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.rfftn", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reconstructs a real array from compact N-dimensional frequency bins.
    pub fn irfftn(&self, length: i64, axes: &[i64], normalization: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.irfftn", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_irfftn(self.raw.as_ptr(), length, axes.as_ptr(), axes_length, normalization as i32, &mut raw) };
        check_status("ndarray.array.irfftn", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.irfftn", status: -1 })?;
        Ok(Self { raw })
    }

    /// Transforms a compact Hermitian signal to a real spectrum.
    pub fn hfft(&self, length: i64, axis: i64, normalization: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_hfft(self.raw.as_ptr(), length, axis, normalization as i32, &mut raw) };
        check_status("ndarray.array.hfft", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.hfft", status: -1 })?;
        Ok(Self { raw })
    }

    /// Transforms a real spectrum to a compact Hermitian signal.
    pub fn ihfft(&self, axis: i64, normalization: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_ihfft(self.raw.as_ptr(), axis, normalization as i32, &mut raw) };
        check_status("ndarray.array.ihfft", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.ihfft", status: -1 })?;
        Ok(Self { raw })
    }

}
