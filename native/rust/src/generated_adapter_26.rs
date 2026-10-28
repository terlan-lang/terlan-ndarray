use super::*;

impl RandomStream {
    /// Creates a deterministic ChaCha8 stream at an explicit seed.
    pub fn new(seed: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdRandomStream = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_new(seed, &mut raw) };
        check_status("ndarray.random_stream.new", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.new", status: -1 })?;
        Ok(Self { raw })
    }

    /// Samples one package-private broadcast distribution family.
    pub fn sample_broadcast(&mut self, parameter_handles: &[&Array], distribution: i64, dtype: i64) -> Result<Array, CAbiError> {
        let parameter_handles_handles = parameter_handles.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
        let parameter_handles_length = i64::try_from(parameter_handles.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.sample_broadcast", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_sample_broadcast(self.raw.as_ptr(), parameter_handles_handles.as_ptr(), parameter_handles_length, distribution, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.sample_broadcast", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.sample_broadcast", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while creating uniform floating samples.
    pub fn uniform(&mut self, shape: &[i64], low: f64, high: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.uniform", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_uniform(self.raw.as_ptr(), shape.as_ptr(), shape_length, low, high, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.uniform", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.uniform", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while atomically filling a C-contiguous floating destination with uniform samples in the half-open interval from zero through one.
    pub fn uniform_into(&mut self, destination: &Array) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_uniform_into(self.raw.as_ptr(), destination.raw.as_ptr()) };
        check_status("ndarray.random_stream.uniform_into", status, 0)?;
        Ok(())
    }

    /// Advances the stream while creating normally distributed floating samples.
    pub fn normal(&mut self, shape: &[i64], mean: f64, standard_deviation: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.normal", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_normal(self.raw.as_ptr(), shape.as_ptr(), shape_length, mean, standard_deviation, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.normal", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.normal", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling a beta distribution.
    pub fn beta(&mut self, shape: &[i64], alpha: f64, beta: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.beta", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_beta(self.raw.as_ptr(), shape.as_ptr(), shape_length, alpha, beta, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.beta", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.beta", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling a gamma distribution.
    pub fn gamma(&mut self, shape: &[i64], distribution_shape: f64, scale: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.gamma", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_gamma(self.raw.as_ptr(), shape.as_ptr(), shape_length, distribution_shape, scale, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.gamma", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.gamma", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling binomial counts.
    pub fn binomial(&mut self, shape: &[i64], trials: i64, probability: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.binomial", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_binomial(self.raw.as_ptr(), shape.as_ptr(), shape_length, trials, probability, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.binomial", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.binomial", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling Poisson counts.
    pub fn poisson(&mut self, shape: &[i64], lambda: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.poisson", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_poisson(self.raw.as_ptr(), shape.as_ptr(), shape_length, lambda, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.poisson", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.poisson", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling a standard normal distribution.
    pub fn standard_normal(&mut self, shape: &[i64], dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.standard_normal", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_standard_normal(self.raw.as_ptr(), shape.as_ptr(), shape_length, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.standard_normal", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.standard_normal", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while atomically filling a C-contiguous floating destination with standard normal samples.
    pub fn standard_normal_into(&mut self, destination: &Array) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_standard_normal_into(self.raw.as_ptr(), destination.raw.as_ptr()) };
        check_status("ndarray.random_stream.standard_normal_into", status, 0)?;
        Ok(())
    }

    /// Advances the stream while sampling a chi-square distribution.
    pub fn chisquare(&mut self, shape: &[i64], degrees_of_freedom: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.chisquare", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_chisquare(self.raw.as_ptr(), shape.as_ptr(), shape_length, degrees_of_freedom, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.chisquare", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.chisquare", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling a Fisher F distribution.
    pub fn f(&mut self, shape: &[i64], numerator_degrees: f64, denominator_degrees: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.f", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_f(self.raw.as_ptr(), shape.as_ptr(), shape_length, numerator_degrees, denominator_degrees, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.f", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.f", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling a Gumbel distribution.
    pub fn gumbel(&mut self, shape: &[i64], location: f64, scale: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.gumbel", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_gumbel(self.raw.as_ptr(), shape.as_ptr(), shape_length, location, scale, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.gumbel", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.gumbel", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling a log-normal distribution.
    pub fn lognormal(&mut self, shape: &[i64], mean: f64, sigma: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.lognormal", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_lognormal(self.raw.as_ptr(), shape.as_ptr(), shape_length, mean, sigma, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.lognormal", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.lognormal", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling NumPy's zero-based Pareto distribution.
    pub fn pareto(&mut self, shape: &[i64], distribution_shape: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.pareto", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_pareto(self.raw.as_ptr(), shape.as_ptr(), shape_length, distribution_shape, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.pareto", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.pareto", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling NumPy's power distribution.
    pub fn power(&mut self, shape: &[i64], distribution_shape: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.power", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_power(self.raw.as_ptr(), shape.as_ptr(), shape_length, distribution_shape, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.power", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.power", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling an exponential distribution parameterized by scale.
    pub fn exponential(&mut self, shape: &[i64], scale: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.exponential", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_exponential(self.raw.as_ptr(), shape.as_ptr(), shape_length, scale, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.exponential", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.exponential", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling a unit-scale exponential distribution.
    pub fn standard_exponential(&mut self, shape: &[i64], dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.standard_exponential", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_standard_exponential(self.raw.as_ptr(), shape.as_ptr(), shape_length, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.standard_exponential", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.standard_exponential", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while atomically filling a C-contiguous floating destination with unit-scale exponential samples.
    pub fn standard_exponential_into(&mut self, destination: &Array) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_standard_exponential_into(self.raw.as_ptr(), destination.raw.as_ptr()) };
        check_status("ndarray.random_stream.standard_exponential_into", status, 0)?;
        Ok(())
    }

    /// Advances the stream while sampling a unit-scale gamma distribution.
    pub fn standard_gamma(&mut self, shape: &[i64], distribution_shape: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.standard_gamma", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_standard_gamma(self.raw.as_ptr(), shape.as_ptr(), shape_length, distribution_shape, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.standard_gamma", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.standard_gamma", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while atomically filling a C-contiguous floating destination with unit-scale gamma samples using broadcast distribution-shape parameters.
    pub fn standard_gamma_into(&mut self, destination: &Array, distribution_shapes: &Array) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_standard_gamma_into(self.raw.as_ptr(), destination.raw.as_ptr(), distribution_shapes.raw.as_ptr()) };
        check_status("ndarray.random_stream.standard_gamma_into", status, 0)?;
        Ok(())
    }

    /// Advances the stream while sampling a standard Student t distribution.
    pub fn standard_t(&mut self, shape: &[i64], degrees_of_freedom: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.standard_t", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_standard_t(self.raw.as_ptr(), shape.as_ptr(), shape_length, degrees_of_freedom, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.standard_t", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.standard_t", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling a bounded triangular distribution.
    pub fn triangular(&mut self, shape: &[i64], left: f64, mode: f64, right: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.triangular", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_triangular(self.raw.as_ptr(), shape.as_ptr(), shape_length, left, mode, right, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.triangular", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.triangular", status: -1 })?;
        Ok(Array { raw })
    }

}
