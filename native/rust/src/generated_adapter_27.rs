use super::*;

impl RandomStream {
    /// Advances the stream while sampling NumPy's unit-scale Weibull distribution.
    pub fn weibull(&mut self, shape: &[i64], distribution_shape: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.weibull", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_weibull(self.raw.as_ptr(), shape.as_ptr(), shape_length, distribution_shape, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.weibull", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.weibull", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling one-based geometric trial counts.
    pub fn geometric(&mut self, shape: &[i64], probability: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.geometric", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_geometric(self.raw.as_ptr(), shape.as_ptr(), shape_length, probability, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.geometric", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.geometric", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling hypergeometric success counts.
    pub fn hypergeometric(&mut self, shape: &[i64], good: i64, bad: i64, samples: i64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.hypergeometric", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_hypergeometric(self.raw.as_ptr(), shape.as_ptr(), shape_length, good, bad, samples, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.hypergeometric", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.hypergeometric", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling Zipf counts through a zeta distribution.
    pub fn zipf(&mut self, shape: &[i64], exponent: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.zipf", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_zipf(self.raw.as_ptr(), shape.as_ptr(), shape_length, exponent, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.zipf", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.zipf", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling a standard Cauchy distribution.
    pub fn standard_cauchy(&mut self, shape: &[i64], dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.standard_cauchy", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_standard_cauchy(self.raw.as_ptr(), shape.as_ptr(), shape_length, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.standard_cauchy", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.standard_cauchy", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling a Laplace distribution.
    pub fn laplace(&mut self, shape: &[i64], location: f64, scale: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.laplace", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_laplace(self.raw.as_ptr(), shape.as_ptr(), shape_length, location, scale, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.laplace", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.laplace", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling a logistic distribution.
    pub fn logistic(&mut self, shape: &[i64], location: f64, scale: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.logistic", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_logistic(self.raw.as_ptr(), shape.as_ptr(), shape_length, location, scale, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.logistic", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.logistic", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling negative-binomial failure counts.
    pub fn negative_binomial(&mut self, shape: &[i64], successes: f64, probability: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.negative_binomial", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_negative_binomial(self.raw.as_ptr(), shape.as_ptr(), shape_length, successes, probability, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.negative_binomial", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.negative_binomial", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling a noncentral chi-square distribution.
    pub fn noncentral_chisquare(&mut self, shape: &[i64], degrees_of_freedom: f64, noncentrality: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.noncentral_chisquare", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_noncentral_chisquare(self.raw.as_ptr(), shape.as_ptr(), shape_length, degrees_of_freedom, noncentrality, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.noncentral_chisquare", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.noncentral_chisquare", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling a noncentral Fisher F distribution.
    pub fn noncentral_f(&mut self, shape: &[i64], numerator_degrees: f64, denominator_degrees: f64, noncentrality: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.noncentral_f", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_noncentral_f(self.raw.as_ptr(), shape.as_ptr(), shape_length, numerator_degrees, denominator_degrees, noncentrality, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.noncentral_f", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.noncentral_f", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling a Rayleigh distribution.
    pub fn rayleigh(&mut self, shape: &[i64], scale: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.rayleigh", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_rayleigh(self.raw.as_ptr(), shape.as_ptr(), shape_length, scale, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.rayleigh", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.rayleigh", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling a Wald distribution.
    pub fn wald(&mut self, shape: &[i64], mean: f64, scale: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.wald", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_wald(self.raw.as_ptr(), shape.as_ptr(), shape_length, mean, scale, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.wald", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.wald", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling positive logarithmic-series counts.
    pub fn logseries(&mut self, shape: &[i64], probability: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.logseries", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_logseries(self.raw.as_ptr(), shape.as_ptr(), shape_length, probability, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.logseries", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.logseries", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while sampling circular von Mises values.
    pub fn vonmises(&mut self, shape: &[i64], mean: f64, concentration: f64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.vonmises", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_vonmises(self.raw.as_ptr(), shape.as_ptr(), shape_length, mean, concentration, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.vonmises", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.vonmises", status: -1 })?;
        Ok(Array { raw })
    }

    /// Samples rank-one source values with weighted replacement policy.
    pub fn choice(&mut self, source: &Array, shape: &[i64], probabilities: &Array, replacement: bool) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.choice", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_choice(self.raw.as_ptr(), source.raw.as_ptr(), shape.as_ptr(), shape_length, probabilities.raw.as_ptr(), replacement as i32, &mut raw) };
        check_status("ndarray.random_stream.choice", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.choice", status: -1 })?;
        Ok(Array { raw })
    }

    /// Samples source values along one axis with explicit output-order policy.
    pub fn sample_choice_axis(&mut self, source: &Array, shape: &[i64], probabilities: &Array, replacement: bool, axis: i64, shuffle: bool) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.choice_axis", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_choice_axis(self.raw.as_ptr(), source.raw.as_ptr(), shape.as_ptr(), shape_length, probabilities.raw.as_ptr(), replacement as i32, axis, shuffle as i32, &mut raw) };
        check_status("ndarray.random_stream.choice_axis", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.choice_axis", status: -1 })?;
        Ok(Array { raw })
    }

    /// Samples uniformly along one source axis without a probability array.
    pub fn sample_choice_uniform_axis(&mut self, source: &Array, shape: &[i64], replacement: bool, axis: i64, shuffle: bool) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.choice_uniform_axis", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_choice_uniform_axis(self.raw.as_ptr(), source.raw.as_ptr(), shape.as_ptr(), shape_length, replacement as i32, axis, shuffle as i32, &mut raw) };
        check_status("ndarray.random_stream.choice_uniform_axis", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.choice_uniform_axis", status: -1 })?;
        Ok(Array { raw })
    }

    /// Samples simplex vectors parameterized by rank-one positive concentrations.
    pub fn dirichlet(&mut self, alpha: &Array, shape: &[i64], dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.dirichlet", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_dirichlet(self.raw.as_ptr(), alpha.raw.as_ptr(), shape.as_ptr(), shape_length, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.dirichlet", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.dirichlet", status: -1 })?;
        Ok(Array { raw })
    }

    /// Samples category counts whose final axis sums to the trial count.
    pub fn multinomial(&mut self, trials: i64, probabilities: &Array, shape: &[i64], dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.multinomial", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_multinomial(self.raw.as_ptr(), trials, probabilities.raw.as_ptr(), shape.as_ptr(), shape_length, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.multinomial", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.multinomial", status: -1 })?;
        Ok(Array { raw })
    }

    /// Samples category counts without replacement from a finite population.
    pub fn multivariate_hypergeometric(&mut self, colors: &Array, samples: i64, shape: &[i64], dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.multivariate_hypergeometric", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_multivariate_hypergeometric(self.raw.as_ptr(), colors.raw.as_ptr(), samples, shape.as_ptr(), shape_length, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.multivariate_hypergeometric", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.multivariate_hypergeometric", status: -1 })?;
        Ok(Array { raw })
    }

    /// Samples correlated vectors from a positive-definite covariance matrix.
    pub fn multivariate_normal(&mut self, mean: &Array, covariance: &Array, shape: &[i64], dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.multivariate_normal", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_multivariate_normal(self.raw.as_ptr(), mean.raw.as_ptr(), covariance.raw.as_ptr(), shape.as_ptr(), shape_length, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.multivariate_normal", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.multivariate_normal", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while creating bounded integer samples.
    pub fn integers(&mut self, shape: &[i64], low: i64, high: i64, dtype: i64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.integers", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_integers(self.raw.as_ptr(), shape.as_ptr(), shape_length, low, high, dtype as i32, &mut raw) };
        check_status("ndarray.random_stream.integers", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.integers", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while creating Bernoulli samples.
    pub fn bernoulli(&mut self, shape: &[i64], probability: f64) -> Result<Array, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.random_stream.bernoulli", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_bernoulli(self.raw.as_ptr(), shape.as_ptr(), shape_length, probability, &mut raw) };
        check_status("ndarray.random_stream.bernoulli", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.bernoulli", status: -1 })?;
        Ok(Array { raw })
    }

    /// Advances the stream while creating a permutation of consecutive integers.
    pub fn permutation(&mut self, length: i64) -> Result<Array, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_random_stream_permutation(self.raw.as_ptr(), length, &mut raw) };
        check_status("ndarray.random_stream.permutation", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.random_stream.permutation", status: -1 })?;
        Ok(Array { raw })
    }

}
