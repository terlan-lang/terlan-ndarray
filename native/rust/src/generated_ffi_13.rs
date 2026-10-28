unsafe extern "C" {
        /// Invokes the raw `terlan_ndarray_v1_random_stream_geometric` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_geometric(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, probability: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_gumbel` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_gumbel(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, location: f64, scale: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_hypergeometric` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_hypergeometric(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, good: i64, bad: i64, samples: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_integers` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_integers(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, low: i64, high: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_laplace` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_laplace(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, location: f64, scale: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_logistic` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_logistic(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, location: f64, scale: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_lognormal` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_lognormal(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, mean: f64, sigma: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_logseries` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_logseries(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, probability: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_multinomial` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_multinomial(stream: *mut TerlanNdRandomStream, trials: i64, probabilities: *mut TerlanNdArray, shape: *const i64, rank: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_multivariate_hypergeometric` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_multivariate_hypergeometric(stream: *mut TerlanNdRandomStream, colors: *mut TerlanNdArray, samples: i64, shape: *const i64, rank: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_multivariate_normal` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_multivariate_normal(stream: *mut TerlanNdRandomStream, mean: *mut TerlanNdArray, covariance: *mut TerlanNdArray, shape: *const i64, rank: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_negative_binomial` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_negative_binomial(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, successes: f64, probability: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_new` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_new(seed: i64, out_stream: *mut *mut TerlanNdRandomStream) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_noncentral_chisquare` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_noncentral_chisquare(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, degrees_of_freedom: f64, noncentrality: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_noncentral_f` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_noncentral_f(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, numerator_degrees: f64, denominator_degrees: f64, noncentrality: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_normal` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_normal(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, mean: f64, standard_deviation: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_pareto` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_pareto(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, distribution_shape: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_permutation` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_permutation(stream: *mut TerlanNdRandomStream, length: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_poisson` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_poisson(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, lambda: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_power` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_power(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, distribution_shape: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_rayleigh` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_rayleigh(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, scale: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_sample_broadcast` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_sample_broadcast(stream: *mut TerlanNdRandomStream, parameter_handles: *const u64, parameter_count: i64, distribution: i64, dtype: i32, output: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_shuffle` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_shuffle(stream: *mut TerlanNdRandomStream, array: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_standard_cauchy` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_standard_cauchy(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_standard_exponential` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_standard_exponential(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_standard_exponential_into` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_standard_exponential_into(stream: *mut TerlanNdRandomStream, destination: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_standard_gamma` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_standard_gamma(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, distribution_shape: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_standard_gamma_into` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_standard_gamma_into(stream: *mut TerlanNdRandomStream, destination: *mut TerlanNdArray, distribution_shapes: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_standard_normal` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_standard_normal(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_standard_normal_into` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_standard_normal_into(stream: *mut TerlanNdRandomStream, destination: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_standard_t` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_standard_t(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, degrees_of_freedom: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_triangular` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_triangular(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, left: f64, mode: f64, right: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_uniform` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_uniform(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, low: f64, high: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_uniform_into` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_uniform_into(stream: *mut TerlanNdRandomStream, destination: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_vonmises` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_vonmises(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, mean: f64, concentration: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_wald` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_wald(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, mean: f64, scale: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_weibull` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_weibull(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, distribution_shape: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_zipf` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_zipf(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, exponent: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_uniform` C ABI symbol.
        pub fn terlan_ndarray_v1_random_uniform(shape: *const i64, rank: i64, low: f64, high: f64, seed: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_rank` C ABI symbol.
        pub fn terlan_ndarray_v1_rank(array: *mut TerlanNdArray, out_rank: *mut i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_ravel` C ABI symbol.
        pub fn terlan_ndarray_v1_ravel(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_ravel_multi_index` C ABI symbol.
        pub fn terlan_ndarray_v1_ravel_multi_index(coordinates: *mut TerlanNdArray, dimensions: *const i64, rank: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_ravel_multi_index_with` C ABI symbol.
        pub fn terlan_ndarray_v1_ravel_multi_index_with(coordinates: *mut TerlanNdArray, dimensions: *const i64, rank: i64, modes: *const i64, mode_count: i64, order: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_ravel_ordered` C ABI symbol.
        pub fn terlan_ndarray_v1_ravel_ordered(array: *mut TerlanNdArray, order: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_real` C ABI symbol.
        pub fn terlan_ndarray_v1_real(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_real_if_close` C ABI symbol.
        pub fn terlan_ndarray_v1_real_if_close(array: *mut TerlanNdArray, tolerance: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_reciprocal` C ABI symbol.
        pub fn terlan_ndarray_v1_reciprocal(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_reciprocal_into` C ABI symbol.
        pub fn terlan_ndarray_v1_reciprocal_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray) -> i32;
}
