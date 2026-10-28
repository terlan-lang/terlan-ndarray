unsafe extern "C" {
        /// Invokes the raw `terlan_ndarray_v1_polyint_axis` C ABI symbol.
        pub fn terlan_ndarray_v1_polyint_axis(coefficients: *mut TerlanNdArray, order: i64, constants: *mut TerlanNdArray, lower_bound: f64, scale: f64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyline` C ABI symbol.
        pub fn terlan_ndarray_v1_polyline(offset: f64, scale: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polymul` C ABI symbol.
        pub fn terlan_ndarray_v1_polymul(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polymulx` C ABI symbol.
        pub fn terlan_ndarray_v1_polymulx(coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polypow` C ABI symbol.
        pub fn terlan_ndarray_v1_polypow(coefficients: *mut TerlanNdArray, exponent: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyroots` C ABI symbol.
        pub fn terlan_ndarray_v1_polyroots(coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polysub` C ABI symbol.
        pub fn terlan_ndarray_v1_polysub(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polytrim` C ABI symbol.
        pub fn terlan_ndarray_v1_polytrim(coefficients: *mut TerlanNdArray, tolerance: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyval` C ABI symbol.
        pub fn terlan_ndarray_v1_polyval(coordinates: *mut TerlanNdArray, coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyval2d` C ABI symbol.
        pub fn terlan_ndarray_v1_polyval2d(x: *mut TerlanNdArray, y: *mut TerlanNdArray, coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyval3d` C ABI symbol.
        pub fn terlan_ndarray_v1_polyval3d(x: *mut TerlanNdArray, y: *mut TerlanNdArray, z: *mut TerlanNdArray, coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyvalfromroots` C ABI symbol.
        pub fn terlan_ndarray_v1_polyvalfromroots(coordinates: *mut TerlanNdArray, roots: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyvalfromroots_tensor` C ABI symbol.
        pub fn terlan_ndarray_v1_polyvalfromroots_tensor(coordinates: *mut TerlanNdArray, roots: *mut TerlanNdArray, tensor: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyvander` C ABI symbol.
        pub fn terlan_ndarray_v1_polyvander(coordinates: *mut TerlanNdArray, degree: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyvander2d` C ABI symbol.
        pub fn terlan_ndarray_v1_polyvander2d(x: *mut TerlanNdArray, y: *mut TerlanNdArray, x_degree: i64, y_degree: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyvander3d` C ABI symbol.
        pub fn terlan_ndarray_v1_polyvander3d(x: *mut TerlanNdArray, y: *mut TerlanNdArray, z: *mut TerlanNdArray, x_degree: i64, y_degree: i64, z_degree: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_positive` C ABI symbol.
        pub fn terlan_ndarray_v1_positive(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_power` C ABI symbol.
        pub fn terlan_ndarray_v1_power(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_product` C ABI symbol.
        pub fn terlan_ndarray_v1_product(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_product_into` C ABI symbol.
        pub fn terlan_ndarray_v1_product_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_put` C ABI symbol.
        pub fn terlan_ndarray_v1_put(array: *mut TerlanNdArray, indices: *const i64, index_count: i64, values: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_put_along_axis` C ABI symbol.
        pub fn terlan_ndarray_v1_put_along_axis(array: *mut TerlanNdArray, indices: *mut TerlanNdArray, values: *mut TerlanNdArray, axis: i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_put_coordinates` C ABI symbol.
        pub fn terlan_ndarray_v1_put_coordinates(array: *mut TerlanNdArray, coordinates: *mut TerlanNdArray, values: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_putmask` C ABI symbol.
        pub fn terlan_ndarray_v1_putmask(array: *mut TerlanNdArray, mask: *mut TerlanNdArray, values: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_qr` C ABI symbol.
        pub fn terlan_ndarray_v1_qr(array: *mut TerlanNdArray, reduced: i32, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_qr_mode` C ABI symbol.
        pub fn terlan_ndarray_v1_qr_mode(array: *mut TerlanNdArray, mode: i64, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_quantile` C ABI symbol.
        pub fn terlan_ndarray_v1_quantile(array: *mut TerlanNdArray, quantile: f64, axes: *const i64, axis_count: i64, keep_dims: i32, interpolation: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_quantile_into` C ABI symbol.
        pub fn terlan_ndarray_v1_quantile_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, quantile: f64, axes: *const i64, axis_count: i64, keep_dims: i32, interpolation: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_quantiles` C ABI symbol.
        pub fn terlan_ndarray_v1_quantiles(array: *mut TerlanNdArray, quantiles: *const f64, quantile_count: i64, axes: *const i64, axis_count: i64, keep_dims: i32, interpolation: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_quantiles_into` C ABI symbol.
        pub fn terlan_ndarray_v1_quantiles_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, quantiles: *const f64, quantile_count: i64, axes: *const i64, axis_count: i64, keep_dims: i32, interpolation: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_radians` C ABI symbol.
        pub fn terlan_ndarray_v1_radians(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_bernoulli` C ABI symbol.
        pub fn terlan_ndarray_v1_random_bernoulli(shape: *const i64, rank: i64, probability: f64, seed: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_integers` C ABI symbol.
        pub fn terlan_ndarray_v1_random_integers(shape: *const i64, rank: i64, low: i64, high: i64, seed: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_normal` C ABI symbol.
        pub fn terlan_ndarray_v1_random_normal(shape: *const i64, rank: i64, mean: f64, standard_deviation: f64, seed: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_permutation` C ABI symbol.
        pub fn terlan_ndarray_v1_random_permutation(length: i64, seed: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_shuffle` C ABI symbol.
        pub fn terlan_ndarray_v1_random_shuffle(array: *mut TerlanNdArray, axis: i64, seed: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_bernoulli` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_bernoulli(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, probability: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_beta` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_beta(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, alpha: f64, beta: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_binomial` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_binomial(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, trials: i64, probability: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_chisquare` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_chisquare(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, degrees_of_freedom: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_choice` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_choice(stream: *mut TerlanNdRandomStream, source: *mut TerlanNdArray, shape: *const i64, rank: i64, probabilities: *mut TerlanNdArray, replacement: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_choice_axis` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_choice_axis(stream: *mut TerlanNdRandomStream, source: *mut TerlanNdArray, shape: *const i64, rank: i64, probabilities: *mut TerlanNdArray, replacement: i32, axis: i64, shuffle: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_choice_uniform_axis` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_choice_uniform_axis(stream: *mut TerlanNdRandomStream, source: *mut TerlanNdArray, shape: *const i64, rank: i64, replacement: i32, axis: i64, shuffle: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_delete` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_delete(stream: *mut TerlanNdRandomStream) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_dirichlet` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_dirichlet(stream: *mut TerlanNdRandomStream, alpha: *mut TerlanNdArray, shape: *const i64, rank: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_exponential` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_exponential(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, scale: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_f` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_f(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, numerator_degrees: f64, denominator_degrees: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_random_stream_gamma` C ABI symbol.
        pub fn terlan_ndarray_v1_random_stream_gamma(stream: *mut TerlanNdRandomStream, shape: *const i64, rank: i64, distribution_shape: f64, scale: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
}
