unsafe extern "C" {
        /// Invokes the raw `terlan_ndarray_v1_pad` C ABI symbol.
        pub fn terlan_ndarray_v1_pad(array: *mut TerlanNdArray, leading: *const i64, leading_count: i64, trailing: *const i64, trailing_count: i64, value: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_pad_constant` C ABI symbol.
        pub fn terlan_ndarray_v1_pad_constant(array: *mut TerlanNdArray, leading: *const i64, leading_count: i64, trailing: *const i64, trailing_count: i64, leading_values: *const f64, leading_value_count: i64, trailing_values: *const f64, trailing_value_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_pad_constant_values` C ABI symbol.
        pub fn terlan_ndarray_v1_pad_constant_values(array: *mut TerlanNdArray, leading: *const i64, leading_count: i64, trailing: *const i64, trailing_count: i64, leading_values: *mut TerlanNdArray, trailing_values: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_pad_edge` C ABI symbol.
        pub fn terlan_ndarray_v1_pad_edge(array: *mut TerlanNdArray, leading: *const i64, leading_count: i64, trailing: *const i64, trailing_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_pad_empty` C ABI symbol.
        pub fn terlan_ndarray_v1_pad_empty(array: *mut TerlanNdArray, leading: *const i64, leading_count: i64, trailing: *const i64, trailing_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_pad_linear_ramp` C ABI symbol.
        pub fn terlan_ndarray_v1_pad_linear_ramp(array: *mut TerlanNdArray, leading: *const i64, leading_count: i64, trailing: *const i64, trailing_count: i64, leading_values: *const f64, leading_value_count: i64, trailing_values: *const f64, trailing_value_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_pad_linear_ramp_values` C ABI symbol.
        pub fn terlan_ndarray_v1_pad_linear_ramp_values(array: *mut TerlanNdArray, leading: *const i64, leading_count: i64, trailing: *const i64, trailing_count: i64, leading_values: *mut TerlanNdArray, trailing_values: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_pad_maximum` C ABI symbol.
        pub fn terlan_ndarray_v1_pad_maximum(array: *mut TerlanNdArray, leading: *const i64, leading_count: i64, trailing: *const i64, trailing_count: i64, leading_lengths: *const i64, leading_length_count: i64, trailing_lengths: *const i64, trailing_length_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_pad_mean` C ABI symbol.
        pub fn terlan_ndarray_v1_pad_mean(array: *mut TerlanNdArray, leading: *const i64, leading_count: i64, trailing: *const i64, trailing_count: i64, leading_lengths: *const i64, leading_length_count: i64, trailing_lengths: *const i64, trailing_length_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_pad_median` C ABI symbol.
        pub fn terlan_ndarray_v1_pad_median(array: *mut TerlanNdArray, leading: *const i64, leading_count: i64, trailing: *const i64, trailing_count: i64, leading_lengths: *const i64, leading_length_count: i64, trailing_lengths: *const i64, trailing_length_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_pad_minimum` C ABI symbol.
        pub fn terlan_ndarray_v1_pad_minimum(array: *mut TerlanNdArray, leading: *const i64, leading_count: i64, trailing: *const i64, trailing_count: i64, leading_lengths: *const i64, leading_length_count: i64, trailing_lengths: *const i64, trailing_length_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_pad_reflect` C ABI symbol.
        pub fn terlan_ndarray_v1_pad_reflect(array: *mut TerlanNdArray, leading: *const i64, leading_count: i64, trailing: *const i64, trailing_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_pad_reflect_odd` C ABI symbol.
        pub fn terlan_ndarray_v1_pad_reflect_odd(array: *mut TerlanNdArray, leading: *const i64, leading_count: i64, trailing: *const i64, trailing_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_pad_statistical` C ABI symbol.
        pub fn terlan_ndarray_v1_pad_statistical(array: *mut TerlanNdArray, leading: *const i64, leading_count: i64, trailing: *const i64, trailing_count: i64, leading_lengths: *const i64, leading_length_count: i64, trailing_lengths: *const i64, trailing_length_count: i64, mode: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_pad_symmetric` C ABI symbol.
        pub fn terlan_ndarray_v1_pad_symmetric(array: *mut TerlanNdArray, leading: *const i64, leading_count: i64, trailing: *const i64, trailing_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_pad_symmetric_odd` C ABI symbol.
        pub fn terlan_ndarray_v1_pad_symmetric_odd(array: *mut TerlanNdArray, leading: *const i64, leading_count: i64, trailing: *const i64, trailing_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_pad_wrap` C ABI symbol.
        pub fn terlan_ndarray_v1_pad_wrap(array: *mut TerlanNdArray, leading: *const i64, leading_count: i64, trailing: *const i64, trailing_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_partition` C ABI symbol.
        pub fn terlan_ndarray_v1_partition(array: *mut TerlanNdArray, kth: i64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_partition_flat` C ABI symbol.
        pub fn terlan_ndarray_v1_partition_flat(array: *mut TerlanNdArray, kth: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_partition_inplace` C ABI symbol.
        pub fn terlan_ndarray_v1_partition_inplace(array: *mut TerlanNdArray, kth: i64, axis: i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_partition_kths` C ABI symbol.
        pub fn terlan_ndarray_v1_partition_kths(array: *mut TerlanNdArray, kths: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_partition_kths_flat` C ABI symbol.
        pub fn terlan_ndarray_v1_partition_kths_flat(array: *mut TerlanNdArray, kths: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_partition_kths_inplace` C ABI symbol.
        pub fn terlan_ndarray_v1_partition_kths_inplace(array: *mut TerlanNdArray, kths: *mut TerlanNdArray, axis: i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_peak_to_peak` C ABI symbol.
        pub fn terlan_ndarray_v1_peak_to_peak(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_percentile` C ABI symbol.
        pub fn terlan_ndarray_v1_percentile(array: *mut TerlanNdArray, percentile: f64, axes: *const i64, axis_count: i64, keep_dims: i32, interpolation: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_percentile_into` C ABI symbol.
        pub fn terlan_ndarray_v1_percentile_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, percentile: f64, axes: *const i64, axis_count: i64, keep_dims: i32, interpolation: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_percentiles` C ABI symbol.
        pub fn terlan_ndarray_v1_percentiles(array: *mut TerlanNdArray, percentiles: *const f64, percentile_count: i64, axes: *const i64, axis_count: i64, keep_dims: i32, interpolation: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_percentiles_into` C ABI symbol.
        pub fn terlan_ndarray_v1_percentiles_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, percentiles: *const f64, percentile_count: i64, axes: *const i64, axis_count: i64, keep_dims: i32, interpolation: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_permute` C ABI symbol.
        pub fn terlan_ndarray_v1_permute(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_pinv` C ABI symbol.
        pub fn terlan_ndarray_v1_pinv(array: *mut TerlanNdArray, tolerance: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_place` C ABI symbol.
        pub fn terlan_ndarray_v1_place(array: *mut TerlanNdArray, mask: *mut TerlanNdArray, values: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_poly2cheb` C ABI symbol.
        pub fn terlan_ndarray_v1_poly2cheb(coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_poly2leg` C ABI symbol.
        pub fn terlan_ndarray_v1_poly2leg(coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyadd` C ABI symbol.
        pub fn terlan_ndarray_v1_polyadd(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polycompanion` C ABI symbol.
        pub fn terlan_ndarray_v1_polycompanion(coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyconvert` C ABI symbol.
        pub fn terlan_ndarray_v1_polyconvert(coefficients: *mut TerlanNdArray, source_basis: i32, target_basis: i32, domain_minimum: f64, domain_maximum: f64, window_minimum: f64, window_maximum: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyder` C ABI symbol.
        pub fn terlan_ndarray_v1_polyder(coefficients: *mut TerlanNdArray, order: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyder_axis` C ABI symbol.
        pub fn terlan_ndarray_v1_polyder_axis(coefficients: *mut TerlanNdArray, order: i64, scale: f64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polydiv_all` C ABI symbol.
        pub fn terlan_ndarray_v1_polydiv_all(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyfit` C ABI symbol.
        pub fn terlan_ndarray_v1_polyfit(coordinates: *mut TerlanNdArray, values: *mut TerlanNdArray, degree: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyfit_covariance` C ABI symbol.
        pub fn terlan_ndarray_v1_polyfit_covariance(coordinates: *mut TerlanNdArray, values: *mut TerlanNdArray, degree: i64, scale_covariance: i32, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyfit_degrees` C ABI symbol.
        pub fn terlan_ndarray_v1_polyfit_degrees(coordinates: *mut TerlanNdArray, values: *mut TerlanNdArray, degrees: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyfit_weighted_covariance` C ABI symbol.
        pub fn terlan_ndarray_v1_polyfit_weighted_covariance(coordinates: *mut TerlanNdArray, values: *mut TerlanNdArray, degree: i64, weights: *mut TerlanNdArray, scale_covariance: i32, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyfit_weighted_degrees` C ABI symbol.
        pub fn terlan_ndarray_v1_polyfit_weighted_degrees(coordinates: *mut TerlanNdArray, values: *mut TerlanNdArray, degrees: *mut TerlanNdArray, weights: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyfromroots` C ABI symbol.
        pub fn terlan_ndarray_v1_polyfromroots(roots: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polygrid2d` C ABI symbol.
        pub fn terlan_ndarray_v1_polygrid2d(x: *mut TerlanNdArray, y: *mut TerlanNdArray, coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polygrid3d` C ABI symbol.
        pub fn terlan_ndarray_v1_polygrid3d(x: *mut TerlanNdArray, y: *mut TerlanNdArray, z: *mut TerlanNdArray, coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_polyint` C ABI symbol.
        pub fn terlan_ndarray_v1_polyint(coefficients: *mut TerlanNdArray, order: i64, constant: f64, out_array: *mut *mut TerlanNdArray) -> i32;
}
