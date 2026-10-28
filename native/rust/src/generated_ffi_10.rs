unsafe extern "C" {
        /// Invokes the raw `terlan_ndarray_v1_nanmedian` C ABI symbol.
        pub fn terlan_ndarray_v1_nanmedian(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nanmedian_axes` C ABI symbol.
        pub fn terlan_ndarray_v1_nanmedian_axes(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nanmedian_into` C ABI symbol.
        pub fn terlan_ndarray_v1_nanmedian_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nanpercentile` C ABI symbol.
        pub fn terlan_ndarray_v1_nanpercentile(array: *mut TerlanNdArray, percentile: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nanpercentile_axes` C ABI symbol.
        pub fn terlan_ndarray_v1_nanpercentile_axes(array: *mut TerlanNdArray, percentile: f64, axes: *const i64, axis_count: i64, keep_dims: i32, interpolation: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nanpercentile_into` C ABI symbol.
        pub fn terlan_ndarray_v1_nanpercentile_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, percentile: f64, axes: *const i64, axis_count: i64, keep_dims: i32, interpolation: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nanpercentiles` C ABI symbol.
        pub fn terlan_ndarray_v1_nanpercentiles(array: *mut TerlanNdArray, percentiles: *const f64, percentile_count: i64, axes: *const i64, axis_count: i64, keep_dims: i32, interpolation: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nanpercentiles_into` C ABI symbol.
        pub fn terlan_ndarray_v1_nanpercentiles_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, percentiles: *const f64, percentile_count: i64, axes: *const i64, axis_count: i64, keep_dims: i32, interpolation: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nanquantile` C ABI symbol.
        pub fn terlan_ndarray_v1_nanquantile(array: *mut TerlanNdArray, quantile: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nanquantile_axes` C ABI symbol.
        pub fn terlan_ndarray_v1_nanquantile_axes(array: *mut TerlanNdArray, quantile: f64, axes: *const i64, axis_count: i64, keep_dims: i32, interpolation: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nanquantile_into` C ABI symbol.
        pub fn terlan_ndarray_v1_nanquantile_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, quantile: f64, axes: *const i64, axis_count: i64, keep_dims: i32, interpolation: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nanquantiles` C ABI symbol.
        pub fn terlan_ndarray_v1_nanquantiles(array: *mut TerlanNdArray, quantiles: *const f64, quantile_count: i64, axes: *const i64, axis_count: i64, keep_dims: i32, interpolation: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nanquantiles_into` C ABI symbol.
        pub fn terlan_ndarray_v1_nanquantiles_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, quantiles: *const f64, quantile_count: i64, axes: *const i64, axis_count: i64, keep_dims: i32, interpolation: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_negative` C ABI symbol.
        pub fn terlan_ndarray_v1_negative(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nextafter` C ABI symbol.
        pub fn terlan_ndarray_v1_nextafter(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nonzero` C ABI symbol.
        pub fn terlan_ndarray_v1_nonzero(array: *mut TerlanNdArray, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_norm` C ABI symbol.
        pub fn terlan_ndarray_v1_norm(array: *mut TerlanNdArray, out_value: *mut f64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_norm_general` C ABI symbol.
        pub fn terlan_ndarray_v1_norm_general(array: *mut TerlanNdArray, order_kind: i64, order_value: f64, axes: *const i64, axis_count: i64, keepdims: bool, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_norm_l1` C ABI symbol.
        pub fn terlan_ndarray_v1_norm_l1(array: *mut TerlanNdArray, out_value: *mut f64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_norm_l2` C ABI symbol.
        pub fn terlan_ndarray_v1_norm_l2(array: *mut TerlanNdArray, out_value: *mut f64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_norm_max` C ABI symbol.
        pub fn terlan_ndarray_v1_norm_max(array: *mut TerlanNdArray, out_value: *mut f64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_not_equal` C ABI symbol.
        pub fn terlan_ndarray_v1_not_equal(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nuclear_norm` C ABI symbol.
        pub fn terlan_ndarray_v1_nuclear_norm(array: *mut TerlanNdArray, out_value: *mut f64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_numel` C ABI symbol.
        pub fn terlan_ndarray_v1_numel(array: *mut TerlanNdArray, out_numel: *mut i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_numpy_histogram` C ABI symbol.
        pub fn terlan_ndarray_v1_numpy_histogram(array: *mut TerlanNdArray, bins: i64, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_numpy_histogram_bin_edges` C ABI symbol.
        pub fn terlan_ndarray_v1_numpy_histogram_bin_edges(array: *mut TerlanNdArray, bins: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_numpy_histogram_bin_edges_estimator` C ABI symbol.
        pub fn terlan_ndarray_v1_numpy_histogram_bin_edges_estimator(array: *mut TerlanNdArray, estimator: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_numpy_histogram_edges` C ABI symbol.
        pub fn terlan_ndarray_v1_numpy_histogram_edges(array: *mut TerlanNdArray, edges: *mut TerlanNdArray, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_numpy_histogram_estimator` C ABI symbol.
        pub fn terlan_ndarray_v1_numpy_histogram_estimator(array: *mut TerlanNdArray, estimator: i64, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_numpy_histogram_estimator_range` C ABI symbol.
        pub fn terlan_ndarray_v1_numpy_histogram_estimator_range(array: *mut TerlanNdArray, estimator: i64, lower: f64, upper: f64, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_numpy_histogram_range` C ABI symbol.
        pub fn terlan_ndarray_v1_numpy_histogram_range(array: *mut TerlanNdArray, bins: i64, lower: f64, upper: f64, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_numpy_histogram_weighted` C ABI symbol.
        pub fn terlan_ndarray_v1_numpy_histogram_weighted(array: *mut TerlanNdArray, weights: *mut TerlanNdArray, bins: i64, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_numpy_histogram_weighted_edges` C ABI symbol.
        pub fn terlan_ndarray_v1_numpy_histogram_weighted_edges(array: *mut TerlanNdArray, weights: *mut TerlanNdArray, edges: *mut TerlanNdArray, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_numpy_histogram_weighted_range` C ABI symbol.
        pub fn terlan_ndarray_v1_numpy_histogram_weighted_range(array: *mut TerlanNdArray, weights: *mut TerlanNdArray, bins: i64, lower: f64, upper: f64, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_numpy_histogramdd` C ABI symbol.
        pub fn terlan_ndarray_v1_numpy_histogramdd(samples: *mut TerlanNdArray, bins: i64, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_numpy_histogramdd_bins_array` C ABI symbol.
        pub fn terlan_ndarray_v1_numpy_histogramdd_bins_array(samples: *mut TerlanNdArray, bins: *mut TerlanNdArray, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_numpy_histogramdd_edges_packed` C ABI symbol.
        pub fn terlan_ndarray_v1_numpy_histogramdd_edges_packed(samples: *mut TerlanNdArray, edges: *mut TerlanNdArray, offsets: *mut TerlanNdArray, dtypes: *mut TerlanNdArray, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_numpy_histogramdd_range_arrays` C ABI symbol.
        pub fn terlan_ndarray_v1_numpy_histogramdd_range_arrays(samples: *mut TerlanNdArray, bins: *mut TerlanNdArray, minima: *mut TerlanNdArray, maxima: *mut TerlanNdArray, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_numpy_histogramdd_weighted` C ABI symbol.
        pub fn terlan_ndarray_v1_numpy_histogramdd_weighted(samples: *mut TerlanNdArray, weights: *mut TerlanNdArray, bins: i64, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_numpy_histogramdd_weighted_bins_array` C ABI symbol.
        pub fn terlan_ndarray_v1_numpy_histogramdd_weighted_bins_array(samples: *mut TerlanNdArray, weights: *mut TerlanNdArray, bins: *mut TerlanNdArray, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_numpy_histogramdd_weighted_edges_packed` C ABI symbol.
        pub fn terlan_ndarray_v1_numpy_histogramdd_weighted_edges_packed(samples: *mut TerlanNdArray, weights: *mut TerlanNdArray, edges: *mut TerlanNdArray, offsets: *mut TerlanNdArray, dtypes: *mut TerlanNdArray, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_numpy_histogramdd_weighted_range_arrays` C ABI symbol.
        pub fn terlan_ndarray_v1_numpy_histogramdd_weighted_range_arrays(samples: *mut TerlanNdArray, weights: *mut TerlanNdArray, bins: *mut TerlanNdArray, minima: *mut TerlanNdArray, maxima: *mut TerlanNdArray, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_ones` C ABI symbol.
        pub fn terlan_ndarray_v1_ones(shape: *const i64, rank: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_ones_like` C ABI symbol.
        pub fn terlan_ndarray_v1_ones_like(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_outer` C ABI symbol.
        pub fn terlan_ndarray_v1_outer(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_outer_into` C ABI symbol.
        pub fn terlan_ndarray_v1_outer_into(destination: *mut TerlanNdArray, lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_packbits` C ABI symbol.
        pub fn terlan_ndarray_v1_packbits(array: *mut TerlanNdArray, bit_order: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_packbits_axis` C ABI symbol.
        pub fn terlan_ndarray_v1_packbits_axis(array: *mut TerlanNdArray, axis: i64, bit_order: i64, out_array: *mut *mut TerlanNdArray) -> i32;
}
