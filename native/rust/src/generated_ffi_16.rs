unsafe extern "C" {
        /// Invokes the raw `terlan_ndarray_v1_trace` C ABI symbol.
        pub fn terlan_ndarray_v1_trace(array: *mut TerlanNdArray, offset: i64, axis_a: i64, axis_b: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_trace_into` C ABI symbol.
        pub fn terlan_ndarray_v1_trace_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, offset: i64, axis_a: i64, axis_b: i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_transpose` C ABI symbol.
        pub fn terlan_ndarray_v1_transpose(array: *mut TerlanNdArray, axis_a: i64, axis_b: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_trapezoidal_integral` C ABI symbol.
        pub fn terlan_ndarray_v1_trapezoidal_integral(array: *mut TerlanNdArray, coordinates: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_trapezoidal_integral_uniform` C ABI symbol.
        pub fn terlan_ndarray_v1_trapezoidal_integral_uniform(array: *mut TerlanNdArray, spacing: f64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_tri` C ABI symbol.
        pub fn terlan_ndarray_v1_tri(rows: i64, columns: i64, diagonal: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_tril` C ABI symbol.
        pub fn terlan_ndarray_v1_tril(array: *mut TerlanNdArray, diagonal: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_tril_indices` C ABI symbol.
        pub fn terlan_ndarray_v1_tril_indices(rows: i64, offset: i64, columns: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_tril_indices_from` C ABI symbol.
        pub fn terlan_ndarray_v1_tril_indices_from(array: *mut TerlanNdArray, offset: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_trim_zeros` C ABI symbol.
        pub fn terlan_ndarray_v1_trim_zeros(array: *mut TerlanNdArray, mode: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_triu` C ABI symbol.
        pub fn terlan_ndarray_v1_triu(array: *mut TerlanNdArray, diagonal: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_triu_indices` C ABI symbol.
        pub fn terlan_ndarray_v1_triu_indices(rows: i64, offset: i64, columns: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_triu_indices_from` C ABI symbol.
        pub fn terlan_ndarray_v1_triu_indices_from(array: *mut TerlanNdArray, offset: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_true_divide` C ABI symbol.
        pub fn terlan_ndarray_v1_true_divide(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_trunc` C ABI symbol.
        pub fn terlan_ndarray_v1_trunc(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_union` C ABI symbol.
        pub fn terlan_ndarray_v1_union(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_unique` C ABI symbol.
        pub fn terlan_ndarray_v1_unique(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_unique_all` C ABI symbol.
        pub fn terlan_ndarray_v1_unique_all(array: *mut TerlanNdArray, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_unique_axis_all` C ABI symbol.
        pub fn terlan_ndarray_v1_unique_axis_all(array: *mut TerlanNdArray, axis: i64, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_unpackbits` C ABI symbol.
        pub fn terlan_ndarray_v1_unpackbits(array: *mut TerlanNdArray, count: i64, bit_order: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_unpackbits_all` C ABI symbol.
        pub fn terlan_ndarray_v1_unpackbits_all(array: *mut TerlanNdArray, bit_order: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_unpackbits_axis` C ABI symbol.
        pub fn terlan_ndarray_v1_unpackbits_axis(array: *mut TerlanNdArray, axis: i64, count: i64, bit_order: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_unpackbits_axis_all` C ABI symbol.
        pub fn terlan_ndarray_v1_unpackbits_axis_all(array: *mut TerlanNdArray, axis: i64, bit_order: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_unravel_index` C ABI symbol.
        pub fn terlan_ndarray_v1_unravel_index(indexes: *mut TerlanNdArray, dimensions: *const i64, rank: i64, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_unravel_index_ordered` C ABI symbol.
        pub fn terlan_ndarray_v1_unravel_index_ordered(indexes: *mut TerlanNdArray, dimensions: *const i64, rank: i64, order: i64, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_unwrap` C ABI symbol.
        pub fn terlan_ndarray_v1_unwrap(array: *mut TerlanNdArray, period: f64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_unwrap_discontinuity` C ABI symbol.
        pub fn terlan_ndarray_v1_unwrap_discontinuity(array: *mut TerlanNdArray, period: f64, discontinuity: f64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_unwrap_integer_period` C ABI symbol.
        pub fn terlan_ndarray_v1_unwrap_integer_period(array: *mut TerlanNdArray, period: i64, discontinuity: i64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_vander` C ABI symbol.
        pub fn terlan_ndarray_v1_vander(array: *mut TerlanNdArray, columns: i64, increasing: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_variance` C ABI symbol.
        pub fn terlan_ndarray_v1_variance(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, correction: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_variance_as` C ABI symbol.
        pub fn terlan_ndarray_v1_variance_as(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, correction: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_variance_into` C ABI symbol.
        pub fn terlan_ndarray_v1_variance_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, correction: i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_variance_where` C ABI symbol.
        pub fn terlan_ndarray_v1_variance_where(array: *mut TerlanNdArray, mask: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, correction: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_vdot` C ABI symbol.
        pub fn terlan_ndarray_v1_vdot(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_vsplit` C ABI symbol.
        pub fn terlan_ndarray_v1_vsplit(array: *mut TerlanNdArray, sections: i64, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_vstack` C ABI symbol.
        pub fn terlan_ndarray_v1_vstack(arrays: *const u64, array_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_weighted_average` C ABI symbol.
        pub fn terlan_ndarray_v1_weighted_average(array: *mut TerlanNdArray, weights: *mut TerlanNdArray, axis: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_weighted_average_axes` C ABI symbol.
        pub fn terlan_ndarray_v1_weighted_average_axes(array: *mut TerlanNdArray, weights: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_weighted_average_weight_sum` C ABI symbol.
        pub fn terlan_ndarray_v1_weighted_average_weight_sum(array: *mut TerlanNdArray, weights: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_weighted_average_with_sum` C ABI symbol.
        pub fn terlan_ndarray_v1_weighted_average_with_sum(array: *mut TerlanNdArray, weights: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_where` C ABI symbol.
        pub fn terlan_ndarray_v1_where(mask: *mut TerlanNdArray, selected: *mut TerlanNdArray, rejected: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_zeros` C ABI symbol.
        pub fn terlan_ndarray_v1_zeros(shape: *const i64, rank: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_zeros_like` C ABI symbol.
        pub fn terlan_ndarray_v1_zeros_like(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_zeros_ordered` C ABI symbol.
        pub fn terlan_ndarray_v1_zeros_ordered(shape: *const i64, rank: i64, dtype: i32, order: i32, out_array: *mut *mut TerlanNdArray) -> i32;
}
