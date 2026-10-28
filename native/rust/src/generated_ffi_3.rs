unsafe extern "C" {
        /// Invokes the raw `terlan_ndarray_v1_convolve` C ABI symbol.
        pub fn terlan_ndarray_v1_convolve(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, mode: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_copy` C ABI symbol.
        pub fn terlan_ndarray_v1_copy(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_copy_ordered` C ABI symbol.
        pub fn terlan_ndarray_v1_copy_ordered(array: *mut TerlanNdArray, order: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_copysign` C ABI symbol.
        pub fn terlan_ndarray_v1_copysign(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_copyto` C ABI symbol.
        pub fn terlan_ndarray_v1_copyto(destination: *mut TerlanNdArray, source: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_copyto_casting` C ABI symbol.
        pub fn terlan_ndarray_v1_copyto_casting(destination: *mut TerlanNdArray, source: *mut TerlanNdArray, casting: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_copyto_where` C ABI symbol.
        pub fn terlan_ndarray_v1_copyto_where(destination: *mut TerlanNdArray, source: *mut TerlanNdArray, mask: *mut TerlanNdArray, casting: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_correlate` C ABI symbol.
        pub fn terlan_ndarray_v1_correlate(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, mode: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_correlation` C ABI symbol.
        pub fn terlan_ndarray_v1_correlation(array: *mut TerlanNdArray, row_variables: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_correlation_pair` C ABI symbol.
        pub fn terlan_ndarray_v1_correlation_pair(array: *mut TerlanNdArray, other: *mut TerlanNdArray, row_variables: i32, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_correlation_typed` C ABI symbol.
        pub fn terlan_ndarray_v1_correlation_typed(array: *mut TerlanNdArray, row_variables: i32, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_cos` C ABI symbol.
        pub fn terlan_ndarray_v1_cos(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_cosh` C ABI symbol.
        pub fn terlan_ndarray_v1_cosh(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_count_nonzero` C ABI symbol.
        pub fn terlan_ndarray_v1_count_nonzero(array: *mut TerlanNdArray, out_count: *mut i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_count_nonzero_axes` C ABI symbol.
        pub fn terlan_ndarray_v1_count_nonzero_axes(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_covariance` C ABI symbol.
        pub fn terlan_ndarray_v1_covariance(array: *mut TerlanNdArray, row_variables: i32, correction: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_covariance_analytic_weighted` C ABI symbol.
        pub fn terlan_ndarray_v1_covariance_analytic_weighted(array: *mut TerlanNdArray, analytic_weights: *mut TerlanNdArray, row_variables: i32, correction: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_covariance_frequency_weighted` C ABI symbol.
        pub fn terlan_ndarray_v1_covariance_frequency_weighted(array: *mut TerlanNdArray, frequency_weights: *mut TerlanNdArray, row_variables: i32, correction: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_covariance_pair` C ABI symbol.
        pub fn terlan_ndarray_v1_covariance_pair(array: *mut TerlanNdArray, other: *mut TerlanNdArray, row_variables: i32, correction: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_covariance_pair_weighted` C ABI symbol.
        pub fn terlan_ndarray_v1_covariance_pair_weighted(array: *mut TerlanNdArray, other: *mut TerlanNdArray, frequency_weights: *mut TerlanNdArray, analytic_weights: *mut TerlanNdArray, row_variables: i32, correction: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_covariance_typed` C ABI symbol.
        pub fn terlan_ndarray_v1_covariance_typed(array: *mut TerlanNdArray, row_variables: i32, correction: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_covariance_weighted` C ABI symbol.
        pub fn terlan_ndarray_v1_covariance_weighted(array: *mut TerlanNdArray, frequency_weights: *mut TerlanNdArray, analytic_weights: *mut TerlanNdArray, row_variables: i32, correction: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_cross` C ABI symbol.
        pub fn terlan_ndarray_v1_cross(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_cumulative_product` C ABI symbol.
        pub fn terlan_ndarray_v1_cumulative_product(array: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_cumulative_product_as` C ABI symbol.
        pub fn terlan_ndarray_v1_cumulative_product_as(array: *mut TerlanNdArray, axis: i64, requested_dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_cumulative_sum` C ABI symbol.
        pub fn terlan_ndarray_v1_cumulative_sum(array: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_cumulative_sum_as` C ABI symbol.
        pub fn terlan_ndarray_v1_cumulative_sum_as(array: *mut TerlanNdArray, axis: i64, requested_dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_degrees` C ABI symbol.
        pub fn terlan_ndarray_v1_degrees(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_delete` C ABI symbol.
        pub fn terlan_ndarray_v1_delete(array: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_delete_bool_values` C ABI symbol.
        pub fn terlan_ndarray_v1_delete_bool_values(values: *mut u8);
        /// Invokes the raw `terlan_ndarray_v1_delete_float_values` C ABI symbol.
        pub fn terlan_ndarray_v1_delete_float_values(values: *mut f64);
        /// Invokes the raw `terlan_ndarray_v1_delete_handle_values` C ABI symbol.
        pub fn terlan_ndarray_v1_delete_handle_values(values: *mut u64);
        /// Invokes the raw `terlan_ndarray_v1_delete_indexes` C ABI symbol.
        pub fn terlan_ndarray_v1_delete_indexes(array: *mut TerlanNdArray, indexes: *const i64, index_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_delete_indexes_axis` C ABI symbol.
        pub fn terlan_ndarray_v1_delete_indexes_axis(array: *mut TerlanNdArray, indexes: *const i64, index_count: i64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_delete_int_values` C ABI symbol.
        pub fn terlan_ndarray_v1_delete_int_values(values: *mut i64);
        /// Invokes the raw `terlan_ndarray_v1_delete_mask_axis` C ABI symbol.
        pub fn terlan_ndarray_v1_delete_mask_axis(array: *mut TerlanNdArray, mask: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_delete_tensor_packet` C ABI symbol.
        pub fn terlan_ndarray_v1_delete_tensor_packet(packet: *mut u8);
        /// Invokes the raw `terlan_ndarray_v1_delete_uint_values` C ABI symbol.
        pub fn terlan_ndarray_v1_delete_uint_values(values: *mut u64);
        /// Invokes the raw `terlan_ndarray_v1_deserialize` C ABI symbol.
        pub fn terlan_ndarray_v1_deserialize(bytes: *const u8, length: i64, maximum: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_det` C ABI symbol.
        pub fn terlan_ndarray_v1_det(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_determinant` C ABI symbol.
        pub fn terlan_ndarray_v1_determinant(array: *mut TerlanNdArray, out_value: *mut f64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_device_type` C ABI symbol.
        pub fn terlan_ndarray_v1_device_type(array: *mut TerlanNdArray, out_device_type: *mut i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_diag` C ABI symbol.
        pub fn terlan_ndarray_v1_diag(array: *mut TerlanNdArray, offset: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_diag_indices` C ABI symbol.
        pub fn terlan_ndarray_v1_diag_indices(size: i64, rank: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_diag_indices_from` C ABI symbol.
        pub fn terlan_ndarray_v1_diag_indices_from(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_diagflat` C ABI symbol.
        pub fn terlan_ndarray_v1_diagflat(array: *mut TerlanNdArray, diagonal: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_diagonal` C ABI symbol.
        pub fn terlan_ndarray_v1_diagonal(array: *mut TerlanNdArray, offset: i64, axis_a: i64, axis_b: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_difference` C ABI symbol.
        pub fn terlan_ndarray_v1_difference(array: *mut TerlanNdArray, order: i64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
}
