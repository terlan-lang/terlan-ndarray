unsafe extern "C" {
        /// Invokes the raw `terlan_ndarray_v1_minimum` C ABI symbol.
        pub fn terlan_ndarray_v1_minimum(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_minimum_initial` C ABI symbol.
        pub fn terlan_ndarray_v1_minimum_initial(array: *mut TerlanNdArray, initial: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_minimum_into` C ABI symbol.
        pub fn terlan_ndarray_v1_minimum_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_minimum_where` C ABI symbol.
        pub fn terlan_ndarray_v1_minimum_where(array: *mut TerlanNdArray, mask: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_minimum_where_initial` C ABI symbol.
        pub fn terlan_ndarray_v1_minimum_where_initial(array: *mut TerlanNdArray, mask: *mut TerlanNdArray, initial: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_mmap_npy` C ABI symbol.
        pub fn terlan_ndarray_v1_mmap_npy(path: *const std::ffi::c_char, writable: i32, maximum: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_modf_all` C ABI symbol.
        pub fn terlan_ndarray_v1_modf_all(array: *mut TerlanNdArray, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_moveaxis` C ABI symbol.
        pub fn terlan_ndarray_v1_moveaxis(array: *mut TerlanNdArray, sources: *const i64, source_count: i64, destinations: *const i64, destination_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_msort` C ABI symbol.
        pub fn terlan_ndarray_v1_msort(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_multi_dot` C ABI symbol.
        pub fn terlan_ndarray_v1_multi_dot(arrays: *const u64, array_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_multi_dot_into` C ABI symbol.
        pub fn terlan_ndarray_v1_multi_dot_into(destination: *mut TerlanNdArray, arrays: *const u64, array_count: i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_multiply` C ABI symbol.
        pub fn terlan_ndarray_v1_multiply(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_argmax` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_argmax(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_argmin` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_argmin(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_maximum` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_maximum(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_maximum_into` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_maximum_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_mean` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_mean(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_mean_as` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_mean_as(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_mean_into` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_mean_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_minimum` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_minimum(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_minimum_into` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_minimum_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_product` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_product(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_product_as` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_product_as(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_product_into` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_product_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_standard_deviation` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_standard_deviation(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, correction: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_standard_deviation_as` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_standard_deviation_as(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, correction: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_standard_deviation_into` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_standard_deviation_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, correction: i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_sum` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_sum(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_sum_as` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_sum_as(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_sum_into` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_sum_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_to_num` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_to_num(array: *mut TerlanNdArray, nan: f64, positive_infinity: f64, negative_infinity: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_to_num_default` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_to_num_default(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_to_num_into` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_to_num_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, nan: f64, positive_infinity: f64, negative_infinity: f64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_variance` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_variance(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, correction: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_variance_as` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_variance_as(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, correction: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nan_variance_into` C ABI symbol.
        pub fn terlan_ndarray_v1_nan_variance_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, correction: i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nanargmax` C ABI symbol.
        pub fn terlan_ndarray_v1_nanargmax(array: *mut TerlanNdArray, out_value: *mut i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nanargmin` C ABI symbol.
        pub fn terlan_ndarray_v1_nanargmin(array: *mut TerlanNdArray, out_value: *mut i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nancumprod` C ABI symbol.
        pub fn terlan_ndarray_v1_nancumprod(array: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nancumprod_as` C ABI symbol.
        pub fn terlan_ndarray_v1_nancumprod_as(array: *mut TerlanNdArray, axis: i64, requested_dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nancumprod_flat` C ABI symbol.
        pub fn terlan_ndarray_v1_nancumprod_flat(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nancumprod_flat_into` C ABI symbol.
        pub fn terlan_ndarray_v1_nancumprod_flat_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nancumprod_into` C ABI symbol.
        pub fn terlan_ndarray_v1_nancumprod_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axis: i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nancumsum` C ABI symbol.
        pub fn terlan_ndarray_v1_nancumsum(array: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nancumsum_as` C ABI symbol.
        pub fn terlan_ndarray_v1_nancumsum_as(array: *mut TerlanNdArray, axis: i64, requested_dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nancumsum_flat` C ABI symbol.
        pub fn terlan_ndarray_v1_nancumsum_flat(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nancumsum_flat_into` C ABI symbol.
        pub fn terlan_ndarray_v1_nancumsum_flat_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_nancumsum_into` C ABI symbol.
        pub fn terlan_ndarray_v1_nancumsum_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axis: i64) -> i32;
}
