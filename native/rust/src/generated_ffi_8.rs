unsafe extern "C" {
        /// Invokes the raw `terlan_ndarray_v1_less` C ABI symbol.
        pub fn terlan_ndarray_v1_less(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_less_equal` C ABI symbol.
        pub fn terlan_ndarray_v1_less_equal(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_lexsort_axis` C ABI symbol.
        pub fn terlan_ndarray_v1_lexsort_axis(keys: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_lexsort_keys` C ABI symbol.
        pub fn terlan_ndarray_v1_lexsort_keys(keys: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_linspace` C ABI symbol.
        pub fn terlan_ndarray_v1_linspace(start: f64, stop: f64, count: i64, endpoint: i32, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_linspace_arrays` C ABI symbol.
        pub fn terlan_ndarray_v1_linspace_arrays(start: *mut TerlanNdArray, stop: *mut TerlanNdArray, count: i64, endpoint: i32, axis: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_linspace_steps` C ABI symbol.
        pub fn terlan_ndarray_v1_linspace_steps(start: *mut TerlanNdArray, stop: *mut TerlanNdArray, count: i64, endpoint: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_live_count` C ABI symbol.
        pub fn terlan_ndarray_v1_live_count() -> i64;
        /// Invokes the raw `terlan_ndarray_v1_load_npy` C ABI symbol.
        pub fn terlan_ndarray_v1_load_npy(path: *const std::ffi::c_char, maximum: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_log` C ABI symbol.
        pub fn terlan_ndarray_v1_log(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_log10` C ABI symbol.
        pub fn terlan_ndarray_v1_log10(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_log1p` C ABI symbol.
        pub fn terlan_ndarray_v1_log1p(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_log2` C ABI symbol.
        pub fn terlan_ndarray_v1_log2(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_logaddexp` C ABI symbol.
        pub fn terlan_ndarray_v1_logaddexp(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_logaddexp2` C ABI symbol.
        pub fn terlan_ndarray_v1_logaddexp2(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_logical_and` C ABI symbol.
        pub fn terlan_ndarray_v1_logical_and(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_logical_and_into` C ABI symbol.
        pub fn terlan_ndarray_v1_logical_and_into(destination: *mut TerlanNdArray, lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_logical_not` C ABI symbol.
        pub fn terlan_ndarray_v1_logical_not(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_logical_not_into` C ABI symbol.
        pub fn terlan_ndarray_v1_logical_not_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_logical_or` C ABI symbol.
        pub fn terlan_ndarray_v1_logical_or(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_logical_or_into` C ABI symbol.
        pub fn terlan_ndarray_v1_logical_or_into(destination: *mut TerlanNdArray, lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_logical_xor` C ABI symbol.
        pub fn terlan_ndarray_v1_logical_xor(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_logical_xor_into` C ABI symbol.
        pub fn terlan_ndarray_v1_logical_xor_into(destination: *mut TerlanNdArray, lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_logical_xor_reduce` C ABI symbol.
        pub fn terlan_ndarray_v1_logical_xor_reduce(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_logspace` C ABI symbol.
        pub fn terlan_ndarray_v1_logspace(start: f64, stop: f64, count: i64, endpoint: i32, base: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_logspace_arrays` C ABI symbol.
        pub fn terlan_ndarray_v1_logspace_arrays(start: *mut TerlanNdArray, stop: *mut TerlanNdArray, count: i64, endpoint: i32, axis: i64, base: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_lstsq` C ABI symbol.
        pub fn terlan_ndarray_v1_lstsq(coefficients: *mut TerlanNdArray, rhs: *mut TerlanNdArray, rcond: f64, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_mask_assign` C ABI symbol.
        pub fn terlan_ndarray_v1_mask_assign(array: *mut TerlanNdArray, mask: *mut TerlanNdArray, values: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_mask_select` C ABI symbol.
        pub fn terlan_ndarray_v1_mask_select(array: *mut TerlanNdArray, mask: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_matmul` C ABI symbol.
        pub fn terlan_ndarray_v1_matmul(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_matmul_into` C ABI symbol.
        pub fn terlan_ndarray_v1_matmul_into(destination: *mut TerlanNdArray, lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_matrix_power` C ABI symbol.
        pub fn terlan_ndarray_v1_matrix_power(array: *mut TerlanNdArray, exponent: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_matrix_rank` C ABI symbol.
        pub fn terlan_ndarray_v1_matrix_rank(array: *mut TerlanNdArray, tolerance: f64, out_rank: *mut i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_matrix_rank_general` C ABI symbol.
        pub fn terlan_ndarray_v1_matrix_rank_general(array: *mut TerlanNdArray, tolerance: f64, hermitian: bool, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_maximum` C ABI symbol.
        pub fn terlan_ndarray_v1_maximum(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_maximum_initial` C ABI symbol.
        pub fn terlan_ndarray_v1_maximum_initial(array: *mut TerlanNdArray, initial: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_maximum_into` C ABI symbol.
        pub fn terlan_ndarray_v1_maximum_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_maximum_where` C ABI symbol.
        pub fn terlan_ndarray_v1_maximum_where(array: *mut TerlanNdArray, mask: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_maximum_where_initial` C ABI symbol.
        pub fn terlan_ndarray_v1_maximum_where_initial(array: *mut TerlanNdArray, mask: *mut TerlanNdArray, initial: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_may_share_memory` C ABI symbol.
        pub fn terlan_ndarray_v1_may_share_memory(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_value: *mut bool) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_may_share_memory_with_work` C ABI symbol.
        pub fn terlan_ndarray_v1_may_share_memory_with_work(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, max_work: i64, out_value: *mut bool) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_mean` C ABI symbol.
        pub fn terlan_ndarray_v1_mean(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_mean_as` C ABI symbol.
        pub fn terlan_ndarray_v1_mean_as(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_mean_into` C ABI symbol.
        pub fn terlan_ndarray_v1_mean_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_mean_where` C ABI symbol.
        pub fn terlan_ndarray_v1_mean_where(array: *mut TerlanNdArray, mask: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_median` C ABI symbol.
        pub fn terlan_ndarray_v1_median(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_median_into` C ABI symbol.
        pub fn terlan_ndarray_v1_median_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_meshgrid` C ABI symbol.
        pub fn terlan_ndarray_v1_meshgrid(first: *mut TerlanNdArray, remaining: *const u64, remaining_count: i64, indexing: i64, sparse: bool, copy: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
}
