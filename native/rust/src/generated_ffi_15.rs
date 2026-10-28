unsafe extern "C" {
        /// Invokes the raw `terlan_ndarray_v1_sort` C ABI symbol.
        pub fn terlan_ndarray_v1_sort(array: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_sort_complex` C ABI symbol.
        pub fn terlan_ndarray_v1_sort_complex(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_sort_flat` C ABI symbol.
        pub fn terlan_ndarray_v1_sort_flat(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_sort_inplace` C ABI symbol.
        pub fn terlan_ndarray_v1_sort_inplace(array: *mut TerlanNdArray, axis: i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_sort_unstable` C ABI symbol.
        pub fn terlan_ndarray_v1_sort_unstable(array: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_spacing` C ABI symbol.
        pub fn terlan_ndarray_v1_spacing(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_split` C ABI symbol.
        pub fn terlan_ndarray_v1_split(array: *mut TerlanNdArray, sections: i64, axis: i64, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_split_at` C ABI symbol.
        pub fn terlan_ndarray_v1_split_at(array: *mut TerlanNdArray, indices: *const i64, index_count: i64, axis: i64, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_sqrt` C ABI symbol.
        pub fn terlan_ndarray_v1_sqrt(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_square` C ABI symbol.
        pub fn terlan_ndarray_v1_square(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_squeeze` C ABI symbol.
        pub fn terlan_ndarray_v1_squeeze(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_stack` C ABI symbol.
        pub fn terlan_ndarray_v1_stack(arrays: *const u64, array_count: i64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_standard_deviation` C ABI symbol.
        pub fn terlan_ndarray_v1_standard_deviation(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, correction: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_standard_deviation_as` C ABI symbol.
        pub fn terlan_ndarray_v1_standard_deviation_as(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, correction: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_standard_deviation_into` C ABI symbol.
        pub fn terlan_ndarray_v1_standard_deviation_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, correction: i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_standard_deviation_where` C ABI symbol.
        pub fn terlan_ndarray_v1_standard_deviation_where(array: *mut TerlanNdArray, mask: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, correction: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_strides` C ABI symbol.
        pub fn terlan_ndarray_v1_strides(array: *mut TerlanNdArray, out_strides: *mut *mut i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_subtract` C ABI symbol.
        pub fn terlan_ndarray_v1_subtract(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_subtract_accumulate` C ABI symbol.
        pub fn terlan_ndarray_v1_subtract_accumulate(array: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_subtract_reduce` C ABI symbol.
        pub fn terlan_ndarray_v1_subtract_reduce(array: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_sum` C ABI symbol.
        pub fn terlan_ndarray_v1_sum(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_sum_initial` C ABI symbol.
        pub fn terlan_ndarray_v1_sum_initial(array: *mut TerlanNdArray, initial: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_sum_into` C ABI symbol.
        pub fn terlan_ndarray_v1_sum_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_sum_where` C ABI symbol.
        pub fn terlan_ndarray_v1_sum_where(array: *mut TerlanNdArray, mask: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_sum_where_initial` C ABI symbol.
        pub fn terlan_ndarray_v1_sum_where_initial(array: *mut TerlanNdArray, mask: *mut TerlanNdArray, initial: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_svd` C ABI symbol.
        pub fn terlan_ndarray_v1_svd(array: *mut TerlanNdArray, full_matrices: i32, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_symmetric_eigen` C ABI symbol.
        pub fn terlan_ndarray_v1_symmetric_eigen(array: *mut TerlanNdArray, tolerance: f64, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_symmetric_eigen_general` C ABI symbol.
        pub fn terlan_ndarray_v1_symmetric_eigen_general(array: *mut TerlanNdArray, tolerance: f64, upper: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_take` C ABI symbol.
        pub fn terlan_ndarray_v1_take(array: *mut TerlanNdArray, indices: *const i64, index_count: i64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_take_along_axis` C ABI symbol.
        pub fn terlan_ndarray_v1_take_along_axis(array: *mut TerlanNdArray, indices: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_take_indices` C ABI symbol.
        pub fn terlan_ndarray_v1_take_indices(array: *mut TerlanNdArray, indices: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_take_indices_into` C ABI symbol.
        pub fn terlan_ndarray_v1_take_indices_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, indices: *mut TerlanNdArray, axis: i64, mode: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_take_indices_mode` C ABI symbol.
        pub fn terlan_ndarray_v1_take_indices_mode(array: *mut TerlanNdArray, indices: *mut TerlanNdArray, axis: i64, mode: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_take_mode` C ABI symbol.
        pub fn terlan_ndarray_v1_take_mode(array: *mut TerlanNdArray, indices: *const i64, index_count: i64, axis: i64, mode: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_tan` C ABI symbol.
        pub fn terlan_ndarray_v1_tan(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_tanh` C ABI symbol.
        pub fn terlan_ndarray_v1_tanh(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_tensordot` C ABI symbol.
        pub fn terlan_ndarray_v1_tensordot(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, lhs_axes: *const i64, lhs_axis_count: i64, rhs_axes: *const i64, rhs_axis_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_tensorinv` C ABI symbol.
        pub fn terlan_ndarray_v1_tensorinv(array: *mut TerlanNdArray, index: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_tensorsolve` C ABI symbol.
        pub fn terlan_ndarray_v1_tensorsolve(coefficients: *mut TerlanNdArray, rhs: *mut TerlanNdArray, axes: *const i64, axis_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_tile` C ABI symbol.
        pub fn terlan_ndarray_v1_tile(array: *mut TerlanNdArray, repetitions: *const i64, repetition_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_to_arrow` C ABI symbol.
        pub fn terlan_ndarray_v1_to_arrow(array: *mut TerlanNdArray, out_arrow: *mut *mut TerlanNdArrowArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_to_bools` C ABI symbol.
        pub fn terlan_ndarray_v1_to_bools(array: *mut TerlanNdArray, out_values: *mut *mut u8, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_to_floats` C ABI symbol.
        pub fn terlan_ndarray_v1_to_floats(array: *mut TerlanNdArray, out_values: *mut *mut f64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_to_floats_cast` C ABI symbol.
        pub fn terlan_ndarray_v1_to_floats_cast(array: *mut TerlanNdArray, out_values: *mut *mut f64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_to_ints` C ABI symbol.
        pub fn terlan_ndarray_v1_to_ints(array: *mut TerlanNdArray, out_values: *mut *mut i64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_to_ints_cast` C ABI symbol.
        pub fn terlan_ndarray_v1_to_ints_cast(array: *mut TerlanNdArray, out_values: *mut *mut i64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_to_npy` C ABI symbol.
        pub fn terlan_ndarray_v1_to_npy(array: *mut TerlanNdArray, out_bytes: *mut *mut u8, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_to_uints_cast` C ABI symbol.
        pub fn terlan_ndarray_v1_to_uints_cast(array: *mut TerlanNdArray, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
}
