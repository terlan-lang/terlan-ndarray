unsafe extern "C" {
        /// Invokes the raw `terlan_ndarray_v1_difference_append` C ABI symbol.
        pub fn terlan_ndarray_v1_difference_append(array: *mut TerlanNdArray, append: *mut TerlanNdArray, order: i64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_difference_prepend` C ABI symbol.
        pub fn terlan_ndarray_v1_difference_prepend(array: *mut TerlanNdArray, prepend: *mut TerlanNdArray, order: i64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_difference_surround` C ABI symbol.
        pub fn terlan_ndarray_v1_difference_surround(array: *mut TerlanNdArray, prepend: *mut TerlanNdArray, append: *mut TerlanNdArray, order: i64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_digitize` C ABI symbol.
        pub fn terlan_ndarray_v1_digitize(values: *mut TerlanNdArray, bins: *mut TerlanNdArray, right: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_divide_accumulate` C ABI symbol.
        pub fn terlan_ndarray_v1_divide_accumulate(array: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_divide_reduce` C ABI symbol.
        pub fn terlan_ndarray_v1_divide_reduce(array: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_divmod` C ABI symbol.
        pub fn terlan_ndarray_v1_divmod(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_dot` C ABI symbol.
        pub fn terlan_ndarray_v1_dot(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_dot_into` C ABI symbol.
        pub fn terlan_ndarray_v1_dot_into(destination: *mut TerlanNdArray, lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_dsplit` C ABI symbol.
        pub fn terlan_ndarray_v1_dsplit(array: *mut TerlanNdArray, sections: i64, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_dstack` C ABI symbol.
        pub fn terlan_ndarray_v1_dstack(arrays: *const u64, array_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_dtype` C ABI symbol.
        pub fn terlan_ndarray_v1_dtype(array: *mut TerlanNdArray, out_dtype: *mut i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_ediff1d` C ABI symbol.
        pub fn terlan_ndarray_v1_ediff1d(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_ediff1d_edges` C ABI symbol.
        pub fn terlan_ndarray_v1_ediff1d_edges(array: *mut TerlanNdArray, begin: *mut TerlanNdArray, end: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_eig` C ABI symbol.
        pub fn terlan_ndarray_v1_eig(array: *mut TerlanNdArray, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_eigvals` C ABI symbol.
        pub fn terlan_ndarray_v1_eigvals(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_eigvalsh` C ABI symbol.
        pub fn terlan_ndarray_v1_eigvalsh(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_eigvalsh_general` C ABI symbol.
        pub fn terlan_ndarray_v1_eigvalsh_general(array: *mut TerlanNdArray, upper: bool, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_einsum` C ABI symbol.
        pub fn terlan_ndarray_v1_einsum(equation: *const std::ffi::c_char, operands: *const u64, operand_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_einsum_into` C ABI symbol.
        pub fn terlan_ndarray_v1_einsum_into(destination: *mut TerlanNdArray, equation: *const std::ffi::c_char, operands: *const u64, operand_count: i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_einsum_ordered` C ABI symbol.
        pub fn terlan_ndarray_v1_einsum_ordered(equation: *const std::ffi::c_char, operands: *const u64, operand_count: i64, order: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_elementwise_maximum` C ABI symbol.
        pub fn terlan_ndarray_v1_elementwise_maximum(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_elementwise_minimum` C ABI symbol.
        pub fn terlan_ndarray_v1_elementwise_minimum(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_empty` C ABI symbol.
        pub fn terlan_ndarray_v1_empty(shape: *const i64, rank: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_empty_like` C ABI symbol.
        pub fn terlan_ndarray_v1_empty_like(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_equal` C ABI symbol.
        pub fn terlan_ndarray_v1_equal(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_exp` C ABI symbol.
        pub fn terlan_ndarray_v1_exp(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_exp2` C ABI symbol.
        pub fn terlan_ndarray_v1_exp2(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_expand_dims` C ABI symbol.
        pub fn terlan_ndarray_v1_expand_dims(array: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_expand_dims_axes` C ABI symbol.
        pub fn terlan_ndarray_v1_expand_dims_axes(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_expm1` C ABI symbol.
        pub fn terlan_ndarray_v1_expm1(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_export_tensor_packet` C ABI symbol.
        pub fn terlan_ndarray_v1_export_tensor_packet(array: *mut TerlanNdArray, consumer: *const std::ffi::c_char, out_packet: *mut *mut u8, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_eye` C ABI symbol.
        pub fn terlan_ndarray_v1_eye(rows: i64, columns: i64, diagonal: i64, dtype: i32, order: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_fft` C ABI symbol.
        pub fn terlan_ndarray_v1_fft(array: *mut TerlanNdArray, axis: i64, normalization: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_fft_sized` C ABI symbol.
        pub fn terlan_ndarray_v1_fft_sized(array: *mut TerlanNdArray, length: i64, axis: i64, normalization: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_fftfreq` C ABI symbol.
        pub fn terlan_ndarray_v1_fftfreq(length: i64, spacing: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_fftn` C ABI symbol.
        pub fn terlan_ndarray_v1_fftn(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, normalization: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_fftshift` C ABI symbol.
        pub fn terlan_ndarray_v1_fftshift(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_fill` C ABI symbol.
        pub fn terlan_ndarray_v1_fill(array: *mut TerlanNdArray, value: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_fill_diagonal` C ABI symbol.
        pub fn terlan_ndarray_v1_fill_diagonal(array: *mut TerlanNdArray, value: f64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_fill_diagonal_with_wrap` C ABI symbol.
        pub fn terlan_ndarray_v1_fill_diagonal_with_wrap(array: *mut TerlanNdArray, value: f64, wrap: bool) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_fix` C ABI symbol.
        pub fn terlan_ndarray_v1_fix(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_flatnonzero` C ABI symbol.
        pub fn terlan_ndarray_v1_flatnonzero(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_flatten` C ABI symbol.
        pub fn terlan_ndarray_v1_flatten(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_flatten_ordered` C ABI symbol.
        pub fn terlan_ndarray_v1_flatten_ordered(array: *mut TerlanNdArray, order: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_flip` C ABI symbol.
        pub fn terlan_ndarray_v1_flip(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_flip_all` C ABI symbol.
        pub fn terlan_ndarray_v1_flip_all(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_fliplr` C ABI symbol.
        pub fn terlan_ndarray_v1_fliplr(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
}
