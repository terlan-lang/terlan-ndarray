unsafe extern "C" {
        /// Invokes the raw `terlan_ndarray_v1_reciprocal_where_into` C ABI symbol.
        pub fn terlan_ndarray_v1_reciprocal_where_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, mask: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_remainder` C ABI symbol.
        pub fn terlan_ndarray_v1_remainder(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_repeat` C ABI symbol.
        pub fn terlan_ndarray_v1_repeat(array: *mut TerlanNdArray, repeats: i64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_repeat_counts` C ABI symbol.
        pub fn terlan_ndarray_v1_repeat_counts(array: *mut TerlanNdArray, repeats: *const i64, repeat_count: i64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_repeat_flat` C ABI symbol.
        pub fn terlan_ndarray_v1_repeat_flat(array: *mut TerlanNdArray, repeats: *const i64, repeat_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_reshape` C ABI symbol.
        pub fn terlan_ndarray_v1_reshape(array: *mut TerlanNdArray, shape: *const i64, rank: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_reshape_ordered` C ABI symbol.
        pub fn terlan_ndarray_v1_reshape_ordered(array: *mut TerlanNdArray, shape: *const i64, rank: i64, order: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_resize` C ABI symbol.
        pub fn terlan_ndarray_v1_resize(array: *mut TerlanNdArray, shape: *const i64, rank: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_resize_in_place` C ABI symbol.
        pub fn terlan_ndarray_v1_resize_in_place(array: *mut TerlanNdArray, shape: *const i64, rank: i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_rfft` C ABI symbol.
        pub fn terlan_ndarray_v1_rfft(array: *mut TerlanNdArray, axis: i64, normalization: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_rfft_sized` C ABI symbol.
        pub fn terlan_ndarray_v1_rfft_sized(array: *mut TerlanNdArray, length: i64, axis: i64, normalization: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_rfftfreq` C ABI symbol.
        pub fn terlan_ndarray_v1_rfftfreq(length: i64, spacing: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_rfftn` C ABI symbol.
        pub fn terlan_ndarray_v1_rfftn(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, normalization: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_right_shift` C ABI symbol.
        pub fn terlan_ndarray_v1_right_shift(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_roll` C ABI symbol.
        pub fn terlan_ndarray_v1_roll(array: *mut TerlanNdArray, shift: i64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_roll_axes` C ABI symbol.
        pub fn terlan_ndarray_v1_roll_axes(array: *mut TerlanNdArray, shifts: *const i64, shift_count: i64, axes: *const i64, axis_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_roll_axes_by` C ABI symbol.
        pub fn terlan_ndarray_v1_roll_axes_by(array: *mut TerlanNdArray, shift: i64, axes: *const i64, axis_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_roll_flat` C ABI symbol.
        pub fn terlan_ndarray_v1_roll_flat(array: *mut TerlanNdArray, shift: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_rollaxis` C ABI symbol.
        pub fn terlan_ndarray_v1_rollaxis(array: *mut TerlanNdArray, axis: i64, start: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_rot90` C ABI symbol.
        pub fn terlan_ndarray_v1_rot90(array: *mut TerlanNdArray, rotations: i64, axis_a: i64, axis_b: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_round` C ABI symbol.
        pub fn terlan_ndarray_v1_round(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_round_decimals` C ABI symbol.
        pub fn terlan_ndarray_v1_round_decimals(array: *mut TerlanNdArray, decimals: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_round_into` C ABI symbol.
        pub fn terlan_ndarray_v1_round_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, decimals: i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_row_stack` C ABI symbol.
        pub fn terlan_ndarray_v1_row_stack(arrays: *const u64, array_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_save_npy` C ABI symbol.
        pub fn terlan_ndarray_v1_save_npy(array: *mut TerlanNdArray, path: *const std::ffi::c_char) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_searchsorted` C ABI symbol.
        pub fn terlan_ndarray_v1_searchsorted(sorted: *mut TerlanNdArray, values: *mut TerlanNdArray, right: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_searchsorted_with_sorter` C ABI symbol.
        pub fn terlan_ndarray_v1_searchsorted_with_sorter(array: *mut TerlanNdArray, values: *mut TerlanNdArray, sorter: *mut TerlanNdArray, right: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_select` C ABI symbol.
        pub fn terlan_ndarray_v1_select(conditions: *const u64, condition_count: i64, choices: *const u64, choice_count: i64, default_value: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_serialize` C ABI symbol.
        pub fn terlan_ndarray_v1_serialize(array: *mut TerlanNdArray, out_bytes: *mut *mut u8, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_set_bool` C ABI symbol.
        pub fn terlan_ndarray_v1_set_bool(array: *mut TerlanNdArray, indices: *const i64, index_count: i64, value: bool) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_set_difference` C ABI symbol.
        pub fn terlan_ndarray_v1_set_difference(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_set_difference_unique` C ABI symbol.
        pub fn terlan_ndarray_v1_set_difference_unique(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_set_float` C ABI symbol.
        pub fn terlan_ndarray_v1_set_float(array: *mut TerlanNdArray, indices: *const i64, index_count: i64, value: f64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_set_int` C ABI symbol.
        pub fn terlan_ndarray_v1_set_int(array: *mut TerlanNdArray, indices: *const i64, index_count: i64, value: i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_set_xor` C ABI symbol.
        pub fn terlan_ndarray_v1_set_xor(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_shape` C ABI symbol.
        pub fn terlan_ndarray_v1_shape(array: *mut TerlanNdArray, out_shape: *mut *mut i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_shares_memory` C ABI symbol.
        pub fn terlan_ndarray_v1_shares_memory(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_value: *mut bool) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_shares_memory_with_work` C ABI symbol.
        pub fn terlan_ndarray_v1_shares_memory_with_work(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, max_work: i64, out_value: *mut bool) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_sign` C ABI symbol.
        pub fn terlan_ndarray_v1_sign(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_sign_bit` C ABI symbol.
        pub fn terlan_ndarray_v1_sign_bit(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_sin` C ABI symbol.
        pub fn terlan_ndarray_v1_sin(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_sinc` C ABI symbol.
        pub fn terlan_ndarray_v1_sinc(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_singular_values` C ABI symbol.
        pub fn terlan_ndarray_v1_singular_values(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_sinh` C ABI symbol.
        pub fn terlan_ndarray_v1_sinh(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_slice_axis` C ABI symbol.
        pub fn terlan_ndarray_v1_slice_axis(array: *mut TerlanNdArray, axis: i64, start: i64, stop: i64, step: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_sliding_window_view` C ABI symbol.
        pub fn terlan_ndarray_v1_sliding_window_view(array: *mut TerlanNdArray, window_shape: *const i64, window_count: i64, axes: *const i64, axis_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_slogdet` C ABI symbol.
        pub fn terlan_ndarray_v1_slogdet(array: *mut TerlanNdArray, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_solve` C ABI symbol.
        pub fn terlan_ndarray_v1_solve(matrix: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
}
