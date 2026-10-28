unsafe extern "C" {
        /// Invokes the raw `terlan_ndarray_v1_flipud` C ABI symbol.
        pub fn terlan_ndarray_v1_flipud(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_float_power` C ABI symbol.
        pub fn terlan_ndarray_v1_float_power(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_floor` C ABI symbol.
        pub fn terlan_ndarray_v1_floor(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_floor_divide` C ABI symbol.
        pub fn terlan_ndarray_v1_floor_divide(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_flush` C ABI symbol.
        pub fn terlan_ndarray_v1_flush(array: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_fmax` C ABI symbol.
        pub fn terlan_ndarray_v1_fmax(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_fmax_reduce` C ABI symbol.
        pub fn terlan_ndarray_v1_fmax_reduce(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_fmin` C ABI symbol.
        pub fn terlan_ndarray_v1_fmin(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_fmin_reduce` C ABI symbol.
        pub fn terlan_ndarray_v1_fmin_reduce(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_fmod` C ABI symbol.
        pub fn terlan_ndarray_v1_fmod(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_frexp_all` C ABI symbol.
        pub fn terlan_ndarray_v1_frexp_all(array: *mut TerlanNdArray, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_frobenius_norm` C ABI symbol.
        pub fn terlan_ndarray_v1_frobenius_norm(array: *mut TerlanNdArray, out_value: *mut f64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_from_arrow` C ABI symbol.
        pub fn terlan_ndarray_v1_from_arrow(arrow: *mut TerlanNdArrowArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_from_bools` C ABI symbol.
        pub fn terlan_ndarray_v1_from_bools(values: *const u8, value_count: i64, shape: *const i64, rank: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_from_complex` C ABI symbol.
        pub fn terlan_ndarray_v1_from_complex(real_values: *const f64, real_count: i64, imag_values: *const f64, imag_count: i64, shape: *const i64, rank: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_from_floats` C ABI symbol.
        pub fn terlan_ndarray_v1_from_floats(values: *const f64, value_count: i64, shape: *const i64, rank: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_from_floats_as` C ABI symbol.
        pub fn terlan_ndarray_v1_from_floats_as(values: *const f64, value_count: i64, shape: *const i64, rank: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_from_ints` C ABI symbol.
        pub fn terlan_ndarray_v1_from_ints(values: *const i64, value_count: i64, shape: *const i64, rank: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_from_ints_as` C ABI symbol.
        pub fn terlan_ndarray_v1_from_ints_as(values: *const i64, value_count: i64, shape: *const i64, rank: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_from_npy` C ABI symbol.
        pub fn terlan_ndarray_v1_from_npy(bytes: *const u8, length: i64, maximum: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_from_uints` C ABI symbol.
        pub fn terlan_ndarray_v1_from_uints(values: *const u64, value_count: i64, shape: *const i64, rank: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_frombuffer` C ABI symbol.
        pub fn terlan_ndarray_v1_frombuffer(bytes: *const u8, length: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_full` C ABI symbol.
        pub fn terlan_ndarray_v1_full(shape: *const i64, rank: i64, value: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_full_float` C ABI symbol.
        pub fn terlan_ndarray_v1_full_float(shape: *const i64, rank: i64, value: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_full_int` C ABI symbol.
        pub fn terlan_ndarray_v1_full_int(shape: *const i64, rank: i64, value: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_full_like` C ABI symbol.
        pub fn terlan_ndarray_v1_full_like(array: *mut TerlanNdArray, value: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_full_like_float` C ABI symbol.
        pub fn terlan_ndarray_v1_full_like_float(array: *mut TerlanNdArray, value: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_full_like_int` C ABI symbol.
        pub fn terlan_ndarray_v1_full_like_int(array: *mut TerlanNdArray, value: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_gather_coordinates` C ABI symbol.
        pub fn terlan_ndarray_v1_gather_coordinates(array: *mut TerlanNdArray, coordinates: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_gcd` C ABI symbol.
        pub fn terlan_ndarray_v1_gcd(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_geomspace` C ABI symbol.
        pub fn terlan_ndarray_v1_geomspace(start: f64, stop: f64, count: i64, endpoint: i32, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_geomspace_arrays` C ABI symbol.
        pub fn terlan_ndarray_v1_geomspace_arrays(start: *mut TerlanNdArray, stop: *mut TerlanNdArray, count: i64, endpoint: i32, axis: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_get_bool` C ABI symbol.
        pub fn terlan_ndarray_v1_get_bool(array: *mut TerlanNdArray, indices: *const i64, index_count: i64, out_value: *mut bool) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_get_float` C ABI symbol.
        pub fn terlan_ndarray_v1_get_float(array: *mut TerlanNdArray, indices: *const i64, index_count: i64, out_value: *mut f64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_get_int` C ABI symbol.
        pub fn terlan_ndarray_v1_get_int(array: *mut TerlanNdArray, indices: *const i64, index_count: i64, out_value: *mut i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_gradient` C ABI symbol.
        pub fn terlan_ndarray_v1_gradient(array: *mut TerlanNdArray, spacing: f64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_gradient_all` C ABI symbol.
        pub fn terlan_ndarray_v1_gradient_all(array: *mut TerlanNdArray, spacing: f64, edge_order: i32, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_gradient_axes` C ABI symbol.
        pub fn terlan_ndarray_v1_gradient_axes(array: *mut TerlanNdArray, spacings: *const u64, spacing_count: i64, axes: *const i64, axis_count: i64, edge_order: i32, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_gradient_coordinates` C ABI symbol.
        pub fn terlan_ndarray_v1_gradient_coordinates(array: *mut TerlanNdArray, coordinates: *mut TerlanNdArray, axis: i64, edge_order: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_gradient_edge` C ABI symbol.
        pub fn terlan_ndarray_v1_gradient_edge(array: *mut TerlanNdArray, spacing: f64, axis: i64, edge_order: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_gradient_uniform_axes` C ABI symbol.
        pub fn terlan_ndarray_v1_gradient_uniform_axes(array: *mut TerlanNdArray, spacings: *const f64, spacing_count: i64, axes: *const i64, axis_count: i64, edge_order: i32, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_greater` C ABI symbol.
        pub fn terlan_ndarray_v1_greater(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_greater_equal` C ABI symbol.
        pub fn terlan_ndarray_v1_greater_equal(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_hamming` C ABI symbol.
        pub fn terlan_ndarray_v1_hamming(length: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_hanning` C ABI symbol.
        pub fn terlan_ndarray_v1_hanning(length: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_heaviside` C ABI symbol.
        pub fn terlan_ndarray_v1_heaviside(lhs: *mut TerlanNdArray, zero_value: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_hfft` C ABI symbol.
        pub fn terlan_ndarray_v1_hfft(array: *mut TerlanNdArray, length: i64, axis: i64, normalization: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_histogram` C ABI symbol.
        pub fn terlan_ndarray_v1_histogram(array: *mut TerlanNdArray, bins: i64, minimum: f64, maximum: f64, out_array: *mut *mut TerlanNdArray) -> i32;
}
