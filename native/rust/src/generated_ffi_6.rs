unsafe extern "C" {
        /// Invokes the raw `terlan_ndarray_v1_histogram2d` C ABI symbol.
        pub fn terlan_ndarray_v1_histogram2d(x: *mut TerlanNdArray, y: *mut TerlanNdArray, bins: i64, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_histogram2d_edges` C ABI symbol.
        pub fn terlan_ndarray_v1_histogram2d_edges(x: *mut TerlanNdArray, y: *mut TerlanNdArray, x_edges: *mut TerlanNdArray, y_edges: *mut TerlanNdArray, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_histogram2d_range` C ABI symbol.
        pub fn terlan_ndarray_v1_histogram2d_range(x: *mut TerlanNdArray, y: *mut TerlanNdArray, x_bins: i64, y_bins: i64, minima: *mut TerlanNdArray, maxima: *mut TerlanNdArray, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_histogram2d_x_edges` C ABI symbol.
        pub fn terlan_ndarray_v1_histogram2d_x_edges(x: *mut TerlanNdArray, y: *mut TerlanNdArray, x_edges: *mut TerlanNdArray, y_bins: i64, y_lower: f64, y_upper: f64, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_histogram2d_y_edges` C ABI symbol.
        pub fn terlan_ndarray_v1_histogram2d_y_edges(x: *mut TerlanNdArray, y: *mut TerlanNdArray, x_bins: i64, x_lower: f64, x_upper: f64, y_edges: *mut TerlanNdArray, density: bool, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_histogram_bin_edges` C ABI symbol.
        pub fn terlan_ndarray_v1_histogram_bin_edges(array: *mut TerlanNdArray, bins: i64, minimum: f64, maximum: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_histogramdd` C ABI symbol.
        pub fn terlan_ndarray_v1_histogramdd(samples: *mut TerlanNdArray, bins: i64, minima: *mut TerlanNdArray, maxima: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_hsplit` C ABI symbol.
        pub fn terlan_ndarray_v1_hsplit(array: *mut TerlanNdArray, sections: i64, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_hstack` C ABI symbol.
        pub fn terlan_ndarray_v1_hstack(arrays: *const u64, array_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_hypot` C ABI symbol.
        pub fn terlan_ndarray_v1_hypot(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_i0` C ABI symbol.
        pub fn terlan_ndarray_v1_i0(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_identity` C ABI symbol.
        pub fn terlan_ndarray_v1_identity(rows: i64, columns: i64, diagonal: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_ifft` C ABI symbol.
        pub fn terlan_ndarray_v1_ifft(array: *mut TerlanNdArray, axis: i64, normalization: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_ifft_sized` C ABI symbol.
        pub fn terlan_ndarray_v1_ifft_sized(array: *mut TerlanNdArray, length: i64, axis: i64, normalization: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_ifftn` C ABI symbol.
        pub fn terlan_ndarray_v1_ifftn(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, normalization: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_ifftshift` C ABI symbol.
        pub fn terlan_ndarray_v1_ifftshift(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_ihfft` C ABI symbol.
        pub fn terlan_ndarray_v1_ihfft(array: *mut TerlanNdArray, axis: i64, normalization: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_ihfft_sized` C ABI symbol.
        pub fn terlan_ndarray_v1_ihfft_sized(array: *mut TerlanNdArray, length: i64, axis: i64, normalization: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_imag` C ABI symbol.
        pub fn terlan_ndarray_v1_imag(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_import_tensor_packet` C ABI symbol.
        pub fn terlan_ndarray_v1_import_tensor_packet(packet: *const u8, packet_length: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_in1d` C ABI symbol.
        pub fn terlan_ndarray_v1_in1d(elements: *mut TerlanNdArray, test_elements: *mut TerlanNdArray, invert: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_index_axis` C ABI symbol.
        pub fn terlan_ndarray_v1_index_axis(array: *mut TerlanNdArray, axis: i64, index: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_indices` C ABI symbol.
        pub fn terlan_ndarray_v1_indices(shape: *const i64, rank: i64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_indices_sparse` C ABI symbol.
        pub fn terlan_ndarray_v1_indices_sparse(shape: *const i64, rank: i64, dtype: i32, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_inner` C ABI symbol.
        pub fn terlan_ndarray_v1_inner(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_insert` C ABI symbol.
        pub fn terlan_ndarray_v1_insert(array: *mut TerlanNdArray, index: i64, values: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_insert_indexes` C ABI symbol.
        pub fn terlan_ndarray_v1_insert_indexes(array: *mut TerlanNdArray, indexes: *const i64, index_count: i64, values: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_insert_indexes_axis` C ABI symbol.
        pub fn terlan_ndarray_v1_insert_indexes_axis(array: *mut TerlanNdArray, indexes: *const i64, index_count: i64, values: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_insert_scalar_axis` C ABI symbol.
        pub fn terlan_ndarray_v1_insert_scalar_axis(array: *mut TerlanNdArray, index: i64, values: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_internal_overlap` C ABI symbol.
        pub fn terlan_ndarray_v1_internal_overlap(array: *mut TerlanNdArray, out_value: *mut bool) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_interpolate` C ABI symbol.
        pub fn terlan_ndarray_v1_interpolate(values: *mut TerlanNdArray, coordinates: *mut TerlanNdArray, samples: *mut TerlanNdArray, left: f64, right: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_interpolate_boundaries` C ABI symbol.
        pub fn terlan_ndarray_v1_interpolate_boundaries(values: *mut TerlanNdArray, coordinates: *mut TerlanNdArray, samples: *mut TerlanNdArray, left: *mut TerlanNdArray, right: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_interpolate_default` C ABI symbol.
        pub fn terlan_ndarray_v1_interpolate_default(values: *mut TerlanNdArray, coordinates: *mut TerlanNdArray, samples: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_interpolate_periodic` C ABI symbol.
        pub fn terlan_ndarray_v1_interpolate_periodic(values: *mut TerlanNdArray, coordinates: *mut TerlanNdArray, samples: *mut TerlanNdArray, period: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_intersect` C ABI symbol.
        pub fn terlan_ndarray_v1_intersect(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_intersect_all` C ABI symbol.
        pub fn terlan_ndarray_v1_intersect_all(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_inverse` C ABI symbol.
        pub fn terlan_ndarray_v1_inverse(matrix: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_invert` C ABI symbol.
        pub fn terlan_ndarray_v1_invert(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_invert_at` C ABI symbol.
        pub fn terlan_ndarray_v1_invert_at(array: *mut TerlanNdArray, indexes: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_invert_into` C ABI symbol.
        pub fn terlan_ndarray_v1_invert_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_invert_where_into` C ABI symbol.
        pub fn terlan_ndarray_v1_invert_where_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, mask: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_irfft` C ABI symbol.
        pub fn terlan_ndarray_v1_irfft(array: *mut TerlanNdArray, length: i64, axis: i64, normalization: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_irfftn` C ABI symbol.
        pub fn terlan_ndarray_v1_irfftn(array: *mut TerlanNdArray, length: i64, axes: *const i64, axis_count: i64, normalization: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_is_close` C ABI symbol.
        pub fn terlan_ndarray_v1_is_close(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, relative_tolerance: f64, absolute_tolerance: f64, equal_nan: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_is_contiguous` C ABI symbol.
        pub fn terlan_ndarray_v1_is_contiguous(array: *mut TerlanNdArray, out_contiguous: *mut i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_is_finite` C ABI symbol.
        pub fn terlan_ndarray_v1_is_finite(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_is_fortran_contiguous` C ABI symbol.
        pub fn terlan_ndarray_v1_is_fortran_contiguous(array: *mut TerlanNdArray, out_contiguous: *mut i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_is_in` C ABI symbol.
        pub fn terlan_ndarray_v1_is_in(elements: *mut TerlanNdArray, test_elements: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
}
