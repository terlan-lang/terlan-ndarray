unsafe extern "C" {
        /// Invokes the raw `terlan_ndarray_v1_arrow_length` C ABI symbol.
        pub fn terlan_ndarray_v1_arrow_length(arrow: *mut TerlanNdArrowArray, out_length: *mut i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_arrow_null_count` C ABI symbol.
        pub fn terlan_ndarray_v1_arrow_null_count(arrow: *mut TerlanNdArrowArray, out_null_count: *mut i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_as_strided` C ABI symbol.
        pub fn terlan_ndarray_v1_as_strided(array: *mut TerlanNdArray, shape: *const i64, strides: *const i64, shape_count: i64, stride_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_asarray_chkfinite` C ABI symbol.
        pub fn terlan_ndarray_v1_asarray_chkfinite(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_asarray_chkfinite_as` C ABI symbol.
        pub fn terlan_ndarray_v1_asarray_chkfinite_as(array: *mut TerlanNdArray, requested_dtype: i32, order: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_asfarray` C ABI symbol.
        pub fn terlan_ndarray_v1_asfarray(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_astype` C ABI symbol.
        pub fn terlan_ndarray_v1_astype(array: *mut TerlanNdArray, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_astype_ordered` C ABI symbol.
        pub fn terlan_ndarray_v1_astype_ordered(array: *mut TerlanNdArray, dtype: i32, order: i32, copy: i32, casting: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_atleast_1d` C ABI symbol.
        pub fn terlan_ndarray_v1_atleast_1d(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_atleast_2d` C ABI symbol.
        pub fn terlan_ndarray_v1_atleast_2d(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_atleast_3d` C ABI symbol.
        pub fn terlan_ndarray_v1_atleast_3d(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_average` C ABI symbol.
        pub fn terlan_ndarray_v1_average(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_average_weight_sum` C ABI symbol.
        pub fn terlan_ndarray_v1_average_weight_sum(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_average_with_sum` C ABI symbol.
        pub fn terlan_ndarray_v1_average_with_sum(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_bartlett` C ABI symbol.
        pub fn terlan_ndarray_v1_bartlett(length: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_basic_slice` C ABI symbol.
        pub fn terlan_ndarray_v1_basic_slice(array: *mut TerlanNdArray, starts: *const i64, start_count: i64, stops: *const i64, stop_count: i64, steps: *const i64, step_count: i64, ellipsis_index: i64, squeeze_axes: *const i64, squeeze_count: i64, new_axes: *const i64, new_axis_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_bincount` C ABI symbol.
        pub fn terlan_ndarray_v1_bincount(values: *mut TerlanNdArray, weights: *mut TerlanNdArray, minimum_length: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_bincount_unweighted` C ABI symbol.
        pub fn terlan_ndarray_v1_bincount_unweighted(values: *mut TerlanNdArray, minimum_length: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_bitwise_and` C ABI symbol.
        pub fn terlan_ndarray_v1_bitwise_and(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_bitwise_and_reduce` C ABI symbol.
        pub fn terlan_ndarray_v1_bitwise_and_reduce(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_bitwise_not` C ABI symbol.
        pub fn terlan_ndarray_v1_bitwise_not(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_bitwise_or` C ABI symbol.
        pub fn terlan_ndarray_v1_bitwise_or(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_bitwise_or_reduce` C ABI symbol.
        pub fn terlan_ndarray_v1_bitwise_or_reduce(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_bitwise_xor` C ABI symbol.
        pub fn terlan_ndarray_v1_bitwise_xor(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_bitwise_xor_reduce` C ABI symbol.
        pub fn terlan_ndarray_v1_bitwise_xor_reduce(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_blackman` C ABI symbol.
        pub fn terlan_ndarray_v1_blackman(length: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_block` C ABI symbol.
        pub fn terlan_ndarray_v1_block(arrays: *const u64, array_count: i64, layout: *const i64, layout_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_broadcast_arrays` C ABI symbol.
        pub fn terlan_ndarray_v1_broadcast_arrays(first: *mut TerlanNdArray, remaining: *const u64, remaining_count: i64, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_broadcast_shapes` C ABI symbol.
        pub fn terlan_ndarray_v1_broadcast_shapes(lhs: *const i64, lhs_count: i64, rhs: *const i64, rhs_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_broadcast_to` C ABI symbol.
        pub fn terlan_ndarray_v1_broadcast_to(array: *mut TerlanNdArray, shape: *const i64, rank: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_byte_count` C ABI symbol.
        pub fn terlan_ndarray_v1_byte_count(array: *mut TerlanNdArray, out_byte_count: *mut i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_cbrt` C ABI symbol.
        pub fn terlan_ndarray_v1_cbrt(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_ceil` C ABI symbol.
        pub fn terlan_ndarray_v1_ceil(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_cheb2poly` C ABI symbol.
        pub fn terlan_ndarray_v1_cheb2poly(coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebadd` C ABI symbol.
        pub fn terlan_ndarray_v1_chebadd(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebcompanion` C ABI symbol.
        pub fn terlan_ndarray_v1_chebcompanion(coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebder` C ABI symbol.
        pub fn terlan_ndarray_v1_chebder(coefficients: *mut TerlanNdArray, order: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebder_axis` C ABI symbol.
        pub fn terlan_ndarray_v1_chebder_axis(coefficients: *mut TerlanNdArray, order: i64, scale: f64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebdiv` C ABI symbol.
        pub fn terlan_ndarray_v1_chebdiv(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebfit` C ABI symbol.
        pub fn terlan_ndarray_v1_chebfit(coordinates: *mut TerlanNdArray, values: *mut TerlanNdArray, degree: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebfit_degrees` C ABI symbol.
        pub fn terlan_ndarray_v1_chebfit_degrees(coordinates: *mut TerlanNdArray, values: *mut TerlanNdArray, degrees: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebfit_weighted_degrees` C ABI symbol.
        pub fn terlan_ndarray_v1_chebfit_weighted_degrees(coordinates: *mut TerlanNdArray, values: *mut TerlanNdArray, degrees: *mut TerlanNdArray, weights: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebfromroots` C ABI symbol.
        pub fn terlan_ndarray_v1_chebfromroots(roots: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebgauss` C ABI symbol.
        pub fn terlan_ndarray_v1_chebgauss(degree: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebgrid2d` C ABI symbol.
        pub fn terlan_ndarray_v1_chebgrid2d(x: *mut TerlanNdArray, y: *mut TerlanNdArray, coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebgrid3d` C ABI symbol.
        pub fn terlan_ndarray_v1_chebgrid3d(x: *mut TerlanNdArray, y: *mut TerlanNdArray, z: *mut TerlanNdArray, coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebint` C ABI symbol.
        pub fn terlan_ndarray_v1_chebint(coefficients: *mut TerlanNdArray, order: i64, constant: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebint_axis` C ABI symbol.
        pub fn terlan_ndarray_v1_chebint_axis(coefficients: *mut TerlanNdArray, order: i64, constants: *mut TerlanNdArray, lower_bound: f64, scale: f64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
}
