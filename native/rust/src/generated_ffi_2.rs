unsafe extern "C" {
        /// Invokes the raw `terlan_ndarray_v1_chebinterpolate` C ABI symbol.
        pub fn terlan_ndarray_v1_chebinterpolate(values: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebline` C ABI symbol.
        pub fn terlan_ndarray_v1_chebline(offset: f64, scale: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebmul` C ABI symbol.
        pub fn terlan_ndarray_v1_chebmul(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebmulx` C ABI symbol.
        pub fn terlan_ndarray_v1_chebmulx(coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebpow` C ABI symbol.
        pub fn terlan_ndarray_v1_chebpow(coefficients: *mut TerlanNdArray, exponent: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebpts1` C ABI symbol.
        pub fn terlan_ndarray_v1_chebpts1(count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebpts2` C ABI symbol.
        pub fn terlan_ndarray_v1_chebpts2(count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebroots` C ABI symbol.
        pub fn terlan_ndarray_v1_chebroots(coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebsub` C ABI symbol.
        pub fn terlan_ndarray_v1_chebsub(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebtrim` C ABI symbol.
        pub fn terlan_ndarray_v1_chebtrim(coefficients: *mut TerlanNdArray, tolerance: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebval` C ABI symbol.
        pub fn terlan_ndarray_v1_chebval(coordinates: *mut TerlanNdArray, coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebval2d` C ABI symbol.
        pub fn terlan_ndarray_v1_chebval2d(x: *mut TerlanNdArray, y: *mut TerlanNdArray, coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebval3d` C ABI symbol.
        pub fn terlan_ndarray_v1_chebval3d(x: *mut TerlanNdArray, y: *mut TerlanNdArray, z: *mut TerlanNdArray, coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebval_tensor` C ABI symbol.
        pub fn terlan_ndarray_v1_chebval_tensor(coordinates: *mut TerlanNdArray, coefficients: *mut TerlanNdArray, tensor: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebvander` C ABI symbol.
        pub fn terlan_ndarray_v1_chebvander(coordinates: *mut TerlanNdArray, degree: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebvander2d` C ABI symbol.
        pub fn terlan_ndarray_v1_chebvander2d(x: *mut TerlanNdArray, y: *mut TerlanNdArray, x_degree: i64, y_degree: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebvander3d` C ABI symbol.
        pub fn terlan_ndarray_v1_chebvander3d(x: *mut TerlanNdArray, y: *mut TerlanNdArray, z: *mut TerlanNdArray, x_degree: i64, y_degree: i64, z_degree: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_chebweight` C ABI symbol.
        pub fn terlan_ndarray_v1_chebweight(coordinates: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_cholesky` C ABI symbol.
        pub fn terlan_ndarray_v1_cholesky(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_choose` C ABI symbol.
        pub fn terlan_ndarray_v1_choose(indexes: *mut TerlanNdArray, choices: *const u64, choice_count: i64, mode: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_choose_into` C ABI symbol.
        pub fn terlan_ndarray_v1_choose_into(destination: *mut TerlanNdArray, indexes: *mut TerlanNdArray, choices: *const u64, choice_count: i64, mode: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_clip` C ABI symbol.
        pub fn terlan_ndarray_v1_clip(array: *mut TerlanNdArray, minimum: *mut TerlanNdArray, maximum: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_clip_inplace` C ABI symbol.
        pub fn terlan_ndarray_v1_clip_inplace(array: *mut TerlanNdArray, minimum: *mut TerlanNdArray, maximum: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_clip_into` C ABI symbol.
        pub fn terlan_ndarray_v1_clip_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, minimum: *mut TerlanNdArray, maximum: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_clip_lower` C ABI symbol.
        pub fn terlan_ndarray_v1_clip_lower(array: *mut TerlanNdArray, minimum: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_clip_lower_inplace` C ABI symbol.
        pub fn terlan_ndarray_v1_clip_lower_inplace(array: *mut TerlanNdArray, minimum: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_clip_lower_into` C ABI symbol.
        pub fn terlan_ndarray_v1_clip_lower_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, minimum: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_clip_upper` C ABI symbol.
        pub fn terlan_ndarray_v1_clip_upper(array: *mut TerlanNdArray, maximum: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_clip_upper_inplace` C ABI symbol.
        pub fn terlan_ndarray_v1_clip_upper_inplace(array: *mut TerlanNdArray, maximum: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_clip_upper_into` C ABI symbol.
        pub fn terlan_ndarray_v1_clip_upper_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, maximum: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_column_stack` C ABI symbol.
        pub fn terlan_ndarray_v1_column_stack(arrays: *const u64, array_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_complex_imags` C ABI symbol.
        pub fn terlan_ndarray_v1_complex_imags(array: *mut TerlanNdArray, out_values: *mut *mut f64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_complex_reals` C ABI symbol.
        pub fn terlan_ndarray_v1_complex_reals(array: *mut TerlanNdArray, out_values: *mut *mut f64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_compress` C ABI symbol.
        pub fn terlan_ndarray_v1_compress(array: *mut TerlanNdArray, mask: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_compress_flat` C ABI symbol.
        pub fn terlan_ndarray_v1_compress_flat(array: *mut TerlanNdArray, mask: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_compress_flat_into` C ABI symbol.
        pub fn terlan_ndarray_v1_compress_flat_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, mask: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_compress_into` C ABI symbol.
        pub fn terlan_ndarray_v1_compress_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, mask: *mut TerlanNdArray, axis: i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_concatenate` C ABI symbol.
        pub fn terlan_ndarray_v1_concatenate(arrays: *const u64, array_count: i64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_concatenate_flat` C ABI symbol.
        pub fn terlan_ndarray_v1_concatenate_flat(arrays: *const u64, array_count: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_concatenate_flat_into` C ABI symbol.
        pub fn terlan_ndarray_v1_concatenate_flat_into(arrays: *const u64, array_count: i64, destination: *mut TerlanNdArray, casting: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_concatenate_flat_with_dtype` C ABI symbol.
        pub fn terlan_ndarray_v1_concatenate_flat_with_dtype(arrays: *const u64, array_count: i64, dtype: i32, casting: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_concatenate_into` C ABI symbol.
        pub fn terlan_ndarray_v1_concatenate_into(arrays: *const u64, array_count: i64, axis: i64, destination: *mut TerlanNdArray, casting: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_concatenate_with_dtype` C ABI symbol.
        pub fn terlan_ndarray_v1_concatenate_with_dtype(arrays: *const u64, array_count: i64, axis: i64, dtype: i32, casting: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_cond` C ABI symbol.
        pub fn terlan_ndarray_v1_cond(array: *mut TerlanNdArray, order: i64, out_value: *mut f64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_condition` C ABI symbol.
        pub fn terlan_ndarray_v1_condition(array: *mut TerlanNdArray, order: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_conjugate` C ABI symbol.
        pub fn terlan_ndarray_v1_conjugate(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_conjugate_into` C ABI symbol.
        pub fn terlan_ndarray_v1_conjugate_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_contiguous` C ABI symbol.
        pub fn terlan_ndarray_v1_contiguous(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
}
