unsafe extern "C" {
        /// Invokes the raw `terlan_ndarray_v1_is_infinite` C ABI symbol.
        pub fn terlan_ndarray_v1_is_infinite(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_is_nan` C ABI symbol.
        pub fn terlan_ndarray_v1_is_nan(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_isclose` C ABI symbol.
        pub fn terlan_ndarray_v1_isclose(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, relative_tolerance: f64, absolute_tolerance: f64, equal_nan: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_iscomplex` C ABI symbol.
        pub fn terlan_ndarray_v1_iscomplex(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_iscomplexobj` C ABI symbol.
        pub fn terlan_ndarray_v1_iscomplexobj(array: *mut TerlanNdArray, out_value: *mut bool) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_isneginf` C ABI symbol.
        pub fn terlan_ndarray_v1_isneginf(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_isposinf` C ABI symbol.
        pub fn terlan_ndarray_v1_isposinf(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_isreal` C ABI symbol.
        pub fn terlan_ndarray_v1_isreal(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_isrealobj` C ABI symbol.
        pub fn terlan_ndarray_v1_isrealobj(array: *mut TerlanNdArray, out_value: *mut bool) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_item_bool` C ABI symbol.
        pub fn terlan_ndarray_v1_item_bool(array: *mut TerlanNdArray, out_value: *mut bool) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_item_int` C ABI symbol.
        pub fn terlan_ndarray_v1_item_int(array: *mut TerlanNdArray, out_value: *mut i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_ix` C ABI symbol.
        pub fn terlan_ndarray_v1_ix(first: *mut TerlanNdArray, remaining: *const u64, remaining_count: i64, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_kaiser` C ABI symbol.
        pub fn terlan_ndarray_v1_kaiser(length: i64, beta: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_kron` C ABI symbol.
        pub fn terlan_ndarray_v1_kron(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_lcm` C ABI symbol.
        pub fn terlan_ndarray_v1_lcm(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_ldexp` C ABI symbol.
        pub fn terlan_ndarray_v1_ldexp(values: *mut TerlanNdArray, exponents: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_least_squares` C ABI symbol.
        pub fn terlan_ndarray_v1_least_squares(matrix: *mut TerlanNdArray, rhs: *mut TerlanNdArray, tolerance: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_left_shift` C ABI symbol.
        pub fn terlan_ndarray_v1_left_shift(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_leg2poly` C ABI symbol.
        pub fn terlan_ndarray_v1_leg2poly(coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legadd` C ABI symbol.
        pub fn terlan_ndarray_v1_legadd(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legcompanion` C ABI symbol.
        pub fn terlan_ndarray_v1_legcompanion(coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legder` C ABI symbol.
        pub fn terlan_ndarray_v1_legder(coefficients: *mut TerlanNdArray, order: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legder_axis` C ABI symbol.
        pub fn terlan_ndarray_v1_legder_axis(coefficients: *mut TerlanNdArray, order: i64, scale: f64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legdiv` C ABI symbol.
        pub fn terlan_ndarray_v1_legdiv(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legfit` C ABI symbol.
        pub fn terlan_ndarray_v1_legfit(coordinates: *mut TerlanNdArray, values: *mut TerlanNdArray, degree: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legfit_degrees` C ABI symbol.
        pub fn terlan_ndarray_v1_legfit_degrees(coordinates: *mut TerlanNdArray, values: *mut TerlanNdArray, degrees: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legfit_weighted_degrees` C ABI symbol.
        pub fn terlan_ndarray_v1_legfit_weighted_degrees(coordinates: *mut TerlanNdArray, values: *mut TerlanNdArray, degrees: *mut TerlanNdArray, weights: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legfromroots` C ABI symbol.
        pub fn terlan_ndarray_v1_legfromroots(roots: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_leggauss` C ABI symbol.
        pub fn terlan_ndarray_v1_leggauss(degree: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_leggrid2d` C ABI symbol.
        pub fn terlan_ndarray_v1_leggrid2d(x: *mut TerlanNdArray, y: *mut TerlanNdArray, coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_leggrid3d` C ABI symbol.
        pub fn terlan_ndarray_v1_leggrid3d(x: *mut TerlanNdArray, y: *mut TerlanNdArray, z: *mut TerlanNdArray, coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legint` C ABI symbol.
        pub fn terlan_ndarray_v1_legint(coefficients: *mut TerlanNdArray, order: i64, constant: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legint_axis` C ABI symbol.
        pub fn terlan_ndarray_v1_legint_axis(coefficients: *mut TerlanNdArray, order: i64, constants: *mut TerlanNdArray, lower_bound: f64, scale: f64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legline` C ABI symbol.
        pub fn terlan_ndarray_v1_legline(offset: f64, scale: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legmul` C ABI symbol.
        pub fn terlan_ndarray_v1_legmul(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legmulx` C ABI symbol.
        pub fn terlan_ndarray_v1_legmulx(coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legpow` C ABI symbol.
        pub fn terlan_ndarray_v1_legpow(coefficients: *mut TerlanNdArray, exponent: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legroots` C ABI symbol.
        pub fn terlan_ndarray_v1_legroots(coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legsub` C ABI symbol.
        pub fn terlan_ndarray_v1_legsub(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legtrim` C ABI symbol.
        pub fn terlan_ndarray_v1_legtrim(coefficients: *mut TerlanNdArray, tolerance: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legval` C ABI symbol.
        pub fn terlan_ndarray_v1_legval(coordinates: *mut TerlanNdArray, coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legval2d` C ABI symbol.
        pub fn terlan_ndarray_v1_legval2d(x: *mut TerlanNdArray, y: *mut TerlanNdArray, coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legval3d` C ABI symbol.
        pub fn terlan_ndarray_v1_legval3d(x: *mut TerlanNdArray, y: *mut TerlanNdArray, z: *mut TerlanNdArray, coefficients: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legval_tensor` C ABI symbol.
        pub fn terlan_ndarray_v1_legval_tensor(coordinates: *mut TerlanNdArray, coefficients: *mut TerlanNdArray, tensor: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legvander` C ABI symbol.
        pub fn terlan_ndarray_v1_legvander(coordinates: *mut TerlanNdArray, degree: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legvander2d` C ABI symbol.
        pub fn terlan_ndarray_v1_legvander2d(x: *mut TerlanNdArray, y: *mut TerlanNdArray, x_degree: i64, y_degree: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legvander3d` C ABI symbol.
        pub fn terlan_ndarray_v1_legvander3d(x: *mut TerlanNdArray, y: *mut TerlanNdArray, z: *mut TerlanNdArray, x_degree: i64, y_degree: i64, z_degree: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_legweight` C ABI symbol.
        pub fn terlan_ndarray_v1_legweight(coordinates: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
}
