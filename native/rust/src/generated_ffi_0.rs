unsafe extern "C" {
        /// Invokes the raw `terlan_ndarray_v1_absolute` C ABI symbol.
        pub fn terlan_ndarray_v1_absolute(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_absolute_into` C ABI symbol.
        pub fn terlan_ndarray_v1_absolute_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_add` C ABI symbol.
        pub fn terlan_ndarray_v1_add(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_add_at` C ABI symbol.
        pub fn terlan_ndarray_v1_add_at(array: *mut TerlanNdArray, indexes: *mut TerlanNdArray, values: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_add_at_coordinates` C ABI symbol.
        pub fn terlan_ndarray_v1_add_at_coordinates(array: *mut TerlanNdArray, coordinates: *mut TerlanNdArray, values: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_add_into` C ABI symbol.
        pub fn terlan_ndarray_v1_add_into(destination: *mut TerlanNdArray, lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_add_reduceat` C ABI symbol.
        pub fn terlan_ndarray_v1_add_reduceat(array: *mut TerlanNdArray, indexes: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_add_reduceat_into` C ABI symbol.
        pub fn terlan_ndarray_v1_add_reduceat_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, indexes: *mut TerlanNdArray, axis: i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_add_where_into` C ABI symbol.
        pub fn terlan_ndarray_v1_add_where_into(destination: *mut TerlanNdArray, lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, mask: *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_alen` C ABI symbol.
        pub fn terlan_ndarray_v1_alen(array: *mut TerlanNdArray, out_value: *mut i64) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_all` C ABI symbol.
        pub fn terlan_ndarray_v1_all(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_all_where` C ABI symbol.
        pub fn terlan_ndarray_v1_all_where(array: *mut TerlanNdArray, mask: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_allclose` C ABI symbol.
        pub fn terlan_ndarray_v1_allclose(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, relative_tolerance: f64, absolute_tolerance: f64, equal_nan: i32, out_value: *mut bool) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_angle` C ABI symbol.
        pub fn terlan_ndarray_v1_angle(array: *mut TerlanNdArray, degrees: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_any` C ABI symbol.
        pub fn terlan_ndarray_v1_any(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_any_where` C ABI symbol.
        pub fn terlan_ndarray_v1_any_where(array: *mut TerlanNdArray, mask: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_append` C ABI symbol.
        pub fn terlan_ndarray_v1_append(array: *mut TerlanNdArray, values: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_arange` C ABI symbol.
        pub fn terlan_ndarray_v1_arange(start: f64, stop: f64, step: f64, dtype: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_arange_float_interval` C ABI symbol.
        pub fn terlan_ndarray_v1_arange_float_interval(start: f64, stop: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_arange_float_step` C ABI symbol.
        pub fn terlan_ndarray_v1_arange_float_step(start: f64, stop: f64, step: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_arange_float_stop` C ABI symbol.
        pub fn terlan_ndarray_v1_arange_float_stop(stop: f64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_arange_int` C ABI symbol.
        pub fn terlan_ndarray_v1_arange_int(start: i64, stop: i64, step: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_arange_int_interval` C ABI symbol.
        pub fn terlan_ndarray_v1_arange_int_interval(start: i64, stop: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_arange_int_stop` C ABI symbol.
        pub fn terlan_ndarray_v1_arange_int_stop(stop: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_arccos` C ABI symbol.
        pub fn terlan_ndarray_v1_arccos(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_arccosh` C ABI symbol.
        pub fn terlan_ndarray_v1_arccosh(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_arcsin` C ABI symbol.
        pub fn terlan_ndarray_v1_arcsin(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_arcsinh` C ABI symbol.
        pub fn terlan_ndarray_v1_arcsinh(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_arctan` C ABI symbol.
        pub fn terlan_ndarray_v1_arctan(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_arctan2` C ABI symbol.
        pub fn terlan_ndarray_v1_arctan2(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_arctanh` C ABI symbol.
        pub fn terlan_ndarray_v1_arctanh(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_argmax` C ABI symbol.
        pub fn terlan_ndarray_v1_argmax(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_argmax_into` C ABI symbol.
        pub fn terlan_ndarray_v1_argmax_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_argmin` C ABI symbol.
        pub fn terlan_ndarray_v1_argmin(array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_argmin_into` C ABI symbol.
        pub fn terlan_ndarray_v1_argmin_into(destination: *mut TerlanNdArray, array: *mut TerlanNdArray, axes: *const i64, axis_count: i64, keep_dims: i32) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_argpartition` C ABI symbol.
        pub fn terlan_ndarray_v1_argpartition(array: *mut TerlanNdArray, kth: i64, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_argpartition_flat` C ABI symbol.
        pub fn terlan_ndarray_v1_argpartition_flat(array: *mut TerlanNdArray, kth: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_argpartition_kths` C ABI symbol.
        pub fn terlan_ndarray_v1_argpartition_kths(array: *mut TerlanNdArray, kths: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_argpartition_kths_flat` C ABI symbol.
        pub fn terlan_ndarray_v1_argpartition_kths_flat(array: *mut TerlanNdArray, kths: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_argsort` C ABI symbol.
        pub fn terlan_ndarray_v1_argsort(array: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_argsort_flat` C ABI symbol.
        pub fn terlan_ndarray_v1_argsort_flat(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_argsort_unstable` C ABI symbol.
        pub fn terlan_ndarray_v1_argsort_unstable(array: *mut TerlanNdArray, axis: i64, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_argwhere` C ABI symbol.
        pub fn terlan_ndarray_v1_argwhere(array: *mut TerlanNdArray, out_array: *mut *mut TerlanNdArray) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_array_equal` C ABI symbol.
        pub fn terlan_ndarray_v1_array_equal(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_value: *mut bool) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_array_equal_nan` C ABI symbol.
        pub fn terlan_ndarray_v1_array_equal_nan(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_value: *mut bool) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_array_equiv` C ABI symbol.
        pub fn terlan_ndarray_v1_array_equiv(lhs: *mut TerlanNdArray, rhs: *mut TerlanNdArray, out_value: *mut bool) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_array_split` C ABI symbol.
        pub fn terlan_ndarray_v1_array_split(array: *mut TerlanNdArray, sections: i64, axis: i64, out_values: *mut *mut u64, out_length: *mut usize) -> i32;
        /// Invokes the raw `terlan_ndarray_v1_arrow_delete` C ABI symbol.
        pub fn terlan_ndarray_v1_arrow_delete(arrow: *mut TerlanNdArrowArray) -> i32;
}
