#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static void assert_ints(TerlanNdArrayHandle array, const int64_t *expected,
                        size_t count) {
  int64_t *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_ints_cast(array, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(values[index] == expected[index]);
  }
  terlan_ndarray_v1_delete_int_values(values);
}

static void assert_floats(TerlanNdArrayHandle array, const double *expected,
                          size_t count) {
  double *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_floats_cast(array, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert((isnan(values[index]) && isnan(expected[index])) ||
           fabs(values[index] - expected[index]) < 1e-12);
  }
  terlan_ndarray_v1_delete_float_values(values);
}

static void assert_complex(TerlanNdArrayHandle array,
                           const double *expected_real,
                           const double *expected_imag, size_t count) {
  double *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_complex_reals(array, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(fabs(values[index] - expected_real[index]) < 1e-12);
  }
  terlan_ndarray_v1_delete_float_values(values);
  assert(terlan_ndarray_v1_complex_imags(array, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(fabs(values[index] - expected_imag[index]) < 1e-12);
  }
  terlan_ndarray_v1_delete_float_values(values);
}

static void assert_bools(TerlanNdArrayHandle array, const uint8_t *expected,
                         size_t count) {
  uint8_t *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_bools(array, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(values[index] == expected[index]);
  }
  terlan_ndarray_v1_delete_bool_values(values);
}

static void test_trailing_axis_broadcast_and_promotion(void) {
  const int64_t matrix_values[] = {1, 2, 3, 4, 5, 6};
  const int64_t vector_values[] = {10, 20, 30};
  const int64_t matrix_shape[] = {2, 3};
  const int64_t vector_shape[] = {3};
  const int64_t expected[] = {11, 22, 33, 14, 25, 36};
  const double scalar_value[] = {0.5};
  const double promoted[] = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5};
  TerlanNdArrayHandle matrix = NULL;
  TerlanNdArrayHandle vector = NULL;
  TerlanNdArrayHandle scalar = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_from_ints_as(matrix_values, 6, matrix_shape, 2,
                                         TERLAN_NDARRAY_DTYPE_INT16,
                                         &matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints_as(vector_values, 3, vector_shape, 1,
                                         TERLAN_NDARRAY_DTYPE_UINT8,
                                         &vector) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_add(matrix, vector, &result) == TERLAN_NDARRAY_OK);
  assert_ints(result, expected, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(scalar_value, 1, NULL, 0, &scalar) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_add(matrix, scalar, &result) == TERLAN_NDARRAY_OK);
  assert_floats(result, promoted, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(vector) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(matrix) == TERLAN_NDARRAY_OK);
}

static void test_broadcast_view_is_read_only_and_handles_zero_dimensions(void) {
  const int64_t values[] = {7, 8, 9};
  const int64_t shape[] = {1, 3};
  const int64_t target_shape[] = {2, 3};
  const int64_t expected[] = {7, 8, 9, 7, 8, 9};
  const int64_t index[] = {1, 0};
  const int64_t empty_shape[] = {0, 3};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle view = NULL;
  TerlanNdArrayHandle empty = NULL;
  TerlanNdArrayHandle empty_view = NULL;
  assert(terlan_ndarray_v1_from_ints(values, 3, shape, 2, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_broadcast_to(source, target_shape, 2, &view) ==
         TERLAN_NDARRAY_OK);
  assert_ints(view, expected, 6);
  assert(terlan_ndarray_v1_set_int(view, index, 2, 99) ==
         TERLAN_NDARRAY_READ_ONLY);
  assert(terlan_ndarray_v1_from_ints(NULL, 0, empty_shape, 2, &empty) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_broadcast_to(empty, empty_shape, 2, &empty_view) ==
         TERLAN_NDARRAY_OK);
  assert_ints(empty_view, NULL, 0);
  assert(terlan_ndarray_v1_delete(empty_view) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(view) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_division_power_comparison_and_logic(void) {
  const int64_t lhs_values[] = {-3, -2, 2, 3};
  const int64_t rhs_values[] = {2, 2, 2, 2};
  const int64_t shape[] = {4};
  const int64_t floors[] = {-2, -1, 1, 1};
  const int64_t remainders[] = {1, 0, 0, 1};
  const int64_t powers[] = {9, 4, 4, 9};
  const double quotients[] = {-1.5, -1.0, 1.0, 1.5};
  const uint8_t comparisons[] = {1, 1, 0, 0};
  const uint8_t bool_lhs[] = {1, 1, 0, 0};
  const uint8_t bool_rhs[] = {1, 0, 1, 0};
  const uint8_t and_values[] = {1, 0, 0, 0};
  const uint8_t xor_values[] = {0, 1, 1, 0};
  TerlanNdArrayHandle lhs = NULL;
  TerlanNdArrayHandle rhs = NULL;
  TerlanNdArrayHandle left_bools = NULL;
  TerlanNdArrayHandle right_bools = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_from_ints(lhs_values, 4, shape, 1, &lhs) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(rhs_values, 4, shape, 1, &rhs) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_true_divide(lhs, rhs, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, quotients, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_floor_divide(lhs, rhs, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, floors, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_remainder(lhs, rhs, &result) == TERLAN_NDARRAY_OK);
  assert_ints(result, remainders, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_power(lhs, rhs, &result) == TERLAN_NDARRAY_OK);
  assert_ints(result, powers, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_less(lhs, rhs, &result) == TERLAN_NDARRAY_OK);
  assert_bools(result, comparisons, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_bools(bool_lhs, 4, shape, 1, &left_bools) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_bools(bool_rhs, 4, shape, 1, &right_bools) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_logical_and(left_bools, right_bools, &result) ==
         TERLAN_NDARRAY_OK);
  assert_bools(result, and_values, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_logical_xor(left_bools, right_bools, &result) ==
         TERLAN_NDARRAY_OK);
  assert_bools(result, xor_values, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(right_bools) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(left_bools) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(rhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(lhs) == TERLAN_NDARRAY_OK);
}

static void test_incompatible_and_zero_division_fail_cleanly(void) {
  const int64_t left_shape[] = {2, 3};
  const int64_t right_shape[] = {2};
  const int64_t left_values[] = {1, 2, 3, 4, 5, 6};
  const int64_t right_values[] = {1, 2};
  const int64_t zero[] = {0};
  TerlanNdArrayHandle left = NULL;
  TerlanNdArrayHandle right = NULL;
  TerlanNdArrayHandle divisor = NULL;
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_from_ints(left_values, 6, left_shape, 2, &left) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(right_values, 2, right_shape, 1, &right) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_add(left, right, &output) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(output == NULL);
  assert(terlan_ndarray_v1_from_ints(zero, 1, NULL, 0, &divisor) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_true_divide(left, divisor, &output) ==
         TERLAN_NDARRAY_DIVIDE_BY_ZERO);
  assert(output == NULL);
  assert(terlan_ndarray_v1_delete(divisor) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(right) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(left) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_where_broadcasts_condition_and_promotes_branches(void) {
  const uint8_t condition_values[] = {1, 0, 1};
  const int64_t vector_shape[] = {3};
  const int64_t true_values[] = {1, 2, 3, 4, 5, 6};
  const int64_t matrix_shape[] = {2, 3};
  const double false_value[] = {0.5};
  const double expected[] = {1.0, 0.5, 3.0, 4.0, 0.5, 6.0};
  TerlanNdArrayHandle condition = NULL;
  TerlanNdArrayHandle when_true = NULL;
  TerlanNdArrayHandle when_false = NULL;
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayHandle column_condition = NULL;
  TerlanNdArrayHandle row_values = NULL;
  TerlanNdArrayHandle numeric_condition = NULL;
  TerlanNdArrayHandle integer_scalar = NULL;
  TerlanNdArrayHandle incompatible = NULL;
  TerlanNdArrayHandle scalar_condition = NULL;
  const uint8_t column_values[] = {1, 0};
  const int64_t column_shape[] = {2, 1};
  const int64_t row_shape[] = {1, 3};
  const int64_t row_data[] = {7, 8, 9};
  const double column_expected[] = {7.0, 8.0, 9.0, 0.5, 0.5, 0.5};
  const int64_t numeric_values[] = {0, 34242324, 0};
  const int64_t integer_value[] = {1};
  const double numeric_expected[] = {0.5, 1.0, 0.5};
  const int64_t incompatible_shape[] = {2, 2};
  assert(terlan_ndarray_v1_from_bools(condition_values, 3, vector_shape, 1,
                                      &condition) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(true_values, 6, matrix_shape, 2,
                                     &when_true) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(false_value, 1, NULL, 0, &when_false) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_where(condition, when_true, when_false, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, expected, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_bools(column_values, 2, column_shape, 2,
                                      &column_condition) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(row_data, 3, row_shape, 2, &row_values) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_where(column_condition, row_values, when_false,
                                 &result) == TERLAN_NDARRAY_OK);
  assert_floats(result, column_expected, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_ints(numeric_values, 3, vector_shape, 1,
                                     &numeric_condition) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints_as(
             integer_value, 1, NULL, 0, TERLAN_NDARRAY_DTYPE_UINT32,
             &integer_scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_where(numeric_condition, integer_scalar, when_false,
                                 &result) == TERLAN_NDARRAY_OK);
  assert_floats(result, numeric_expected, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_ints(true_values, 4, incompatible_shape, 2,
                                     &incompatible) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_where(condition, incompatible, incompatible,
                                 &result) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(result == NULL);
  assert(terlan_ndarray_v1_from_bools((const uint8_t[]){1}, 1, NULL, 0,
                                      &scalar_condition) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_where(scalar_condition, when_true, incompatible,
                                 &result) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(result == NULL);

  assert(terlan_ndarray_v1_delete(scalar_condition) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(incompatible) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(integer_scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(numeric_condition) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(row_values) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(column_condition) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(when_false) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(when_true) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(condition) == TERLAN_NDARRAY_OK);
}

static void test_extrema_and_clip_broadcast_with_nan_policy(void) {
  const double values[] = {NAN, -2.0, 5.0};
  const double limits[] = {1.0, 4.0, 3.0};
  const double zero[] = {0.0};
  const int64_t shape[] = {3};
  const double minimums[] = {NAN, -2.0, 3.0};
  const double finite_minimums[] = {1.0, -2.0, 3.0};
  const double clipped[] = {NAN, 0.0, 3.0};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle upper = NULL;
  TerlanNdArrayHandle lower = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_from_floats(values, 3, shape, 1, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(limits, 3, shape, 1, &upper) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(zero, 1, NULL, 0, &lower) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_elementwise_minimum(array, upper, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, minimums, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_fmin(array, upper, &result) == TERLAN_NDARRAY_OK);
  assert_floats(result, finite_minimums, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip(array, lower, upper, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, clipped, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(lower) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(upper) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_clip_bounds_promotion_strides_and_mutation(void) {
  const int64_t source_shape[] = {2, 6};
  const int64_t source_values[] = {
      -5, 0, 5, 10, 15, 20, -4, 1, 6, 11, 16, 21};
  const int64_t bound_shape[] = {3};
  const double lower_values[] = {-1.0, 4.0, 12.0};
  const double upper_values[] = {3.0, 5.0, 15.0};
  const double nan_value[] = {NAN};
  const double expected_two_sided[] = {-1.0, 5.0, 15.0, -1.0, 5.0, 15.0};
  const double expected_lower[] = {-1.0, 5.0, 15.0, -1.0, 6.0, 16.0};
  const double expected_upper[] = {-5.0, 5.0, 15.0, -4.0, 5.0, 15.0};
  const double expected_nan[] = {-5.0, 5.0, 15.0, -4.0, 6.0, 16.0};
  const int64_t destination_shape[] = {3, 2};
  const int64_t strided_shape[] = {2, 3};
  const int64_t zero_values[] = {0, 0, 0, 0, 0, 0};
  const int64_t expected_destination[] = {-1, 5, 15, -1, 5, 15};
  const int64_t narrow_values[] = {2, 0};
  const int64_t narrow_cast_expected[] = {2, 1};
  const double fractional_values[] = {2.0, 1.5};
  const int64_t narrow_shape[] = {2};
  const int64_t wrong_shape[] = {6};
  const int64_t overlap_values[] = {0, 1, 2, 3, 4, 5};
  const int64_t overlap_expected[] = {0, 1, 1, 2, 3, 3};
  const int64_t strided_inplace_expected[] = {-1, 5, 15, -1, 5, 15};
  const int64_t overlap_shape[] = {6};
  const int64_t scalar_zero[] = {0};
  const int64_t scalar_one[] = {1};
  const int64_t scalar_three[] = {3};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle strided = NULL;
  TerlanNdArrayHandle lower = NULL;
  TerlanNdArrayHandle upper = NULL;
  TerlanNdArrayHandle nan = NULL;
  TerlanNdArrayHandle clipped = NULL;
  TerlanNdArrayHandle destination_backing = NULL;
  TerlanNdArrayHandle destination = NULL;
  TerlanNdArrayHandle narrow = NULL;
  TerlanNdArrayHandle fractional = NULL;
  TerlanNdArrayHandle wrong = NULL;
  TerlanNdArrayHandle scalar = NULL;
  TerlanNdArrayHandle read_only = NULL;
  TerlanNdArrayHandle overlap = NULL;
  TerlanNdArrayHandle overlap_source = NULL;
  TerlanNdArrayHandle overlap_destination = NULL;
  TerlanNdArrayHandle one = NULL;
  TerlanNdArrayHandle three = NULL;
  int32_t dtype = 0;

  assert(terlan_ndarray_v1_from_ints(source_values, 12, source_shape, 2,
                                    &source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_slice_axis(source, 1, 0, 6, 2, &strided) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(lower_values, 3, bound_shape, 1,
                                      &lower) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(upper_values, 3, bound_shape, 1,
                                      &upper) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(nan_value, 1, NULL, 0, &nan) ==
         TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_clip(strided, lower, upper, &clipped) ==
         TERLAN_NDARRAY_OK);
  assert_floats(clipped, expected_two_sided, 6);
  assert(terlan_ndarray_v1_dtype(clipped, &dtype) == TERLAN_NDARRAY_OK);
  assert(dtype == TERLAN_NDARRAY_DTYPE_FLOAT64);
  assert(terlan_ndarray_v1_delete(clipped) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip_lower(strided, lower, &clipped) ==
         TERLAN_NDARRAY_OK);
  assert_floats(clipped, expected_lower, 6);
  assert(terlan_ndarray_v1_delete(clipped) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip_upper(strided, upper, &clipped) ==
         TERLAN_NDARRAY_OK);
  assert_floats(clipped, expected_upper, 6);
  assert(terlan_ndarray_v1_delete(clipped) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip_lower(strided, nan, &clipped) ==
         TERLAN_NDARRAY_OK);
  assert_floats(clipped, expected_nan, 6);
  assert(terlan_ndarray_v1_delete(clipped) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip_upper(strided, nan, &clipped) ==
         TERLAN_NDARRAY_OK);
  assert_floats(clipped, expected_nan, 6);
  assert(terlan_ndarray_v1_delete(clipped) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip(strided, nan, nan, &clipped) ==
         TERLAN_NDARRAY_OK);
  assert_floats(clipped, expected_nan, 6);
  assert(terlan_ndarray_v1_delete(clipped) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip(strided, lower, nan, &clipped) ==
         TERLAN_NDARRAY_OK);
  assert_floats(clipped, expected_lower, 6);
  assert(terlan_ndarray_v1_delete(clipped) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip(strided, nan, upper, &clipped) ==
         TERLAN_NDARRAY_OK);
  assert_floats(clipped, expected_upper, 6);
  assert(terlan_ndarray_v1_delete(clipped) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_ints_as(
             zero_values, 6, destination_shape, 2,
             TERLAN_NDARRAY_DTYPE_INT32, &destination_backing) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_transpose(destination_backing, 0, 1,
                                    &destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip_into(destination, strided, lower, upper) ==
         TERLAN_NDARRAY_OK);
  assert_ints(destination, expected_destination, 6);

  assert(terlan_ndarray_v1_from_ints_as(
             narrow_values, 2, narrow_shape, 1, TERLAN_NDARRAY_DTYPE_INT32,
             &narrow) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(fractional_values, 2, narrow_shape, 1,
                                      &fractional) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip_lower_into(narrow, fractional, lower) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert_ints(narrow, narrow_values, 2);
  assert(terlan_ndarray_v1_clip_lower_into(narrow, fractional, nan) ==
         TERLAN_NDARRAY_OK);
  assert_ints(narrow, narrow_cast_expected, 2);

  assert(terlan_ndarray_v1_from_ints(zero_values, 6, wrong_shape, 1, &wrong) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip_into(wrong, strided, lower, upper) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(terlan_ndarray_v1_from_ints(scalar_zero, 1, NULL, 0, &scalar) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_broadcast_to(scalar, strided_shape, 2, &read_only) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip_into(read_only, strided, lower, upper) ==
         TERLAN_NDARRAY_READ_ONLY);

  assert(terlan_ndarray_v1_from_ints(overlap_values, 6, overlap_shape, 1,
                                    &overlap) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_slice_axis(overlap, 0, 0, 5, 1,
                                     &overlap_source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_slice_axis(overlap, 0, 1, 6, 1,
                                     &overlap_destination) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(scalar_one, 1, NULL, 0, &one) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(scalar_three, 1, NULL, 0, &three) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip_into(overlap_destination, overlap_source, one,
                                    three) == TERLAN_NDARRAY_OK);
  assert_ints(overlap, overlap_expected, 6);
  assert(terlan_ndarray_v1_clip_inplace(strided, lower, upper) ==
         TERLAN_NDARRAY_OK);
  assert_ints(strided, strided_inplace_expected, 6);

  assert(terlan_ndarray_v1_delete(three) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(one) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(overlap_destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(overlap_source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(overlap) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(read_only) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(wrong) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(fractional) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(narrow) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(destination_backing) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(nan) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(upper) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(lower) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(strided) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_clip_scalars_complex_inverted_and_exact_inplace(void) {
  const int64_t shape[] = {3};
  const double floating_values[] = {-2.0, 0.2, 0.8};
  const double floating_minimum[] = {0.1};
  const double floating_maximum[] = {0.6};
  const double floating_expected[] = {0.1, 0.2, 0.6};
  const double real_values[] = {-2.0, 0.0, 2.0};
  const double imag_values[] = {2.0, 0.0, -3.0};
  const double minimum_real[] = {-0.5};
  const double minimum_imag[] = {1.0};
  const double maximum_real[] = {1.0};
  const double maximum_imag[] = {2.0};
  const double expected_real[] = {-0.5, 0.0, 1.0};
  const double expected_imag[] = {1.0, 0.0, 2.0};
  const int64_t integers[] = {-2, 0, 4};
  const int64_t inverted_expected[] = {1, 1, 1};
  const int64_t inplace_expected[] = {-1, 0, 2};
  const int64_t negative_one[] = {-1};
  const int64_t one[] = {1};
  const int64_t two[] = {2};
  const int64_t three[] = {3};
  TerlanNdArrayHandle complex_values = NULL;
  TerlanNdArrayHandle complex_minimum = NULL;
  TerlanNdArrayHandle complex_maximum = NULL;
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle lower = NULL;
  TerlanNdArrayHandle upper = NULL;
  TerlanNdArrayHandle inverted_lower = NULL;
  TerlanNdArrayHandle inverted_upper = NULL;
  TerlanNdArrayHandle floating = NULL;
  TerlanNdArrayHandle floating_lower = NULL;
  TerlanNdArrayHandle floating_upper = NULL;

  assert(terlan_ndarray_v1_from_floats(floating_values, 3, shape, 1,
                                      &floating) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(floating_minimum, 1, NULL, 0,
                                      &floating_lower) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(floating_maximum, 1, NULL, 0,
                                      &floating_upper) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip(floating, floating_lower, floating_upper,
                               &result) == TERLAN_NDARRAY_OK);
  assert_floats(result, floating_expected, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip_inplace(floating, floating_lower,
                                       floating_upper) == TERLAN_NDARRAY_OK);
  assert_floats(floating, floating_expected, 3);

  assert(terlan_ndarray_v1_from_complex(
             real_values, 3, imag_values, 3, shape, 1,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &complex_values) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_complex(
             minimum_real, 1, minimum_imag, 1, NULL, 0,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &complex_minimum) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_complex(
             maximum_real, 1, maximum_imag, 1, NULL, 0,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &complex_maximum) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip(complex_values, complex_minimum,
                               complex_maximum, &result) == TERLAN_NDARRAY_OK);
  assert_complex(result, expected_real, expected_imag, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_ints(integers, 3, shape, 1, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(negative_one, 1, NULL, 0, &lower) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(two, 1, NULL, 0, &upper) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip_inplace(array, lower, upper) ==
         TERLAN_NDARRAY_OK);
  assert_ints(array, inplace_expected, 3);
  assert(terlan_ndarray_v1_from_ints(three, 1, NULL, 0, &inverted_lower) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(one, 1, NULL, 0, &inverted_upper) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip(array, inverted_lower, inverted_upper,
                               &result) == TERLAN_NDARRAY_OK);
  assert_ints(result, inverted_expected, 3);

  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(inverted_upper) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(inverted_lower) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(upper) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(lower) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(complex_maximum) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(complex_minimum) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(complex_values) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(floating_upper) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(floating_lower) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(floating) == TERLAN_NDARRAY_OK);
}

static void test_clip_all_admitted_dtypes_and_one_sided_complex(void) {
  const int32_t dtypes[] = {
      TERLAN_NDARRAY_DTYPE_BOOL,       TERLAN_NDARRAY_DTYPE_INT8,
      TERLAN_NDARRAY_DTYPE_INT16,      TERLAN_NDARRAY_DTYPE_INT32,
      TERLAN_NDARRAY_DTYPE_INT64,      TERLAN_NDARRAY_DTYPE_UINT8,
      TERLAN_NDARRAY_DTYPE_UINT16,     TERLAN_NDARRAY_DTYPE_UINT32,
      TERLAN_NDARRAY_DTYPE_UINT64,     TERLAN_NDARRAY_DTYPE_FLOAT16,
      TERLAN_NDARRAY_DTYPE_FLOAT32,    TERLAN_NDARRAY_DTYPE_FLOAT64,
      TERLAN_NDARRAY_DTYPE_COMPLEX64,  TERLAN_NDARRAY_DTYPE_COMPLEX128};
  const int64_t shape[] = {10};
  const int64_t zeros[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const int64_t ones[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  size_t index;

  for (index = 0; index < sizeof(dtypes) / sizeof(dtypes[0]); ++index) {
    TerlanNdArrayHandle array = NULL;
    TerlanNdArrayHandle lower = NULL;
    TerlanNdArrayHandle upper = NULL;
    TerlanNdArrayHandle result = NULL;
    int32_t dtype = 0;
    assert(terlan_ndarray_v1_ones(shape, 1, dtypes[index], &array) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_full_int(NULL, 0, 1, dtypes[index], &lower) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_full_int(NULL, 0, 0, dtypes[index], &upper) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_clip(array, lower, upper, &result) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_dtype(result, &dtype) == TERLAN_NDARRAY_OK);
    assert(dtype == dtypes[index]);
    assert_ints(result, zeros, 10);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(upper) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(lower) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  }

  {
    TerlanNdArrayHandle array = NULL;
    TerlanNdArrayHandle bound = NULL;
    TerlanNdArrayHandle result = NULL;
    assert(terlan_ndarray_v1_ones(shape, 1, TERLAN_NDARRAY_DTYPE_COMPLEX128,
                                 &array) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_full_int(NULL, 0, 1,
                                     TERLAN_NDARRAY_DTYPE_COMPLEX128,
                                     &bound) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_clip_lower(array, bound, &result) ==
           TERLAN_NDARRAY_OK);
    assert_ints(result, ones, 10);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_clip_upper(array, bound, &result) ==
           TERLAN_NDARRAY_OK);
    assert_ints(result, ones, 10);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(bound) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  }
}

static void test_clip_output_casting_matrix(void) {
  const int64_t shape[] = {3};
  const int64_t integer_values[] = {-3, 0, 5};
  const double floating_values[] = {-2.8, 0.2, 3.7};
  const double lower_float[] = {-2.0};
  const double upper_float[] = {4.0};
  const int64_t lower_int[] = {-1};
  const int64_t upper_int[] = {1};
  const double promoted_expected[] = {-2.0, 0.0, 4.0};
  const int64_t integer_expected[] = {-1, 0, 1};
  const int64_t narrowed_expected[] = {-1, 0, 3};
  TerlanNdArrayHandle integers = NULL;
  TerlanNdArrayHandle floating = NULL;
  TerlanNdArrayHandle lower_f32 = NULL;
  TerlanNdArrayHandle upper_f32 = NULL;
  TerlanNdArrayHandle lower_i32 = NULL;
  TerlanNdArrayHandle upper_i32 = NULL;
  TerlanNdArrayHandle output_f32 = NULL;
  TerlanNdArrayHandle output_i32 = NULL;
  TerlanNdArrayHandle output_i64 = NULL;
  TerlanNdArrayHandle result = NULL;
  int32_t dtype = 0;

  assert(terlan_ndarray_v1_from_ints_as(integer_values, 3, shape, 1,
                                       TERLAN_NDARRAY_DTYPE_INT32,
                                       &integers) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats_as(floating_values, 3, shape, 1,
                                         TERLAN_NDARRAY_DTYPE_FLOAT64,
                                         &floating) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats_as(lower_float, 1, NULL, 0,
                                         TERLAN_NDARRAY_DTYPE_FLOAT32,
                                         &lower_f32) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats_as(upper_float, 1, NULL, 0,
                                         TERLAN_NDARRAY_DTYPE_FLOAT32,
                                         &upper_f32) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints_as(lower_int, 1, NULL, 0,
                                       TERLAN_NDARRAY_DTYPE_INT32,
                                       &lower_i32) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints_as(upper_int, 1, NULL, 0,
                                       TERLAN_NDARRAY_DTYPE_INT32,
                                       &upper_i32) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_clip(integers, lower_f32, upper_f32, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, promoted_expected, 3);
  assert(terlan_ndarray_v1_dtype(result, &dtype) == TERLAN_NDARRAY_OK);
  assert(dtype == TERLAN_NDARRAY_DTYPE_FLOAT64);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_zeros(shape, 1, TERLAN_NDARRAY_DTYPE_FLOAT32,
                                &output_f32) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip_into(output_f32, integers, lower_f32,
                                    upper_f32) == TERLAN_NDARRAY_OK);
  assert_floats(output_f32, promoted_expected, 3);
  assert(terlan_ndarray_v1_clip_into(output_f32, integers, lower_i32,
                                    upper_i32) == TERLAN_NDARRAY_OK);
  assert_ints(output_f32, integer_expected, 3);

  assert(terlan_ndarray_v1_zeros(shape, 1, TERLAN_NDARRAY_DTYPE_INT64,
                                &output_i64) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip_into(output_i64, integers, lower_i32,
                                    upper_i32) == TERLAN_NDARRAY_OK);
  assert_ints(output_i64, integer_expected, 3);

  assert(terlan_ndarray_v1_zeros(shape, 1, TERLAN_NDARRAY_DTYPE_INT32,
                                &output_i32) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_clip_into(output_i32, floating, lower_i32,
                                    upper_f32) == TERLAN_NDARRAY_OK);
  assert_ints(output_i32, narrowed_expected, 3);

  assert(terlan_ndarray_v1_delete(output_i32) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(output_i64) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(output_f32) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(upper_i32) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(lower_i32) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(upper_f32) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(lower_f32) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(floating) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(integers) == TERLAN_NDARRAY_OK);
}

int main(void) {
  test_trailing_axis_broadcast_and_promotion();
  test_broadcast_view_is_read_only_and_handles_zero_dimensions();
  test_division_power_comparison_and_logic();
  test_incompatible_and_zero_division_fail_cleanly();
  test_where_broadcasts_condition_and_promotes_branches();
  test_extrema_and_clip_broadcast_with_nan_policy();
  test_clip_bounds_promotion_strides_and_mutation();
  test_clip_scalars_complex_inverted_and_exact_inplace();
  test_clip_all_admitted_dtypes_and_one_sided_complex();
  test_clip_output_casting_matrix();
  return 0;
}
