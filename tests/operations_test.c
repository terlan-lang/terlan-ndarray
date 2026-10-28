#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static void assert_int_values(TerlanNdArrayHandle array,
                              const int64_t *expected, size_t length) {
  int64_t *values = NULL;
  size_t actual_length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_ints(array, &values, &actual_length) ==
         TERLAN_NDARRAY_OK);
  assert(actual_length == length);
  for (index = 0; index < length; ++index) {
    assert(values[index] == expected[index]);
  }
  terlan_ndarray_v1_delete_int_values(values);
}

static void assert_shape(TerlanNdArrayHandle array, const int64_t *expected,
                         int64_t rank) {
  int64_t actual_rank = -1;
  int64_t *actual = NULL;
  int64_t index;
  assert(terlan_ndarray_v1_rank(array, &actual_rank) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_shape(array, &actual) == TERLAN_NDARRAY_OK);
  assert(actual_rank == rank);
  for (index = 0; index < rank; ++index) {
    assert(actual[index] == expected[index]);
  }
}

static void assert_float_values(TerlanNdArrayHandle array,
                                const double *expected, size_t length) {
  double *values = NULL;
  size_t actual_length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_floats(array, &values, &actual_length) ==
         TERLAN_NDARRAY_OK);
  assert(actual_length == length);
  for (index = 0; index < length; ++index) {
    if (isnan(expected[index])) {
      assert(isnan(values[index]));
    } else {
      assert(values[index] == expected[index]);
    }
  }
  terlan_ndarray_v1_delete_float_values(values);
}

static void test_reshape_and_transpose_are_shared_views(void) {
  const int64_t values[] = {1, 2, 3, 4, 5, 6};
  const int64_t source_shape[] = {2, 3};
  const int64_t reshaped_shape[] = {3, 2};
  const int64_t transposed_shape[] = {2, 3};
  const int64_t reshaped_values[] = {99, 2, 3, 4, 5, 6};
  const int64_t transposed_values[] = {99, 3, 5, 88, 4, 6};
  const int64_t source_first[] = {0, 0};
  const int64_t reshaped_second[] = {0, 1};
  const int64_t scalar_value[] = {7};
  const int64_t vector_shape[] = {3};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle reshaped = NULL;
  TerlanNdArrayHandle transposed = NULL;
  TerlanNdArrayHandle scalar = NULL;
  TerlanNdArrayHandle scalar_copy = NULL;
  TerlanNdArrayHandle vector = NULL;
  TerlanNdArrayHandle vector_copy = NULL;

  assert(terlan_ndarray_v1_from_ints(values, 6, source_shape, 2, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_reshape(source, reshaped_shape, 2, &reshaped) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_set_int(source, source_first, 2, 99) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(reshaped, reshaped_values, 6);
  assert(terlan_ndarray_v1_transpose(reshaped, 0, -1, &transposed) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_set_int(reshaped, reshaped_second, 2, 88) ==
         TERLAN_NDARRAY_OK);
  assert_shape(transposed, transposed_shape, 2);
  assert_int_values(transposed, transposed_values, 6);

  assert(terlan_ndarray_v1_from_ints(scalar_value, 1, NULL, 0, &scalar) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_transpose(scalar, 0, 0, &scalar_copy) ==
         TERLAN_NDARRAY_OK);
  assert_shape(scalar_copy, NULL, 0);
  assert_int_values(scalar_copy, scalar_value, 1);
  assert(terlan_ndarray_v1_from_ints(values, 3, vector_shape, 1, &vector) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_transpose(vector, -1, 0, &vector_copy) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(vector_copy, values, 3);

  assert(terlan_ndarray_v1_delete(reshaped) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(vector_copy) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(vector) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(transposed) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(scalar_copy) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_exact_shape_arithmetic_and_rejections(void) {
  const int64_t left_values[] = {1, -2, 3, -4};
  const int64_t right_values[] = {5, 6, -7, -8};
  const int64_t shape[] = {2, 2};
  const int64_t flat_shape[] = {4};
  const int64_t added[] = {6, 4, -4, -12};
  const int64_t subtracted[] = {-4, -8, 10, 4};
  const int64_t multiplied[] = {5, -12, -21, 32};
  const uint8_t bool_values[] = {1, 0, 1, 0};
  const double float_values[] = {1.0, 2.0, 3.0, 4.0};
  const double promoted_values[] = {2.0, 0.0, 6.0, 0.0};
  TerlanNdArrayHandle left = NULL;
  TerlanNdArrayHandle right = NULL;
  TerlanNdArrayHandle flat = NULL;
  TerlanNdArrayHandle bools = NULL;
  TerlanNdArrayHandle floats = NULL;
  TerlanNdArrayHandle result = NULL;

  assert(terlan_ndarray_v1_from_ints(left_values, 4, shape, 2, &left) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(right_values, 4, shape, 2, &right) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(right_values, 4, flat_shape, 1, &flat) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_add(left, right, &result) == TERLAN_NDARRAY_OK);
  assert_int_values(result, added, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_subtract(left, right, &result) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(result, subtracted, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_multiply(left, right, &result) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(result, multiplied, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_add(left, flat, &result) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(result == NULL);
  assert(terlan_ndarray_v1_from_bools(bool_values, 4, shape, 2, &bools) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_add(bools, bools, &result) == TERLAN_NDARRAY_OK);
  {
    uint8_t *actual = NULL;
    size_t length = 0;
    assert(terlan_ndarray_v1_to_bools(result, &actual, &length) ==
           TERLAN_NDARRAY_OK);
    assert(length == 4 && actual[0] == 1 && actual[1] == 0 &&
           actual[2] == 1 && actual[3] == 0);
    terlan_ndarray_v1_delete_bool_values(actual);
  }
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(float_values, 4, shape, 2, &floats) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_add(left, floats, &result) == TERLAN_NDARRAY_OK);
  assert_float_values(result, promoted_values, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_delete(left) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(right) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(flat) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(bools) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(floats) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_arbitrary_rank_transpose_and_float_arithmetic(void) {
  const int64_t cube_values[] = {1,  2,  3,  4, 5,  6,
                                 7,  8,  9, 10, 11, 12};
  const int64_t cube_shape[] = {2, 2, 3};
  const int64_t transposed_shape[] = {3, 2, 2};
  const int64_t transposed_values[] = {1, 7, 4, 10, 2, 8,
                                       5, 11, 3, 9,  6, 12};
  const double left_values[] = {NAN, INFINITY, -INFINITY, 2.0};
  const double right_values[] = {1.0, 2.0, -2.0, 0.5};
  const double added_values[] = {NAN, INFINITY, -INFINITY, 2.5};
  const double multiplied_values[] = {NAN, INFINITY, INFINITY, 1.0};
  const int64_t float_shape[] = {4};
  TerlanNdArrayHandle cube = NULL;
  TerlanNdArrayHandle transposed = NULL;
  TerlanNdArrayHandle left = NULL;
  TerlanNdArrayHandle right = NULL;
  TerlanNdArrayHandle result = NULL;

  assert(terlan_ndarray_v1_from_ints(cube_values, 12, cube_shape, 3, &cube) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_transpose(cube, 0, -1, &transposed) ==
         TERLAN_NDARRAY_OK);
  assert_shape(transposed, transposed_shape, 3);
  assert_int_values(transposed, transposed_values, 12);
  assert(terlan_ndarray_v1_from_floats(left_values, 4, float_shape, 1, &left) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(right_values, 4, float_shape, 1,
                                       &right) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_add(left, right, &result) == TERLAN_NDARRAY_OK);
  assert_float_values(result, added_values, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_multiply(left, right, &result) ==
         TERLAN_NDARRAY_OK);
  assert_float_values(result, multiplied_values, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(cube) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(transposed) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(left) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(right) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_numpy_binop_broadcast_views_and_chains_preserve_inputs(void) {
  const double half_values[] = {0.5, 0.5};
  const double vector_values[] = {1.0, 2.0};
  TerlanNdArrayHandle halves = NULL;
  TerlanNdArrayHandle strided_half = NULL;
  TerlanNdArrayHandle vector = NULL;
  TerlanNdArrayHandle inner = NULL;
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayHandle integer_five = NULL;
  TerlanNdArrayHandle integer_three = NULL;
  TerlanNdArrayHandle square = NULL;

  assert(terlan_ndarray_v1_from_floats(half_values, 2,
                                       (const int64_t[]){2}, 1,
                                       &halves) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_slice_axis(halves, 0, 0, 2, 2,
                                      &strided_half) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(vector_values, 2,
                                       (const int64_t[]){2}, 1,
                                       &vector) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_multiply(strided_half, vector, &inner) ==
         TERLAN_NDARRAY_OK);
  assert_float_values(inner, (const double[]){0.5, 1.0}, 2);
  assert(terlan_ndarray_v1_multiply(strided_half, inner, &result) ==
         TERLAN_NDARRAY_OK);
  assert_float_values(result, (const double[]){0.25, 0.5}, 2);
  assert_float_values(halves, half_values, 2);
  assert_float_values(vector, vector_values, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(inner) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_ints((const int64_t[]){5}, 1,
                                     (const int64_t[]){1}, 1,
                                     &integer_five) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints((const int64_t[]){3}, 1,
                                     (const int64_t[]){1}, 1,
                                     &integer_three) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_multiply(integer_five, integer_five, &square) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_true_divide(square, integer_three, &result) ==
         TERLAN_NDARRAY_OK);
  assert_float_values(result, (const double[]){25.0 / 3.0}, 1);
  assert_int_values(integer_five, (const int64_t[]){5}, 1);
  assert_int_values(integer_three, (const int64_t[]){3}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(square) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(integer_three) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(integer_five) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(vector) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(strided_half) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(halves) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_int64_arithmetic_wraps_like_numpy(void) {
  const int64_t shape[] = {1};
  const int64_t maximum[] = {INT64_MAX};
  const int64_t minimum[] = {INT64_MIN};
  const int64_t one[] = {1};
  const int64_t negative_one[] = {-1};
  TerlanNdArrayHandle max_array = NULL;
  TerlanNdArrayHandle min_array = NULL;
  TerlanNdArrayHandle one_array = NULL;
  TerlanNdArrayHandle negative_one_array = NULL;
  TerlanNdArrayHandle result = NULL;

  assert(terlan_ndarray_v1_from_ints(maximum, 1, shape, 1, &max_array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(minimum, 1, shape, 1, &min_array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(one, 1, shape, 1, &one_array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(negative_one, 1, shape, 1,
                                     &negative_one_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_add(max_array, one_array, &result) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(result, minimum, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_subtract(min_array, one_array, &result) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(result, maximum, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_multiply(min_array, negative_one_array, &result) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(result, minimum, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 4);
  assert(terlan_ndarray_v1_delete(max_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(min_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(one_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(negative_one_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_sum_axes_empty_identity_and_non_finite_values(void) {
  const int64_t values[] = {1, 2, 3, 4, 5, 6};
  const int64_t shape[] = {2, 3};
  const int64_t axis_zero[] = {0};
  const int64_t axis_one[] = {1};
  const int64_t negative_axis[] = {-1};
  const int64_t duplicate_axes[] = {0, -2};
  const int64_t out_of_range[] = {2};
  const int64_t columns[] = {5, 7, 9};
  const int64_t rows[] = {6, 15};
  const int64_t total[] = {21};
  const int64_t kept_shape[] = {2, 1};
  const int64_t empty_shape[] = {2, 0, 3};
  const int64_t empty_sum_shape[] = {2, 3};
  const int64_t empty_sum_values[] = {0, 0, 0, 0, 0, 0};
  const double non_finite[] = {INFINITY, -INFINITY};
  const int64_t two_shape[] = {2};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayHandle empty = NULL;
  TerlanNdArrayHandle floats = NULL;
  double *float_result = NULL;
  size_t float_length = 0;

  assert(terlan_ndarray_v1_from_ints(values, 6, shape, 2, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sum(array, axis_zero, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(result, columns, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sum(array, axis_one, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(result, rows, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sum(array, negative_axis, 1, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_shape(result, kept_shape, 2);
  assert_int_values(result, rows, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sum(array, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_shape(result, NULL, 0);
  assert_int_values(result, total, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sum(array, duplicate_axes, 2, 0, &result) ==
         TERLAN_NDARRAY_DUPLICATE_AXIS);
  assert(terlan_ndarray_v1_sum(array, out_of_range, 1, 0, &result) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(terlan_ndarray_v1_sum(array, axis_zero, 1, 2, &result) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);

  assert(terlan_ndarray_v1_from_ints(NULL, 0, empty_shape, 3, &empty) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sum(empty, axis_one, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_shape(result, empty_sum_shape, 2);
  assert_int_values(result, empty_sum_values, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_floats(non_finite, 2, two_shape, 1, &floats) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sum(floats, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_to_floats(result, &float_result, &float_length) ==
         TERLAN_NDARRAY_OK);
  assert(float_length == 1 && isnan(float_result[0]));
  terlan_ndarray_v1_delete_float_values(float_result);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(floats) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_operation_failures_leave_sources_usable(void) {
  const int64_t values[] = {1, 2, 3, 4};
  const int64_t shape[] = {2, 2};
  const int64_t bad_shape[] = {3, 2};
  const int64_t negative_shape[] = {-1, 4};
  const int64_t axis[] = {0};
  const int64_t overflow_values[] = {INT64_MAX, 1};
  const int64_t overflow_shape[] = {2};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle rhs = NULL;
  TerlanNdArrayHandle overflow = NULL;
  TerlanNdArrayHandle result = NULL;
  int64_t fail_after;

  assert(terlan_ndarray_v1_from_ints(values, 4, shape, 2, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(values, 4, shape, 2, &rhs) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_reshape(source, bad_shape, 2, &result) ==
         TERLAN_NDARRAY_ELEMENT_COUNT_MISMATCH);
  assert(result == NULL);
  assert(terlan_ndarray_v1_reshape(source, negative_shape, 2, &result) ==
         TERLAN_NDARRAY_NEGATIVE_DIMENSION);
  assert(result == NULL);

  for (fail_after = 0; fail_after < 4; ++fail_after) {
    terlan_ndarray_test_fail_allocation_after(fail_after);
    assert(terlan_ndarray_v1_reshape(source, shape, 2, &result) ==
           TERLAN_NDARRAY_ALLOCATION_FAILED);
    assert(result == NULL && terlan_ndarray_v1_live_count() == 2);
    terlan_ndarray_test_reset_allocation_failures();
  }
  for (fail_after = 0; fail_after < 5; ++fail_after) {
    terlan_ndarray_test_fail_allocation_after(fail_after);
    assert(terlan_ndarray_v1_transpose(source, 0, 1, &result) ==
           TERLAN_NDARRAY_ALLOCATION_FAILED);
    assert(result == NULL && terlan_ndarray_v1_live_count() == 2);
    terlan_ndarray_test_reset_allocation_failures();
  }
  for (fail_after = 0; fail_after < 4; ++fail_after) {
    terlan_ndarray_test_fail_allocation_after(fail_after);
    assert(terlan_ndarray_v1_add(source, rhs, &result) ==
           TERLAN_NDARRAY_ALLOCATION_FAILED);
    assert(result == NULL && terlan_ndarray_v1_live_count() == 2);
    terlan_ndarray_test_reset_allocation_failures();
  }
  for (fail_after = 0; fail_after < 7; ++fail_after) {
    terlan_ndarray_test_fail_allocation_after(fail_after);
    assert(terlan_ndarray_v1_sum(source, axis, 1, 0, &result) ==
           TERLAN_NDARRAY_ALLOCATION_FAILED);
    assert(result == NULL && terlan_ndarray_v1_live_count() == 2);
    terlan_ndarray_test_reset_allocation_failures();
  }
  assert_int_values(source, values, 4);
  assert(terlan_ndarray_v1_from_ints(overflow_values, 2, overflow_shape, 1,
                                     &overflow) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sum(overflow, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_ARITHMETIC_OVERFLOW);
  assert(result == NULL && terlan_ndarray_v1_live_count() == 3);
  assert_int_values(overflow, overflow_values, 2);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(rhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(overflow) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}


int main(void) {
  test_reshape_and_transpose_are_shared_views();
  test_exact_shape_arithmetic_and_rejections();
  test_arbitrary_rank_transpose_and_float_arithmetic();
  test_numpy_binop_broadcast_views_and_chains_preserve_inputs();
  test_int64_arithmetic_wraps_like_numpy();
  test_sum_axes_empty_identity_and_non_finite_values();
  test_operation_failures_leave_sources_usable();
  return 0;
}
