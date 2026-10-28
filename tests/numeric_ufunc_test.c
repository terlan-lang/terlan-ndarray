#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static TerlanNdArrayHandle ints(const int64_t *values, int64_t count) {
  TerlanNdArrayHandle array = NULL;
  assert(terlan_ndarray_v1_from_ints(values, count, &count, 1, &array) == 0);
  return array;
}

static TerlanNdArrayHandle floats(const double *values, int64_t count) {
  TerlanNdArrayHandle array = NULL;
  assert(terlan_ndarray_v1_from_floats(values, count, &count, 1, &array) == 0);
  return array;
}

static void assert_ints(TerlanNdArrayHandle array, const int64_t *expected,
                        size_t count) {
  int64_t *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_ints_cast(array, &values, &length) == 0);
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
  assert(terlan_ndarray_v1_to_floats_cast(array, &values, &length) == 0);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(fabs(values[index] - expected[index]) < 1e-10);
  }
  terlan_ndarray_v1_delete_float_values(values);
}

static void test_bitwise_shift_gcd_and_lcm(void) {
  TerlanNdArrayHandle lhs = ints((int64_t[]){12, 7, -8}, 3);
  TerlanNdArrayHandle rhs = ints((int64_t[]){10, 3, 1}, 3);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_bitwise_and(lhs, rhs, &output) == 0);
  assert_ints(output, (int64_t[]){8, 3, 0}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_bitwise_or(lhs, rhs, &output) == 0);
  assert_ints(output, (int64_t[]){14, 7, -7}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_bitwise_xor(lhs, rhs, &output) == 0);
  assert_ints(output, (int64_t[]){6, 4, -7}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_left_shift(lhs, rhs, &output) == 0);
  assert_ints(output, (int64_t[]){12288, 56, -16}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_right_shift(lhs, rhs, &output) == 0);
  assert_ints(output, (int64_t[]){0, 0, -4}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_gcd(lhs, rhs, &output) == 0);
  assert_ints(output, (int64_t[]){2, 1, 1}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_lcm(lhs, rhs, &output) == 0);
  assert_ints(output, (int64_t[]){60, 21, 8}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(rhs) == 0);
  assert(terlan_ndarray_v1_delete(lhs) == 0);
}

static void test_transcendental_angle_and_tolerance_families(void) {
  TerlanNdArrayHandle lhs = floats((double[]){3.0, 0.0, 1000.0}, 3);
  TerlanNdArrayHandle rhs = floats((double[]){4.0, -1.0, 999.999999}, 3);
  TerlanNdArrayHandle angles =
      floats((double[]){0.0, acos(-1.0)}, 2);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_hypot(lhs, rhs, &output) == 0);
  assert_floats(output, (double[]){5.0, 1.0, hypot(1000.0, 999.999999)}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_arctan2(lhs, rhs, &output) == 0);
  assert_floats(output, (double[]){atan2(3.0, 4.0), acos(-1.0),
                                   atan2(1000.0, 999.999999)},
                3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_is_close(lhs, rhs, 1e-5, 1e-8, 0, &output) == 0);
  assert_ints(output, (int64_t[]){0, 0, 1}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_degrees(angles, &output) == 0);
  assert_floats(output, (double[]){0.0, 180.0}, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(angles) == 0);
  assert(terlan_ndarray_v1_exp2(lhs, &output) == 0);
  assert_floats(output, (double[]){8.0, 1.0, exp2(1000.0)}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(rhs) == 0);
  assert(terlan_ndarray_v1_delete(lhs) == 0);
}

static void test_floating_decomposition_and_sign_bits(void) {
  TerlanNdArrayHandle array = floats((double[]){-0.0, 6.5, -2.25}, 3);
  TerlanNdArrayHandle output = NULL;
  uint64_t *outputs = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_sign_bit(array, &output) == 0);
  assert_ints(output, (int64_t[]){1, 0, 1}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_frexp_all(array, &outputs, &length) == 0);
  assert(length == 2);
  assert_floats((TerlanNdArrayHandle)(uintptr_t)outputs[0],
                (double[]){-0.0, 0.8125, -0.5625}, 3);
  assert_ints((TerlanNdArrayHandle)(uintptr_t)outputs[1],
              (int64_t[]){0, 3, 2}, 3);
  for (index = 0; index < length; ++index) {
    assert(terlan_ndarray_v1_delete(
               (TerlanNdArrayHandle)(uintptr_t)outputs[index]) == 0);
  }
  terlan_ndarray_v1_delete_handle_values(outputs);
  assert(terlan_ndarray_v1_modf_all(array, &outputs, &length) == 0);
  assert_floats((TerlanNdArrayHandle)(uintptr_t)outputs[0],
                (double[]){-0.0, 0.5, -0.25}, 3);
  assert_floats((TerlanNdArrayHandle)(uintptr_t)outputs[1],
                (double[]){-0.0, 6.0, -2.0}, 3);
  for (index = 0; index < length; ++index) {
    assert(terlan_ndarray_v1_delete(
               (TerlanNdArrayHandle)(uintptr_t)outputs[index]) == 0);
  }
  terlan_ndarray_v1_delete_handle_values(outputs);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_numeric_ufunc_rejections(void) {
  TerlanNdArrayHandle integers = ints((int64_t[]){1}, 1);
  TerlanNdArrayHandle wide_shift = ints((int64_t[]){64}, 1);
  TerlanNdArrayHandle negative_shift = ints((int64_t[]){-1}, 1);
  TerlanNdArrayHandle floating = floats((double[]){1.0}, 1);
  TerlanNdArrayHandle output = NULL;
  TerlanNdArrayHandle large = ints((int64_t[]){INT64_MAX}, 1);
  TerlanNdArrayHandle two = ints((int64_t[]){2}, 1);
  uint64_t *outputs = NULL;
  size_t length = 0;
  int64_t failure;
  assert(terlan_ndarray_v1_left_shift(integers, wide_shift, &output) ==
         TERLAN_NDARRAY_OK);
  assert_ints(output, (int64_t[]){0}, 1);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  output = NULL;
  assert(terlan_ndarray_v1_left_shift(integers, negative_shift, &output) ==
         TERLAN_NDARRAY_DOMAIN_ERROR);
  assert(output == NULL);
  assert(terlan_ndarray_v1_bitwise_and(floating, floating, &output) ==
         TERLAN_NDARRAY_UNSUPPORTED_DTYPE);
  assert(terlan_ndarray_v1_is_close(floating, floating, -1.0, 0.0, 0,
                                    &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_lcm(large, two, &output) ==
         TERLAN_NDARRAY_ARITHMETIC_OVERFLOW);
  assert(output == NULL);
  for (failure = 0; failure < 4; ++failure) {
    terlan_ndarray_test_fail_allocation_after(failure);
    outputs = NULL;
    length = 0;
    if (terlan_ndarray_v1_frexp_all(floating, &outputs, &length) == 0) {
      size_t index;
      for (index = 0; index < length; ++index) {
        assert(terlan_ndarray_v1_delete(
                   (TerlanNdArrayHandle)(uintptr_t)outputs[index]) == 0);
      }
      terlan_ndarray_v1_delete_handle_values(outputs);
    } else {
      assert(outputs == NULL && length == 0);
    }
    terlan_ndarray_test_reset_allocation_failures();
  }
  assert(terlan_ndarray_v1_delete(two) == 0);
  assert(terlan_ndarray_v1_delete(large) == 0);
  assert(terlan_ndarray_v1_delete(floating) == 0);
  assert(terlan_ndarray_v1_delete(negative_shift) == 0);
  assert(terlan_ndarray_v1_delete(wide_shift) == 0);
  assert(terlan_ndarray_v1_delete(integers) == 0);
}

int main(void) {
  test_bitwise_shift_gcd_and_lcm();
  test_transcendental_angle_and_tolerance_families();
  test_floating_decomposition_and_sign_bits();
  test_numeric_ufunc_rejections();
  return 0;
}
