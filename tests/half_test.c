#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef TerlanNdArrayStatus (*BinaryFunction)(TerlanNdArrayHandle,
                                              TerlanNdArrayHandle,
                                              TerlanNdArrayHandle *);

static TerlanNdArrayHandle half_bits(const uint16_t *bits, size_t count) {
  const int64_t shape[] = {(int64_t)count};
  TerlanNdArrayHandle output = NULL;
  assert(create_array(bits, (int64_t)count, shape, 1, sizeof(uint16_t),
                      TERLAN_NDARRAY_DTYPE_FLOAT16, &output) ==
         TERLAN_NDARRAY_OK);
  return output;
}

static TerlanNdArrayHandle half_values(const double *values, size_t count) {
  const int64_t shape[] = {(int64_t)count};
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_from_floats_as(
             values, (int64_t)count, shape, 1, TERLAN_NDARRAY_DTYPE_FLOAT16,
             &output) == TERLAN_NDARRAY_OK);
  return output;
}

static void assert_dtype(TerlanNdArrayHandle array, int32_t expected) {
  int32_t actual = 0;
  assert(terlan_ndarray_v1_dtype(array, &actual) == TERLAN_NDARRAY_OK);
  assert(actual == expected);
}

static void assert_half(TerlanNdArrayHandle array, const uint16_t *expected,
                        size_t count) {
  size_t index;
  assert_dtype(array, TERLAN_NDARRAY_DTYPE_FLOAT16);
  assert(array->numel == (int64_t)count);
  for (index = 0; index < count; ++index) {
    uint16_t actual = 0;
    memcpy(&actual, array_const_element_pointer(array, (int64_t)index),
           sizeof(actual));
    if (actual != expected[index]) {
      fprintf(stderr, "half[%zu]: actual=0x%04x expected=0x%04x\n", index,
              actual, expected[index]);
      assert(0);
    }
  }
}

static void assert_ints(TerlanNdArrayHandle array, const int64_t *expected,
                        size_t count) {
  int64_t *actual = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_ints_cast(array, &actual, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(actual[index] == expected[index]);
  }
  terlan_ndarray_v1_delete_int_values(actual);
}

static double manual_half_value(uint16_t bits) {
  const int sign = (bits & UINT16_C(0x8000)) == 0 ? 1 : -1;
  const int exponent = (bits >> 10) & 0x1f;
  const int fraction = bits & 0x03ff;
  if (exponent == 0) {
    return sign * ldexp((double)fraction, -24);
  }
  return sign * ldexp((double)(1024 + fraction), exponent - 25);
}

static void test_every_encoding_converts_exactly(void) {
  uint32_t raw;
  for (raw = 0; raw <= UINT16_MAX; ++raw) {
    const uint16_t bits = (uint16_t)raw;
    const uint16_t exponent = (uint16_t)((bits >> 10) & 0x1fU);
    const uint16_t fraction = (uint16_t)(bits & 0x03ffU);
    const float single = dtype_half_to_float(bits);
    const double doubled = dtype_half_to_double(bits);
    if (exponent == 0x1fU && fraction != 0) {
      assert(isnan(single));
      assert(isnan(doubled));
    } else if (exponent == 0x1fU) {
      assert(isinf(single) && isinf(doubled));
      assert((signbit(single) != 0) == (signbit(doubled) != 0));
    } else {
      const double expected = manual_half_value(bits);
      assert((double)single == expected);
      assert(doubled == expected);
      if (expected == 0.0) {
        assert((signbit(single) != 0) ==
               ((bits & UINT16_C(0x8000)) != 0));
        assert((signbit(doubled) != 0) ==
               ((bits & UINT16_C(0x8000)) != 0));
      }
    }
    assert(dtype_float_to_half(single) == bits);
    assert(dtype_double_to_half(doubled) == bits);
  }
}

static void test_known_values_and_round_to_even(void) {
  static const double values[] = {
      1.0,          -1.0,       2.0,          -2.0,
      0.0999755859375, 0.333251953125, 65504.0, -65504.0,
      0x1p-14,      -0x1p-14,   0x1p-24,      -0x1p-24,
      0.0,          -0.0,       INFINITY,      -INFINITY};
  static const uint16_t expected[] = {
      0x3c00, 0xbc00, 0x4000, 0xc000, 0x2e66, 0x3555, 0x7bff, 0xfbff,
      0x0400, 0x8400, 0x0001, 0x8001, 0x0000, 0x8000, 0x7c00, 0xfc00};
  static const double boundaries[] = {
      0x1p-25 + 0x1p-35, 0x1p-25, 0x1p-26,
      1.0 + 0x1p-11 + 0x1p-16, 1.0 + 0x1p-11, 1.0 + 0x1p-12,
      65519.0, 65520.0};
  static const uint16_t rounded[] = {0x0001, 0x0000, 0x0000, 0x3c01,
                                      0x3c00, 0x3c00, 0x7bff, 0x7c00};
  size_t index;
  for (index = 0; index < sizeof(values) / sizeof(values[0]); ++index) {
    assert(dtype_double_to_half(values[index]) == expected[index]);
  }
  for (index = 0; index < sizeof(boundaries) / sizeof(boundaries[0]);
       ++index) {
    assert(dtype_double_to_half(boundaries[index]) == rounded[index]);
    assert(dtype_float_to_half((float)boundaries[index]) == rounded[index]);
  }
}

static void test_denormal_rounding_uses_every_source_bit(void) {
  double double_center = 0x1p-25;
  float single_center = 0x1p-25f;
  uint64_t double_bits;
  uint32_t single_bits;
  uint32_t bit;
  assert(dtype_double_to_half(double_center) == 0);
  assert(dtype_float_to_half(single_center) == 0);
  memcpy(&double_bits, &double_center, sizeof(double_bits));
  memcpy(&single_bits, &single_center, sizeof(single_bits));
  for (bit = 0; bit < 52; ++bit) {
    const uint64_t larger_bits = double_bits | (UINT64_C(1) << bit);
    double larger;
    memcpy(&larger, &larger_bits, sizeof(larger));
    assert(dtype_double_to_half(larger) == 1);
  }
  for (bit = 0; bit < 23; ++bit) {
    const uint32_t larger_bits = single_bits | (UINT32_C(1) << bit);
    float larger;
    memcpy(&larger, &larger_bits, sizeof(larger));
    assert(dtype_float_to_half(larger) == 1);
  }
}

static void test_array_casts_do_not_double_round(void) {
  static const double values[] = {0x1p-25 + 0x1p-77, 0x1.0020000000001p0,
                                  65519.0, 65520.0, -0.0, NAN};
  static const uint16_t expected[] = {0x0001, 0x3c01, 0x7bff,
                                      0x7c00, 0x8000, 0x7e00};
  const int64_t shape[] = {6};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle converted = NULL;
  assert(terlan_ndarray_v1_from_floats_as(
             values, 6, shape, 1, TERLAN_NDARRAY_DTYPE_FLOAT64, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_astype(source, TERLAN_NDARRAY_DTYPE_FLOAT16,
                                  &converted) == TERLAN_NDARRAY_OK);
  assert_half(converted, expected, 6);
  assert(terlan_ndarray_v1_delete(converted) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_nextafter_walks_binary16_encodings(void) {
  const size_t count = UINT16_C(0x7c00);
  uint16_t *bits = malloc(count * sizeof(*bits));
  uint16_t *expected = malloc(count * sizeof(*expected));
  const uint16_t positive_infinity[] = {UINT16_C(0x7c00)};
  const uint16_t negative_infinity[] = {UINT16_C(0xfc00)};
  TerlanNdArrayHandle values;
  TerlanNdArrayHandle direction;
  TerlanNdArrayHandle output = NULL;
  size_t index;
  assert(bits != NULL && expected != NULL);
  for (index = 0; index < count; ++index) {
    bits[index] = (uint16_t)index;
    expected[index] = (uint16_t)(index + 1);
  }
  values = half_bits(bits, count);
  direction = half_bits(positive_infinity, 1);
  assert(terlan_ndarray_v1_nextafter(values, direction, &output) ==
         TERLAN_NDARRAY_OK);
  assert_half(output, expected, count);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(direction) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(values) == TERLAN_NDARRAY_OK);

  values = half_bits((uint16_t[]){0x0000, 0x0001, 0x7c00, 0x8000,
                                  0x8001, 0xfc00}, 6);
  direction = half_bits(negative_infinity, 1);
  assert(terlan_ndarray_v1_nextafter(values, direction, &output) ==
         TERLAN_NDARRAY_OK);
  assert_half(output,
              (uint16_t[]){0x8001, 0x0000, 0x7bff, 0x8001, 0x8002, 0xfc00},
              6);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(direction) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(values) == TERLAN_NDARRAY_OK);
  free(expected);
  free(bits);
}

static void assert_binary(BinaryFunction function, const double *lhs_values,
                          const double *rhs_values, const uint16_t *expected,
                          size_t count) {
  TerlanNdArrayHandle lhs = half_values(lhs_values, count);
  TerlanNdArrayHandle rhs = half_values(rhs_values, count);
  TerlanNdArrayHandle output = NULL;
  assert(function(lhs, rhs, &output) == TERLAN_NDARRAY_OK);
  assert_half(output, expected, count);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(rhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(lhs) == TERLAN_NDARRAY_OK);
}

static void assert_binary_bools(BinaryFunction function,
                                const double *lhs_values,
                                const double *rhs_values,
                                const int64_t *expected, size_t count) {
  TerlanNdArrayHandle lhs = half_values(lhs_values, count);
  TerlanNdArrayHandle rhs = half_values(rhs_values, count);
  TerlanNdArrayHandle output = NULL;
  assert(function(lhs, rhs, &output) == TERLAN_NDARRAY_OK);
  assert_ints(output, expected, count);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(rhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(lhs) == TERLAN_NDARRAY_OK);
}

static void test_ufuncs_classification_and_coercion(void) {
  static const double lhs[] = {0, 1, 2, 4, 2};
  static const double rhs[] = {-2, 5, 1, 4, 3};
  static const double special[] = {0, -1, -INFINITY, NAN, 6};
  TerlanNdArrayHandle classified = half_values(
      (double[]){0.0, -1.0, -INFINITY, NAN, 6.0}, 5);
  TerlanNdArrayHandle output = NULL;
  assert_binary(terlan_ndarray_v1_add, lhs, rhs,
                (uint16_t[]){0xc000, 0x4600, 0x4200, 0x4800, 0x4500}, 5);
  assert_binary(terlan_ndarray_v1_subtract, lhs, rhs,
                (uint16_t[]){0x4000, 0xc400, 0x3c00, 0x0000, 0xbc00}, 5);
  assert_binary(terlan_ndarray_v1_multiply, lhs, rhs,
                (uint16_t[]){0x8000, 0x4500, 0x4000, 0x4c00, 0x4600}, 5);
  assert_binary(terlan_ndarray_v1_true_divide, lhs, rhs,
                (uint16_t[]){0x8000, 0x3266, 0x4000, 0x3c00, 0x3955}, 5);
  assert_binary_bools(terlan_ndarray_v1_equal, lhs, rhs,
                      (int64_t[]){0, 0, 0, 1, 0}, 5);
  assert_binary_bools(terlan_ndarray_v1_not_equal, lhs, rhs,
                      (int64_t[]){1, 1, 1, 0, 1}, 5);
  assert_binary_bools(terlan_ndarray_v1_less, lhs, rhs,
                      (int64_t[]){0, 1, 0, 0, 1}, 5);
  assert_binary_bools(terlan_ndarray_v1_less_equal, lhs, rhs,
                      (int64_t[]){0, 1, 0, 1, 1}, 5);
  assert_binary_bools(terlan_ndarray_v1_greater, lhs, rhs,
                      (int64_t[]){1, 0, 1, 0, 0}, 5);
  assert_binary_bools(terlan_ndarray_v1_greater_equal, lhs, rhs,
                      (int64_t[]){1, 0, 1, 1, 0}, 5);
  assert_binary_bools(terlan_ndarray_v1_logical_and, lhs, rhs,
                      (int64_t[]){0, 1, 1, 1, 1}, 5);
  assert_binary_bools(terlan_ndarray_v1_logical_or, lhs, rhs,
                      (int64_t[]){1, 1, 1, 1, 1}, 5);
  assert_binary_bools(terlan_ndarray_v1_logical_xor, lhs, rhs,
                      (int64_t[]){1, 0, 0, 0, 0}, 5);
  assert_binary(terlan_ndarray_v1_copysign, rhs, lhs,
                (uint16_t[]){0x4000, 0x4500, 0x3c00, 0x4400, 0x4200}, 5);
  assert_binary(terlan_ndarray_v1_elementwise_maximum, rhs, special,
                (uint16_t[]){0x0000, 0x4500, 0x3c00, 0x7e00, 0x4600}, 5);
  assert_binary(terlan_ndarray_v1_elementwise_minimum, rhs, special,
                (uint16_t[]){0xc000, 0xbc00, 0xfc00, 0x7e00, 0x4200}, 5);
  assert_binary(terlan_ndarray_v1_fmax, rhs, special,
                (uint16_t[]){0x0000, 0x4500, 0x3c00, 0x4400, 0x4600}, 5);
  assert_binary(terlan_ndarray_v1_fmin, rhs, special,
                (uint16_t[]){0xc000, 0xbc00, 0xfc00, 0x4400, 0x4200}, 5);
  assert_binary(terlan_ndarray_v1_floor_divide, lhs, rhs,
                (uint16_t[]){0x8000, 0x0000, 0x4000, 0x3c00, 0x0000}, 5);
  assert_binary(terlan_ndarray_v1_remainder, lhs, rhs,
                (uint16_t[]){0x8000, 0x3c00, 0x0000, 0x0000, 0x4000}, 5);
  assert(terlan_ndarray_v1_is_nan(classified, &output) == TERLAN_NDARRAY_OK);
  assert_ints(output, (int64_t[]){0, 0, 0, 1, 0}, 5);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_is_infinite(classified, &output) ==
         TERLAN_NDARRAY_OK);
  assert_ints(output, (int64_t[]){0, 0, 1, 0, 0}, 5);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_is_finite(classified, &output) ==
         TERLAN_NDARRAY_OK);
  assert_ints(output, (int64_t[]){1, 1, 0, 0, 1}, 5);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sign_bit(classified, &output) == TERLAN_NDARRAY_OK);
  assert_ints(output, (int64_t[]){0, 1, 1, 0, 0}, 5);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(classified) == TERLAN_NDARRAY_OK);

  {
    TerlanNdArrayHandle unary = half_values(rhs, 5);
    assert(terlan_ndarray_v1_square(unary, &output) == TERLAN_NDARRAY_OK);
    assert_half(output, (uint16_t[]){0x4400, 0x4e40, 0x3c00, 0x4c00, 0x4880},
                5);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_reciprocal(unary, &output) == TERLAN_NDARRAY_OK);
    assert_half(output, (uint16_t[]){0xb800, 0x3266, 0x3c00, 0x3400, 0x3555},
                5);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_absolute(unary, &output) == TERLAN_NDARRAY_OK);
    assert_half(output, (uint16_t[]){0x4000, 0x4500, 0x3c00, 0x4400, 0x4200},
                5);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_negative(unary, &output) == TERLAN_NDARRAY_OK);
    assert_half(output, (uint16_t[]){0x4000, 0xc500, 0xbc00, 0xc400, 0xc200},
                5);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_sign(unary, &output) == TERLAN_NDARRAY_OK);
    assert_half(output, (uint16_t[]){0xbc00, 0x3c00, 0x3c00, 0x3c00, 0x3c00},
                5);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_logical_not(unary, &output) == TERLAN_NDARRAY_OK);
    assert_ints(output, (int64_t[]){0, 0, 0, 0, 0}, 5);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_conjugate(unary, &output) == TERLAN_NDARRAY_OK);
    assert_half(output, (uint16_t[]){0xc000, 0x4500, 0x3c00, 0x4400, 0x4200},
                5);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_ones_like(unary, &output) == TERLAN_NDARRAY_OK);
    assert_half(output, (uint16_t[]){0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00},
                5);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(unary) == TERLAN_NDARRAY_OK);
  }

  {
    TerlanNdArrayHandle half = half_values((double[]){1.0}, 1);
    const int64_t shape[] = {1};
    TerlanNdArrayHandle single = NULL;
    assert(terlan_ndarray_v1_from_floats_as(
               (double[]){1.0}, 1, shape, 1, TERLAN_NDARRAY_DTYPE_FLOAT32,
               &single) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_power(half, half, &output) == TERLAN_NDARRAY_OK);
    assert_dtype(output, TERLAN_NDARRAY_DTYPE_FLOAT16);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_power(half, single, &output) == TERLAN_NDARRAY_OK);
    assert_dtype(output, TERLAN_NDARRAY_DTYPE_FLOAT32);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(single) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(half) == TERLAN_NDARRAY_OK);
  }
}

static void test_all_non_nan_encodings_sort_numerically(void) {
  const size_t side = UINT16_C(0x7c01);
  const size_t count = side * 2;
  uint16_t *bits = malloc(count * sizeof(*bits));
  TerlanNdArrayHandle values;
  TerlanNdArrayHandle sorted = NULL;
  double *actual = NULL;
  size_t length = 0;
  size_t index;
  size_t strict = 0;
  assert(bits != NULL);
  for (index = 0; index < side; ++index) {
    bits[index] = (uint16_t)index;
    bits[side + index] = (uint16_t)(UINT16_C(0x8000) + index);
  }
  values = half_bits(bits, count);
  assert(terlan_ndarray_v1_sort(values, 0, &sorted) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_to_floats_cast(sorted, &actual, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  assert(isinf(actual[0]) && signbit(actual[0]));
  assert(isinf(actual[count - 1]) && !signbit(actual[count - 1]));
  for (index = 1; index < count; ++index) {
    assert(actual[index - 1] <= actual[index]);
    strict += actual[index - 1] < actual[index];
  }
  assert(strict == count - 2);
  terlan_ndarray_v1_delete_float_values(actual);
  assert(terlan_ndarray_v1_delete(sorted) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(values) == TERLAN_NDARRAY_OK);
  free(bits);
}

static void test_array_functions_preserve_half_semantics(void) {
  TerlanNdArrayHandle values = half_values(
      (double[]){0.0, 0.0, -1.0, -1e-20, 0.0, 0x1p-24, 7.629e-6}, 7);
  TerlanNdArrayHandle scalar = half_values((double[]){1.0}, 1);
  TerlanNdArrayHandle output = NULL;
  uint64_t *coordinates = NULL;
  size_t coordinate_count = 0;
  assert(terlan_ndarray_v1_nonzero(values, &coordinates, &coordinate_count) ==
         TERLAN_NDARRAY_OK);
  assert(coordinate_count == 1);
  assert_ints((TerlanNdArrayHandle)(uintptr_t)coordinates[0],
              (int64_t[]){2, 5, 6}, 3);
  assert(terlan_ndarray_v1_delete(
             (TerlanNdArrayHandle)(uintptr_t)coordinates[0]) ==
         TERLAN_NDARRAY_OK);
  terlan_ndarray_v1_delete_handle_values(coordinates);
  assert(terlan_ndarray_v1_fill(values, scalar) == TERLAN_NDARRAY_OK);
  assert_half(values,
              (uint16_t[]){0x3c00, 0x3c00, 0x3c00, 0x3c00,
                           0x3c00, 0x3c00, 0x3c00}, 7);
  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(values) == TERLAN_NDARRAY_OK);

  values = half_values((double[]){0, -INFINITY, -2, 0.5, 12.55, 7.3, 2.1,
                                  12.4}, 8);
  assert(terlan_ndarray_v1_argmax(values, NULL, 0, 0, &output) ==
         TERLAN_NDARRAY_OK);
  assert_ints(output, (int64_t[]){4}, 1);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(values) == TERLAN_NDARRAY_OK);

  values = half_values((double[]){0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5,
                                  5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5},
                       20);
  scalar = half_values((double[]){1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 20);
  assert(terlan_ndarray_v1_dot(values, scalar, &output) == TERLAN_NDARRAY_OK);
  assert_half(output, (uint16_t[]){0x55f0}, 1);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(values) == TERLAN_NDARRAY_OK);
}

int main(void) {
  test_every_encoding_converts_exactly();
  test_known_values_and_round_to_even();
  test_denormal_rounding_uses_every_source_bit();
  test_array_casts_do_not_double_round();
  test_nextafter_walks_binary16_encodings();
  test_ufuncs_classification_and_coercion();
  test_all_non_nan_encodings_sort_numerically();
  test_array_functions_preserve_half_semantics();
  assert(terlan_ndarray_v1_live_count() == 0);
  return 0;
}
