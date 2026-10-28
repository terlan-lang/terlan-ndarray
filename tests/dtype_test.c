#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static void assert_dtype(TerlanNdArrayHandle array, int32_t expected) {
  int32_t actual = 0;
  assert(terlan_ndarray_v1_dtype(array, &actual) == TERLAN_NDARRAY_OK);
  assert(actual == expected);
}

static void test_every_real_dtype_constructs_and_casts(void) {
  static const int32_t integer_dtypes[] = {
      TERLAN_NDARRAY_DTYPE_BOOL,   TERLAN_NDARRAY_DTYPE_INT8,
      TERLAN_NDARRAY_DTYPE_INT16,  TERLAN_NDARRAY_DTYPE_INT32,
      TERLAN_NDARRAY_DTYPE_INT64,  TERLAN_NDARRAY_DTYPE_UINT8,
      TERLAN_NDARRAY_DTYPE_UINT16, TERLAN_NDARRAY_DTYPE_UINT32,
      TERLAN_NDARRAY_DTYPE_UINT64};
  static const int32_t float_dtypes[] = {TERLAN_NDARRAY_DTYPE_FLOAT16,
                                         TERLAN_NDARRAY_DTYPE_FLOAT32,
                                         TERLAN_NDARRAY_DTYPE_FLOAT64};
  const int64_t values[] = {0, 1};
  const int64_t shape[] = {2};
  size_t index;
  for (index = 0; index < sizeof(integer_dtypes) / sizeof(integer_dtypes[0]);
       ++index) {
    int64_t *readback = NULL;
    size_t length = 0;
    TerlanNdArrayHandle array = NULL;
    assert(terlan_ndarray_v1_from_ints_as(values, 2, shape, 1,
                                           integer_dtypes[index], &array) ==
           TERLAN_NDARRAY_OK);
    assert_dtype(array, integer_dtypes[index]);
    assert(terlan_ndarray_v1_to_ints_cast(array, &readback, &length) ==
           TERLAN_NDARRAY_OK);
    assert(length == 2 && readback[0] == 0 && readback[1] == 1);
    terlan_ndarray_v1_delete_int_values(readback);
    assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  }
  for (index = 0; index < sizeof(float_dtypes) / sizeof(float_dtypes[0]);
       ++index) {
    const int32_t dtype = float_dtypes[index];
    const double float_values[] = {-1.25, 2.5};
    double *readback = NULL;
    size_t length = 0;
    TerlanNdArrayHandle array = NULL;
    assert(terlan_ndarray_v1_from_floats_as(float_values, 2, shape, 1, dtype,
                                             &array) == TERLAN_NDARRAY_OK);
    assert_dtype(array, dtype);
    assert(terlan_ndarray_v1_to_floats_cast(array, &readback, &length) ==
           TERLAN_NDARRAY_OK);
    assert(length == 2 && readback[0] == -1.25 && readback[1] == 2.5);
    terlan_ndarray_v1_delete_float_values(readback);
    assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  }
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_complex_components_and_checked_casts(void) {
  const double real[] = {1.5, -2.0};
  const double imag[] = {-0.5, 4.0};
  const int64_t shape[] = {2};
  int32_t dtype;
  for (dtype = TERLAN_NDARRAY_DTYPE_COMPLEX64;
       dtype <= TERLAN_NDARRAY_DTYPE_COMPLEX128; ++dtype) {
    TerlanNdArrayHandle array = NULL;
    TerlanNdArrayHandle converted = NULL;
    TerlanNdArrayHandle rejected = NULL;
    double *parts = NULL;
    size_t length = 0;
    assert(terlan_ndarray_v1_from_complex(real, 2, imag, 2, shape, 1, dtype,
                                           &array) == TERLAN_NDARRAY_OK);
    assert_dtype(array, dtype);
    assert(terlan_ndarray_v1_complex_reals(array, &parts, &length) ==
           TERLAN_NDARRAY_OK);
    assert(length == 2 && parts[0] == 1.5 && parts[1] == -2.0);
    terlan_ndarray_v1_delete_float_values(parts);
    assert(terlan_ndarray_v1_complex_imags(array, &parts, &length) ==
           TERLAN_NDARRAY_OK);
    assert(length == 2 && parts[0] == -0.5 && parts[1] == 4.0);
    terlan_ndarray_v1_delete_float_values(parts);
    assert(terlan_ndarray_v1_astype(array, TERLAN_NDARRAY_DTYPE_FLOAT64,
                                    &converted) == TERLAN_NDARRAY_OK);
    assert_dtype(converted, TERLAN_NDARRAY_DTYPE_FLOAT64);
    assert(terlan_ndarray_v1_to_floats(converted, &parts, &length) ==
           TERLAN_NDARRAY_OK);
    assert(length == 2 && parts[0] == 1.5 && parts[1] == -2.0);
    terlan_ndarray_v1_delete_float_values(parts);
    assert(terlan_ndarray_v1_astype_ordered(
               array, TERLAN_NDARRAY_DTYPE_FLOAT64,
               TERLAN_NDARRAY_ORDER_KEEP, 1, TERLAN_NDARRAY_CAST_SAFE,
               &rejected) == TERLAN_NDARRAY_DTYPE_MISMATCH);
    assert(rejected == NULL);
    assert(terlan_ndarray_v1_delete(converted) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  }
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_promotion_table_matches_numpy_array_dtypes(void) {
  static const int32_t expected[14][14] = {
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
      {2, 2, 3, 2, 2, 2, 2, 2, 2, 3, 3, 13, 13, 3},
      {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 13, 13, 3},
      {4, 2, 3, 4, 5, 6, 5, 6, 2, 3, 11, 12, 13, 14},
      {5, 2, 3, 5, 5, 6, 5, 6, 2, 3, 11, 12, 13, 11},
      {6, 2, 3, 6, 6, 6, 6, 6, 2, 3, 3, 13, 13, 3},
      {7, 2, 3, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
      {8, 2, 3, 6, 6, 6, 8, 8, 9, 10, 11, 12, 13, 11},
      {9, 2, 3, 2, 2, 2, 9, 9, 9, 10, 3, 13, 13, 3},
      {10, 3, 3, 3, 3, 3, 10, 10, 10, 10, 3, 13, 13, 3},
      {11, 3, 3, 11, 11, 3, 11, 11, 3, 3, 11, 12, 13, 11},
      {12, 13, 13, 12, 12, 13, 12, 12, 13, 13, 12, 12, 13, 12},
      {13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13},
      {14, 3, 3, 14, 11, 3, 14, 11, 3, 3, 11, 12, 13, 14}};
  int32_t lhs;
  int32_t rhs;
  for (lhs = 1; lhs <= 14; ++lhs) {
    assert(dtype_size(lhs) > 0);
    for (rhs = 1; rhs <= 14; ++rhs) {
      assert(dtype_promote(lhs, rhs) == expected[lhs - 1][rhs - 1]);
      assert(dtype_promote(lhs, rhs) == dtype_promote(rhs, lhs));
    }
  }
  assert(dtype_promote(0, 1) == 0);
  assert(dtype_promote(1, 15) == 0);
}

static void test_invalid_and_overflowing_casts_fail_without_outputs(void) {
  const int64_t negative[] = {-1};
  const int64_t too_large[] = {256};
  const double fractional[] = {1.5};
  const int64_t shape[] = {1};
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_from_ints_as(negative, 1, shape, 1,
                                         TERLAN_NDARRAY_DTYPE_UINT8,
                                         &output) ==
         TERLAN_NDARRAY_CAST_OVERFLOW);
  assert(output == NULL);
  assert(terlan_ndarray_v1_from_ints_as(too_large, 1, shape, 1,
                                         TERLAN_NDARRAY_DTYPE_UINT8,
                                         &output) ==
         TERLAN_NDARRAY_CAST_OVERFLOW);
  assert(terlan_ndarray_v1_from_floats_as(fractional, 1, shape, 1,
                                           TERLAN_NDARRAY_DTYPE_INT64,
                                           &output) ==
         TERLAN_NDARRAY_CAST_OVERFLOW);
  assert(terlan_ndarray_v1_from_ints_as(negative, 1, shape, 1, 999, &output) ==
         TERLAN_NDARRAY_UNSUPPORTED_DTYPE);
  assert(terlan_ndarray_v1_live_count() == 0);
}

int main(void) {
  test_every_real_dtype_constructs_and_casts();
  test_complex_components_and_checked_casts();
  test_promotion_table_matches_numpy_array_dtypes();
  test_invalid_and_overflowing_casts_fail_without_outputs();
  return 0;
}
