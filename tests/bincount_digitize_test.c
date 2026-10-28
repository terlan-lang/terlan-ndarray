#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>

static TerlanNdArrayHandle integers(const int64_t *values, int64_t count,
                                    const int64_t *shape, int64_t rank) {
  TerlanNdArrayHandle array = NULL;
  assert(terlan_ndarray_v1_from_ints(values, count, shape, rank, &array) == 0);
  return array;
}

static TerlanNdArrayHandle floats_as(const double *values, int64_t count,
                                     const int64_t *shape, int64_t rank,
                                     int32_t dtype) {
  TerlanNdArrayHandle array = NULL;
  assert(terlan_ndarray_v1_from_floats_as(values, count, shape, rank, dtype,
                                          &array) == 0);
  return array;
}

static TerlanNdArrayHandle floats(const double *values, int64_t count,
                                  const int64_t *shape, int64_t rank) {
  return floats_as(values, count, shape, rank,
                   TERLAN_NDARRAY_DTYPE_FLOAT64);
}

static void assert_dtype(TerlanNdArrayHandle array, int32_t expected) {
  int32_t actual = 0;
  assert(terlan_ndarray_v1_dtype(array, &actual) == 0);
  assert(actual == expected);
}

static void assert_ints(TerlanNdArrayHandle array, const int64_t *expected,
                        size_t count) {
  int64_t *actual = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_ints_cast(array, &actual, &length) == 0);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(actual[index] == expected[index]);
  }
  terlan_ndarray_v1_delete_int_values(actual);
}

static void assert_floats(TerlanNdArrayHandle array, const double *expected,
                          size_t count) {
  double *actual = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_floats_cast(array, &actual, &length) == 0);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    if (isnan(expected[index])) {
      assert(isnan(actual[index]));
    } else if (isinf(expected[index])) {
      assert(isinf(actual[index]) &&
             signbit(actual[index]) == signbit(expected[index]));
    } else {
      assert(fabs(actual[index] - expected[index]) <= 1e-6);
    }
  }
  terlan_ndarray_v1_delete_float_values(actual);
}

static void test_unweighted_bincount_values_empty_and_minimum_length(void) {
  TerlanNdArrayHandle simple =
      integers((int64_t[]){1, 5, 2, 4, 1}, 5, (int64_t[]){5}, 1);
  TerlanNdArrayHandle repeated = integers(
      (int64_t[]){0, 1, 1, 2, 2, 3, 3}, 7, (int64_t[]){7}, 1);
  TerlanNdArrayHandle empty = integers(NULL, 0, (int64_t[]){0}, 1);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_bincount_unweighted(simple, 0, &output) == 0);
  assert_dtype(output, TERLAN_NDARRAY_DTYPE_INT64);
  assert_ints(output, (int64_t[]){0, 2, 1, 0, 1, 1}, 6);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_bincount_unweighted(repeated, 2, &output) == 0);
  assert_ints(output, (int64_t[]){1, 2, 2, 2}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_bincount_unweighted(empty, 0, &output) == 0);
  assert_ints(output, NULL, 0);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_bincount_unweighted(empty, 5, &output) == 0);
  assert_ints(output, (int64_t[]){0, 0, 0, 0, 0}, 5);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(empty) == 0);
  assert(terlan_ndarray_v1_delete(repeated) == 0);
  assert(terlan_ndarray_v1_delete(simple) == 0);
}

static void test_weighted_bincount_is_float64_and_keeps_nonfinite_values(void) {
  TerlanNdArrayHandle values =
      integers((int64_t[]){1, 2, 4, 5, 2}, 5, (int64_t[]){5}, 1);
  TerlanNdArrayHandle weights = floats_as(
      (double[]){0.2, 0.3, 0.5, 0.1, 0.2}, 5, (int64_t[]){5}, 1,
      TERLAN_NDARRAY_DTYPE_FLOAT32);
  TerlanNdArrayHandle special_values =
      integers((int64_t[]){0, 1, 1}, 3, (int64_t[]){3}, 1);
  TerlanNdArrayHandle special_weights = floats(
      (double[]){INFINITY, 1, NAN}, 3, (int64_t[]){3}, 1);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_bincount(values, weights, 8, &output) == 0);
  assert_dtype(output, TERLAN_NDARRAY_DTYPE_FLOAT64);
  assert_floats(output, (double[]){0, 0.2, 0.5, 0, 0.5, 0.1, 0, 0}, 8);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_bincount(special_values, special_weights, 0,
                                    &output) == 0);
  assert_floats(output, (double[]){INFINITY, NAN}, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(special_weights) == 0);
  assert(terlan_ndarray_v1_delete(special_values) == 0);
  assert(terlan_ndarray_v1_delete(weights) == 0);
  assert(terlan_ndarray_v1_delete(values) == 0);
}

static void test_digitize_increasing_decreasing_repeated_and_right_edges(void) {
  TerlanNdArrayHandle values = integers(
      (int64_t[]){-1, 0, 1, 2}, 4, (int64_t[]){2, 2}, 2);
  TerlanNdArrayHandle increasing =
      integers((int64_t[]){0, 0, 1}, 3, (int64_t[]){3}, 1);
  TerlanNdArrayHandle decreasing =
      integers((int64_t[]){1, 1, 0}, 3, (int64_t[]){3}, 1);
  TerlanNdArrayHandle equal =
      integers((int64_t[]){1, 1, 1, 1}, 4, (int64_t[]){4}, 1);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_digitize(values, increasing, 0, &output) == 0);
  assert_ints(output, (int64_t[]){0, 2, 3, 3}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_digitize(values, increasing, 1, &output) == 0);
  assert_ints(output, (int64_t[]){0, 0, 2, 3}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_digitize(values, decreasing, 0, &output) == 0);
  assert_ints(output, (int64_t[]){3, 2, 0, 0}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_digitize(values, decreasing, 1, &output) == 0);
  assert_ints(output, (int64_t[]){3, 3, 2, 0}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_digitize(values, equal, 0, &output) == 0);
  assert_ints(output, (int64_t[]){0, 0, 4, 4}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_digitize(values, equal, 1, &output) == 0);
  assert_ints(output, (int64_t[]){0, 0, 0, 4}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(equal) == 0);
  assert(terlan_ndarray_v1_delete(decreasing) == 0);
  assert(terlan_ndarray_v1_delete(increasing) == 0);
  assert(terlan_ndarray_v1_delete(values) == 0);
}

static void test_digitize_large_integers_and_nonfinite_values(void) {
  const int64_t large = INT64_C(1) << 54;
  TerlanNdArrayHandle value = integers(&large, 1, NULL, 0);
  TerlanNdArrayHandle increasing = integers(
      (int64_t[]){large - 1, large + 1}, 2, (int64_t[]){2}, 1);
  TerlanNdArrayHandle decreasing = integers(
      (int64_t[]){large + 1, large - 1}, 2, (int64_t[]){2}, 1);
  TerlanNdArrayHandle special = floats(
      (double[]){NAN, -INFINITY, INFINITY}, 3, (int64_t[]){3}, 1);
  TerlanNdArrayHandle real_bins =
      floats((double[]){0, 1, 2}, 3, (int64_t[]){3}, 1);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_digitize(value, increasing, 0, &output) == 0);
  assert_ints(output, (int64_t[]){1}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_digitize(value, decreasing, 0, &output) == 0);
  assert_ints(output, (int64_t[]){1}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_digitize(special, real_bins, 0, &output) == 0);
  assert_ints(output, (int64_t[]){3, 0, 3}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(real_bins) == 0);
  assert(terlan_ndarray_v1_delete(special) == 0);
  assert(terlan_ndarray_v1_delete(decreasing) == 0);
  assert(terlan_ndarray_v1_delete(increasing) == 0);
  assert(terlan_ndarray_v1_delete(value) == 0);
}

static void test_counting_rejections_and_allocation_cleanup(void) {
  TerlanNdArrayHandle negative =
      integers((int64_t[]){0, -1}, 2, (int64_t[]){2}, 1);
  TerlanNdArrayHandle matrix =
      integers((int64_t[]){0, 1}, 2, (int64_t[]){1, 2}, 2);
  TerlanNdArrayHandle values =
      integers((int64_t[]){0, 1}, 2, (int64_t[]){2}, 1);
  TerlanNdArrayHandle mixed_bins =
      integers((int64_t[]){0, 0, 1, 0}, 4, (int64_t[]){4}, 1);
  TerlanNdArrayHandle output = NULL;
  const int64_t baseline = terlan_ndarray_v1_live_count();
  int64_t failure;
  assert(terlan_ndarray_v1_bincount_unweighted(negative, 0, &output) ==
         TERLAN_NDARRAY_DOMAIN_ERROR);
  assert(terlan_ndarray_v1_bincount_unweighted(matrix, 0, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_bincount_unweighted(values, -1, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_digitize(values, mixed_bins, 0, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(output == NULL);
  for (failure = 0; failure < 2; ++failure) {
    terlan_ndarray_test_fail_allocation_after(failure);
    output = NULL;
    if (terlan_ndarray_v1_bincount_unweighted(values, 4, &output) == 0) {
      assert(terlan_ndarray_v1_delete(output) == 0);
    } else {
      assert(output == NULL);
    }
    terlan_ndarray_test_reset_allocation_failures();
    assert(terlan_ndarray_v1_live_count() == baseline);
  }
  assert(terlan_ndarray_v1_delete(mixed_bins) == 0);
  assert(terlan_ndarray_v1_delete(values) == 0);
  assert(terlan_ndarray_v1_delete(matrix) == 0);
  assert(terlan_ndarray_v1_delete(negative) == 0);
}

int main(void) {
  test_unweighted_bincount_values_empty_and_minimum_length();
  test_weighted_bincount_is_float64_and_keeps_nonfinite_values();
  test_digitize_increasing_decreasing_repeated_and_right_edges();
  test_digitize_large_integers_and_nonfinite_values();
  test_counting_rejections_and_allocation_cleanup();
  return 0;
}
