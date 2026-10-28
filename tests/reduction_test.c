#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

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
  const double tolerance =
      array->dtype == TERLAN_NDARRAY_DTYPE_FLOAT16
          ? 1e-3
          : array->dtype == TERLAN_NDARRAY_DTYPE_FLOAT32 ? 1e-6 : 1e-12;
  assert(terlan_ndarray_v1_to_floats_cast(array, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert((isnan(values[index]) && isnan(expected[index])) ||
           fabs(values[index] - expected[index]) < tolerance);
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
    assert(fabs(values[index] - expected_real[index]) < 1e-6);
  }
  terlan_ndarray_v1_delete_float_values(values);
  assert(terlan_ndarray_v1_complex_imags(array, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(fabs(values[index] - expected_imag[index]) < 1e-6);
  }
  terlan_ndarray_v1_delete_float_values(values);
}

static void test_numeric_reductions_and_axis_shapes(void) {
  const int64_t values[] = {1, 2, 3, 4, 5, 6};
  const int64_t shape[] = {2, 3};
  const int64_t axis[] = {1};
  const int64_t sums[] = {6, 15};
  const int64_t products[] = {6, 120};
  const int64_t minimums[] = {1, 4};
  const int64_t maximums[] = {3, 6};
  const double means[] = {2.0, 5.0};
  const double variances[] = {2.0 / 3.0, 2.0 / 3.0};
  const double deviations[] = {sqrt(2.0 / 3.0), sqrt(2.0 / 3.0)};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_from_ints_as(values, 6, shape, 2,
                                         TERLAN_NDARRAY_DTYPE_INT8, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sum(array, axis, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, sums, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_product(array, axis, 1, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, products, 2);
  assert(result->rank == 2 && result->shape[0] == 2 && result->shape[1] == 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_mean(array, axis, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, means, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_minimum(array, axis, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, minimums, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_maximum(array, axis, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, maximums, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance(array, axis, 1, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, variances, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_standard_deviation(array, axis, 1, 0, 0,
                                               &result) == TERLAN_NDARRAY_OK);
  assert_floats(result, deviations, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_truth_and_argument_reductions(void) {
  const int64_t values[] = {4, 1, 3, 0, 7, 2};
  const int64_t shape[] = {2, 3};
  const int64_t axis[] = {-1};
  const int64_t argmins[] = {1, 0};
  const int64_t argmaxs[] = {0, 1};
  const uint8_t any_values[] = {1, 1};
  const uint8_t all_values[] = {1, 0};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_from_ints(values, 6, shape, 2, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argmin(array, axis, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, argmins, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argmax(array, axis, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, argmaxs, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_any(array, axis, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_bools(result, any_values, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_all(array, axis, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_bools(result, all_values, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_empty_identities_and_domains(void) {
  const int64_t shape[] = {2, 0, 3};
  const int64_t axis[] = {1};
  const int64_t sums[] = {0, 0, 0, 0, 0, 0};
  const int64_t products[] = {1, 1, 1, 1, 1, 1};
  const uint8_t anys[] = {0, 0, 0, 0, 0, 0};
  const uint8_t alls[] = {1, 1, 1, 1, 1, 1};
  const double undefined[] = {NAN, NAN, NAN, NAN, NAN, NAN};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_from_ints(NULL, 0, shape, 3, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sum(array, axis, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, sums, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_product(array, axis, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, products, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_any(array, axis, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_bools(result, anys, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_all(array, axis, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_bools(result, alls, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_mean(array, axis, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, undefined, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance(array, axis, 1, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, undefined, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_standard_deviation(array, axis, 1, 0, 0,
                                               &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, undefined, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_minimum(array, axis, 1, 0, &result) ==
         TERLAN_NDARRAY_DOMAIN_ERROR);
  assert(terlan_ndarray_v1_argmax(array, axis, 1, 0, &result) ==
         TERLAN_NDARRAY_DOMAIN_ERROR);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_axis_and_correction_rejections(void) {
  const int64_t value[] = {1};
  const int64_t duplicate_axes[] = {0, 0};
  const int64_t bad_axis[] = {1};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_from_ints(value, 1, (const int64_t[]){1}, 1,
                                     &array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sum(array, duplicate_axes, 2, 0, &result) ==
         TERLAN_NDARRAY_DUPLICATE_AXIS);
  assert(terlan_ndarray_v1_sum(array, bad_axis, 1, 0, &result) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(terlan_ndarray_v1_mean(array, bad_axis, 1, 0, &result) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(terlan_ndarray_v1_variance(array, bad_axis, 1, 0, 0, &result) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(terlan_ndarray_v1_variance(array, NULL, 0, 0, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){NAN}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_standard_deviation(array, NULL, 0, 0, 2,
                                               &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){NAN}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_nan_infinity_ties_and_correction_are_frozen(void) {
  const double non_finite[] = {3.0, NAN, -INFINITY, NAN, INFINITY};
  const int64_t ties[] = {2, 1, 1, 3, 3};
  const double variance_values[] = {1.0, 2.0, 3.0};
  TerlanNdArrayHandle values = NULL;
  TerlanNdArrayHandle tied = NULL;
  TerlanNdArrayHandle variance = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_from_floats(non_finite, 5,
                                        (const int64_t[]){5}, 1, &values) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_minimum(values, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert(isnan(((double *)result->data)[0]));
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_maximum(values, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert(isnan(((double *)result->data)[0]));
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argmin(values, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, (const int64_t[]){1}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argmax(values, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, (const int64_t[]){1}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_ints(ties, 5, (const int64_t[]){5}, 1,
                                     &tied) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argmin(tied, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, (const int64_t[]){1}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argmax(tied, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, (const int64_t[]){3}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_floats(variance_values, 3,
                                        (const int64_t[]){3}, 1,
                                        &variance) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance(variance, NULL, 0, 0, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){1.0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_standard_deviation(variance, NULL, 0, 0, 1,
                                               &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){1.0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(variance) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(tied) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(values) == TERLAN_NDARRAY_OK);
}

static void test_truth_reductions_match_numpy_dtypes_axes_and_strides(void) {
  uint8_t false_values[256] = {0};
  uint8_t true_values[256];
  const int64_t vector_shape[] = {256};
  const int64_t matrix_shape[] = {3, 3};
  const int64_t matrix_values[] = {0, 0, 0, 0, 1, 0, 1, 1, 0};
  const int64_t axis_zero[] = {0};
  const int64_t axis_one[] = {1};
  const double floating_values[] = {0.1, 0.9};
  const double nan_values[] = {NAN};
  const double complex_reals[] = {0.0, 2.0};
  const double complex_imags[] = {1.0, 0.0};
  const uint8_t axis_zero_any[] = {1, 1, 0};
  const uint8_t axis_one_any[] = {0, 1, 1};
  TerlanNdArrayHandle false_backing = NULL;
  TerlanNdArrayHandle true_backing = NULL;
  TerlanNdArrayHandle false_view = NULL;
  TerlanNdArrayHandle true_view = NULL;
  TerlanNdArrayHandle matrix = NULL;
  TerlanNdArrayHandle floating = NULL;
  TerlanNdArrayHandle nan_array = NULL;
  TerlanNdArrayHandle complex = NULL;
  TerlanNdArrayHandle result = NULL;
  int64_t index;

  for (index = 0; index < 256; ++index) {
    true_values[index] = 1;
  }
  assert(terlan_ndarray_v1_from_bools(false_values, 256, vector_shape, 1,
                                      &false_backing) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_bools(true_values, 256, vector_shape, 1,
                                      &true_backing) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_slice_axis(false_backing, 0, 7, 256, 1,
                                      &false_view) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_slice_axis(true_backing, 0, 7, 256, 1,
                                      &true_view) == TERLAN_NDARRAY_OK);
  for (index = 0; index < 249; ++index) {
    const int64_t coordinate[] = {index};
    assert(terlan_ndarray_v1_set_bool(false_view, coordinate, 1, true) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_any(false_view, NULL, 0, 0, &result) ==
           TERLAN_NDARRAY_OK);
    assert_bools(result, (const uint8_t[]){1}, 1);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_set_bool(false_view, coordinate, 1, false) ==
           TERLAN_NDARRAY_OK);

    assert(terlan_ndarray_v1_set_bool(true_view, coordinate, 1, false) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_all(true_view, NULL, 0, 0, &result) ==
           TERLAN_NDARRAY_OK);
    assert_bools(result, (const uint8_t[]){0}, 1);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_set_bool(true_view, coordinate, 1, true) ==
           TERLAN_NDARRAY_OK);
  }

  assert(terlan_ndarray_v1_from_ints(matrix_values, 9, matrix_shape, 2,
                                     &matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_any(matrix, axis_zero, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_bools(result, axis_zero_any, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_any(matrix, axis_one, 1, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert(result->rank == 2 && result->shape[0] == 3 && result->shape[1] == 1);
  assert_bools(result, axis_one_any, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_floats(floating_values, 2,
                                       (const int64_t[]){2}, 1, &floating) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_all(floating, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_bools(result, (const uint8_t[]){1}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(nan_values, 1,
                                       (const int64_t[]){1}, 1, &nan_array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_all(nan_array, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_bools(result, (const uint8_t[]){1}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_complex(
             complex_reals, 2, complex_imags, 2, (const int64_t[]){2}, 1,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &complex) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_all(complex, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_bools(result, (const uint8_t[]){1}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_delete(complex) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(nan_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(floating) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(true_view) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(false_view) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(true_backing) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(false_backing) == TERLAN_NDARRAY_OK);
}

static void test_statistic_keepdims_dtype_and_out_semantics(void) {
  const int64_t eye_values[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const int64_t shape[] = {3, 3};
  const int64_t axis[] = {1};
  const int64_t output_shape[] = {3};
  const double thirds[] = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
  const double variances[] = {2.0 / 9.0, 2.0 / 9.0, 2.0 / 9.0};
  const double deviations[] = {sqrt(2.0) / 3.0, sqrt(2.0) / 3.0,
                               sqrt(2.0) / 3.0};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle destination = NULL;
  TerlanNdArrayHandle wrong = NULL;
  TerlanNdArrayHandle result = NULL;
  int32_t dtype = 0;

  assert(terlan_ndarray_v1_from_ints_as(eye_values, 9, shape, 2,
                                        TERLAN_NDARRAY_DTYPE_INT16, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_mean(array, axis, 1, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert(result->rank == 2 && result->shape[0] == 3 && result->shape[1] == 1);
  assert_floats(result, thirds, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_mean(array, NULL, 0, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert(result->rank == 2 && result->shape[0] == 1 && result->shape[1] == 1);
  assert_floats(result, (const double[]){1.0 / 3.0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance(array, axis, 1, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert(result->rank == 2 && result->shape[0] == 3 && result->shape[1] == 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_standard_deviation(array, NULL, 0, 1, 0,
                                               &result) ==
         TERLAN_NDARRAY_OK);
  assert(result->rank == 2 && result->shape[0] == 1 && result->shape[1] == 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_mean_as(array, axis, 1, 0,
                                   TERLAN_NDARRAY_DTYPE_FLOAT32, &result) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dtype(result, &dtype) == TERLAN_NDARRAY_OK);
  assert(dtype == TERLAN_NDARRAY_DTYPE_FLOAT32);
  assert_floats(result, thirds, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance_as(
             array, axis, 1, 0, 0, TERLAN_NDARRAY_DTYPE_FLOAT32, &result) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dtype(result, &dtype) == TERLAN_NDARRAY_OK);
  assert(dtype == TERLAN_NDARRAY_DTYPE_FLOAT32);
  assert_floats(result, variances, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_standard_deviation_as(
             array, axis, 1, 0, 0, TERLAN_NDARRAY_DTYPE_FLOAT32, &result) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dtype(result, &dtype) == TERLAN_NDARRAY_OK);
  assert(dtype == TERLAN_NDARRAY_DTYPE_FLOAT32);
  assert_floats(result, deviations, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_floats_as(
             (const double[]){0.0, 0.0, 0.0}, 3, output_shape, 1,
             TERLAN_NDARRAY_DTYPE_FLOAT32, &destination) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_mean_into(destination, array, axis, 1, 0) ==
         TERLAN_NDARRAY_OK);
  assert_floats(destination, thirds, 3);
  assert(terlan_ndarray_v1_variance_into(destination, array, axis, 1, 0, 0) ==
         TERLAN_NDARRAY_OK);
  assert_floats(destination, variances, 3);
  assert(terlan_ndarray_v1_standard_deviation_into(
             destination, array, axis, 1, 0, 0) == TERLAN_NDARRAY_OK);
  assert_floats(destination, deviations, 3);
  assert(terlan_ndarray_v1_from_floats((const double[]){0.0, 0.0}, 2,
                                       (const int64_t[]){2}, 1, &wrong) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_mean_into(wrong, array, axis, 1, 0) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(terlan_ndarray_v1_standard_deviation_into(wrong, array, axis, 1, 0,
                                                    0) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert_floats(wrong, (const double[]){0.0, 0.0}, 2);

  assert(terlan_ndarray_v1_delete(wrong) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_masked_truth_reductions_match_numpy_identities(void) {
  const uint8_t values[] = {1, 0, 1, 0, 0, 0, 1, 1, 1};
  const int64_t shape[] = {3, 3};
  const uint8_t all_full_values[] = {1, 0, 1, 0, 0, 0, 1, 0, 1};
  const uint8_t any_full_values[] = {0, 1, 0, 1, 1, 1, 0, 0, 0};
  const uint8_t lower_values[] = {0, 0, 1};
  const uint8_t middle_values[] = {0, 1, 0};
  const int64_t axis[] = {0};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle all_full = NULL;
  TerlanNdArrayHandle any_full = NULL;
  TerlanNdArrayHandle lower = NULL;
  TerlanNdArrayHandle middle = NULL;
  TerlanNdArrayHandle false_mask = NULL;
  TerlanNdArrayHandle result = NULL;

  assert(terlan_ndarray_v1_from_bools(values, 9, shape, 2, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_bools(all_full_values, 9, shape, 2,
                                      &all_full) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_bools(any_full_values, 9, shape, 2,
                                      &any_full) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_bools(lower_values, 3,
                                      (const int64_t[]){3, 1}, 2, &lower) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_bools(middle_values, 3,
                                      (const int64_t[]){3, 1}, 2, &middle) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_bools((const uint8_t[]){0}, 1, NULL, 0,
                                      &false_mask) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_all_where(array, lower, axis, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_bools(result, (const uint8_t[]){1, 1, 1}, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_all_where(array, all_full, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_bools(result, (const uint8_t[]){1}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_all_where(array, false_mask, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_bools(result, (const uint8_t[]){1}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_any_where(array, middle, axis, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_bools(result, (const uint8_t[]){0, 0, 0}, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_any_where(array, any_full, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_bools(result, (const uint8_t[]){0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_any_where(array, false_mask, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_bools(result, (const uint8_t[]){0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_delete(false_mask) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(middle) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(lower) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(any_full) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(all_full) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_masked_statistics_broadcast_and_empty_groups(void) {
  const int64_t values[] = {0, 1, 2, 3, 4, 5, 6, 7,
                            8, 9, 10, 11, 12, 13, 14, 15};
  const int64_t shape[] = {4, 4};
  const uint8_t mask_values[] = {0, 1, 0, 1, 1, 0, 1, 0,
                                 1, 1, 0, 0, 0, 0, 1, 1};
  const uint8_t partial_values[] = {0, 1, 1, 0};
  const double row_means[] = {2.0, 5.0, 8.5, 14.5};
  const double column_means[] = {6.0, 5.0, 10.0, 9.0};
  const double partial_means[] = {6.0, 7.0, 8.0, 9.0};
  const int64_t axis_zero[] = {0};
  const int64_t axis_one[] = {1};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle mask = NULL;
  TerlanNdArrayHandle partial = NULL;
  TerlanNdArrayHandle false_mask = NULL;
  TerlanNdArrayHandle result = NULL;

  assert(terlan_ndarray_v1_from_ints(values, 16, shape, 2, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_bools(mask_values, 16, shape, 2, &mask) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_bools(partial_values, 4,
                                      (const int64_t[]){4, 1}, 2, &partial) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_mean_where(array, mask, axis_zero, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, column_means, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_mean_where(array, mask, axis_one, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, row_means, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_mean_where(array, partial, axis_zero, 1, 0,
                                      &result) == TERLAN_NDARRAY_OK);
  assert_floats(result, partial_means, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_variance_where(array, NULL, axis_zero, 1, 0, 0,
                                           &result) == TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){20.0, 20.0, 20.0, 20.0}, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_standard_deviation_where(
             array, NULL, axis_one, 1, 0, 0, &result) == TERLAN_NDARRAY_OK);
  assert_floats(result,
                (const double[]){sqrt(1.25), sqrt(1.25), sqrt(1.25),
                                 sqrt(1.25)},
                4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_bools((const uint8_t[]){0}, 1, NULL, 0,
                                      &false_mask) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_mean_where(array, false_mask, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){NAN}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance_where(array, false_mask, NULL, 0, 0, 0,
                                           &result) == TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){NAN}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_standard_deviation_where(
             array, false_mask, NULL, 0, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){NAN}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_delete(false_mask) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(partial) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(mask) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_statistic_dtype_defaults_and_float16_accumulation(void) {
  const int32_t integer_dtypes[] = {
      TERLAN_NDARRAY_DTYPE_BOOL,   TERLAN_NDARRAY_DTYPE_INT8,
      TERLAN_NDARRAY_DTYPE_INT16,  TERLAN_NDARRAY_DTYPE_INT32,
      TERLAN_NDARRAY_DTYPE_INT64,  TERLAN_NDARRAY_DTYPE_UINT8,
      TERLAN_NDARRAY_DTYPE_UINT16, TERLAN_NDARRAY_DTYPE_UINT32,
      TERLAN_NDARRAY_DTYPE_UINT64};
  const int32_t float_dtypes[] = {TERLAN_NDARRAY_DTYPE_FLOAT16,
                                  TERLAN_NDARRAY_DTYPE_FLOAT32,
                                  TERLAN_NDARRAY_DTYPE_FLOAT64};
  const int64_t values[] = {1, 0, 0, 1};
  const double floating_values[] = {1.0, 0.0, 0.0, 1.0};
  const int64_t shape[] = {2, 2};
  const int64_t axis[] = {1};
  size_t index;
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle result = NULL;

  for (index = 0; index < sizeof(integer_dtypes) / sizeof(integer_dtypes[0]);
       ++index) {
    assert(terlan_ndarray_v1_from_ints_as(values, 4, shape, 2,
                                           integer_dtypes[index], &array) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_mean(array, axis, 1, 0, &result) ==
           TERLAN_NDARRAY_OK);
    assert(result->dtype == TERLAN_NDARRAY_DTYPE_FLOAT64);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_variance(array, axis, 1, 0, 0, &result) ==
           TERLAN_NDARRAY_OK);
    assert(result->dtype == TERLAN_NDARRAY_DTYPE_FLOAT64);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_standard_deviation(array, axis, 1, 0, 0,
                                                 &result) ==
           TERLAN_NDARRAY_OK);
    assert(result->dtype == TERLAN_NDARRAY_DTYPE_FLOAT64);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  }
  for (index = 0; index < sizeof(float_dtypes) / sizeof(float_dtypes[0]);
       ++index) {
    assert(terlan_ndarray_v1_from_floats_as(floating_values, 4, shape, 2,
                                             float_dtypes[index], &array) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_mean(array, axis, 1, 0, &result) ==
           TERLAN_NDARRAY_OK);
    assert(result->dtype == float_dtypes[index]);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_variance(array, axis, 1, 0, 0, &result) ==
           TERLAN_NDARRAY_OK);
    assert(result->dtype == float_dtypes[index]);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_standard_deviation(array, axis, 1, 0, 0,
                                                 &result) ==
           TERLAN_NDARRAY_OK);
    assert(result->dtype == float_dtypes[index]);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  }
  assert(terlan_ndarray_v1_from_floats(floating_values, 4, shape, 2, &array) ==
         TERLAN_NDARRAY_OK);
  for (index = 0; index < sizeof(float_dtypes) / sizeof(float_dtypes[0]);
       ++index) {
    assert(terlan_ndarray_v1_mean_as(array, axis, 1, 0, float_dtypes[index],
                                     &result) == TERLAN_NDARRAY_OK);
    assert(result->dtype == float_dtypes[index]);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_variance_as(array, axis, 1, 0, 0,
                                         float_dtypes[index], &result) ==
           TERLAN_NDARRAY_OK);
    assert(result->dtype == float_dtypes[index]);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_standard_deviation_as(
               array, axis, 1, 0, 0, float_dtypes[index], &result) ==
           TERLAN_NDARRAY_OK);
    assert(result->dtype == float_dtypes[index]);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  }
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  {
    double *ones = malloc(100000 * sizeof(*ones));
    assert(ones != NULL);
    for (index = 0; index < 100000; ++index) {
      ones[index] = 1.0;
    }
    assert(terlan_ndarray_v1_from_floats_as(
               ones, 100000, (const int64_t[]){100000}, 1,
               TERLAN_NDARRAY_DTYPE_FLOAT16, &array) == TERLAN_NDARRAY_OK);
    free(ones);
    assert(terlan_ndarray_v1_mean(array, NULL, 0, 0, &result) ==
           TERLAN_NDARRAY_OK);
    assert(result->dtype == TERLAN_NDARRAY_DTYPE_FLOAT16);
    assert_floats(result, (const double[]){1.0}, 1);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  }
}

static void test_statistic_correction_and_complex_semantics(void) {
  const int64_t alternating[] = {1, -1, 1, -1};
  const double circle_real[] = {1.0, 0.0, -1.0, 0.0};
  const double circle_imag[] = {0.0, 1.0, 0.0, -1.0};
  const double real[] = {1.0, 3.0, 5.0, 7.0};
  const double imag[] = {1.0, 3.0, 5.0, 7.0};
  const double stacked_real[] = {1.0, 3.0, 5.0, 7.0,
                                 1.0, 3.0, 5.0, 7.0};
  const double stacked_imag[] = {1.0, 3.0, 5.0, 7.0,
                                 1.0, 3.0, 5.0, 7.0};
  const double values[] = {1.0, 2.0, 3.0, 4.0, 5.0};
  const int64_t axis_zero[] = {0};
  const int64_t axis_one[] = {1};
  const int64_t axis_two[] = {2};
  TerlanNdArrayHandle complex = NULL;
  TerlanNdArrayHandle complex_circle = NULL;
  TerlanNdArrayHandle complex_scalar = NULL;
  TerlanNdArrayHandle complex32 = NULL;
  TerlanNdArrayHandle stacked = NULL;
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle scalar = NULL;
  TerlanNdArrayHandle destination = NULL;
  TerlanNdArrayHandle result = NULL;
  int32_t dtype = 0;

  assert(terlan_ndarray_v1_from_ints(alternating, 4,
                                     (const int64_t[]){4}, 1, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance(array, NULL, 0, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){1.0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance(array, NULL, 0, 0, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){4.0 / 3.0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance(array, NULL, 0, 0, 2, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){2.0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_standard_deviation(array, NULL, 0, 0, 0,
                                               &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){1.0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats((const double[]){-1.0}, 1, NULL, 0,
                                       &destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_standard_deviation_into(destination, array, NULL, 0,
                                                    0, 0) ==
         TERLAN_NDARRAY_OK);
  assert_floats(destination, (const double[]){1.0}, 1);
  assert(terlan_ndarray_v1_variance_into(destination, array, NULL, 0, 0, 0) ==
         TERLAN_NDARRAY_OK);
  assert_floats(destination, (const double[]){1.0}, 1);
  assert(terlan_ndarray_v1_mean_into(destination, array, NULL, 0, 0) ==
         TERLAN_NDARRAY_OK);
  assert_floats(destination, (const double[]){0.0}, 1);
  assert(terlan_ndarray_v1_delete(destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_ints((const int64_t[]){1}, 1, NULL, 0,
                                     &scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance(scalar, NULL, 0, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){0.0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_standard_deviation(scalar, NULL, 0, 0, 0,
                                               &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){0.0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_complex(
             circle_real, 4, circle_imag, 4, (const int64_t[]){4}, 1,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &complex_circle) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance(complex_circle, NULL, 0, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){1.0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_standard_deviation(complex_circle, NULL, 0, 0, 0,
                                               &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){1.0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(complex_circle) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_complex(
             (const double[]){0.0}, 1, (const double[]){1.0}, 1, NULL, 0,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &complex_scalar) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance(complex_scalar, NULL, 0, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){0.0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_standard_deviation(complex_scalar, NULL, 0, 0, 0,
                                               &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){0.0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(complex_scalar) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_floats(values, 5, (const int64_t[]){5}, 1,
                                       &array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance(array, NULL, 0, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){2.0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance(array, NULL, 0, 0, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){2.5}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance(array, NULL, 0, 0, 2, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){10.0 / 3.0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_standard_deviation(array, NULL, 0, 0, 0,
                                               &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){sqrt(2.0)}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_standard_deviation(array, NULL, 0, 0, 1,
                                               &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){sqrt(2.5)}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_standard_deviation(array, NULL, 0, 0, 2,
                                               &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){sqrt(10.0 / 3.0)}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_complex(
             real, 4, imag, 4, (const int64_t[]){2, 2}, 2,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &complex) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_mean(complex, axis_zero, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_complex(result, (const double[]){3.0, 5.0},
                 (const double[]){3.0, 5.0}, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_mean(complex, axis_one, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_complex(result, (const double[]){2.0, 6.0},
                 (const double[]){2.0, 6.0}, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance(complex, NULL, 0, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dtype(result, &dtype) == TERLAN_NDARRAY_OK);
  assert(dtype == TERLAN_NDARRAY_DTYPE_FLOAT64);
  assert_floats(result, (const double[]){10.0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_standard_deviation(complex, NULL, 0, 0, 0,
                                               &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){sqrt(10.0)}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_complex(
             real, 4, imag, 4, (const int64_t[]){2, 2}, 2,
             TERLAN_NDARRAY_DTYPE_COMPLEX64, &complex32) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance(complex32, NULL, 0, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert(result->dtype == TERLAN_NDARRAY_DTYPE_FLOAT32);
  assert_floats(result, (const double[]){10.0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_standard_deviation(complex32, axis_one, 1, 0, 0,
                                               &result) ==
         TERLAN_NDARRAY_OK);
  assert(result->dtype == TERLAN_NDARRAY_DTYPE_FLOAT32);
  assert_floats(result, (const double[]){sqrt(2.0), sqrt(2.0)}, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_complex(
             stacked_real, 8, stacked_imag, 8, (const int64_t[]){2, 2, 2}, 3,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &stacked) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance(stacked, axis_zero, 1, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){0.0, 0.0, 0.0, 0.0}, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance(stacked, axis_one, 1, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){8.0, 8.0, 8.0, 8.0}, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_variance(stacked, axis_two, 1, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){2.0, 2.0, 2.0, 2.0}, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_delete(stacked) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(complex32) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(complex) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_cumulative_reductions(void) {
  const int64_t values[] = {1, 2, 3, 4, 5, 6};
  const int64_t shape[] = {2, 3};
  const int64_t sums[] = {1, 3, 6, 4, 9, 15};
  const int64_t products[] = {1, 2, 6, 4, 20, 120};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_from_ints_as(values, 6, shape, 2,
                                         TERLAN_NDARRAY_DTYPE_INT8, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_cumulative_sum(array, -1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, sums, 6);
  assert(result->dtype == TERLAN_NDARRAY_DTYPE_INT64);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_cumulative_product(array, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, products, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_cumulative_sum(array, 2, &result) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(result == NULL);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_axis_aware_count_nonzero(void) {
  const int64_t values[] = {0, 2, 0, 4, 5, 0};
  const int64_t shape[] = {2, 3};
  const int64_t axis[] = {1};
  const int64_t counts[] = {1, 2};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_from_ints(values, 6, shape, 2, &array) == 0);
  assert(terlan_ndarray_v1_count_nonzero_axes(array, axis, 1, 1, &result) ==
         0);
  assert(result->rank == 2 && result->shape[0] == 2 && result->shape[1] == 1);
  assert_ints(result, counts, 2);
  assert(terlan_ndarray_v1_delete(result) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

int main(void) {
  test_numeric_reductions_and_axis_shapes();
  test_truth_and_argument_reductions();
  test_empty_identities_and_domains();
  test_axis_and_correction_rejections();
  test_nan_infinity_ties_and_correction_are_frozen();
  test_truth_reductions_match_numpy_dtypes_axes_and_strides();
  test_statistic_keepdims_dtype_and_out_semantics();
  test_masked_truth_reductions_match_numpy_identities();
  test_masked_statistics_broadcast_and_empty_groups();
  test_statistic_dtype_defaults_and_float16_accumulation();
  test_statistic_correction_and_complex_semantics();
  test_cumulative_reductions();
  test_axis_aware_count_nonzero();
  return 0;
}
