#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static TerlanNdArrayHandle floats(const double *values, int64_t count,
                                  const int64_t *shape, int64_t rank) {
  TerlanNdArrayHandle array = NULL;
  assert(terlan_ndarray_v1_from_floats(values, count, shape, rank, &array) ==
         0);
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

static TerlanNdArrayHandle complex_values(const double *reals,
                                          const double *imags, int64_t count,
                                          const int64_t *shape, int64_t rank,
                                          int32_t dtype) {
  TerlanNdArrayHandle array = NULL;
  assert(terlan_ndarray_v1_from_complex(reals, count, imags, count, shape,
                                        rank, dtype, &array) == 0);
  return array;
}

static void assert_floats(TerlanNdArrayHandle array, const double *expected,
                          size_t count) {
  double *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_floats_cast(array, &values, &length) == 0);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(fabsl(values[index] - expected[index]) < 1e-10L);
  }
  terlan_ndarray_v1_delete_float_values(values);
}

static void assert_complex_reals(TerlanNdArrayHandle array,
                                 const double *expected, size_t count) {
  double *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_complex_reals(array, &values, &length) == 0);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(fabsl(values[index] - expected[index]) < 1e-10L);
  }
  terlan_ndarray_v1_delete_float_values(values);
}

static void assert_complex_values(TerlanNdArrayHandle array,
                                  const double *expected_reals,
                                  const double *expected_imags, size_t count) {
  double *reals = NULL;
  double *imags = NULL;
  size_t real_count = 0;
  size_t imag_count = 0;
  size_t index;
  assert(terlan_ndarray_v1_complex_reals(array, &reals, &real_count) == 0);
  assert(terlan_ndarray_v1_complex_imags(array, &imags, &imag_count) == 0);
  assert(real_count == count && imag_count == count);
  for (index = 0; index < count; ++index) {
    assert(fabsl(reals[index] - expected_reals[index]) < 1e-6L);
    assert(fabsl(imags[index] - expected_imags[index]) < 1e-6L);
  }
  terlan_ndarray_v1_delete_float_values(imags);
  terlan_ndarray_v1_delete_float_values(reals);
}

static void assert_shape(TerlanNdArrayHandle array, const int64_t *expected,
                         size_t count) {
  int64_t *shape = NULL;
  int64_t rank = 0;
  size_t index;
  assert(terlan_ndarray_v1_rank(array, &rank) == 0);
  assert(terlan_ndarray_v1_shape(array, &shape) == 0);
  assert(rank == (int64_t)count);
  for (index = 0; index < count; ++index) {
    assert(shape[index] == expected[index]);
  }
}

static void assert_dtype(TerlanNdArrayHandle array, int32_t expected) {
  int32_t actual = 0;
  assert(terlan_ndarray_v1_dtype(array, &actual) == TERLAN_NDARRAY_OK);
  assert(actual == expected);
}

static void assert_floats_with_nan(TerlanNdArrayHandle array,
                                   const double *expected, size_t count) {
  double *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_floats_cast(array, &values, &length) == 0);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    if (isnan(expected[index])) {
      assert(isnan(values[index]));
    } else if (isinf(expected[index])) {
      assert(values[index] == expected[index]);
    } else {
      assert(fabsl(values[index] - expected[index]) < 1e-10L);
    }
  }
  terlan_ndarray_v1_delete_float_values(values);
}

static void test_quantiles_and_range_across_axes(void) {
  const double values[] = {1.0, 4.0, 2.0, 8.0, 3.0, 6.0};
  const int64_t shape[] = {2, 3};
  const int64_t axis = 1;
  TerlanNdArrayHandle array = floats(values, 6, shape, 2);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_median(array, &axis, 1, 0, &output) == 0);
  assert_floats(output, (double[]){2.0, 6.0}, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_quantile(array, 0.25, &axis, 1, 1, 0, &output) ==
         0);
  assert_floats(output, (double[]){1.5, 4.5}, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_percentile(array, 75.0, &axis, 1, 0, 3,
                                      &output) == 0);
  assert_floats(output, (double[]){3.0, 7.0}, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_peak_to_peak(array, &axis, 1, 0, &output) == 0);
  assert_floats(output, (double[]){3.0, 5.0}, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_quantile_sequences_axes_shapes_and_interpolation(void) {
  const double values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  const double percentiles[] = {25, 50, 100};
  const int64_t shape[] = {3, 4};
  const int64_t axis_zero = 0;
  const int64_t axis_one = 1;
  TerlanNdArrayHandle array = floats(values, 12, shape, 2);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_percentiles(array, percentiles, 3, &axis_zero, 1,
                                       0, 0, &output) == 0);
  assert_shape(output, (int64_t[]){3, 4}, 2);
  assert_floats(output,
                (double[]){2, 3, 4, 5, 4, 5, 6, 7, 8, 9, 10, 11}, 12);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_percentiles(array, percentiles, 3, &axis_one, 1,
                                       0, 0, &output) == 0);
  assert_shape(output, (int64_t[]){3, 3}, 2);
  assert_floats(output,
                (double[]){0.75, 4.75, 8.75, 1.5, 5.5, 9.5, 3, 7, 11},
                9);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_percentiles(array, NULL, 0, NULL, 0, 0, 0,
                                       &output) == 0);
  assert_shape(output, (int64_t[]){0}, 1);
  assert_floats(output, NULL, 0);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_percentile(array, 51, NULL, 0, 0, 0, &output) ==
         0);
  assert_floats(output, (double[]){5.61}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_percentile(array, 51, NULL, 0, 0, 1, &output) ==
         0);
  assert_floats(output, (double[]){5}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_percentile(array, 51, NULL, 0, 0, 2, &output) ==
         0);
  assert_floats(output, (double[]){6}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_percentile(array, 51, NULL, 0, 0, 3, &output) ==
         0);
  assert_floats(output, (double[]){5.5}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_percentile(array, 51, NULL, 0, 0, 4, &output) ==
         0);
  assert_floats(output, (double[]){6}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  {
    const uint8_t truth_values[] = {0, 1, 1};
    const double truth_quantiles[] = {0, 1, 1};
    TerlanNdArrayHandle truth = NULL;
    uint8_t *actual = NULL;
    size_t actual_count = 0;
    int32_t dtype = 0;
    assert(terlan_ndarray_v1_from_bools(truth_values, 3, (int64_t[]){3}, 1,
                                        &truth) == 0);
    assert(terlan_ndarray_v1_quantiles(truth, truth_quantiles, 3, NULL, 0, 0,
                                       0, &output) == 0);
    assert(terlan_ndarray_v1_dtype(output, &dtype) == 0);
    assert(dtype == TERLAN_NDARRAY_DTYPE_BOOL);
    assert(terlan_ndarray_v1_to_bools(output, &actual, &actual_count) == 0);
    assert(actual_count == 3 && actual[0] == 0 && actual[1] == 1 &&
           actual[2] == 1);
    terlan_ndarray_v1_delete_bool_values(actual);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_delete(truth) == 0);
  }
  assert_floats(array, values, 12);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_quantile_sequences_are_monotonic(void) {
  const double values[] = {0.0, 0.1, 0.1, 0.2, 0.2, 0.3,
                           0.3, 0.4, 0.5, 0.5, 0.1, 0.1,
                           0.9, 0.9, 0.9, 0.8, 0.8, 0.7};
  double quantiles[101];
  double *actual = NULL;
  size_t actual_count = 0;
  size_t index;
  TerlanNdArrayHandle array = floats(values, 18, (int64_t[]){18}, 1);
  TerlanNdArrayHandle output = NULL;
  for (index = 0; index < 101; ++index) {
    quantiles[index] = (double)index / 100.0;
  }
  assert(terlan_ndarray_v1_quantiles(array, quantiles, 101, NULL, 0, 0, 0,
                                     &output) == 0);
  assert(terlan_ndarray_v1_to_floats_cast(output, &actual, &actual_count) == 0);
  assert(actual_count == 101);
  for (index = 1; index < actual_count; ++index) {
    assert(actual[index - 1] <= actual[index]);
  }
  terlan_ndarray_v1_delete_float_values(actual);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_quantile_nan_and_empty_dimension_semantics(void) {
  const double values[] = {NAN, 1, 3, NAN, NAN, NAN};
  const double quantiles[] = {0, 1};
  const int64_t axis_one = 1;
  const int64_t axis_zero = 0;
  TerlanNdArrayHandle array = floats(values, 6, (int64_t[]){2, 3}, 2);
  TerlanNdArrayHandle empty = floats(NULL, 0, (int64_t[]){0, 3}, 2);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_quantiles(array, quantiles, 2, &axis_one, 1, 0, 0,
                                     &output) == 0);
  assert_shape(output, (int64_t[]){2, 2}, 2);
  assert_floats_with_nan(output, (double[]){NAN, NAN, NAN, NAN}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_nanquantiles(array, quantiles, 2, &axis_one, 1, 0,
                                        0, &output) == 0);
  assert_shape(output, (int64_t[]){2, 2}, 2);
  assert_floats_with_nan(output, (double[]){1, NAN, 3, NAN}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  {
    TerlanNdArrayHandle destination =
        floats((double[]){0, 0, 0, 0}, 4, (int64_t[]){2, 2}, 2);
    assert(terlan_ndarray_v1_nanquantiles_into(
               destination, array, quantiles, 2, &axis_one, 1, 0, 0) == 0);
    assert_floats_with_nan(destination, (double[]){1, NAN, 3, NAN}, 4);
    assert(terlan_ndarray_v1_delete(destination) == 0);
    destination = floats((double[]){0, 0, 0, 0}, 4,
                         (int64_t[]){2, 2}, 2);
    assert(terlan_ndarray_v1_quantiles_into(
               destination, array, quantiles, 2, &axis_one, 1, 0, 0) == 0);
    assert_floats_with_nan(destination, (double[]){NAN, NAN, NAN, NAN}, 4);
    assert(terlan_ndarray_v1_delete(destination) == 0);
  }
  assert(terlan_ndarray_v1_nanpercentile_axes(array, 50, &axis_one, 1, 1, 0,
                                               &output) == 0);
  assert_shape(output, (int64_t[]){2, 1}, 2);
  assert_floats_with_nan(output, (double[]){2, NAN}, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_percentile(empty, 50, &axis_one, 1, 0, 0,
                                      &output) == 0);
  assert_shape(output, (int64_t[]){0}, 1);
  assert_floats(output, NULL, 0);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_percentile(empty, 50, &axis_zero, 1, 0, 0,
                                      &output) == TERLAN_NDARRAY_DOMAIN_ERROR);
  assert(output == NULL);
  assert(terlan_ndarray_v1_nanpercentile_axes(empty, 50, &axis_zero, 1, 0, 0,
                                               &output) == 0);
  assert_shape(output, (int64_t[]){3}, 1);
  assert_floats_with_nan(output, (double[]){NAN, NAN, NAN}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(empty) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_quantile_destination_writes_are_atomic(void) {
  const double values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  const double percentiles[] = {25, 50, 100};
  const int64_t axis = 1;
  TerlanNdArrayHandle array = floats(values, 12, (int64_t[]){3, 4}, 2);
  TerlanNdArrayHandle destination =
      floats((double[]){0, 0, 0, 0, 0, 0, 0, 0, 0}, 9,
             (int64_t[]){3, 3}, 2);
  TerlanNdArrayHandle wrong = floats((double[]){9, 9}, 2, (int64_t[]){2}, 1);
  assert(terlan_ndarray_v1_percentiles_into(
             destination, array, percentiles, 3, &axis, 1, 0, 0) == 0);
  assert_floats(destination,
                (double[]){0.75, 4.75, 8.75, 1.5, 5.5, 9.5, 3, 7, 11},
                9);
  assert(terlan_ndarray_v1_percentile_into(wrong, array, 50, &axis, 1, 0,
                                           0) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert_floats(wrong, (double[]){9, 9}, 2);
  assert(terlan_ndarray_v1_delete(wrong) == 0);
  assert(terlan_ndarray_v1_delete(destination) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_median_axes_empty_dtype_and_nan_semantics(void) {
  const double values[] = {0, 1, 2, 3, 4, 5};
  const int64_t axis_zero = 0;
  const int64_t axis_one = 1;
  const int64_t both_axes[] = {0, -1};
  TerlanNdArrayHandle array = floats(values, 6, (int64_t[]){2, 3}, 2);
  TerlanNdArrayHandle scalar = floats((double[]){7}, 1, NULL, 0);
  TerlanNdArrayHandle empty = floats(NULL, 0, (int64_t[]){0, 3}, 2);
  TerlanNdArrayHandle with_nan =
      floats((double[]){0, NAN, 2, 3, 4, 5}, 6, (int64_t[]){2, 3}, 2);
  TerlanNdArrayHandle output = NULL;

  assert(terlan_ndarray_v1_median(scalar, NULL, 0, 0, &output) == 0);
  assert_shape(output, NULL, 0);
  assert_floats(output, (double[]){7}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_nanmedian_axes(scalar, NULL, 0, 0, &output) == 0);
  assert_shape(output, NULL, 0);
  assert_floats(output, (double[]){7}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_median(array, NULL, 0, 0, &output) == 0);
  assert_floats(output, (double[]){2.5}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_median(array, &axis_zero, 1, 0, &output) == 0);
  assert_shape(output, (int64_t[]){3}, 1);
  assert_floats(output, (double[]){1.5, 2.5, 3.5}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_median(array, &axis_one, 1, 0, &output) == 0);
  assert_floats(output, (double[]){1, 4}, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_median(array, both_axes, 2, 1, &output) == 0);
  assert_shape(output, (int64_t[]){1, 1}, 2);
  assert_floats(output, (double[]){2.5}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_median(with_nan, &axis_zero, 1, 0, &output) == 0);
  assert_floats_with_nan(output, (double[]){1.5, NAN, 3.5}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_median(empty, &axis_zero, 1, 0, &output) == 0);
  assert_shape(output, (int64_t[]){3}, 1);
  assert_floats_with_nan(output, (double[]){NAN, NAN, NAN}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_nanmedian_axes(empty, &axis_zero, 1, 0, &output) ==
         0);
  assert_shape(output, (int64_t[]){3}, 1);
  assert_floats_with_nan(output, (double[]){NAN, NAN, NAN}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_median(empty, &axis_one, 1, 0, &output) == 0);
  assert_shape(output, (int64_t[]){0}, 1);
  assert_floats(output, NULL, 0);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_nanmedian_axes(empty, &axis_one, 1, 0, &output) ==
         0);
  assert_shape(output, (int64_t[]){0}, 1);
  assert_floats(output, NULL, 0);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert_floats(array, values, 6);

  {
    const int32_t floating_dtypes[] = {
        TERLAN_NDARRAY_DTYPE_FLOAT16, TERLAN_NDARRAY_DTYPE_FLOAT32,
        TERLAN_NDARRAY_DTYPE_FLOAT64};
    size_t index;
    for (index = 0; index < 3; ++index) {
      const int32_t dtype = floating_dtypes[index];
      TerlanNdArrayHandle typed = NULL;
      assert(terlan_ndarray_v1_from_floats_as(values, 6, (int64_t[]){6}, 1,
                                             dtype, &typed) == 0);
      assert(terlan_ndarray_v1_median(typed, NULL, 0, 0, &output) == 0);
      assert_dtype(output, dtype);
      assert_floats(output, (double[]){2.5}, 1);
      assert(terlan_ndarray_v1_delete(output) == 0);
      assert(terlan_ndarray_v1_nanmedian_axes(typed, NULL, 0, 0, &output) ==
             0);
      assert_dtype(output, dtype);
      assert_floats(output, (double[]){2.5}, 1);
      assert(terlan_ndarray_v1_delete(output) == 0);
      assert(terlan_ndarray_v1_delete(typed) == 0);
      typed = NULL;
      assert(terlan_ndarray_v1_from_floats_as((double[]){NAN}, 1,
                                             (int64_t[]){1}, 1, dtype,
                                             &typed) == 0);
      assert(terlan_ndarray_v1_nanmedian_axes(typed, NULL, 0, 0, &output) ==
             0);
      assert_dtype(output, dtype);
      assert_floats_with_nan(output, (double[]){NAN}, 1);
      assert(terlan_ndarray_v1_delete(output) == 0);
      assert(terlan_ndarray_v1_delete(typed) == 0);
    }
  }
  {
    TerlanNdArrayHandle typed = NULL;
    assert(terlan_ndarray_v1_from_complex(
               values, 6, (double[]){0, 0, 0, 0, 0, 0}, 6,
               (int64_t[]){6}, 1, TERLAN_NDARRAY_DTYPE_COMPLEX64, &typed) ==
           0);
    assert(terlan_ndarray_v1_median(typed, NULL, 0, 0, &output) == 0);
    assert_dtype(output, TERLAN_NDARRAY_DTYPE_COMPLEX64);
    assert_complex_reals(output, (double[]){2.5}, 1);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_nanmedian_axes(typed, NULL, 0, 0, &output) == 0);
    assert_dtype(output, TERLAN_NDARRAY_DTYPE_COMPLEX64);
    assert_complex_reals(output, (double[]){2.5}, 1);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_delete(typed) == 0);
  }
  {
    TerlanNdArrayHandle integers = NULL;
    assert(terlan_ndarray_v1_from_ints((int64_t[]){0, 1, 2, 3}, 4,
                                      (int64_t[]){4}, 1, &integers) == 0);
    assert(terlan_ndarray_v1_median(integers, NULL, 0, 0, &output) == 0);
    assert_dtype(output, TERLAN_NDARRAY_DTYPE_FLOAT64);
    assert_floats(output, (double[]){1.5}, 1);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_delete(integers) == 0);
  }
  assert(terlan_ndarray_v1_delete(with_nan) == 0);
  assert(terlan_ndarray_v1_delete(empty) == 0);
  assert(terlan_ndarray_v1_delete(scalar) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_nanmedian_large_infinity_and_destination_semantics(void) {
  const int64_t axis = 1;
  double large_values[802];
  TerlanNdArrayHandle array = floats(
      (double[]){NAN, 1, 3, NAN, NAN, NAN, NAN, NAN, INFINITY, INFINITY,
                 NAN, NAN, -INFINITY, -INFINITY, NAN, NAN, INFINITY, 7,
                 -INFINITY, -9},
      20, (int64_t[]){5, 4}, 2);
  TerlanNdArrayHandle output = NULL;
  TerlanNdArrayHandle destination =
      floats((double[]){0, 0, 0, 0, 0}, 5, (int64_t[]){5}, 1);
  TerlanNdArrayHandle wrong =
      floats((double[]){9, 9, 9, 9}, 4, (int64_t[]){4}, 1);

  assert(terlan_ndarray_v1_nanmedian_axes(array, &axis, 1, 1, &output) == 0);
  assert_shape(output, (int64_t[]){5, 1}, 2);
  assert_floats_with_nan(output,
                         (double[]){2, NAN, INFINITY, -INFINITY, -1}, 5);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_nanmedian_into(destination, array, &axis, 1, 0) ==
         0);
  assert_floats_with_nan(destination,
                         (double[]){2, NAN, INFINITY, -INFINITY, -1}, 5);
  assert(terlan_ndarray_v1_nanmedian_into(wrong, array, &axis, 1, 0) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert_floats(wrong, (double[]){9, 9, 9, 9}, 4);
  assert(terlan_ndarray_v1_median_into(destination, array, &axis, 1, 0) == 0);
  assert_floats_with_nan(destination,
                         (double[]){NAN, NAN, NAN, NAN, -1}, 5);

  for (int64_t index = 0; index < 401; ++index) {
    large_values[index] = (double)index;
    large_values[401 + index] = NAN;
  }
  large_values[401] = 7;
  {
    TerlanNdArrayHandle large =
        floats(large_values, 802, (int64_t[]){2, 401}, 2);
    assert(terlan_ndarray_v1_nanmedian_axes(large, &axis, 1, 0, &output) == 0);
    assert_floats(output, (double[]){200, 7}, 2);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert_floats_with_nan(large, large_values, 802);
    assert(terlan_ndarray_v1_delete(large) == 0);
  }
  assert(terlan_ndarray_v1_nanmedian_axes(array, (int64_t[]){-3}, 1, 0,
                                         &output) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(output == NULL);
  assert(terlan_ndarray_v1_nanmedian_axes(array, (int64_t[]){1, 1}, 2, 0,
                                         &output) ==
         TERLAN_NDARRAY_DUPLICATE_AXIS);
  assert(output == NULL);
  assert(terlan_ndarray_v1_delete(wrong) == 0);
  assert(terlan_ndarray_v1_delete(destination) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_weighted_covariance_and_correlation(void) {
  const double values[] = {1.0, 2.0, 3.0, 2.0, 4.0, 6.0};
  const double weight_values[] = {1.0, 1.0, 2.0};
  const int64_t shape[] = {2, 3};
  TerlanNdArrayHandle array = floats(values, 6, shape, 2);
  TerlanNdArrayHandle weights = floats(weight_values, 3, (int64_t[]){3}, 1);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_weighted_average(array, weights, 1, 0, &output) ==
         0);
  assert_floats(output, (double[]){2.25, 4.5}, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_covariance(array, 1, 1, &output) == 0);
  assert_floats(output, (double[]){1.0, 2.0, 2.0, 4.0}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_correlation(array, 1, &output) == 0);
  assert_floats(output, (double[]){1.0, 1.0, 1.0, 1.0}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(weights) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_covariance_numpy_pair_weight_dtype_and_empty_semantics(void) {
  TerlanNdArrayHandle x = floats((double[]){0, 1, 2, 2, 1, 0}, 6,
                                 (int64_t[]){2, 3}, 2);
  TerlanNdArrayHandle y = complex_values(
      (double[]){0, 0, 0}, (double[]){1, 2, 3}, 3,
      (int64_t[]){1, 3}, 2, TERLAN_NDARRAY_DTYPE_COMPLEX128);
  TerlanNdArrayHandle frequency = floats((double[]){1, 4, 1}, 3,
                                         (int64_t[]){3}, 1);
  TerlanNdArrayHandle analytic = floats((double[]){1, 4, 1}, 3,
                                        (int64_t[]){3}, 1);
  TerlanNdArrayHandle unit = floats((double[]){1, 1, 1}, 3,
                                    (int64_t[]){3}, 1);
  TerlanNdArrayHandle float32 = floats_as(
      (double[]){0, 1, 2, 2, 1, 0}, 6, (int64_t[]){2, 3}, 2,
      TERLAN_NDARRAY_DTYPE_FLOAT32);
  TerlanNdArrayHandle vector = floats(
      (double[]){0.3942, 0.5969, 0.7730, 0.9918, 0.7964}, 5,
      (int64_t[]){5}, 1);
  TerlanNdArrayHandle empty_vector = floats(NULL, 0, (int64_t[]){0}, 1);
  TerlanNdArrayHandle empty_variables =
      floats(NULL, 0, (int64_t[]){0, 2}, 2);
  TerlanNdArrayHandle empty_observations =
      floats(NULL, 0, (int64_t[]){2, 0}, 2);
  TerlanNdArrayHandle output = NULL;

  assert(terlan_ndarray_v1_covariance_pair(
             x, y, 1, 1, TERLAN_NDARRAY_DTYPE_COMPLEX128, &output) == 0);
  assert_shape(output, (int64_t[]){3, 3}, 2);
  assert_complex_values(output, (double[]){1, -1, 0, -1, 1, 0, 0, 0, 1},
                        (double[]){0, 0, -1, 0, 0, 1, 1, -1, 0}, 9);
  assert(terlan_ndarray_v1_delete(output) == 0);

  assert(terlan_ndarray_v1_covariance(vector, 0, 1, &output) == 0);
  assert_shape(output, NULL, 0);
  assert_floats(output, (double[]){0.050841348}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_covariance(x, 1, 5, &output) == 0);
  assert_floats_with_nan(output,
                         (double[]){INFINITY, -INFINITY, -INFINITY, INFINITY},
                         4);
  assert(terlan_ndarray_v1_delete(output) == 0);

  assert(terlan_ndarray_v1_covariance_frequency_weighted(
             x, frequency, 1, 1, 0, &output) == 0);
  assert_floats(output, (double[]){0.4, -0.4, -0.4, 0.4}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_covariance_analytic_weighted(
             x, analytic, 1, 1, 0, &output) == 0);
  assert_floats(output,
                (double[]){2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0,
                           2.0 / 3.0},
                4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_covariance_weighted(
             x, unit, analytic, 1, 1, 0, &output) == 0);
  assert_floats(output,
                (double[]){2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0,
                           2.0 / 3.0},
                4);
  assert(terlan_ndarray_v1_delete(output) == 0);

  assert(terlan_ndarray_v1_covariance_typed(
             float32, 1, 1, TERLAN_NDARRAY_DTYPE_FLOAT32, &output) == 0);
  assert_dtype(output, TERLAN_NDARRAY_DTYPE_FLOAT32);
  assert_floats(output, (double[]){1, -1, -1, 1}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);

  assert(terlan_ndarray_v1_covariance(empty_vector, 1, 1, &output) == 0);
  assert_shape(output, NULL, 0);
  assert_floats_with_nan(output, (double[]){NAN}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_covariance(empty_variables, 1, 1, &output) == 0);
  assert_shape(output, (int64_t[]){0, 0}, 2);
  assert_floats(output, NULL, 0);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_covariance(empty_observations, 1, 1, &output) ==
         0);
  assert_shape(output, (int64_t[]){2, 2}, 2);
  assert_floats_with_nan(output, (double[]){NAN, NAN, NAN, NAN}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);

  assert_floats(x, (double[]){0, 1, 2, 2, 1, 0}, 6);
  assert(terlan_ndarray_v1_delete(empty_observations) == 0);
  assert(terlan_ndarray_v1_delete(empty_variables) == 0);
  assert(terlan_ndarray_v1_delete(empty_vector) == 0);
  assert(terlan_ndarray_v1_delete(vector) == 0);
  assert(terlan_ndarray_v1_delete(float32) == 0);
  assert(terlan_ndarray_v1_delete(unit) == 0);
  assert(terlan_ndarray_v1_delete(analytic) == 0);
  assert(terlan_ndarray_v1_delete(frequency) == 0);
  assert(terlan_ndarray_v1_delete(y) == 0);
  assert(terlan_ndarray_v1_delete(x) == 0);
}

static void test_correlation_numpy_pair_complex_degenerate_and_extreme(void) {
  TerlanNdArrayHandle x = complex_values(
      (double[]){1, 2, 3, 0, 0, 0}, (double[]){0, 0, 0, 1, 2, 3}, 6,
      (int64_t[]){2, 3}, 2, TERLAN_NDARRAY_DTYPE_COMPLEX128);
  TerlanNdArrayHandle first = floats((double[]){1, 2, 3}, 3,
                                     (int64_t[]){1, 3}, 2);
  TerlanNdArrayHandle second = complex_values(
      (double[]){0, 0, 0}, (double[]){1, 2, 3}, 3,
      (int64_t[]){1, 3}, 2, TERLAN_NDARRAY_DTYPE_COMPLEX128);
  TerlanNdArrayHandle constant = floats((double[]){7, 7, 7}, 3,
                                        (int64_t[]){3}, 1);
  TerlanNdArrayHandle extreme = floats(
      (double[]){1e-100, 1e100, 1e100, 1e-100}, 4,
      (int64_t[]){2, 2}, 2);
  TerlanNdArrayHandle output = NULL;

  assert(terlan_ndarray_v1_correlation(x, 1, &output) == 0);
  assert_complex_values(output, (double[]){1, 0, 0, 1},
                        (double[]){0, -1, 1, 0}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_correlation_pair(
             first, second, 1, TERLAN_NDARRAY_DTYPE_COMPLEX64, &output) == 0);
  assert_dtype(output, TERLAN_NDARRAY_DTYPE_COMPLEX64);
  assert_complex_values(output, (double[]){1, 0, 0, 1},
                        (double[]){0, -1, 1, 0}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_correlation(constant, 1, &output) == 0);
  assert_floats_with_nan(output, (double[]){NAN}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_correlation(extreme, 1, &output) == 0);
  assert_floats(output, (double[]){1, -1, -1, 1}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);

  assert(terlan_ndarray_v1_delete(extreme) == 0);
  assert(terlan_ndarray_v1_delete(constant) == 0);
  assert(terlan_ndarray_v1_delete(second) == 0);
  assert(terlan_ndarray_v1_delete(first) == 0);
  assert(terlan_ndarray_v1_delete(x) == 0);
}

static void test_covariance_rejects_invalid_weights_and_shapes(void) {
  TerlanNdArrayHandle array = floats((double[]){0, 1, 2, 2, 1, 0}, 6,
                                     (int64_t[]){2, 3}, 2);
  TerlanNdArrayHandle non_integer = floats((double[]){1.5, 2, 3}, 3,
                                           (int64_t[]){3}, 1);
  TerlanNdArrayHandle negative = floats((double[]){1, -1, 1}, 3,
                                        (int64_t[]){3}, 1);
  TerlanNdArrayHandle wrong = floats((double[]){1, 1}, 2,
                                     (int64_t[]){2}, 1);
  TerlanNdArrayHandle matrix = floats((double[]){1, 1, 1, 1, 1, 1}, 6,
                                      (int64_t[]){2, 3}, 2);
  TerlanNdArrayHandle zeros = floats((double[]){0, 0, 0}, 3,
                                     (int64_t[]){3}, 1);
  TerlanNdArrayHandle other = floats((double[]){1, 2}, 2,
                                     (int64_t[]){2}, 1);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_covariance_frequency_weighted(
             array, non_integer, 1, 1, 0, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_covariance_frequency_weighted(
             array, matrix, 1, 1, 0, &output) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(terlan_ndarray_v1_covariance_frequency_weighted(
             array, wrong, 1, 1, 0, &output) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(terlan_ndarray_v1_covariance_analytic_weighted(
             array, negative, 1, 1, 0, &output) ==
         TERLAN_NDARRAY_DOMAIN_ERROR);
  assert(terlan_ndarray_v1_covariance_analytic_weighted(
             array, zeros, 1, 1, 0, &output) ==
         TERLAN_NDARRAY_DIVIDE_BY_ZERO);
  assert(terlan_ndarray_v1_covariance_pair(array, other, 1, 1, 0, &output) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(terlan_ndarray_v1_covariance_typed(
             array, 1, 1, TERLAN_NDARRAY_DTYPE_INT64, &output) ==
         TERLAN_NDARRAY_UNSUPPORTED_DTYPE);
  assert(output == NULL);
  assert(terlan_ndarray_v1_delete(other) == 0);
  assert(terlan_ndarray_v1_delete(zeros) == 0);
  assert(terlan_ndarray_v1_delete(matrix) == 0);
  assert(terlan_ndarray_v1_delete(wrong) == 0);
  assert(terlan_ndarray_v1_delete(negative) == 0);
  assert(terlan_ndarray_v1_delete(non_integer) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_average_weights_returned_dtype_and_axes(void) {
  const double flat_values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const double matrix_values[] = {1, 2, 3, 4, 5, 6};
  const int64_t axis_zero = 0;
  const int64_t axis_one = 1;
  const int64_t cube_axes[] = {0, 2};
  TerlanNdArrayHandle flat =
      floats(flat_values, 10, (int64_t[]){10}, 1);
  TerlanNdArrayHandle flat_weights =
      floats(flat_values, 10, (int64_t[]){10}, 1);
  TerlanNdArrayHandle matrix =
      floats(matrix_values, 6, (int64_t[]){2, 3}, 2);
  TerlanNdArrayHandle row_weights =
      floats((double[]){1, 2}, 2, (int64_t[]){2}, 1);
  TerlanNdArrayHandle column_weights =
      floats((double[]){0, 0, 1}, 3, (int64_t[]){3}, 1);
  TerlanNdArrayHandle matrix_weights =
      floats((double[]){0, 0, 1, 0, 0, 2}, 6, (int64_t[]){2, 3}, 2);
  TerlanNdArrayHandle output = NULL;

  assert(terlan_ndarray_v1_weighted_average_axes(
             flat, flat_weights, NULL, 0, 0, &output) == 0);
  assert_floats(output, (double[]){285.0 / 45.0}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_weighted_average_axes(
             matrix, row_weights, &axis_zero, 1, 0, &output) == 0);
  assert_floats(output, (double[]){3, 4, 5}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_weighted_average_axes(
             matrix, column_weights, &axis_one, 1, 0, &output) == 0);
  assert_floats(output, (double[]){3, 6}, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_weighted_average_axes(
             matrix, matrix_weights, NULL, 0, 0, &output) == 0);
  assert_floats(output, (double[]){5}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_weighted_average_weight_sum(
             matrix, matrix_weights, NULL, 0, 0, &output) == 0);
  assert_floats(output, (double[]){3}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_average_weight_sum(
             matrix, &axis_zero, 1, 0, &output) == 0);
  assert_floats(output, (double[]){2, 2, 2}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  {
    TerlanNdArrayHandle cube = floats(
        (double[]){0, 1, 2, 3, 4, 5, 6, 7}, 8,
        (int64_t[]){2, 2, 2}, 3);
    TerlanNdArrayHandle weights = floats(
        (double[]){1, 1, 1, 1, 1, 1, 1, 1}, 8,
        (int64_t[]){2, 2, 2}, 3);
    assert(terlan_ndarray_v1_weighted_average_axes(
               cube, weights, cube_axes, 2, 1, &output) == 0);
    assert_shape(output, (int64_t[]){1, 2, 1}, 3);
    assert_floats(output, (double[]){2.5, 4.5}, 2);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_weighted_average_weight_sum(
               cube, weights, cube_axes, 2, 1, &output) == 0);
    assert_floats(output, (double[]){4, 4}, 2);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_delete(weights) == 0);
    assert(terlan_ndarray_v1_delete(cube) == 0);
  }
  {
    const int32_t source_dtypes[] = {TERLAN_NDARRAY_DTYPE_FLOAT32,
                                     TERLAN_NDARRAY_DTYPE_FLOAT32};
    const int32_t weight_dtypes[] = {TERLAN_NDARRAY_DTYPE_FLOAT32,
                                     TERLAN_NDARRAY_DTYPE_FLOAT64};
    const int32_t expected_dtypes[] = {TERLAN_NDARRAY_DTYPE_FLOAT32,
                                       TERLAN_NDARRAY_DTYPE_FLOAT64};
    size_t index;
    for (index = 0; index < 2; ++index) {
      TerlanNdArrayHandle source = NULL;
      TerlanNdArrayHandle weights = NULL;
      assert(terlan_ndarray_v1_from_floats_as(
                 (double[]){1, 2}, 2, (int64_t[]){2}, 1,
                 source_dtypes[index], &source) == 0);
      assert(terlan_ndarray_v1_from_floats_as(
                 (double[]){1, 2}, 2, (int64_t[]){2}, 1,
                 weight_dtypes[index], &weights) == 0);
      assert(terlan_ndarray_v1_weighted_average_axes(
                 source, weights, NULL, 0, 0, &output) == 0);
      assert_dtype(output, expected_dtypes[index]);
      assert(terlan_ndarray_v1_delete(output) == 0);
      assert(terlan_ndarray_v1_delete(weights) == 0);
      assert(terlan_ndarray_v1_delete(source) == 0);
    }
  }
  {
    TerlanNdArrayHandle source = NULL;
    TerlanNdArrayHandle weights = NULL;
    assert(terlan_ndarray_v1_from_ints((int64_t[]){1, 2}, 2,
                                      (int64_t[]){2}, 1, &source) == 0);
    assert(terlan_ndarray_v1_from_floats_as(
               (double[]){1, 2}, 2, (int64_t[]){2}, 1,
               TERLAN_NDARRAY_DTYPE_FLOAT32, &weights) == 0);
    assert(terlan_ndarray_v1_weighted_average_axes(
               source, weights, NULL, 0, 0, &output) == 0);
    assert_dtype(output, TERLAN_NDARRAY_DTYPE_FLOAT64);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_delete(weights) == 0);
    assert(terlan_ndarray_v1_delete(source) == 0);
  }
  {
    TerlanNdArrayHandle source = NULL;
    TerlanNdArrayHandle weights = NULL;
    assert(terlan_ndarray_v1_from_floats_as(
               (double[]){1, 2}, 2, (int64_t[]){2}, 1,
               TERLAN_NDARRAY_DTYPE_FLOAT32, &source) == 0);
    assert(terlan_ndarray_v1_from_complex(
               (double[]){1, 2}, 2, (double[]){0, 1}, 2,
               (int64_t[]){2}, 1, TERLAN_NDARRAY_DTYPE_COMPLEX64,
               &weights) == 0);
    assert(terlan_ndarray_v1_weighted_average_axes(
               source, weights, NULL, 0, 0, &output) == 0);
    assert_dtype(output, TERLAN_NDARRAY_DTYPE_COMPLEX64);
    assert_complex_values(output, (double[]){1.7}, (double[]){0.1}, 1);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_weighted_average_weight_sum(
               source, weights, NULL, 0, 0, &output) == 0);
    assert_complex_values(output, (double[]){3}, (double[]){1}, 1);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_delete(weights) == 0);
    assert(terlan_ndarray_v1_delete(source) == 0);
  }
  {
    TerlanNdArrayHandle zero =
        floats((double[]){0, 0, 0}, 3, (int64_t[]){3}, 1);
    TerlanNdArrayHandle wrong =
        floats((double[]){1, 2, 3, 4}, 4, (int64_t[]){2, 2}, 2);
    assert(terlan_ndarray_v1_weighted_average_axes(
               matrix, zero, &axis_one, 1, 0, &output) ==
           TERLAN_NDARRAY_DOMAIN_ERROR);
    assert(output == NULL);
    assert(terlan_ndarray_v1_weighted_average_axes(
               matrix, wrong, &axis_one, 1, 0, &output) ==
           TERLAN_NDARRAY_SHAPE_MISMATCH);
    assert(output == NULL);
    assert(terlan_ndarray_v1_weighted_average_axes(
               matrix, column_weights, (int64_t[]){-3}, 1, 0, &output) ==
           TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
    assert(output == NULL);
    assert(terlan_ndarray_v1_delete(wrong) == 0);
    assert(terlan_ndarray_v1_delete(zero) == 0);
  }
  assert_floats(matrix, matrix_values, 6);
  assert(terlan_ndarray_v1_delete(matrix_weights) == 0);
  assert(terlan_ndarray_v1_delete(column_weights) == 0);
  assert(terlan_ndarray_v1_delete(row_weights) == 0);
  assert(terlan_ndarray_v1_delete(matrix) == 0);
  assert(terlan_ndarray_v1_delete(flat_weights) == 0);
  assert(terlan_ndarray_v1_delete(flat) == 0);
}

static void test_nan_reductions_skip_non_finite_missing_values(void) {
  const double values[] = {1.0, NAN, 3.0, NAN};
  const int64_t shape[] = {2, 2};
  const int64_t axis = 1;
  TerlanNdArrayHandle array = floats(values, 4, shape, 2);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_nan_sum(array, &axis, 1, 0, &output) == 0);
  assert_floats(output, (double[]){1.0, 3.0}, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_nan_product(array, &axis, 1, 0, &output) == 0);
  assert_floats(output, (double[]){1.0, 3.0}, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_nan_mean(array, &axis, 1, 0, &output) == 0);
  assert_floats(output, (double[]){1.0, 3.0}, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_nan_minimum(array, &axis, 1, 0, &output) == 0);
  assert_floats(output, (double[]){1.0, 3.0}, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_nan_maximum(array, &axis, 1, 0, &output) == 0);
  assert_floats(output, (double[]){1.0, 3.0}, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_statistics_rejections_and_allocation_cleanup(void) {
  const double values[] = {1.0, 2.0};
  const double zeros[] = {0.0, 0.0};
  TerlanNdArrayHandle array = floats(values, 2, (int64_t[]){2}, 1);
  TerlanNdArrayHandle weights = floats(zeros, 2, (int64_t[]){2}, 1);
  TerlanNdArrayHandle output = (TerlanNdArrayHandle)1;
  TerlanNdArrayHandle empty = floats(NULL, 0, (int64_t[]){0}, 1);
  TerlanNdArrayHandle non_finite =
      floats((double[]){1.0, NAN}, 2, (int64_t[]){2}, 1);
  int64_t failure;
  assert(terlan_ndarray_v1_quantile(array, -0.1, NULL, 0, 0, 0, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_quantile(array, 0.5, NULL, 0, 0, 9, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_quantiles(array, (double[]){0.5, NAN}, 2, NULL, 0,
                                     0, 0, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(output == NULL);
  assert(terlan_ndarray_v1_percentiles(array, (double[]){50, 101}, 2, NULL, 0,
                                       0, 0, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(output == NULL);
  assert(terlan_ndarray_v1_percentile(array, 50, (int64_t[]){-2}, 1, 0, 0,
                                      &output) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(output == NULL);
  assert(terlan_ndarray_v1_nanpercentile_axes(
             array, 50, (int64_t[]){0, 0}, 2, 0, 0, &output) ==
         TERLAN_NDARRAY_DUPLICATE_AXIS);
  assert(output == NULL);
  assert(terlan_ndarray_v1_weighted_average(array, weights, 0, 0, &output) ==
         TERLAN_NDARRAY_DOMAIN_ERROR);
  assert(output == NULL);
  assert(terlan_ndarray_v1_median(empty, NULL, 0, 0, &output) == 0);
  assert_floats_with_nan(output, (double[]){NAN}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_covariance(non_finite, 1, 1, &output) == 0);
  assert_floats_with_nan(output, (double[]){NAN}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  for (failure = 0; failure < 8; ++failure) {
    terlan_ndarray_test_fail_allocation_after(failure);
    output = NULL;
    if (terlan_ndarray_v1_median(array, NULL, 0, 0, &output) == 0) {
      assert(terlan_ndarray_v1_delete(output) == 0);
    } else {
      assert(output == NULL);
    }
    terlan_ndarray_test_reset_allocation_failures();
  }
  assert(terlan_ndarray_v1_delete(non_finite) == 0);
  assert(terlan_ndarray_v1_delete(empty) == 0);
  assert(terlan_ndarray_v1_delete(weights) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

int main(void) {
  test_quantiles_and_range_across_axes();
  test_quantile_sequences_axes_shapes_and_interpolation();
  test_quantile_sequences_are_monotonic();
  test_quantile_nan_and_empty_dimension_semantics();
  test_quantile_destination_writes_are_atomic();
  test_median_axes_empty_dtype_and_nan_semantics();
  test_nanmedian_large_infinity_and_destination_semantics();
  test_weighted_covariance_and_correlation();
  test_covariance_numpy_pair_weight_dtype_and_empty_semantics();
  test_correlation_numpy_pair_complex_degenerate_and_extreme();
  test_covariance_rejects_invalid_weights_and_shapes();
  test_average_weights_returned_dtype_and_axes();
  test_nan_reductions_skip_non_finite_missing_values();
  test_statistics_rejections_and_allocation_cleanup();
  return 0;
}
