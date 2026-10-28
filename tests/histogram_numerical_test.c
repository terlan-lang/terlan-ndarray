#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static TerlanNdArrayHandle ints(const int64_t *values, int64_t count,
                                const int64_t *shape, int64_t rank) {
  TerlanNdArrayHandle array = NULL;
  assert(terlan_ndarray_v1_from_ints(values, count, shape, rank, &array) == 0);
  return array;
}

static TerlanNdArrayHandle floats(const double *values, int64_t count,
                                  const int64_t *shape, int64_t rank) {
  TerlanNdArrayHandle array = NULL;
  assert(terlan_ndarray_v1_from_floats(values, count, shape, rank, &array) ==
         0);
  return array;
}

static TerlanNdArrayHandle bools(const uint8_t *values, int64_t count,
                                 const int64_t *shape, int64_t rank) {
  TerlanNdArrayHandle array = NULL;
  assert(terlan_ndarray_v1_from_bools(values, count, shape, rank, &array) ==
         0);
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

static void assert_bools(TerlanNdArrayHandle array, const uint8_t *expected,
                         size_t count) {
  uint8_t *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_bools(array, &values, &length) == 0);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(values[index] == expected[index]);
  }
  terlan_ndarray_v1_delete_bool_values(values);
}

static void test_histograms_bincount_and_digitize(void) {
  const double values[] = {-1.0, 0.0, 0.5, 1.0, 2.0, NAN};
  TerlanNdArrayHandle array = floats(values, 6, (int64_t[]){6}, 1);
  TerlanNdArrayHandle output = NULL;
  TerlanNdArrayHandle counts = NULL;
  TerlanNdArrayHandle weights = NULL;
  TerlanNdArrayHandle bins = NULL;
  assert(terlan_ndarray_v1_histogram(array, 2, 0.0, 1.0, &output) == 0);
  assert_ints(output, (int64_t[]){1, 2}, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_histogram_bin_edges(array, 2, 0.0, 1.0,
                                               &output) == 0);
  assert_floats(output, (double[]){0.0, 0.5, 1.0}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  counts = ints((int64_t[]){0, 1, 1, 3}, 4, (int64_t[]){4}, 1);
  weights = floats((double[]){1.0, 0.5, 1.5, 2.0}, 4,
                   (int64_t[]){4}, 1);
  assert(terlan_ndarray_v1_bincount(counts, weights, 5, &output) == 0);
  assert_floats(output, (double[]){1.0, 2.0, 0.0, 2.0, 0.0}, 5);
  assert(terlan_ndarray_v1_delete(output) == 0);
  bins = floats((double[]){0.0, 1.0, 2.0}, 3, (int64_t[]){3}, 1);
  assert(terlan_ndarray_v1_digitize(array, bins, 0, &output) == 0);
  assert_ints(output, (int64_t[]){0, 1, 1, 2, 3, 3}, 6);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(bins) == 0);
  assert(terlan_ndarray_v1_delete(weights) == 0);
  assert(terlan_ndarray_v1_delete(counts) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_multidimensional_histogram(void) {
  const double samples[] = {0.1, 0.1, 0.9, 0.9, 0.2, 0.8, 2.0, 2.0};
  TerlanNdArrayHandle array = floats(samples, 8, (int64_t[]){4, 2}, 2);
  TerlanNdArrayHandle minima =
      floats((double[]){0.0, 0.0}, 2, (int64_t[]){2}, 1);
  TerlanNdArrayHandle maxima =
      floats((double[]){1.0, 1.0}, 2, (int64_t[]){2}, 1);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_histogramdd(array, 2, minima, maxima, &output) == 0);
  assert_ints(output, (int64_t[]){1, 1, 0, 1}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(maxima) == 0);
  assert(terlan_ndarray_v1_delete(minima) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_difference_gradient_unwrap_and_interpolation(void) {
  const double values[] = {0.0, 1.0, 4.0, 9.0};
  TerlanNdArrayHandle array = floats(values, 4, (int64_t[]){4}, 1);
  TerlanNdArrayHandle output = NULL;
  TerlanNdArrayHandle coordinates = NULL;
  TerlanNdArrayHandle samples = NULL;
  TerlanNdArrayHandle queries = NULL;
  TerlanNdArrayHandle angles = NULL;
  assert(terlan_ndarray_v1_difference(array, 2, 0, &output) == 0);
  assert_floats(output, (double[]){2.0, 2.0}, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_gradient(array, 1.0, 0, &output) == 0);
  assert_floats(output, (double[]){1.0, 2.0, 4.0, 5.0}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  angles = floats((double[]){0.0, 3.0, -3.0, -2.0}, 4,
                  (int64_t[]){4}, 1);
  assert(terlan_ndarray_v1_unwrap(angles, 2.0 * acos(-1.0), 0, &output) == 0);
  assert_floats(output, (double[]){0.0, 3.0, 3.283185307179586,
                                   4.283185307179586},
                4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(angles) == 0);
  coordinates = floats((double[]){0.0, 1.0, 2.0}, 3, (int64_t[]){3}, 1);
  samples = floats((double[]){0.0, 10.0, 20.0}, 3, (int64_t[]){3}, 1);
  queries = floats((double[]){-1.0, 0.5, 2.0, 3.0}, 4,
                   (int64_t[]){2, 2}, 2);
  assert(terlan_ndarray_v1_interpolate(queries, coordinates, samples, -5.0,
                                      25.0, &output) == 0);
  assert_floats(output, (double[]){-5.0, 5.0, 20.0, 25.0}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(queries) == 0);
  assert(terlan_ndarray_v1_delete(samples) == 0);
  assert(terlan_ndarray_v1_delete(coordinates) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_difference_numpy_orders_axes_boolean_and_boundaries(void) {
  TerlanNdArrayHandle vector = ints((int64_t[]){1, 4, 6, 7, 12}, 5,
                                    (int64_t[]){5}, 1);
  TerlanNdArrayHandle truth = bools(
      (uint8_t[]){1, 1, 0, 0}, 4, (int64_t[]){4}, 1);
  TerlanNdArrayHandle matrix = ints((int64_t[]){0, 1, 2, 3}, 4,
                                    (int64_t[]){2, 2}, 2);
  TerlanNdArrayHandle cube = ints((int64_t[]){0, 1, 2, 3, 4, 5, 6, 7}, 8,
                                  (int64_t[]){2, 2, 2}, 3);
  TerlanNdArrayHandle prepend = ints((int64_t[]){0}, 1,
                                     (int64_t[]){1}, 1);
  TerlanNdArrayHandle append = ints((int64_t[]){0, 2}, 2,
                                    (int64_t[]){2}, 1);
  TerlanNdArrayHandle matrix_prepend = ints((int64_t[]){0, 0}, 2,
                                            (int64_t[]){2, 1}, 2);
  TerlanNdArrayHandle matrix_append = ints((int64_t[]){0, 0}, 2,
                                           (int64_t[]){1, 2}, 2);
  TerlanNdArrayHandle wrong = ints((int64_t[]){0, 0, 0}, 3,
                                   (int64_t[]){3, 1}, 2);
  TerlanNdArrayHandle output = NULL;

  assert(terlan_ndarray_v1_difference(vector, 1, -1, &output) == 0);
  assert_ints(output, (int64_t[]){3, 2, 1, 5}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_difference(vector, 3, 0, &output) == 0);
  assert_ints(output, (int64_t[]){0, 5}, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_difference(vector, 7, 0, &output) == 0);
  assert_ints(output, NULL, 0);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_difference(truth, 1, 0, &output) == 0);
  assert_bools(output, (uint8_t[]){0, 1, 0}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_difference(truth, 2, 0, &output) == 0);
  assert_bools(output, (uint8_t[]){1, 1}, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_difference(cube, 1, 0, &output) == 0);
  assert_ints(output, (int64_t[]){4, 4, 4, 4}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_difference(cube, 1, -1, &output) == 0);
  assert_ints(output, (int64_t[]){1, 1, 1, 1}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);

  assert(terlan_ndarray_v1_difference_prepend(vector, prepend, 1, 0,
                                               &output) == 0);
  assert_ints(output, (int64_t[]){1, 3, 2, 1, 5}, 5);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_difference_append(vector, append, 1, 0, &output) ==
         0);
  assert_ints(output, (int64_t[]){3, 2, 1, 5, -12, 2}, 6);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_difference_surround(
             vector, prepend, append, 1, 0, &output) == 0);
  assert_ints(output, (int64_t[]){1, 3, 2, 1, 5, -12, 2}, 7);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_difference_prepend(
             matrix, matrix_prepend, 1, 1, &output) == 0);
  assert_ints(output, (int64_t[]){0, 1, 2, 1}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_difference_append(
             matrix, matrix_append, 1, 0, &output) == 0);
  assert_ints(output, (int64_t[]){2, 2, -2, -3}, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_difference_prepend(matrix, wrong, 1, 1,
                                               &output) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(terlan_ndarray_v1_difference(matrix, 1, 2, &output) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(output == NULL);
  assert_ints(vector, (int64_t[]){1, 4, 6, 7, 12}, 5);

  assert(terlan_ndarray_v1_delete(wrong) == 0);
  assert(terlan_ndarray_v1_delete(matrix_append) == 0);
  assert(terlan_ndarray_v1_delete(matrix_prepend) == 0);
  assert(terlan_ndarray_v1_delete(append) == 0);
  assert(terlan_ndarray_v1_delete(prepend) == 0);
  assert(terlan_ndarray_v1_delete(matrix) == 0);
  assert(terlan_ndarray_v1_delete(cube) == 0);
  assert(terlan_ndarray_v1_delete(truth) == 0);
  assert(terlan_ndarray_v1_delete(vector) == 0);
}

static void test_trapezoids_and_rejections(void) {
  TerlanNdArrayHandle array =
      floats((double[]){0.0, 1.0, 4.0}, 3, (int64_t[]){3}, 1);
  TerlanNdArrayHandle coordinates =
      floats((double[]){0.0, 1.0, 3.0}, 3, (int64_t[]){3}, 1);
  TerlanNdArrayHandle output = NULL;
  TerlanNdArrayHandle duplicates =
      floats((double[]){0.0, 1.0, 1.0}, 3, (int64_t[]){3}, 1);
  TerlanNdArrayHandle queries =
      floats((double[]){0.5}, 1, (int64_t[]){1}, 1);
  assert(terlan_ndarray_v1_trapezoidal_integral_uniform(array, 1.0, 0,
                                                        &output) == 0);
  assert_floats(output, (double[]){3.0}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_trapezoidal_integral(array, coordinates, 0,
                                                &output) == 0);
  assert_floats(output, (double[]){5.5}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_histogram(array, 0, 0.0, 1.0, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_histogram(array, 2, 1.0, 1.0, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_gradient(array, 0.0, 0, &output) == 0);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_interpolate(queries, duplicates, array, -1.0, 1.0,
                                      &output) == 0);
  assert_floats(output, (double[]){0.5}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(queries) == 0);
  assert(terlan_ndarray_v1_delete(duplicates) == 0);
  assert(terlan_ndarray_v1_delete(coordinates) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

int main(void) {
  test_histograms_bincount_and_digitize();
  test_multidimensional_histogram();
  test_difference_gradient_unwrap_and_interpolation();
  test_difference_numpy_orders_axes_boolean_and_boundaries();
  test_trapezoids_and_rejections();
  return 0;
}
