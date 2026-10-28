#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>

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

static TerlanNdArrayHandle integers(const int64_t *values, int64_t count,
                                    const int64_t *shape, int64_t rank) {
  TerlanNdArrayHandle array = NULL;
  assert(terlan_ndarray_v1_from_ints(values, count, shape, rank, &array) == 0);
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

static void assert_dtype(TerlanNdArrayHandle array, int32_t expected) {
  int32_t actual = 0;
  assert(terlan_ndarray_v1_dtype(array, &actual) == 0);
  assert(actual == expected);
}

static void assert_shape(TerlanNdArrayHandle array, const int64_t *expected,
                         size_t count) {
  int64_t *shape = NULL;
  int64_t rank = 0;
  size_t index;
  assert(terlan_ndarray_v1_rank(array, &rank) == 0);
  assert(terlan_ndarray_v1_shape(array, &shape) == 0);
  assert((size_t)rank == count);
  for (index = 0; index < count; ++index) {
    assert(shape[index] == expected[index]);
  }
}

static void assert_floats(TerlanNdArrayHandle array, const double *expected,
                          size_t count, double tolerance) {
  double *actual = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_floats_cast(array, &actual, &length) == 0);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    if (isnan(expected[index])) {
      assert(isnan(actual[index]));
    } else if (isinf(expected[index])) {
      assert(isinf(actual[index]) && signbit(actual[index]) ==
                                        signbit(expected[index]));
    } else {
      assert(fabs(actual[index] - expected[index]) <= tolerance);
    }
  }
  terlan_ndarray_v1_delete_float_values(actual);
}

static void assert_complex(TerlanNdArrayHandle array,
                           const double *expected_reals,
                           const double *expected_imags, size_t count,
                           double tolerance) {
  double *reals = NULL;
  double *imags = NULL;
  size_t real_count = 0;
  size_t imag_count = 0;
  size_t index;
  assert(terlan_ndarray_v1_complex_reals(array, &reals, &real_count) == 0);
  assert(terlan_ndarray_v1_complex_imags(array, &imags, &imag_count) == 0);
  assert(real_count == count && imag_count == count);
  for (index = 0; index < count; ++index) {
    if (isnan(expected_reals[index])) {
      assert(isnan(reals[index]));
    } else if (isinf(expected_reals[index])) {
      assert(isinf(reals[index]) &&
             signbit(reals[index]) == signbit(expected_reals[index]));
    } else {
      assert(fabs(reals[index] - expected_reals[index]) <= tolerance);
    }
    if (isnan(expected_imags[index])) {
      assert(isnan(imags[index]));
    } else if (isinf(expected_imags[index])) {
      assert(isinf(imags[index]) &&
             signbit(imags[index]) == signbit(expected_imags[index]));
    } else {
      assert(fabs(imags[index] - expected_imags[index]) <= tolerance);
    }
  }
  terlan_ndarray_v1_delete_float_values(imags);
  terlan_ndarray_v1_delete_float_values(reals);
}

static void test_unwrap_period_discontinuity_axes_and_dtypes(void) {
  TerlanNdArrayHandle wrapped = integers(
      (int64_t[]){0, 75, 150, 225, 50, 180}, 6, (int64_t[]){6}, 1);
  TerlanNdArrayHandle matrix = floats_as(
      (double[]){0, 3, -3, 0, -3, 3}, 6, (int64_t[]){2, 3}, 2,
      TERLAN_NDARRAY_DTYPE_FLOAT32);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_unwrap_integer_period(wrapped, 250, 125, 0,
                                                 &output) == 0);
  assert_dtype(output, TERLAN_NDARRAY_DTYPE_INT64);
  assert_floats(output, (double[]){0, 75, 150, 225, 300, 180}, 6, 0.0);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_unwrap_integer_period(wrapped, 250, 140, 0,
                                                 &output) == 0);
  assert_floats(output, (double[]){0, 75, 150, 225, 300, 430}, 6, 0.0);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_unwrap(matrix, 2.0 * acos(-1.0), -1, &output) ==
         0);
  assert_dtype(output, TERLAN_NDARRAY_DTYPE_FLOAT32);
  assert_shape(output, (int64_t[]){2, 3}, 2);
  assert_floats(output,
                (double[]){0, 3, 3.283185307179586, 0, -3,
                           -3.283185307179586},
                6, 1e-5);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(matrix) == 0);
  assert(terlan_ndarray_v1_delete(wrapped) == 0);
}

static void test_interpolation_defaults_nonfinite_and_scalar_shapes(void) {
  TerlanNdArrayHandle coordinates =
      floats((double[]){0, 1, 2}, 3, (int64_t[]){3}, 1);
  TerlanNdArrayHandle samples = floats(
      (double[]){0, INFINITY, 4}, 3, (int64_t[]){3}, 1);
  TerlanNdArrayHandle queries = floats(
      (double[]){-INFINITY, 0, 0.5, 1, 1.5, 2, INFINITY, NAN}, 8,
      (int64_t[]){2, 4}, 2);
  TerlanNdArrayHandle scalar = floats((double[]){0.5}, 1, NULL, 0);
  TerlanNdArrayHandle one_coordinate =
      floats((double[]){0}, 1, (int64_t[]){1}, 1);
  TerlanNdArrayHandle one_sample =
      floats((double[]){1}, 1, (int64_t[]){1}, 1);
  TerlanNdArrayHandle boundary_queries =
      floats((double[]){-1, 0, 1}, 3, (int64_t[]){3}, 1);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_interpolate_default(queries, coordinates, samples,
                                               &output) == 0);
  assert_shape(output, (int64_t[]){2, 4}, 2);
  assert_floats(output,
                (double[]){0, 0, INFINITY, INFINITY, INFINITY, 4, 4, NAN},
                8, 0.0);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_interpolate_default(scalar, coordinates, samples,
                                               &output) == 0);
  assert_shape(output, NULL, 0);
  assert_floats(output, (double[]){INFINITY}, 1, 0.0);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_interpolate_default(
             boundary_queries, one_coordinate, one_sample, &output) == 0);
  assert_floats(output, (double[]){1, 1, 1}, 3, 0.0);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_interpolate(boundary_queries, one_coordinate,
                                       one_sample, -1, 2, &output) == 0);
  assert_floats(output, (double[]){-1, 1, 2}, 3, 0.0);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(boundary_queries) == 0);
  assert(terlan_ndarray_v1_delete(one_sample) == 0);
  assert(terlan_ndarray_v1_delete(one_coordinate) == 0);
  assert(terlan_ndarray_v1_delete(scalar) == 0);
  assert(terlan_ndarray_v1_delete(queries) == 0);
  assert(terlan_ndarray_v1_delete(samples) == 0);
  assert(terlan_ndarray_v1_delete(coordinates) == 0);
}

static void test_interpolation_nonfinite_fallbacks(void) {
  static const double coordinate_cases[][2] = {
      {-INFINITY, INFINITY}, {-INFINITY, -INFINITY},
      {0, INFINITY},         {NAN, 1},
  };
  static const double sample_cases[][2] = {
      {0, 10}, {0, 10}, {0, 10}, {0, 10},
  };
  static const double expected[] = {NAN, 10, 0, NAN};
  TerlanNdArrayHandle query = floats((double[]){0.5}, 1, NULL, 0);
  size_t index;
  for (index = 0; index < 4; ++index) {
    TerlanNdArrayHandle coordinates =
        floats(coordinate_cases[index], 2, (int64_t[]){2}, 1);
    TerlanNdArrayHandle samples =
        floats(sample_cases[index], 2, (int64_t[]){2}, 1);
    TerlanNdArrayHandle output = NULL;
    assert(terlan_ndarray_v1_interpolate_default(query, coordinates, samples,
                                                 &output) == 0);
    assert_floats(output, &expected[index], 1, 0.0);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_delete(samples) == 0);
    assert(terlan_ndarray_v1_delete(coordinates) == 0);
  }
  {
    TerlanNdArrayHandle coordinates =
        floats((double[]){0, 1}, 2, (int64_t[]){2}, 1);
    TerlanNdArrayHandle samples =
        floats((double[]){-INFINITY, INFINITY}, 2, (int64_t[]){2}, 1);
    TerlanNdArrayHandle output = NULL;
    assert(terlan_ndarray_v1_interpolate_default(query, coordinates, samples,
                                                 &output) == 0);
    assert_floats(output, (double[]){NAN}, 1, 0.0);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_delete(samples) == 0);
    assert(terlan_ndarray_v1_delete(coordinates) == 0);
  }
  assert(terlan_ndarray_v1_delete(query) == 0);
}

static void test_complex_and_periodic_interpolation(void) {
  TerlanNdArrayHandle coordinates =
      floats((double[]){0, 0.25, 0.5, 0.75, 1}, 5, (int64_t[]){5}, 1);
  TerlanNdArrayHandle samples = complex_values(
      (double[]){0, 0.25, 0.5, 0.75, 1},
      (double[]){1, 1.25, 1.5, 1.75, 2}, 5, (int64_t[]){5}, 1,
      TERLAN_NDARRAY_DTYPE_COMPLEX64);
  TerlanNdArrayHandle queries =
      floats((double[]){-1, 0.3, 2}, 3, (int64_t[]){3}, 1);
  TerlanNdArrayHandle left = complex_values(
      (double[]){2}, (double[]){3}, 1, NULL, 0,
      TERLAN_NDARRAY_DTYPE_COMPLEX128);
  TerlanNdArrayHandle right = complex_values(
      (double[]){4}, (double[]){5}, 1, NULL, 0,
      TERLAN_NDARRAY_DTYPE_COMPLEX128);
  TerlanNdArrayHandle periodic_coordinates =
      floats((double[]){190, -190, 350, -350}, 4, (int64_t[]){4}, 1);
  TerlanNdArrayHandle periodic_samples = complex_values(
      (double[]){5, 10, 3, 4}, (double[]){1, 2, 3, 4}, 4,
      (int64_t[]){4}, 1, TERLAN_NDARRAY_DTYPE_COMPLEX128);
  TerlanNdArrayHandle periodic_queries = floats(
      (double[]){-180, -170, -185, 185, -10, -5, 0, 365}, 8,
      (int64_t[]){2, 4}, 2);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_interpolate_boundaries(
             queries, coordinates, samples, left, right, &output) == 0);
  assert_dtype(output, TERLAN_NDARRAY_DTYPE_COMPLEX128);
  assert_complex(output, (double[]){2, 0.3, 4}, (double[]){3, 1.3, 5}, 3,
                 1e-12);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_interpolate_periodic(
             periodic_queries, periodic_coordinates, periodic_samples, 360,
             &output) == 0);
  assert_shape(output, (int64_t[]){2, 4}, 2);
  assert_complex(output,
                 (double[]){7.5, 5, 8.75, 6.25, 3, 3.25, 3.5, 3.75},
                 (double[]){1.5, 1, 1.75, 1.25, 3, 3.25, 3.5, 3.75}, 8,
                 1e-12);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(periodic_queries) == 0);
  assert(terlan_ndarray_v1_delete(periodic_samples) == 0);
  assert(terlan_ndarray_v1_delete(periodic_coordinates) == 0);
  assert(terlan_ndarray_v1_delete(right) == 0);
  assert(terlan_ndarray_v1_delete(left) == 0);
  assert(terlan_ndarray_v1_delete(queries) == 0);
  assert(terlan_ndarray_v1_delete(samples) == 0);
  assert(terlan_ndarray_v1_delete(coordinates) == 0);
}

static void test_trapezoids_broadcast_coordinates_and_preserve_dtypes(void) {
  const double q_values[] = {0, 3, 2, 5, 0.5, 3.5, 2.5, 5.5,
                             1, 4, 3, 6};
  TerlanNdArrayHandle q =
      floats(q_values, 12, (int64_t[]){3, 2, 2}, 3);
  TerlanNdArrayHandle x =
      floats((double[]){0, 0.5, 1}, 3, (int64_t[]){3, 1, 1}, 3);
  TerlanNdArrayHandle y =
      floats((double[]){0, 2}, 2, (int64_t[]){1, 2, 1}, 3);
  TerlanNdArrayHandle single = floats_as(
      (double[]){1, 2, 3}, 3, (int64_t[]){3}, 1,
      TERLAN_NDARRAY_DTYPE_FLOAT32);
  TerlanNdArrayHandle complex = complex_values(
      (double[]){1, 2, 3}, (double[]){1, 2, 3}, 3, (int64_t[]){3}, 1,
      TERLAN_NDARRAY_DTYPE_COMPLEX64);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_trapezoidal_integral(q, x, 0, &output) == 0);
  assert_shape(output, (int64_t[]){2, 2}, 2);
  assert_floats(output, (double[]){0.5, 3.5, 2.5, 5.5}, 4, 1e-12);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_trapezoidal_integral(q, y, 1, &output) == 0);
  assert_shape(output, (int64_t[]){3, 2}, 2);
  assert_floats(output, (double[]){2, 8, 3, 9, 4, 10}, 6, 1e-12);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_trapezoidal_integral_uniform(single, 1, 0,
                                                        &output) == 0);
  assert_dtype(output, TERLAN_NDARRAY_DTYPE_FLOAT32);
  assert_floats(output, (double[]){4}, 1, 0.0);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_trapezoidal_integral_uniform(complex, 1, 0,
                                                        &output) == 0);
  assert_dtype(output, TERLAN_NDARRAY_DTYPE_COMPLEX64);
  assert_complex(output, (double[]){4}, (double[]){4}, 1, 0.0);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(complex) == 0);
  assert(terlan_ndarray_v1_delete(single) == 0);
  assert(terlan_ndarray_v1_delete(y) == 0);
  assert(terlan_ndarray_v1_delete(x) == 0);
  assert(terlan_ndarray_v1_delete(q) == 0);
}

static void test_calculus_rejections_and_allocation_cleanup(void) {
  TerlanNdArrayHandle values =
      floats((double[]){0.5}, 1, (int64_t[]){1}, 1);
  TerlanNdArrayHandle coordinates =
      floats((double[]){0, 1}, 2, (int64_t[]){2}, 1);
  TerlanNdArrayHandle samples =
      floats((double[]){0, 1}, 2, (int64_t[]){2}, 1);
  TerlanNdArrayHandle wrong =
      floats((double[]){0, 1, 2}, 3, (int64_t[]){3}, 1);
  TerlanNdArrayHandle output = NULL;
  const int64_t baseline = terlan_ndarray_v1_live_count();
  int64_t failure;
  assert(terlan_ndarray_v1_unwrap(values, 0, 0, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_unwrap_discontinuity(values, 2, -1, 0, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_interpolate_default(values, coordinates, wrong,
                                               &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_interpolate_periodic(values, coordinates, samples,
                                                0, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_trapezoidal_integral(values, coordinates, 0,
                                                &output) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(output == NULL);
  for (failure = 0; failure < 3; ++failure) {
    terlan_ndarray_test_fail_allocation_after(failure);
    output = NULL;
    if (terlan_ndarray_v1_interpolate_periodic(
            values, coordinates, samples, 2, &output) == 0) {
      assert(terlan_ndarray_v1_delete(output) == 0);
    } else {
      assert(output == NULL);
    }
    terlan_ndarray_test_reset_allocation_failures();
    assert(terlan_ndarray_v1_live_count() == baseline);
  }
  assert(terlan_ndarray_v1_delete(wrong) == 0);
  assert(terlan_ndarray_v1_delete(samples) == 0);
  assert(terlan_ndarray_v1_delete(coordinates) == 0);
  assert(terlan_ndarray_v1_delete(values) == 0);
}

int main(void) {
  test_unwrap_period_discontinuity_axes_and_dtypes();
  test_interpolation_defaults_nonfinite_and_scalar_shapes();
  test_interpolation_nonfinite_fallbacks();
  test_complex_and_periodic_interpolation();
  test_trapezoids_broadcast_coordinates_and_preserve_dtypes();
  test_calculus_rejections_and_allocation_cleanup();
  return 0;
}
