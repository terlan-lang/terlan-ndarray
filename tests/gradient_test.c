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

static TerlanNdArrayHandle ints_as(const int64_t *values, int64_t count,
                                   const int64_t *shape, int64_t rank,
                                   int32_t dtype) {
  TerlanNdArrayHandle array = NULL;
  assert(terlan_ndarray_v1_from_ints_as(values, count, shape, rank, dtype,
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
                          size_t count, double tolerance) {
  double *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_floats_cast(array, &values, &length) == 0);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    const double scale = fmax(1.0, fabs(expected[index]));
    assert(fabs(values[index] - expected[index]) <= tolerance * scale);
  }
  terlan_ndarray_v1_delete_float_values(values);
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
    assert(fabs(reals[index] - expected_reals[index]) <= tolerance);
    assert(fabs(imags[index] - expected_imags[index]) <= tolerance);
  }
  terlan_ndarray_v1_delete_float_values(imags);
  terlan_ndarray_v1_delete_float_values(reals);
}

static void assert_dtype(TerlanNdArrayHandle array, int32_t expected) {
  int32_t actual = 0;
  assert(terlan_ndarray_v1_dtype(array, &actual) == 0);
  assert(actual == expected);
}

static TerlanNdArrayHandle output_at(const uint64_t *values, size_t index) {
  return (TerlanNdArrayHandle)(uintptr_t)values[index];
}

static void delete_outputs(uint64_t *values, size_t count) {
  size_t index;
  for (index = 0; index < count; ++index) {
    assert(terlan_ndarray_v1_delete(output_at(values, index)) == 0);
  }
  terlan_ndarray_v1_delete_handle_values(values);
}

static void test_gradient_all_axes_and_axis_order(void) {
  TerlanNdArrayHandle array =
      ints_as((int64_t[]){1, 1, 3, 4}, 4, (int64_t[]){2, 2}, 2,
              TERLAN_NDARRAY_DTYPE_INT64);
  uint64_t *outputs = NULL;
  size_t output_count = 0;
  assert(terlan_ndarray_v1_gradient_all(array, 1.0, 1, &outputs,
                                        &output_count) == 0);
  assert(output_count == 2);
  assert_floats(output_at(outputs, 0), (double[]){2, 3, 2, 3}, 4, 1e-12);
  assert_floats(output_at(outputs, 1), (double[]){0, 0, 1, 1}, 4, 1e-12);
  delete_outputs(outputs, output_count);

  assert(terlan_ndarray_v1_gradient_uniform_axes(
             array, (double[]){2, 3}, 2, (int64_t[]){1, 0}, 2, 1,
             &outputs, &output_count) == 0);
  assert(output_count == 2);
  assert_floats(output_at(outputs, 0), (double[]){0, 0, 0.5, 0.5}, 4,
                1e-12);
  assert_floats(output_at(outputs, 1),
                (double[]){2.0 / 3.0, 1, 2.0 / 3.0, 1}, 4, 1e-12);
  delete_outputs(outputs, output_count);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_gradient_coordinate_spacing_and_edge_orders(void) {
  TerlanNdArrayHandle values = floats(
      (double[]){0, 2, 3, 4, 5, 5}, 6, (int64_t[]){6}, 1);
  TerlanNdArrayHandle coordinates = floats(
      (double[]){0, 0.5, 1, 3, 5, 7}, 6, (int64_t[]){6}, 1);
  TerlanNdArrayHandle quadratic = floats(
      (double[]){0, 1, 4, 9, 16}, 5, (int64_t[]){5}, 1);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_gradient_coordinates(values, coordinates, 0, 1,
                                                 &output) == 0);
  assert_floats(output, (double[]){4, 3, 1.7, 0.5, 0.25, 0}, 6, 1e-12);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_gradient_coordinates(values, coordinates, 0, 2,
                                                 &output) == 0);
  assert_floats(output, (double[]){5, 3, 1.7, 0.5, 0.25, -0.25}, 6,
                1e-12);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_gradient_edge(quadratic, 1.0, 0, 2, &output) ==
         0);
  assert_floats(output, (double[]){0, 2, 4, 6, 8}, 5, 1e-12);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(quadratic) == 0);
  assert(terlan_ndarray_v1_delete(coordinates) == 0);
  assert(terlan_ndarray_v1_delete(values) == 0);
}

static void test_gradient_mixed_spacing_arrays(void) {
  TerlanNdArrayHandle array = floats(
      (double[]){1, 2, 6, 3, 4, 5}, 6, (int64_t[]){2, 3}, 2);
  TerlanNdArrayHandle row_spacing = floats((double[]){2}, 1, NULL, 0);
  TerlanNdArrayHandle column_coordinates = floats(
      (double[]){1, 1.5, 3.5}, 3, (int64_t[]){3}, 1);
  const uint64_t spacings[] = {
      (uint64_t)(uintptr_t)row_spacing,
      (uint64_t)(uintptr_t)column_coordinates,
  };
  uint64_t *outputs = NULL;
  size_t output_count = 0;
  assert(terlan_ndarray_v1_gradient_axes(
             array, spacings, 2, (int64_t[]){0, 1}, 2, 1, &outputs,
             &output_count) == 0);
  assert(output_count == 2);
  assert_floats(output_at(outputs, 0), (double[]){1, 1, -0.5, 1, 1, -0.5},
                6, 1e-12);
  assert_floats(output_at(outputs, 1), (double[]){2, 2, 2, 2, 1.7, 0.5}, 6,
                1e-12);
  delete_outputs(outputs, output_count);
  assert(terlan_ndarray_v1_delete(column_coordinates) == 0);
  assert(terlan_ndarray_v1_delete(row_spacing) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_gradient_dtype_and_integer_range_semantics(void) {
  const int32_t float_dtypes[] = {TERLAN_NDARRAY_DTYPE_FLOAT16,
                                  TERLAN_NDARRAY_DTYPE_FLOAT32,
                                  TERLAN_NDARRAY_DTYPE_FLOAT64};
  size_t index;
  for (index = 0; index < sizeof(float_dtypes) / sizeof(float_dtypes[0]);
       ++index) {
    TerlanNdArrayHandle array = floats_as(
        (double[]){1, 2, 3}, 3, (int64_t[]){3}, 1, float_dtypes[index]);
    TerlanNdArrayHandle output = NULL;
    assert(terlan_ndarray_v1_gradient(array, 1.0, 0, &output) == 0);
    assert_dtype(output, float_dtypes[index]);
    assert_floats(output, (double[]){1, 1, 1}, 3, 1e-3);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_delete(array) == 0);
  }
  {
    TerlanNdArrayHandle array = complex_values(
        (double[]){1, 2, 4}, (double[]){2, 4, 8}, 3,
        (int64_t[]){3}, 1, TERLAN_NDARRAY_DTYPE_COMPLEX64);
    TerlanNdArrayHandle output = NULL;
    assert(terlan_ndarray_v1_gradient(array, 1.0, 0, &output) == 0);
    assert_dtype(output, TERLAN_NDARRAY_DTYPE_COMPLEX64);
    assert_complex(output, (double[]){1, 1.5, 2}, (double[]){2, 3, 4}, 3,
                   1e-6);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_delete(array) == 0);
  }
  {
    TerlanNdArrayHandle decreasing = ints_as(
        (int64_t[]){5, 4, 3, 2, 1}, 5, (int64_t[]){5}, 1,
        TERLAN_NDARRAY_DTYPE_UINT8);
    TerlanNdArrayHandle positions = ints_as(
        (int64_t[]){3, 2, 1}, 3, (int64_t[]){3}, 1,
        TERLAN_NDARRAY_DTYPE_UINT64);
    TerlanNdArrayHandle increasing = ints_as(
        (int64_t[]){0, 2, 4}, 3, (int64_t[]){3}, 1,
        TERLAN_NDARRAY_DTYPE_INT64);
    TerlanNdArrayHandle output = NULL;
    assert(terlan_ndarray_v1_gradient(decreasing, 1.0, 0, &output) == 0);
    assert_dtype(output, TERLAN_NDARRAY_DTYPE_FLOAT64);
    assert_floats(output, (double[]){-1, -1, -1, -1, -1}, 5, 1e-12);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_gradient_coordinates(increasing, positions, 0, 1,
                                                   &output) == 0);
    assert_floats(output, (double[]){-2, -2, -2}, 3, 1e-12);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_delete(increasing) == 0);
    assert(terlan_ndarray_v1_delete(positions) == 0);
    assert(terlan_ndarray_v1_delete(decreasing) == 0);
  }
  {
    const int64_t maximum = INT64_MAX;
    const int64_t minimum = INT64_MIN;
    TerlanNdArrayHandle values = ints_as(
        (int64_t[]){minimum / 2, 0}, 2, (int64_t[]){2}, 1,
        TERLAN_NDARRAY_DTYPE_INT64);
    TerlanNdArrayHandle positions = ints_as(
        (int64_t[]){-1, maximum}, 2, (int64_t[]){2}, 1,
        TERLAN_NDARRAY_DTYPE_INT64);
    TerlanNdArrayHandle output = NULL;
    assert(terlan_ndarray_v1_gradient_coordinates(values, positions, 0, 1,
                                                   &output) == 0);
    assert_floats(output, (double[]){0.5, 0.5}, 2, 1e-12);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_delete(positions) == 0);
    assert(terlan_ndarray_v1_delete(values) == 0);
  }
}

static void test_gradient_rejections_and_allocation_cleanup(void) {
  TerlanNdArrayHandle array = floats(
      (double[]){0, 1, 4, 9, 16, 25}, 6, (int64_t[]){2, 3}, 2);
  TerlanNdArrayHandle short_array = floats(
      (double[]){0, 1}, 2, (int64_t[]){2}, 1);
  TerlanNdArrayHandle wrong_coordinates = floats(
      (double[]){0, 1}, 2, (int64_t[]){2}, 1);
  TerlanNdArrayHandle matrix_coordinates = floats(
      (double[]){0, 1, 2, 3}, 4, (int64_t[]){2, 2}, 2);
  const uint64_t coordinate_handle =
      (uint64_t)(uintptr_t)wrong_coordinates;
  TerlanNdArrayHandle output = NULL;
  uint64_t *outputs = NULL;
  size_t output_count = 0;
  int64_t failure;
  const int64_t baseline = terlan_ndarray_v1_live_count();

  assert(terlan_ndarray_v1_gradient_edge(array, 1, 0, 3, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_gradient_edge(short_array, 1, 0, 2, &output) ==
         TERLAN_NDARRAY_DOMAIN_ERROR);
  assert(terlan_ndarray_v1_gradient_coordinates(
             array, wrong_coordinates, 1, 1, &output) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(terlan_ndarray_v1_gradient_coordinates(
             array, matrix_coordinates, 1, 1, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_gradient_uniform_axes(
             array, (double[]){1}, 1, (int64_t[]){0, 0}, 2, 1, &outputs,
             &output_count) == TERLAN_NDARRAY_DUPLICATE_AXIS);
  assert(terlan_ndarray_v1_gradient_uniform_axes(
             array, (double[]){1, 2, 3}, 3, (int64_t[]){0, 1}, 2, 1,
             &outputs, &output_count) == TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_gradient_uniform_axes(
             array, (double[]){1}, 1, (int64_t[]){2}, 1, 1, &outputs,
             &output_count) == TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(terlan_ndarray_v1_gradient_axes(
             array, &coordinate_handle, 1, (int64_t[]){0, 1}, 2, 1,
             &outputs, &output_count) == TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(output == NULL && outputs == NULL && output_count == 0);

  for (failure = 0; failure < 8; ++failure) {
    terlan_ndarray_test_fail_allocation_after(failure);
    outputs = NULL;
    output_count = 0;
    if (terlan_ndarray_v1_gradient_all(array, 1, 1, &outputs,
                                       &output_count) == 0) {
      delete_outputs(outputs, output_count);
    } else {
      assert(outputs == NULL && output_count == 0);
    }
    terlan_ndarray_test_reset_allocation_failures();
    assert(terlan_ndarray_v1_live_count() == baseline);
  }
  assert_floats(array, (double[]){0, 1, 4, 9, 16, 25}, 6, 1e-12);
  assert(terlan_ndarray_v1_delete(matrix_coordinates) == 0);
  assert(terlan_ndarray_v1_delete(wrong_coordinates) == 0);
  assert(terlan_ndarray_v1_delete(short_array) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

int main(void) {
  test_gradient_all_axes_and_axis_order();
  test_gradient_coordinate_spacing_and_edge_orders();
  test_gradient_mixed_spacing_arrays();
  test_gradient_dtype_and_integer_range_semantics();
  test_gradient_rejections_and_allocation_cleanup();
  return 0;
}
