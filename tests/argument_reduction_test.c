#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static void assert_indexes(TerlanNdArrayHandle array,
                           const int64_t *expected, size_t count) {
  int64_t *values = NULL;
  size_t length = 0;
  size_t index;
  assert(array != NULL);
  assert(terlan_ndarray_v1_to_ints_cast(array, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(values[index] == expected[index]);
  }
  terlan_ndarray_v1_delete_int_values(values);
}

static void assert_shape(TerlanNdArrayHandle array, const int64_t *expected,
                         int64_t rank) {
  int64_t axis;
  assert(array != NULL);
  assert(array->rank == rank);
  for (axis = 0; axis < rank; ++axis) {
    assert(array->shape[axis] == expected[axis]);
  }
}

static void test_numpy_argmin_argmax_cover_every_axis(void) {
  const int64_t values[] = {4, 9, 2, 8, 5, 7, 6, 1, 3, 0, 5, 10};
  const int64_t shape[] = {2, 3, 2};
  const int64_t expected_minimums[][6] = {
      {0, 1, 0, 1, 0, 0},
      {1, 2, 1, 1},
      {0, 0, 0, 1, 1, 0},
  };
  const int64_t expected_maximums[][6] = {
      {1, 0, 1, 0, 0, 1},
      {2, 0, 0, 2},
      {1, 1, 1, 0, 0, 1},
  };
  const size_t counts[] = {6, 4, 6};
  const int64_t output_shapes[][3] = {{3, 2, 0}, {2, 2, 0}, {2, 3, 0}};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle result = NULL;
  int64_t axis;

  assert(terlan_ndarray_v1_from_ints(values, 12, shape, 3, &source) ==
         TERLAN_NDARRAY_OK);
  for (axis = 0; axis < 3; ++axis) {
    assert(terlan_ndarray_v1_argmin(source, &axis, 1, 0, &result) ==
           TERLAN_NDARRAY_OK);
    assert_shape(result, output_shapes[axis], 2);
    assert_indexes(result, expected_minimums[axis], counts[axis]);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    result = NULL;

    assert(terlan_ndarray_v1_argmax(source, &axis, 1, 1, &result) ==
           TERLAN_NDARRAY_OK);
    assert(result->rank == 3 && result->shape[axis] == 1);
    assert_indexes(result, expected_maximums[axis], counts[axis]);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    result = NULL;
  }
  assert(terlan_ndarray_v1_argmin(source, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_shape(result, NULL, 0);
  assert_indexes(result, (const int64_t[]){9}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  result = NULL;
  assert(terlan_ndarray_v1_argmax(source, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_indexes(result, (const int64_t[]){11}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);

  source = NULL;
  result = NULL;
  assert(terlan_ndarray_v1_zeros((const int64_t[]){4, 5, 6, 7, 8}, 5,
                                  TERLAN_NDARRAY_DTYPE_FLOAT64, &source) ==
         TERLAN_NDARRAY_OK);
  for (axis = 0; axis < 5; ++axis) {
    assert(terlan_ndarray_v1_argmax(source, &axis, 1, 0, &result) ==
           TERLAN_NDARRAY_OK);
    assert(result->rank == 4 &&
           result->numel == source->numel / source->shape[axis]);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    result = NULL;
  }
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_numpy_argmin_argmax_numeric_combinations(void) {
  const double non_finite[] = {0.0, 1.0, NAN, 3.0, NAN};
  const double complex_real[] = {1.0, 0.0, 1.0};
  const double complex_imag[] = {0.0, 2.0, 1.0};
  const double complex_nan_real[] = {0.0, NAN, NAN};
  const double complex_nan_imag[] = {0.0, 2.0, NAN};
  const uint8_t booleans[] = {1, 0, 1, 0, 1};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle result = NULL;

  assert(terlan_ndarray_v1_from_floats(non_finite, 5,
                                        (const int64_t[]){5}, 1, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argmin(source, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_indexes(result, (const int64_t[]){2}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argmax(source, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_indexes(result, (const int64_t[]){2}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_complex(
             complex_real, 3, complex_imag, 3, (const int64_t[]){3}, 1,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argmin(source, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_indexes(result, (const int64_t[]){1}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argmax(source, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_indexes(result, (const int64_t[]){2}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_complex(
             complex_nan_real, 3, complex_nan_imag, 3,
             (const int64_t[]){3}, 1, TERLAN_NDARRAY_DTYPE_COMPLEX128,
             &source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argmin(source, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_indexes(result, (const int64_t[]){1}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argmax(source, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_indexes(result, (const int64_t[]){1}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_bools(booleans, 5, (const int64_t[]){5}, 1,
                                      &source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argmin(source, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_indexes(result, (const int64_t[]){1}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argmax(source, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_indexes(result, (const int64_t[]){0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_numpy_argmin_signed_integer_minima(void) {
  const int32_t dtypes[] = {
      TERLAN_NDARRAY_DTYPE_INT8,
      TERLAN_NDARRAY_DTYPE_INT16,
      TERLAN_NDARRAY_DTYPE_INT32,
      TERLAN_NDARRAY_DTYPE_INT64,
  };
  const int64_t minima[] = {INT8_MIN, INT16_MIN, INT32_MIN, INT64_MIN};
  size_t index;
  for (index = 0; index < sizeof(dtypes) / sizeof(dtypes[0]); ++index) {
    const int64_t values[] = {1, minima[index], minima[index] + 1};
    TerlanNdArrayHandle source = NULL;
    TerlanNdArrayHandle result = NULL;
    assert(terlan_ndarray_v1_from_ints_as(values, 3,
                                           (const int64_t[]){3}, 1,
                                           dtypes[index], &source) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_argmin(source, NULL, 0, 0, &result) ==
           TERLAN_NDARRAY_OK);
    assert_indexes(result, (const int64_t[]){1}, 1);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
  }
}

static void test_numpy_argument_reduction_destinations_are_exact_and_atomic(
    void) {
  const int64_t values[] = {4, 1, 3, 0, 7, 2};
  const int64_t axis[] = {1};
  const int64_t sentinel[] = {9, 9};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle minimum = NULL;
  TerlanNdArrayHandle maximum = NULL;
  TerlanNdArrayHandle wrong_shape = NULL;
  TerlanNdArrayHandle wrong_dtype = NULL;

  assert(terlan_ndarray_v1_from_ints(values, 6,
                                     (const int64_t[]){2, 3}, 2, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints_as(
             sentinel, 2, (const int64_t[]){2}, 1,
             TERLAN_NDARRAY_DTYPE_INT32, &minimum) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_bools((const uint8_t[]){0, 0}, 2,
                                      (const int64_t[]){2}, 1, &maximum) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argmin_into(minimum, source, axis, 1, 0) ==
         TERLAN_NDARRAY_OK);
  assert_indexes(minimum, (const int64_t[]){1, 0}, 2);
  assert(terlan_ndarray_v1_argmax_into(maximum, source, axis, 1, 0) ==
         TERLAN_NDARRAY_OK);
  assert_indexes(maximum, (const int64_t[]){0, 1}, 2);

  assert(terlan_ndarray_v1_from_ints(
             (const int64_t[]){8, 8, 8}, 3, (const int64_t[]){3}, 1,
             &wrong_shape) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argmin_into(wrong_shape, source, axis, 1, 0) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert_indexes(wrong_shape, (const int64_t[]){8, 8, 8}, 3);

  assert(terlan_ndarray_v1_from_floats((const double[]){8.0, 8.0}, 2,
                                        (const int64_t[]){2}, 1,
                                        &wrong_dtype) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argmax_into(wrong_dtype, source, axis, 1, 0) ==
         TERLAN_NDARRAY_UNSUPPORTED_DTYPE);
  assert_indexes(wrong_dtype, (const int64_t[]){8, 8}, 2);

  assert(terlan_ndarray_v1_delete(wrong_dtype) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(wrong_shape) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(maximum) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(minimum) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

int main(void) {
  test_numpy_argmin_argmax_cover_every_axis();
  test_numpy_argmin_argmax_numeric_combinations();
  test_numpy_argmin_signed_integer_minima();
  test_numpy_argument_reduction_destinations_are_exact_and_atomic();
  assert(terlan_ndarray_v1_live_count() == 0);
  return 0;
}
