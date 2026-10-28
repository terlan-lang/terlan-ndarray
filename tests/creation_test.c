#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

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
  assert(terlan_ndarray_v1_to_floats_cast(array, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(values[index] == expected[index]);
  }
  terlan_ndarray_v1_delete_float_values(values);
}

static void assert_shape(TerlanNdArrayHandle array, const int64_t *expected,
                         size_t count) {
  int64_t rank = 0;
  int64_t *shape = NULL;
  size_t index;
  assert(terlan_ndarray_v1_rank(array, &rank) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_shape(array, &shape) == TERLAN_NDARRAY_OK);
  assert(rank == (int64_t)count);
  for (index = 0; index < count; ++index) {
    assert(shape[index] == expected[index]);
  }
}

static void test_fill_constructors_cover_scalar_empty_and_matrix(void) {
  const int64_t shape[] = {2, 3};
  const int64_t zeros[] = {0, 0, 0, 0, 0, 0};
  const int64_t ones[] = {1, 1, 1, 1, 1, 1};
  const int64_t sevens[] = {7, 7, 7, 7, 7, 7};
  TerlanNdArrayHandle array = NULL;
  assert(terlan_ndarray_v1_empty(shape, 2, TERLAN_NDARRAY_DTYPE_INT64,
                                &array) == TERLAN_NDARRAY_OK);
  assert_ints(array, zeros, 6);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_zeros(shape, 2, TERLAN_NDARRAY_DTYPE_INT16,
                                 &array) == TERLAN_NDARRAY_OK);
  assert_ints(array, zeros, 6);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_ones(shape, 2, TERLAN_NDARRAY_DTYPE_UINT8,
                                &array) == TERLAN_NDARRAY_OK);
  assert_ints(array, ones, 6);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_full_int(shape, 2, 7,
                                    TERLAN_NDARRAY_DTYPE_INT32, &array) ==
         TERLAN_NDARRAY_OK);
  assert_ints(array, sevens, 6);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_full_float(NULL, 0, 2.5,
                                      TERLAN_NDARRAY_DTYPE_FLOAT32, &array) ==
         TERLAN_NDARRAY_OK);
  assert_floats(array, (const double[]){2.5}, 1);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_ranges_spacing_and_identity(void) {
  const int64_t range[] = {-3, -1, 1, 3};
  const double spacing[] = {0.0, 0.25, 0.5, 0.75, 1.0};
  const int64_t identity[] = {0, 1, 0, 0, 0, 1};
  TerlanNdArrayHandle array = NULL;
  assert(terlan_ndarray_v1_arange(-3.0, 5.0, 2.0,
                                   TERLAN_NDARRAY_DTYPE_INT16, &array) ==
         TERLAN_NDARRAY_OK);
  assert_ints(array, range, 4);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_linspace(0.0, 1.0, 5, 1,
                                     TERLAN_NDARRAY_DTYPE_FLOAT64, &array) ==
         TERLAN_NDARRAY_OK);
  assert_floats(array, spacing, 5);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_identity(2, 3, 1, TERLAN_NDARRAY_DTYPE_INT64,
                                    &array) == TERLAN_NDARRAY_OK);
  assert_ints(array, identity, 6);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_arange_matches_numpy_numeric_boundaries(void) {
  const int64_t ascending[] = {0, 1, 2, 3, 4};
  const int64_t descending[] = {5, 3, 1, -1};
  const int64_t high[] = {INT64_MAX - 2, INT64_MAX - 1};
  const int64_t low[] = {INT64_MIN + 2, INT64_MIN + 1};
  TerlanNdArrayHandle array = NULL;

  assert(terlan_ndarray_v1_arange_int_stop(5, &array) ==
         TERLAN_NDARRAY_OK);
  assert_ints(array, ascending, 5);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_arange_float_stop(3.0, &array) ==
         TERLAN_NDARRAY_OK);
  assert_floats(array, (const double[]){0.0, 1.0, 2.0}, 3);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_arange_int(5, -2, -2, &array) ==
         TERLAN_NDARRAY_OK);
  assert_ints(array, descending, 4);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_arange_int(INT64_MAX - 2, INT64_MAX, 1,
                                      &array) == TERLAN_NDARRAY_OK);
  assert_ints(array, high, 2);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_arange_int(INT64_MIN + 2, INT64_MIN, -1,
                                      &array) == TERLAN_NDARRAY_OK);
  assert_ints(array, low, 2);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_arange(0.0, 1.0, INFINITY,
                                  TERLAN_NDARRAY_DTYPE_FLOAT64, &array) ==
         TERLAN_NDARRAY_OK);
  assert_floats(array, (const double[]){0.0}, 1);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_arange(0.0, 1.0, -INFINITY,
                                  TERLAN_NDARRAY_DTYPE_FLOAT64, &array) ==
         TERLAN_NDARRAY_OK);
  assert_floats(array, NULL, 0);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_arange(0.0, -1.0, -INFINITY,
                                  TERLAN_NDARRAY_DTYPE_FLOAT64, &array) ==
         TERLAN_NDARRAY_OK);
  assert_floats(array, (const double[]){0.0}, 1);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_arange(0.0, -1.0, INFINITY,
                                  TERLAN_NDARRAY_DTYPE_FLOAT64, &array) ==
         TERLAN_NDARRAY_OK);
  assert_floats(array, NULL, 0);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_arange(0.0, DBL_EPSILON, DBL_MAX,
                                  TERLAN_NDARRAY_DTYPE_FLOAT64, &array) ==
         TERLAN_NDARRAY_OK);
  assert_floats(array, (const double[]){0.0}, 1);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_arange(0.0, -DBL_EPSILON, -DBL_MAX,
                                  TERLAN_NDARRAY_DTYPE_FLOAT64, &array) ==
         TERLAN_NDARRAY_OK);
  assert_floats(array, (const double[]){0.0}, 1);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_arange_int(INT64_MIN, INT64_MAX, 1, &array) ==
         TERLAN_NDARRAY_SIZE_OVERFLOW);
  assert(array == NULL);
  assert(terlan_ndarray_v1_arange(0.0, DBL_MAX, DBL_TRUE_MIN,
                                  TERLAN_NDARRAY_DTYPE_FLOAT64, &array) ==
         TERLAN_NDARRAY_SIZE_OVERFLOW);
  assert(array == NULL);
  assert(terlan_ndarray_v1_arange(0.0, 1.0, NAN,
                                  TERLAN_NDARRAY_DTYPE_FLOAT64, &array) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_arange_int(0, 1, 0, &array) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_arange(0.0, INFINITY, 1.0,
                                  TERLAN_NDARRAY_DTYPE_FLOAT64, &array) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(array == NULL);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_linspace_matches_numpy_scalar_dtype_and_rounding_semantics(
    void) {
  const int64_t rounded[] = {-1, -1, 0, 0, 1, 1, 2, 3};
  const double scalar_limits[] = {-120.0, -65.0, -10.0, 45.0, 100.0};
  const double descending_limits[] = {120.0, 65.0, 10.0, -45.0, -100.0};
  const double unsigned_limits[] = {1200.0, 1150.0, 1100.0, 1050.0, 1000.0};
  const int64_t sample_counts[] = {0, 1, 2, 17, 999};
  const double denormal_stops[] = {
      0x1p-24 * 5.0,
      (double)FLT_TRUE_MIN * 5.0,
      DBL_TRUE_MIN * 5.0,
  };
  const int32_t denormal_dtypes[] = {
      TERLAN_NDARRAY_DTYPE_FLOAT16,
      TERLAN_NDARRAY_DTYPE_FLOAT32,
      TERLAN_NDARRAY_DTYPE_FLOAT64,
  };
  TerlanNdArrayHandle array = NULL;
  int64_t numel = -1;
  int32_t dtype = 0;
  size_t index;

  assert(terlan_ndarray_v1_linspace(0.0, 10.0, 50, 1,
                                    TERLAN_NDARRAY_DTYPE_FLOAT64, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_numel(array, &numel) == TERLAN_NDARRAY_OK);
  assert(numel == 50);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_linspace(2.0, 10.0, 100, 1,
                                    TERLAN_NDARRAY_DTYPE_FLOAT64, &array) ==
         TERLAN_NDARRAY_OK);
  {
    double *values = NULL;
    size_t length = 0;
    assert(terlan_ndarray_v1_to_floats_cast(array, &values, &length) ==
           TERLAN_NDARRAY_OK);
    assert(length == 100);
    assert(values[99] == 10.0);
    terlan_ndarray_v1_delete_float_values(values);
  }
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_linspace(2.0, 10.0, 50, 0,
                                    TERLAN_NDARRAY_DTYPE_FLOAT64, &array) ==
         TERLAN_NDARRAY_OK);
  {
    double *values = NULL;
    size_t length = 0;
    assert(terlan_ndarray_v1_to_floats_cast(array, &values, &length) ==
           TERLAN_NDARRAY_OK);
    assert(length == 50);
    assert(values[49] < 10.0);
    terlan_ndarray_v1_delete_float_values(values);
  }
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_linspace(0.0, 1.0, 0, 1,
                                    TERLAN_NDARRAY_DTYPE_FLOAT32, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_numel(array, &numel) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dtype(array, &dtype) == TERLAN_NDARRAY_OK);
  assert(numel == 0 && dtype == TERLAN_NDARRAY_DTYPE_FLOAT32);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_linspace(0.0, 1.0, 1, 1,
                                    TERLAN_NDARRAY_DTYPE_FLOAT32, &array) ==
         TERLAN_NDARRAY_OK);
  assert_floats(array, (const double[]){0.0}, 1);
  assert(terlan_ndarray_v1_dtype(array, &dtype) == TERLAN_NDARRAY_OK);
  assert(dtype == TERLAN_NDARRAY_DTYPE_FLOAT32);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_linspace(-1.0, 3.0, 8, 1,
                                    TERLAN_NDARRAY_DTYPE_INT32, &array) ==
         TERLAN_NDARRAY_OK);
  assert_ints(array, rounded, 8);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_linspace(-120.0, 100.0, 5, 1,
                                    TERLAN_NDARRAY_DTYPE_FLOAT64, &array) ==
         TERLAN_NDARRAY_OK);
  assert_floats(array, scalar_limits, 5);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_linspace(120.0, -100.0, 5, 1,
                                    TERLAN_NDARRAY_DTYPE_FLOAT64, &array) ==
         TERLAN_NDARRAY_OK);
  assert_floats(array, descending_limits, 5);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_linspace(1200.0, 1000.0, 5, 1,
                                    TERLAN_NDARRAY_DTYPE_FLOAT64, &array) ==
         TERLAN_NDARRAY_OK);
  assert_floats(array, unsigned_limits, 5);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  for (index = 0; index < sizeof(sample_counts) / sizeof(sample_counts[0]);
       ++index) {
    const int64_t stop = sample_counts[index];
    TerlanNdArrayHandle range = NULL;
    assert(terlan_ndarray_v1_linspace(0.0, (double)stop, stop + 1, 1,
                                      TERLAN_NDARRAY_DTYPE_INT64, &array) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_arange(0.0, (double)(stop + 1), 1.0,
                                    TERLAN_NDARRAY_DTYPE_INT64, &range) ==
           TERLAN_NDARRAY_OK);
    {
      int64_t *actual = NULL;
      int64_t *expected = NULL;
      size_t actual_length = 0;
      size_t expected_length = 0;
      size_t value_index;
      assert(terlan_ndarray_v1_to_ints_cast(array, &actual, &actual_length) ==
             TERLAN_NDARRAY_OK);
      assert(terlan_ndarray_v1_to_ints_cast(range, &expected,
                                            &expected_length) ==
             TERLAN_NDARRAY_OK);
      assert(actual_length == expected_length);
      for (value_index = 0; value_index < actual_length; ++value_index) {
        assert(actual[value_index] == expected[value_index]);
      }
      terlan_ndarray_v1_delete_int_values(actual);
      terlan_ndarray_v1_delete_int_values(expected);
    }
    assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(range) == TERLAN_NDARRAY_OK);
  }

  for (index = 0;
       index < sizeof(denormal_stops) / sizeof(denormal_stops[0]); ++index) {
    double *values = NULL;
    size_t length = 0;
    size_t value_index;
    int any_nonzero = 0;
    assert(terlan_ndarray_v1_linspace(0.0, denormal_stops[index], 10, 0,
                                      denormal_dtypes[index], &array) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_to_floats_cast(array, &values, &length) ==
           TERLAN_NDARRAY_OK);
    for (value_index = 0; value_index < length; ++value_index) {
      any_nonzero = any_nonzero || values[value_index] != 0.0;
    }
    assert(any_nonzero);
    terlan_ndarray_v1_delete_float_values(values);
    assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  }
}

static void test_linspace_arrays_match_numpy_broadcast_axis_complex_and_step(
    void) {
  const int64_t vector_shape[] = {2};
  const int64_t incompatible_shape[] = {3};
  const int64_t starts[] = {-120, 120};
  const int64_t stops[] = {100, -100};
  const int64_t incompatible[] = {1, 2, 3};
  const double axis_zero[] = {-120, 120, -65, 65, -10,
                              10,   45,  -45, 100, -100};
  const double axis_last[] = {-120, -65, -10, 45, 100,
                              120,  65,  10,  -45, -100};
  const double scalar_stop[] = {-120, 120, -65, 115, -10,
                                110,  45,  105, 100, 100};
  const double expected_reals[] = {1.0, 1.5, 2.0, 2.5, 3.0};
  const double expected_imags[] = {2.0, 2.5, 3.0, 3.5, 4.0};
  TerlanNdArrayHandle start = NULL;
  TerlanNdArrayHandle stop = NULL;
  TerlanNdArrayHandle scalar = NULL;
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayHandle steps = NULL;
  TerlanNdArrayHandle bad = NULL;
  TerlanNdArrayHandle other = NULL;

  assert(terlan_ndarray_v1_from_ints_as(starts, 2, vector_shape, 1,
                                        TERLAN_NDARRAY_DTYPE_INT8, &start) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints_as(stops, 2, vector_shape, 1,
                                        TERLAN_NDARRAY_DTYPE_INT8, &stop) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_linspace_arrays(
             start, stop, 5, 1, 0, TERLAN_NDARRAY_DTYPE_FLOAT64, &result) ==
         TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){5, 2}, 2);
  assert_floats(result, axis_zero, 10);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_linspace_arrays(
             start, stop, 5, 1, -1, TERLAN_NDARRAY_DTYPE_FLOAT64, &result) ==
         TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){2, 5}, 2);
  assert_floats(result, axis_last, 10);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_ints_as((const int64_t[]){100}, 1, NULL, 0,
                                        TERLAN_NDARRAY_DTYPE_INT16,
                                        &scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_linspace_arrays(
             start, scalar, 5, 1, 0, TERLAN_NDARRAY_DTYPE_FLOAT64, &result) ==
         TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){5, 2}, 2);
  assert_floats(result, scalar_stop, 10);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_linspace_steps(start, stop, 5, 1, &steps) ==
         TERLAN_NDARRAY_OK);
  assert_shape(steps, vector_shape, 1);
  assert_floats(steps, (const double[]){55.0, -55.0}, 2);
  assert(terlan_ndarray_v1_delete(steps) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(start) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(stop) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_complex(
             (const double[]){1.0}, 1, (const double[]){2.0}, 1, NULL, 0,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &start) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_complex(
             (const double[]){3.0}, 1, (const double[]){4.0}, 1, NULL, 0,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &stop) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_linspace_arrays(
             start, stop, 5, 1, 0, TERLAN_NDARRAY_DTYPE_COMPLEX128, &result) ==
         TERLAN_NDARRAY_OK);
  {
    double *parts = NULL;
    size_t length = 0;
    size_t index;
    assert(terlan_ndarray_v1_complex_reals(result, &parts, &length) ==
           TERLAN_NDARRAY_OK);
    assert(length == 5);
    for (index = 0; index < length; ++index) {
      assert(parts[index] == expected_reals[index]);
    }
    terlan_ndarray_v1_delete_float_values(parts);
    assert(terlan_ndarray_v1_complex_imags(result, &parts, &length) ==
           TERLAN_NDARRAY_OK);
    assert(length == 5);
    for (index = 0; index < length; ++index) {
      assert(parts[index] == expected_imags[index]);
    }
    terlan_ndarray_v1_delete_float_values(parts);
  }
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_linspace_steps(start, stop, 2, 1, &steps) ==
         TERLAN_NDARRAY_OK);
  {
    double *parts = NULL;
    size_t length = 0;
    assert(terlan_ndarray_v1_complex_reals(steps, &parts, &length) ==
           TERLAN_NDARRAY_OK);
    assert(length == 1 && parts[0] == 2.0);
    terlan_ndarray_v1_delete_float_values(parts);
    assert(terlan_ndarray_v1_complex_imags(steps, &parts, &length) ==
           TERLAN_NDARRAY_OK);
    assert(length == 1 && parts[0] == 2.0);
    terlan_ndarray_v1_delete_float_values(parts);
  }
  assert(terlan_ndarray_v1_delete(steps) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(start) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(stop) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_floats((const double[]){0.0}, 1, NULL, 0,
                                       &start) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats((const double[]){1.0}, 1, NULL, 0,
                                       &stop) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_linspace_steps(start, stop, 0, 1, &steps) ==
         TERLAN_NDARRAY_OK);
  {
    double *values = NULL;
    size_t length = 0;
    assert(terlan_ndarray_v1_to_floats_cast(steps, &values, &length) ==
           TERLAN_NDARRAY_OK);
    assert(length == 1 && isnan(values[0]));
    terlan_ndarray_v1_delete_float_values(values);
  }
  assert(terlan_ndarray_v1_delete(steps) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_linspace_steps(start, stop, 1, 0, &steps) ==
         TERLAN_NDARRAY_OK);
  assert_floats(steps, (const double[]){1.0}, 1);
  assert(terlan_ndarray_v1_delete(steps) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_ints(incompatible, 3, incompatible_shape, 1,
                                     &bad) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_linspace_arrays(
             start, bad, 5, 1, 0, TERLAN_NDARRAY_DTYPE_FLOAT64, &result) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints((const int64_t[]){1, 2}, 2,
                                     vector_shape, 1, &other) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_linspace_arrays(
             other, bad, 5, 1, 0, TERLAN_NDARRAY_DTYPE_FLOAT64, &result) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(result == NULL);
  assert(terlan_ndarray_v1_linspace_arrays(
             bad, stop, 5, 1, 2, TERLAN_NDARRAY_DTYPE_FLOAT64, &result) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(result == NULL);
  assert(terlan_ndarray_v1_linspace_arrays(
             other, other, -1, 1, 0, TERLAN_NDARRAY_DTYPE_FLOAT64, &result) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_linspace_arrays(
             other, other, 5, 2, 0, TERLAN_NDARRAY_DTYPE_FLOAT64, &result) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_delete(start) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(stop) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(bad) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(other) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_eye_matches_numpy_shape_diagonal_dtype_bounds_and_order(void) {
  TerlanNdArrayHandle array = NULL;
  int32_t dtype = -1;
  int32_t contiguous = -1;
  int64_t *strides = NULL;

  assert(terlan_ndarray_v1_eye(4, 4, 0, TERLAN_NDARRAY_DTYPE_FLOAT64, 0,
                               &array) == TERLAN_NDARRAY_OK);
  assert_shape(array, (const int64_t[]){4, 4}, 2);
  assert_floats(array,
                (const double[]){1, 0, 0, 0, 0, 1, 0, 0,
                                 0, 0, 1, 0, 0, 0, 0, 1},
                16);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_eye(3, 4, 2, TERLAN_NDARRAY_DTYPE_FLOAT32, 0,
                               &array) == TERLAN_NDARRAY_OK);
  assert_shape(array, (const int64_t[]){3, 4}, 2);
  assert_ints(array, (const int64_t[]){0, 0, 1, 0, 0, 0,
                                       0, 1, 0, 0, 0, 0}, 12);
  assert(terlan_ndarray_v1_dtype(array, &dtype) == TERLAN_NDARRAY_OK);
  assert(dtype == TERLAN_NDARRAY_DTYPE_FLOAT32);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_eye(4, 3, -2, TERLAN_NDARRAY_DTYPE_BOOL, 0,
                               &array) == TERLAN_NDARRAY_OK);
  assert_ints(array, (const int64_t[]){0, 0, 0, 0, 0, 0,
                                       1, 0, 0, 0, 1, 0}, 12);
  assert(terlan_ndarray_v1_dtype(array, &dtype) == TERLAN_NDARRAY_OK);
  assert(dtype == TERLAN_NDARRAY_DTYPE_BOOL);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_eye(3, 2, -3, TERLAN_NDARRAY_DTYPE_INT64, 0,
                               &array) == TERLAN_NDARRAY_OK);
  assert_ints(array, (const int64_t[]){0, 0, 0, 0, 0, 0}, 6);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_eye(4, 3, -1, TERLAN_NDARRAY_DTYPE_INT64, 0,
                               &array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_strides(array, &strides) == TERLAN_NDARRAY_OK);
  assert(strides[0] == 3 && strides[1] == 1);
  assert(terlan_ndarray_v1_is_contiguous(array, &contiguous) ==
         TERLAN_NDARRAY_OK);
  assert(contiguous == 1);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_eye(4, 3, -1, TERLAN_NDARRAY_DTYPE_INT64, 1,
                               &array) == TERLAN_NDARRAY_OK);
  assert_ints(array, (const int64_t[]){0, 0, 0, 1, 0, 0,
                                       0, 1, 0, 0, 0, 1}, 12);
  assert(terlan_ndarray_v1_strides(array, &strides) == TERLAN_NDARRAY_OK);
  assert(strides[0] == 1 && strides[1] == 4);
  assert(terlan_ndarray_v1_is_contiguous(array, &contiguous) ==
         TERLAN_NDARRAY_OK);
  assert(contiguous == 0);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_eye(2, 2, 0, TERLAN_NDARRAY_DTYPE_INT64, 2,
                               &array) == TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(array == NULL);
  assert(terlan_ndarray_v1_eye(-1, 2, 0, TERLAN_NDARRAY_DTYPE_INT64, 0,
                               &array) == TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(array == NULL);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_diagonal_is_a_mutable_shared_view(void) {
  const int64_t values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const int64_t shape[] = {3, 3};
  const int64_t diagonal[] = {2, 6};
  const int64_t source_after[] = {1, 11, 3, 4, 5, 6, 7, 8, 9};
  const int64_t index[] = {0};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle view = NULL;
  assert(terlan_ndarray_v1_from_ints(values, 9, shape, 2, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_diagonal(array, 1, 0, 1, &view) ==
         TERLAN_NDARRAY_OK);
  assert_ints(view, diagonal, 2);
  assert(terlan_ndarray_v1_set_int(view, index, 1, 11) == TERLAN_NDARRAY_OK);
  assert_ints(array, source_after, 9);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert_ints(view, (const int64_t[]){11, 6}, 2);
  assert(terlan_ndarray_v1_delete(view) == TERLAN_NDARRAY_OK);
}

static void test_creation_rejections_are_typed(void) {
  TerlanNdArrayHandle array = NULL;
  assert(terlan_ndarray_v1_empty(NULL, 1, TERLAN_NDARRAY_DTYPE_INT64,
                                &array) == TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_arange(0.0, 1.0, 0.0,
                                   TERLAN_NDARRAY_DTYPE_FLOAT64, &array) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_linspace(0.0, 1.0, -1, 1,
                                     TERLAN_NDARRAY_DTYPE_FLOAT64, &array) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_full_int(NULL, 0, 256,
                                    TERLAN_NDARRAY_DTYPE_UINT8, &array) ==
         TERLAN_NDARRAY_CAST_OVERFLOW);
  assert(array == NULL);
  assert(terlan_ndarray_v1_live_count() == 0);
}

int main(void) {
  test_fill_constructors_cover_scalar_empty_and_matrix();
  test_ranges_spacing_and_identity();
  test_arange_matches_numpy_numeric_boundaries();
  test_linspace_matches_numpy_scalar_dtype_and_rounding_semantics();
  test_linspace_arrays_match_numpy_broadcast_axis_complex_and_step();
  test_eye_matches_numpy_shape_diagonal_dtype_bounds_and_order();
  test_diagonal_is_a_mutable_shared_view();
  test_creation_rejections_are_typed();
  return 0;
}
