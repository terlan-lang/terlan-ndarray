#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
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
    assert(fabs(values[index] - expected[index]) < 1e-12);
  }
  terlan_ndarray_v1_delete_float_values(values);
}

static void assert_floats_close(TerlanNdArrayHandle array,
                                const double *expected, size_t count) {
  double *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_floats_cast(array, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    if (isnan(expected[index])) {
      assert(isnan(values[index]));
    } else if (isinf(expected[index])) {
      assert(isinf(values[index]));
      assert(signbit(values[index]) == signbit(expected[index]));
    } else {
      const double scale = fmax(1.0, fabs(expected[index]));
      assert(fabs(values[index] - expected[index]) <= 1e-10 * scale);
    }
  }
  terlan_ndarray_v1_delete_float_values(values);
}

static void assert_dtype(TerlanNdArrayHandle array, int32_t expected) {
  int32_t actual = -1;
  assert(terlan_ndarray_v1_dtype(array, &actual) == TERLAN_NDARRAY_OK);
  assert(actual == expected);
}

static TerlanNdArrayStatus create_ones(int32_t dtype, const int64_t *shape,
                                       int64_t rank, int64_t count,
                                       TerlanNdArrayHandle *out_array) {
  const int64_t integer_values[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  const double real_values[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  const double imaginary_values[] = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                                     2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
  if (dtype == TERLAN_NDARRAY_DTYPE_COMPLEX64 ||
      dtype == TERLAN_NDARRAY_DTYPE_COMPLEX128) {
    return terlan_ndarray_v1_from_complex(real_values, count, imaginary_values,
                                          count, shape, rank, dtype, out_array);
  }
  if (dtype == TERLAN_NDARRAY_DTYPE_FLOAT16 ||
      dtype == TERLAN_NDARRAY_DTYPE_FLOAT32 ||
      dtype == TERLAN_NDARRAY_DTYPE_FLOAT64) {
    return terlan_ndarray_v1_from_floats_as(real_values, count, shape, rank,
                                            dtype, out_array);
  }
  return terlan_ndarray_v1_from_ints_as(integer_values, count, shape, rank,
                                        dtype, out_array);
}

static void assert_complex_close(TerlanNdArrayHandle array,
                                 const double *expected_reals,
                                 const double *expected_imags, size_t count) {
  double *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_complex_reals(array, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    const double scale = fmax(1.0, fabs(expected_reals[index]));
    assert(fabs(values[index] - expected_reals[index]) <= 1e-8 * scale);
  }
  terlan_ndarray_v1_delete_float_values(values);
  values = NULL;
  assert(terlan_ndarray_v1_complex_imags(array, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    const double scale = fmax(1.0, fabs(expected_imags[index]));
    assert(fabs(values[index] - expected_imags[index]) <= 1e-8 * scale);
  }
  terlan_ndarray_v1_delete_float_values(values);
}

static void assert_shape(TerlanNdArrayHandle array, const int64_t *expected,
                         size_t count) {
  int64_t *shape = NULL;
  int64_t rank = 0;
  size_t index;
  assert(terlan_ndarray_v1_rank(array, &rank) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_shape(array, &shape) == TERLAN_NDARRAY_OK);
  assert(rank == (int64_t)count);
  for (index = 0; index < count; ++index) {
    assert(shape[index] == expected[index]);
  }
}

static void test_like_constructors_preserve_shape_and_dtype(void) {
  const int64_t values[] = {1, 2, 3, 4};
  const int64_t shape[] = {2, 2};
  const int64_t zeros[] = {0, 0, 0, 0};
  const int64_t ones[] = {1, 1, 1, 1};
  const int64_t sevens[] = {7, 7, 7, 7};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle result = NULL;
  int32_t dtype = -1;
  assert(terlan_ndarray_v1_from_ints_as(values, 4, shape, 2,
                                        TERLAN_NDARRAY_DTYPE_INT16,
                                        &source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_empty_like(source, &result) == TERLAN_NDARRAY_OK);
  assert_ints(result, zeros, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_zeros_like(source, &result) == TERLAN_NDARRAY_OK);
  assert_ints(result, zeros, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_ones_like(source, &result) == TERLAN_NDARRAY_OK);
  assert_ints(result, ones, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_full_like_int(source, 7, &result) ==
         TERLAN_NDARRAY_OK);
  assert_shape(result, shape, 2);
  assert(terlan_ndarray_v1_dtype(result, &dtype) == TERLAN_NDARRAY_OK);
  assert(dtype == TERLAN_NDARRAY_DTYPE_INT16);
  assert_ints(result, sevens, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_zeros_and_zeros_like_cover_all_admitted_dtypes(void) {
  const int32_t dtypes[] = {
      TERLAN_NDARRAY_DTYPE_BOOL,       TERLAN_NDARRAY_DTYPE_INT8,
      TERLAN_NDARRAY_DTYPE_INT16,      TERLAN_NDARRAY_DTYPE_INT32,
      TERLAN_NDARRAY_DTYPE_INT64,      TERLAN_NDARRAY_DTYPE_UINT8,
      TERLAN_NDARRAY_DTYPE_UINT16,     TERLAN_NDARRAY_DTYPE_UINT32,
      TERLAN_NDARRAY_DTYPE_UINT64,     TERLAN_NDARRAY_DTYPE_FLOAT16,
      TERLAN_NDARRAY_DTYPE_FLOAT32,    TERLAN_NDARRAY_DTYPE_FLOAT64,
      TERLAN_NDARRAY_DTYPE_COMPLEX64,  TERLAN_NDARRAY_DTYPE_COMPLEX128};
  const int64_t shape[] = {13};
  const int64_t zeros[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  size_t index;

  for (index = 0; index < sizeof(dtypes) / sizeof(dtypes[0]); ++index) {
    TerlanNdArrayHandle array = NULL;
    TerlanNdArrayHandle like = NULL;
    TerlanNdArrayHandle reduced = NULL;
    int32_t dtype = -1;
    int64_t count = -1;
    assert(terlan_ndarray_v1_zeros(shape, 1, dtypes[index], &array) ==
           TERLAN_NDARRAY_OK);
    assert_dtype(array, dtypes[index]);
    assert_ints(array, zeros, 13);
    assert(terlan_ndarray_v1_count_nonzero(array, &count) ==
           TERLAN_NDARRAY_OK);
    assert(count == 0);
    assert(terlan_ndarray_v1_sum(array, NULL, 0, 0, &reduced) ==
           TERLAN_NDARRAY_OK);
    assert_ints(reduced, (const int64_t[]){0}, 1);
    assert(terlan_ndarray_v1_delete(reduced) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_any(array, NULL, 0, 0, &reduced) ==
           TERLAN_NDARRAY_OK);
    assert_ints(reduced, (const int64_t[]){0}, 1);
    assert(terlan_ndarray_v1_delete(reduced) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_zeros_like(array, &like) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_dtype(like, &dtype) == TERLAN_NDARRAY_OK);
    assert(dtype == dtypes[index]);
    assert_shape(like, shape, 1);
    assert_ints(like, zeros, 13);
    assert(terlan_ndarray_v1_delete(like) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  }
}

static void test_spaced_creation_matches_numpy(void) {
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_logspace(0.0, 2.0, 3, 1, 10.0,
                                    TERLAN_NDARRAY_DTYPE_FLOAT64,
                                    &result) == TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){1.0, 10.0, 100.0}, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_geomspace(1.0, 16.0, 5, 1,
                                     TERLAN_NDARRAY_DTYPE_FLOAT64,
                                     &result) == TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){1.0, 2.0, 4.0, 8.0, 16.0}, 5);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_geomspace(-1.0, -16.0, 5, 1,
                                     TERLAN_NDARRAY_DTYPE_FLOAT64,
                                     &result) == TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){-1.0, -2.0, -4.0, -8.0, -16.0}, 5);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
}

static void test_logarithmic_spaces_match_numpy_array_and_complex_semantics(
    void) {
  const int64_t vector_shape[] = {2};
  const double starts[] = {0.0, 1.0};
  const double stops[] = {6.0, 7.0};
  const double log_axis_zero[] = {
      1.0, 10.0, 10.0, 100.0, 100.0, 1000.0, 1000.0,
      10000.0, 10000.0, 100000.0, 100000.0, 1000000.0,
      1000000.0, 10000000.0,
  };
  const double log_axis_last[] = {
      1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0,
      10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0,
  };
  TerlanNdArrayHandle start = NULL;
  TerlanNdArrayHandle stop = NULL;
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayHandle complex_start = NULL;
  TerlanNdArrayHandle complex_stop = NULL;
  TerlanNdArrayHandle incompatible = NULL;

  assert(terlan_ndarray_v1_logspace(0.0, 1.0, 4, 1, 10.0,
                                    TERLAN_NDARRAY_DTYPE_INT32,
                                    &result) == TERLAN_NDARRAY_OK);
  assert_ints(result, (const int64_t[]){1, 2, 4, 10}, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_geomspace(1.0, 10.0, 4, 1,
                                     TERLAN_NDARRAY_DTYPE_INT32,
                                     &result) == TERLAN_NDARRAY_OK);
  assert_ints(result, (const int64_t[]){1, 2, 4, 10}, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_geomspace(-3.0, 3.0, 4, 1,
                                     TERLAN_NDARRAY_DTYPE_FLOAT64,
                                     &result) == TERLAN_NDARRAY_OK);
  assert_floats_close(result, (const double[]){-3.0, NAN, NAN, 3.0}, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_geomspace(-3.0, 3.0, 4, 0,
                                     TERLAN_NDARRAY_DTYPE_FLOAT64,
                                     &result) == TERLAN_NDARRAY_OK);
  assert_floats_close(result, (const double[]){-3.0, NAN, NAN, NAN}, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_geomspace(0.3, 20.3, 1, 1,
                                     TERLAN_NDARRAY_DTYPE_FLOAT64,
                                     &result) == TERLAN_NDARRAY_OK);
  assert_floats(result, (const double[]){0.3}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_geomspace(0.3, 20.3, 3, 1,
                                     TERLAN_NDARRAY_DTYPE_FLOAT64,
                                     &result) == TERLAN_NDARRAY_OK);
  {
    double *values = NULL;
    size_t length = 0;
    assert(terlan_ndarray_v1_to_floats_cast(result, &values, &length) ==
           TERLAN_NDARRAY_OK);
    assert(length == 3 && values[0] == 0.3 && values[2] == 20.3);
    terlan_ndarray_v1_delete_float_values(values);
  }
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_floats(starts, 2, vector_shape, 1, &start) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(stops, 2, vector_shape, 1, &stop) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_logspace_arrays(
             start, stop, 7, 1, 0, 10.0, TERLAN_NDARRAY_DTYPE_FLOAT64,
             &result) == TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){7, 2}, 2);
  assert_floats_close(result, log_axis_zero, 14);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_logspace_arrays(
             start, stop, 7, 1, -1, 10.0, TERLAN_NDARRAY_DTYPE_FLOAT64,
             &result) == TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){2, 7}, 2);
  assert_floats_close(result, log_axis_last, 14);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(start) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(stop) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_floats((const double[]){1.0, 32.0}, 2,
                                       vector_shape, 1, &start) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats((const double[]){10000.0, 2.0}, 2,
                                       vector_shape, 1, &stop) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_geomspace_arrays(
             start, stop, 5, 1, 0, TERLAN_NDARRAY_DTYPE_FLOAT64,
             &result) == TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){5, 2}, 2);
  assert_floats_close(
      result,
      (const double[]){1.0, 32.0, 10.0, 16.0, 100.0, 8.0,
                       1000.0, 4.0, 10000.0, 2.0},
      10);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_geomspace_arrays(
             start, stop, 5, 1, -1, TERLAN_NDARRAY_DTYPE_FLOAT64,
             &result) == TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){2, 5}, 2);
  assert_floats_close(
      result,
      (const double[]){1.0, 10.0, 100.0, 1000.0, 10000.0,
                       32.0, 16.0, 8.0, 4.0, 2.0},
      10);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(start) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(stop) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_complex(
             (const double[]){0.0}, 1, (const double[]){1.0}, 1, NULL, 0,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &complex_start) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_complex(
             (const double[]){0.0}, 1, (const double[]){16.0}, 1, NULL, 0,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &complex_stop) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_geomspace_arrays(
             complex_start, complex_stop, 5, 1, 0,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &result) == TERLAN_NDARRAY_OK);
  assert_complex_close(result, (const double[]){0.0, 0.0, 0.0, 0.0, 0.0},
                       (const double[]){1.0, 2.0, 4.0, 8.0, 16.0}, 5);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(complex_start) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(complex_stop) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_complex(
             (const double[]){-1.0}, 1, (const double[]){0.0}, 1, NULL, 0,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &complex_start) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_complex(
             (const double[]){1.0}, 1, (const double[]){0.0}, 1, NULL, 0,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &complex_stop) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_geomspace_arrays(
             complex_start, complex_stop, 3, 1, 0,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &result) == TERLAN_NDARRAY_OK);
  assert_complex_close(result, (const double[]){-1.0, 0.0, 1.0},
                       (const double[]){0.0, 1.0, 0.0}, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(complex_start) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(complex_stop) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_floats((const double[]){0.0, 1.0, 2.0}, 3,
                                       (const int64_t[]){3}, 1,
                                       &incompatible) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(starts, 2, vector_shape, 1, &start) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_logspace_arrays(
             start, incompatible, 4, 1, 0, 10.0,
             TERLAN_NDARRAY_DTYPE_FLOAT64, &result) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(result == NULL);
  assert(terlan_ndarray_v1_logspace_arrays(
             start, start, 4, 1, 2, 10.0, TERLAN_NDARRAY_DTYPE_FLOAT64,
             &result) == TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(result == NULL);
  assert(terlan_ndarray_v1_logspace_arrays(
             start, start, 4, 1, 0, 0.0, TERLAN_NDARRAY_DTYPE_FLOAT64,
             &result) == TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(result == NULL);
  assert(terlan_ndarray_v1_geomspace_arrays(
             start, start, -1, 1, 0, TERLAN_NDARRAY_DTYPE_FLOAT64,
             &result) == TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(result == NULL);
  assert(terlan_ndarray_v1_delete(start) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(incompatible) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_triangular_and_diagonal_creation(void) {
  const int64_t values[] = {1, 2, 3, 4, 5, 6};
  const int64_t shape[] = {2, 3};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_tri(3, 4, 0, TERLAN_NDARRAY_DTYPE_INT64,
                               &result) == TERLAN_NDARRAY_OK);
  assert_ints(result, (const int64_t[]){1, 0, 0, 0, 1, 1, 0, 0,
                                        1, 1, 1, 0}, 12);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(values, 6, shape, 2, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_tril(source, 0, &result) == TERLAN_NDARRAY_OK);
  assert_ints(result, (const int64_t[]){1, 0, 0, 4, 5, 0}, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_triu(source, 0, &result) == TERLAN_NDARRAY_OK);
  assert_ints(result, (const int64_t[]){1, 2, 3, 0, 5, 6}, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_diagflat(source, 1, &result) == TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){7, 7}, 2);
  assert_ints(result,
              (const int64_t[]){0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
                                0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4,
                                0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0,
                                0, 0, 6, 0, 0, 0, 0, 0, 0, 0},
              49);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_triangular_numpy_dtype_rank_and_infinity_contracts(void) {
  const int32_t dtypes[] = {
      TERLAN_NDARRAY_DTYPE_BOOL,       TERLAN_NDARRAY_DTYPE_INT64,
      TERLAN_NDARRAY_DTYPE_FLOAT64,    TERLAN_NDARRAY_DTYPE_INT8,
      TERLAN_NDARRAY_DTYPE_INT16,      TERLAN_NDARRAY_DTYPE_INT32,
      TERLAN_NDARRAY_DTYPE_UINT8,      TERLAN_NDARRAY_DTYPE_UINT16,
      TERLAN_NDARRAY_DTYPE_UINT32,     TERLAN_NDARRAY_DTYPE_UINT64,
      TERLAN_NDARRAY_DTYPE_FLOAT32,    TERLAN_NDARRAY_DTYPE_COMPLEX64,
      TERLAN_NDARRAY_DTYPE_COMPLEX128, TERLAN_NDARRAY_DTYPE_FLOAT16};
  const int64_t matrix_shape[] = {2, 2};
  const int64_t cube_shape[] = {3, 2, 2};
  const int64_t cube_values[] = {1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0};
  const double infinity_values[] = {1.0, 1.0, INFINITY, 1.0, 1.0,
                                    1.0, INFINITY, 1.0, 1.0};
  const int64_t infinity_shape[] = {3, 3};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle lower = NULL;
  TerlanNdArrayHandle upper = NULL;
  size_t index;

  assert(terlan_ndarray_v1_tri(3, -1, 0, TERLAN_NDARRAY_DTYPE_FLOAT64,
                               &lower) == TERLAN_NDARRAY_OK);
  assert_shape(lower, (const int64_t[]){3, 3}, 2);
  assert_dtype(lower, TERLAN_NDARRAY_DTYPE_FLOAT64);
  assert_floats(lower, (const double[]){1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
                                        1.0, 1.0, 1.0}, 9);
  assert(terlan_ndarray_v1_delete(lower) == TERLAN_NDARRAY_OK);
  lower = NULL;
  assert(terlan_ndarray_v1_tri(3, -1, 0, TERLAN_NDARRAY_DTYPE_BOOL,
                               &lower) == TERLAN_NDARRAY_OK);
  assert_dtype(lower, TERLAN_NDARRAY_DTYPE_BOOL);
  assert_ints(lower, (const int64_t[]){1, 0, 0, 1, 1, 0, 1, 1, 1}, 9);
  assert(terlan_ndarray_v1_delete(lower) == TERLAN_NDARRAY_OK);

  for (index = 0; index < sizeof(dtypes) / sizeof(dtypes[0]); ++index) {
    assert(create_ones(dtypes[index], matrix_shape, 2, 4, &source) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_tril(source, 0, &lower) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_triu(source, 0, &upper) == TERLAN_NDARRAY_OK);
    assert_dtype(lower, dtypes[index]);
    assert_dtype(upper, dtypes[index]);
    if (dtypes[index] == TERLAN_NDARRAY_DTYPE_COMPLEX64 ||
        dtypes[index] == TERLAN_NDARRAY_DTYPE_COMPLEX128) {
      assert_complex_close(lower, (const double[]){1.0, 0.0, 1.0, 1.0},
                           (const double[]){2.0, 0.0, 2.0, 2.0}, 4);
      assert_complex_close(upper, (const double[]){1.0, 1.0, 0.0, 1.0},
                           (const double[]){2.0, 2.0, 0.0, 2.0}, 4);
      assert_complex_close(source, (const double[]){1.0, 1.0, 1.0, 1.0},
                           (const double[]){2.0, 2.0, 2.0, 2.0}, 4);
    } else {
      assert_ints(lower, (const int64_t[]){1, 0, 1, 1}, 4);
      assert_ints(upper, (const int64_t[]){1, 1, 0, 1}, 4);
      assert_ints(source, (const int64_t[]){1, 1, 1, 1}, 4);
    }
    assert(terlan_ndarray_v1_delete(upper) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(lower) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
    source = NULL;
    lower = NULL;
    upper = NULL;
  }

  assert(terlan_ndarray_v1_from_ints(cube_values, 12, cube_shape, 3,
                                     &source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_tril(source, 0, &lower) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_triu(source, 0, &upper) == TERLAN_NDARRAY_OK);
  assert_ints(lower, (const int64_t[]){1, 0, 1, 1, 1, 0, 1, 0,
                                       1, 0, 0, 0}, 12);
  assert_ints(upper, (const int64_t[]){1, 1, 0, 1, 1, 1, 0, 0,
                                       1, 1, 0, 0}, 12);
  assert_ints(source, cube_values, 12);
  assert(terlan_ndarray_v1_delete(upper) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(lower) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_floats(infinity_values, 9, infinity_shape, 2,
                                       &source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_tril(source, 0, &lower) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_triu(source, 0, &upper) == TERLAN_NDARRAY_OK);
  assert_floats_close(lower, (const double[]){1.0, 0.0, 0.0, 1.0, 1.0,
                                              0.0, INFINITY, 1.0, 1.0}, 9);
  assert_floats_close(upper, (const double[]){1.0, 1.0, INFINITY, 0.0, 1.0,
                                              1.0, 0.0, 0.0, 1.0}, 9);
  assert_floats_close(source, infinity_values, 9);
  assert(terlan_ndarray_v1_delete(upper) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(lower) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);

  lower = (TerlanNdArrayHandle)(uintptr_t)1;
  assert(terlan_ndarray_v1_tri(-1, -1, 0, TERLAN_NDARRAY_DTYPE_FLOAT64,
                               &lower) == TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(lower == NULL);
  assert(terlan_ndarray_v1_tri(1, 1, 0, TERLAN_NDARRAY_DTYPE_FLOAT64, NULL) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_coordinate_and_vandermonde_creation(void) {
  const int64_t input[] = {1, 2, 3};
  const int64_t input_shape[] = {3};
  const int64_t grid_shape[] = {2, 3};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_indices(grid_shape, 2,
                                   TERLAN_NDARRAY_DTYPE_INT64,
                                   &result) == TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){2, 2, 3}, 3);
  assert_ints(result, (const int64_t[]){0, 0, 0, 1, 1, 1,
                                        0, 1, 2, 0, 1, 2}, 12);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_indices(grid_shape, 2,
                                   TERLAN_NDARRAY_DTYPE_FLOAT32,
                                   &result) == TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){2, 2, 3}, 3);
  assert_floats_close(result, (const double[]){0, 0, 0, 1, 1, 1,
                                               0, 1, 2, 0, 1, 2}, 12);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_indices(grid_shape, 2,
                                   TERLAN_NDARRAY_DTYPE_COMPLEX128,
                                   &result) == TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){2, 2, 3}, 3);
  assert_complex_close(result,
                       (const double[]){0, 0, 0, 1, 1, 1,
                                        0, 1, 2, 0, 1, 2},
                       (const double[]){0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0},
                       12);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(input, 3, input_shape, 1, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_vander(source, 3, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, (const int64_t[]){1, 1, 1, 4, 2, 1, 9, 3, 1}, 9);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_vander(source, 3, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, (const int64_t[]){1, 1, 1, 1, 2, 4, 1, 3, 9}, 9);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_creation_extra_rejections_release_everything(void) {
  const int64_t shape[] = {2};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle result = (TerlanNdArrayHandle)(uintptr_t)1;
  assert(terlan_ndarray_v1_geomspace(0.0, 1.0, 2, 1,
                                     TERLAN_NDARRAY_DTYPE_FLOAT64,
                                     &result) == TERLAN_NDARRAY_DOMAIN_ERROR);
  assert(result == NULL);
  assert(terlan_ndarray_v1_indices(shape, 1, TERLAN_NDARRAY_DTYPE_BOOL,
                                   &result) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(result == NULL);
  assert(terlan_ndarray_v1_indices(shape, 1, 99, &result) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(result == NULL);
  assert(terlan_ndarray_v1_from_ints((const int64_t[]){1, 2}, 2, shape, 1,
                                     &source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_tril(source, 0, &result) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(result == NULL);
  assert(terlan_ndarray_v1_vander(source, -2, 0, &result) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(result == NULL);
  assert(terlan_ndarray_v1_full_like_float(source, 1.5, &result) ==
         TERLAN_NDARRAY_CAST_OVERFLOW);
  assert(result == NULL);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

int main(void) {
  test_like_constructors_preserve_shape_and_dtype();
  test_zeros_and_zeros_like_cover_all_admitted_dtypes();
  test_spaced_creation_matches_numpy();
  test_logarithmic_spaces_match_numpy_array_and_complex_semantics();
  test_triangular_and_diagonal_creation();
  test_triangular_numpy_dtype_rank_and_infinity_contracts();
  test_coordinate_and_vandermonde_creation();
  test_creation_extra_rejections_release_everything();
  return 0;
}
