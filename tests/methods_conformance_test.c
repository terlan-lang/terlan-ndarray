#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static const int32_t admitted_dtypes[] = {
    TERLAN_NDARRAY_DTYPE_BOOL,       TERLAN_NDARRAY_DTYPE_INT8,
    TERLAN_NDARRAY_DTYPE_INT16,      TERLAN_NDARRAY_DTYPE_INT32,
    TERLAN_NDARRAY_DTYPE_INT64,      TERLAN_NDARRAY_DTYPE_UINT8,
    TERLAN_NDARRAY_DTYPE_UINT16,     TERLAN_NDARRAY_DTYPE_UINT32,
    TERLAN_NDARRAY_DTYPE_UINT64,     TERLAN_NDARRAY_DTYPE_FLOAT16,
    TERLAN_NDARRAY_DTYPE_FLOAT32,    TERLAN_NDARRAY_DTYPE_FLOAT64,
    TERLAN_NDARRAY_DTYPE_COMPLEX64,  TERLAN_NDARRAY_DTYPE_COMPLEX128,
};

static int dtype_is_test_complex(int32_t dtype) {
  return dtype == TERLAN_NDARRAY_DTYPE_COMPLEX64 ||
         dtype == TERLAN_NDARRAY_DTYPE_COMPLEX128;
}

static int dtype_is_test_float(int32_t dtype) {
  return dtype == TERLAN_NDARRAY_DTYPE_FLOAT16 ||
         dtype == TERLAN_NDARRAY_DTYPE_FLOAT32 ||
         dtype == TERLAN_NDARRAY_DTYPE_FLOAT64;
}

static TerlanNdArrayHandle numeric_array(const int64_t *values,
                                         const int64_t *imaginary,
                                         int64_t count,
                                         const int64_t *shape, int64_t rank,
                                         int32_t dtype) {
  TerlanNdArrayHandle array = NULL;
  if (dtype == TERLAN_NDARRAY_DTYPE_BOOL) {
    uint8_t *converted = calloc((size_t)count, sizeof(*converted));
    int64_t index;
    assert(converted != NULL || count == 0);
    for (index = 0; index < count; ++index) {
      converted[index] = values[index] != 0;
    }
    assert(terlan_ndarray_v1_from_bools(converted, count, shape, rank,
                                        &array) == TERLAN_NDARRAY_OK);
    free(converted);
  } else if (dtype_is_test_float(dtype)) {
    double *converted = calloc((size_t)count, sizeof(*converted));
    int64_t index;
    assert(converted != NULL || count == 0);
    for (index = 0; index < count; ++index) {
      converted[index] = (double)values[index];
    }
    assert(terlan_ndarray_v1_from_floats_as(converted, count, shape, rank,
                                            dtype, &array) ==
           TERLAN_NDARRAY_OK);
    free(converted);
  } else if (dtype_is_test_complex(dtype)) {
    double *real = calloc((size_t)count, sizeof(*real));
    double *imag = calloc((size_t)count, sizeof(*imag));
    int64_t index;
    assert((real != NULL && imag != NULL) || count == 0);
    for (index = 0; index < count; ++index) {
      real[index] = (double)values[index];
      imag[index] = imaginary == NULL ? 0.0 : (double)imaginary[index];
    }
    assert(terlan_ndarray_v1_from_complex(real, count, imag, count, shape,
                                           rank, dtype, &array) ==
           TERLAN_NDARRAY_OK);
    free(imag);
    free(real);
  } else {
    assert(terlan_ndarray_v1_from_ints_as(values, count, shape, rank, dtype,
                                          &array) == TERLAN_NDARRAY_OK);
  }
  return array;
}

static void assert_shape(TerlanNdArrayHandle array, const int64_t *expected,
                         int64_t rank) {
  int64_t axis;
  assert(array != NULL && array->rank == rank);
  for (axis = 0; axis < rank; ++axis) {
    assert(array->shape[axis] == expected[axis]);
  }
}

static void assert_scalars(TerlanNdArrayHandle array,
                           const int64_t *expected_real,
                           const int64_t *expected_imaginary, int64_t count) {
  int64_t index;
  assert(array != NULL && array->numel == count);
  for (index = 0; index < count; ++index) {
    TerlanNdArrayScalar scalar;
    assert(dtype_load(array_const_element_pointer(array, index), array->dtype,
                      &scalar) == TERLAN_NDARRAY_OK);
    assert(fabsl(scalar.real - (long double)expected_real[index]) < 1e-5L);
    assert(fabsl(scalar.imag -
                 (long double)(expected_imaginary == NULL
                                   ? 0
                                   : expected_imaginary[index])) < 1e-5L);
  }
}

static void test_numpy_product_dtypes_and_axes(void) {
  const int32_t dtypes[] = {
      TERLAN_NDARRAY_DTYPE_INT16,     TERLAN_NDARRAY_DTYPE_UINT16,
      TERLAN_NDARRAY_DTYPE_INT32,     TERLAN_NDARRAY_DTYPE_UINT32,
      TERLAN_NDARRAY_DTYPE_FLOAT32,   TERLAN_NDARRAY_DTYPE_FLOAT64,
      TERLAN_NDARRAY_DTYPE_COMPLEX64, TERLAN_NDARRAY_DTYPE_COMPLEX128,
  };
  const int64_t vector[] = {1, 2, 10, 11, 6, 5, 4};
  const int64_t matrix[] = {1, 2, 3, 4, 5, 6, 7, 9, 10, 3, 4, 5};
  const int64_t shape[] = {3, 4};
  const int64_t axis_zero[] = {50, 36, 84, 180};
  const int64_t axis_one[] = {24, 1890, 600};
  size_t index;
  for (index = 0; index < sizeof(dtypes) / sizeof(dtypes[0]); ++index) {
    TerlanNdArrayHandle a =
        numeric_array(vector, NULL, 7, (const int64_t[]){7}, 1, dtypes[index]);
    TerlanNdArrayHandle b =
        numeric_array(matrix, NULL, 12, shape, 2, dtypes[index]);
    TerlanNdArrayHandle result = NULL;
    int64_t axis = 0;
    assert(terlan_ndarray_v1_product(a, &axis, 1, 0, &result) ==
           TERLAN_NDARRAY_OK);
    assert_scalars(result, (const int64_t[]){26400}, NULL, 1);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_product(b, &axis, 1, 0, &result) ==
           TERLAN_NDARRAY_OK);
    assert_shape(result, (const int64_t[]){4}, 1);
    assert_scalars(result, axis_zero, NULL, 4);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    axis = -1;
    assert(terlan_ndarray_v1_product(b, &axis, 1, 0, &result) ==
           TERLAN_NDARRAY_OK);
    assert_shape(result, (const int64_t[]){3}, 1);
    assert_scalars(result, axis_one, NULL, 3);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(b) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(a) == TERLAN_NDARRAY_OK);
  }
}

static void test_numpy_conjugate_all_admitted_dtypes(void) {
  const int64_t real[] = {1, 1, 23};
  const int64_t boolean_real[] = {1, 1, 1};
  const int64_t imaginary[] = {-1, 1, 23};
  const int64_t conjugated[] = {1, -1, -23};
  size_t index;
  for (index = 0;
       index < sizeof(admitted_dtypes) / sizeof(admitted_dtypes[0]); ++index) {
    const int32_t dtype = admitted_dtypes[index];
    TerlanNdArrayHandle source = numeric_array(
        real, imaginary, 3, (const int64_t[]){3}, 1, dtype);
    TerlanNdArrayHandle result = NULL;
    const int64_t *expected_real =
        dtype == TERLAN_NDARRAY_DTYPE_BOOL ? boolean_real : real;
    assert(terlan_ndarray_v1_conjugate(source, &result) ==
           TERLAN_NDARRAY_OK);
    assert(result->dtype == dtype);
    assert_scalars(result, expected_real,
                   dtype_is_test_complex(dtype) ? conjugated : NULL, 3);
    assert_scalars(source, expected_real,
                   dtype_is_test_complex(dtype) ? imaginary : NULL, 3);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
  }
}

static void test_numpy_fill_mutates_all_admitted_dtypes_and_views(void) {
  const int64_t zeros[] = {0, 0, 0, 0, 0, 0};
  const int64_t expected[] = {7, 7, 7, 7, 7, 7};
  const int64_t expected_bool[] = {1, 1, 1, 1, 1, 1};
  const int64_t expected_imaginary[] = {-3, -3, -3, -3, -3, -3};
  size_t index;
  for (index = 0;
       index < sizeof(admitted_dtypes) / sizeof(admitted_dtypes[0]); ++index) {
    const int32_t dtype = admitted_dtypes[index];
    TerlanNdArrayHandle target = numeric_array(
        zeros, NULL, 6, (const int64_t[]){3, 2, 1}, 3, dtype);
    TerlanNdArrayHandle scalar = numeric_array(
        (const int64_t[]){7}, (const int64_t[]){-3}, 1, NULL, 0, dtype);
    assert(terlan_ndarray_v1_fill(target, scalar) == TERLAN_NDARRAY_OK);
    assert_scalars(target,
                   dtype == TERLAN_NDARRAY_DTYPE_BOOL ? expected_bool
                                                      : expected,
                   dtype_is_test_complex(dtype) ? expected_imaginary : NULL,
                   6);
    assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(target) == TERLAN_NDARRAY_OK);
  }
  {
    TerlanNdArrayHandle target = numeric_array(
        zeros, NULL, 6, (const int64_t[]){3, 2, 1}, 3,
        TERLAN_NDARRAY_DTYPE_UINT64);
    TerlanNdArrayHandle scalar = numeric_array(
        (const int64_t[]){0}, NULL, 1, NULL, 0,
        TERLAN_NDARRAY_DTYPE_UINT64);
    TerlanNdArrayScalar maximum = {
        .real = (long double)UINT64_MAX,
        .imag = 0.0L,
    };
    int64_t element;
    assert(dtype_store(array_element_pointer(scalar, 0), scalar->dtype,
                       maximum) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_fill(target, scalar) == TERLAN_NDARRAY_OK);
    for (element = 0; element < target->numel; ++element) {
      TerlanNdArrayScalar actual;
      assert(dtype_load(array_const_element_pointer(target, element),
                        target->dtype, &actual) == TERLAN_NDARRAY_OK);
      assert(actual.real == (long double)UINT64_MAX);
    }
    assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(target) == TERLAN_NDARRAY_OK);
  }
  {
    TerlanNdArrayHandle base = numeric_array(
        (const int64_t[]){0, 1, 2, 3, 4, 5}, NULL, 6,
        (const int64_t[]){6}, 1, TERLAN_NDARRAY_DTYPE_INT64);
    TerlanNdArrayHandle scalar = numeric_array(
        (const int64_t[]){9}, NULL, 1, NULL, 0,
        TERLAN_NDARRAY_DTYPE_FLOAT64);
    TerlanNdArrayHandle values = numeric_array(
        (const int64_t[]){3, 4}, NULL, 2, (const int64_t[]){2}, 1,
        TERLAN_NDARRAY_DTYPE_INT64);
    TerlanNdArrayHandle view = NULL;
    TerlanNdArrayHandle read_only = NULL;
    assert(terlan_ndarray_v1_slice_axis(base, 0, 0, 6, 2, &view) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_fill(view, scalar) == TERLAN_NDARRAY_OK);
    assert_scalars(base, (const int64_t[]){9, 1, 9, 3, 9, 5}, NULL, 6);
    assert(terlan_ndarray_v1_fill(base, values) ==
           TERLAN_NDARRAY_INVALID_ARGUMENT);
    assert_scalars(base, (const int64_t[]){9, 1, 9, 3, 9, 5}, NULL, 6);
    assert(terlan_ndarray_v1_broadcast_to(
               scalar, (const int64_t[]){4}, 1, &read_only) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_fill(read_only, scalar) ==
           TERLAN_NDARRAY_READ_ONLY);
    assert_scalars(read_only, (const int64_t[]){9, 9, 9, 9}, NULL, 4);
    assert(terlan_ndarray_v1_delete(read_only) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(view) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(values) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(base) == TERLAN_NDARRAY_OK);
  }
}

static void test_numpy_putmask_all_admitted_numeric_dtypes(void) {
  const int64_t real[] = {0, 2, 0};
  const int64_t imaginary[] = {1, 2, 3};
  const int64_t expected[] = {7, 2, 7};
  const int64_t expected_bool[] = {1, 1, 1};
  const int64_t expected_imaginary[] = {-3, 2, -3};
  const uint8_t truth[] = {1, 0, 1};
  size_t index;
  for (index = 0;
       index < sizeof(admitted_dtypes) / sizeof(admitted_dtypes[0]); ++index) {
    const int32_t dtype = admitted_dtypes[index];
    TerlanNdArrayHandle target = numeric_array(
        real, imaginary, 3, (const int64_t[]){3}, 1, dtype);
    TerlanNdArrayHandle replacement = numeric_array(
        (const int64_t[]){7}, (const int64_t[]){-3}, 1, NULL, 0, dtype);
    TerlanNdArrayHandle mask = NULL;
    assert(terlan_ndarray_v1_from_bools(truth, 3, (const int64_t[]){3}, 1,
                                        &mask) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_putmask(target, mask, replacement) ==
           TERLAN_NDARRAY_OK);
    assert_scalars(target,
                   dtype == TERLAN_NDARRAY_DTYPE_BOOL ? expected_bool
                                                      : expected,
                   dtype_is_test_complex(dtype) ? expected_imaginary : NULL,
                   3);
    assert(terlan_ndarray_v1_delete(mask) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(replacement) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(target) == TERLAN_NDARRAY_OK);
  }
}

static void test_numpy_diagonal_offsets_axes_and_lifecycle(void) {
  const int64_t matrix_values[] = {0, 1, 2, 3, 4, 5,
                                   6, 7, 8, 9, 10, 11};
  const int64_t cube_values[] = {0, 1, 2, 3, 4, 5, 6, 7};
  TerlanNdArrayHandle matrix = numeric_array(
      matrix_values, NULL, 12, (const int64_t[]){3, 4}, 2,
      TERLAN_NDARRAY_DTYPE_INT64);
  TerlanNdArrayHandle cube = numeric_array(
      cube_values, NULL, 8, (const int64_t[]){2, 2, 2}, 3,
      TERLAN_NDARRAY_DTYPE_INT64);
  TerlanNdArrayHandle diagonal = NULL;
  int64_t iteration;
  assert(terlan_ndarray_v1_diagonal(matrix, 0, 0, 1, &diagonal) ==
         TERLAN_NDARRAY_OK);
  assert_scalars(diagonal, (const int64_t[]){0, 5, 10}, NULL, 3);
  assert(terlan_ndarray_v1_delete(diagonal) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_diagonal(matrix, 1, 0, 1, &diagonal) ==
         TERLAN_NDARRAY_OK);
  assert_scalars(diagonal, (const int64_t[]){1, 6, 11}, NULL, 3);
  assert(terlan_ndarray_v1_delete(diagonal) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_diagonal(matrix, -1, 0, 1, &diagonal) ==
         TERLAN_NDARRAY_OK);
  assert_scalars(diagonal, (const int64_t[]){4, 9}, NULL, 2);
  assert(terlan_ndarray_v1_delete(diagonal) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_diagonal(cube, 0, 1, 2, &diagonal) ==
         TERLAN_NDARRAY_OK);
  assert_shape(diagonal, (const int64_t[]){2, 2}, 2);
  assert_scalars(diagonal, (const int64_t[]){0, 3, 4, 7}, NULL, 4);
  assert(terlan_ndarray_v1_delete(diagonal) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_diagonal(cube, 0, 2, 1, &diagonal) ==
         TERLAN_NDARRAY_OK);
  assert_scalars(diagonal, (const int64_t[]){0, 3, 4, 7}, NULL, 4);
  assert(terlan_ndarray_v1_delete(diagonal) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_diagonal(cube, 1, 0, 2, &diagonal) ==
         TERLAN_NDARRAY_OK);
  assert_scalars(diagonal, (const int64_t[]){1, 3}, NULL, 2);
  assert(terlan_ndarray_v1_delete(diagonal) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_diagonal(cube, 0, 0, 5, &diagonal) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(terlan_ndarray_v1_diagonal(cube, 0, 1, 1, &diagonal) ==
         TERLAN_NDARRAY_DUPLICATE_AXIS);
  for (iteration = 0; iteration < 100; ++iteration) {
    assert(terlan_ndarray_v1_diagonal(matrix, 0, 0, 1, &diagonal) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(diagonal) == TERLAN_NDARRAY_OK);
  }
  assert(terlan_ndarray_v1_delete(cube) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(matrix) == TERLAN_NDARRAY_OK);
}

static void test_numpy_put_all_admitted_dtypes_and_rejections(void) {
  const int64_t zeros[] = {0, 0, 0, 0, 0, 0};
  const int64_t values[] = {1, 3, 5};
  const int64_t expected[] = {0, 1, 0, 3, 0, 5};
  const int64_t bool_expected[] = {0, 1, 0, 1, 0, 1};
  const int64_t indices[] = {1, 3, 5};
  size_t index;
  for (index = 0;
       index < sizeof(admitted_dtypes) / sizeof(admitted_dtypes[0]); ++index) {
    const int32_t dtype = admitted_dtypes[index];
    TerlanNdArrayHandle vector = numeric_array(
        zeros, NULL, 6, (const int64_t[]){6}, 1, dtype);
    TerlanNdArrayHandle matrix = numeric_array(
        zeros, NULL, 6, (const int64_t[]){2, 3}, 2, dtype);
    TerlanNdArrayHandle replacements = numeric_array(
        values, NULL, 3, (const int64_t[]){3}, 1, dtype);
    const int64_t *wanted =
        dtype == TERLAN_NDARRAY_DTYPE_BOOL ? bool_expected : expected;
    assert(terlan_ndarray_v1_put(vector, indices, 3, replacements) ==
           TERLAN_NDARRAY_OK);
    assert_scalars(vector, wanted, NULL, 6);
    assert(terlan_ndarray_v1_put(matrix, indices, 3, replacements) ==
           TERLAN_NDARRAY_OK);
    assert_scalars(matrix, wanted, NULL, 6);
    vector->writable = 0;
    assert(terlan_ndarray_v1_put(vector, indices, 3, replacements) ==
           TERLAN_NDARRAY_READ_ONLY);
    vector->writable = 1;
    assert(terlan_ndarray_v1_put(vector, (const int64_t[]){1, 6, 3}, 3,
                                 replacements) ==
           TERLAN_NDARRAY_INDEX_OUT_OF_RANGE);
    assert_scalars(vector, wanted, NULL, 6);
    assert(terlan_ndarray_v1_delete(replacements) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(matrix) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(vector) == TERLAN_NDARRAY_OK);
  }
}

static void test_numpy_squeeze_and_transpose_methods(void) {
  TerlanNdArrayHandle source = numeric_array(
      (const int64_t[]){1, 2, 3}, NULL, 3,
      (const int64_t[]){1, 3, 1}, 3, TERLAN_NDARRAY_DTYPE_INT64);
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayHandle matrix = numeric_array(
      (const int64_t[]){1, 2, 3, 4}, NULL, 4,
      (const int64_t[]){2, 2}, 2, TERLAN_NDARRAY_DTYPE_INT64);
  assert(terlan_ndarray_v1_squeeze(source, NULL, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){3}, 1);
  assert_scalars(result, (const int64_t[]){1, 2, 3}, NULL, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_squeeze(source, (const int64_t[]){0}, 1,
                                   &result) == TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){3, 1}, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_squeeze(source, (const int64_t[]){1}, 1,
                                   &result) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(terlan_ndarray_v1_squeeze(source, (const int64_t[]){2}, 1,
                                   &result) == TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){1, 3}, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_transpose(matrix, 0, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_scalars(result, (const int64_t[]){1, 3, 2, 4}, NULL, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_transpose(matrix, 0, 2, &result) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(result == NULL);
  assert(terlan_ndarray_v1_delete(matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_numpy_ordered_reshape_flatten_and_ravel(void) {
  const int64_t matrix_values[] = {1, 2, 3, 4, 5, 6,
                                   7, 8, 9, 10, 11, 12};
  TerlanNdArrayHandle matrix = numeric_array(
      matrix_values, NULL, 12, (const int64_t[]){4, 3}, 2,
      TERLAN_NDARRAY_DTYPE_INT64);
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayHandle transposed = NULL;
  TerlanNdArrayHandle cube = numeric_array(
      (const int64_t[]){1, 2, 3, 4, 5, 6, 7, 8}, NULL, 8,
      (const int64_t[]){2, 2, 2}, 3, TERLAN_NDARRAY_DTYPE_INT32);
  TerlanNdArrayHandle small = numeric_array(
      (const int64_t[]){0, 1, 2, 3}, NULL, 4,
      (const int64_t[]){2, 2}, 2, TERLAN_NDARRAY_DTYPE_INT64);
  TerlanNdArrayHandle fortran = NULL;
  TerlanNdArrayHandle reversed = NULL;
  TerlanNdArrayHandle vector = numeric_array(
      (const int64_t[]){0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, NULL, 10,
      (const int64_t[]){10}, 1, TERLAN_NDARRAY_DTYPE_INT64);
  TerlanNdArrayHandle strided = NULL;
  TerlanNdArrayHandle scalar = numeric_array(
      (const int64_t[]){0}, NULL, 1, NULL, 0,
      TERLAN_NDARRAY_DTYPE_INT64);
  int32_t order;

  assert(terlan_ndarray_v1_reshape_ordered(
             matrix, (const int64_t[]){2, 6}, 2, TERLAN_NDARRAY_ORDER_C,
             &result) == TERLAN_NDARRAY_OK);
  assert_scalars(result, matrix_values, NULL, 12);
  assert(result->storage == matrix->storage);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_reshape_ordered(
             matrix, (const int64_t[]){3, 4}, 2,
             TERLAN_NDARRAY_ORDER_FORTRAN, &result) == TERLAN_NDARRAY_OK);
  assert_scalars(result,
                 (const int64_t[]){1, 10, 8, 6, 4, 2,
                                   11, 9, 7, 5, 3, 12},
                 NULL, 12);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_transpose(matrix, 0, 1, &transposed) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_reshape_ordered(
             transposed, (const int64_t[]){3, 4}, 2,
             TERLAN_NDARRAY_ORDER_C, &result) == TERLAN_NDARRAY_OK);
  assert_scalars(result,
                 (const int64_t[]){1, 4, 7, 10, 2, 5,
                                   8, 11, 3, 6, 9, 12},
                 NULL, 12);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_flatten_ordered(
             small, TERLAN_NDARRAY_ORDER_C, &result) == TERLAN_NDARRAY_OK);
  assert_scalars(result, (const int64_t[]){0, 1, 2, 3}, NULL, 4);
  assert(result->storage != small->storage);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_flatten_ordered(
             small, TERLAN_NDARRAY_ORDER_FORTRAN, &result) ==
         TERLAN_NDARRAY_OK);
  assert_scalars(result, (const int64_t[]){0, 2, 1, 3}, NULL, 4);
  assert(result->storage != small->storage);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_flatten_ordered(
             cube, TERLAN_NDARRAY_ORDER_FORTRAN, &result) ==
         TERLAN_NDARRAY_OK);
  assert_scalars(result, (const int64_t[]){1, 5, 3, 7, 2, 6, 4, 8}, NULL,
                 8);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_copy_ordered(small, TERLAN_NDARRAY_ORDER_FORTRAN,
                                        &fortran) == TERLAN_NDARRAY_OK);
  for (order = TERLAN_NDARRAY_ORDER_C; order <= TERLAN_NDARRAY_ORDER_ANY;
       ++order) {
    const int64_t *expected =
        order == TERLAN_NDARRAY_ORDER_FORTRAN
                ? (const int64_t[]){0, 2, 1, 3}
                : (const int64_t[]){0, 1, 2, 3};
    assert(terlan_ndarray_v1_ravel_ordered(small, order, &result) ==
           TERLAN_NDARRAY_OK);
    assert_scalars(result, expected, NULL, 4);
    if (order != TERLAN_NDARRAY_ORDER_FORTRAN) {
      assert(result->storage == small->storage);
    }
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  }
  assert(terlan_ndarray_v1_ravel_ordered(fortran, TERLAN_NDARRAY_ORDER_C,
                                         &result) == TERLAN_NDARRAY_OK);
  assert_scalars(result, (const int64_t[]){0, 1, 2, 3}, NULL, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  for (order = TERLAN_NDARRAY_ORDER_FORTRAN;
       order <= TERLAN_NDARRAY_ORDER_ANY; ++order) {
    assert(terlan_ndarray_v1_ravel_ordered(fortran, order, &result) ==
           TERLAN_NDARRAY_OK);
    assert_scalars(result, (const int64_t[]){0, 2, 1, 3}, NULL, 4);
    assert(result->storage == fortran->storage);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  }

  assert(terlan_ndarray_v1_flip(small, (const int64_t[]){0}, 1, &reversed) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_ravel_ordered(reversed, TERLAN_NDARRAY_ORDER_C,
                                         &result) == TERLAN_NDARRAY_OK);
  assert_scalars(result, (const int64_t[]){2, 3, 0, 1}, NULL, 4);
  assert(result->storage != reversed->storage);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_ravel_ordered(
             reversed, TERLAN_NDARRAY_ORDER_FORTRAN, &result) ==
         TERLAN_NDARRAY_OK);
  assert_scalars(result, (const int64_t[]){2, 0, 3, 1}, NULL, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  for (order = TERLAN_NDARRAY_ORDER_KEEP;
       order <= TERLAN_NDARRAY_ORDER_ANY; ++order) {
    assert(terlan_ndarray_v1_ravel_ordered(reversed, order, &result) ==
           TERLAN_NDARRAY_OK);
    assert_scalars(result, (const int64_t[]){2, 3, 0, 1}, NULL, 4);
    assert(result->storage != reversed->storage);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  }
  assert(terlan_ndarray_v1_slice_axis(vector, 0, 0, 10, 2, &strided) ==
         TERLAN_NDARRAY_OK);
  for (order = TERLAN_NDARRAY_ORDER_C; order <= TERLAN_NDARRAY_ORDER_ANY;
       ++order) {
    assert(terlan_ndarray_v1_ravel_ordered(strided, order, &result) ==
           TERLAN_NDARRAY_OK);
    assert_scalars(result, (const int64_t[]){0, 2, 4, 6, 8}, NULL, 5);
    assert(result->storage != strided->storage);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_ravel_ordered(scalar, order, &result) ==
           TERLAN_NDARRAY_OK);
    assert_scalars(result, (const int64_t[]){0}, NULL, 1);
    assert(result->storage == scalar->storage);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  }
  assert(terlan_ndarray_v1_reshape_ordered(
             matrix, (const int64_t[]){3, 4}, 2,
             TERLAN_NDARRAY_ORDER_KEEP, &result) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_ravel_ordered(matrix, 7, &result) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(result == NULL);

  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(strided) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(vector) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(reversed) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(fortran) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(small) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(cube) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(transposed) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(matrix) == TERLAN_NDARRAY_OK);
}

static void assert_transpose_values(TerlanNdArrayHandle source,
                                    TerlanNdArrayHandle transposed,
                                    int64_t axis_a, int64_t axis_b) {
  int64_t output_logical;
  for (output_logical = 0; output_logical < transposed->numel;
       ++output_logical) {
    int64_t coordinates[4];
    int64_t source_logical = 0;
    int64_t remainder = output_logical;
    int64_t axis;
    TerlanNdArrayScalar source_value;
    TerlanNdArrayScalar transposed_value;
    for (axis = transposed->rank; axis > 0; --axis) {
      const int64_t selected = axis - 1;
      coordinates[selected] = remainder % transposed->shape[selected];
      remainder /= transposed->shape[selected];
    }
    {
      const int64_t temporary = coordinates[axis_a];
      coordinates[axis_a] = coordinates[axis_b];
      coordinates[axis_b] = temporary;
    }
    for (axis = 0; axis < source->rank; ++axis) {
      source_logical = source_logical * source->shape[axis] + coordinates[axis];
    }
    assert(dtype_load(array_const_element_pointer(source, source_logical),
                      source->dtype, &source_value) == TERLAN_NDARRAY_OK);
    assert(dtype_load(array_const_element_pointer(transposed, output_logical),
                      transposed->dtype, &transposed_value) ==
           TERLAN_NDARRAY_OK);
    assert(source_value.real == transposed_value.real);
    assert(source_value.imag == transposed_value.imag);
  }
}

static void test_numpy_swapaxes_exhaustive_views(void) {
  int64_t values[24];
  TerlanNdArrayHandle source;
  TerlanNdArrayHandle noncontiguous;
  TerlanNdArrayHandle invalid = NULL;
  int64_t index;
  int64_t axis_a;
  int64_t axis_b;
  for (index = 0; index < 24; ++index) {
    values[index] = index;
  }
  source = numeric_array(values, NULL, 24, (const int64_t[]){1, 2, 3, 4}, 4,
                         TERLAN_NDARRAY_DTYPE_INT64);
  assert(terlan_ndarray_v1_copy(source, &noncontiguous) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_transpose(source, -5, 0, &invalid) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(invalid == NULL);
  for (axis_a = -4; axis_a < 4; ++axis_a) {
    for (axis_b = -4; axis_b < 4; ++axis_b) {
      const int64_t normalized_a = axis_a < 0 ? axis_a + 4 : axis_a;
      const int64_t normalized_b = axis_b < 0 ? axis_b + 4 : axis_b;
      TerlanNdArrayHandle result = NULL;
      TerlanNdArrayHandle chained = NULL;
      assert(terlan_ndarray_v1_transpose(source, axis_a, axis_b, &result) ==
             TERLAN_NDARRAY_OK);
      assert_transpose_values(source, result, normalized_a, normalized_b);
      assert(result->storage == source->storage);
      assert(terlan_ndarray_v1_transpose(noncontiguous, axis_a, axis_b,
                                         &chained) == TERLAN_NDARRAY_OK);
      assert_transpose_values(noncontiguous, chained, normalized_a,
                              normalized_b);
      assert(chained->storage == noncontiguous->storage);
      assert(terlan_ndarray_v1_delete(noncontiguous) == TERLAN_NDARRAY_OK);
      noncontiguous = chained;
      assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    }
  }
  assert(terlan_ndarray_v1_delete(noncontiguous) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_numpy_arr_mult_and_dot_overlap_dtypes(void) {
  const int32_t multiplication_dtypes[] = {
      TERLAN_NDARRAY_DTYPE_FLOAT32, TERLAN_NDARRAY_DTYPE_FLOAT64,
      TERLAN_NDARRAY_DTYPE_COMPLEX64, TERLAN_NDARRAY_DTYPE_COMPLEX128,
  };
  const int64_t matrix[] = {0, 1, 2, 3, 4, 5, 6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  const int64_t expected[] = {55,  145, 235, 325, 145, 451, 757, 1063,
                              235, 757, 1279, 1801, 325, 1063, 1801, 2539};
  const int64_t reversed_expected[] = {
      325, 1063, 1801, 2539, 235, 757, 1279, 1801,
      145, 451,  757,  1063, 55,  145, 235,  325,
  };
  size_t index;
  for (index = 0; index < sizeof(multiplication_dtypes) /
                                  sizeof(multiplication_dtypes[0]);
       ++index) {
    TerlanNdArrayHandle lhs = numeric_array(
        matrix, NULL, 24, (const int64_t[]){4, 6}, 2,
        multiplication_dtypes[index]);
    TerlanNdArrayHandle rhs = NULL;
    TerlanNdArrayHandle reversed = NULL;
    TerlanNdArrayHandle result = NULL;
    assert(terlan_ndarray_v1_transpose(lhs, 0, 1, &rhs) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_dot(lhs, rhs, &result) == TERLAN_NDARRAY_OK);
    assert_scalars(result, expected, NULL, 16);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_matmul(lhs, rhs, &result) == TERLAN_NDARRAY_OK);
    assert_scalars(result, expected, NULL, 16);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_flip(lhs, (const int64_t[]){0}, 1, &reversed) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_dot(reversed, rhs, &result) ==
           TERLAN_NDARRAY_OK);
    assert_scalars(result, reversed_expected, NULL, 16);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(reversed) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(rhs) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(lhs) == TERLAN_NDARRAY_OK);
  }
  for (index = 0;
       index < sizeof(admitted_dtypes) / sizeof(admitted_dtypes[0]); ++index) {
    TerlanNdArrayHandle identity = numeric_array(
        (const int64_t[]){1, 0, 0, 1}, NULL, 4,
        (const int64_t[]){2, 2}, 2, admitted_dtypes[index]);
    TerlanNdArrayHandle destination = numeric_array(
        (const int64_t[]){0, 1, 1, 0}, NULL, 4,
        (const int64_t[]){2, 2}, 2, admitted_dtypes[index]);
    assert(terlan_ndarray_v1_dot_into(destination, identity, destination) ==
           TERLAN_NDARRAY_OK);
    assert_scalars(destination, (const int64_t[]){0, 1, 1, 0}, NULL, 4);
    assert(terlan_ndarray_v1_delete(destination) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(identity) == TERLAN_NDARRAY_OK);
  }
}

static void test_numpy_broadcast_vector_multiplication(void) {
  const int32_t dtypes[] = {
      TERLAN_NDARRAY_DTYPE_INT32, TERLAN_NDARRAY_DTYPE_FLOAT32,
      TERLAN_NDARRAY_DTYPE_FLOAT64, TERLAN_NDARRAY_DTYPE_COMPLEX64,
      TERLAN_NDARRAY_DTYPE_COMPLEX128,
  };
  const int64_t values[] = {0, 1, 2, 3, 4, 5, 6, 7};
  const int64_t expected[] = {6, 22};
  size_t index;
  for (index = 0; index < sizeof(dtypes) / sizeof(dtypes[0]); ++index) {
    TerlanNdArrayHandle matrix = numeric_array(
        values, NULL, 8, (const int64_t[]){2, 4}, 2, dtypes[index]);
    TerlanNdArrayHandle scalar = numeric_array(
        (const int64_t[]){1}, NULL, 1, NULL, 0, dtypes[index]);
    TerlanNdArrayHandle broadcast = NULL;
    TerlanNdArrayHandle copied = NULL;
    TerlanNdArrayHandle result = NULL;
    TerlanNdArrayHandle reference = NULL;
    TerlanNdArrayHandle matrix_transpose = NULL;
    TerlanNdArrayHandle broadcast_transpose = NULL;
    assert(terlan_ndarray_v1_broadcast_to(
               scalar, (const int64_t[]){4, 1}, 2, &broadcast) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_copy(broadcast, &copied) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_matmul(matrix, broadcast, &result) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_matmul(matrix, copied, &reference) ==
           TERLAN_NDARRAY_OK);
    assert_scalars(result, expected, NULL, 2);
    assert_scalars(reference, expected, NULL, 2);
    assert(terlan_ndarray_v1_delete(reference) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_transpose(matrix, 0, 1, &matrix_transpose) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_transpose(broadcast, 0, 1,
                                       &broadcast_transpose) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_dot(broadcast_transpose, matrix_transpose,
                                 &result) == TERLAN_NDARRAY_OK);
    assert_scalars(result, expected, NULL, 2);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(broadcast_transpose) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(matrix_transpose) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(copied) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(broadcast) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(matrix) == TERLAN_NDARRAY_OK);
  }
}

static void test_numpy_zero_dot_lifecycle(void) {
  TerlanNdArrayHandle empty = numeric_array(
      NULL, NULL, 0, (const int64_t[]){0}, 1, TERLAN_NDARRAY_DTYPE_FLOAT64);
  TerlanNdArrayHandle destination = numeric_array(
      (const int64_t[]){2}, NULL, 1, NULL, 0,
      TERLAN_NDARRAY_DTYPE_FLOAT64);
  int64_t iteration;
  for (iteration = 0; iteration < 100; ++iteration) {
    assert(terlan_ndarray_v1_dot_into(destination, empty, empty) ==
           TERLAN_NDARRAY_OK);
    assert_scalars(destination, (const int64_t[]){0}, NULL, 1);
  }
  assert(terlan_ndarray_v1_delete(destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty) == TERLAN_NDARRAY_OK);
}

static void test_numpy_degraded_sort_dataset(void) {
  const int64_t count = 1000000;
  int64_t *values = malloc((size_t)count * sizeof(*values));
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle sorted = NULL;
  TerlanNdArrayHandle order = NULL;
  int64_t active = count;
  int64_t index;
  assert(values != NULL);
  for (index = 0; index < count; ++index) {
    values[index] = index;
  }
  while (active > 3) {
    const int64_t middle = active / 2;
    const int64_t temporary = values[middle];
    values[middle] = values[active - 2];
    values[active - 2] = temporary;
    active -= 2;
  }
  assert(terlan_ndarray_v1_from_ints(values, count,
                                     (const int64_t[]){count}, 1, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sort(source, 0, &sorted) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argsort(source, 0, &order) == TERLAN_NDARRAY_OK);
  for (index = 0; index < count; ++index) {
    int64_t selected;
    TerlanNdArrayScalar value;
    assert(dtype_load(array_const_element_pointer(sorted, index),
                      sorted->dtype, &value) == TERLAN_NDARRAY_OK);
    assert(value.real == (long double)index);
    assert(dtype_load(array_const_element_pointer(order, index), order->dtype,
                      &value) == TERLAN_NDARRAY_OK);
    selected = (int64_t)value.real;
    assert(values[selected] == index);
  }
  assert(terlan_ndarray_v1_delete(order) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(sorted) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
  free(values);
}

int main(void) {
  test_numpy_product_dtypes_and_axes();
  test_numpy_conjugate_all_admitted_dtypes();
  test_numpy_fill_mutates_all_admitted_dtypes_and_views();
  test_numpy_putmask_all_admitted_numeric_dtypes();
  test_numpy_diagonal_offsets_axes_and_lifecycle();
  test_numpy_put_all_admitted_dtypes_and_rejections();
  test_numpy_squeeze_and_transpose_methods();
  test_numpy_ordered_reshape_flatten_and_ravel();
  test_numpy_swapaxes_exhaustive_views();
  test_numpy_arr_mult_and_dot_overlap_dtypes();
  test_numpy_broadcast_vector_multiplication();
  test_numpy_zero_dot_lifecycle();
  test_numpy_degraded_sort_dataset();
  assert(terlan_ndarray_v1_live_count() == 0);
  return 0;
}
