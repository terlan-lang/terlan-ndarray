#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static const int32_t numerical_dtypes[] = {
    TERLAN_NDARRAY_DTYPE_BOOL,       TERLAN_NDARRAY_DTYPE_INT8,
    TERLAN_NDARRAY_DTYPE_INT16,      TERLAN_NDARRAY_DTYPE_INT32,
    TERLAN_NDARRAY_DTYPE_INT64,      TERLAN_NDARRAY_DTYPE_UINT8,
    TERLAN_NDARRAY_DTYPE_UINT16,     TERLAN_NDARRAY_DTYPE_UINT32,
    TERLAN_NDARRAY_DTYPE_UINT64,     TERLAN_NDARRAY_DTYPE_FLOAT16,
    TERLAN_NDARRAY_DTYPE_FLOAT32,    TERLAN_NDARRAY_DTYPE_FLOAT64,
    TERLAN_NDARRAY_DTYPE_COMPLEX64,  TERLAN_NDARRAY_DTYPE_COMPLEX128,
};

static int is_float_dtype(int32_t dtype) {
  return dtype == TERLAN_NDARRAY_DTYPE_FLOAT16 ||
         dtype == TERLAN_NDARRAY_DTYPE_FLOAT32 ||
         dtype == TERLAN_NDARRAY_DTYPE_FLOAT64;
}

static int is_complex_dtype(int32_t dtype) {
  return dtype == TERLAN_NDARRAY_DTYPE_COMPLEX64 ||
         dtype == TERLAN_NDARRAY_DTYPE_COMPLEX128;
}

static TerlanNdArrayHandle numeric_array(const int64_t *real,
                                         const int64_t *imaginary,
                                         int64_t count,
                                         const int64_t *shape, int64_t rank,
                                         int32_t dtype) {
  TerlanNdArrayHandle output = NULL;
  int64_t index;
  if (dtype == TERLAN_NDARRAY_DTYPE_BOOL) {
    uint8_t *values = calloc((size_t)count, sizeof(*values));
    assert(values != NULL || count == 0);
    for (index = 0; index < count; ++index) {
      values[index] = real[index] != 0;
    }
    assert(terlan_ndarray_v1_from_bools(values, count, shape, rank, &output) ==
           TERLAN_NDARRAY_OK);
    free(values);
  } else if (is_float_dtype(dtype)) {
    double *values = calloc((size_t)count, sizeof(*values));
    assert(values != NULL || count == 0);
    for (index = 0; index < count; ++index) {
      values[index] = (double)real[index];
    }
    assert(terlan_ndarray_v1_from_floats_as(values, count, shape, rank, dtype,
                                            &output) == TERLAN_NDARRAY_OK);
    free(values);
  } else if (is_complex_dtype(dtype)) {
    double *real_values = calloc((size_t)count, sizeof(*real_values));
    double *imaginary_values = calloc((size_t)count,
                                      sizeof(*imaginary_values));
    assert((real_values != NULL && imaginary_values != NULL) || count == 0);
    for (index = 0; index < count; ++index) {
      real_values[index] = (double)real[index];
      imaginary_values[index] = imaginary == NULL ? 0.0 : imaginary[index];
    }
    assert(terlan_ndarray_v1_from_complex(
               real_values, count, imaginary_values, count, shape, rank,
               dtype, &output) == TERLAN_NDARRAY_OK);
    free(imaginary_values);
    free(real_values);
  } else {
    assert(terlan_ndarray_v1_from_ints_as(real, count, shape, rank, dtype,
                                          &output) == TERLAN_NDARRAY_OK);
  }
  return output;
}

static int64_t wrapped_expected(int64_t value, int32_t dtype) {
  int64_t modulus;
  int64_t wrapped;
  if (dtype == TERLAN_NDARRAY_DTYPE_BOOL) {
    return value != 0;
  }
  if (dtype == TERLAN_NDARRAY_DTYPE_INT8 ||
      dtype == TERLAN_NDARRAY_DTYPE_UINT8) {
    modulus = 256;
  } else if (dtype == TERLAN_NDARRAY_DTYPE_INT16 ||
             dtype == TERLAN_NDARRAY_DTYPE_UINT16) {
    modulus = 65536;
  } else {
    return value;
  }
  wrapped = value % modulus;
  if (wrapped < 0) {
    wrapped += modulus;
  }
  if ((dtype == TERLAN_NDARRAY_DTYPE_INT8 ||
       dtype == TERLAN_NDARRAY_DTYPE_INT16) &&
      wrapped >= modulus / 2) {
    wrapped -= modulus;
  }
  return wrapped;
}

static void assert_values(TerlanNdArrayHandle array, const int64_t *expected,
                          int64_t count, int32_t dtype) {
  int64_t index;
  assert(array != NULL && array->numel == count && array->dtype == dtype);
  for (index = 0; index < count; ++index) {
    TerlanNdArrayScalar actual;
    assert(dtype_load(array_const_element_pointer(array, index), array->dtype,
                      &actual) == TERLAN_NDARRAY_OK);
    assert(actual.real == (long double)wrapped_expected(expected[index], dtype));
    assert(actual.imag == 0.0L);
  }
}

static void test_numpy_inner_scalar_vector_and_contiguous_views(void) {
  const int64_t matrix_values[] = {1, 2, 3, 4};
  const int64_t rows_values[] = {1, 3, 2, 4};
  const int64_t vector_values[] = {1, 1};
  size_t dtype_index;
  for (dtype_index = 0;
       dtype_index < sizeof(numerical_dtypes) / sizeof(numerical_dtypes[0]);
       ++dtype_index) {
    const int32_t dtype = numerical_dtypes[dtype_index];
    TerlanNdArrayHandle scalar = numeric_array(
        (const int64_t[]){3}, NULL, 1, NULL, 0, dtype);
    TerlanNdArrayHandle pair = numeric_array(
        (const int64_t[]){1, 2}, NULL, 2, (const int64_t[]){2}, 1, dtype);
    TerlanNdArrayHandle matrix = numeric_array(
        matrix_values, NULL, 4, (const int64_t[]){2, 2}, 2, dtype);
    TerlanNdArrayHandle rows = numeric_array(
        rows_values, NULL, 4, (const int64_t[]){2, 2}, 2, dtype);
    TerlanNdArrayHandle vector = numeric_array(
        vector_values, NULL, 2, (const int64_t[]){2}, 1, dtype);
    TerlanNdArrayHandle transposed = NULL;
    TerlanNdArrayHandle sequence = numeric_array(
        (const int64_t[]){0, 1, 2, 3, 4}, NULL, 5,
        (const int64_t[]){5}, 1, dtype);
    TerlanNdArrayHandle reversed = NULL;
    TerlanNdArrayHandle output = NULL;
    assert(terlan_ndarray_v1_inner(pair, scalar, &output) == TERLAN_NDARRAY_OK);
    assert_values(output, (const int64_t[]){3, 6}, 2, dtype);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_inner(scalar, pair, &output) == TERLAN_NDARRAY_OK);
    assert_values(output, (const int64_t[]){3, 6}, 2, dtype);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_transpose(matrix, 0, 1, &transposed) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_inner(transposed, vector, &output) ==
           TERLAN_NDARRAY_OK);
    assert_values(output, (const int64_t[]){4, 6}, 2, dtype);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_inner(matrix, rows, &output) == TERLAN_NDARRAY_OK);
    assert_values(output, (const int64_t[]){7, 10, 15, 22}, 4, dtype);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_inner(matrix, matrix, &output) ==
           TERLAN_NDARRAY_OK);
    assert_values(output, (const int64_t[]){5, 11, 11, 25}, 4, dtype);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_slice_axis(sequence, 0, INT64_MIN, INT64_MIN, -1,
                                        &reversed) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_inner(reversed, sequence, &output) ==
           TERLAN_NDARRAY_OK);
    assert_values(output, (const int64_t[]){10}, 1, dtype);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(reversed) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(sequence) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(transposed) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(vector) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(rows) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(matrix) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(pair) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
  }
}

static void test_numpy_inner_three_dimensional_dtype_and_shape_contract(void) {
  const int64_t expected[] = {
      158, 182, 206, 230, 254, 278, 566, 654, 742,
      830, 918, 1006, 974, 1126, 1278, 1430, 1582, 1734,
      1382, 1598, 1814, 2030, 2246, 2462, 1790, 2070, 2350,
      2630, 2910, 3190, 2198, 2542, 2886, 3230, 3574, 3918,
  };
  int64_t lhs_values[24];
  int64_t rhs_values[24];
  size_t dtype_index;
  int64_t index;
  for (index = 0; index < 24; ++index) {
    lhs_values[index] = index;
    rhs_values[index] = index + 24;
  }
  for (dtype_index = 0;
       dtype_index < sizeof(numerical_dtypes) / sizeof(numerical_dtypes[0]);
       ++dtype_index) {
    const int32_t dtype = numerical_dtypes[dtype_index];
    TerlanNdArrayHandle lhs = numeric_array(
        lhs_values, NULL, 24, (const int64_t[]){2, 3, 4}, 3, dtype);
    TerlanNdArrayHandle rhs = numeric_array(
        rhs_values, NULL, 24, (const int64_t[]){2, 3, 4}, 3, dtype);
    TerlanNdArrayHandle output = NULL;
    assert(terlan_ndarray_v1_inner(lhs, rhs, &output) == TERLAN_NDARRAY_OK);
    assert(output->rank == 4 && output->shape[0] == 2 &&
           output->shape[1] == 3 && output->shape[2] == 2 &&
           output->shape[3] == 3);
    assert_values(output, expected, 36, dtype);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(rhs) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(lhs) == TERLAN_NDARRAY_OK);
  }
}

static void test_numpy_vdot_dtype_order_conjugation_and_strides(void) {
  const int64_t identity[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  size_t dtype_index;
  for (dtype_index = 0;
       dtype_index < sizeof(numerical_dtypes) / sizeof(numerical_dtypes[0]);
       ++dtype_index) {
    const int32_t dtype = numerical_dtypes[dtype_index];
    const int64_t imaginary[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const int64_t zeros[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    TerlanNdArrayHandle value = numeric_array(
        is_complex_dtype(dtype) ? zeros : identity,
        is_complex_dtype(dtype) ? imaginary : NULL, 9,
        (const int64_t[]){3, 3}, 2, dtype);
    TerlanNdArrayHandle output = NULL;
    assert(terlan_ndarray_v1_vdot(value, value, &output) == TERLAN_NDARRAY_OK);
    assert_values(output,
                  dtype == TERLAN_NDARRAY_DTYPE_BOOL
                      ? (const int64_t[]){1}
                      : (const int64_t[]){3},
                  1, dtype);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(value) == TERLAN_NDARRAY_OK);
  }
  {
    TerlanNdArrayHandle c_order = numeric_array(
        (const int64_t[]){1, 2, 3, 4}, NULL, 4,
        (const int64_t[]){2, 2}, 2, TERLAN_NDARRAY_DTYPE_INT64);
    TerlanNdArrayHandle f_storage = numeric_array(
        (const int64_t[]){1, 3, 2, 4}, NULL, 4,
        (const int64_t[]){2, 2}, 2, TERLAN_NDARRAY_DTYPE_INT64);
    TerlanNdArrayHandle f_order = NULL;
    TerlanNdArrayHandle output = NULL;
    assert(terlan_ndarray_v1_transpose(f_storage, 0, 1, &f_order) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_vdot(c_order, f_order, &output) ==
           TERLAN_NDARRAY_OK);
    assert_values(output, (const int64_t[]){30}, 1,
                  TERLAN_NDARRAY_DTYPE_INT64);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_vdot(f_order, c_order, &output) ==
           TERLAN_NDARRAY_OK);
    assert_values(output, (const int64_t[]){30}, 1,
                  TERLAN_NDARRAY_DTYPE_INT64);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(f_order) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(f_storage) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(c_order) == TERLAN_NDARRAY_OK);
  }
}

static void test_numpy_vdot_large_uncontiguous_views(void) {
  const int64_t sizes[] = {2, 1000};
  size_t size_index;
  for (size_index = 0; size_index < 2; ++size_index) {
    const int64_t size = sizes[size_index];
    int64_t *lhs_values = calloc((size_t)size * 4, sizeof(*lhs_values));
    int64_t *rhs_values = calloc((size_t)size * 4, sizeof(*rhs_values));
    TerlanNdArrayHandle lhs;
    TerlanNdArrayHandle rhs;
    TerlanNdArrayHandle lhs_view = NULL;
    TerlanNdArrayHandle rhs_view = NULL;
    TerlanNdArrayHandle output = NULL;
    TerlanNdArrayScalar actual;
    long double expected = 0.0L;
    int64_t index;
    assert(lhs_values != NULL && rhs_values != NULL);
    for (index = 0; index < size; ++index) {
      lhs_values[index * 4] = index;
      rhs_values[index * 4] = index + 1;
      expected += (long double)index * (long double)(index + 1);
    }
    lhs = numeric_array(lhs_values, NULL, size * 4,
                        (const int64_t[]){size, 2, 2}, 3,
                        TERLAN_NDARRAY_DTYPE_FLOAT64);
    rhs = numeric_array(rhs_values, NULL, size * 4,
                        (const int64_t[]){size, 2, 2}, 3,
                        TERLAN_NDARRAY_DTYPE_FLOAT64);
    assert(terlan_ndarray_v1_index_axis(lhs, 2, 0, &lhs_view) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_index_axis(rhs, 2, 0, &rhs_view) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_vdot(lhs_view, rhs_view, &output) ==
           TERLAN_NDARRAY_OK);
    assert(dtype_load(array_const_element_pointer(output, 0), output->dtype,
                      &actual) == TERLAN_NDARRAY_OK);
    assert(actual.real == expected && actual.imag == 0.0L);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(rhs_view) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(lhs_view) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(rhs) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(lhs) == TERLAN_NDARRAY_OK);
    free(rhs_values);
    free(lhs_values);
  }
}

static void test_numpy_inner_vdot_rejections_are_atomic(void) {
  TerlanNdArrayHandle pair = numeric_array(
      (const int64_t[]){1, 2}, NULL, 2, (const int64_t[]){2}, 1,
      TERLAN_NDARRAY_DTYPE_FLOAT64);
  TerlanNdArrayHandle triple = numeric_array(
      (const int64_t[]){1, 2, 3}, NULL, 3, (const int64_t[]){3}, 1,
      TERLAN_NDARRAY_DTYPE_FLOAT64);
  TerlanNdArrayHandle output = (TerlanNdArrayHandle)(uintptr_t)1;
  assert(terlan_ndarray_v1_inner(pair, triple, &output) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(output == NULL);
  output = (TerlanNdArrayHandle)(uintptr_t)1;
  assert(terlan_ndarray_v1_vdot(pair, triple, &output) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(output == NULL);
  assert(terlan_ndarray_v1_inner(pair, pair, NULL) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_vdot(pair, pair, NULL) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_delete(triple) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(pair) == TERLAN_NDARRAY_OK);
}

int main(void) {
  test_numpy_inner_scalar_vector_and_contiguous_views();
  test_numpy_inner_three_dimensional_dtype_and_shape_contract();
  test_numpy_vdot_dtype_order_conjugation_and_strides();
  test_numpy_vdot_large_uncontiguous_views();
  test_numpy_inner_vdot_rejections_are_atomic();
  assert(terlan_ndarray_v1_live_count() == 0);
  return 0;
}
