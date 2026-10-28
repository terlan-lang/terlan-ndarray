#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static void assert_scalar_values(TerlanNdArrayHandle array,
                                 const long double *expected_real,
                                 const long double *expected_imag,
                                 size_t count) {
  size_t index;
  assert(array != NULL);
  assert((size_t)array->numel == count);
  for (index = 0; index < count; ++index) {
    TerlanNdArrayScalar value;
    assert(dtype_load(array_const_element_pointer(array, (int64_t)index),
                      array->dtype, &value) == TERLAN_NDARRAY_OK);
    assert(fabsl(value.real - expected_real[index]) < 1e-6L);
    assert(fabsl(value.imag - expected_imag[index]) < 1e-6L);
  }
}

static void test_numpy_where_covers_all_admitted_dtypes_and_scalars(void) {
  const uint8_t predicates[] = {1, 0, 1, 0, 1};
  const int64_t shape[] = {5};
  const int64_t true_integers[] = {1, 2, 3, 4, 5};
  const int64_t false_integers[] = {9, 8, 7, 6, 5};
  const double true_floats[] = {1.0, 2.0, 3.0, 4.0, 5.0};
  const double false_floats[] = {9.0, 8.0, 7.0, 6.0, 5.0};
  const double true_imag[] = {10.0, 20.0, 30.0, 40.0, 50.0};
  const double false_imag[] = {90.0, 80.0, 70.0, 60.0, 50.0};
  const long double expected_real[] = {1.0L, 8.0L, 3.0L, 6.0L, 5.0L};
  const long double expected_imag[] = {10.0L, 80.0L, 30.0L, 60.0L, 50.0L};
  TerlanNdArrayHandle condition = NULL;
  TerlanNdArrayHandle when_true = NULL;
  TerlanNdArrayHandle when_false = NULL;
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayHandle scalar = NULL;
  size_t index;

  assert(terlan_ndarray_v1_from_bools(predicates, 5, shape, 1, &condition) ==
         TERLAN_NDARRAY_OK);
  for (index = 0;
       index < sizeof((int32_t[]){TERLAN_NDARRAY_DTYPE_INT8,
                                  TERLAN_NDARRAY_DTYPE_INT16,
                                  TERLAN_NDARRAY_DTYPE_INT32,
                                  TERLAN_NDARRAY_DTYPE_INT64,
                                  TERLAN_NDARRAY_DTYPE_UINT8,
                                  TERLAN_NDARRAY_DTYPE_UINT16,
                                  TERLAN_NDARRAY_DTYPE_UINT32,
                                  TERLAN_NDARRAY_DTYPE_UINT64}) /
                   sizeof(int32_t);
       ++index) {
    const int32_t dtypes[] = {
        TERLAN_NDARRAY_DTYPE_INT8,   TERLAN_NDARRAY_DTYPE_INT16,
        TERLAN_NDARRAY_DTYPE_INT32,  TERLAN_NDARRAY_DTYPE_INT64,
        TERLAN_NDARRAY_DTYPE_UINT8,  TERLAN_NDARRAY_DTYPE_UINT16,
        TERLAN_NDARRAY_DTYPE_UINT32, TERLAN_NDARRAY_DTYPE_UINT64,
    };
    assert(terlan_ndarray_v1_from_ints_as(true_integers, 5, shape, 1,
                                          dtypes[index], &when_true) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_from_ints_as(false_integers, 5, shape, 1,
                                          dtypes[index], &when_false) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_where(condition, when_true, when_false,
                                   &result) == TERLAN_NDARRAY_OK);
    assert(result->dtype == dtypes[index]);
    assert_scalar_values(result, expected_real,
                         (const long double[]){0, 0, 0, 0, 0}, 5);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(when_false) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(when_true) == TERLAN_NDARRAY_OK);
  }

  for (index = 0; index < 3; ++index) {
    const int32_t dtypes[] = {TERLAN_NDARRAY_DTYPE_FLOAT16,
                              TERLAN_NDARRAY_DTYPE_FLOAT32,
                              TERLAN_NDARRAY_DTYPE_FLOAT64};
    assert(terlan_ndarray_v1_from_floats_as(true_floats, 5, shape, 1,
                                            dtypes[index], &when_true) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_from_floats_as(false_floats, 5, shape, 1,
                                            dtypes[index], &when_false) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_where(condition, when_true, when_false,
                                   &result) == TERLAN_NDARRAY_OK);
    assert(result->dtype == dtypes[index]);
    assert_scalar_values(result, expected_real,
                         (const long double[]){0, 0, 0, 0, 0}, 5);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(when_false) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(when_true) == TERLAN_NDARRAY_OK);
  }

  for (index = 0; index < 2; ++index) {
    const int32_t dtypes[] = {TERLAN_NDARRAY_DTYPE_COMPLEX64,
                              TERLAN_NDARRAY_DTYPE_COMPLEX128};
    assert(terlan_ndarray_v1_from_complex(
               true_floats, 5, true_imag, 5, shape, 1, dtypes[index],
               &when_true) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_from_complex(
               false_floats, 5, false_imag, 5, shape, 1, dtypes[index],
               &when_false) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_where(condition, when_true, when_false,
                                   &result) == TERLAN_NDARRAY_OK);
    assert(result->dtype == dtypes[index]);
    assert_scalar_values(result, expected_real, expected_imag, 5);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(when_false) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(when_true) == TERLAN_NDARRAY_OK);
  }

  assert(terlan_ndarray_v1_from_bools((const uint8_t[]){1}, 1, NULL, 0,
                                      &scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(true_integers, 5, shape, 1, &when_true) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(false_integers, 5, shape, 1,
                                     &when_false) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_where(scalar, when_true, when_false, &result) ==
         TERLAN_NDARRAY_OK);
  assert_scalar_values(result, (const long double[]){1, 2, 3, 4, 5},
                       (const long double[]){0, 0, 0, 0, 0}, 5);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(when_false) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(when_true) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(condition) == TERLAN_NDARRAY_OK);
}

static void test_numpy_where_preserves_strided_and_empty_shapes(void) {
  const uint8_t predicates[] = {1, 0, 0, 1, 1, 0, 0, 1, 1, 0};
  const int64_t true_values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const int64_t false_values[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
  TerlanNdArrayHandle condition = NULL;
  TerlanNdArrayHandle when_true = NULL;
  TerlanNdArrayHandle when_false = NULL;
  TerlanNdArrayHandle condition_view = NULL;
  TerlanNdArrayHandle true_view = NULL;
  TerlanNdArrayHandle false_view = NULL;
  TerlanNdArrayHandle result = NULL;

  assert(terlan_ndarray_v1_from_bools(predicates, 10,
                                      (const int64_t[]){10}, 1, &condition) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(true_values, 10,
                                     (const int64_t[]){10}, 1, &when_true) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(false_values, 10,
                                     (const int64_t[]){10}, 1, &when_false) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_slice_axis(condition, 0, 0, 10, 2,
                                      &condition_view) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_slice_axis(when_true, 0, 0, 10, 2, &true_view) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_slice_axis(when_false, 0, 0, 10, 2, &false_view) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_where(condition_view, true_view, false_view,
                                 &result) == TERLAN_NDARRAY_OK);
  assert_scalar_values(result, (const long double[]){0, 12, 4, 16, 8},
                       (const long double[]){0, 0, 0, 0, 0}, 5);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(false_view) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(true_view) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(condition_view) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(when_false) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(when_true) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(condition) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_bools(NULL, 0, (const int64_t[]){0, 3}, 2,
                                      &condition) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(NULL, 0, (const int64_t[]){0, 3}, 2,
                                       &when_true) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(NULL, 0, (const int64_t[]){0, 3}, 2,
                                       &when_false) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_where(condition, when_true, when_false, &result) ==
         TERLAN_NDARRAY_OK);
  assert(result->rank == 2 && result->shape[0] == 0 && result->shape[1] == 3);
  assert(result->numel == 0);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(when_false) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(when_true) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(condition) == TERLAN_NDARRAY_OK);
}

static void test_numpy_where_basic_matches_all_dtype_scalar_and_stride_cases(
    void) {
  test_numpy_where_covers_all_admitted_dtypes_and_scalars();
  test_numpy_where_preserves_strided_and_empty_shapes();
}

int main(void) {
  test_numpy_where_basic_matches_all_dtype_scalar_and_stride_cases();
  assert(terlan_ndarray_v1_live_count() == 0);
  return 0;
}
