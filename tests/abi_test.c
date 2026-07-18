#include "../bindings/include/terlan_ndarray.h"

#include <assert.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

static void test_owned_metadata(void) {
  const int64_t values[] = {1, 2, 3, 4, 5, 6};
  const int64_t dimensions[] = {2, 3};
  TerlanNdArrayHandle array = NULL;
  int64_t *shape = NULL;
  int64_t rank = -1;
  int64_t numel = -1;
  int32_t dtype = -1;
  int32_t contiguous = 0;

  assert(terlan_ndarray_v1_from_ints(values, 6, dimensions, 2, &array) ==
         TERLAN_NDARRAY_OK);
  assert(array != NULL);
  assert(terlan_ndarray_v1_live_count() == 1);
  assert(terlan_ndarray_v1_rank(array, &rank) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_numel(array, &numel) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dtype(array, &dtype) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_is_contiguous(array, &contiguous) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_shape(array, &shape) == TERLAN_NDARRAY_OK);
  assert(rank == 2);
  assert(numel == 6);
  assert(dtype == TERLAN_NDARRAY_DTYPE_INT64);
  assert(contiguous == 1);
  assert(shape != NULL && shape[0] == 2 && shape[1] == 3);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_scalar_and_empty_shapes(void) {
  const double scalar = 2.5;
  const int64_t empty_shape[] = {2, 0, 3};
  TerlanNdArrayHandle scalar_array = NULL;
  TerlanNdArrayHandle empty_array = NULL;
  int64_t rank = -1;
  int64_t numel = -1;

  assert(terlan_ndarray_v1_from_floats(&scalar, 1, NULL, 0,
                                       &scalar_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_rank(scalar_array, &rank) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_numel(scalar_array, &numel) == TERLAN_NDARRAY_OK);
  assert(rank == 0 && numel == 1);
  assert(terlan_ndarray_v1_from_ints(NULL, 0, empty_shape, 3, &empty_array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_numel(empty_array, &numel) == TERLAN_NDARRAY_OK);
  assert(numel == 0);
  assert(terlan_ndarray_v1_delete(scalar_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_rejections_leave_no_live_array(void) {
  const int64_t value = 1;
  const int64_t mismatch_shape[] = {2};
  const int64_t negative_shape[] = {-1};
  const int64_t overflow_shape[] = {INT64_MAX, 2};
  const uint8_t invalid_bool[] = {0, 2};
  TerlanNdArrayHandle array = NULL;

  assert(terlan_ndarray_v1_from_ints(&value, 1, mismatch_shape, 1, &array) ==
         TERLAN_NDARRAY_ELEMENT_COUNT_MISMATCH);
  assert(array == NULL);
  assert(terlan_ndarray_v1_from_ints(&value, 1, negative_shape, 1, &array) ==
         TERLAN_NDARRAY_NEGATIVE_DIMENSION);
  assert(array == NULL);
  assert(terlan_ndarray_v1_from_ints(&value, 1, overflow_shape, 2, &array) ==
         TERLAN_NDARRAY_SIZE_OVERFLOW);
  assert(array == NULL);
  assert(terlan_ndarray_v1_from_bools(invalid_bool, 2, mismatch_shape, 1,
                                      &array) == TERLAN_NDARRAY_INVALID_BOOL);
  assert(array == NULL);
  assert(terlan_ndarray_v1_live_count() == 0);
}

int main(void) {
  test_owned_metadata();
  test_scalar_and_empty_shapes();
  test_rejections_leave_no_live_array();
  return 0;
}
