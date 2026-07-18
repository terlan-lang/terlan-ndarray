#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static void test_owned_metadata_and_int_readback(void) {
  const int64_t values[] = {1, 2, 3, 4, 5, 6};
  const int64_t dimensions[] = {2, 3};
  TerlanNdArrayHandle array = NULL;
  int64_t *shape = NULL;
  int64_t *strides = NULL;
  int64_t *copied = NULL;
  size_t copied_length = 0;
  int64_t rank = -1;
  int64_t numel = -1;
  int64_t byte_count = -1;
  int32_t dtype = -1;
  int32_t device_type = -1;
  int32_t contiguous = 0;
  size_t index;

  assert(terlan_ndarray_v1_from_ints(values, 6, dimensions, 2, &array) ==
         TERLAN_NDARRAY_OK);
  assert(array != NULL);
  assert(array->state == TERLAN_NDARRAY_STATE_LIVE);
  assert(terlan_ndarray_v1_live_count() == 1);
  assert(terlan_ndarray_v1_rank(array, &rank) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_numel(array, &numel) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_byte_count(array, &byte_count) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dtype(array, &dtype) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_device_type(array, &device_type) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_is_contiguous(array, &contiguous) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_shape(array, &shape) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_strides(array, &strides) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_to_ints(array, &copied, &copied_length) ==
         TERLAN_NDARRAY_OK);
  assert(rank == 2);
  assert(numel == 6);
  assert(byte_count == 6 * (int64_t)sizeof(int64_t));
  assert(dtype == TERLAN_NDARRAY_DTYPE_INT64);
  assert(device_type == TERLAN_NDARRAY_DEVICE_CPU);
  assert(contiguous == 1);
  assert(shape != NULL && shape[0] == 2 && shape[1] == 3);
  assert(strides != NULL && strides[0] == 3 && strides[1] == 1);
  assert(copied != NULL && copied_length == 6);
  for (index = 0; index < copied_length; ++index) {
    assert(copied[index] == values[index]);
  }
  terlan_ndarray_v1_delete_int_values(copied);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_all_dtype_readback(void) {
  const uint8_t bools[] = {1, 0, 1};
  const int64_t integers[] = {INT64_MIN, 0, INT64_MAX};
  const double floats[] = {NAN, -INFINITY, INFINITY};
  const int64_t shape[] = {3};
  TerlanNdArrayHandle bool_array = NULL;
  TerlanNdArrayHandle int_array = NULL;
  TerlanNdArrayHandle float_array = NULL;
  uint8_t *copied_bools = NULL;
  int64_t *copied_ints = NULL;
  double *copied_floats = NULL;
  size_t length = 0;

  assert(terlan_ndarray_v1_from_bools(bools, 3, shape, 1, &bool_array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(integers, 3, shape, 1, &int_array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(floats, 3, shape, 1, &float_array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_to_bools(bool_array, &copied_bools, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == 3 && copied_bools[0] == 1 && copied_bools[1] == 0 &&
         copied_bools[2] == 1);
  assert(terlan_ndarray_v1_to_ints(int_array, &copied_ints, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == 3 && copied_ints[0] == INT64_MIN && copied_ints[1] == 0 &&
         copied_ints[2] == INT64_MAX);
  assert(terlan_ndarray_v1_to_floats(float_array, &copied_floats, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == 3 && isnan(copied_floats[0]) &&
         copied_floats[1] == -INFINITY && copied_floats[2] == INFINITY);
  terlan_ndarray_v1_delete_bool_values(copied_bools);
  terlan_ndarray_v1_delete_int_values(copied_ints);
  terlan_ndarray_v1_delete_float_values(copied_floats);
  assert(terlan_ndarray_v1_delete(bool_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(int_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(float_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_scalar_empty_and_high_rank_shapes(void) {
  const double scalar = 2.5;
  const int64_t empty_shape[] = {2, 0, 3};
  int64_t high_rank_shape[64];
  TerlanNdArrayHandle scalar_array = NULL;
  TerlanNdArrayHandle empty_array = NULL;
  TerlanNdArrayHandle high_rank_array = NULL;
  int64_t *strides = NULL;
  double *copied = NULL;
  size_t copied_length = 1;
  int64_t rank = -1;
  int64_t numel = -1;
  size_t index;

  for (index = 0; index < 64; ++index) {
    high_rank_shape[index] = 1;
  }
  assert(terlan_ndarray_v1_from_floats(&scalar, 1, NULL, 0,
                                       &scalar_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_rank(scalar_array, &rank) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_numel(scalar_array, &numel) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_strides(scalar_array, &strides) ==
         TERLAN_NDARRAY_OK);
  assert(rank == 0 && numel == 1 && strides == NULL);
  assert(terlan_ndarray_v1_from_ints(NULL, 0, empty_shape, 3, &empty_array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_numel(empty_array, &numel) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_strides(empty_array, &strides) ==
         TERLAN_NDARRAY_OK);
  assert(numel == 0 && strides[0] == 0 && strides[1] == 3 && strides[2] == 1);
  assert(terlan_ndarray_v1_to_floats(empty_array, &copied, &copied_length) ==
         TERLAN_NDARRAY_DTYPE_MISMATCH);
  assert(copied == NULL && copied_length == 0);
  assert(terlan_ndarray_v1_from_floats(&scalar, 1, high_rank_shape, 64,
                                       &high_rank_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_rank(high_rank_array, &rank) == TERLAN_NDARRAY_OK);
  assert(rank == 64);
  assert(terlan_ndarray_v1_strides(high_rank_array, &strides) ==
         TERLAN_NDARRAY_OK);
  for (index = 0; index < 64; ++index) {
    assert(strides[index] == 1);
  }
  assert(terlan_ndarray_v1_delete(scalar_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(high_rank_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_rejections_leave_no_live_array(void) {
  const int64_t value = 1;
  const int64_t mismatch_shape[] = {2};
  const int64_t negative_shape[] = {-1};
  const int64_t overflow_shape[] = {INT64_MAX, 2};
  const int64_t zero_prefix_overflow_shape[] = {0, INT64_MAX, 2};
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
  assert(terlan_ndarray_v1_from_ints(NULL, 0, zero_prefix_overflow_shape, 3,
                                     &array) == TERLAN_NDARRAY_SIZE_OVERFLOW);
  assert(array == NULL);
  assert(terlan_ndarray_v1_from_bools(invalid_bool, 2, mismatch_shape, 1,
                                      &array) == TERLAN_NDARRAY_INVALID_BOOL);
  assert(array == NULL);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_allocation_failures_release_partial_state(void) {
  const int64_t values[] = {1, 2, 3, 4, 5, 6};
  const int64_t dimensions[] = {2, 3};
  TerlanNdArrayHandle array = NULL;
  int64_t *copied = NULL;
  size_t copied_length = 0;
  int64_t fail_after;

  for (fail_after = 0; fail_after < 4; ++fail_after) {
    terlan_ndarray_test_fail_allocation_after(fail_after);
    assert(terlan_ndarray_v1_from_ints(values, 6, dimensions, 2, &array) ==
           TERLAN_NDARRAY_ALLOCATION_FAILED);
    assert(array == NULL);
    assert(terlan_ndarray_v1_live_count() == 0);
    terlan_ndarray_test_reset_allocation_failures();
  }
  assert(terlan_ndarray_v1_from_ints(values, 6, dimensions, 2, &array) ==
         TERLAN_NDARRAY_OK);
  terlan_ndarray_test_fail_allocation_after(0);
  assert(terlan_ndarray_v1_to_ints(array, &copied, &copied_length) ==
         TERLAN_NDARRAY_ALLOCATION_FAILED);
  assert(copied == NULL && copied_length == 0);
  terlan_ndarray_test_reset_allocation_failures();
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

int main(void) {
  test_owned_metadata_and_int_readback();
  test_all_dtype_readback();
  test_scalar_empty_and_high_rank_shapes();
  test_rejections_leave_no_live_array();
  test_allocation_failures_release_partial_state();
  return 0;
}
