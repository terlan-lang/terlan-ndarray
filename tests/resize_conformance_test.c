#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

static void assert_shape(TerlanNdArrayHandle array, const int64_t *expected,
                         int64_t rank) {
  int64_t axis;
  assert(array->rank == rank);
  for (axis = 0; axis < rank; ++axis) {
    assert(array->shape[axis] == expected[axis]);
  }
}

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

static void test_numpy_resize_growth_shrink_and_freeform_shapes(void) {
  const int64_t identity[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const int64_t grown[] = {
      1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  TerlanNdArrayHandle array = NULL;

  assert(terlan_ndarray_v1_from_ints(identity, 9,
                                     (const int64_t[]){3, 3}, 2,
                                     &array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_resize_in_place(array,
                                            (const int64_t[]){5, 5}, 2) ==
         TERLAN_NDARRAY_OK);
  assert_shape(array, (const int64_t[]){5, 5}, 2);
  assert_ints(array, grown, 25);
  assert(terlan_ndarray_v1_resize_in_place(array,
                                            (const int64_t[]){3}, 1) ==
         TERLAN_NDARRAY_OK);
  assert_shape(array, (const int64_t[]){3}, 1);
  assert_ints(array, (const int64_t[]){1, 0, 0}, 3);
  assert(terlan_ndarray_v1_resize_in_place(
             array, (const int64_t[]){3, 2, 1}, 3) == TERLAN_NDARRAY_OK);
  assert_shape(array, (const int64_t[]){3, 2, 1}, 3);
  assert_ints(array, (const int64_t[]){1, 0, 0, 0, 0, 0}, 6);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_numpy_resize_rank_zero_and_appended_zero_contract(void) {
  TerlanNdArrayHandle array = NULL;
  int64_t iteration;
  for (iteration = 0; iteration < 10; ++iteration) {
    assert(terlan_ndarray_v1_from_ints((const int64_t[]){iteration}, 1,
                                       (const int64_t[]){1}, 1,
                                       &array) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_resize_in_place(array, NULL, 0) ==
           TERLAN_NDARRAY_OK);
    assert_shape(array, NULL, 0);
    assert_ints(array, &iteration, 1);
    assert(terlan_ndarray_v1_resize_in_place(
               array, (const int64_t[]){1}, 1) == TERLAN_NDARRAY_OK);
    assert_shape(array, (const int64_t[]){1}, 1);
    assert_ints(array, &iteration, 1);
    assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  }
  assert(terlan_ndarray_v1_from_ints(
             (const int64_t[]){1, 0, 0, 0, 1, 0, 0, 0, 1}, 9,
             (const int64_t[]){3, 3}, 2, &array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_resize_in_place(
             array, (const int64_t[]){2, 3, 3}, 3) == TERLAN_NDARRAY_OK);
  assert_ints(array,
              (const int64_t[]){1, 0, 0, 0, 1, 0, 0, 0, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 0},
              18);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_numpy_resize_alias_empty_and_atomic_rejections(void) {
  const int64_t values[] = {1, 2, 3, 4};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle alias = NULL;
  TerlanNdArrayHandle empty = NULL;
  TerlanNdArrayHandle empty_view = NULL;

  assert(terlan_ndarray_v1_from_ints(values, 4,
                                     (const int64_t[]){2, 2}, 2,
                                     &source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_reshape(source, (const int64_t[]){4}, 1,
                                   &alias) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_resize_in_place(source,
                                            (const int64_t[]){5, 1}, 2) ==
         TERLAN_NDARRAY_UNSUPPORTED_LAYOUT);
  assert_shape(source, (const int64_t[]){2, 2}, 2);
  assert_ints(source, values, 4);
  assert(terlan_ndarray_v1_resize_in_place(source,
                                            (const int64_t[]){-1}, 1) ==
         TERLAN_NDARRAY_NEGATIVE_DIMENSION);
  assert_shape(source, (const int64_t[]){2, 2}, 2);
  assert_ints(source, values, 4);
  assert(terlan_ndarray_v1_delete(alias) == TERLAN_NDARRAY_OK);
  terlan_ndarray_test_fail_allocation_after(0);
  assert(terlan_ndarray_v1_resize_in_place(source,
                                            (const int64_t[]){3, 2}, 2) ==
         TERLAN_NDARRAY_ALLOCATION_FAILED);
  terlan_ndarray_test_reset_allocation_failures();
  assert_shape(source, (const int64_t[]){2, 2}, 2);
  assert_ints(source, values, 4);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_ints(NULL, 0,
                                     (const int64_t[]){10, 0}, 2,
                                     &empty) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_reshape(empty, (const int64_t[]){0, 10}, 2,
                                   &empty_view) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_resize_in_place(
             empty_view, (const int64_t[]){0, 100}, 2) == TERLAN_NDARRAY_OK);
  assert_shape(empty_view, (const int64_t[]){0, 100}, 2);
  assert(terlan_ndarray_v1_delete(empty_view) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_resize_in_place(NULL, NULL, 0) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_live_count() == 0);
}

int main(void) {
  test_numpy_resize_growth_shrink_and_freeform_shapes();
  test_numpy_resize_rank_zero_and_appended_zero_contract();
  test_numpy_resize_alias_empty_and_atomic_rejections();
  return 0;
}
