#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

static void assert_shape(TerlanNdArrayHandle array, const int64_t *expected,
                         int64_t expected_rank) {
  int64_t rank = 0;
  int64_t *shape = NULL;
  int64_t axis;
  assert(terlan_ndarray_v1_rank(array, &rank) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_shape(array, &shape) == TERLAN_NDARRAY_OK);
  assert(rank == expected_rank);
  for (axis = 0; axis < rank; ++axis) {
    assert(shape[axis] == expected[axis]);
  }
}

static void assert_values(TerlanNdArrayHandle array, const int64_t *expected,
                          size_t expected_count) {
  int64_t *values = NULL;
  size_t count = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_ints_cast(array, &values, &count) ==
         TERLAN_NDARRAY_OK);
  assert(count == expected_count);
  for (index = 0; index < count; ++index) {
    assert(values[index] == expected[index]);
  }
  terlan_ndarray_v1_delete_int_values(values);
}

static TerlanNdArrayHandle int_array(const int64_t *values, size_t count,
                                     const int64_t *shape, int64_t rank) {
  TerlanNdArrayHandle array = NULL;
  assert(terlan_ndarray_v1_from_ints(values, count, shape, rank, &array) ==
         TERLAN_NDARRAY_OK);
  return array;
}

static void test_atleast_rank_promotion_matches_numpy(void) {
  TerlanNdArrayHandle scalar = int_array((const int64_t[]){7}, 1, NULL, 0);
  TerlanNdArrayHandle vector =
      int_array((const int64_t[]){1, 2, 3}, 3, (const int64_t[]){3}, 1);
  TerlanNdArrayHandle matrix = int_array(
      (const int64_t[]){1, 2, 3, 4, 5, 6}, 6,
      (const int64_t[]){2, 3}, 2);
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_atleast_1d(scalar, &result) == TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){1}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_atleast_2d(vector, &result) == TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){1, 3}, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_atleast_3d(vector, &result) == TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){1, 3, 1}, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_atleast_3d(matrix, &result) == TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){2, 3, 1}, 3);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(vector) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
}

static void test_ravel_and_flatten_use_logical_order(void) {
  TerlanNdArrayHandle source = int_array(
      (const int64_t[]){1, 2, 3, 4, 5, 6}, 6,
      (const int64_t[]){2, 3}, 2);
  TerlanNdArrayHandle transposed = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_transpose(source, 0, 1, &transposed) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_ravel(transposed, &result) == TERLAN_NDARRAY_OK);
  assert_values(result, (const int64_t[]){1, 4, 2, 5, 3, 6}, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_flatten(transposed, &result) == TERLAN_NDARRAY_OK);
  assert_values(result, (const int64_t[]){1, 4, 2, 5, 3, 6}, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(transposed) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_moveaxis_and_rollaxis_permute_strides(void) {
  const int64_t values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  const int64_t moved[] = {0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21,
                           2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23};
  TerlanNdArrayHandle source =
      int_array(values, 24, (const int64_t[]){2, 3, 4}, 3);
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_moveaxis(
             source, (const int64_t[]){0, 2}, 2,
             (const int64_t[]){2, 0}, 2, &result) == TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){4, 3, 2}, 3);
  assert_values(result, moved, 24);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_rollaxis(source, 2, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_shape(result, (const int64_t[]){4, 2, 3}, 3);
  assert_values(result,
                (const int64_t[]){0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21,
                                  2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23},
                24);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_flips_rotations_and_rejections(void) {
  TerlanNdArrayHandle source = int_array(
      (const int64_t[]){1, 2, 3, 4, 5, 6}, 6,
      (const int64_t[]){2, 3}, 2);
  TerlanNdArrayHandle vector =
      int_array((const int64_t[]){1, 2, 3, 4}, 4,
                (const int64_t[]){4}, 1);
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_flipud(source, &result) == TERLAN_NDARRAY_OK);
  assert_values(result, (const int64_t[]){4, 5, 6, 1, 2, 3}, 6);
  assert(terlan_ndarray_v1_set_int(result, (const int64_t[]){0, 0}, 2, 9) ==
         TERLAN_NDARRAY_OK);
  assert_values(source, (const int64_t[]){1, 2, 3, 9, 5, 6}, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_fliplr(source, &result) == TERLAN_NDARRAY_OK);
  assert_values(result, (const int64_t[]){3, 2, 1, 6, 5, 9}, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_fliplr(vector, &result) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(result == NULL);
  assert(terlan_ndarray_v1_flipud(NULL, &result) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(result == NULL);
  assert(terlan_ndarray_v1_flipud(source, NULL) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_rot90(source, 1, 0, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_values(result, (const int64_t[]){3, 6, 2, 5, 1, 9}, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_rot90(source, 2, 0, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_values(result, (const int64_t[]){6, 5, 9, 3, 2, 1}, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_moveaxis(source, (const int64_t[]){0, 0}, 2,
                                    (const int64_t[]){0, 1}, 2,
                                    &result) == TERLAN_NDARRAY_DUPLICATE_AXIS);
  assert(result == NULL);
  assert(terlan_ndarray_v1_rot90(source, 1, 0, 0, &result) ==
         TERLAN_NDARRAY_DUPLICATE_AXIS);
  assert(result == NULL);
  assert(terlan_ndarray_v1_delete(vector) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_flip_numpy_axes_and_rank_semantics(void) {
  const int64_t values[] = {0, 1, 2, 3, 4, 5, 6, 7};
  TerlanNdArrayHandle cube =
      int_array(values, 8, (const int64_t[]){2, 2, 2}, 3);
  TerlanNdArrayHandle result = NULL;

  assert(terlan_ndarray_v1_flip(cube, (const int64_t[]){0}, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_values(result, (const int64_t[]){4, 5, 6, 7, 0, 1, 2, 3}, 8);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_flip(cube, (const int64_t[]){1}, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_values(result, (const int64_t[]){2, 3, 0, 1, 6, 7, 4, 5}, 8);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_flip(cube, (const int64_t[]){-1}, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_values(result, (const int64_t[]){1, 0, 3, 2, 5, 4, 7, 6}, 8);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_flip(cube, NULL, 0, &result) == TERLAN_NDARRAY_OK);
  assert_values(result, values, 8);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_flip_all(cube, &result) == TERLAN_NDARRAY_OK);
  assert_values(result, (const int64_t[]){7, 6, 5, 4, 3, 2, 1, 0}, 8);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_flip(cube, (const int64_t[]){0, 2}, 2, &result) ==
         TERLAN_NDARRAY_OK);
  assert_values(result, (const int64_t[]){5, 4, 7, 6, 1, 0, 3, 2}, 8);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_flip(cube, (const int64_t[]){1, 2}, 2, &result) ==
         TERLAN_NDARRAY_OK);
  assert_values(result, (const int64_t[]){3, 2, 1, 0, 7, 6, 5, 4}, 8);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_flip(cube, (const int64_t[]){3}, 1, &result) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(result == NULL);
  assert(terlan_ndarray_v1_flip(cube, (const int64_t[]){-4}, 1, &result) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(result == NULL);
  assert(terlan_ndarray_v1_delete(cube) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_rot90_numpy_turns_axes_and_views(void) {
  const int64_t matrix_values[] = {0, 1, 2, 3, 4, 5};
  const int64_t cube_values[] = {0, 1, 2, 3, 4, 5, 6, 7};
  TerlanNdArrayHandle matrix =
      int_array(matrix_values, 6, (const int64_t[]){2, 3}, 2);
  TerlanNdArrayHandle cube =
      int_array(cube_values, 8, (const int64_t[]){2, 2, 2}, 3);
  TerlanNdArrayHandle vector =
      int_array((const int64_t[]){0, 1, 2, 3}, 4,
                (const int64_t[]){4}, 1);
  TerlanNdArrayHandle result = NULL;
  int64_t k;

  for (k = -3; k <= 9; k += 4) {
    assert(terlan_ndarray_v1_rot90(matrix, k, 0, 1, &result) ==
           TERLAN_NDARRAY_OK);
    assert_shape(result, (const int64_t[]){3, 2}, 2);
    assert_values(result, (const int64_t[]){2, 5, 1, 4, 0, 3}, 6);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  }
  assert(terlan_ndarray_v1_rot90(matrix, -2, 0, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_values(result, (const int64_t[]){5, 4, 3, 2, 1, 0}, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_rot90(matrix, -1, 0, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_values(result, (const int64_t[]){3, 0, 4, 1, 5, 2}, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_rot90(matrix, 4, 0, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_values(result, matrix_values, 6);
  assert(terlan_ndarray_v1_set_int(result, (const int64_t[]){0, 0}, 2, 9) ==
         TERLAN_NDARRAY_OK);
  assert_values(matrix, (const int64_t[]){9, 1, 2, 3, 4, 5}, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_rot90(cube, 1, 0, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_values(result, (const int64_t[]){2, 3, 6, 7, 0, 1, 4, 5}, 8);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_rot90(cube, 1, 1, 2, &result) ==
         TERLAN_NDARRAY_OK);
  assert_values(result, (const int64_t[]){1, 3, 0, 2, 5, 7, 4, 6}, 8);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_rot90(cube, 1, 2, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_values(result, (const int64_t[]){4, 0, 6, 2, 5, 1, 7, 3}, 8);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_rot90(cube, 1, 0, -1, &result) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_rot90(vector, 1, 0, 1, &result) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(result == NULL);
  assert(terlan_ndarray_v1_rot90(cube, 1, -2, 1, &result) ==
         TERLAN_NDARRAY_DUPLICATE_AXIS);
  assert(result == NULL);
  assert(terlan_ndarray_v1_rot90(cube, 1, 0, 3, &result) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(result == NULL);
  assert(terlan_ndarray_v1_delete(vector) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(cube) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

int main(void) {
  test_atleast_rank_promotion_matches_numpy();
  test_ravel_and_flatten_use_logical_order();
  test_moveaxis_and_rollaxis_permute_strides();
  test_flips_rotations_and_rejections();
  test_flip_numpy_axes_and_rank_semantics();
  test_rot90_numpy_turns_axes_and_views();
  return 0;
}
