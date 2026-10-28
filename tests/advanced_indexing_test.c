#include "../bindings/include/terlan_ndarray.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static void assert_ints(TerlanNdArrayHandle array, const int64_t *expected,
                        size_t count) {
  int64_t *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_ints_cast(array, &values, &length) == 0);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(values[index] == expected[index]);
  }
  free(values);
}

static void assert_bools(TerlanNdArrayHandle array, const uint8_t *expected,
                         size_t count) {
  uint8_t *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_bools(array, &values, &length) == 0);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(values[index] == expected[index]);
  }
  free(values);
}

static void assert_floats(TerlanNdArrayHandle array, const double *expected,
                          size_t count) {
  double *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_floats_cast(array, &values, &length) == 0);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(fabs(values[index] - expected[index]) <= 1e-12);
  }
  free(values);
}

static TerlanNdArrayHandle ints(const int64_t *values, int64_t count,
                                const int64_t *shape, int64_t rank) {
  TerlanNdArrayHandle array = NULL;
  assert(terlan_ndarray_v1_from_ints(values, count, shape, rank, &array) == 0);
  return array;
}

static TerlanNdArrayHandle floats(const double *values, int64_t count,
                                  const int64_t *shape, int64_t rank) {
  TerlanNdArrayHandle array = NULL;
  assert(terlan_ndarray_v1_from_floats(values, count, shape, rank, &array) ==
         0);
  return array;
}

static void test_take_put_and_coordinate_gather(void) {
  const int64_t source_values[] = {10, 11, 12, 20, 21, 22};
  const int64_t shape[] = {2, 3};
  const int64_t index_values[] = {2, 1, 0, 0, 1, 2};
  const int64_t expected_take[] = {12, 11, 10, 20, 21, 22};
  const int64_t coordinate_values[] = {0, 2, -1, 0};
  const int64_t coordinate_shape[] = {2, 2};
  const int64_t expected_gather[] = {12, 20};
  const int64_t axis_replacements[] = {100, 101, 102, 200, 201, 202};
  const int64_t expected_put_axis[] = {102, 101, 100, 200, 201, 202};
  const int64_t replacements[] = {90, 91};
  const int64_t flat_indexes[] = {0, -1};
  const int64_t expected_put[] = {90, 11, 12, 20, 21, 91};
  TerlanNdArrayHandle source = ints(source_values, 6, shape, 2);
  TerlanNdArrayHandle indices = ints(index_values, 6, shape, 2);
  TerlanNdArrayHandle output = NULL;
  TerlanNdArrayHandle coordinates =
      ints(coordinate_values, 4, coordinate_shape, 2);
  TerlanNdArrayHandle values = ints(replacements, 2, (int64_t[]){2}, 1);
  TerlanNdArrayHandle axis_values = ints(axis_replacements, 6, shape, 2);
  TerlanNdArrayHandle axis_target = ints(source_values, 6, shape, 2);
  assert(terlan_ndarray_v1_take_along_axis(source, indices, 1, &output) == 0);
  assert_ints(output, expected_take, 6);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_gather_coordinates(source, coordinates, &output) ==
         0);
  assert_ints(output, expected_gather, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_put_along_axis(axis_target, indices, axis_values, 1) ==
         0);
  assert_ints(axis_target, expected_put_axis, 6);
  assert(terlan_ndarray_v1_put(source, flat_indexes, 2, values) == 0);
  assert_ints(source, expected_put, 6);
  assert(terlan_ndarray_v1_delete(axis_target) == 0);
  assert(terlan_ndarray_v1_delete(axis_values) == 0);
  assert(terlan_ndarray_v1_delete(values) == 0);
  assert(terlan_ndarray_v1_delete(coordinates) == 0);
  assert(terlan_ndarray_v1_delete(indices) == 0);
  assert(terlan_ndarray_v1_delete(source) == 0);
}

static void test_take_modes_match_numpy_raise_wrap_and_clip(void) {
  const int64_t source_values[] = {0, 1, 2, 3, 4, 5};
  const int64_t shape[] = {2, 3};
  const int64_t wrap_expected[] = {3, 4, 5, 0, 1, 2, 3, 4, 5};
  const int64_t clip_expected[] = {0, 1, 2, 3, 4, 5, 3, 4, 5};
  TerlanNdArrayHandle source = ints(source_values, 6, shape, 2);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_take_mode(
             source, (const int64_t[]){-1, 2, 3}, 3, 0, 1, &output) == 0);
  assert_ints(output, wrap_expected, 9);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_take_mode(
             source, (const int64_t[]){-1, 2, 3}, 3, 0, 2, &output) == 0);
  assert_ints(output, clip_expected, 9);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_take_mode(
             source, (const int64_t[]){2}, 1, 0, 0, &output) ==
         TERLAN_NDARRAY_INDEX_OUT_OF_RANGE);
  assert(output == NULL);
  assert(terlan_ndarray_v1_take_mode(
             source, (const int64_t[]){0}, 1, 0, 3, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(output == NULL);
  assert(terlan_ndarray_v1_delete(source) == 0);
}

static void test_masks_compress_and_assignment(void) {
  const int64_t source_values[] = {1, 2, 3, 4, 5, 6};
  const int64_t shape[] = {2, 3};
  const uint8_t mask_values[] = {1, 0, 1, 0, 1, 0};
  const uint8_t axis_mask_values[] = {1, 0, 1};
  const int64_t selected[] = {1, 3, 5};
  const int64_t compressed[] = {1, 3, 4, 6};
  const int64_t assigned[] = {9, 2, 9, 4, 9, 6};
  const int64_t scalar_value = 9;
  TerlanNdArrayHandle source = ints(source_values, 6, shape, 2);
  TerlanNdArrayHandle mask = NULL;
  TerlanNdArrayHandle axis_mask = NULL;
  TerlanNdArrayHandle scalar = ints(&scalar_value, 1, NULL, 0);
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_from_bools(mask_values, 6, shape, 2, &mask) == 0);
  assert(terlan_ndarray_v1_from_bools(axis_mask_values, 3, (int64_t[]){3}, 1,
                                      &axis_mask) == 0);
  assert(terlan_ndarray_v1_mask_select(source, mask, &output) == 0);
  assert_ints(output, selected, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_compress(source, axis_mask, 1, &output) == 0);
  assert_ints(output, compressed, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_mask_assign(source, mask, scalar) == 0);
  assert_ints(source, assigned, 6);
  assert(terlan_ndarray_v1_delete(scalar) == 0);
  assert(terlan_ndarray_v1_delete(axis_mask) == 0);
  assert(terlan_ndarray_v1_delete(mask) == 0);
  assert(terlan_ndarray_v1_delete(source) == 0);
}

static void test_compress_truncates_axes_and_flattens_logically(void) {
  const int64_t source_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const int64_t shape[] = {2, 5};
  const uint8_t short_values[] = {0, 1};
  const uint8_t flat_values[] = {0, 1};
  const int64_t axis_expected[] = {2, 7};
  const int64_t flat_expected[] = {2};
  const int64_t transposed_expected[] = {6};
  TerlanNdArrayHandle source = ints(source_values, 10, shape, 2);
  TerlanNdArrayHandle short_mask = NULL;
  TerlanNdArrayHandle flat_mask = NULL;
  TerlanNdArrayHandle transposed = NULL;
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_from_bools(short_values, 2, (int64_t[]){2}, 1,
                                      &short_mask) == 0);
  assert(terlan_ndarray_v1_from_bools(flat_values, 2, (int64_t[]){2}, 1,
                                      &flat_mask) == 0);
  assert(terlan_ndarray_v1_compress(source, short_mask, 1, &output) == 0);
  assert_ints(output, axis_expected, 2);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_compress_flat(source, flat_mask, &output) == 0);
  assert_ints(output, flat_expected, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_transpose(source, 0, 1, &transposed) == 0);
  assert(terlan_ndarray_v1_compress_flat(transposed, flat_mask, &output) == 0);
  assert_ints(output, transposed_expected, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(transposed) == 0);
  assert(terlan_ndarray_v1_delete(flat_mask) == 0);
  assert(terlan_ndarray_v1_delete(short_mask) == 0);
  assert(terlan_ndarray_v1_delete(source) == 0);
}

static void test_putmask_stages_overlaps_and_rejects_invalid_destinations(void) {
  const uint8_t source_values[] = {1, 0, 1, 0};
  const uint8_t all_values[] = {1, 1, 1};
  const uint8_t independent_values[] = {1, 0, 1};
  const uint8_t first_expected[] = {1, 1, 0, 1};
  const uint8_t second_expected[] = {1, 1, 1, 1};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle destination = NULL;
  TerlanNdArrayHandle values = NULL;
  TerlanNdArrayHandle mask = NULL;
  TerlanNdArrayHandle independent = NULL;
  TerlanNdArrayHandle integer_source = ints((const int64_t[]){1, 2, 3}, 3,
                                             (const int64_t[]){3}, 1);
  TerlanNdArrayHandle integer_value = ints((const int64_t[]){5}, 1, NULL, 0);
  TerlanNdArrayHandle short_mask = NULL;
  TerlanNdArrayHandle broadcast = NULL;

  assert(terlan_ndarray_v1_from_bools(source_values, 4,
                                      (const int64_t[]){4}, 1, &source) == 0);
  assert(terlan_ndarray_v1_slice_axis(source, 0, 1, 4, 1, &destination) == 0);
  assert(terlan_ndarray_v1_slice_axis(source, 0, 0, 3, 1, &values) == 0);
  assert(terlan_ndarray_v1_from_bools(all_values, 3,
                                      (const int64_t[]){3}, 1, &mask) == 0);
  assert(terlan_ndarray_v1_putmask(destination, mask, values) == 0);
  assert_bools(source, first_expected, 4);
  assert(terlan_ndarray_v1_delete(values) == 0);
  assert(terlan_ndarray_v1_delete(destination) == 0);
  assert(terlan_ndarray_v1_delete(mask) == 0);
  assert(terlan_ndarray_v1_delete(source) == 0);

  assert(terlan_ndarray_v1_from_bools(source_values, 4,
                                      (const int64_t[]){4}, 1, &source) == 0);
  assert(terlan_ndarray_v1_slice_axis(source, 0, 1, 4, 1, &destination) == 0);
  assert(terlan_ndarray_v1_slice_axis(source, 0, 0, 3, 1, &mask) == 0);
  assert(terlan_ndarray_v1_from_bools(independent_values, 3,
                                      (const int64_t[]){3}, 1,
                                      &independent) == 0);
  assert(terlan_ndarray_v1_putmask(destination, mask, independent) == 0);
  assert_bools(source, second_expected, 4);
  assert(terlan_ndarray_v1_delete(independent) == 0);
  assert(terlan_ndarray_v1_delete(mask) == 0);
  assert(terlan_ndarray_v1_delete(destination) == 0);
  assert(terlan_ndarray_v1_delete(source) == 0);

  assert(terlan_ndarray_v1_from_bools((const uint8_t[]){1}, 1,
                                      (const int64_t[]){1}, 1,
                                      &short_mask) == 0);
  assert(terlan_ndarray_v1_putmask(integer_source, short_mask, integer_value) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(terlan_ndarray_v1_broadcast_to(integer_value,
                                        (const int64_t[]){3}, 1,
                                        &broadcast) == 0);
  assert(terlan_ndarray_v1_from_bools(all_values, 3,
                                      (const int64_t[]){3}, 1, &mask) == 0);
  assert(terlan_ndarray_v1_putmask(broadcast, mask, integer_value) ==
         TERLAN_NDARRAY_READ_ONLY);
  assert(terlan_ndarray_v1_delete(mask) == 0);
  assert(terlan_ndarray_v1_delete(broadcast) == 0);
  assert(terlan_ndarray_v1_delete(short_mask) == 0);
  assert(terlan_ndarray_v1_delete(integer_value) == 0);
  assert(terlan_ndarray_v1_delete(integer_source) == 0);
}

static void test_advanced_indexing_rejects_bad_shapes_and_indexes(void) {
  const int64_t shape[] = {2};
  const int64_t values[] = {1, 2};
  const int64_t bad_indexes[] = {0, 2};
  TerlanNdArrayHandle source = ints(values, 2, shape, 1);
  TerlanNdArrayHandle indices = ints(bad_indexes, 2, shape, 1);
  TerlanNdArrayHandle output = (TerlanNdArrayHandle)1;
  assert(terlan_ndarray_v1_take_along_axis(source, indices, 0, &output) ==
         TERLAN_NDARRAY_INDEX_OUT_OF_RANGE);
  assert(output == NULL);
  assert(terlan_ndarray_v1_put(source, bad_indexes, 2, source) ==
         TERLAN_NDARRAY_INDEX_OUT_OF_RANGE);
  assert(terlan_ndarray_v1_compress(source, source, 0, &output) ==
         TERLAN_NDARRAY_DTYPE_MISMATCH);
  assert(output == NULL);
  assert(terlan_ndarray_v1_delete(indices) == 0);
  assert(terlan_ndarray_v1_delete(source) == 0);
}

static void test_repeated_indexes_apply_in_input_order(void) {
  const int64_t source_values[] = {1, 2, 3};
  const int64_t indexes[] = {1, 1, -1};
  const int64_t replacement_values[] = {10, 20, 30};
  const int64_t expected[] = {1, 20, 30};
  const int64_t shape[] = {3};
  TerlanNdArrayHandle source = ints(source_values, 3, shape, 1);
  TerlanNdArrayHandle replacements =
      ints(replacement_values, 3, shape, 1);
  assert(terlan_ndarray_v1_put(source, indexes, 3, replacements) == 0);
  assert_ints(source, expected, 3);
  assert(terlan_ndarray_v1_delete(replacements) == 0);
  assert(terlan_ndarray_v1_delete(source) == 0);
}

static void test_coordinate_assignment_is_atomic_and_alias_safe(void) {
  const int64_t shape[] = {4, 4};
  const int64_t source_values[] = {1,  2,  3,  4,  5,  6,  7,  8,
                                   9,  10, 11, 12, 13, 14, 15, 16};
  const int64_t lower_coordinates[] = {0, 0, 1, 0, 1, 1, 2, 0, 2, 1,
                                       2, 2, 3, 0, 3, 1, 3, 2, 3, 3};
  const int64_t lower_shape[] = {10, 2};
  const int64_t selected[] = {1, 5, 6, 9, 10, 11, 13, 14, 15, 16};
  const int64_t assigned[] = {-1, 2,  3,  4,  -1, -1, 7,  8,
                              -1, -1, -1, 12, -1, -1, -1, -1};
  const int64_t scalar_value = -1;
  const int64_t bad_coordinates[] = {0, 0, 4, 0};
  const int64_t bad_shape[] = {2, 2};
  const int64_t alias_values[] = {1, 2, 3, 4};
  const int64_t alias_shape[] = {2, 2};
  const int64_t reverse_coordinates[] = {1, 1, 1, 0, 0, 1, 0, 0};
  const int64_t coordinate_shape[] = {4, 2};
  TerlanNdArrayHandle source = ints(source_values, 16, shape, 2);
  TerlanNdArrayHandle coordinates =
      ints(lower_coordinates, 20, lower_shape, 2);
  TerlanNdArrayHandle scalar = ints(&scalar_value, 1, NULL, 0);
  TerlanNdArrayHandle selected_values = NULL;
  TerlanNdArrayHandle bad = ints(bad_coordinates, 4, bad_shape, 2);
  TerlanNdArrayHandle aliased = ints(alias_values, 4, alias_shape, 2);
  TerlanNdArrayHandle reverse =
      ints(reverse_coordinates, 8, coordinate_shape, 2);
  TerlanNdArrayHandle wrong_dtype = NULL;

  assert(terlan_ndarray_v1_gather_coordinates(source, coordinates,
                                               &selected_values) == 0);
  assert_ints(selected_values, selected, 10);
  assert(terlan_ndarray_v1_put_coordinates(source, coordinates, scalar) == 0);
  assert_ints(source, assigned, 16);
  assert(terlan_ndarray_v1_put_coordinates(source, bad, scalar) ==
         TERLAN_NDARRAY_INDEX_OUT_OF_RANGE);
  assert_ints(source, assigned, 16);
  assert(terlan_ndarray_v1_put_coordinates(aliased, reverse, aliased) == 0);
  assert_ints(aliased, (const int64_t[]){4, 3, 2, 1}, 4);

  assert(terlan_ndarray_v1_from_floats(
             (const double[]){0.0, 0.0}, 2, (const int64_t[]){1, 2}, 2,
             &wrong_dtype) == 0);
  assert(terlan_ndarray_v1_put_coordinates(source, wrong_dtype, scalar) ==
         TERLAN_NDARRAY_DTYPE_MISMATCH);
  assert(terlan_ndarray_v1_put_coordinates(source, coordinates, wrong_dtype) ==
         TERLAN_NDARRAY_DTYPE_MISMATCH);

  assert(terlan_ndarray_v1_delete(wrong_dtype) == 0);
  assert(terlan_ndarray_v1_delete(reverse) == 0);
  assert(terlan_ndarray_v1_delete(aliased) == 0);
  assert(terlan_ndarray_v1_delete(bad) == 0);
  assert(terlan_ndarray_v1_delete(selected_values) == 0);
  assert(terlan_ndarray_v1_delete(scalar) == 0);
  assert(terlan_ndarray_v1_delete(coordinates) == 0);
  assert(terlan_ndarray_v1_delete(source) == 0);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_coordinate_add_at_matches_numpy_map_iterator(void) {
  const double matrix_values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  const double matrix_expected[] = {0, 1, 2, 19, 104, 5, 6, 7,
                                    8, 9, 40, 11};
  const int64_t matrix_shape[] = {3, 4};
  const int64_t coordinates_values[] = {1, 0, 1, 0, 2, 2, 0, 3};
  const int64_t coordinates_shape[] = {4, 2};
  const double additions_values[] = {50, 50, 30, 16};
  const int64_t additions_shape[] = {4};
  const double vector_values[] = {0, 1, 2, 3, 4, 5};
  const double vector_expected[] = {100.1, 51, 6, 3, 4, 5};
  const int64_t vector_shape[] = {6};
  const int64_t vector_coordinates_values[] = {1, 2, 0};
  const int64_t vector_coordinates_shape[] = {3, 1};
  const double vector_additions_values[] = {50, 4, 100.1};
  const int64_t vector_additions_shape[] = {3};
  const int64_t bad_coordinates_values[] = {0, 0, 3, 0};
  const int64_t bad_coordinates_shape[] = {2, 2};
  const double bad_additions_values[] = {1, 1};
  const int64_t bad_additions_shape[] = {2};
  TerlanNdArrayHandle matrix =
      floats(matrix_values, 12, matrix_shape, 2);
  TerlanNdArrayHandle coordinates =
      ints(coordinates_values, 8, coordinates_shape, 2);
  TerlanNdArrayHandle additions =
      floats(additions_values, 4, additions_shape, 1);
  TerlanNdArrayHandle vector = floats(vector_values, 6, vector_shape, 1);
  TerlanNdArrayHandle vector_coordinates =
      ints(vector_coordinates_values, 3, vector_coordinates_shape, 2);
  TerlanNdArrayHandle vector_additions =
      floats(vector_additions_values, 3, vector_additions_shape, 1);
  TerlanNdArrayHandle bad_coordinates =
      ints(bad_coordinates_values, 4, bad_coordinates_shape, 2);
  TerlanNdArrayHandle bad_additions =
      floats(bad_additions_values, 2, bad_additions_shape, 1);

  assert(terlan_ndarray_v1_add_at_coordinates(matrix, coordinates,
                                               additions) == 0);
  assert_floats(matrix, matrix_expected, 12);
  assert(terlan_ndarray_v1_add_at_coordinates(
             vector, vector_coordinates, vector_additions) == 0);
  assert_floats(vector, vector_expected, 6);
  assert(terlan_ndarray_v1_add_at_coordinates(matrix, bad_coordinates,
                                               bad_additions) ==
         TERLAN_NDARRAY_INDEX_OUT_OF_RANGE);
  assert_floats(matrix, matrix_expected, 12);

  assert(terlan_ndarray_v1_delete(bad_additions) == 0);
  assert(terlan_ndarray_v1_delete(bad_coordinates) == 0);
  assert(terlan_ndarray_v1_delete(vector_additions) == 0);
  assert(terlan_ndarray_v1_delete(vector_coordinates) == 0);
  assert(terlan_ndarray_v1_delete(vector) == 0);
  assert(terlan_ndarray_v1_delete(additions) == 0);
  assert(terlan_ndarray_v1_delete(coordinates) == 0);
  assert(terlan_ndarray_v1_delete(matrix) == 0);
  assert(terlan_ndarray_v1_live_count() == 0);
}

int main(void) {
  test_take_put_and_coordinate_gather();
  test_take_modes_match_numpy_raise_wrap_and_clip();
  test_masks_compress_and_assignment();
  test_compress_truncates_axes_and_flattens_logically();
  test_putmask_stages_overlaps_and_rejects_invalid_destinations();
  test_advanced_indexing_rejects_bad_shapes_and_indexes();
  test_repeated_indexes_apply_in_input_order();
  test_coordinate_assignment_is_atomic_and_alias_safe();
  test_coordinate_add_at_matches_numpy_map_iterator();
  return 0;
}
