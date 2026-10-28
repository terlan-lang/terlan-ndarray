#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

static void assert_int_values(TerlanNdArrayHandle array,
                              const int64_t *expected, size_t length) {
  int64_t *values = NULL;
  size_t actual_length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_ints(array, &values, &actual_length) ==
         TERLAN_NDARRAY_OK);
  assert(actual_length == length);
  for (index = 0; index < length; ++index) {
    assert(values[index] == expected[index]);
  }
  terlan_ndarray_v1_delete_int_values(values);
}

static void assert_int_values_cast(TerlanNdArrayHandle array,
                                   const int64_t *expected, size_t length) {
  int64_t *values = NULL;
  size_t actual_length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_ints_cast(array, &values, &actual_length) ==
         TERLAN_NDARRAY_OK);
  assert(actual_length == length);
  for (index = 0; index < length; ++index) {
    assert(values[index] == expected[index]);
  }
  terlan_ndarray_v1_delete_int_values(values);
}

static void assert_shape(TerlanNdArrayHandle array, const int64_t *expected,
                         int64_t rank) {
  int64_t *shape = NULL;
  int64_t actual_rank = -1;
  int64_t index;
  assert(terlan_ndarray_v1_rank(array, &actual_rank) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_shape(array, &shape) == TERLAN_NDARRAY_OK);
  assert(actual_rank == rank);
  for (index = 0; index < rank; ++index) {
    assert(shape[index] == expected[index]);
  }
}

static void test_views_share_storage_and_materialization_detaches(void) {
  const int64_t values[] = {1, 2, 3, 4, 5, 6};
  const int64_t shape[] = {2, 3};
  const int64_t transpose_shape[] = {3, 2};
  const int64_t transpose_values[] = {1, 4, 2, 5, 3, 6};
  const int64_t source_after_set[] = {1, 9, 3, 4, 5, 6};
  const int64_t copied_values[] = {1, 4, 9, 5, 3, 6};
  const int64_t transpose_index[] = {1, 0};
  const int64_t copy_index[] = {0, 0};
  int32_t contiguous = -1;
  int64_t value = 0;
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle transposed = NULL;
  TerlanNdArrayHandle copied = NULL;

  assert(terlan_ndarray_v1_from_ints(values, 6, shape, 2, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_transpose(source, 0, 1, &transposed) ==
         TERLAN_NDARRAY_OK);
  assert_shape(transposed, transpose_shape, 2);
  assert_int_values(transposed, transpose_values, 6);
  assert(terlan_ndarray_v1_is_contiguous(transposed, &contiguous) ==
         TERLAN_NDARRAY_OK);
  assert(contiguous == 0);
  assert(terlan_ndarray_v1_set_int(transposed, transpose_index, 2, 9) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(source, source_after_set, 6);
  assert(terlan_ndarray_v1_get_int(source, transpose_index, 2, &value) ==
         TERLAN_NDARRAY_OK);
  assert(value == 4);

  assert(terlan_ndarray_v1_contiguous(transposed, &copied) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(copied, copied_values, 6);
  assert(terlan_ndarray_v1_set_int(copied, copy_index, 2, 77) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(source, source_after_set, 6);
  assert(terlan_ndarray_v1_is_contiguous(copied, &contiguous) ==
         TERLAN_NDARRAY_OK);
  assert(contiguous == 1);

  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
  assert_int_values(transposed, copied_values, 6);
  assert(terlan_ndarray_v1_delete(copied) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(transposed) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_index_slice_reverse_and_permute_views(void) {
  const int64_t values[] = {1, 2, 3, 4, 5, 6};
  const int64_t shape[] = {2, 3};
  const int64_t row_shape[] = {3};
  const int64_t last_row[] = {4, 5, 6};
  const int64_t stepped_shape[] = {2, 2};
  const int64_t stepped_values[] = {1, 3, 4, 6};
  const int64_t reversed_values[] = {3, 2, 1, 6, 5, 4};
  const int64_t axes[] = {-1, 0};
  const int64_t permuted_shape[] = {3, 2};
  const int64_t permuted_values[] = {1, 4, 2, 5, 3, 6};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle row = NULL;
  TerlanNdArrayHandle stepped = NULL;
  TerlanNdArrayHandle reversed = NULL;
  TerlanNdArrayHandle permuted = NULL;
  TerlanNdArrayHandle rejected = NULL;

  assert(terlan_ndarray_v1_from_ints(values, 6, shape, 2, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_index_axis(source, 0, -1, &row) ==
         TERLAN_NDARRAY_OK);
  assert_shape(row, row_shape, 1);
  assert_int_values(row, last_row, 3);
  assert(terlan_ndarray_v1_slice_axis(source, 1, 0, 3, 2, &stepped) ==
         TERLAN_NDARRAY_OK);
  assert_shape(stepped, stepped_shape, 2);
  assert_int_values(stepped, stepped_values, 4);
  assert(terlan_ndarray_v1_slice_axis(source, 1, INT64_MIN, INT64_MIN, -1,
                                      &reversed) == TERLAN_NDARRAY_OK);
  assert_int_values(reversed, reversed_values, 6);
  assert(terlan_ndarray_v1_permute(source, axes, 2, &permuted) ==
         TERLAN_NDARRAY_OK);
  assert_shape(permuted, permuted_shape, 2);
  assert_int_values(permuted, permuted_values, 6);

  assert(terlan_ndarray_v1_index_axis(source, 0, 2, &rejected) ==
         TERLAN_NDARRAY_INDEX_OUT_OF_RANGE);
  assert(rejected == NULL);
  assert(terlan_ndarray_v1_slice_axis(source, 1, 0, 2, 0, &rejected) ==
         TERLAN_NDARRAY_INVALID_SLICE);
  assert(terlan_ndarray_v1_permute(source, (const int64_t[]){0, 0}, 2,
                                   &rejected) == TERLAN_NDARRAY_DUPLICATE_AXIS);

  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
  assert_int_values(row, last_row, 3);
  assert(terlan_ndarray_v1_delete(row) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(stepped) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(reversed) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(permuted) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_scalar_and_empty_index_contracts(void) {
  const int64_t scalar_value[] = {5};
  const int64_t empty_shape[] = {0, 3};
  TerlanNdArrayHandle scalar = NULL;
  TerlanNdArrayHandle empty = NULL;
  TerlanNdArrayHandle empty_slice = NULL;
  TerlanNdArrayHandle rejected = NULL;
  int64_t value = 0;

  assert(terlan_ndarray_v1_from_ints(scalar_value, 1, NULL, 0, &scalar) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_get_int(scalar, NULL, 0, &value) ==
         TERLAN_NDARRAY_OK);
  assert(value == 5);
  assert(terlan_ndarray_v1_set_int(scalar, NULL, 0, -8) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_get_int(scalar, NULL, 0, &value) ==
         TERLAN_NDARRAY_OK);
  assert(value == -8);
  assert(terlan_ndarray_v1_from_ints(NULL, 0, empty_shape, 2, &empty) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_slice_axis(empty, 0, INT64_MIN, INT64_MIN, 1,
                                      &empty_slice) == TERLAN_NDARRAY_OK);
  assert_shape(empty_slice, empty_shape, 2);
  assert(terlan_ndarray_v1_index_axis(empty, 0, 0, &rejected) ==
         TERLAN_NDARRAY_INDEX_OUT_OF_RANGE);

  assert(terlan_ndarray_v1_delete(empty_slice) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_take_uses_normalized_integer_indices(void) {
  const int64_t values[] = {1, 2, 3, 4, 5, 6};
  const int64_t shape[] = {2, 3};
  const int64_t indices[] = {-1, 0, 0};
  const int64_t expected[] = {3, 1, 1, 6, 4, 4};
  const int64_t expected_shape[] = {2, 3};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle selected = NULL;
  assert(terlan_ndarray_v1_from_ints(values, 6, shape, 2, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_take(array, indices, 3, -1, &selected) ==
         TERLAN_NDARRAY_OK);
  assert_shape(selected, expected_shape, 2);
  assert_int_values(selected, expected, 6);
  assert(terlan_ndarray_v1_delete(selected) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_take(array, (const int64_t[]){3}, 1, 1,
                                &selected) ==
         TERLAN_NDARRAY_INDEX_OUT_OF_RANGE);
  assert(selected == NULL);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_numpy_take_preserves_index_shape_and_stages_outputs(void) {
  const int64_t source_values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  const int64_t source_shape[] = {3, 4};
  const int64_t matrix_index_values[] = {0, 2};
  const int64_t matrix_index_shape[] = {1, 2};
  const int64_t matrix_output_shape[] = {3, 1, 2};
  const int64_t matrix_output_values[] = {0, 2, 4, 6, 8, 10};
  const int64_t scalar_index_value[] = {1};
  const int64_t scalar_output_shape[] = {3};
  const int64_t scalar_output_values[] = {1, 5, 9};
  const int64_t overlap_values[] = {0, 1, 2, 3, 4};
  const int64_t overlap_shape[] = {5};
  const int64_t overlap_index_values[] = {1, 2, 3};
  const int64_t overlap_index_shape[] = {3};
  const int64_t overlap_source_expected[] = {0, 1, 1, 2, 3};
  const int64_t overlap_destination_expected[] = {1, 2, 3};
  const int64_t unchanged_values[] = {91, 92, 93, 94, 95, 96};
  const int64_t flat_output_shape[] = {6};
  const int64_t invalid_index_values[] = {0, 4};
  const double floating_index_values[] = {0.0, 1.0};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle matrix_indices = NULL;
  TerlanNdArrayHandle matrix_output = NULL;
  TerlanNdArrayHandle scalar_index = NULL;
  TerlanNdArrayHandle scalar_output = NULL;
  TerlanNdArrayHandle destination = NULL;
  TerlanNdArrayHandle wrong_shape = NULL;
  TerlanNdArrayHandle invalid_indices = NULL;
  TerlanNdArrayHandle floating_indices = NULL;
  TerlanNdArrayHandle overlap_source = NULL;
  TerlanNdArrayHandle overlap_destination = NULL;
  TerlanNdArrayHandle overlap_indices = NULL;
  TerlanNdArrayHandle rejected = NULL;

  assert(terlan_ndarray_v1_from_ints(source_values, 12, source_shape, 2,
                                     &source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(matrix_index_values, 2,
                                     matrix_index_shape, 2,
                                     &matrix_indices) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_take_indices(source, matrix_indices, 1,
                                        &matrix_output) == TERLAN_NDARRAY_OK);
  assert_shape(matrix_output, matrix_output_shape, 3);
  assert_int_values(matrix_output, matrix_output_values, 6);

  assert(terlan_ndarray_v1_from_ints(scalar_index_value, 1, NULL, 0,
                                     &scalar_index) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_take_indices(source, scalar_index, 1,
                                        &scalar_output) == TERLAN_NDARRAY_OK);
  assert_shape(scalar_output, scalar_output_shape, 1);
  assert_int_values(scalar_output, scalar_output_values, 3);

  assert(terlan_ndarray_v1_from_ints(unchanged_values, 6, matrix_output_shape,
                                     3, &destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_take_indices_into(destination, source,
                                             matrix_indices, 1, 0) ==
         TERLAN_NDARRAY_OK);
  assert_shape(destination, matrix_output_shape, 3);
  assert_int_values(destination, matrix_output_values, 6);

  assert(terlan_ndarray_v1_from_ints(unchanged_values, 6, flat_output_shape, 1,
                                     &wrong_shape) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_take_indices_into(wrong_shape, source,
                                             matrix_indices, 1, 0) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert_int_values(wrong_shape, unchanged_values, 6);
  assert(terlan_ndarray_v1_from_ints(invalid_index_values, 2,
                                     matrix_index_shape, 2,
                                     &invalid_indices) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_take_indices_into(wrong_shape, source,
                                             invalid_indices, 1, 0) ==
         TERLAN_NDARRAY_INDEX_OUT_OF_RANGE);
  assert_int_values(wrong_shape, unchanged_values, 6);
  assert(terlan_ndarray_v1_from_floats(floating_index_values, 2,
                                       matrix_index_shape, 2,
                                       &floating_indices) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_take_indices(source, floating_indices, 1,
                                        &rejected) ==
         TERLAN_NDARRAY_DTYPE_MISMATCH);
  assert(rejected == NULL);
  assert(terlan_ndarray_v1_take_indices_mode(source, matrix_indices, 1, 3,
                                             &rejected) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(rejected == NULL);

  assert(terlan_ndarray_v1_from_ints(overlap_values, 5, overlap_shape, 1,
                                     &overlap_source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_slice_axis(overlap_source, 0, 2, 5, 1,
                                      &overlap_destination) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(overlap_index_values, 3,
                                     overlap_index_shape, 1,
                                     &overlap_indices) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_take_indices_into(
             overlap_destination, overlap_source, overlap_indices, 0, 1) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(overlap_destination, overlap_destination_expected, 3);
  assert_int_values(overlap_source, overlap_source_expected, 5);

  assert(terlan_ndarray_v1_delete(overlap_indices) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(overlap_destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(overlap_source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(floating_indices) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(invalid_indices) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(wrong_shape) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(scalar_output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(scalar_index) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(matrix_output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(matrix_indices) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_numpy_take_all_admitted_numeric_dtypes(void) {
  const int64_t values[] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  const int64_t shape[] = {2, 3, 4};
  const int64_t index_values[] = {0, 1};
  const int64_t index_shape[] = {2};
  TerlanNdArrayHandle indices = NULL;
  int32_t dtype;

  assert(terlan_ndarray_v1_from_ints(index_values, 2, index_shape, 1,
                                     &indices) == TERLAN_NDARRAY_OK);
  for (dtype = TERLAN_NDARRAY_DTYPE_BOOL;
       dtype <= TERLAN_NDARRAY_DTYPE_COMPLEX128; ++dtype) {
    TerlanNdArrayHandle source = NULL;
    TerlanNdArrayHandle selected = NULL;
    int32_t selected_dtype = 0;
    assert(terlan_ndarray_v1_from_ints_as(values, 24, shape, 3, dtype,
                                          &source) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_take_indices(source, indices, 0, &selected) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_dtype(selected, &selected_dtype) ==
           TERLAN_NDARRAY_OK);
    assert(selected_dtype == dtype);
    assert_shape(selected, shape, 3);
    assert_int_values_cast(selected, values, 24);
    assert(terlan_ndarray_v1_delete(selected) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
  }
  assert(terlan_ndarray_v1_delete(indices) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_nonzero_coordinate_family(void) {
  const int64_t shape[] = {2, 3};
  const int64_t values[] = {0, 2, 0, -1, 0, 4};
  const int64_t flat_expected[] = {1, 3, 5};
  const int64_t argwhere_shape[] = {3, 2};
  const int64_t argwhere_expected[] = {0, 1, 1, 0, 1, 2};
  const int64_t rows_expected[] = {0, 1, 1};
  const int64_t columns_expected[] = {1, 0, 2};
  const int64_t scalar_value[] = {1};
  const int64_t scalar_argwhere_shape[] = {1, 0};
  uint64_t *coordinates = NULL;
  size_t coordinate_count = 0;
  int64_t count = -1;
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle flat = NULL;
  TerlanNdArrayHandle locations = NULL;
  TerlanNdArrayHandle scalar = NULL;

  assert(terlan_ndarray_v1_from_ints(values, 6, shape, 2, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_count_nonzero(array, &count) == TERLAN_NDARRAY_OK);
  assert(count == 3);
  assert(terlan_ndarray_v1_flatnonzero(array, &flat) == TERLAN_NDARRAY_OK);
  assert_int_values(flat, flat_expected, 3);
  assert(terlan_ndarray_v1_argwhere(array, &locations) == TERLAN_NDARRAY_OK);
  assert_shape(locations, argwhere_shape, 2);
  assert_int_values(locations, argwhere_expected, 6);
  assert(terlan_ndarray_v1_nonzero(array, &coordinates, &coordinate_count) ==
         TERLAN_NDARRAY_OK);
  assert(coordinate_count == 2);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)coordinates[0],
                    rows_expected, 3);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)coordinates[1],
                    columns_expected, 3);
  assert(terlan_ndarray_v1_delete(
             (TerlanNdArrayHandle)(uintptr_t)coordinates[0]) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(
             (TerlanNdArrayHandle)(uintptr_t)coordinates[1]) ==
         TERLAN_NDARRAY_OK);
  terlan_ndarray_v1_delete_handle_values(coordinates);
  assert(terlan_ndarray_v1_delete(locations) == TERLAN_NDARRAY_OK);
  locations = NULL;

  assert(terlan_ndarray_v1_from_ints(scalar_value, 1, NULL, 0, &scalar) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argwhere(scalar, &locations) == TERLAN_NDARRAY_OK);
  assert_shape(locations, scalar_argwhere_shape, 2);
  assert(terlan_ndarray_v1_delete(locations) == TERLAN_NDARRAY_OK);
  locations = NULL;
  coordinates = NULL;
  coordinate_count = 0;
  assert(terlan_ndarray_v1_nonzero(scalar, &coordinates, &coordinate_count) ==
         TERLAN_NDARRAY_OK);
  assert(coordinate_count == 1);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)coordinates[0],
                    (const int64_t[]){0}, 1);
  assert(terlan_ndarray_v1_delete(
             (TerlanNdArrayHandle)(uintptr_t)coordinates[0]) ==
         TERLAN_NDARRAY_OK);
  terlan_ndarray_v1_delete_handle_values(coordinates);

  assert(terlan_ndarray_v1_delete(flat) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

int main(void) {
  test_views_share_storage_and_materialization_detaches();
  test_index_slice_reverse_and_permute_views();
  test_scalar_and_empty_index_contracts();
  test_take_uses_normalized_integer_indices();
  test_numpy_take_preserves_index_shape_and_stages_outputs();
  test_numpy_take_all_admitted_numeric_dtypes();
  test_nonzero_coordinate_family();
  return 0;
}
