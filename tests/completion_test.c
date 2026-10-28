#include "../bindings/include/terlan_ndarray.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

static TerlanNdArrayHandle ints(const int64_t *values, int64_t count,
                               const int64_t *shape, int64_t rank) {
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_from_ints(values, count, shape, rank, &output) ==
         TERLAN_NDARRAY_OK);
  return output;
}

static TerlanNdArrayHandle floats(const double *values, int64_t count,
                                 const int64_t *shape, int64_t rank) {
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_from_floats(values, count, shape, rank, &output) ==
         TERLAN_NDARRAY_OK);
  return output;
}

static void expect_ints(TerlanNdArrayHandle array, const int64_t *expected,
                        size_t count) {
  int64_t *values = NULL;
  size_t length = 0;
  assert(terlan_ndarray_v1_to_ints_cast(array, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (size_t index = 0; index < count; ++index) {
    assert(values[index] == expected[index]);
  }
  terlan_ndarray_v1_delete_int_values(values);
}

static void test_numpy_resize_repeats_zero_shapes_and_dtype(void) {
  const int64_t matrix_values[] = {1, 2, 3, 4};
  TerlanNdArrayHandle matrix =
      ints(matrix_values, 4, (const int64_t[]){2, 2}, 2);
  TerlanNdArrayHandle empty = NULL;
  TerlanNdArrayHandle result = NULL;
  int32_t dtype = -1;
  int64_t rank = -1;
  int64_t *shape = NULL;

  assert(terlan_ndarray_v1_resize(matrix, (const int64_t[]){2, 4}, 2,
                                  &result) == TERLAN_NDARRAY_OK);
  expect_ints(result, (const int64_t[]){1, 2, 3, 4, 1, 2, 3, 4}, 8);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_resize(matrix, (const int64_t[]){4, 2}, 2,
                                  &result) == TERLAN_NDARRAY_OK);
  expect_ints(result, (const int64_t[]){1, 2, 3, 4, 1, 2, 3, 4}, 8);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_resize(matrix, (const int64_t[]){4, 3}, 2,
                                  &result) == TERLAN_NDARRAY_OK);
  expect_ints(result,
              (const int64_t[]){1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}, 12);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_resize(matrix, (const int64_t[]){0, 2}, 2,
                                  &result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_rank(result, &rank) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_shape(result, &shape) == TERLAN_NDARRAY_OK);
  assert(rank == 2 && shape[0] == 0 && shape[1] == 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_floats_as(NULL, 0, (const int64_t[]){0}, 1,
                                          TERLAN_NDARRAY_DTYPE_FLOAT32,
                                          &empty) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_resize(empty, (const int64_t[]){2, 1}, 2,
                                  &result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dtype(result, &dtype) == TERLAN_NDARRAY_OK);
  assert(dtype == TERLAN_NDARRAY_DTYPE_FLOAT32);
  expect_ints(result, (const int64_t[]){0, 0}, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_resize(matrix, (const int64_t[]){-1}, 1,
                                  &result) == TERLAN_NDARRAY_NEGATIVE_DIMENSION);
  assert(result == NULL);
  assert(terlan_ndarray_v1_delete(empty) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_choose_fill_and_resize_preserve_broadcast_and_dtype(void) {
  const int64_t index_shape[2] = {2, 1};
  const int64_t index_values[2] = {0, 1};
  const int64_t first_shape[2] = {1, 3};
  const int64_t first_values[3] = {10, 20, 30};
  const int64_t second_shape[2] = {2, 1};
  const int64_t second_values[2] = {100, 200};
  const int64_t expected_choice[6] = {10, 20, 30, 200, 200, 200};
  const int64_t source_shape[1] = {3};
  const int64_t source_values[3] = {1, 2, 3};
  const int64_t scalar_shape[1] = {1};
  const double scalar_value[1] = {7.0};
  const int64_t resize_shape[2] = {2, 4};
  const int64_t expected_resize[8] = {1, 2, 3, 1, 2, 3, 1, 2};
  const int64_t expected_fill[3] = {7, 7, 7};
  TerlanNdArrayHandle indexes =
      ints(index_values, 2, index_shape, 2);
  TerlanNdArrayHandle first = ints(first_values, 3, first_shape, 2);
  TerlanNdArrayHandle second = ints(second_values, 2, second_shape, 2);
  TerlanNdArrayHandle source = ints(source_values, 3, source_shape, 1);
  TerlanNdArrayHandle scalar = floats(scalar_value, 1, scalar_shape, 1);
  const uint64_t choices[2] = {(uint64_t)(uintptr_t)first,
                               (uint64_t)(uintptr_t)second};
  TerlanNdArrayHandle selected = NULL;
  TerlanNdArrayHandle resized = NULL;

  assert(terlan_ndarray_v1_choose(indexes, choices, 2, 0, &selected) ==
         TERLAN_NDARRAY_OK);
  expect_ints(selected, expected_choice, 6);
  assert(terlan_ndarray_v1_resize(source, resize_shape, 2, &resized) ==
         TERLAN_NDARRAY_OK);
  expect_ints(resized, expected_resize, 8);
  assert(terlan_ndarray_v1_fill(source, scalar) == TERLAN_NDARRAY_OK);
  expect_ints(source, expected_fill, 3);

  assert(terlan_ndarray_v1_delete(resized) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(selected) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(second) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(first) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(indexes) == TERLAN_NDARRAY_OK);
}

static void test_choose_promotes_all_admitted_mixed_scalar_dtypes(void) {
  const int64_t shape[] = {1};
  const int64_t index_values[] = {0};
  TerlanNdArrayHandle indexes = ints(index_values, 1, shape, 1);
  TerlanNdArrayHandle integer = ints((const int64_t[]){1000}, 1, shape, 1);
  TerlanNdArrayHandle negative = ints((const int64_t[]){-1}, 1, shape, 1);
  TerlanNdArrayHandle byte = NULL;
  TerlanNdArrayHandle real = floats((const double[]){1.0}, 1, shape, 1);
  TerlanNdArrayHandle single = NULL;
  TerlanNdArrayHandle output = NULL;
  int32_t dtype = -1;

  assert(terlan_ndarray_v1_from_ints_as((const int64_t[]){1}, 1, shape, 1,
                                        TERLAN_NDARRAY_DTYPE_UINT8,
                                        &byte) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats_as((const double[]){3.0}, 1, shape, 1,
                                          TERLAN_NDARRAY_DTYPE_FLOAT32,
                                          &single) == TERLAN_NDARRAY_OK);
  {
    const uint64_t choices[] = {(uint64_t)(uintptr_t)integer,
                                (uint64_t)(uintptr_t)byte};
    assert(terlan_ndarray_v1_choose(indexes, choices, 2, 0, &output) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_dtype(output, &dtype) == TERLAN_NDARRAY_OK);
    assert(dtype == TERLAN_NDARRAY_DTYPE_INT64);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  }
  {
    const uint64_t choices[] = {(uint64_t)(uintptr_t)negative,
                                (uint64_t)(uintptr_t)byte};
    assert(terlan_ndarray_v1_choose(indexes, choices, 2, 0, &output) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_dtype(output, &dtype) == TERLAN_NDARRAY_OK);
    assert(dtype == TERLAN_NDARRAY_DTYPE_INT64);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  }
  {
    const uint64_t choices[] = {(uint64_t)(uintptr_t)real,
                                (uint64_t)(uintptr_t)single};
    assert(terlan_ndarray_v1_choose(indexes, choices, 2, 0, &output) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_dtype(output, &dtype) == TERLAN_NDARRAY_OK);
    assert(dtype == TERLAN_NDARRAY_DTYPE_FLOAT64);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  }
  assert(terlan_ndarray_v1_delete(single) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(real) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(byte) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(negative) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(integer) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(indexes) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_completion_rejections_leave_no_partial_output(void) {
  const int64_t shape[1] = {2};
  const int64_t bad_indexes_values[2] = {-1, 2};
  const int64_t choice_values[2] = {1, 2};
  const double float_indexes_values[2] = {0.0, 1.0};
  TerlanNdArrayHandle bad_indexes =
      ints(bad_indexes_values, 2, shape, 1);
  TerlanNdArrayHandle choice = ints(choice_values, 2, shape, 1);
  TerlanNdArrayHandle float_indexes =
      floats(float_indexes_values, 2, shape, 1);
  const uint64_t choices[1] = {(uint64_t)(uintptr_t)choice};
  TerlanNdArrayHandle output = (TerlanNdArrayHandle)(uintptr_t)1;

  assert(terlan_ndarray_v1_choose(bad_indexes, choices, 1, 0, &output) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(output == NULL);
  output = (TerlanNdArrayHandle)(uintptr_t)1;
  assert(terlan_ndarray_v1_choose(float_indexes, choices, 1, 0, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(output == NULL);
  output = (TerlanNdArrayHandle)(uintptr_t)1;
  assert(terlan_ndarray_v1_choose(choice, NULL, 0, 0, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(output == NULL);
  assert(terlan_ndarray_v1_fill(choice, choice) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  output = (TerlanNdArrayHandle)(uintptr_t)1;
  assert(terlan_ndarray_v1_resize(choice, shape, -1, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(output == NULL);

  assert(terlan_ndarray_v1_delete(float_indexes) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(choice) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(bad_indexes) == TERLAN_NDARRAY_OK);
}

static void test_choose_into_stages_overlap_and_rejects_invalid_outputs(void) {
  const int64_t source_values[] = {0, 1, 2, 3, 4};
  const int64_t index_values[] = {0, 0, 0};
  const int64_t zero_values[] = {9, 9, 9};
  const int64_t expected_values[] = {0, 0, 1, 2, 4};
  const int64_t unchanged_values[] = {9, 9, 9};
  const int64_t shape[] = {5};
  const int64_t selected_shape[] = {3};
  const int64_t wrong_shape[] = {2};
  TerlanNdArrayHandle source = ints(source_values, 5, shape, 1);
  TerlanNdArrayHandle indexes = ints(index_values, 3, selected_shape, 1);
  TerlanNdArrayHandle choice = NULL;
  TerlanNdArrayHandle destination = NULL;
  TerlanNdArrayHandle wrong = ints(zero_values, 2, wrong_shape, 1);
  TerlanNdArrayHandle scalar = ints((const int64_t[]){9}, 1, NULL, 0);
  TerlanNdArrayHandle read_only = NULL;
  uint64_t choices[1] = {(uint64_t)(uintptr_t)source};

  assert(terlan_ndarray_v1_slice_axis(source, 0, 0, 3, 1, &choice) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_slice_axis(source, 0, 1, 4, 1, &destination) ==
         TERLAN_NDARRAY_OK);
  choices[0] = (uint64_t)(uintptr_t)choice;
  assert(terlan_ndarray_v1_choose_into(destination, indexes, choices, 1, 0) ==
         TERLAN_NDARRAY_OK);
  expect_ints(source, expected_values, 5);

  assert(terlan_ndarray_v1_choose_into(wrong, indexes, choices, 1, 0) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  expect_ints(wrong, unchanged_values, 2);
  assert(terlan_ndarray_v1_broadcast_to(scalar, selected_shape, 1,
                                        &read_only) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_choose_into(read_only, indexes, choices, 1, 0) ==
         TERLAN_NDARRAY_READ_ONLY);
  expect_ints(read_only, unchanged_values, 3);

  assert(terlan_ndarray_v1_delete(read_only) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(wrong) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(choice) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(indexes) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

int main(void) {
  test_numpy_resize_repeats_zero_shapes_and_dtype();
  test_choose_fill_and_resize_preserve_broadcast_and_dtype();
  test_choose_promotes_all_admitted_mixed_scalar_dtypes();
  test_completion_rejections_leave_no_partial_output();
  test_choose_into_stages_overlap_and_rejects_invalid_outputs();
  assert(terlan_ndarray_v1_live_count() == 0);
  return 0;
}
