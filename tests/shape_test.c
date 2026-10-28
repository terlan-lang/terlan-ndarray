#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

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

static void test_expand_squeeze_and_flip_are_views(void) {
  const int64_t values[] = {1, 2, 3, 4, 5, 6};
  const int64_t shape[] = {2, 3};
  const int64_t expanded_shape[] = {2, 1, 3};
  const int64_t flipped[] = {3, 2, 1, 6, 5, 4};
  const int64_t flipped_index[] = {0, 0};
  const int64_t source_after[] = {1, 2, 9, 4, 5, 6};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle expanded = NULL;
  TerlanNdArrayHandle squeezed = NULL;
  TerlanNdArrayHandle reversed = NULL;
  assert(terlan_ndarray_v1_from_ints(values, 6, shape, 2, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_expand_dims(source, -2, &expanded) ==
         TERLAN_NDARRAY_OK);
  assert(expanded->rank == 3 && expanded->shape[0] == expanded_shape[0] &&
         expanded->shape[1] == expanded_shape[1] &&
         expanded->shape[2] == expanded_shape[2]);
  assert(terlan_ndarray_v1_squeeze(expanded, NULL, 0, &squeezed) ==
         TERLAN_NDARRAY_OK);
  assert_ints(squeezed, values, 6);
  assert(terlan_ndarray_v1_flip(source, (const int64_t[]){1}, 1, &reversed) ==
         TERLAN_NDARRAY_OK);
  assert_ints(reversed, flipped, 6);
  assert(terlan_ndarray_v1_set_int(reversed, flipped_index, 2, 9) ==
         TERLAN_NDARRAY_OK);
  assert_ints(source, source_after, 6);
  assert(terlan_ndarray_v1_delete(reversed) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(squeezed) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(expanded) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_roll_repeat_and_tile_materialize_in_c_order(void) {
  const int64_t values[] = {1, 2, 3, 4, 5, 6};
  const int64_t shape[] = {2, 3};
  const int64_t rolled[] = {3, 1, 2, 6, 4, 5};
  const int64_t repeated[] = {1, 1, 2, 2, 3, 3,
                              4, 4, 5, 5, 6, 6};
  const int64_t tiled[] = {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_from_ints(values, 6, shape, 2, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_roll(source, 1, 1, &result) == TERLAN_NDARRAY_OK);
  assert_ints(result, rolled, 6);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_repeat(source, 2, 1, &result) == TERLAN_NDARRAY_OK);
  assert_ints(result, repeated, 12);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_tile(source, (const int64_t[]){1, 2}, 2,
                                &result) == TERLAN_NDARRAY_OK);
  assert_ints(result, tiled, 12);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_repeat_counts_match_numpy_axis_and_flatten_semantics(void) {
  const int64_t values[] = {1, 2, 3, 4, 5, 6};
  const int64_t shape[] = {2, 3};
  const int64_t axis_zero_expected[] = {1, 2, 3, 1, 2, 3, 4, 5, 6};
  const int64_t axis_one_expected[] = {1, 2, 2, 2, 3, 3,
                                       4, 5, 5, 5, 6, 6};
  const int64_t flat_expected[] = {1, 2, 2, 2, 3, 3, 4, 5, 6, 6};
  const int64_t transposed_expected[] = {1, 1, 4, 4, 2, 2,
                                         5, 5, 3, 3, 6, 6};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle transposed = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_from_ints(values, 6, shape, 2, &source) ==
         TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_repeat_counts(
             source, (const int64_t[]){2, 1}, 2, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, axis_zero_expected, 9);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_repeat_counts(
             source, (const int64_t[]){1, 3, 2}, 3, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, axis_one_expected, 12);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_repeat_flat(
             source, (const int64_t[]){1, 3, 2, 1, 1, 2}, 6, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, flat_expected, 10);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_transpose(source, 0, 1, &transposed) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_repeat_counts(transposed, (const int64_t[]){2}, 1,
                                         1, &result) == TERLAN_NDARRAY_OK);
  assert_ints(result, transposed_expected, 12);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(transposed) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_compress_outputs_are_exact_atomic_and_overlap_safe(void) {
  const int64_t values[] = {1, 2, 3, 4};
  const int64_t shape[] = {2, 2};
  const int64_t output_shape[] = {2, 1};
  const uint8_t mask_values[] = {0, 1};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle mask = NULL;
  TerlanNdArrayHandle destination = NULL;
  TerlanNdArrayHandle flat_destination = NULL;
  TerlanNdArrayHandle overlap_destination = NULL;
  TerlanNdArrayHandle wrong_shape = NULL;
  TerlanNdArrayHandle wrong_dtype = NULL;
  double *floating_values = NULL;
  size_t floating_count = 0;

  assert(terlan_ndarray_v1_from_ints(values, 4, shape, 2, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_bools(mask_values, 2,
                                      (const int64_t[]){2}, 1,
                                      &mask) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints((const int64_t[]){9, 9}, 2,
                                     output_shape, 2,
                                     &destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_compress_into(destination, source, mask, 1) ==
         TERLAN_NDARRAY_OK);
  assert_ints(destination, (const int64_t[]){2, 4}, 2);

  assert(terlan_ndarray_v1_from_ints((const int64_t[]){9}, 1,
                                     (const int64_t[]){1}, 1,
                                     &flat_destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_compress_flat_into(flat_destination, source, mask) ==
         TERLAN_NDARRAY_OK);
  assert_ints(flat_destination, (const int64_t[]){2}, 1);

  assert(terlan_ndarray_v1_slice_axis(source, 1, 0, 1, 1,
                                      &overlap_destination) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_compress_into(overlap_destination, source, mask,
                                         1) == TERLAN_NDARRAY_OK);
  assert_ints(source, (const int64_t[]){2, 2, 4, 4}, 4);

  assert(terlan_ndarray_v1_from_ints((const int64_t[]){7, 8}, 2,
                                     (const int64_t[]){2}, 1,
                                     &wrong_shape) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_compress_into(wrong_shape, source, mask, 1) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert_ints(wrong_shape, (const int64_t[]){7, 8}, 2);
  assert(terlan_ndarray_v1_from_floats_as(
             (const double[]){7.0, 8.0}, 2, output_shape, 2,
             TERLAN_NDARRAY_DTYPE_FLOAT32, &wrong_dtype) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_compress_into(wrong_dtype, source, mask, 1) ==
         TERLAN_NDARRAY_DTYPE_MISMATCH);
  assert(terlan_ndarray_v1_to_floats_cast(wrong_dtype, &floating_values,
                                          &floating_count) ==
         TERLAN_NDARRAY_OK);
  assert(floating_count == 2 && floating_values[0] == 7.0 &&
         floating_values[1] == 8.0);
  terlan_ndarray_v1_delete_float_values(floating_values);

  assert(terlan_ndarray_v1_delete(wrong_dtype) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(wrong_shape) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(overlap_destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(flat_destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(mask) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_concatenate_and_stack_support_non_contiguous_inputs(void) {
  const int64_t left_values[] = {1, 2, 3, 4};
  const int64_t right_values[] = {5, 6, 7, 8};
  const int64_t shape[] = {2, 2};
  const int64_t concatenated[] = {1, 3, 5, 7, 2, 4, 6, 8};
  const int64_t stacked[] = {1, 3, 2, 4, 5, 7, 6, 8};
  TerlanNdArrayHandle left = NULL;
  TerlanNdArrayHandle right = NULL;
  TerlanNdArrayHandle left_t = NULL;
  TerlanNdArrayHandle right_t = NULL;
  TerlanNdArrayHandle left_expanded = NULL;
  TerlanNdArrayHandle right_expanded = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_from_ints(left_values, 4, shape, 2, &left) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(right_values, 4, shape, 2, &right) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_transpose(left, 0, 1, &left_t) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_transpose(right, 0, 1, &right_t) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_concatenate_pair(left_t, right_t, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, concatenated, 8);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_expand_dims(left_t, 0, &left_expanded) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_expand_dims(right_t, 0, &right_expanded) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_concatenate_pair(left_expanded, right_expanded, 0,
                                         &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, stacked, 8);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(right_expanded) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(left_expanded) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(right_t) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(left_t) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(right) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(left) == TERLAN_NDARRAY_OK);
}

static void delete_handle_list(uint64_t *values, size_t length) {
  size_t index;
  for (index = 0; index < length; ++index) {
    assert(terlan_ndarray_v1_delete(
               (TerlanNdArrayHandle)(uintptr_t)values[index]) ==
           TERLAN_NDARRAY_OK);
  }
  terlan_ndarray_v1_delete_handle_values(values);
}

static void test_split_families_return_shared_views(void) {
  const int64_t shape[] = {2, 6};
  const int64_t values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  const int64_t split_zero[] = {0, 1, 6, 7};
  const int64_t split_one[] = {2, 3, 8, 9};
  const int64_t split_two[] = {4, 5, 10, 11};
  const int64_t uneven_two[] = {4, 10};
  const int64_t uneven_three[] = {5, 11};
  const int64_t indexed_middle[] = {1, 2, 3, 7, 8, 9};
  const int64_t source_after[] = {99, 1, 2, 3, 4, 5,
                                  6,  7, 8, 9, 10, 11};
  uint64_t *parts = NULL;
  size_t length = 0;
  TerlanNdArrayHandle source = NULL;

  assert(terlan_ndarray_v1_from_ints(values, 12, shape, 2, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_split(source, 3, 1, &parts, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == 3);
  assert_ints((TerlanNdArrayHandle)(uintptr_t)parts[0], split_zero, 4);
  assert_ints((TerlanNdArrayHandle)(uintptr_t)parts[1], split_one, 4);
  assert_ints((TerlanNdArrayHandle)(uintptr_t)parts[2], split_two, 4);
  assert(terlan_ndarray_v1_set_int(
             (TerlanNdArrayHandle)(uintptr_t)parts[0],
             (const int64_t[]){0, 0}, 2, 99) == TERLAN_NDARRAY_OK);
  assert_ints(source, source_after, 12);
  delete_handle_list(parts, length);

  assert(terlan_ndarray_v1_array_split(source, 4, -1, &parts, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == 4);
  assert_ints((TerlanNdArrayHandle)(uintptr_t)parts[2], uneven_two, 2);
  assert_ints((TerlanNdArrayHandle)(uintptr_t)parts[3], uneven_three, 2);
  delete_handle_list(parts, length);

  assert(terlan_ndarray_v1_split_at(source, (const int64_t[]){1, 4}, 2, 1,
                                    &parts, &length) == TERLAN_NDARRAY_OK);
  assert(length == 3);
  assert_ints((TerlanNdArrayHandle)(uintptr_t)parts[1], indexed_middle, 6);
  delete_handle_list(parts, length);

  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_shape_rejections_leave_no_results(void) {
  const int64_t values[] = {1, 2};
  const int64_t shape[] = {2};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_from_ints(values, 2, shape, 1, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_squeeze(array, (const int64_t[]){0}, 1, &result) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(result == NULL);
  assert(terlan_ndarray_v1_flip(array, (const int64_t[]){0, 0}, 2, &result) ==
         TERLAN_NDARRAY_DUPLICATE_AXIS);
  assert(terlan_ndarray_v1_repeat(array, -1, 0, &result) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_repeat_counts(
             array, (const int64_t[]){1, 2, 3}, 3, 0, &result) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(result == NULL);
  assert(terlan_ndarray_v1_repeat_counts(
             array, (const int64_t[]){1, -1}, 2, 0, &result) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(result == NULL);
  assert(terlan_ndarray_v1_repeat_flat(array, NULL, 1, &result) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(result == NULL);
  assert(terlan_ndarray_v1_tile(array, (const int64_t[]){-1}, 1, &result) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  {
    uint64_t *parts = NULL;
    size_t length = 0;
    assert(terlan_ndarray_v1_split(array, 3, 0, &parts, &length) ==
           TERLAN_NDARRAY_SHAPE_MISMATCH);
    assert(parts == NULL && length == 0);
    assert(terlan_ndarray_v1_array_split(array, 0, 0, &parts, &length) ==
           TERLAN_NDARRAY_INVALID_ARGUMENT);
    assert(parts == NULL && length == 0);
  }
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

int main(void) {
  test_expand_squeeze_and_flip_are_views();
  test_roll_repeat_and_tile_materialize_in_c_order();
  test_repeat_counts_match_numpy_axis_and_flatten_semantics();
  test_compress_outputs_are_exact_atomic_and_overlap_safe();
  test_concatenate_and_stack_support_non_contiguous_inputs();
  test_split_families_return_shared_views();
  test_shape_rejections_leave_no_results();
  return 0;
}
