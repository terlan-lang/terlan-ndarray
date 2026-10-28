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

static void test_bounded_shape_reference_model(void) {
  int64_t rows;
  for (rows = 1; rows <= 4; ++rows) {
    int64_t columns;
    for (columns = 1; columns <= 4; ++columns) {
      int64_t values[16];
      int64_t expected_transpose[16];
      int64_t expected_sum[4];
      const int64_t shape[] = {rows, columns};
      const int64_t axis_zero[] = {0};
      TerlanNdArrayHandle array = NULL;
      TerlanNdArrayHandle transposed = NULL;
      TerlanNdArrayHandle sum = NULL;
      int64_t row;
      int64_t column;
      for (column = 0; column < columns; ++column) {
        expected_sum[column] = 0;
      }
      for (row = 0; row < rows; ++row) {
        for (column = 0; column < columns; ++column) {
          const int64_t index = row * columns + column;
          values[index] = index - 5;
          expected_transpose[column * rows + row] = values[index];
          expected_sum[column] += values[index];
        }
      }
      assert(terlan_ndarray_v1_from_ints(values, rows * columns, shape, 2,
                                         &array) == TERLAN_NDARRAY_OK);
      assert(terlan_ndarray_v1_transpose(array, 0, 1, &transposed) ==
             TERLAN_NDARRAY_OK);
      assert_int_values(transposed, expected_transpose,
                        (size_t)(rows * columns));
      assert(terlan_ndarray_v1_sum(array, axis_zero, 1, 0, &sum) ==
             TERLAN_NDARRAY_OK);
      assert_int_values(sum, expected_sum, (size_t)columns);
      assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
      assert(terlan_ndarray_v1_delete(transposed) == TERLAN_NDARRAY_OK);
      assert(terlan_ndarray_v1_delete(sum) == TERLAN_NDARRAY_OK);
    }
  }
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_arbitrary_rank_rejections_and_operation_cleanup(void) {
  const int64_t values[] = {0, 1, 2, 3, 4, 5, 6, 7};
  const int64_t shape[] = {2, 2, 2};
  const int64_t transposed_values[] = {0, 4, 2, 6, 1, 5, 3, 7};
  const int64_t bad_shape[] = {3, 3};
  const int64_t negative_shape[] = {-1, 8};
  const int64_t overflow_values[] = {INT64_MAX, 1};
  const int64_t overflow_shape[] = {2};
  const int64_t axis_zero[] = {0};
  const uint8_t bool_values[] = {1, 0};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle transposed = NULL;
  TerlanNdArrayHandle peer = NULL;
  TerlanNdArrayHandle overflow = NULL;
  TerlanNdArrayHandle bools = NULL;
  TerlanNdArrayHandle result = NULL;
  int64_t fail_after;

  assert(terlan_ndarray_v1_from_ints(values, 8, shape, 3, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(values, 8, shape, 3, &peer) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_transpose(array, 0, 2, &transposed) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(transposed, transposed_values, 8);
  assert(terlan_ndarray_v1_delete(transposed) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_reshape(array, bad_shape, 2, &result) ==
         TERLAN_NDARRAY_ELEMENT_COUNT_MISMATCH);
  assert(terlan_ndarray_v1_reshape(array, negative_shape, 2, &result) ==
         TERLAN_NDARRAY_NEGATIVE_DIMENSION);
  assert(terlan_ndarray_v1_transpose(array, 0, 3, &result) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(result == NULL);

  assert(terlan_ndarray_v1_from_ints(overflow_values, 2, overflow_shape, 1,
                                     &overflow) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sum(overflow, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_ARITHMETIC_OVERFLOW);
  assert(result == NULL);
  assert(terlan_ndarray_v1_from_bools(bool_values, 2, overflow_shape, 1,
                                      &bools) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sum(bools, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(result, (const int64_t[]){1}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  for (fail_after = 0; fail_after < 4; ++fail_after) {
    terlan_ndarray_test_fail_allocation_after(fail_after);
    assert(terlan_ndarray_v1_reshape(array, shape, 3, &result) ==
           TERLAN_NDARRAY_ALLOCATION_FAILED);
    assert(result == NULL && terlan_ndarray_v1_live_count() == 4);
    terlan_ndarray_test_reset_allocation_failures();
  }
  for (fail_after = 0; fail_after < 5; ++fail_after) {
    terlan_ndarray_test_fail_allocation_after(fail_after);
    assert(terlan_ndarray_v1_transpose(array, 0, 2, &result) ==
           TERLAN_NDARRAY_ALLOCATION_FAILED);
    assert(result == NULL && terlan_ndarray_v1_live_count() == 4);
    terlan_ndarray_test_reset_allocation_failures();
  }
  for (fail_after = 0; fail_after < 4; ++fail_after) {
    terlan_ndarray_test_fail_allocation_after(fail_after);
    assert(terlan_ndarray_v1_add(array, peer, &result) ==
           TERLAN_NDARRAY_ALLOCATION_FAILED);
    assert(result == NULL && terlan_ndarray_v1_live_count() == 4);
    terlan_ndarray_test_reset_allocation_failures();
  }
  for (fail_after = 0; fail_after < 7; ++fail_after) {
    terlan_ndarray_test_fail_allocation_after(fail_after);
    assert(terlan_ndarray_v1_sum(array, axis_zero, 1, 0, &result) ==
           TERLAN_NDARRAY_ALLOCATION_FAILED);
    assert(result == NULL && terlan_ndarray_v1_live_count() == 4);
    terlan_ndarray_test_reset_allocation_failures();
  }

  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(peer) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(overflow) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(bools) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}


int main(void) {
  test_bounded_shape_reference_model();
  test_arbitrary_rank_rejections_and_operation_cleanup();
  return 0;
}
