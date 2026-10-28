#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static void assert_int_values(TerlanNdArrayHandle array,
                              const int64_t *expected, size_t count) {
  int64_t *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_ints_cast(array, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(values[index] == expected[index]);
  }
  free(values);
}

static void test_unique_returns_values_indexes_inverse_and_counts(void) {
  const double input[] = {3.0, 1.0, 3.0, 2.0, NAN, NAN};
  const int64_t shape[] = {2, 3};
  const int64_t expected_indexes[] = {1, 3, 0, 4};
  const int64_t expected_inverse[] = {2, 0, 2, 1, 3, 3};
  const int64_t expected_counts[] = {1, 1, 2, 2};
  TerlanNdArrayHandle source = NULL;
  uint64_t *values = NULL;
  size_t length = 0;
  double *unique = NULL;
  size_t unique_length = 0;
  assert(terlan_ndarray_v1_from_floats(input, 6, shape, 2, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_unique_all(source, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == 4);
  assert(terlan_ndarray_v1_to_floats(
             (TerlanNdArrayHandle)(uintptr_t)values[0], &unique,
             &unique_length) == TERLAN_NDARRAY_OK);
  assert(unique_length == 4);
  assert(unique[0] == 1.0 && unique[1] == 2.0 && unique[2] == 3.0 &&
         isnan(unique[3]));
  free(unique);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)values[1],
                    expected_indexes, 4);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)values[2],
                    expected_inverse, 6);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)values[3], expected_counts,
                    4);
  for (length = 0; length < 4; ++length) {
    assert(terlan_ndarray_v1_delete(
               (TerlanNdArrayHandle)(uintptr_t)values[length]) ==
           TERLAN_NDARRAY_OK);
  }
  terlan_ndarray_v1_delete_handle_values(values);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_set_algebra_and_membership_are_sorted_and_exact(void) {
  const int64_t lhs_values[] = {4, 1, 2, 2};
  const int64_t rhs_values[] = {3, 2, 5, 3};
  const int64_t shape[] = {2, 2};
  const int64_t union_values[] = {1, 2, 3, 4, 5};
  const int64_t intersection_values[] = {2};
  const int64_t difference_values[] = {1, 4};
  const int64_t xor_values[] = {1, 3, 4, 5};
  const int64_t membership_values[] = {0, 0, 1, 1};
  TerlanNdArrayHandle lhs = NULL;
  TerlanNdArrayHandle rhs = NULL;
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_from_ints(lhs_values, 4, shape, 2, &lhs) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(rhs_values, 4, shape, 2, &rhs) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_union(lhs, rhs, &output) == TERLAN_NDARRAY_OK);
  assert_int_values(output, union_values, 5);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_intersect(lhs, rhs, &output) == TERLAN_NDARRAY_OK);
  assert_int_values(output, intersection_values, 1);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_set_difference(lhs, rhs, &output) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(output, difference_values, 2);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_set_xor(lhs, rhs, &output) == TERLAN_NDARRAY_OK);
  assert_int_values(output, xor_values, 4);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_is_in(lhs, rhs, &output) == TERLAN_NDARRAY_OK);
  assert_int_values(output, membership_values, 4);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(rhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(lhs) == TERLAN_NDARRAY_OK);
}

static void test_numpy_set_variants_preserve_indexes_order_and_shape(void) {
  const int64_t left_values[] = {1, 2, 2, 3, 4, 3, 2};
  const int64_t right_values[] = {1, 8, 4, 2, 2, 3, 2, 3};
  const int64_t left_shape[] = {7};
  const int64_t right_shape[] = {2, 4};
  const int64_t expected_intersection[] = {1, 2, 3, 4};
  const int64_t expected_left_indexes[] = {0, 1, 3, 4};
  const int64_t expected_right_indexes[] = {0, 3, 5, 2};
  const int64_t unique_left_values[] = {3, 2, 1, 2};
  const int64_t unique_right_values[] = {7, 5, 2};
  const int64_t unique_left_shape[] = {2, 2};
  const int64_t unique_right_shape[] = {3};
  const int64_t expected_difference[] = {3, 1};
  TerlanNdArrayHandle left = NULL;
  TerlanNdArrayHandle right = NULL;
  TerlanNdArrayHandle unique_left = NULL;
  TerlanNdArrayHandle unique_right = NULL;
  TerlanNdArrayHandle difference = NULL;
  uint64_t *outputs = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_from_ints(left_values, 7, left_shape, 1, &left) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(right_values, 8, right_shape, 2, &right) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_intersect_all(left, right, &outputs, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == 3);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)outputs[0],
                    expected_intersection, 4);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)outputs[1],
                    expected_left_indexes, 4);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)outputs[2],
                    expected_right_indexes, 4);
  for (index = 0; index < length; ++index) {
    assert(terlan_ndarray_v1_delete(
               (TerlanNdArrayHandle)(uintptr_t)outputs[index]) ==
           TERLAN_NDARRAY_OK);
  }
  terlan_ndarray_v1_delete_handle_values(outputs);
  assert(terlan_ndarray_v1_from_ints(unique_left_values, 4, unique_left_shape,
                                    2, &unique_left) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(unique_right_values, 3,
                                    unique_right_shape, 1, &unique_right) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_set_difference_unique(unique_left, unique_right,
                                                 &difference) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(difference, expected_difference, 2);
  assert(terlan_ndarray_v1_delete(difference) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(unique_right) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(unique_left) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(right) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(left) == TERLAN_NDARRAY_OK);
}

static void test_in1d_and_ediff1d_match_numpy_flattening_and_edges(void) {
  const int64_t element_values[] = {0, 1, 2, 3, 4, 5};
  const int64_t candidate_values[] = {3, 4, 5, 6, 7, 8};
  const int64_t element_shape[] = {2, 3};
  const int64_t candidate_shape[] = {3, 2};
  const int64_t expected_membership[] = {0, 0, 0, 1, 1, 1};
  const int64_t expected_inverted[] = {1, 1, 1, 0, 0, 0};
  const int64_t source_values[] = {1, 2};
  const int64_t source_shape[] = {2};
  const int64_t begin_values[] = {5, 6};
  const int64_t end_values[] = {7, 8};
  const int64_t edge_shape[] = {2};
  const int64_t expected_difference[] = {1};
  const int64_t expected_edges[] = {5, 6, 1, 7, 8};
  TerlanNdArrayHandle elements = NULL;
  TerlanNdArrayHandle candidates = NULL;
  TerlanNdArrayHandle membership = NULL;
  TerlanNdArrayHandle inverted = NULL;
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle begin = NULL;
  TerlanNdArrayHandle end = NULL;
  TerlanNdArrayHandle difference = NULL;
  assert(terlan_ndarray_v1_from_ints(element_values, 6, element_shape, 2,
                                    &elements) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(candidate_values, 6, candidate_shape, 2,
                                    &candidates) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_in1d(elements, candidates, 0, &membership) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(membership, expected_membership, 6);
  assert(terlan_ndarray_v1_in1d(elements, candidates, 1, &inverted) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(inverted, expected_inverted, 6);
  assert(terlan_ndarray_v1_from_ints(source_values, 2, source_shape, 1,
                                    &source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_ediff1d(source, &difference) == TERLAN_NDARRAY_OK);
  assert_int_values(difference, expected_difference, 1);
  assert(terlan_ndarray_v1_delete(difference) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(begin_values, 2, edge_shape, 1, &begin) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(end_values, 2, edge_shape, 1, &end) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_ediff1d_edges(source, begin, end, &difference) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(difference, expected_edges, 5);
  assert(terlan_ndarray_v1_delete(difference) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(end) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(begin) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(inverted) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(membership) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(candidates) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(elements) == TERLAN_NDARRAY_OK);
}

static void test_unique_axis_returns_values_indexes_inverse_and_counts(void) {
  const int64_t values[] = {0, 1, 0, 0, 1, 0, 0, 0,
                            0, 1, 0, 0, 1, 0, 0, 0};
  const int64_t shape[] = {4, 4};
  const int64_t axis0_values[] = {0, 1, 0, 0, 1, 0, 0, 0};
  const int64_t axis0_indexes[] = {0, 1};
  const int64_t axis0_inverse[] = {0, 1, 0, 1};
  const int64_t axis0_counts[] = {2, 2};
  const int64_t axis1_values[] = {0, 0, 1, 0, 1, 0,
                                  0, 0, 1, 0, 1, 0};
  const int64_t axis1_indexes[] = {2, 0, 1};
  const int64_t axis1_inverse[] = {1, 2, 0, 0};
  const int64_t axis1_counts[] = {2, 1, 1};
  const int64_t zero_shape[] = {2, 0};
  const int64_t zero_axis0_indexes[] = {0};
  const int64_t zero_axis0_inverse[] = {0, 0};
  const int64_t zero_axis0_counts[] = {2};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle zero = NULL;
  uint64_t *outputs = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_from_ints(values, 16, shape, 2, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_unique_axis_all(array, 0, &outputs, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == 4);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)outputs[0], axis0_values,
                    8);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)outputs[1], axis0_indexes,
                    2);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)outputs[2], axis0_inverse,
                    4);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)outputs[3], axis0_counts,
                    2);
  for (index = 0; index < length; ++index) {
    assert(terlan_ndarray_v1_delete(
               (TerlanNdArrayHandle)(uintptr_t)outputs[index]) ==
           TERLAN_NDARRAY_OK);
  }
  terlan_ndarray_v1_delete_handle_values(outputs);
  assert(terlan_ndarray_v1_unique_axis_all(array, -1, &outputs, &length) ==
         TERLAN_NDARRAY_OK);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)outputs[0], axis1_values,
                    12);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)outputs[1], axis1_indexes,
                    3);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)outputs[2], axis1_inverse,
                    4);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)outputs[3], axis1_counts,
                    3);
  for (index = 0; index < length; ++index) {
    assert(terlan_ndarray_v1_delete(
               (TerlanNdArrayHandle)(uintptr_t)outputs[index]) ==
           TERLAN_NDARRAY_OK);
  }
  terlan_ndarray_v1_delete_handle_values(outputs);
  assert(terlan_ndarray_v1_from_ints(NULL, 0, zero_shape, 2, &zero) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_unique_axis_all(zero, 0, &outputs, &length) ==
         TERLAN_NDARRAY_OK);
  assert(((TerlanNdArrayHandle)(uintptr_t)outputs[0])->shape[0] == 1);
  assert(((TerlanNdArrayHandle)(uintptr_t)outputs[0])->shape[1] == 0);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)outputs[1],
                    zero_axis0_indexes, 1);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)outputs[2],
                    zero_axis0_inverse, 2);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)outputs[3],
                    zero_axis0_counts, 1);
  for (index = 0; index < length; ++index) {
    assert(terlan_ndarray_v1_delete(
               (TerlanNdArrayHandle)(uintptr_t)outputs[index]) ==
           TERLAN_NDARRAY_OK);
  }
  terlan_ndarray_v1_delete_handle_values(outputs);
  assert(terlan_ndarray_v1_unique_axis_all(zero, 1, &outputs, &length) ==
         TERLAN_NDARRAY_OK);
  assert(((TerlanNdArrayHandle)(uintptr_t)outputs[0])->shape[0] == 2);
  assert(((TerlanNdArrayHandle)(uintptr_t)outputs[0])->shape[1] == 0);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)outputs[1], NULL, 0);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)outputs[2], NULL, 0);
  assert_int_values((TerlanNdArrayHandle)(uintptr_t)outputs[3], NULL, 0);
  for (index = 0; index < length; ++index) {
    assert(terlan_ndarray_v1_delete(
               (TerlanNdArrayHandle)(uintptr_t)outputs[index]) ==
           TERLAN_NDARRAY_OK);
  }
  terlan_ndarray_v1_delete_handle_values(outputs);
  assert(terlan_ndarray_v1_delete(zero) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_set_rejections_and_partial_allocation_cleanup(void) {
  const int64_t integers[] = {1, 2};
  const double floats[] = {1.0, 2.0};
  const int64_t shape[] = {2};
  TerlanNdArrayHandle integer_array = NULL;
  TerlanNdArrayHandle float_array = NULL;
  TerlanNdArrayHandle output = (TerlanNdArrayHandle)1;
  uint64_t *outputs = (uint64_t *)1;
  size_t length = 99;
  int64_t failure;
  assert(terlan_ndarray_v1_from_ints(integers, 2, shape, 1, &integer_array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(floats, 2, shape, 1, &float_array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_union(integer_array, float_array, &output) ==
         TERLAN_NDARRAY_DTYPE_MISMATCH);
  assert(output == NULL);
  output = (TerlanNdArrayHandle)1;
  assert(terlan_ndarray_v1_in1d(integer_array, float_array, 0, &output) ==
         TERLAN_NDARRAY_DTYPE_MISMATCH);
  assert(output == NULL);
  output = (TerlanNdArrayHandle)1;
  assert(terlan_ndarray_v1_ediff1d_edges(integer_array, float_array,
                                        float_array, &output) ==
         TERLAN_NDARRAY_DTYPE_MISMATCH);
  assert(output == NULL);
  outputs = (uint64_t *)1;
  length = 99;
  assert(terlan_ndarray_v1_intersect_all(integer_array, float_array, &outputs,
                                         &length) ==
         TERLAN_NDARRAY_DTYPE_MISMATCH);
  assert(outputs == NULL);
  assert(length == 0);
  outputs = (uint64_t *)1;
  length = 99;
  assert(terlan_ndarray_v1_unique_axis_all(integer_array, 1, &outputs,
                                           &length) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(outputs == NULL);
  assert(length == 0);
  assert(terlan_ndarray_v1_unique_all(integer_array, NULL, &length) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  for (failure = 0; failure < 10; ++failure) {
    outputs = (uint64_t *)1;
    length = 99;
    terlan_ndarray_test_fail_allocation_after(failure);
    if (terlan_ndarray_v1_unique_all(integer_array, &outputs, &length) ==
        TERLAN_NDARRAY_OK) {
      size_t index;
      for (index = 0; index < length; ++index) {
        assert(terlan_ndarray_v1_delete(
                   (TerlanNdArrayHandle)(uintptr_t)outputs[index]) ==
               TERLAN_NDARRAY_OK);
      }
      terlan_ndarray_v1_delete_handle_values(outputs);
    } else {
      assert(outputs == NULL);
      assert(length == 0);
    }
    terlan_ndarray_test_reset_allocation_failures();
  }
  assert(terlan_ndarray_v1_delete(float_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(integer_array) == TERLAN_NDARRAY_OK);
}

int main(void) {
  test_unique_returns_values_indexes_inverse_and_counts();
  test_set_algebra_and_membership_are_sorted_and_exact();
  test_numpy_set_variants_preserve_indexes_order_and_shape();
  test_in1d_and_ediff1d_match_numpy_flattening_and_edges();
  test_unique_axis_returns_values_indexes_inverse_and_counts();
  test_set_rejections_and_partial_allocation_cleanup();
  return 0;
}
