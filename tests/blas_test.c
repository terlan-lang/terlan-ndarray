#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <cblas.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static void assert_shape(TerlanNdArrayHandle array, const int64_t *expected,
                         int64_t rank) {
  int64_t actual_rank = -1;
  int64_t *actual_shape = NULL;
  int64_t index;
  assert(terlan_ndarray_v1_rank(array, &actual_rank) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_shape(array, &actual_shape) == TERLAN_NDARRAY_OK);
  assert(actual_rank == rank);
  for (index = 0; index < rank; ++index) {
    assert(actual_shape[index] == expected[index]);
  }
}

static void assert_values(TerlanNdArrayHandle array, const double *expected,
                          size_t length, double tolerance) {
  double *actual = NULL;
  size_t actual_length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_floats(array, &actual, &actual_length) ==
         TERLAN_NDARRAY_OK);
  assert(actual_length == length);
  for (index = 0; index < length; ++index) {
    if (isnan(expected[index])) {
      assert(isnan(actual[index]));
    } else {
      assert(fabs(actual[index] - expected[index]) <= tolerance);
    }
  }
  terlan_ndarray_v1_delete_float_values(actual);
}

static void reference_matmul(const double *lhs, const double *rhs,
                             double *output, size_t rows, size_t shared,
                             size_t columns) {
  size_t row;
  size_t column;
  size_t inner;
  for (row = 0; row < rows; ++row) {
    for (column = 0; column < columns; ++column) {
      double value = 0.0;
      for (inner = 0; inner < shared; ++inner) {
        value += lhs[row * shared + inner] *
                 rhs[inner * columns + column];
      }
      output[row * columns + column] = value;
    }
  }
}

static void test_rectangular_and_fractional_matmul(void) {
  const int64_t left_shape[] = {2, 3};
  const int64_t right_shape[] = {3, 2};
  const int64_t result_shape[] = {2, 2};
  const double left_values[] = {1, 2, 3, 4, 5, 6};
  const double right_values[] = {7, 8, 9, 10, 11, 12};
  double expected[4];
  const int64_t fraction_shape[] = {2, 2};
  const double fraction_left[] = {0.5, 1.25, -2.0, 3.5};
  const double fraction_right[] = {4.0, -1.0, 0.25, 2.0};
  double fraction_expected[4];
  TerlanNdArrayHandle left = NULL;
  TerlanNdArrayHandle right = NULL;
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayHandle fraction_a = NULL;
  TerlanNdArrayHandle fraction_b = NULL;
  TerlanNdArrayHandle fraction_result = NULL;

  assert(sizeof(blasint) == sizeof(int32_t));
  reference_matmul(left_values, right_values, expected, 2, 3, 2);
  reference_matmul(fraction_left, fraction_right, fraction_expected, 2, 2, 2);
  assert(expected[0] == 58.0 && expected[1] == 64.0 &&
         expected[2] == 139.0 && expected[3] == 154.0);
  assert(terlan_ndarray_v1_from_floats(left_values, 6, left_shape, 2, &left) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(right_values, 6, right_shape, 2,
                                       &right) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul(left, right, &result) == TERLAN_NDARRAY_OK);
  assert_shape(result, result_shape, 2);
  assert_values(result, expected, 4, 0.0);

  assert(terlan_ndarray_v1_from_floats(fraction_left, 4, fraction_shape, 2,
                                       &fraction_a) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(fraction_right, 4, fraction_shape, 2,
                                       &fraction_b) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul(fraction_a, fraction_b, &fraction_result) ==
         TERLAN_NDARRAY_OK);
  assert_values(fraction_result, fraction_expected, 4, 1e-12);

  assert(terlan_ndarray_v1_delete(left) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(right) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(fraction_a) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(fraction_b) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(fraction_result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_identity_zero_and_non_finite_matmul(void) {
  const int64_t square_shape[] = {2, 2};
  const double values[] = {2.0, -3.0, 4.5, 7.0};
  const double identity[] = {1.0, 0.0, 0.0, 1.0};
  const int64_t empty_left_shape[] = {2, 0};
  const int64_t empty_right_shape[] = {0, 3};
  const int64_t empty_rows_shape[] = {0, 2};
  const int64_t empty_columns_shape[] = {2, 0};
  const int64_t zero_result_shape[] = {2, 3};
  const double zeroes[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const int64_t nan_shape[] = {1, 1};
  const double nan_value[] = {NAN};
  const double one[] = {1.0};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle eye = NULL;
  TerlanNdArrayHandle copied = NULL;
  TerlanNdArrayHandle empty_left = NULL;
  TerlanNdArrayHandle empty_right = NULL;
  TerlanNdArrayHandle zero_result = NULL;
  TerlanNdArrayHandle empty_rows = NULL;
  TerlanNdArrayHandle empty_columns = NULL;
  TerlanNdArrayHandle empty_row_output = NULL;
  TerlanNdArrayHandle empty_column_output = NULL;
  TerlanNdArrayHandle nan_array = NULL;
  TerlanNdArrayHandle one_array = NULL;
  TerlanNdArrayHandle nan_result = NULL;

  assert(terlan_ndarray_v1_from_floats(values, 4, square_shape, 2, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(identity, 4, square_shape, 2, &eye) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul(source, eye, &copied) == TERLAN_NDARRAY_OK);
  assert_values(copied, values, 4, 0.0);

  assert(terlan_ndarray_v1_from_floats(NULL, 0, empty_left_shape, 2,
                                       &empty_left) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(NULL, 0, empty_right_shape, 2,
                                       &empty_right) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul(empty_left, empty_right, &zero_result) ==
         TERLAN_NDARRAY_OK);
  assert_shape(zero_result, zero_result_shape, 2);
  assert_values(zero_result, zeroes, 6, 0.0);

  assert(terlan_ndarray_v1_from_floats(NULL, 0, empty_rows_shape, 2,
                                       &empty_rows) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(NULL, 0, empty_columns_shape, 2,
                                       &empty_columns) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul(empty_rows, eye, &empty_row_output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(empty_row_output, empty_rows_shape, 2);
  assert_values(empty_row_output, NULL, 0, 0.0);
  assert(terlan_ndarray_v1_matmul(source, empty_columns,
                                  &empty_column_output) == TERLAN_NDARRAY_OK);
  assert_shape(empty_column_output, empty_columns_shape, 2);
  assert_values(empty_column_output, NULL, 0, 0.0);

  assert(terlan_ndarray_v1_from_floats(nan_value, 1, nan_shape, 2,
                                       &nan_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(one, 1, nan_shape, 2, &one_array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul(nan_array, one_array, &nan_result) ==
         TERLAN_NDARRAY_OK);
  assert_values(nan_result, nan_value, 1, 0.0);

  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(eye) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(copied) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty_left) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty_right) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(zero_result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty_rows) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty_columns) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty_row_output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty_column_output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(nan_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(one_array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(nan_result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_matmul_rejections_and_allocation_cleanup(void) {
  const int64_t matrix_shape[] = {2, 2};
  const int64_t bad_shape[] = {3, 1};
  const double matrix_values[] = {1, 2, 3, 4};
  const double bad_values[] = {1, 2, 3};
  TerlanNdArrayHandle matrix = NULL;
  TerlanNdArrayHandle peer = NULL;
  TerlanNdArrayHandle bad = NULL;
  TerlanNdArrayHandle result = NULL;
  int64_t fail_after;

  assert(terlan_ndarray_v1_from_floats(matrix_values, 4, matrix_shape, 2,
                                       &matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(matrix_values, 4, matrix_shape, 2,
                                       &peer) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(bad_values, 3, bad_shape, 2, &bad) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul(matrix, bad, &result) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(result == NULL);
  assert(terlan_ndarray_v1_matmul(matrix, peer, NULL) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);

  for (fail_after = 0; fail_after < 5; ++fail_after) {
    terlan_ndarray_test_fail_allocation_after(fail_after);
    assert(terlan_ndarray_v1_matmul(matrix, peer, &result) ==
           TERLAN_NDARRAY_ALLOCATION_FAILED);
    assert(result == NULL);
    assert(terlan_ndarray_v1_live_count() == 3);
    terlan_ndarray_test_reset_allocation_failures();
  }

  assert(terlan_ndarray_v1_delete(matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(peer) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(bad) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

int main(void) {
  test_rectangular_and_fractional_matmul();
  test_identity_zero_and_non_finite_matmul();
  test_matmul_rejections_and_allocation_cleanup();
  return 0;
}
