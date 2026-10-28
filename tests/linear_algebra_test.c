#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static void assert_shape(TerlanNdArrayHandle array, const int64_t *expected,
                         int64_t rank) {
  int64_t actual_rank = -1;
  int64_t *actual_shape = NULL;
  int64_t dimension;
  assert(terlan_ndarray_v1_rank(array, &actual_rank) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_shape(array, &actual_shape) == TERLAN_NDARRAY_OK);
  assert(actual_rank == rank);
  for (dimension = 0; dimension < rank; ++dimension) {
    assert(actual_shape[dimension] == expected[dimension]);
  }
}

static void assert_floats(TerlanNdArrayHandle array, const double *expected,
                          size_t length) {
  double *actual = NULL;
  size_t actual_length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_floats(array, &actual, &actual_length) ==
         TERLAN_NDARRAY_OK);
  assert(actual_length == length);
  for (index = 0; index < length; ++index) {
    assert(fabs(actual[index] - expected[index]) <= 1e-12);
  }
  terlan_ndarray_v1_delete_float_values(actual);
}

static void test_vector_and_matrix_matmul(void) {
  const int64_t vector_shape[] = {3};
  const int64_t matrix_shape[] = {2, 3};
  const int64_t rhs_shape[] = {3, 2};
  const int64_t matrix_vector_shape[] = {2};
  const double lhs_values[] = {1, 2, 3};
  const double rhs_values[] = {4, 5, 6};
  const double matrix_values[] = {1, 2, 3, 4, 5, 6};
  const double matrix_rhs_values[] = {1, 2, 3, 4, 5, 6};
  const double inner_expected[] = {32};
  const double matrix_vector_expected[] = {32, 77};
  const double vector_matrix_expected[] = {22, 28};
  TerlanNdArrayHandle lhs = NULL;
  TerlanNdArrayHandle rhs = NULL;
  TerlanNdArrayHandle matrix = NULL;
  TerlanNdArrayHandle matrix_rhs = NULL;
  TerlanNdArrayHandle result = NULL;

  assert(terlan_ndarray_v1_from_floats(lhs_values, 3, vector_shape, 1, &lhs) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(rhs_values, 3, vector_shape, 1, &rhs) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(matrix_values, 6, matrix_shape, 2,
                                       &matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(matrix_rhs_values, 6, rhs_shape, 2,
                                       &matrix_rhs) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_matmul(lhs, rhs, &result) == TERLAN_NDARRAY_OK);
  assert_shape(result, NULL, 0);
  assert_floats(result, inner_expected, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_matmul(matrix, rhs, &result) == TERLAN_NDARRAY_OK);
  assert_shape(result, matrix_vector_shape, 1);
  assert_floats(result, matrix_vector_expected, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_matmul(lhs, matrix_rhs, &result) ==
         TERLAN_NDARRAY_OK);
  assert_shape(result, matrix_vector_shape, 1);
  assert_floats(result, vector_matrix_expected, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_delete(lhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(rhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(matrix_rhs) == TERLAN_NDARRAY_OK);
}

static void test_batched_broadcast_and_non_contiguous_matmul(void) {
  const int64_t lhs_shape[] = {2, 2, 3};
  const int64_t rhs_shape[] = {1, 3, 2};
  const int64_t output_shape[] = {2, 2, 2};
  const double lhs_values[] = {1, 2, 3, 4, 5, 6,
                               7, 8, 9, 10, 11, 12};
  const double rhs_values[] = {1, 0, 0, 1, 1, 1};
  const double expected[] = {4, 5, 10, 11, 16, 17, 22, 23};
  const int64_t source_shape[] = {3, 2};
  const int64_t vector_shape[] = {3};
  const int64_t vector_output_shape[] = {2};
  const double source_values[] = {1, 2, 3, 4, 5, 6};
  const double vector_values[] = {1, 2, 3};
  const double vector_expected[] = {22, 28};
  TerlanNdArrayHandle lhs = NULL;
  TerlanNdArrayHandle rhs = NULL;
  TerlanNdArrayHandle output = NULL;
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle transposed = NULL;
  TerlanNdArrayHandle vector = NULL;

  assert(terlan_ndarray_v1_from_floats(lhs_values, 12, lhs_shape, 3, &lhs) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(rhs_values, 6, rhs_shape, 3, &rhs) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul(lhs, rhs, &output) == TERLAN_NDARRAY_OK);
  assert_shape(output, output_shape, 3);
  assert_floats(output, expected, 8);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_floats(source_values, 6, source_shape, 2,
                                       &source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_transpose(source, 0, 1, &transposed) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(vector_values, 3, vector_shape, 1,
                                       &vector) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul(transposed, vector, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, vector_output_shape, 1);
  assert_floats(output, vector_expected, 2);

  assert(terlan_ndarray_v1_delete(lhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(rhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(transposed) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(vector) == TERLAN_NDARRAY_OK);
}

static void test_promoted_and_complex_matmul(void) {
  const int64_t matrix_shape[] = {2, 2};
  const int64_t int_values[] = {1, 2, 3, 4};
  const double float_values[] = {0.5, 1.0, 1.5, 2.0};
  const double expected[] = {3.5, 5.0, 7.5, 11.0};
  const int64_t vector_shape[] = {2};
  const double real_lhs[] = {1, 3};
  const double imag_lhs[] = {2, -1};
  const double real_rhs[] = {2, 1};
  const double imag_rhs[] = {-1, 2};
  TerlanNdArrayScalar complex_value;
  int32_t dtype = 0;
  TerlanNdArrayHandle ints = NULL;
  TerlanNdArrayHandle floats = NULL;
  TerlanNdArrayHandle complex_lhs = NULL;
  TerlanNdArrayHandle complex_rhs = NULL;
  TerlanNdArrayHandle result = NULL;

  assert(terlan_ndarray_v1_from_ints(int_values, 4, matrix_shape, 2, &ints) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(float_values, 4, matrix_shape, 2,
                                       &floats) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul(ints, floats, &result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dtype(result, &dtype) == TERLAN_NDARRAY_OK);
  assert(dtype == TERLAN_NDARRAY_DTYPE_FLOAT64);
  assert_floats(result, expected, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_complex(
             real_lhs, 2, imag_lhs, 2, vector_shape, 1,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &complex_lhs) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_complex(
             real_rhs, 2, imag_rhs, 2, vector_shape, 1,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &complex_rhs) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul(complex_lhs, complex_rhs, &result) ==
         TERLAN_NDARRAY_OK);
  assert(dtype_load(array_const_element_pointer(result, 0), result->dtype,
                    &complex_value) == TERLAN_NDARRAY_OK);
  assert(fabsl(complex_value.real - 9.0L) <= 1e-12L);
  assert(fabsl(complex_value.imag - 8.0L) <= 1e-12L);

  assert(terlan_ndarray_v1_delete(ints) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(floats) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(complex_lhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(complex_rhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
}

static void test_numpy_dot_shapes(void) {
  const int64_t array_shape[] = {2, 2};
  const double scalar_value[] = {2};
  const double array_values[] = {1, 2, 3, 4};
  const double scalar_expected[] = {2, 4, 6, 8};
  const int64_t lhs_shape[] = {2, 3};
  const int64_t rhs_shape[] = {2, 3, 2};
  const int64_t output_shape[] = {2, 2, 2};
  const double lhs_values[] = {1, 2, 3, 4, 5, 6};
  const double rhs_values[] = {1, 0, 0, 1, 1, 1, 2, 1, 1, 0, 0, 2};
  const double expected[] = {4, 5, 4, 7, 10, 11, 13, 16};
  TerlanNdArrayHandle scalar = NULL;
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle lhs = NULL;
  TerlanNdArrayHandle rhs = NULL;
  TerlanNdArrayHandle output = NULL;

  assert(terlan_ndarray_v1_from_floats(scalar_value, 1, NULL, 0, &scalar) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(array_values, 4, array_shape, 2,
                                       &array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dot(scalar, array, &output) == TERLAN_NDARRAY_OK);
  assert_shape(output, array_shape, 2);
  assert_floats(output, scalar_expected, 4);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_floats(lhs_values, 6, lhs_shape, 2, &lhs) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(rhs_values, 12, rhs_shape, 3, &rhs) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dot(lhs, rhs, &output) == TERLAN_NDARRAY_OK);
  assert_shape(output, output_shape, 3);
  assert_floats(output, expected, 8);

  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(lhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(rhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
}

static void test_numpy_dot_rank_matrix_and_orientations(void) {
  const int64_t matrix_shape[] = {4, 2};
  const int64_t vector_two_shape[] = {2};
  const int64_t vector_four_shape[] = {4};
  const int64_t column_two_shape[] = {2, 1};
  const int64_t row_two_shape[] = {1, 2};
  const int64_t column_three_shape[] = {3, 1};
  const int64_t row_three_shape[] = {1, 3};
  const int64_t singleton_shape[] = {1};
  const double matrix_values[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const double vector_two_values[] = {1, 2};
  const double vector_four_values[] = {1, 2, 3, 4};
  const double column_two_values[] = {1, 2};
  const double row_two_values[] = {1, 2};
  const double column_three_values[] = {1, 1, 1};
  const double row_three_values[] = {1, 1, 1};
  const double singleton_five[] = {5.3};
  const double singleton_six[] = {6.2};
  TerlanNdArrayHandle matrix = NULL;
  TerlanNdArrayHandle transposed = NULL;
  TerlanNdArrayHandle vector_two = NULL;
  TerlanNdArrayHandle vector_four = NULL;
  TerlanNdArrayHandle column_two = NULL;
  TerlanNdArrayHandle row_two = NULL;
  TerlanNdArrayHandle column_three = NULL;
  TerlanNdArrayHandle row_three = NULL;
  TerlanNdArrayHandle five = NULL;
  TerlanNdArrayHandle six = NULL;
  TerlanNdArrayHandle output = NULL;

  assert(terlan_ndarray_v1_from_floats(matrix_values, 8, matrix_shape, 2,
                                       &matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_transpose(matrix, 0, 1, &transposed) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(vector_two_values, 2,
                                       vector_two_shape, 1, &vector_two) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(vector_four_values, 4,
                                       vector_four_shape, 1, &vector_four) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(column_two_values, 2,
                                       column_two_shape, 2, &column_two) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(row_two_values, 2, row_two_shape, 2,
                                       &row_two) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(column_three_values, 3,
                                       column_three_shape, 2,
                                       &column_three) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(row_three_values, 3, row_three_shape, 2,
                                       &row_three) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(singleton_five, 1, singleton_shape, 1,
                                       &five) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(singleton_six, 1, singleton_shape, 1,
                                       &six) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_dot(transposed, matrix, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, (const int64_t[]){2, 2}, 2);
  assert_floats(output, (const double[]){84, 100, 100, 120}, 4);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dot(matrix, column_two, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, (const int64_t[]){4, 1}, 2);
  assert_floats(output, (const double[]){5, 11, 17, 23}, 4);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dot(matrix, vector_two, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, (const int64_t[]){4}, 1);
  assert_floats(output, (const double[]){5, 11, 17, 23}, 4);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dot(vector_four, matrix, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, (const int64_t[]){2}, 1);
  assert_floats(output, (const double[]){50, 60}, 2);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dot(row_two, transposed, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, (const int64_t[]){1, 4}, 2);
  assert_floats(output, (const double[]){5, 11, 17, 23}, 4);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dot(transposed, vector_four, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, (const int64_t[]){2}, 1);
  assert_floats(output, (const double[]){50, 60}, 2);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dot(column_two, row_two, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, (const int64_t[]){2, 2}, 2);
  assert_floats(output, (const double[]){1, 2, 2, 4}, 4);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dot(row_two, column_two, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, (const int64_t[]){1, 1}, 2);
  assert_floats(output, (const double[]){5}, 1);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dot(column_three, five, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, (const int64_t[]){3}, 1);
  assert_floats(output, (const double[]){5.3, 5.3, 5.3}, 3);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dot(six, row_three, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, (const int64_t[]){3}, 1);
  assert_floats(output, (const double[]){6.2, 6.2, 6.2}, 3);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_delete(six) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(five) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(row_three) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(column_three) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(row_two) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(column_two) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(vector_four) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(vector_two) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(transposed) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(matrix) == TERLAN_NDARRAY_OK);
}

static void test_numpy_dot_scalar_vector_matrix_shape_matrix(void) {
  const int64_t vector_shape[] = {1};
  const int64_t matrix_shape[] = {1, 1};
  const int64_t expected_ranks[] = {0, 1, 2, 1, 0, 1, 2, 1, 2};
  const int64_t *expected_shapes[] = {
      NULL,         vector_shape, matrix_shape,
      vector_shape, NULL,         vector_shape,
      matrix_shape, vector_shape, matrix_shape};
  TerlanNdArrayHandle values[3] = {NULL, NULL, NULL};
  size_t lhs_index;
  size_t rhs_index;
  size_t output_index = 0;

  assert(terlan_ndarray_v1_zeros(NULL, 0, TERLAN_NDARRAY_DTYPE_FLOAT64,
                                &values[0]) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_zeros(vector_shape, 1,
                                TERLAN_NDARRAY_DTYPE_FLOAT64, &values[1]) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_zeros(matrix_shape, 2,
                                TERLAN_NDARRAY_DTYPE_FLOAT64, &values[2]) ==
         TERLAN_NDARRAY_OK);
  for (lhs_index = 0; lhs_index < 3; ++lhs_index) {
    for (rhs_index = 0; rhs_index < 3; ++rhs_index) {
      TerlanNdArrayHandle output = NULL;
      assert(terlan_ndarray_v1_dot(values[lhs_index], values[rhs_index],
                                   &output) == TERLAN_NDARRAY_OK);
      assert_shape(output, expected_shapes[output_index],
                   expected_ranks[output_index]);
      assert_floats(output, (const double[]){0}, 1);
      assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
      ++output_index;
    }
  }
  assert(terlan_ndarray_v1_delete(values[2]) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(values[1]) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(values[0]) == TERLAN_NDARRAY_OK);
}

static void test_inner_outer_vdot_and_trace(void) {
  const int64_t matrix_shape[] = {2, 3};
  const int64_t inner_shape[] = {2, 2};
  const double matrix_values[] = {1, 2, 3, 4, 5, 6};
  const double inner_expected[] = {14, 32, 32, 77};
  const int64_t lhs_shape[] = {2};
  const int64_t rhs_shape[] = {3};
  const int64_t outer_shape[] = {2, 3};
  const double lhs_values[] = {1, 2};
  const double rhs_values[] = {3, 4, 5};
  const double outer_expected[] = {3, 4, 5, 6, 8, 10};
  const double trace_values[] = {1, 2, 3, 4};
  const double trace_expected[] = {5};
  const double real_lhs[] = {1, 3};
  const double imag_lhs[] = {2, -1};
  const double real_rhs[] = {2, 1};
  const double imag_rhs[] = {-1, 2};
  TerlanNdArrayScalar vdot_value;
  TerlanNdArrayHandle matrix = NULL;
  TerlanNdArrayHandle lhs = NULL;
  TerlanNdArrayHandle rhs = NULL;
  TerlanNdArrayHandle complex_lhs = NULL;
  TerlanNdArrayHandle complex_rhs = NULL;
  TerlanNdArrayHandle trace_source = NULL;
  TerlanNdArrayHandle output = NULL;

  assert(terlan_ndarray_v1_from_floats(matrix_values, 6, matrix_shape, 2,
                                       &matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_inner(matrix, matrix, &output) == TERLAN_NDARRAY_OK);
  assert_shape(output, inner_shape, 2);
  assert_floats(output, inner_expected, 4);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_floats(lhs_values, 2, lhs_shape, 1, &lhs) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(rhs_values, 3, rhs_shape, 1, &rhs) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_outer(lhs, rhs, &output) == TERLAN_NDARRAY_OK);
  assert_shape(output, outer_shape, 2);
  assert_floats(output, outer_expected, 6);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_complex(
             real_lhs, 2, imag_lhs, 2, lhs_shape, 1,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &complex_lhs) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_complex(
             real_rhs, 2, imag_rhs, 2, lhs_shape, 1,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &complex_rhs) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_vdot(complex_lhs, complex_rhs, &output) ==
         TERLAN_NDARRAY_OK);
  assert(dtype_load(array_const_element_pointer(output, 0), output->dtype,
                    &vdot_value) == TERLAN_NDARRAY_OK);
  assert(fabsl(vdot_value.real - 1.0L) <= 1e-12L);
  assert(fabsl(vdot_value.imag - 2.0L) <= 1e-12L);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_floats(trace_values, 4, inner_shape, 2,
                                       &trace_source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_trace(trace_source, 0, 0, 1, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, NULL, 0);
  assert_floats(output, trace_expected, 1);

  assert(terlan_ndarray_v1_delete(matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(lhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(rhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(complex_lhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(complex_rhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(trace_source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
}

static void test_linear_algebra_rejections(void) {
  const int64_t scalar_shape_rank = 0;
  const int64_t bad_shape[] = {2};
  const double scalar_value[] = {1};
  const double bad_values[] = {1, 2};
  TerlanNdArrayHandle scalar = NULL;
  TerlanNdArrayHandle bad = NULL;
  TerlanNdArrayHandle output = NULL;
  (void)scalar_shape_rank;
  assert(terlan_ndarray_v1_from_floats(scalar_value, 1, NULL, 0, &scalar) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(bad_values, 2, bad_shape, 1, &bad) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul(scalar, bad, &output) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(output == NULL);
  assert(terlan_ndarray_v1_dot(scalar, bad, NULL) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(bad) == TERLAN_NDARRAY_OK);
}

static void test_output_destinations_are_atomic_and_operation_specific(void) {
  const int64_t matrix_shape[] = {2, 2};
  const double identity_values[] = {1, 0, 0, 1};
  const double swap_values[] = {0, 1, 1, 0};
  const double zero_values[] = {0, 0, 0, 0};
  TerlanNdArrayHandle identity = NULL;
  TerlanNdArrayHandle swap = NULL;
  TerlanNdArrayHandle destination = NULL;
  TerlanNdArrayHandle strided_storage = NULL;
  TerlanNdArrayHandle strided_destination = NULL;
  TerlanNdArrayHandle wrong = NULL;
  TerlanNdArrayHandle integer_destination = NULL;
  TerlanNdArrayHandle trace_destination = NULL;
  TerlanNdArrayHandle empty_lhs = NULL;
  TerlanNdArrayHandle empty_rhs = NULL;
  TerlanNdArrayHandle empty_destination = NULL;

  assert(terlan_ndarray_v1_from_floats(identity_values, 4, matrix_shape, 2,
                                       &identity) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(swap_values, 4, matrix_shape, 2,
                                       &swap) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(zero_values, 4, matrix_shape, 2,
                                       &destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dot_into(destination, identity, swap) ==
         TERLAN_NDARRAY_OK);
  assert_floats(destination, swap_values, 4);
  assert(terlan_ndarray_v1_matmul_into(destination, identity, swap) ==
         TERLAN_NDARRAY_OK);
  assert_floats(destination, swap_values, 4);

  assert(terlan_ndarray_v1_dot_into(swap, identity, swap) ==
         TERLAN_NDARRAY_OK);
  assert_floats(swap, swap_values, 4);

  assert(terlan_ndarray_v1_from_floats(zero_values, 4, matrix_shape, 2,
                                       &strided_storage) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_transpose(strided_storage, 0, 1,
                                     &strided_destination) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dot_into(strided_destination, identity, swap) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert_floats(strided_destination, zero_values, 4);
  assert(terlan_ndarray_v1_matmul_into(strided_destination, identity, swap) ==
         TERLAN_NDARRAY_OK);
  assert_floats(strided_destination, swap_values, 4);

  assert(terlan_ndarray_v1_from_floats((const double[]){7, 8}, 2,
                                       (const int64_t[]){2}, 1, &wrong) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dot_into(wrong, identity, swap) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert_floats(wrong, (const double[]){7, 8}, 2);
  assert(terlan_ndarray_v1_from_ints((const int64_t[]){7, 8, 9, 10}, 4,
                                     matrix_shape, 2,
                                     &integer_destination) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul_into(integer_destination, identity, swap) ==
         TERLAN_NDARRAY_DTYPE_MISMATCH);

  assert(terlan_ndarray_v1_from_floats((const double[]){0}, 1, NULL, 0,
                                       &trace_destination) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_trace_into(trace_destination, identity, 0, 0, 1) ==
         TERLAN_NDARRAY_OK);
  assert_floats(trace_destination, (const double[]){2}, 1);

  assert(terlan_ndarray_v1_from_floats(NULL, 0, (const int64_t[]){0}, 1,
                                       &empty_lhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(NULL, 0, (const int64_t[]){0}, 1,
                                       &empty_rhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats((const double[]){1}, 1, NULL, 0,
                                       &empty_destination) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dot_into(empty_destination, empty_lhs, empty_rhs) ==
         TERLAN_NDARRAY_OK);
  assert_floats(empty_destination, (const double[]){0}, 1);

  assert(terlan_ndarray_v1_delete(empty_destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty_rhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty_lhs) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(trace_destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(integer_destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(wrong) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(strided_destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(strided_storage) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(swap) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(identity) == TERLAN_NDARRAY_OK);
}

static void test_batched_matmul_output_overlap_matches_numpy(void) {
  const int64_t shape[] = {2, 3, 3};
  const double values[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,
                           9, 10, 11, 12, 13, 14, 15, 16, 17};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle expected = NULL;
  TerlanNdArrayHandle reversed_source = NULL;
  TerlanNdArrayHandle reversed_expected = NULL;
  TerlanNdArrayHandle reversed_destination = NULL;

  assert(terlan_ndarray_v1_from_floats(values, 18, shape, 3, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul(source, source, &expected) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul_into(source, source, source) ==
         TERLAN_NDARRAY_OK);
  {
    double *expected_values = NULL;
    size_t expected_length = 0;
    assert(terlan_ndarray_v1_to_floats(expected, &expected_values,
                                       &expected_length) == TERLAN_NDARRAY_OK);
    assert_floats(source, expected_values, expected_length);
    terlan_ndarray_v1_delete_float_values(expected_values);
  }
  assert(terlan_ndarray_v1_delete(expected) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_floats(values, 18, shape, 3,
                                       &reversed_source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul(reversed_source, reversed_source,
                                  &reversed_expected) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_slice_axis(reversed_source, 0, INT64_MIN,
                                     INT64_MIN, -1, &reversed_destination) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul_into(reversed_destination, reversed_source,
                                       reversed_source) == TERLAN_NDARRAY_OK);
  {
    double *expected_values = NULL;
    size_t expected_length = 0;
    assert(terlan_ndarray_v1_to_floats(reversed_expected, &expected_values,
                                       &expected_length) == TERLAN_NDARRAY_OK);
    assert_floats(reversed_destination, expected_values, expected_length);
    terlan_ndarray_v1_delete_float_values(expected_values);
  }

  assert(terlan_ndarray_v1_delete(reversed_destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(reversed_expected) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(reversed_source) == TERLAN_NDARRAY_OK);
}

static void test_trace_offsets_axes_and_output_match_numpy(void) {
  const double matrix_values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  const double cube_values[] = {0, 1, 2, 3, 4, 5, 6, 7};
  TerlanNdArrayHandle matrix = NULL;
  TerlanNdArrayHandle cube = NULL;
  TerlanNdArrayHandle output = NULL;
  TerlanNdArrayHandle destination = NULL;

  assert(terlan_ndarray_v1_from_floats(matrix_values, 12,
                                       (const int64_t[]){3, 4}, 2, &matrix) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_trace(matrix, 0, 0, 1, &output) ==
         TERLAN_NDARRAY_OK);
  assert_floats(output, (const double[]){15}, 1);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_trace(matrix, 1, 0, 1, &output) ==
         TERLAN_NDARRAY_OK);
  assert_floats(output, (const double[]){18}, 1);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_trace(matrix, -1, 0, 1, &output) ==
         TERLAN_NDARRAY_OK);
  assert_floats(output, (const double[]){13}, 1);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_floats(cube_values, 8,
                                       (const int64_t[]){2, 2, 2}, 3, &cube) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_trace(cube, 0, 0, 1, &output) ==
         TERLAN_NDARRAY_OK);
  assert_floats(output, (const double[]){6, 8}, 2);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_trace(cube, 1, 0, 1, &output) ==
         TERLAN_NDARRAY_OK);
  assert_floats(output, (const double[]){2, 3}, 2);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_trace(cube, -1, 0, 1, &output) ==
         TERLAN_NDARRAY_OK);
  assert_floats(output, (const double[]){4, 5}, 2);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_trace(cube, 0, 0, 2, &output) ==
         TERLAN_NDARRAY_OK);
  assert_floats(output, (const double[]){5, 9}, 2);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_trace(cube, 0, 1, 2, &output) ==
         TERLAN_NDARRAY_OK);
  assert_floats(output, (const double[]){3, 11}, 2);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_trace(cube, 1, 0, 2, &output) ==
         TERLAN_NDARRAY_OK);
  assert_floats(output, (const double[]){1, 3}, 2);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_floats((const double[]){1}, 1, NULL, 0,
                                       &destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_trace_into(destination, matrix, 0, 0, 1) ==
         TERLAN_NDARRAY_OK);
  assert_floats(destination, (const double[]){15}, 1);

  assert(terlan_ndarray_v1_delete(destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(cube) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(matrix) == TERLAN_NDARRAY_OK);
}

static void test_numpy_matmul_common_contract(void) {
  const int64_t vector_shape[] = {2};
  const int64_t matrix_shape[] = {2, 2};
  const int64_t wrong_shape[] = {3};
  const double vector_values[] = {1, 2};
  const double matrix_values[] = {1, 2, 3, 4};
  const int64_t integer_values[] = {1, 2, 3, 4};
  const double wrong_values[] = {1, 2, 3};
  const double scalar_value[] = {2};
  TerlanNdArrayHandle vector = NULL;
  TerlanNdArrayHandle matrix = NULL;
  TerlanNdArrayHandle integers = NULL;
  TerlanNdArrayHandle wrong = NULL;
  TerlanNdArrayHandle scalar = NULL;
  TerlanNdArrayHandle output = NULL;
  int32_t dtype = 0;

  assert(terlan_ndarray_v1_from_floats(vector_values, 2, vector_shape, 1,
                                       &vector) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(matrix_values, 4, matrix_shape, 2,
                                       &matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(integer_values, 4, matrix_shape, 2,
                                     &integers) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(wrong_values, 3, wrong_shape, 1,
                                       &wrong) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(scalar_value, 1, NULL, 0, &scalar) ==
         TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_matmul(vector, vector, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, NULL, 0);
  assert_floats(output, (const double[]){5}, 1);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  output = NULL;

  assert(terlan_ndarray_v1_matmul(matrix, vector, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, vector_shape, 1);
  assert_floats(output, (const double[]){5, 11}, 2);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  output = NULL;

  assert(terlan_ndarray_v1_matmul(vector, matrix, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, vector_shape, 1);
  assert_floats(output, (const double[]){7, 10}, 2);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  output = NULL;

  assert(terlan_ndarray_v1_matmul(matrix, matrix, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, matrix_shape, 2);
  assert_floats(output, (const double[]){7, 10, 15, 22}, 4);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  output = NULL;

  assert(terlan_ndarray_v1_matmul(integers, matrix, &output) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dtype(output, &dtype) == TERLAN_NDARRAY_OK);
  assert(dtype == TERLAN_NDARRAY_DTYPE_FLOAT64);
  assert_floats(output, (const double[]){7, 10, 15, 22}, 4);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  output = NULL;

  assert(terlan_ndarray_v1_matmul(scalar, matrix, &output) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(output == NULL);
  assert(terlan_ndarray_v1_matmul(matrix, wrong, &output) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(output == NULL);

  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(wrong) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(integers) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(vector) == TERLAN_NDARRAY_OK);
}

static void test_numpy_matmul_bool_and_empty_contract(void) {
  const uint8_t bool_values[] = {1, 0, 1, 1};
  const uint8_t bool_expected[] = {1, 0, 1, 1};
  const double sentinel[] = {7};
  TerlanNdArrayHandle boolean = NULL;
  TerlanNdArrayHandle bool_result = NULL;
  TerlanNdArrayHandle empty_batch = NULL;
  TerlanNdArrayHandle empty_result = NULL;
  TerlanNdArrayHandle wrong_destination = NULL;

  assert(terlan_ndarray_v1_from_bools(bool_values, 4,
                                      (const int64_t[]){2, 2}, 2,
                                      &boolean) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul(boolean, boolean, &bool_result) ==
         TERLAN_NDARRAY_OK);
  assert_shape(bool_result, (const int64_t[]){2, 2}, 2);
  {
    uint8_t *values = NULL;
    size_t length = 0;
    size_t index;
    assert(terlan_ndarray_v1_to_bools(bool_result, &values, &length) ==
           TERLAN_NDARRAY_OK);
    assert(length == 4);
    for (index = 0; index < length; ++index) {
      assert(values[index] == bool_expected[index]);
    }
    terlan_ndarray_v1_delete_bool_values(values);
  }
  assert(terlan_ndarray_v1_delete(bool_result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(boolean) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_floats(NULL, 0,
                                       (const int64_t[]){0, 1, 1}, 3,
                                       &empty_batch) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul(empty_batch, empty_batch, &empty_result) ==
         TERLAN_NDARRAY_OK);
  assert_shape(empty_result, (const int64_t[]){0, 1, 1}, 3);
  assert(empty_result->numel == 0);
  assert(terlan_ndarray_v1_from_floats(sentinel, 1,
                                       (const int64_t[]){1, 1, 1}, 3,
                                       &wrong_destination) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_matmul_into(wrong_destination, empty_batch,
                                       empty_batch) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert_floats(wrong_destination, sentinel, 1);
  assert(terlan_ndarray_v1_delete(wrong_destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty_result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty_batch) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

int main(void) {
  test_vector_and_matrix_matmul();
  test_batched_broadcast_and_non_contiguous_matmul();
  test_promoted_and_complex_matmul();
  test_numpy_dot_shapes();
  test_numpy_dot_rank_matrix_and_orientations();
  test_numpy_dot_scalar_vector_matrix_shape_matrix();
  test_inner_outer_vdot_and_trace();
  test_linear_algebra_rejections();
  test_output_destinations_are_atomic_and_operation_specific();
  test_batched_matmul_output_overlap_matches_numpy();
  test_trace_offsets_axes_and_output_match_numpy();
  test_numpy_matmul_common_contract();
  test_numpy_matmul_bool_and_empty_contract();
  assert(terlan_ndarray_v1_live_count() == 0);
  return 0;
}
