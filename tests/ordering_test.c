#include "../bindings/include/terlan_ndarray.h"

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

static void assert_shape(TerlanNdArrayHandle array, const int64_t *expected,
                         int64_t expected_rank) {
  int64_t *shape = NULL;
  int64_t rank = -1;
  int64_t dimension;
  assert(terlan_ndarray_v1_rank(array, &rank) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_shape(array, &shape) == TERLAN_NDARRAY_OK);
  assert(rank == expected_rank);
  for (dimension = 0; dimension < rank; ++dimension) {
    assert(shape[dimension] == expected[dimension]);
  }
}

static void test_sort_and_argsort_each_axis(void) {
  const int64_t values[] = {3, 1, 2, 9, 7, 8};
  const int64_t shape[] = {2, 3};
  const int64_t sorted_last_values[] = {1, 2, 3, 7, 8, 9};
  const int64_t sorted_first_values[] = {3, 1, 2, 9, 7, 8};
  const int64_t indices[] = {1, 2, 0, 1, 2, 0};
  const int64_t flat_values[] = {1, 2, 3, 7, 8, 9};
  const int64_t flat_indices[] = {1, 2, 0, 4, 5, 3};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle inplace = NULL;
  TerlanNdArrayHandle sorted_last = NULL;
  TerlanNdArrayHandle sorted_first = NULL;
  TerlanNdArrayHandle order = NULL;
  TerlanNdArrayHandle flat = NULL;
  TerlanNdArrayHandle flat_order = NULL;
  assert(terlan_ndarray_v1_from_ints(values, 6, shape, 2, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sort(array, -1, &sorted_last) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sort(array, 0, &sorted_first) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argsort(array, 1, &order) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sort_flat(array, &flat) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argsort_flat(array, &flat_order) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(values, 6, shape, 2, &inplace) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sort_inplace(inplace, -1) == TERLAN_NDARRAY_OK);
  assert_int_values(sorted_last, sorted_last_values, 6);
  assert_int_values(sorted_first, sorted_first_values, 6);
  assert_int_values(order, indices, 6);
  assert_int_values(flat, flat_values, 6);
  assert_int_values(flat_order, flat_indices, 6);
  assert_int_values(inplace, sorted_last_values, 6);
  assert_int_values(array, values, 6);
  assert(terlan_ndarray_v1_delete(flat_order) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(flat) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(inplace) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(order) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(sorted_first) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(sorted_last) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_noncontiguous_nan_and_stable_ties(void) {
  const double values[] = {NAN, 2.0, 2.0, -1.0, 4.0, 3.0};
  const int64_t shape[] = {2, 3};
  const int64_t expected_indices[] = {1, 0, 0, 1, 0, 1};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle transposed = NULL;
  TerlanNdArrayHandle order = NULL;
  TerlanNdArrayHandle sorted = NULL;
  double *result = NULL;
  size_t length = 0;
  assert(terlan_ndarray_v1_from_floats(values, 6, shape, 2, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_transpose(array, 0, 1, &transposed) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argsort(transposed, 1, &order) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sort(transposed, 1, &sorted) == TERLAN_NDARRAY_OK);
  assert_int_values(order, expected_indices, 6);
  assert(terlan_ndarray_v1_to_floats(sorted, &result, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == 6);
  assert(result[0] == -1.0 && isnan(result[1]));
  assert(result[2] == 2.0 && result[3] == 4.0);
  assert(result[4] == 2.0 && result[5] == 3.0);
  free(result);
  assert(terlan_ndarray_v1_delete(sorted) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(order) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(transposed) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_complex_nan_order_matches_numpy(void) {
  const double real[] = {NAN, NAN, NAN, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0};
  const double imag[] = {NAN, 1.0, 0.0, NAN, NAN, 1.0, 0.0, 1.0, 0.0};
  const double expected_real[] = {0.0, 0.0, 0.0, 1.0, 1.0,
                                  1.0, NAN, NAN, NAN};
  const double expected_imag[] = {0.0, 1.0, NAN, 0.0, 1.0,
                                  NAN, 0.0, 1.0, NAN};
  const int64_t shape[] = {9};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle sorted = NULL;
  TerlanNdArrayHandle partitioned = NULL;
  TerlanNdArrayHandle order = NULL;
  double *parts = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_from_complex(
             real, 9, imag, 9, shape, 1, TERLAN_NDARRAY_DTYPE_COMPLEX128,
             &array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sort(array, 0, &sorted) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_complex_reals(sorted, &parts, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == 9);
  for (index = 0; index < 9; ++index) {
    assert((isnan(parts[index]) && isnan(expected_real[index])) ||
           parts[index] == expected_real[index]);
  }
  free(parts);
  assert(terlan_ndarray_v1_complex_imags(sorted, &parts, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == 9);
  for (index = 0; index < 9; ++index) {
    assert((isnan(parts[index]) && isnan(expected_imag[index])) ||
           parts[index] == expected_imag[index]);
  }
  free(parts);
  assert(terlan_ndarray_v1_partition(array, 4, 0, &partitioned) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argpartition(array, 4, 0, &order) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_complex_reals(partitioned, &parts, &length) ==
         TERLAN_NDARRAY_OK);
  for (index = 0; index < 9; ++index) {
    assert((isnan(parts[index]) && isnan(expected_real[index])) ||
           parts[index] == expected_real[index]);
  }
  free(parts);
  assert_int_values(order, (int64_t[]){8, 7, 4, 6, 5, 3, 2, 1, 0}, 9);
  assert(terlan_ndarray_v1_delete(order) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(partitioned) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(sorted) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_sort_all_admitted_scalar_dtypes(void) {
  const int32_t dtypes[] = {
      TERLAN_NDARRAY_DTYPE_UINT8,  TERLAN_NDARRAY_DTYPE_UINT16,
      TERLAN_NDARRAY_DTYPE_UINT32, TERLAN_NDARRAY_DTYPE_UINT64,
      TERLAN_NDARRAY_DTYPE_INT8,   TERLAN_NDARRAY_DTYPE_INT16,
      TERLAN_NDARRAY_DTYPE_INT32,  TERLAN_NDARRAY_DTYPE_INT64,
      TERLAN_NDARRAY_DTYPE_FLOAT16, TERLAN_NDARRAY_DTYPE_FLOAT32,
      TERLAN_NDARRAY_DTYPE_FLOAT64};
  const int64_t shape[] = {101};
  int64_t reversed[101];
  int64_t expected[101];
  size_t dtype_index;
  int64_t index;
  for (index = 0; index < 101; ++index) {
    reversed[index] = 100 - index;
    expected[index] = index;
  }
  for (dtype_index = 0; dtype_index < sizeof(dtypes) / sizeof(dtypes[0]);
       ++dtype_index) {
    TerlanNdArrayHandle array = NULL;
    TerlanNdArrayHandle sorted = NULL;
    TerlanNdArrayHandle order = NULL;
    assert(terlan_ndarray_v1_from_ints_as(reversed, 101, shape, 1,
                                         dtypes[dtype_index], &array) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_sort(array, 0, &sorted) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_argsort(array, 0, &order) == TERLAN_NDARRAY_OK);
    assert_int_values(sorted, expected, 101);
    assert_int_values(order, reversed, 101);
    assert(terlan_ndarray_v1_delete(order) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(sorted) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_sort_inplace(array, 0) == TERLAN_NDARRAY_OK);
    assert_int_values(array, expected, 101);
    assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  }
}

static void test_argsort_complex_is_stable_and_uses_numpy_nan_order(void) {
  const double real[] = {1.0, 0.0, 1.0, 0.0, NAN, NAN, NAN, 1.0, 0.0};
  const double imag[] = {1.0, 0.0, 0.0, 1.0, 1.0, NAN, 0.0, NAN, NAN};
  const int64_t expected[] = {1, 3, 8, 2, 0, 7, 6, 4, 5};
  const int64_t shape[] = {9};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle order = NULL;
  assert(terlan_ndarray_v1_from_complex(
             real, 9, imag, 9, shape, 1, TERLAN_NDARRAY_DTYPE_COMPLEX128,
             &array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_argsort(array, 0, &order) == TERLAN_NDARRAY_OK);
  assert_int_values(order, expected, 9);
  assert(terlan_ndarray_v1_delete(order) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_searchsorted_left_right_and_nan_order(void) {
  const double sorted_values[] = {1.0, 3.0, 3.0, 5.0, NAN};
  const int64_t sorted_shape[] = {5};
  const double query_values[] = {0.0, 3.0, 4.0, NAN};
  const int64_t query_shape[] = {2, 2};
  const int64_t left_expected[] = {0, 1, 3, 4};
  const int64_t right_expected[] = {0, 3, 3, 5};
  TerlanNdArrayHandle sorted = NULL;
  TerlanNdArrayHandle queries = NULL;
  TerlanNdArrayHandle indexes = NULL;
  assert(terlan_ndarray_v1_from_floats(sorted_values, 5, sorted_shape, 1,
                                       &sorted) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(query_values, 4, query_shape, 2,
                                       &queries) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_searchsorted(sorted, queries, 0, &indexes) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(indexes, left_expected, 4);
  assert(terlan_ndarray_v1_delete(indexes) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_searchsorted(sorted, queries, 1, &indexes) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(indexes, right_expected, 4);
  assert(terlan_ndarray_v1_delete(indexes) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_searchsorted(queries, sorted, 0, &indexes) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(indexes == NULL);
  assert(terlan_ndarray_v1_searchsorted(sorted, queries, 2, &indexes) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(indexes == NULL);
  assert(terlan_ndarray_v1_delete(queries) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(sorted) == TERLAN_NDARRAY_OK);
}

static void test_searchsorted_all_admitted_scalar_dtypes(void) {
  const int32_t dtypes[] = {
      TERLAN_NDARRAY_DTYPE_INT8,   TERLAN_NDARRAY_DTYPE_INT16,
      TERLAN_NDARRAY_DTYPE_INT32,  TERLAN_NDARRAY_DTYPE_INT64,
      TERLAN_NDARRAY_DTYPE_UINT8,  TERLAN_NDARRAY_DTYPE_UINT16,
      TERLAN_NDARRAY_DTYPE_UINT32, TERLAN_NDARRAY_DTYPE_UINT64,
      TERLAN_NDARRAY_DTYPE_FLOAT16, TERLAN_NDARRAY_DTYPE_FLOAT32,
      TERLAN_NDARRAY_DTYPE_FLOAT64};
  const int64_t values[] = {0, 1, 2, 3, 4};
  const int64_t shape[] = {5};
  const int64_t left[] = {0, 1, 2, 3, 4};
  const int64_t right[] = {1, 2, 3, 4, 5};
  size_t index;
  for (index = 0; index < sizeof(dtypes) / sizeof(dtypes[0]); ++index) {
    TerlanNdArrayHandle array = NULL;
    TerlanNdArrayHandle result = NULL;
    if (index < 8) {
      assert(terlan_ndarray_v1_from_ints_as(values, 5, shape, 1,
                                            dtypes[index], &array) == 0);
    } else {
      assert(terlan_ndarray_v1_from_floats_as(
                 (double[]){0.0, 1.0, 2.0, 3.0, 4.0}, 5, shape, 1,
                 dtypes[index], &array) == 0);
    }
    assert(terlan_ndarray_v1_searchsorted(array, array, 0, &result) == 0);
    assert_int_values(result, left, 5);
    assert(terlan_ndarray_v1_delete(result) == 0);
    assert(terlan_ndarray_v1_searchsorted(array, array, 1, &result) == 0);
    assert_int_values(result, right, 5);
    assert(terlan_ndarray_v1_delete(result) == 0);
    assert(terlan_ndarray_v1_delete(array) == 0);
  }
  {
    TerlanNdArrayHandle array = NULL;
    TerlanNdArrayHandle result = NULL;
    assert(terlan_ndarray_v1_from_bools((uint8_t[]){0, 1}, 2,
                                        (int64_t[]){2}, 1, &array) == 0);
    assert(terlan_ndarray_v1_searchsorted(array, array, 0, &result) == 0);
    assert_int_values(result, (int64_t[]){0, 1}, 2);
    assert(terlan_ndarray_v1_delete(result) == 0);
    assert(terlan_ndarray_v1_delete(array) == 0);
  }
  {
    const double real[] = {0.0, 0.0, 0.0, 1.0, 1.0,
                           1.0, NAN, NAN, NAN};
    const double imag[] = {0.0, 1.0, NAN, 0.0, 1.0,
                           NAN, 0.0, 1.0, NAN};
    TerlanNdArrayHandle array = NULL;
    TerlanNdArrayHandle result = NULL;
    assert(terlan_ndarray_v1_from_complex(
               real, 9, imag, 9, (int64_t[]){9}, 1,
               TERLAN_NDARRAY_DTYPE_COMPLEX128, &array) == 0);
    assert(terlan_ndarray_v1_searchsorted(array, array, 0, &result) == 0);
    assert_int_values(result, (int64_t[]){0, 1, 2, 3, 4, 5, 6, 7, 8}, 9);
    assert(terlan_ndarray_v1_delete(result) == 0);
    assert(terlan_ndarray_v1_searchsorted(array, array, 1, &result) == 0);
    assert_int_values(result, (int64_t[]){1, 2, 3, 4, 5, 6, 7, 8, 9}, 9);
    assert(terlan_ndarray_v1_delete(result) == 0);
    assert(terlan_ndarray_v1_delete(array) == 0);
  }
}

static void test_ordering_rejects_invalid_inputs(void) {
  const int64_t value = 1;
  TerlanNdArrayHandle scalar = NULL;
  TerlanNdArrayHandle output = (TerlanNdArrayHandle)1;
  assert(terlan_ndarray_v1_from_ints(&value, 1, NULL, 0, &scalar) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sort(scalar, 0, &output) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(output == NULL);
  assert(terlan_ndarray_v1_argsort(scalar, 0, NULL) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
}

static void test_partition_and_argpartition_satisfy_kth_contract(void) {
  const int64_t values[] = {4, 1, 3, 2};
  const int64_t shape[] = {4};
  const int64_t ordered[] = {1, 2, 3, 4};
  const int64_t indexes[] = {1, 3, 2, 0};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_from_ints(values, 4, shape, 1, &array) == 0);
  assert(terlan_ndarray_v1_partition(array, 2, 0, &output) == 0);
  assert_int_values(output, ordered, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_argpartition(array, -2, 0, &output) == 0);
  assert_int_values(output, indexes, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_partition(array, 4, 0, &output) ==
         TERLAN_NDARRAY_INDEX_OUT_OF_RANGE);
  assert(output == NULL);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_partition_multiple_kths_flattening_and_mutation(void) {
  const int64_t values[] = {6, 1, 5, 2, 4, 3};
  const int64_t shape[] = {2, 3};
  const int64_t kth_storage[] = {0, 99, -1, 99, 1, 99};
  const int64_t kth_shape[] = {6};
  const int64_t slice_shape[] = {3};
  const int64_t axis_sorted[] = {1, 5, 6, 2, 3, 4};
  const int64_t axis_order[] = {1, 2, 0, 0, 2, 1};
  const int64_t flat_sorted[] = {1, 2, 3, 4, 5, 6};
  const int64_t flat_order[] = {1, 3, 5, 4, 2, 0};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle kth_values = NULL;
  TerlanNdArrayHandle kths = NULL;
  TerlanNdArrayHandle output = NULL;
  TerlanNdArrayHandle indices = NULL;
  TerlanNdArrayHandle inplace = NULL;
  assert(terlan_ndarray_v1_from_ints(values, 6, shape, 2, &array) == 0);
  assert(terlan_ndarray_v1_from_ints_as(
             kth_storage, 6, kth_shape, 1, TERLAN_NDARRAY_DTYPE_INT16,
             &kth_values) == 0);
  assert(terlan_ndarray_v1_slice_axis(kth_values, 0, 0, 6, 2, &kths) == 0);
  assert_shape(kths, slice_shape, 1);
  assert(terlan_ndarray_v1_partition_kths(array, kths, -1, &output) == 0);
  assert(terlan_ndarray_v1_argpartition_kths(array, kths, -1, &indices) == 0);
  assert_int_values(output, axis_sorted, 6);
  assert_int_values(indices, axis_order, 6);
  assert_int_values(kths, (int64_t[]){0, -1, 1}, 3);
  assert(terlan_ndarray_v1_delete(indices) == 0);
  assert(terlan_ndarray_v1_delete(output) == 0);

  assert(terlan_ndarray_v1_partition_kths_flat(array, kths, &output) == 0);
  assert(terlan_ndarray_v1_argpartition_kths_flat(array, kths, &indices) == 0);
  assert_shape(output, (int64_t[]){6}, 1);
  assert_int_values(output, flat_sorted, 6);
  assert_int_values(indices, flat_order, 6);
  assert(terlan_ndarray_v1_delete(indices) == 0);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_partition_flat(array, -2, &output) == 0);
  assert(terlan_ndarray_v1_argpartition_flat(array, -2, &indices) == 0);
  assert_int_values(output, flat_sorted, 6);
  assert_int_values(indices, flat_order, 6);
  assert(terlan_ndarray_v1_delete(indices) == 0);
  assert(terlan_ndarray_v1_delete(output) == 0);

  assert(terlan_ndarray_v1_copy(array, &inplace) == 0);
  assert(terlan_ndarray_v1_partition_kths_inplace(inplace, kths, -1) == 0);
  assert_int_values(inplace, axis_sorted, 6);
  assert(terlan_ndarray_v1_delete(inplace) == 0);
  assert(terlan_ndarray_v1_copy(array, &inplace) == 0);
  assert(terlan_ndarray_v1_partition_inplace(inplace, 1, -1) == 0);
  assert_int_values(inplace, axis_sorted, 6);
  assert(terlan_ndarray_v1_delete(inplace) == 0);
  assert_int_values(array, values, 6);
  assert(terlan_ndarray_v1_delete(kths) == 0);
  assert(terlan_ndarray_v1_delete(kth_values) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_partition_empty_axes_and_invalid_kths(void) {
  const int64_t empty_shape[] = {3, 2, 1, 0};
  const int64_t values[] = {3, 2, 1, 0};
  const int64_t shape[] = {4};
  TerlanNdArrayHandle empty = NULL;
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle output = NULL;
  TerlanNdArrayHandle kths = NULL;
  TerlanNdArrayHandle readonly = NULL;
  int64_t axis;
  assert(terlan_ndarray_v1_from_ints(NULL, 0, empty_shape, 4, &empty) == 0);
  for (axis = -4; axis < 4; ++axis) {
    assert(terlan_ndarray_v1_partition(empty, 0, axis, &output) == 0);
    assert_shape(output, empty_shape, 4);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_argpartition(empty, 0, axis, &output) == 0);
    assert_shape(output, empty_shape, 4);
    assert(terlan_ndarray_v1_delete(output) == 0);
  }
  assert(terlan_ndarray_v1_partition_flat(empty, 0, &output) == 0);
  assert_shape(output, (int64_t[]){0}, 1);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_partition(empty, 1, -1, &output) ==
         TERLAN_NDARRAY_INDEX_OUT_OF_RANGE);
  assert(output == NULL);
  assert(terlan_ndarray_v1_delete(empty) == 0);

  assert(terlan_ndarray_v1_from_ints(values, 4, shape, 1, &array) == 0);
  assert(terlan_ndarray_v1_from_ints((int64_t[]){0, 4}, 2,
                                     (int64_t[]){2}, 1, &kths) == 0);
  assert(terlan_ndarray_v1_partition_kths(array, kths, 0, &output) ==
         TERLAN_NDARRAY_INDEX_OUT_OF_RANGE);
  assert(output == NULL);
  assert(terlan_ndarray_v1_delete(kths) == 0);
  assert(terlan_ndarray_v1_partition(array, -5, 0, &output) ==
         TERLAN_NDARRAY_INDEX_OUT_OF_RANGE);
  assert(output == NULL);
  assert(terlan_ndarray_v1_from_floats((double[]){1.0}, 1,
                                       (int64_t[]){1}, 1, &kths) == 0);
  assert(terlan_ndarray_v1_partition_kths(array, kths, 0, &output) ==
         TERLAN_NDARRAY_DTYPE_MISMATCH);
  assert(output == NULL);
  assert(terlan_ndarray_v1_delete(kths) == 0);
  assert(terlan_ndarray_v1_broadcast_to(array, shape, 1, &readonly) == 0);
  assert(terlan_ndarray_v1_partition_inplace(readonly, 1, 0) ==
         TERLAN_NDARRAY_READ_ONLY);
  assert_int_values(readonly, values, 4);
  assert(terlan_ndarray_v1_delete(readonly) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_partition_fuzz_and_axis_contracts(void) {
  int64_t length;
  for (length = 10; length < 30; ++length) {
    int64_t *values = malloc((size_t)length * sizeof(*values));
    int64_t *partitioned = NULL;
    int64_t *indices = NULL;
    const int64_t shape[] = {length};
    const int64_t selected = length / 3;
    const int64_t kth_values[] = {0, selected, selected + 1, -1, selected};
    TerlanNdArrayHandle array = NULL;
    TerlanNdArrayHandle kths = NULL;
    TerlanNdArrayHandle output = NULL;
    TerlanNdArrayHandle order = NULL;
    size_t result_length = 0;
    int64_t index;
    assert(values != NULL);
    for (index = 0; index < length; ++index) {
      values[index] = ((index * 17) + (length * 3)) % 11;
    }
    assert(terlan_ndarray_v1_from_ints(values, length, shape, 1, &array) == 0);
    assert(terlan_ndarray_v1_from_ints(kth_values, 5, (int64_t[]){5}, 1,
                                       &kths) == 0);
    assert(terlan_ndarray_v1_partition_kths(array, kths, 0, &output) == 0);
    assert(terlan_ndarray_v1_argpartition_kths(array, kths, 0, &order) == 0);
    assert(terlan_ndarray_v1_to_ints(output, &partitioned, &result_length) ==
           0);
    assert(result_length == (size_t)length);
    assert(terlan_ndarray_v1_to_ints(order, &indices, &result_length) == 0);
    for (index = 0; index < length; ++index) {
      assert(indices[index] >= 0 && indices[index] < length);
      assert(partitioned[index] == values[indices[index]]);
      if (index > 0) {
        assert(partitioned[index - 1] <= partitioned[index]);
      }
    }
    free(indices);
    free(partitioned);
    assert(terlan_ndarray_v1_delete(order) == 0);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_delete(kths) == 0);
    assert(terlan_ndarray_v1_delete(array) == 0);
    free(values);
  }
  {
    const int64_t values[] = {9, 4, 8, 3, 7, 2, 6, 1, 5, 0};
    const int64_t shape[] = {2, 5};
    TerlanNdArrayHandle array = NULL;
    TerlanNdArrayHandle output = NULL;
    assert(terlan_ndarray_v1_from_ints(values, 10, shape, 2, &array) == 0);
    assert(terlan_ndarray_v1_partition(array, 1, 0, &output) == 0);
    assert_int_values(output, (int64_t[]){2, 4, 1, 3, 0, 9, 6, 8, 5, 7},
                      10);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_partition(array, 4, 1, &output) == 0);
    assert_int_values(output, (int64_t[]){3, 4, 7, 8, 9, 0, 1, 2, 5, 6},
                      10);
    assert(terlan_ndarray_v1_delete(output) == 0);
    assert(terlan_ndarray_v1_delete(array) == 0);
  }
}

static void test_sorter_search_lexsort_and_unstable_modes(void) {
  const int64_t unsorted_values[] = {30, 10, 20};
  const int64_t sorter_values[] = {1, 2, 0};
  const int64_t query_values[] = {5, 20, 25, 40};
  const int64_t search_expected[] = {0, 1, 2, 3};
  const int64_t key_values[] = {1, 1, 0, 0, 1, 1, 0, 1};
  const int64_t key_shape[] = {2, 4};
  const int64_t lex_expected[] = {2, 3, 0, 1};
  const int64_t shape_three[] = {3};
  const int64_t shape_four[] = {4};
  TerlanNdArrayHandle unsorted = NULL;
  TerlanNdArrayHandle sorter = NULL;
  TerlanNdArrayHandle queries = NULL;
  TerlanNdArrayHandle keys = NULL;
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_from_ints(unsorted_values, 3, shape_three, 1,
                                     &unsorted) == 0);
  assert(terlan_ndarray_v1_from_ints(sorter_values, 3, shape_three, 1,
                                     &sorter) == 0);
  assert(terlan_ndarray_v1_from_ints(query_values, 4, shape_four, 1,
                                     &queries) == 0);
  assert(terlan_ndarray_v1_searchsorted_with_sorter(
             unsorted, queries, sorter, 0, &output) == 0);
  assert_int_values(output, search_expected, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_from_ints(key_values, 8, key_shape, 2, &keys) == 0);
  assert(terlan_ndarray_v1_lexsort_keys(keys, &output) == 0);
  assert_int_values(output, lex_expected, 4);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_sort_unstable(unsorted, 0, &output) == 0);
  assert_int_values(output, (int64_t[]){10, 20, 30}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_argsort_unstable(unsorted, 0, &output) == 0);
  assert_int_values(output, sorter_values, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_delete(keys) == 0);
  assert(terlan_ndarray_v1_delete(queries) == 0);
  assert(terlan_ndarray_v1_delete(sorter) == 0);
  assert(terlan_ndarray_v1_delete(unsorted) == 0);
}

static void test_numpy_lexsort_all_admitted_numeric_dtypes(void) {
  const int32_t dtypes[] = {
      TERLAN_NDARRAY_DTYPE_UINT8,   TERLAN_NDARRAY_DTYPE_UINT16,
      TERLAN_NDARRAY_DTYPE_UINT32,  TERLAN_NDARRAY_DTYPE_UINT64,
      TERLAN_NDARRAY_DTYPE_INT8,    TERLAN_NDARRAY_DTYPE_INT16,
      TERLAN_NDARRAY_DTYPE_INT32,   TERLAN_NDARRAY_DTYPE_INT64,
      TERLAN_NDARRAY_DTYPE_FLOAT16, TERLAN_NDARRAY_DTYPE_FLOAT32,
      TERLAN_NDARRAY_DTYPE_FLOAT64,
  };
  const int64_t keys[] = {0, 4, 5, 6, 2, 3, 1, 2, 1, 3, 1, 5};
  const int64_t expected[] = {0, 4, 2, 1, 3, 5};
  size_t index;
  for (index = 0; index < sizeof(dtypes) / sizeof(dtypes[0]); ++index) {
    TerlanNdArrayHandle array = NULL;
    TerlanNdArrayHandle output = NULL;
    assert(terlan_ndarray_v1_from_ints_as(
               keys, 12, (const int64_t[]){2, 6}, 2, dtypes[index],
               &array) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_lexsort_keys(array, &output) ==
           TERLAN_NDARRAY_OK);
    assert_int_values(output, expected, 6);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  }
}

static void test_numpy_lexsort_normalizes_axes_and_rejects_invalid_axes(void) {
  const int64_t keys[] = {1, 2, 1, 3, 1, 2, 0, 0, 0, 1, 0, 1};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_from_ints(keys, 12,
                                     (const int64_t[]){2, 2, 3}, 3,
                                     &array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_lexsort_axis(array, -1, &output) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(output, (const int64_t[]){0, 2, 1, 1, 2, 0}, 6);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_lexsort_axis(array, 0, &output) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(output, (const int64_t[]){0, 1, 0, 1, 0, 1}, 6);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_lexsort_keys(array, &output) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(output, (const int64_t[]){0, 2, 1, 1, 2, 0}, 6);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  for (int64_t axis = -3; axis <= 2; axis += 5) {
    output = (TerlanNdArrayHandle)(uintptr_t)1;
    assert(terlan_ndarray_v1_lexsort_axis(array, axis, &output) ==
           TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
    assert(output == NULL);
  }
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_numpy_lexsort_empty_strided_and_buffer_bounds(void) {
  int64_t equal_keys[200];
  int64_t stable_indices[100];
  TerlanNdArrayHandle empty = NULL;
  TerlanNdArrayHandle empty_view = NULL;
  TerlanNdArrayHandle keys = NULL;
  TerlanNdArrayHandle reversed = NULL;
  TerlanNdArrayHandle output = NULL;
  size_t index;

  assert(terlan_ndarray_v1_from_ints(NULL, 0, (const int64_t[]){1, 0}, 2,
                                     &empty) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_slice_axis(empty, 1, INT64_MIN, INT64_MIN, -1,
                                      &empty_view) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_lexsort_keys(empty_view, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, (const int64_t[]){0}, 1);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty_view) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_ints(NULL, 0,
                                     (const int64_t[]){1, 0, 2}, 3,
                                     &empty) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_lexsort_axis(empty, 0, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, (const int64_t[]){0, 2}, 2);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_ints(NULL, 0,
                                     (const int64_t[]){1, 2, 0}, 3,
                                     &empty) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_lexsort_axis(empty, 0, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, (const int64_t[]){2, 0}, 2);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty) == TERLAN_NDARRAY_OK);

  for (index = 0; index < 200; ++index) {
    equal_keys[index] = 1;
  }
  for (index = 0; index < 100; ++index) {
    stable_indices[index] = (int64_t)index;
  }
  assert(terlan_ndarray_v1_from_ints(equal_keys, 200,
                                     (const int64_t[]){2, 100}, 2,
                                     &keys) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_slice_axis(keys, 1, INT64_MIN, INT64_MIN, -1,
                                      &reversed) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_lexsort_keys(reversed, &output) ==
         TERLAN_NDARRAY_OK);
  assert_int_values(output, stable_indices, 100);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(reversed) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(keys) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_searchsorted_strided_sorters_and_bad_permutations(void) {
  const int64_t values[] = {30, 10, 20};
  const int64_t sorter_storage[] = {1, 99, 2, 99, 0, 99};
  const int64_t query_values[] = {10, 20, 30};
  const int64_t shape_three[] = {3};
  const int64_t shape_six[] = {6};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle storage = NULL;
  TerlanNdArrayHandle sorter = NULL;
  TerlanNdArrayHandle queries = NULL;
  TerlanNdArrayHandle output = NULL;
  TerlanNdArrayHandle invalid = NULL;
  assert(terlan_ndarray_v1_from_ints(values, 3, shape_three, 1, &array) == 0);
  assert(terlan_ndarray_v1_from_ints_as(
             sorter_storage, 6, shape_six, 1, TERLAN_NDARRAY_DTYPE_INT16,
             &storage) == 0);
  assert(terlan_ndarray_v1_slice_axis(storage, 0, 0, 6, 2, &sorter) == 0);
  assert(terlan_ndarray_v1_from_ints(query_values, 3, shape_three, 1,
                                     &queries) == 0);
  assert(terlan_ndarray_v1_searchsorted_with_sorter(
             array, queries, sorter, 0, &output) == 0);
  assert_int_values(output, (int64_t[]){0, 1, 2}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);
  assert(terlan_ndarray_v1_searchsorted_with_sorter(
             array, queries, sorter, 1, &output) == 0);
  assert_int_values(output, (int64_t[]){1, 2, 3}, 3);
  assert(terlan_ndarray_v1_delete(output) == 0);

  assert(terlan_ndarray_v1_from_ints((int64_t[]){0, 0, 2}, 3, shape_three, 1,
                                     &invalid) == 0);
  assert(terlan_ndarray_v1_searchsorted_with_sorter(
             array, queries, invalid, 0, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(output == NULL);
  assert(terlan_ndarray_v1_delete(invalid) == 0);
  assert(terlan_ndarray_v1_from_ints((int64_t[]){0, 1, 3}, 3, shape_three, 1,
                                     &invalid) == 0);
  assert(terlan_ndarray_v1_searchsorted_with_sorter(
             array, queries, invalid, 0, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(output == NULL);
  assert(terlan_ndarray_v1_delete(invalid) == 0);
  assert(terlan_ndarray_v1_from_floats((double[]){0.0, 1.0, 2.0}, 3,
                                       shape_three, 1, &invalid) == 0);
  assert(terlan_ndarray_v1_searchsorted_with_sorter(
             array, queries, invalid, 0, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(output == NULL);
  assert(terlan_ndarray_v1_delete(invalid) == 0);
  assert(terlan_ndarray_v1_delete(queries) == 0);
  assert(terlan_ndarray_v1_delete(sorter) == 0);
  assert(terlan_ndarray_v1_delete(storage) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_empty_ordering_preserves_shape(void) {
  const int64_t shape[] = {2, 0, 3};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle sorted = NULL;
  TerlanNdArrayHandle order = NULL;
  int64_t *result_shape = NULL;
  int64_t rank = 0;
  int64_t element_count = -1;
  assert(terlan_ndarray_v1_from_ints(NULL, 0, shape, 3, &array) == 0);
  assert(terlan_ndarray_v1_sort(array, 1, &sorted) == 0);
  assert(terlan_ndarray_v1_argsort(array, 1, &order) == 0);
  assert(terlan_ndarray_v1_rank(sorted, &rank) == 0);
  assert(terlan_ndarray_v1_shape(sorted, &result_shape) == 0);
  assert(rank == 3 && result_shape[0] == 2 && result_shape[1] == 0 &&
         result_shape[2] == 3);
  assert(terlan_ndarray_v1_numel(order, &element_count) == 0);
  assert(element_count == 0);
  assert(terlan_ndarray_v1_delete(order) == 0);
  assert(terlan_ndarray_v1_argsort_flat(array, &order) == 0);
  assert(terlan_ndarray_v1_delete(sorted) == 0);
  assert(terlan_ndarray_v1_sort_flat(array, &sorted) == 0);
  assert(terlan_ndarray_v1_numel(sorted, &element_count) == 0);
  assert(element_count == 0);
  assert(terlan_ndarray_v1_delete(order) == 0);
  assert(terlan_ndarray_v1_delete(sorted) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void test_numpy_partition_method_contract(void) {
  test_complex_nan_order_matches_numpy();
  test_partition_and_argpartition_satisfy_kth_contract();
  test_partition_multiple_kths_flattening_and_mutation();
  test_partition_empty_axes_and_invalid_kths();
  test_partition_fuzz_and_axis_contracts();
}

static void test_numpy_searchsorted_method_contract(void) {
  test_searchsorted_left_right_and_nan_order();
  test_searchsorted_all_admitted_scalar_dtypes();
  test_sorter_search_lexsort_and_unstable_modes();
  test_numpy_lexsort_all_admitted_numeric_dtypes();
  test_numpy_lexsort_normalizes_axes_and_rejects_invalid_axes();
  test_numpy_lexsort_empty_strided_and_buffer_bounds();
  test_searchsorted_strided_sorters_and_bad_permutations();
}

int main(void) {
  test_sort_and_argsort_each_axis();
  test_noncontiguous_nan_and_stable_ties();
  test_sort_all_admitted_scalar_dtypes();
  test_argsort_complex_is_stable_and_uses_numpy_nan_order();
  test_numpy_searchsorted_method_contract();
  test_ordering_rejects_invalid_inputs();
  test_numpy_partition_method_contract();
  test_empty_ordering_preserves_shape();
  return 0;
}
