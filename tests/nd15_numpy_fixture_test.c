#include "terlan_ndarray_internal.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* Values are pinned by tests/fixtures/numpy-1.21.5-nd15.json. */

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
  terlan_ndarray_v1_delete_int_values(values);
}

static void all_dtype_ordering(void) {
  const int64_t input[] = {3, 1, 2, 1};
  const int64_t expected[] = {1, 1, 2, 3};
  const int64_t expected_order[] = {1, 3, 2, 0};
  const int64_t shape[] = {4};
  int32_t dtype;
  for (dtype = 2; dtype <= 14; ++dtype) {
    TerlanNdArrayHandle array = NULL;
    TerlanNdArrayHandle sorted = NULL;
    TerlanNdArrayHandle order = NULL;
    if (dtype == 12 || dtype == 13) {
      const double real[] = {3.0, 1.0, 2.0, 1.0};
      const double imag[] = {0.0, 0.0, 0.0, 0.0};
      assert(terlan_ndarray_v1_from_complex(real, 4, imag, 4, shape, 1, dtype,
                                             &array) == 0);
    } else if (dtype == 3 || dtype == 11 || dtype == 14) {
      const double values[] = {3.0, 1.0, 2.0, 1.0};
      assert(terlan_ndarray_v1_from_floats_as(values, 4, shape, 1, dtype,
                                               &array) == 0);
    } else {
      assert(terlan_ndarray_v1_from_ints_as(input, 4, shape, 1, dtype,
                                             &array) == 0);
    }
    assert(terlan_ndarray_v1_sort(array, 0, &sorted) == 0);
    assert(terlan_ndarray_v1_argsort(array, 0, &order) == 0);
    assert_ints(order, expected_order, 4);
    if (dtype == 12 || dtype == 13) {
      double *values = NULL;
      size_t length = 0;
      size_t index;
      assert(terlan_ndarray_v1_complex_reals(sorted, &values, &length) == 0);
      assert(length == 4);
      for (index = 0; index < 4; ++index) {
        assert(values[index] == (double)expected[index]);
      }
      terlan_ndarray_v1_delete_float_values(values);
    } else {
      assert_ints(sorted, expected, 4);
    }
    assert(terlan_ndarray_v1_delete(order) == 0);
    assert(terlan_ndarray_v1_delete(sorted) == 0);
    assert(terlan_ndarray_v1_delete(array) == 0);
  }
}

static void boolean_and_unique_metadata(void) {
  const uint8_t values[] = {1, 0, 1, 0};
  const int64_t shape[] = {4};
  const int64_t order_expected[] = {1, 3, 0, 2};
  const int64_t input[] = {3, 1, 2, 1};
  const int64_t unique_expected[] = {1, 2, 3};
  const int64_t index_expected[] = {1, 2, 0};
  const int64_t inverse_expected[] = {2, 0, 1, 0};
  const int64_t count_expected[] = {2, 1, 1};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle order = NULL;
  uint64_t *outputs = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_from_bools(values, 4, shape, 1, &array) == 0);
  assert(terlan_ndarray_v1_argsort(array, 0, &order) == 0);
  assert_ints(order, order_expected, 4);
  assert(terlan_ndarray_v1_delete(order) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
  assert(terlan_ndarray_v1_from_ints(input, 4, shape, 1, &array) == 0);
  assert(terlan_ndarray_v1_unique_all(array, &outputs, &length) == 0);
  assert(length == 4);
  assert_ints((TerlanNdArrayHandle)(uintptr_t)outputs[0], unique_expected, 3);
  assert_ints((TerlanNdArrayHandle)(uintptr_t)outputs[1], index_expected, 3);
  assert_ints((TerlanNdArrayHandle)(uintptr_t)outputs[2], inverse_expected, 4);
  assert_ints((TerlanNdArrayHandle)(uintptr_t)outputs[3], count_expected, 3);
  for (index = 0; index < 4; ++index) {
    assert(terlan_ndarray_v1_delete(
               (TerlanNdArrayHandle)(uintptr_t)outputs[index]) == 0);
  }
  terlan_ndarray_v1_delete_handle_values(outputs);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

static void floating_and_lexicographic_order(void) {
  const double values[] = {-0.0, INFINITY, NAN, -INFINITY, 0.0};
  const int64_t float_shape[] = {5};
  const int64_t float_order[] = {3, 0, 4, 1, 2};
  const int64_t keys[] = {1, 1, 0, 0, 0, 1, 0, 1};
  const int64_t key_shape[] = {2, 4};
  const int64_t lex_order[] = {2, 0, 3, 1};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_from_floats(values, 5, float_shape, 1, &array) == 0);
  assert(terlan_ndarray_v1_argsort(array, 0, &result) == 0);
  assert_ints(result, float_order, 5);
  assert(terlan_ndarray_v1_delete(result) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
  assert(terlan_ndarray_v1_from_ints(keys, 8, key_shape, 2, &array) == 0);
  assert(terlan_ndarray_v1_lexsort_keys(array, &result) == 0);
  assert_ints(result, lex_order, 4);
  assert(terlan_ndarray_v1_delete(result) == 0);
  assert(terlan_ndarray_v1_delete(array) == 0);
}

int main(void) {
  all_dtype_ordering();
  boolean_and_unique_metadata();
  floating_and_lexicographic_order();
  return 0;
}
