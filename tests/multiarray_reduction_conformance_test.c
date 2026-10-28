#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <math.h>
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

static void test_numpy_minmax_scalar_and_axis_contract(void) {
  const int64_t scalar_value[] = {1};
  const int64_t matrix_values[] = {1, 2, 3};
  const int64_t scalar_axis[] = {0};
  const int64_t scalar_negative_axis[] = {-1};
  const int64_t invalid_scalar_axis[] = {1};
  const int64_t duplicate_scalar_axes[] = {0, -1};
  const int64_t matrix_axis[] = {1};
  const int64_t invalid_matrix_axis[] = {1000};
  TerlanNdArrayHandle scalar = NULL;
  TerlanNdArrayHandle matrix = NULL;
  TerlanNdArrayHandle result = NULL;

  assert(terlan_ndarray_v1_from_ints(scalar_value, 1, NULL, 0, &scalar) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_minimum(scalar, scalar_axis, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert(result->rank == 0);
  assert_ints(result, scalar_value, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  result = NULL;
  assert(terlan_ndarray_v1_maximum(scalar, scalar_negative_axis, 1, 0,
                                    &result) == TERLAN_NDARRAY_OK);
  assert(result->rank == 0);
  assert_ints(result, scalar_value, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  result = NULL;
  assert(terlan_ndarray_v1_minimum(scalar, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert(result->rank == 0);
  assert_ints(result, scalar_value, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  result = NULL;
  assert(terlan_ndarray_v1_maximum(scalar, scalar_axis, 1, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert(result->rank == 0);
  assert_ints(result, scalar_value, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  result = NULL;
  assert(terlan_ndarray_v1_maximum(scalar, invalid_scalar_axis, 1, 0,
                                    &result) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(result == NULL);
  assert(terlan_ndarray_v1_minimum(scalar, duplicate_scalar_axes, 2, 0,
                                    &result) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(result == NULL);

  assert(terlan_ndarray_v1_from_ints(matrix_values, 3,
                                      (const int64_t[]){1, 3}, 2,
                                      &matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_maximum(matrix, matrix_axis, 1, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert(result->rank == 1 && result->shape[0] == 1);
  assert_ints(result, (const int64_t[]){3}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  result = NULL;
  assert(terlan_ndarray_v1_minimum(matrix, invalid_matrix_axis, 1, 0,
                                    &result) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(result == NULL);

  assert(terlan_ndarray_v1_delete(matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_numpy_bool_sum_and_unaligned_count_nonzero(void) {
  uint8_t ones[101];
  int64_t offset;
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle every_second = NULL;
  TerlanNdArrayHandle reversed_every_second = NULL;
  TerlanNdArrayHandle result = NULL;

  for (offset = 0; offset < 101; ++offset) {
    ones[offset] = 1;
  }
  assert(terlan_ndarray_v1_from_bools(ones, 101,
                                      (const int64_t[]){101}, 1,
                                      &source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sum(source, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, (const int64_t[]){101}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  result = NULL;
  assert(terlan_ndarray_v1_slice_axis(source, 0, 0, 101, 2,
                                      &every_second) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sum(every_second, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, (const int64_t[]){51}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  result = NULL;
  assert(terlan_ndarray_v1_slice_axis(source, 0, INT64_MIN, INT64_MIN, -2,
                                      &reversed_every_second) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sum(reversed_every_second, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, (const int64_t[]){51}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(reversed_every_second) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(every_second) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);

  for (offset = 0; offset < 7; ++offset) {
    uint8_t false_values[18] = {0};
    uint8_t true_values[18];
    int64_t index;
    int64_t count = -1;
    TerlanNdArrayHandle false_source = NULL;
    TerlanNdArrayHandle true_source = NULL;
    TerlanNdArrayHandle false_view = NULL;
    TerlanNdArrayHandle true_view = NULL;
    for (index = 0; index < 18; ++index) {
      true_values[index] = 1;
    }
    for (index = 0; index < offset; ++index) {
      false_values[offset + 1 + index] = 1;
      true_values[offset + 1 + index] = 0;
    }
    assert(terlan_ndarray_v1_from_bools(false_values, 18,
                                        (const int64_t[]){18}, 1,
                                        &false_source) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_from_bools(true_values, 18,
                                        (const int64_t[]){18}, 1,
                                        &true_source) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_slice_axis(false_source, 0, offset + 1, 18, 1,
                                        &false_view) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_slice_axis(true_source, 0, offset + 1, 18, 1,
                                        &true_view) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_count_nonzero(false_view, &count) ==
           TERLAN_NDARRAY_OK);
    assert(count == offset);
    assert(terlan_ndarray_v1_count_nonzero(true_view, &count) ==
           TERLAN_NDARRAY_OK);
    assert(count == 17 - (2 * offset));
    assert(terlan_ndarray_v1_delete(true_view) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(false_view) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(true_source) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(false_source) == TERLAN_NDARRAY_OK);
  }
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_numpy_scalar_conversion_contract(void) {
  const uint8_t false_value[] = {0};
  const uint8_t true_value[] = {1};
  const int64_t integer_values[] = {0, 1, 42};
  const double floating_value[] = {-4.75};
  const double complex_real[] = {4.0};
  const double complex_imag[] = {1.0};
  TerlanNdArrayHandle array = NULL;
  bool boolean = true;
  int64_t integer = 99;

  assert(terlan_ndarray_v1_from_bools(false_value, 1,
                                      (const int64_t[]){1}, 1,
                                      &array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_item_bool(array, &boolean) == TERLAN_NDARRAY_OK);
  assert(!boolean);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_bools(true_value, 1,
                                      (const int64_t[]){1}, 1,
                                      &array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_item_bool(array, &boolean) == TERLAN_NDARRAY_OK);
  assert(boolean);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(integer_values + 2, 1,
                                     (const int64_t[]){1, 1}, 2,
                                     &array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_item_bool(array, &boolean) == TERLAN_NDARRAY_OK);
  assert(boolean);
  assert(terlan_ndarray_v1_item_int(array, &integer) == TERLAN_NDARRAY_OK);
  assert(integer == 42);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(integer_values, 1, NULL, 0, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_item_int(array, &integer) == TERLAN_NDARRAY_OK);
  assert(integer == 0);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(integer_values + 1, 1,
                                     (const int64_t[]){1}, 1,
                                     &array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_item_int(array, &integer) == TERLAN_NDARRAY_OK);
  assert(integer == 1);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(floating_value, 1, NULL, 0, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_item_int(array, &integer) == TERLAN_NDARRAY_OK);
  assert(integer == -4);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_ints(integer_values, 3,
                                     (const int64_t[]){3}, 1,
                                     &array) == TERLAN_NDARRAY_OK);
  boolean = true;
  integer = 99;
  assert(terlan_ndarray_v1_item_bool(array, &boolean) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(boolean);
  assert(terlan_ndarray_v1_item_int(array, &integer) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert(integer == 99);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_complex(complex_real, 1, complex_imag, 1,
                                        NULL, 0,
                                        TERLAN_NDARRAY_DTYPE_COMPLEX128,
                                        &array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_item_bool(array, &boolean) == TERLAN_NDARRAY_OK);
  assert(boolean);
  assert(terlan_ndarray_v1_item_int(array, &integer) ==
         TERLAN_NDARRAY_DTYPE_MISMATCH);
  assert(integer == 99);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_item_bool(NULL, &boolean) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_item_bool(NULL, NULL) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_item_int(NULL, NULL) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_numpy_logical_xor_reduction_contract(void) {
  const uint8_t none_values[] = {0, 0, 0, 0};
  const uint8_t some_values[] = {1, 0, 1, 1};
  const uint8_t every_values[] = {1, 1, 1, 1};
  const int64_t axis[] = {1};
  TerlanNdArrayHandle none = NULL;
  TerlanNdArrayHandle some = NULL;
  TerlanNdArrayHandle every = NULL;
  TerlanNdArrayHandle empty = NULL;
  TerlanNdArrayHandle matrix = NULL;
  TerlanNdArrayHandle result = NULL;

  assert(terlan_ndarray_v1_from_bools(none_values, 4,
                                      (const int64_t[]){4}, 1,
                                      &none) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_bools(some_values, 4,
                                      (const int64_t[]){4}, 1,
                                      &some) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_bools(every_values, 4,
                                      (const int64_t[]){4}, 1,
                                      &every) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_bools(NULL, 0,
                                      (const int64_t[]){0}, 1,
                                      &empty) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_logical_xor_reduce(none, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, (const int64_t[]){0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_logical_xor_reduce(some, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, (const int64_t[]){1}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_logical_xor_reduce(every, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, (const int64_t[]){0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_logical_xor_reduce(empty, NULL, 0, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_ints(result, (const int64_t[]){0}, 1);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_bools(
             (const uint8_t[]){1, 0, 1, 1, 1, 1}, 6,
             (const int64_t[]){2, 3}, 2, &matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_logical_xor_reduce(matrix, axis, 1, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert(result->rank == 2 && result->shape[0] == 2 &&
         result->shape[1] == 1);
  assert_ints(result, (const int64_t[]){0, 1}, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_logical_xor_reduce(matrix,
                                               (const int64_t[]){2}, 1, 0,
                                               &result) ==
         TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  assert(result == NULL);

  assert(terlan_ndarray_v1_delete(matrix) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(every) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(some) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(none) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static TerlanNdArrayHandle floating_fixture(const double *values,
                                             int64_t count,
                                             int32_t dtype) {
  TerlanNdArrayHandle array = NULL;
  if (dtype == TERLAN_NDARRAY_DTYPE_COMPLEX64 ||
      dtype == TERLAN_NDARRAY_DTYPE_COMPLEX128) {
    double imaginary[32] = {0};
    assert(count <= 32);
    assert(terlan_ndarray_v1_from_complex(values, (size_t)count, imaginary,
                                           (size_t)count,
                                           (const int64_t[]){count}, 1, dtype,
                                           &array) == TERLAN_NDARRAY_OK);
  } else {
    assert(terlan_ndarray_v1_from_floats_as(
               values, count, (const int64_t[]){count}, 1, dtype, &array) ==
           TERLAN_NDARRAY_OK);
  }
  return array;
}

static void assert_scalar_nan(TerlanNdArrayHandle array) {
  TerlanNdArrayScalar value;
  assert(array->numel == 1);
  assert(dtype_load(array_const_element_pointer(array, 0), array->dtype,
                    &value) == TERLAN_NDARRAY_OK);
  assert(isnan(value.real) || isnan(value.imag));
}

static void test_numpy_minmax_offsets_nan_and_extremes(void) {
  const int32_t dtypes[] = {
      TERLAN_NDARRAY_DTYPE_FLOAT16, TERLAN_NDARRAY_DTYPE_FLOAT32,
      TERLAN_NDARRAY_DTYPE_FLOAT64, TERLAN_NDARRAY_DTYPE_COMPLEX64,
      TERLAN_NDARRAY_DTYPE_COMPLEX128,
  };
  size_t dtype_index;
  for (dtype_index = 0; dtype_index < sizeof(dtypes) / sizeof(dtypes[0]);
       ++dtype_index) {
    int64_t offset;
    for (offset = 0; offset < 8; ++offset) {
      const int64_t length = dtypes[dtype_index] ==
                                     TERLAN_NDARRAY_DTYPE_FLOAT64
                                 ? 7
                                 : 15;
      int64_t nan_index;
      for (nan_index = 0; nan_index < length; ++nan_index) {
        double values[32];
        int64_t index;
        TerlanNdArrayHandle source;
        TerlanNdArrayHandle view = NULL;
        TerlanNdArrayHandle result = NULL;
        for (index = 0; index < 32; ++index) {
          values[index] = (double)index;
        }
        values[offset + nan_index] = NAN;
        source = floating_fixture(values, 32, dtypes[dtype_index]);
        assert(terlan_ndarray_v1_slice_axis(source, 0, offset,
                                             offset + length, 1, &view) ==
               TERLAN_NDARRAY_OK);
        assert(terlan_ndarray_v1_minimum(view, NULL, 0, 0, &result) ==
               TERLAN_NDARRAY_OK);
        assert_scalar_nan(result);
        assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
        assert(terlan_ndarray_v1_maximum(view, NULL, 0, 0, &result) ==
               TERLAN_NDARRAY_OK);
        assert_scalar_nan(result);
        assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
        assert(terlan_ndarray_v1_delete(view) == TERLAN_NDARRAY_OK);
        assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
      }
    }
    {
      const double large = dtypes[dtype_index] == TERLAN_NDARRAY_DTYPE_FLOAT16
                               ? 60000.0
                               : 1e10;
      const double extremes[] = {0.0, large, -large, 4.0};
      TerlanNdArrayHandle source =
          floating_fixture(extremes, 4, dtypes[dtype_index]);
      TerlanNdArrayHandle result = NULL;
      TerlanNdArrayScalar value;
      assert(terlan_ndarray_v1_minimum(source, NULL, 0, 0, &result) ==
             TERLAN_NDARRAY_OK);
      assert(dtype_load(array_const_element_pointer(result, 0), result->dtype,
                        &value) == TERLAN_NDARRAY_OK);
      assert(value.real < -(long double)large * 0.9L);
      assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
      assert(terlan_ndarray_v1_maximum(source, NULL, 0, 0, &result) ==
             TERLAN_NDARRAY_OK);
      assert(dtype_load(array_const_element_pointer(result, 0), result->dtype,
                        &value) == TERLAN_NDARRAY_OK);
      assert(value.real > (long double)large * 0.9L);
      assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
      assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
    }
  }
  {
    TerlanNdArrayHandle nan = floating_fixture((const double[]){NAN}, 1,
                                                TERLAN_NDARRAY_DTYPE_FLOAT64);
    TerlanNdArrayHandle one = floating_fixture((const double[]){1.0}, 1,
                                                TERLAN_NDARRAY_DTYPE_FLOAT64);
    TerlanNdArrayHandle result = NULL;
    assert(terlan_ndarray_v1_elementwise_minimum(nan, one, &result) ==
           TERLAN_NDARRAY_OK);
    assert_scalar_nan(result);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(one) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(nan) == TERLAN_NDARRAY_OK);
  }
  assert(terlan_ndarray_v1_live_count() == 0);
}

int main(void) {
  test_numpy_minmax_scalar_and_axis_contract();
  test_numpy_bool_sum_and_unaligned_count_nonzero();
  test_numpy_scalar_conversion_contract();
  test_numpy_logical_xor_reduction_contract();
  test_numpy_minmax_offsets_nan_and_extremes();
  return 0;
}
