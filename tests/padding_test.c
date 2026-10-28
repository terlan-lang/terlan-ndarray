#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>

static void assert_floats(TerlanNdArrayHandle array, const double *expected,
                          size_t count) {
  double *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_floats_cast(array, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    if (values[index] != expected[index]) {
      fprintf(stderr, "padding mismatch at %zu: %.17g != %.17g\n", index,
              values[index], expected[index]);
    }
    assert(values[index] == expected[index]);
  }
  terlan_ndarray_v1_delete_float_values(values);
}

static void assert_shape(TerlanNdArrayHandle array, const int64_t *expected,
                         int64_t rank) {
  int64_t actual_rank = 0;
  int64_t *actual = NULL;
  int64_t axis;
  assert(terlan_ndarray_v1_rank(array, &actual_rank) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_shape(array, &actual) == TERLAN_NDARRAY_OK);
  assert(actual_rank == rank);
  for (axis = 0; axis < rank; ++axis) {
    assert(actual[axis] == expected[axis]);
  }
}

static TerlanNdArrayHandle fixture(void) {
  const double values[] = {1.0, 2.0, 3.0};
  const int64_t shape[] = {3};
  TerlanNdArrayHandle array = NULL;
  assert(terlan_ndarray_v1_from_floats(values, 3, shape, 1, &array) ==
         TERLAN_NDARRAY_OK);
  return array;
}

static void test_constant_and_mapping_modes_match_numpy_order(void) {
  const int64_t before[] = {2};
  const int64_t after[] = {2};
  const double before_value[] = {-1.0};
  const double after_value[] = {9.0};
  const double constant[] = {-1.0, -1.0, 1.0, 2.0, 3.0, 9.0, 9.0};
  const double edge[] = {1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0};
  const double reflect[] = {3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0};
  const double symmetric[] = {2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 2.0};
  const double wrap[] = {2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0};
  TerlanNdArrayHandle source = fixture();
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_pad_constant(
             source, before, 1, after, 1, before_value, 1, after_value, 1,
             &output) == TERLAN_NDARRAY_OK);
  assert_floats(output, constant, 7);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
#define CHECK_MODE(name, expected)                                             \
  assert(terlan_ndarray_v1_pad_##name(source, before, 1, after, 1, &output) == \
         TERLAN_NDARRAY_OK);                                                   \
  assert_floats(output, expected, 7);                                          \
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK)
  CHECK_MODE(edge, edge);
  CHECK_MODE(reflect, reflect);
  CHECK_MODE(symmetric, symmetric);
  CHECK_MODE(wrap, wrap);
#undef CHECK_MODE
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_statistical_modes_and_non_contiguous_inputs(void) {
  const int64_t before[] = {1};
  const int64_t after[] = {1};
  const int64_t lengths[] = {2};
  const double minimum[] = {1.0, 1.0, 2.0, 3.0, 2.0};
  const double maximum[] = {2.0, 1.0, 2.0, 3.0, 3.0};
  const double center[] = {1.5, 1.0, 2.0, 3.0, 2.5};
  TerlanNdArrayHandle source = fixture();
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_pad_statistical(
             source, before, 1, after, 1, lengths, 1, lengths, 1, 0,
             &output) == TERLAN_NDARRAY_OK);
  assert_floats(output, minimum, 5);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_pad_statistical(
             source, before, 1, after, 1, lengths, 1, lengths, 1, 1,
             &output) == TERLAN_NDARRAY_OK);
  assert_floats(output, maximum, 5);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_pad_statistical(
             source, before, 1, after, 1, lengths, 1, lengths, 1, 2,
             &output) == TERLAN_NDARRAY_OK);
  assert_floats(output, center, 5);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_pad_statistical(
             source, before, 1, after, 1, lengths, 1, lengths, 1, 3,
             &output) == TERLAN_NDARRAY_OK);
  assert_floats(output, center, 5);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_linear_and_odd_reflection_modes_match_numpy(void) {
  const int64_t before[] = {2};
  const int64_t after[] = {3};
  const int64_t wide[] = {5};
  const double start[] = {5.0};
  const double end[] = {-4.0};
  const double linear[] = {5.0, 3.0, 1.0,       2.0,
                           3.0, 2.0 / 3.0, -5.0 / 3.0, -4.0};
  const double reflect[] = {-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0,
                            3.0,  4.0,  5.0,  6.0,  7.0, 8.0};
  const double symmetric[] = {-2.0, -1.0, -1.0, 0.0, 1.0, 1.0, 2.0,
                              3.0,  3.0,  4.0,  5.0, 5.0, 6.0};
  TerlanNdArrayHandle source = fixture();
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_pad_linear_ramp(
             source, before, 1, after, 1, start, 1, end, 1, &output) ==
         TERLAN_NDARRAY_OK);
  assert_floats(output, linear, 8);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_pad_reflect_odd(source, wide, 1, wide, 1,
                                           &output) == TERLAN_NDARRAY_OK);
  assert_floats(output, reflect, 13);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_pad_symmetric_odd(source, wide, 1, wide, 1,
                                             &output) == TERLAN_NDARRAY_OK);
  assert_floats(output, symmetric, 13);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_statistical_defaults_clipping_zero_and_integer_rounding(void) {
  const int64_t width[] = {1};
  const int64_t oversized[] = {30};
  const int64_t zero[] = {0};
  const int64_t pair_values[] = {1, 2};
  const int64_t pair_shape[] = {2};
  const double minimum[] = {1.0, 1.0, 2.0, 3.0, 1.0};
  const double maximum[] = {3.0, 1.0, 2.0, 3.0, 3.0};
  const double center[] = {2.0, 1.0, 2.0, 3.0, 2.0};
  const double rounded[] = {2.0, 1.0, 2.0, 2.0};
  TerlanNdArrayHandle source = fixture();
  TerlanNdArrayHandle integers = NULL;
  TerlanNdArrayHandle output = NULL;
#define CHECK_STAT(name, expected)                                            \
  assert(terlan_ndarray_v1_pad_##name(source, width, 1, width, 1, NULL, 0,    \
                                       NULL, 0, &output) ==                    \
         TERLAN_NDARRAY_OK);                                                  \
  assert_floats(output, expected, 5);                                         \
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK)
  CHECK_STAT(minimum, minimum);
  CHECK_STAT(maximum, maximum);
  CHECK_STAT(mean, center);
  CHECK_STAT(median, center);
#undef CHECK_STAT
  assert(terlan_ndarray_v1_pad_mean(source, width, 1, width, 1, oversized, 1,
                                    oversized, 1, &output) == TERLAN_NDARRAY_OK);
  assert_floats(output, center, 5);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(pair_values, 2, pair_shape, 1, &integers) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_pad_mean(integers, width, 1, width, 1, NULL, 0,
                                    NULL, 0, &output) == TERLAN_NDARRAY_OK);
  assert_floats(output, rounded, 4);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_pad_mean(source, width, 1, width, 1, zero, 1, zero,
                                    1, &output) == TERLAN_NDARRAY_OK);
  {
    double *values = NULL;
    size_t count = 0;
    assert(terlan_ndarray_v1_to_floats_cast(output, &values, &count) ==
           TERLAN_NDARRAY_OK);
    assert(count == 5 && isnan(values[0]) && isnan(values[4]));
    terlan_ndarray_v1_delete_float_values(values);
  }
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_pad_minimum(source, width, 1, width, 1, zero, 1,
                                       zero, 1, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(output == NULL);
  assert(terlan_ndarray_v1_delete(integers) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_rank_broadcast_empty_axes_and_transposed_inputs(void) {
  const int64_t values[] = {1, 2, 3, 4};
  const int64_t shape[] = {2, 2};
  const int64_t width[] = {1};
  const int64_t padded_shape[] = {4, 4};
  const double expected[] = {1, 1, 2, 2, 1, 1, 2, 2,
                             3, 3, 4, 4, 3, 3, 4, 4};
  const double transposed[] = {1, 1, 3, 3, 1, 1, 3, 3,
                               2, 2, 4, 4, 2, 2, 4, 4};
  const double empty_padding[] = {0, 0, 0, 0, 0, 1, 2, 0,
                                  0, 3, 4, 0, 0, 0, 0, 0};
  const int64_t empty_shape[] = {2, 0, 2};
  const int64_t empty_before[] = {3, 0, 1};
  const int64_t empty_after[] = {3, 0, 1};
  const int64_t empty_padded_shape[] = {8, 0, 4};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle view = NULL;
  TerlanNdArrayHandle empty = NULL;
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_from_ints(values, 4, shape, 2, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_pad_edge(source, width, 1, width, 1, &output) ==
         TERLAN_NDARRAY_OK);
  assert_shape(output, padded_shape, 2);
  assert_floats(output, expected, 16);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_pad_empty(source, width, 1, width, 1, &output) ==
         TERLAN_NDARRAY_OK);
  assert_floats(output, empty_padding, 16);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_transpose(source, 0, 1, &view) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_pad_edge(view, width, 1, width, 1, &output) ==
         TERLAN_NDARRAY_OK);
  assert_floats(output, transposed, 16);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(NULL, 0, empty_shape, 3, &empty) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_pad_empty(empty, empty_before, 3, empty_after, 3,
                                     &output) == TERLAN_NDARRAY_OK);
  assert_shape(output, empty_padded_shape, 3);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_pad_wrap(empty, empty_before, 3, empty_after, 3,
                                    &output) == TERLAN_NDARRAY_OK);
  assert_shape(output, empty_padded_shape, 3);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(empty) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(view) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_multidimensional_statistics_match_numpy_corner_order(void) {
  const int64_t values[] = {1, 2, 3, 4};
  const int64_t shape[] = {2, 2};
  const int64_t width[] = {1};
  const double minimum[] = {1, 1, 2, 1, 1, 1, 2, 1,
                            3, 3, 4, 3, 1, 1, 2, 1};
  const double maximum[] = {4, 3, 4, 4, 2, 1, 2, 2,
                            4, 3, 4, 4, 4, 3, 4, 4};
  const double center[] = {2, 2, 3, 2, 2, 1, 2, 2,
                           4, 3, 4, 4, 2, 2, 3, 2};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_from_ints(values, 4, shape, 2, &source) ==
         TERLAN_NDARRAY_OK);
#define CHECK_2D_STAT(name, expected)                                         \
  assert(terlan_ndarray_v1_pad_##name(source, width, 1, width, 1, NULL, 0,    \
                                       NULL, 0, &output) ==                    \
         TERLAN_NDARRAY_OK);                                                  \
  assert_floats(output, expected, 16);                                        \
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK)
  CHECK_2D_STAT(minimum, minimum);
  CHECK_2D_STAT(maximum, maximum);
  CHECK_2D_STAT(mean, center);
  CHECK_2D_STAT(median, center);
#undef CHECK_2D_STAT
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_typed_constants_preserve_large_integers_and_complex_ramps(void) {
  const int64_t large = INT64_C(18014398509481985);
  const int64_t large_values[] = {INT64_C(18014398509481987)};
  const int64_t scalar_shape[] = {1};
  const int64_t width[] = {1};
  const int64_t ramp_width[] = {2};
  const double source_real[] = {1.0};
  const double source_imag[] = {1.0};
  const double endpoint_real[] = {3.0};
  const double endpoint_imag[] = {5.0};
  const double expected_real[] = {3.0, 2.0, 1.0, 2.0, 3.0};
  const double expected_imag[] = {5.0, 3.0, 1.0, 3.0, 5.0};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle constants = NULL;
  TerlanNdArrayHandle complex_source = NULL;
  TerlanNdArrayHandle endpoints = NULL;
  TerlanNdArrayHandle output = NULL;
  int64_t *integers = NULL;
  double *parts = NULL;
  size_t count = 0;
  assert(terlan_ndarray_v1_from_ints(large_values, 1, scalar_shape, 1,
                                     &source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_ints(&large, 1, scalar_shape, 1, &constants) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_pad_constant_values(
             source, width, 1, width, 1, constants, constants, &output) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_to_ints_cast(output, &integers, &count) ==
         TERLAN_NDARRAY_OK);
  assert(count == 3 && integers[0] == large && integers[1] == large_values[0] &&
         integers[2] == large);
  terlan_ndarray_v1_delete_int_values(integers);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_complex(
             source_real, 1, source_imag, 1, scalar_shape, 1,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &complex_source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_complex(
             endpoint_real, 1, endpoint_imag, 1, scalar_shape, 1,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &endpoints) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_pad_linear_ramp_values(
             complex_source, ramp_width, 1, ramp_width, 1, endpoints, endpoints,
             &output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_complex_reals(output, &parts, &count) ==
         TERLAN_NDARRAY_OK);
  assert(count == 5);
  for (size_t index = 0; index < count; ++index) {
    assert(parts[index] == expected_real[index]);
  }
  terlan_ndarray_v1_delete_float_values(parts);
  assert(terlan_ndarray_v1_complex_imags(output, &parts, &count) ==
         TERLAN_NDARRAY_OK);
  for (size_t index = 0; index < count; ++index) {
    assert(parts[index] == expected_imag[index]);
  }
  terlan_ndarray_v1_delete_float_values(parts);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(endpoints) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(complex_source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(constants) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_padding_rejections_leave_no_outputs(void) {
  const int64_t before[] = {-1};
  const int64_t after[] = {1};
  const int64_t lengths[] = {1};
  const double constants[] = {0.0};
  TerlanNdArrayHandle source = fixture();
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_pad_edge(source, before, 1, after, 1, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_pad_constant(source, after, 1, after, 1,
                                        constants, 0, constants, 0,
                                        &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(terlan_ndarray_v1_pad_statistical(
             source, after, 1, after, 1, lengths, 1, lengths, 1, 4,
             &output) == TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(output == NULL);
  assert(terlan_ndarray_v1_pad_edge(source, after, 0, after, 1, &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(output == NULL);
  assert(terlan_ndarray_v1_pad_linear_ramp(source, after, 1, after, 1,
                                            constants, 0, constants, 0,
                                            &output) ==
         TERLAN_NDARRAY_INVALID_ARGUMENT);
  assert(output == NULL);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

int main(void) {
  test_constant_and_mapping_modes_match_numpy_order();
  test_statistical_modes_and_non_contiguous_inputs();
  test_linear_and_odd_reflection_modes_match_numpy();
  test_statistical_defaults_clipping_zero_and_integer_rounding();
  test_rank_broadcast_empty_axes_and_transposed_inputs();
  test_multidimensional_statistics_match_numpy_corner_order();
  test_typed_constants_preserve_large_integers_and_complex_ramps();
  test_padding_rejections_leave_no_outputs();
  return 0;
}
