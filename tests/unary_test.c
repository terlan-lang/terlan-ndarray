#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static void assert_floats(TerlanNdArrayHandle array, const double *expected,
                          size_t count, double tolerance) {
  double *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_floats_cast(array, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    if (isnan(expected[index])) {
      assert(isnan(values[index]));
    } else if (isinf(expected[index])) {
      assert(values[index] == expected[index]);
    } else {
      assert(fabs(values[index] - expected[index]) <= tolerance);
    }
  }
  terlan_ndarray_v1_delete_float_values(values);
}

static void assert_bools(TerlanNdArrayHandle array, const uint8_t *expected,
                         size_t count) {
  uint8_t *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_bools(array, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(values[index] == expected[index]);
  }
  terlan_ndarray_v1_delete_bool_values(values);
}

static void test_real_unary_families(void) {
  const double values[] = {-1.5, 0.0, 2.0, 4.0};
  const int64_t shape[] = {4};
  const double absolute[] = {1.5, 0.0, 2.0, 4.0};
  const double signs[] = {-1.0, 0.0, 1.0, 1.0};
  const double floors[] = {-2.0, 0.0, 2.0, 4.0};
  const double ceilings[] = {-1.0, 0.0, 2.0, 4.0};
  const uint8_t logical_not[] = {0, 1, 0, 0};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_from_floats(values, 4, shape, 1, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_absolute(array, &result) == TERLAN_NDARRAY_OK);
  assert_floats(result, absolute, 4, 0.0);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sign(array, &result) == TERLAN_NDARRAY_OK);
  assert_floats(result, signs, 4, 0.0);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_floor(array, &result) == TERLAN_NDARRAY_OK);
  assert_floats(result, floors, 4, 0.0);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_ceil(array, &result) == TERLAN_NDARRAY_OK);
  assert_floats(result, ceilings, 4, 0.0);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_logical_not(array, &result) == TERLAN_NDARRAY_OK);
  assert_bools(result, logical_not, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_transcendental_and_non_finite_classification(void) {
  const double values[] = {0.0, 1.0, INFINITY, NAN};
  const int64_t shape[] = {4};
  const uint8_t nan_values[] = {0, 0, 0, 1};
  const uint8_t infinite[] = {0, 0, 1, 0};
  const uint8_t finite[] = {1, 1, 0, 0};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle result = NULL;
  assert(terlan_ndarray_v1_from_floats(values, 4, shape, 1, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_is_nan(array, &result) == TERLAN_NDARRAY_OK);
  assert_bools(result, nan_values, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_is_infinite(array, &result) == TERLAN_NDARRAY_OK);
  assert_bools(result, infinite, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_is_finite(array, &result) == TERLAN_NDARRAY_OK);
  assert_bools(result, finite, 4);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_exp(array, &result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_sin(array, &result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_complex_absolute_conjugate_and_square(void) {
  const double real[] = {3.0, 1.0};
  const double imag[] = {4.0, -2.0};
  const int64_t shape[] = {2};
  const double absolute[] = {5.0, sqrt(5.0)};
  const double conjugate_imag[] = {-4.0, 2.0};
  const double square_real[] = {-7.0, -3.0};
  const double square_imag[] = {24.0, -4.0};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle result = NULL;
  double *parts = NULL;
  size_t length = 0;
  assert(terlan_ndarray_v1_from_complex(real, 2, imag, 2, shape, 1,
                                         TERLAN_NDARRAY_DTYPE_COMPLEX128,
                                         &array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_absolute(array, &result) == TERLAN_NDARRAY_OK);
  assert_floats(result, absolute, 2, 1e-12);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_conjugate(array, &result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_complex_imags(result, &parts, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == 2 && parts[0] == conjugate_imag[0] &&
         parts[1] == conjugate_imag[1]);
  terlan_ndarray_v1_delete_float_values(parts);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_square(array, &result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_complex_reals(result, &parts, &length) ==
         TERLAN_NDARRAY_OK);
  assert(parts[0] == square_real[0] && parts[1] == square_real[1]);
  terlan_ndarray_v1_delete_float_values(parts);
  assert(terlan_ndarray_v1_complex_imags(result, &parts, &length) ==
         TERLAN_NDARRAY_OK);
  assert(parts[0] == square_imag[0] && parts[1] == square_imag[1]);
  terlan_ndarray_v1_delete_float_values(parts);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_floor(array, &result) ==
         TERLAN_NDARRAY_UNSUPPORTED_DTYPE);
  assert(result == NULL);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void test_reciprocal_zero_and_bool_negative_follow_numeric_rules(void) {
  const double zero[] = {0.0};
  const uint8_t truth[] = {1};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_from_floats(zero, 1, NULL, 0, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_reciprocal(array, &output) == TERLAN_NDARRAY_OK);
  assert_floats(output, (double[]){INFINITY}, 1, 0.0);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  output = NULL;
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_bools(truth, 1, NULL, 0, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_negative(array, &output) ==
         TERLAN_NDARRAY_UNSUPPORTED_DTYPE);
  assert(output == NULL);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_live_count() == 0);
}

static void test_extended_transcendental_family(void) {
  typedef TerlanNdArrayStatus (*UnaryFunction)(TerlanNdArrayHandle,
                                               TerlanNdArrayHandle *);
  const UnaryFunction functions[] = {
      terlan_ndarray_v1_expm1,   terlan_ndarray_v1_log1p,
      terlan_ndarray_v1_log2,    terlan_ndarray_v1_log10,
      terlan_ndarray_v1_sinh,    terlan_ndarray_v1_cosh,
      terlan_ndarray_v1_tanh,    terlan_ndarray_v1_arcsin,
      terlan_ndarray_v1_arccos,  terlan_ndarray_v1_arctan,
      terlan_ndarray_v1_arcsinh, terlan_ndarray_v1_arccosh,
      terlan_ndarray_v1_arctanh};
  const double value[] = {0.25};
  const double logarithm_input[] = {100.0};
  const double logarithm_expected[] = {2.0};
  TerlanNdArrayHandle array = NULL;
  TerlanNdArrayHandle result = NULL;
  size_t index;
  assert(terlan_ndarray_v1_from_floats(value, 1, NULL, 0, &array) ==
         TERLAN_NDARRAY_OK);
  for (index = 0; index < sizeof(functions) / sizeof(functions[0]); ++index) {
    assert(functions[index](array, &result) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  }
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_floats(logarithm_input, 1, NULL, 0, &array) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_log10(array, &result) == TERLAN_NDARRAY_OK);
  assert_floats(result, logarithm_expected, 1, 1e-12);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(array) == TERLAN_NDARRAY_OK);
}

static void assert_complex(TerlanNdArrayHandle array, const double *real,
                           const double *imag, size_t count) {
  double *values = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_complex_reals(array, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(fabs(values[index] - real[index]) <= 1e-12);
  }
  terlan_ndarray_v1_delete_float_values(values);
  assert(terlan_ndarray_v1_complex_imags(array, &values, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    assert(fabs(values[index] - imag[index]) <= 1e-12);
  }
  terlan_ndarray_v1_delete_float_values(values);
}

static void test_round_decimals_and_output_contracts(void) {
  const int64_t pair_shape[] = {2};
  const double decimal_values[] = {12.15, 15.51};
  const double decimal_expected[] = {12.2, 15.5};
  const double tens_expected[] = {10.0, 20.0};
  const double complex_real[] = {4.5, 12.5};
  const double complex_imag[] = {1.5, 15.5};
  const double complex_units_real[] = {4.0, 12.0};
  const double complex_units_imag[] = {2.0, 16.0};
  const double complex_tens_real[] = {0.0, 10.0};
  const double complex_tens_imag[] = {0.0, 20.0};
  const double conjugate_imag[] = {-1.5, -15.5};
  const double overlap_values[] = {1.2, 2.6, 3.4, 4.8};
  const double overlap_expected[] = {1.2, 1.0, 3.0, 3.0};
  TerlanNdArrayHandle decimals = NULL;
  TerlanNdArrayHandle complex = NULL;
  TerlanNdArrayHandle destination = NULL;
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayHandle overlap = NULL;
  TerlanNdArrayHandle overlap_source = NULL;
  TerlanNdArrayHandle overlap_destination = NULL;
  TerlanNdArrayHandle wrong = NULL;
  TerlanNdArrayHandle scalar = NULL;
  TerlanNdArrayHandle read_only = NULL;

  assert(terlan_ndarray_v1_from_floats(decimal_values, 2, pair_shape, 1,
                                       &decimals) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_round_decimals(decimals, 1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, decimal_expected, 2, 1e-12);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_round_decimals(decimals, -1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_floats(result, tens_expected, 2, 0.0);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_complex(
             complex_real, 2, complex_imag, 2, pair_shape, 1,
             TERLAN_NDARRAY_DTYPE_COMPLEX128, &complex) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_round_decimals(complex, 0, &result) ==
         TERLAN_NDARRAY_OK);
  assert_complex(result, complex_units_real, complex_units_imag, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_round_decimals(complex, -1, &result) ==
         TERLAN_NDARRAY_OK);
  assert_complex(result, complex_tens_real, complex_tens_imag, 2);
  assert(terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK);

  assert(terlan_ndarray_v1_from_complex(
             (const double[]){0.0, 0.0}, 2, (const double[]){0.0, 0.0}, 2,
             pair_shape, 1, TERLAN_NDARRAY_DTYPE_COMPLEX128, &destination) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_conjugate_into(destination, complex) ==
         TERLAN_NDARRAY_OK);
  assert_complex(destination, complex_real, conjugate_imag, 2);
  assert(terlan_ndarray_v1_round_into(destination, complex, -1) ==
         TERLAN_NDARRAY_OK);
  assert_complex(destination, complex_tens_real, complex_tens_imag, 2);

  assert(terlan_ndarray_v1_from_floats(overlap_values, 4,
                                       (const int64_t[]){4}, 1, &overlap) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_slice_axis(overlap, 0, 0, 3, 1,
                                     &overlap_source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_slice_axis(overlap, 0, 1, 4, 1,
                                     &overlap_destination) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_round_into(overlap_destination, overlap_source, 0) ==
         TERLAN_NDARRAY_OK);
  assert_floats(overlap, overlap_expected, 4, 0.0);

  assert(terlan_ndarray_v1_from_floats((const double[]){7.0}, 1, NULL, 0,
                                       &wrong) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_round_into(wrong, decimals, 0) ==
         TERLAN_NDARRAY_SHAPE_MISMATCH);
  assert_floats(wrong, (const double[]){7.0}, 1, 0.0);
  assert(terlan_ndarray_v1_from_floats((const double[]){7.0}, 1, NULL, 0,
                                       &scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_broadcast_to(scalar, pair_shape, 1, &read_only) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_round_into(read_only, decimals, 0) ==
         TERLAN_NDARRAY_READ_ONLY);
  assert_floats(read_only, (const double[]){7.0, 7.0}, 2, 0.0);

  assert(terlan_ndarray_v1_delete(read_only) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(scalar) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(wrong) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(overlap_destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(overlap_source) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(overlap) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(destination) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(complex) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(decimals) == TERLAN_NDARRAY_OK);
}

int main(void) {
  test_real_unary_families();
  test_transcendental_and_non_finite_classification();
  test_complex_absolute_conjugate_and_square();
  test_reciprocal_zero_and_bool_negative_follow_numeric_rules();
  test_extended_transcendental_family();
  test_round_decimals_and_output_contracts();
  assert(terlan_ndarray_v1_live_count() == 0);
  return 0;
}
