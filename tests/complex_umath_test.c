#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef TerlanNdArrayStatus (*UnaryFunction)(TerlanNdArrayHandle,
                                             TerlanNdArrayHandle *);

static TerlanNdArrayHandle complex_array(const double *real,
                                         const double *imag, size_t count,
                                         int32_t dtype) {
  const int64_t shape[] = {(int64_t)count};
  TerlanNdArrayHandle array = NULL;
  assert(terlan_ndarray_v1_from_complex(
             real, (int64_t)count, imag, (int64_t)count, shape, 1, dtype,
             &array) == TERLAN_NDARRAY_OK);
  return array;
}

static int component_matches(double actual, double expected,
                             double tolerance, int check_zero_sign) {
  if (isnan(expected)) {
    return isnan(actual);
  }
  if (isinf(expected)) {
    return isinf(actual) && signbit(actual) == signbit(expected);
  }
  if (expected == 0.0 && check_zero_sign) {
    return actual == 0.0 && signbit(actual) == signbit(expected);
  }
  return fabs(actual - expected) <= tolerance;
}

static void assert_real(TerlanNdArrayHandle array, const double *expected,
                        size_t count, double tolerance,
                        int check_zero_sign) {
  double *actual = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_to_floats_cast(array, &actual, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    if (!component_matches(actual[index], expected[index], tolerance,
                           check_zero_sign)) {
      fprintf(stderr, "real[%zu]: actual=%a expected=%a tolerance=%g\n",
              index, actual[index], expected[index], tolerance);
      assert(0);
    }
  }
  terlan_ndarray_v1_delete_float_values(actual);
}

static void assert_complex(TerlanNdArrayHandle array, const double *real,
                           const double *imag, size_t count, double tolerance,
                           int check_zero_sign) {
  double *actual = NULL;
  size_t length = 0;
  size_t index;
  assert(terlan_ndarray_v1_complex_reals(array, &actual, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    if (!component_matches(actual[index], real[index], tolerance,
                           check_zero_sign)) {
      fprintf(stderr,
              "complex.real[%zu]: actual=%a expected=%a tolerance=%g\n",
              index, actual[index], real[index], tolerance);
      assert(0);
    }
  }
  terlan_ndarray_v1_delete_float_values(actual);
  actual = NULL;
  assert(terlan_ndarray_v1_complex_imags(array, &actual, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == count);
  for (index = 0; index < count; ++index) {
    if (!component_matches(actual[index], imag[index], tolerance,
                           check_zero_sign)) {
      fprintf(stderr,
              "complex.imag[%zu]: actual=%a expected=%a tolerance=%g\n",
              index, actual[index], imag[index], tolerance);
      assert(0);
    }
  }
  terlan_ndarray_v1_delete_float_values(actual);
}

static void assert_unary_complex(UnaryFunction function, const double *input_real,
                                 const double *input_imag,
                                 const double *expected_real,
                                 const double *expected_imag, size_t count,
                                 int32_t dtype, int check_zero_sign) {
  const double tolerance =
      dtype == TERLAN_NDARRAY_DTYPE_COMPLEX64 ? 2e-6 : 1e-12;
  TerlanNdArrayHandle input = complex_array(input_real, input_imag, count, dtype);
  TerlanNdArrayHandle output = NULL;
  assert(function(input, &output) == TERLAN_NDARRAY_OK);
  assert_complex(output, expected_real, expected_imag, count, tolerance,
                 check_zero_sign);
  assert_complex(input, input_real, input_imag, count, 0.0, 1);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(input) == TERLAN_NDARRAY_OK);
}

static void test_complex_exp_simple_and_c99_special_values(void) {
  const double pi = acos(-1.0);
  const double finite_real[] = {1.0, 0.0, 1.0};
  const double finite_imag[] = {0.0, 1.0, 1.0};
  const double finite_expected_real[] = {exp(1.0), cos(1.0),
                                         exp(1.0) * cos(1.0)};
  const double finite_expected_imag[] = {0.0, sin(1.0),
                                         exp(1.0) * sin(1.0)};
  const double special_real[] = {
      0.0, -0.0, 1.0, -1.0, 0.0, INFINITY, -INFINITY, -INFINITY,
      INFINITY, INFINITY, -INFINITY, INFINITY, -INFINITY, INFINITY,
      NAN, NAN, NAN, NAN, NAN};
  const double special_imag[] = {
      0.0, 0.0, INFINITY, INFINITY, INFINITY, 0.0, 1.0, 0.75 * pi,
      1.0, 0.75 * pi, INFINITY, INFINITY, NAN, NAN, 1.0, -1.0,
      INFINITY, -INFINITY, NAN};
  const double expected_real[] = {
      1.0, 1.0, NAN, NAN, NAN, INFINITY, 0.0, -0.0, INFINITY,
      -INFINITY, 0.0, INFINITY, 0.0, INFINITY, NAN, NAN, NAN, NAN, NAN};
  const double expected_imag[] = {
      0.0, 0.0, NAN, NAN, NAN, 0.0, 0.0, 0.0, INFINITY, INFINITY,
      0.0, NAN, 0.0, NAN, NAN, NAN, NAN, NAN, NAN};
  assert_unary_complex(terlan_ndarray_v1_exp, finite_real, finite_imag,
                       finite_expected_real, finite_expected_imag, 3,
                       TERLAN_NDARRAY_DTYPE_COMPLEX128, 0);
  assert_unary_complex(terlan_ndarray_v1_exp, special_real, special_imag,
                       expected_real, expected_imag, 19,
                       TERLAN_NDARRAY_DTYPE_COMPLEX128, 0);
}

static void test_complex_log_simple_and_c99_special_values(void) {
  const double pi = acos(-1.0);
  const double real[] = {-0.0, 0.0, 1.0, -1.0, 1.0, INFINITY, -INFINITY,
                         INFINITY, -INFINITY, INFINITY, -INFINITY, NAN, NAN,
                         NAN, 1.0, 1.0};
  const double imag[] = {0.0, 0.0, INFINITY, INFINITY, NAN, 1.0, 1.0,
                         INFINITY, INFINITY, NAN, NAN, 1.0, INFINITY, NAN,
                         0.0, 2.0};
  const double expected_real[] = {
      -INFINITY, -INFINITY, INFINITY, INFINITY, NAN, INFINITY, INFINITY,
      INFINITY, INFINITY, INFINITY, INFINITY, NAN, INFINITY, NAN,
      0.0, 0.5 * log(5.0)};
  const double expected_imag[] = {
      pi, 0.0, 0.5 * pi, 0.5 * pi, NAN, 0.0, pi, 0.25 * pi,
      0.75 * pi, NAN, NAN, NAN, NAN, NAN, 0.0, atan2(2.0, 1.0)};
  assert_unary_complex(terlan_ndarray_v1_log, real, imag, expected_real,
                       expected_imag, 16, TERLAN_NDARRAY_DTYPE_COMPLEX128, 1);

  {
    const double conjugate_real[] = {-INFINITY, INFINITY, 1.0};
    const double conjugate_imag[] = {-1.0, -1.0, -2.0};
    const double conjugate_expected_real[] = {INFINITY, INFINITY,
                                               0.5 * log(5.0)};
    const double conjugate_expected_imag[] = {-pi, -0.0,
                                               -atan2(2.0, 1.0)};
    assert_unary_complex(terlan_ndarray_v1_log, conjugate_real,
                         conjugate_imag, conjugate_expected_real,
                         conjugate_expected_imag, 3,
                         TERLAN_NDARRAY_DTYPE_COMPLEX128, 1);
  }
}

static void test_complex_sqrt_branch_and_c99_special_values(void) {
  const double root_half = 0.5 * sqrt(2.0);
  const double finite_real[] = {1.0, 0.0, -1.0, 1.0, 1.0};
  const double finite_imag[] = {0.0, 1.0, 0.0, 1.0, -1.0};
  const double finite_expected_real[] = {1.0, root_half, 0.0,
                                          1.09868411346781,
                                          1.09868411346781};
  const double finite_expected_imag[] = {0.0, root_half, 1.0,
                                          0.455089860562227,
                                          -0.455089860562227};
  const double special_real[] = {
      0.0, -0.0, 1.0, -1.0, 0.0, -0.0, INFINITY, -INFINITY, NAN,
      1.0, -1.0, 0.0, -INFINITY, INFINITY, -INFINITY, INFINITY,
      NAN, NAN, NAN};
  const double special_imag[] = {
      0.0, 0.0, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY,
      INFINITY, INFINITY, NAN, NAN, NAN, 1.0, 1.0, NAN, NAN,
      0.0, 1.0, NAN};
  const double expected_real[] = {
      0.0, 0.0, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY,
      INFINITY, INFINITY, NAN, NAN, NAN, 0.0, INFINITY, NAN,
      INFINITY, NAN, NAN, NAN};
  const double expected_imag[] = {
      0.0, 0.0, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY,
      INFINITY, INFINITY, NAN, NAN, NAN, INFINITY, 0.0, INFINITY,
      NAN, NAN, NAN, NAN};
  assert_unary_complex(terlan_ndarray_v1_sqrt, finite_real, finite_imag,
                       finite_expected_real, finite_expected_imag, 5,
                       TERLAN_NDARRAY_DTYPE_COMPLEX128, 1);
  assert_unary_complex(terlan_ndarray_v1_sqrt, special_real, special_imag,
                       expected_real, expected_imag, 19,
                       TERLAN_NDARRAY_DTYPE_COMPLEX128, 0);
}

static void test_complex_power_scalar_array_and_non_finite_values(void) {
  const double base_real[] = {1.0, 0.0, 2.0, 2.5, INFINITY, NAN};
  const double base_imag[] = {0.0, 1.0, 0.0, 0.37, 0.0, 0.0};
  const double exponent_real[] = {1.0, 0.0, -0.5, -0.5, 2.0, 3.0};
  const double exponent_imag[] = {0.0, 1.0, 1.5, 1.5, 0.0, 0.0};
  const double expected_real[] = {1.0, 0.20787957635076193,
                                  0.35812203996480685,
                                  0.12659112128185032, INFINITY, NAN};
  const double expected_imag[] = {0.0, 0.0, 0.6097119028618724,
                                  0.48847676699581527, NAN, NAN};
  const double square_real[] = {1.0, 0.0, 1.0, INFINITY, NAN};
  const double square_imag[] = {1.0, 2.0, 2.0, 0.0, 0.0};
  const double squared_expected_real[] = {0.0, -4.0, -3.0, INFINITY, NAN};
  const double squared_expected_imag[] = {2.0, 0.0, 4.0, NAN, NAN};
  const int64_t scalar_shape[] = {1};
  TerlanNdArrayHandle base = complex_array(base_real, base_imag, 6,
                                            TERLAN_NDARRAY_DTYPE_COMPLEX128);
  TerlanNdArrayHandle exponent = complex_array(
      exponent_real, exponent_imag, 6, TERLAN_NDARRAY_DTYPE_COMPLEX128);
  TerlanNdArrayHandle output = NULL;
  TerlanNdArrayHandle square_base = complex_array(
      square_real, square_imag, 5, TERLAN_NDARRAY_DTYPE_COMPLEX128);
  TerlanNdArrayHandle two = NULL;

  assert(terlan_ndarray_v1_power(base, exponent, &output) ==
         TERLAN_NDARRAY_OK);
  assert_complex(output, expected_real, expected_imag, 6, 1e-12, 0);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  output = NULL;
  assert(terlan_ndarray_v1_from_complex((const double[]){2.0}, 1,
                                         (const double[]){0.0}, 1,
                                         scalar_shape, 1,
                                         TERLAN_NDARRAY_DTYPE_COMPLEX128,
                                         &two) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_power(square_base, two, &output) ==
         TERLAN_NDARRAY_OK);
  assert_complex(output, squared_expected_real, squared_expected_imag, 5,
                 1e-12, 0);
  assert_complex(square_base, square_real, square_imag, 5, 0.0, 1);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(two) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(square_base) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(exponent) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(base) == TERLAN_NDARRAY_OK);
}

static void test_complex_absolute_special_values_and_strided_dtypes(void) {
  const double real[] = {NAN, -NAN, INFINITY, -INFINITY, 1.0, 1.0,
                         INFINITY, NAN};
  const double imag[] = {NAN, NAN, NAN, NAN, 0.0, -0.0, -0.0, -0.0};
  const double expected[] = {NAN, NAN, INFINITY, INFINITY, 1.0, 1.0,
                             INFINITY, NAN};
  const double stride_real[] = {3.0, 5.0, 8.0, 7.0, 20.0, 12.0, 9.0, 28.0,
                                11.0};
  const double stride_imag[] = {4.0, 12.0, 15.0, 24.0, 21.0, 35.0, 40.0,
                                45.0, 60.0};
  const int32_t dtypes[] = {TERLAN_NDARRAY_DTYPE_COMPLEX64,
                            TERLAN_NDARRAY_DTYPE_COMPLEX128};
  size_t dtype_index;
  TerlanNdArrayHandle input = complex_array(real, imag, 8,
                                             TERLAN_NDARRAY_DTYPE_COMPLEX128);
  TerlanNdArrayHandle conjugated = NULL;
  TerlanNdArrayHandle output = NULL;
  assert(terlan_ndarray_v1_absolute(input, &output) == TERLAN_NDARRAY_OK);
  assert_real(output, expected, 8, 0.0, 0);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  output = NULL;
  assert(terlan_ndarray_v1_conjugate(input, &conjugated) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_absolute(conjugated, &output) ==
         TERLAN_NDARRAY_OK);
  assert_real(output, expected, 8, 0.0, 0);
  assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(conjugated) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(input) == TERLAN_NDARRAY_OK);

  for (dtype_index = 0; dtype_index < 2; ++dtype_index) {
    TerlanNdArrayHandle owned = complex_array(stride_real, stride_imag, 9,
                                               dtypes[dtype_index]);
    TerlanNdArrayHandle strided = NULL;
    TerlanNdArrayHandle reversed = NULL;
    const double tolerance = dtype_index == 0 ? 2e-6 : 0.0;
    assert(terlan_ndarray_v1_slice_axis(owned, 0, 0, 9, 2, &strided) ==
           TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_absolute(strided, &output) == TERLAN_NDARRAY_OK);
    assert_real(output, (const double[]){5.0, 17.0, 29.0, 41.0, 61.0}, 5,
                tolerance, 0);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    output = NULL;
    assert(terlan_ndarray_v1_slice_axis(owned, 0, INT64_MIN, INT64_MIN, -1,
                                       &reversed) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_absolute(reversed, &output) == TERLAN_NDARRAY_OK);
    assert_real(output,
                (const double[]){61.0, 53.0, 41.0, 37.0, 29.0, 25.0, 17.0,
                                 13.0, 5.0},
                9, tolerance, 0);
    assert_complex(owned, stride_real, stride_imag, 9, tolerance, 1);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    output = NULL;
    assert(terlan_ndarray_v1_delete(reversed) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(strided) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(owned) == TERLAN_NDARRAY_OK);
  }
}

static void test_special_complex_strided_square_and_conjugate(void) {
  const double real[] = {NAN, NAN, INFINITY, INFINITY, 0.0, INFINITY, 0.0,
                         0.0, NAN};
  const double imag[] = {NAN, INFINITY, NAN, INFINITY, INFINITY, 0.0, 0.0,
                         NAN, 0.0};
  const double square_real[] = {NAN, NAN, NAN, NAN, -INFINITY, INFINITY,
                                0.0, NAN, NAN};
  const double square_imag[] = {NAN, NAN, NAN, INFINITY, NAN, NAN, 0.0, NAN,
                                NAN};
  const int32_t dtypes[] = {TERLAN_NDARRAY_DTYPE_COMPLEX64,
                            TERLAN_NDARRAY_DTYPE_COMPLEX128};
  size_t dtype_index;
  for (dtype_index = 0; dtype_index < 2; ++dtype_index) {
    TerlanNdArrayHandle owned = complex_array(real, imag, 9, dtypes[dtype_index]);
    TerlanNdArrayHandle reversed = NULL;
    TerlanNdArrayHandle output = NULL;
    assert(terlan_ndarray_v1_slice_axis(owned, 0, INT64_MIN, INT64_MIN, -1,
                                       &reversed) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_square(reversed, &output) == TERLAN_NDARRAY_OK);
    assert_complex(output,
                   (const double[]){square_real[8], square_real[7],
                                    square_real[6], square_real[5],
                                    square_real[4], square_real[3],
                                    square_real[2], square_real[1],
                                    square_real[0]},
                   (const double[]){square_imag[8], square_imag[7],
                                    square_imag[6], square_imag[5],
                                    square_imag[4], square_imag[3],
                                    square_imag[2], square_imag[1],
                                    square_imag[0]},
                   9, dtype_index == 0 ? 2e-6 : 0.0, 0);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    output = NULL;
    assert(terlan_ndarray_v1_conjugate(reversed, &output) ==
           TERLAN_NDARRAY_OK);
    assert_complex(output,
                   (const double[]){NAN, 0.0, 0.0, INFINITY, 0.0, INFINITY,
                                    INFINITY, NAN, NAN},
                   (const double[]){-0.0, NAN, -0.0, -0.0, -INFINITY,
                                    -INFINITY, NAN, -INFINITY, NAN},
                   9, 0.0, 1);
    assert(terlan_ndarray_v1_delete(output) == TERLAN_NDARRAY_OK);
    assert_complex(owned, real, imag, 9, 0.0, 1);
    assert(terlan_ndarray_v1_delete(reversed) == TERLAN_NDARRAY_OK);
    assert(terlan_ndarray_v1_delete(owned) == TERLAN_NDARRAY_OK);
  }
}

int main(void) {
  test_complex_exp_simple_and_c99_special_values();
  test_complex_log_simple_and_c99_special_values();
  test_complex_sqrt_branch_and_c99_special_values();
  test_complex_power_scalar_array_and_non_finite_values();
  test_complex_absolute_special_values_and_strided_dtypes();
  test_special_complex_strided_square_and_conjugate();
  assert(terlan_ndarray_v1_live_count() == 0);
  return 0;
}
