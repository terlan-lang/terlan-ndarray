#include "terlan_ndarray_internal.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>

typedef enum UnaryOperation {
  UNARY_NEGATIVE,
  UNARY_POSITIVE,
  UNARY_ABSOLUTE,
  UNARY_SIGN,
  UNARY_SQUARE,
  UNARY_SQRT,
  UNARY_RECIPROCAL,
  UNARY_EXP,
  UNARY_EXPM1,
  UNARY_LOG,
  UNARY_LOG1P,
  UNARY_LOG2,
  UNARY_LOG10,
  UNARY_SIN,
  UNARY_COS,
  UNARY_TAN,
  UNARY_SINH,
  UNARY_COSH,
  UNARY_TANH,
  UNARY_ARCSIN,
  UNARY_ARCCOS,
  UNARY_ARCTAN,
  UNARY_ARCSINH,
  UNARY_ARCCOSH,
  UNARY_ARCTANH,
  UNARY_FLOOR,
  UNARY_CEIL,
  UNARY_ROUND,
  UNARY_LOGICAL_NOT,
  UNARY_IS_NAN,
  UNARY_IS_INFINITE,
  UNARY_IS_FINITE,
  UNARY_CONJUGATE,
  UNARY_EXP2,
  UNARY_CBRT,
  UNARY_DEGREES,
  UNARY_RADIANS,
  UNARY_SIGN_BIT
} UnaryOperation;

static int unary_is_transcendental(UnaryOperation operation) {
  return operation == UNARY_SQRT ||
         (operation >= UNARY_EXP && operation <= UNARY_ARCTANH) ||
         (operation >= UNARY_EXP2 && operation <= UNARY_RADIANS);
}

static int32_t unary_dtype(int32_t input, UnaryOperation operation) {
  if (operation == UNARY_LOGICAL_NOT || operation == UNARY_IS_NAN ||
      operation == UNARY_IS_INFINITE || operation == UNARY_IS_FINITE) {
    return TERLAN_NDARRAY_DTYPE_BOOL;
  }
  if (operation == UNARY_SIGN_BIT) {
    return TERLAN_NDARRAY_DTYPE_BOOL;
  }
  if (operation == UNARY_ABSOLUTE && dtype_is_complex(input)) {
    return input == TERLAN_NDARRAY_DTYPE_COMPLEX64
               ? TERLAN_NDARRAY_DTYPE_FLOAT32
               : TERLAN_NDARRAY_DTYPE_FLOAT64;
  }
  if (unary_is_transcendental(operation) &&
      !dtype_is_float(input) && !dtype_is_complex(input)) {
    return TERLAN_NDARRAY_DTYPE_FLOAT64;
  }
  return input;
}

static TerlanNdArrayStatus calculate_unary(UnaryOperation operation,
                                            TerlanNdArrayScalar input,
                                            int32_t input_dtype,
                                            TerlanNdArrayScalar *output) {
  const int input_complex = dtype_is_complex(input_dtype);
  const long double complex value = CMPLXL(input.real, input.imag);
  long double complex result = 0.0L;
  output->real = 0.0L;
  output->imag = 0.0L;
  switch (operation) {
  case UNARY_LOGICAL_NOT:
    output->real = input.real == 0.0L && input.imag == 0.0L;
    return TERLAN_NDARRAY_OK;
  case UNARY_IS_NAN:
    output->real = isnan(input.real) || isnan(input.imag);
    return TERLAN_NDARRAY_OK;
  case UNARY_IS_INFINITE:
    output->real = isinf(input.real) || isinf(input.imag);
    return TERLAN_NDARRAY_OK;
  case UNARY_IS_FINITE:
    output->real = isfinite(input.real) && isfinite(input.imag);
    return TERLAN_NDARRAY_OK;
  case UNARY_NEGATIVE:
    output->real = -input.real;
    output->imag = -input.imag;
    return TERLAN_NDARRAY_OK;
  case UNARY_POSITIVE:
    *output = input;
    return TERLAN_NDARRAY_OK;
  case UNARY_ABSOLUTE:
    output->real = input_complex ? cabsl(value) : fabsl(input.real);
    return TERLAN_NDARRAY_OK;
  case UNARY_SIGN:
    if (input_complex) {
      const int real_sign = (input.real > 0.0L) - (input.real < 0.0L);
      const int imaginary_sign = (input.imag > 0.0L) - (input.imag < 0.0L);
      output->real = isnan(input.real) ||
                             (input.real == 0.0L && isnan(input.imag))
                         ? NAN
                         : real_sign != 0 ? real_sign : imaginary_sign;
      return TERLAN_NDARRAY_OK;
    } else {
      output->real = isnan(input.real)
                         ? NAN
                         : (input.real > 0.0L) - (input.real < 0.0L);
      return TERLAN_NDARRAY_OK;
    }
    break;
  case UNARY_SQUARE:
    output->real = input_complex
                       ? input.real * input.real - input.imag * input.imag
                       : input.real * input.real;
    output->imag = input_complex ? 2.0L * input.real * input.imag : 0.0L;
    return TERLAN_NDARRAY_OK;
  case UNARY_SQRT:
    result = input_complex ? csqrtl(value) : sqrtl(input.real);
    break;
  case UNARY_RECIPROCAL:
    if (input.real == 0.0L && input.imag == 0.0L) {
      if (dtype_is_float(input_dtype)) {
        output->real = copysignl(INFINITY, input.real);
        return TERLAN_NDARRAY_OK;
      }
      if (input_complex) {
        output->real = NAN;
        output->imag = NAN;
        return TERLAN_NDARRAY_OK;
      }
      return TERLAN_NDARRAY_DIVIDE_BY_ZERO;
    }
    if (!input_complex) {
      output->real = 1.0L / input.real;
      return TERLAN_NDARRAY_OK;
    }
    result = 1.0L / value;
    break;
  case UNARY_EXP:
    result = input_complex ? cexpl(value) : expl(input.real);
    break;
  case UNARY_EXPM1:
    if (input_complex) {
      output->real =
          expm1l(input.real) * cosl(input.imag) -
          2.0L * sinl(input.imag / 2.0L) * sinl(input.imag / 2.0L);
      output->imag = expl(input.real) * sinl(input.imag);
      return TERLAN_NDARRAY_OK;
    }
    result = expm1l(input.real);
    break;
  case UNARY_LOG:
    result = input_complex ? clogl(value) : logl(input.real);
    break;
  case UNARY_LOG1P:
    result = input_complex ? clogl(1.0L + value) : log1pl(input.real);
    break;
  case UNARY_LOG2:
    result = input_complex ? clogl(value) / logl(2.0L) : log2l(input.real);
    break;
  case UNARY_LOG10:
    result = input_complex ? clogl(value) / logl(10.0L) : log10l(input.real);
    break;
  case UNARY_SIN:
    result = input_complex ? csinl(value) : sinl(input.real);
    break;
  case UNARY_COS:
    result = input_complex ? ccosl(value) : cosl(input.real);
    break;
  case UNARY_TAN:
    result = input_complex ? ctanl(value) : tanl(input.real);
    break;
  case UNARY_SINH:
    result = input_complex ? csinhl(value) : sinhl(input.real);
    break;
  case UNARY_COSH:
    result = input_complex ? ccoshl(value) : coshl(input.real);
    break;
  case UNARY_TANH:
    result = input_complex ? ctanhl(value) : tanhl(input.real);
    break;
  case UNARY_ARCSIN:
    result = input_complex ? casinl(value) : asinl(input.real);
    break;
  case UNARY_ARCCOS:
    result = input_complex ? cacosl(value) : acosl(input.real);
    break;
  case UNARY_ARCTAN:
    result = input_complex ? catanl(value) : atanl(input.real);
    break;
  case UNARY_ARCSINH:
    result = input_complex ? casinhl(value) : asinhl(input.real);
    break;
  case UNARY_ARCCOSH:
    result = input_complex ? cacoshl(value) : acoshl(input.real);
    break;
  case UNARY_ARCTANH:
    result = input_complex ? catanhl(value) : atanhl(input.real);
    break;
  case UNARY_FLOOR:
  case UNARY_CEIL:
    if (input_complex) {
      return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
    }
    output->real = operation == UNARY_FLOOR ? floorl(input.real)
                                             : ceill(input.real);
    return TERLAN_NDARRAY_OK;
  case UNARY_ROUND:
    output->real = nearbyintl(input.real);
    output->imag = nearbyintl(input.imag);
    return TERLAN_NDARRAY_OK;
  case UNARY_CONJUGATE:
    output->real = input.real;
    output->imag = -input.imag;
    return TERLAN_NDARRAY_OK;
  case UNARY_EXP2:
    result = input_complex ? cpowl(2.0L, value) : exp2l(input.real);
    break;
  case UNARY_CBRT:
    if (input_complex) {
      return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
    }
    output->real = cbrtl(input.real);
    return TERLAN_NDARRAY_OK;
  case UNARY_DEGREES:
  case UNARY_RADIANS:
    if (input_complex) {
      return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
    }
    output->real = operation == UNARY_DEGREES
                       ? input.real * 180.0L / acosl(-1.0L)
                       : input.real * acosl(-1.0L) / 180.0L;
    return TERLAN_NDARRAY_OK;
  case UNARY_SIGN_BIT:
    if (input_complex) {
      return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
    }
    output->real = signbit(input.real) != 0;
    return TERLAN_NDARRAY_OK;
  }
  output->real = creall(result);
  output->imag = cimagl(result);
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus unary_operation(TerlanNdArrayHandle array,
                                            UnaryOperation operation,
                                            TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle result = NULL;
  const int32_t output_dtype =
      status == TERLAN_NDARRAY_OK ? unary_dtype(array->dtype, operation) : 0;
  int64_t index;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if ((operation == UNARY_NEGATIVE || operation == UNARY_POSITIVE) &&
      dtype_is_bool(array->dtype)) {
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  status = create_empty_array(array->shape, array->rank, output_dtype, &result);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (index = 0; index < array->numel; ++index) {
    TerlanNdArrayScalar input;
    TerlanNdArrayScalar output;
    status = dtype_load(array_const_element_pointer(array, index), array->dtype,
                        &input);
    if (status == TERLAN_NDARRAY_OK) {
      status = calculate_unary(operation, input, array->dtype, &output);
    }
    if (status == TERLAN_NDARRAY_OK && operation == UNARY_NEGATIVE &&
        dtype_is_integer(array->dtype)) {
      const uint64_t value = dtype_is_signed(array->dtype)
                                 ? (uint64_t)(int64_t)input.real
                                 : (uint64_t)input.real;
      const uint64_t negated = dtype_truncate_integer(
          UINT64_C(0) - value, array->dtype);
      output.real = dtype_is_signed(array->dtype)
                        ? (long double)dtype_signed_integer(negated,
                                                            array->dtype)
                        : (long double)negated;
    }
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_store(array_element_pointer(result, index), output_dtype,
                           output);
    }
    if (status != TERLAN_NDARRAY_OK) {
      (void)terlan_ndarray_v1_delete(result);
      return status == TERLAN_NDARRAY_CAST_OVERFLOW
                 ? TERLAN_NDARRAY_ARITHMETIC_OVERFLOW
                 : status;
    }
  }
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}

#define UNARY_WRAPPER(name, operation)                                        \
  TerlanNdArrayStatus name(TerlanNdArrayHandle array,                         \
                           TerlanNdArrayHandle *out_array) {                   \
    return unary_operation(array, operation, out_array);                      \
  }

UNARY_WRAPPER(terlan_ndarray_v1_negative, UNARY_NEGATIVE)
UNARY_WRAPPER(terlan_ndarray_v1_positive, UNARY_POSITIVE)
UNARY_WRAPPER(terlan_ndarray_v1_absolute, UNARY_ABSOLUTE)
UNARY_WRAPPER(terlan_ndarray_v1_sign, UNARY_SIGN)
UNARY_WRAPPER(terlan_ndarray_v1_square, UNARY_SQUARE)
UNARY_WRAPPER(terlan_ndarray_v1_sqrt, UNARY_SQRT)
UNARY_WRAPPER(terlan_ndarray_v1_reciprocal, UNARY_RECIPROCAL)
UNARY_WRAPPER(terlan_ndarray_v1_exp, UNARY_EXP)
UNARY_WRAPPER(terlan_ndarray_v1_expm1, UNARY_EXPM1)
UNARY_WRAPPER(terlan_ndarray_v1_log, UNARY_LOG)
UNARY_WRAPPER(terlan_ndarray_v1_log1p, UNARY_LOG1P)
UNARY_WRAPPER(terlan_ndarray_v1_log2, UNARY_LOG2)
UNARY_WRAPPER(terlan_ndarray_v1_log10, UNARY_LOG10)
UNARY_WRAPPER(terlan_ndarray_v1_sin, UNARY_SIN)
UNARY_WRAPPER(terlan_ndarray_v1_cos, UNARY_COS)
UNARY_WRAPPER(terlan_ndarray_v1_tan, UNARY_TAN)
UNARY_WRAPPER(terlan_ndarray_v1_sinh, UNARY_SINH)
UNARY_WRAPPER(terlan_ndarray_v1_cosh, UNARY_COSH)
UNARY_WRAPPER(terlan_ndarray_v1_tanh, UNARY_TANH)
UNARY_WRAPPER(terlan_ndarray_v1_arcsin, UNARY_ARCSIN)
UNARY_WRAPPER(terlan_ndarray_v1_arccos, UNARY_ARCCOS)
UNARY_WRAPPER(terlan_ndarray_v1_arctan, UNARY_ARCTAN)
UNARY_WRAPPER(terlan_ndarray_v1_arcsinh, UNARY_ARCSINH)
UNARY_WRAPPER(terlan_ndarray_v1_arccosh, UNARY_ARCCOSH)
UNARY_WRAPPER(terlan_ndarray_v1_arctanh, UNARY_ARCTANH)
UNARY_WRAPPER(terlan_ndarray_v1_floor, UNARY_FLOOR)
UNARY_WRAPPER(terlan_ndarray_v1_ceil, UNARY_CEIL)
UNARY_WRAPPER(terlan_ndarray_v1_round, UNARY_ROUND)
UNARY_WRAPPER(terlan_ndarray_v1_logical_not, UNARY_LOGICAL_NOT)
UNARY_WRAPPER(terlan_ndarray_v1_is_nan, UNARY_IS_NAN)
UNARY_WRAPPER(terlan_ndarray_v1_is_infinite, UNARY_IS_INFINITE)
UNARY_WRAPPER(terlan_ndarray_v1_is_finite, UNARY_IS_FINITE)
UNARY_WRAPPER(terlan_ndarray_v1_conjugate, UNARY_CONJUGATE)
UNARY_WRAPPER(terlan_ndarray_v1_exp2, UNARY_EXP2)
UNARY_WRAPPER(terlan_ndarray_v1_cbrt, UNARY_CBRT)
UNARY_WRAPPER(terlan_ndarray_v1_degrees, UNARY_DEGREES)
UNARY_WRAPPER(terlan_ndarray_v1_radians, UNARY_RADIANS)
UNARY_WRAPPER(terlan_ndarray_v1_sign_bit, UNARY_SIGN_BIT)

static long double round_decimal_component(long double value,
                                           long double factor,
                                           int64_t decimals) {
  long double scaled;
  if (!isfinite(value)) {
    return value;
  }
  if (!isfinite(factor)) {
    return decimals < 0 ? copysignl(0.0L, value) : value;
  }
  if (decimals < 0) {
    return nearbyintl(value / factor) * factor;
  }
  scaled = value * factor;
  return isfinite(scaled) ? nearbyintl(scaled) / factor : value;
}

TerlanNdArrayStatus terlan_ndarray_v1_round_decimals(
    TerlanNdArrayHandle array, int64_t decimals,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle result = NULL;
  const long double factor = powl(10.0L, fabsl((long double)decimals));
  int64_t index;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = create_empty_array(array->shape, array->rank, array->dtype, &result);
  for (index = 0; status == TERLAN_NDARRAY_OK && index < array->numel;
       ++index) {
    TerlanNdArrayScalar value;
    status = dtype_load(array_const_element_pointer(array, index), array->dtype,
                        &value);
    if (status == TERLAN_NDARRAY_OK) {
      value.real = round_decimal_component(value.real, factor, decimals);
      value.imag = round_decimal_component(value.imag, factor, decimals);
    }
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_store(array_element_pointer(result, index), result->dtype,
                           value);
    }
  }
  if (status != TERLAN_NDARRAY_OK) {
    if (result != NULL) {
      (void)terlan_ndarray_v1_delete(result);
    }
    return status;
  }
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_round_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    int64_t decimals) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status =
      decimals == 0 ? terlan_ndarray_v1_round(array, &result)
                    : terlan_ndarray_v1_round_decimals(array, decimals,
                                                       &result);
  return array_finish_into(destination, status, result,
                           TERLAN_NDARRAY_CAST_SAME_KIND);
}

TerlanNdArrayStatus terlan_ndarray_v1_conjugate_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status =
      terlan_ndarray_v1_conjugate(array, &result);
  return array_finish_into(destination, status, result,
                           TERLAN_NDARRAY_CAST_SAME_KIND);
}
