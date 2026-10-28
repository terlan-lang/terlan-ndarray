#include "terlan_ndarray_internal.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>

static uint64_t integer_operand(long double value, int32_t dtype) {
  return dtype_is_signed(dtype) ? (uint64_t)(int64_t)value : (uint64_t)value;
}

uint64_t scalar_unsigned_gcd(uint64_t lhs, uint64_t rhs) {
  while (rhs != 0U) {
    const uint64_t remainder = lhs % rhs;
    lhs = rhs;
    rhs = remainder;
  }
  return lhs;
}

uint64_t scalar_integer_magnitude(long double value) {
  const int64_t signed_value = (int64_t)value;
  return signed_value < 0 ? (uint64_t)(-(signed_value + 1)) + UINT64_C(1)
                          : (uint64_t)signed_value;
}

void scalar_complex_power(TerlanNdArrayScalar lhs, TerlanNdArrayScalar rhs,
                          TerlanNdArrayScalar *out_value) {
  if (lhs.real == 0.0L && lhs.imag == 0.0L) {
    if (rhs.real == 0.0L && rhs.imag == 0.0L) {
      out_value->real = 1.0L;
      out_value->imag = 0.0L;
    } else if (rhs.real > 0.0L) {
      out_value->real = 0.0L;
      out_value->imag = 0.0L;
    } else {
      out_value->real = NAN;
      out_value->imag = NAN;
    }
    return;
  }
  {
    const long double complex result =
        cpowl(CMPLXL(lhs.real, lhs.imag), CMPLXL(rhs.real, rhs.imag));
    out_value->real = creall(result);
    out_value->imag = cimagl(result);
  }
}

static uint64_t wrapping_integer_power(uint64_t base, uint64_t exponent) {
  uint64_t result = UINT64_C(1);
  while (exponent != 0U) {
    if ((exponent & UINT64_C(1)) != 0U) {
      result *= base;
    }
    exponent >>= 1U;
    if (exponent != 0U) {
      base *= base;
    }
  }
  return result;
}

TerlanNdArrayStatus scalar_wrapping_integer_binary(
    TerlanNdArrayIntegerBinaryOperation operation, long double lhs,
    long double rhs, int32_t dtype, TerlanNdArrayScalar *out_value) {
  const uint64_t left = integer_operand(lhs, dtype);
  const uint64_t right = integer_operand(rhs, dtype);
  uint64_t result;
  if (out_value == NULL || !dtype_is_integer(dtype)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (operation == TERLAN_NDARRAY_INTEGER_ADD) {
    result = left + right;
  } else if (operation == TERLAN_NDARRAY_INTEGER_SUBTRACT) {
    result = left - right;
  } else if (operation == TERLAN_NDARRAY_INTEGER_MULTIPLY) {
    result = left * right;
  } else if (operation == TERLAN_NDARRAY_INTEGER_POWER) {
    if (rhs < 0.0L) {
      return TERLAN_NDARRAY_DOMAIN_ERROR;
    }
    result = wrapping_integer_power(left, right);
  } else {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  result = dtype_truncate_integer(result, dtype);
  out_value->real = dtype_is_signed(dtype)
                        ? (long double)dtype_signed_integer(result, dtype)
                        : (long double)result;
  out_value->imag = 0.0L;
  return TERLAN_NDARRAY_OK;
}

void scalar_floating_floor_divmod(long double lhs, long double rhs,
                                  long double *out_quotient,
                                  long double *out_remainder) {
  long double remainder = fmodl(lhs, rhs);
  long double quotient = (lhs - remainder) / rhs;
  if (remainder != 0.0L) {
    if (signbit(remainder) != signbit(rhs)) {
      remainder += rhs;
      quotient -= 1.0L;
    }
  } else {
    remainder = copysignl(0.0L, rhs);
  }
  if (quotient != 0.0L) {
    const long double floored = floorl(quotient);
    quotient = quotient - floored > 0.5L ? floored + 1.0L : floored;
  } else {
    quotient = copysignl(0.0L, lhs / rhs);
  }
  *out_quotient = quotient;
  *out_remainder = remainder;
}
