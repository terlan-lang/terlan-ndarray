#include "terlan_ndarray_internal.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct TerlanComplex32 {
  float real;
  float imag;
} TerlanComplex32;

typedef struct TerlanComplex64 {
  double real;
  double imag;
} TerlanComplex64;

size_t dtype_size(int32_t dtype) {
  switch (dtype) {
  case TERLAN_NDARRAY_DTYPE_BOOL:
  case TERLAN_NDARRAY_DTYPE_INT8:
  case TERLAN_NDARRAY_DTYPE_UINT8:
    return 1;
  case TERLAN_NDARRAY_DTYPE_INT16:
  case TERLAN_NDARRAY_DTYPE_UINT16:
  case TERLAN_NDARRAY_DTYPE_FLOAT16:
    return 2;
  case TERLAN_NDARRAY_DTYPE_INT32:
  case TERLAN_NDARRAY_DTYPE_UINT32:
  case TERLAN_NDARRAY_DTYPE_FLOAT32:
    return 4;
  case TERLAN_NDARRAY_DTYPE_INT64:
  case TERLAN_NDARRAY_DTYPE_UINT64:
  case TERLAN_NDARRAY_DTYPE_FLOAT64:
  case TERLAN_NDARRAY_DTYPE_COMPLEX64:
    return 8;
  case TERLAN_NDARRAY_DTYPE_COMPLEX128:
    return 16;
  default:
    return 0;
  }
}

int dtype_is_bool(int32_t dtype) {
  return dtype == TERLAN_NDARRAY_DTYPE_BOOL;
}

int dtype_is_integer(int32_t dtype) {
  return dtype >= TERLAN_NDARRAY_DTYPE_INT8 &&
             dtype <= TERLAN_NDARRAY_DTYPE_UINT64
         ? 1
         : dtype == TERLAN_NDARRAY_DTYPE_INT64;
}

int dtype_is_signed(int32_t dtype) {
  return dtype == TERLAN_NDARRAY_DTYPE_INT8 ||
         dtype == TERLAN_NDARRAY_DTYPE_INT16 ||
         dtype == TERLAN_NDARRAY_DTYPE_INT32 ||
         dtype == TERLAN_NDARRAY_DTYPE_INT64;
}

int dtype_is_float(int32_t dtype) {
  return dtype == TERLAN_NDARRAY_DTYPE_FLOAT16 ||
         dtype == TERLAN_NDARRAY_DTYPE_FLOAT32 ||
         dtype == TERLAN_NDARRAY_DTYPE_FLOAT64;
}

int dtype_is_complex(int32_t dtype) {
  return dtype == TERLAN_NDARRAY_DTYPE_COMPLEX64 ||
         dtype == TERLAN_NDARRAY_DTYPE_COMPLEX128;
}

uint32_t dtype_integer_bits(int32_t dtype) {
  return (uint32_t)(dtype_size(dtype) * 8U);
}

uint64_t dtype_truncate_integer(uint64_t value, int32_t dtype) {
  const uint32_t bits = dtype_integer_bits(dtype);
  return bits >= 64U ? value : value & ((UINT64_C(1) << bits) - UINT64_C(1));
}

int64_t dtype_signed_integer(uint64_t value, int32_t dtype) {
  const uint32_t bits = dtype_integer_bits(dtype);
  const uint64_t truncated = dtype_truncate_integer(value, dtype);
  const uint64_t sign = UINT64_C(1) << (bits - 1U);
  return (int64_t)((truncated ^ sign) - sign);
}

static int dtype_kind(int32_t dtype) {
  if (dtype_is_bool(dtype)) {
    return 0;
  }
  if (dtype_is_integer(dtype)) {
    return 1;
  }
  if (dtype_is_float(dtype)) {
    return 2;
  }
  if (dtype_is_complex(dtype)) {
    return 3;
  }
  return -1;
}

static int dtype_precision_bits(int32_t dtype) {
  switch (dtype) {
  case TERLAN_NDARRAY_DTYPE_BOOL:
    return 1;
  case TERLAN_NDARRAY_DTYPE_INT8:
  case TERLAN_NDARRAY_DTYPE_UINT8:
    return 8;
  case TERLAN_NDARRAY_DTYPE_INT16:
  case TERLAN_NDARRAY_DTYPE_UINT16:
    return 16;
  case TERLAN_NDARRAY_DTYPE_INT32:
  case TERLAN_NDARRAY_DTYPE_UINT32:
    return 32;
  case TERLAN_NDARRAY_DTYPE_INT64:
  case TERLAN_NDARRAY_DTYPE_UINT64:
    return 64;
  case TERLAN_NDARRAY_DTYPE_FLOAT16:
    return 11;
  case TERLAN_NDARRAY_DTYPE_FLOAT32:
  case TERLAN_NDARRAY_DTYPE_COMPLEX64:
    return 24;
  case TERLAN_NDARRAY_DTYPE_FLOAT64:
  case TERLAN_NDARRAY_DTYPE_COMPLEX128:
    return 53;
  default:
    return 0;
  }
}

static int dtype_safe_cast(int32_t source, int32_t destination) {
  const int source_kind = dtype_kind(source);
  const int destination_kind = dtype_kind(destination);
  const int source_bits = dtype_precision_bits(source);
  const int destination_bits = dtype_precision_bits(destination);
  if (source == destination || source_kind == 0) {
    return source_kind >= 0 && destination_kind >= 0;
  }
  if (source_kind == 1 && destination_kind == 1) {
    if (dtype_is_signed(source) == dtype_is_signed(destination)) {
      return destination_bits >= source_bits;
    }
    return !dtype_is_signed(source) && dtype_is_signed(destination) &&
           destination_bits > source_bits;
  }
  if (source_kind == 1 &&
      (destination_kind == 2 || destination_kind == 3)) {
    return destination_bits >= source_bits;
  }
  if (source_kind == 2 && destination_kind == 2) {
    return destination_bits >= source_bits;
  }
  if (source_kind == 2 && destination_kind == 3) {
    return destination_bits >= source_bits;
  }
  if (source_kind == 3 && destination_kind == 3) {
    return destination_bits >= source_bits;
  }
  return 0;
}

int dtype_can_cast(int32_t source, int32_t destination, int32_t casting) {
  const int source_kind = dtype_kind(source);
  const int destination_kind = dtype_kind(destination);
  if (source_kind < 0 || destination_kind < 0 ||
      casting < TERLAN_NDARRAY_CAST_NO ||
      casting > TERLAN_NDARRAY_CAST_UNSAFE) {
    return 0;
  }
  if (casting == TERLAN_NDARRAY_CAST_UNSAFE) {
    return 1;
  }
  if (casting == TERLAN_NDARRAY_CAST_NO ||
      casting == TERLAN_NDARRAY_CAST_EQUIVALENT) {
    return source == destination;
  }
  if (dtype_safe_cast(source, destination)) {
    return 1;
  }
  return casting == TERLAN_NDARRAY_CAST_SAME_KIND &&
         source_kind <= destination_kind;
}

int32_t dtype_promote(int32_t lhs, int32_t rhs) {
  static const int32_t table[14][14] = {
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
      {2, 2, 3, 2, 2, 2, 2, 2, 2, 3, 3, 13, 13, 3},
      {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 13, 13, 3},
      {4, 2, 3, 4, 5, 6, 5, 6, 2, 3, 11, 12, 13, 14},
      {5, 2, 3, 5, 5, 6, 5, 6, 2, 3, 11, 12, 13, 11},
      {6, 2, 3, 6, 6, 6, 6, 6, 2, 3, 3, 13, 13, 3},
      {7, 2, 3, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
      {8, 2, 3, 6, 6, 6, 8, 8, 9, 10, 11, 12, 13, 11},
      {9, 2, 3, 2, 2, 2, 9, 9, 9, 10, 3, 13, 13, 3},
      {10, 3, 3, 3, 3, 3, 10, 10, 10, 10, 3, 13, 13, 3},
      {11, 3, 3, 11, 11, 3, 11, 11, 3, 3, 11, 12, 13, 11},
      {12, 13, 13, 12, 12, 13, 12, 12, 13, 13, 12, 12, 13, 12},
      {13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13},
      {14, 3, 3, 14, 11, 3, 14, 11, 3, 3, 11, 12, 13, 14}};
  if (lhs < 1 || lhs > 14 || rhs < 1 || rhs > 14) {
    return 0;
  }
  return table[lhs - 1][rhs - 1];
}

int32_t dtype_inexact_or_float64(int32_t dtype) {
  return dtype_is_float(dtype) || dtype_is_complex(dtype)
             ? dtype
             : TERLAN_NDARRAY_DTYPE_FLOAT64;
}

static int compare_scalar_component(long double lhs, long double rhs) {
  const int lhs_nan = isnan(lhs) != 0;
  const int rhs_nan = isnan(rhs) != 0;
  if (lhs_nan || rhs_nan) {
    if (lhs_nan != rhs_nan) {
      return lhs_nan ? 1 : -1;
    }
    return 0;
  }
  if (lhs < rhs) {
    return -1;
  }
  if (lhs > rhs) {
    return 1;
  }
  return 0;
}

int scalar_order_compare(TerlanNdArrayScalar lhs, TerlanNdArrayScalar rhs) {
  const int real_order = compare_scalar_component(lhs.real, rhs.real);
  return real_order == 0 ? compare_scalar_component(lhs.imag, rhs.imag)
                         : real_order;
}

TerlanNdArrayScalar scalar_multiply(TerlanNdArrayScalar lhs,
                                    TerlanNdArrayScalar rhs) {
  TerlanNdArrayScalar value;
  value.real = lhs.real * rhs.real - lhs.imag * rhs.imag;
  value.imag = lhs.real * rhs.imag + lhs.imag * rhs.real;
  return value;
}

TerlanNdArrayScalar scalar_divide(TerlanNdArrayScalar lhs,
                                  TerlanNdArrayScalar rhs) {
  const long double denominator = rhs.real * rhs.real + rhs.imag * rhs.imag;
  TerlanNdArrayScalar value;
  value.real = (lhs.real * rhs.real + lhs.imag * rhs.imag) / denominator;
  value.imag = (lhs.imag * rhs.real - lhs.real * rhs.imag) / denominator;
  return value;
}

TerlanNdArrayScalar scalar_add(TerlanNdArrayScalar lhs,
                               TerlanNdArrayScalar rhs) {
  const TerlanNdArrayScalar value = {lhs.real + rhs.real,
                                     lhs.imag + rhs.imag};
  return value;
}

TerlanNdArrayScalar scalar_subtract(TerlanNdArrayScalar lhs,
                                    TerlanNdArrayScalar rhs) {
  const TerlanNdArrayScalar value = {lhs.real - rhs.real,
                                     lhs.imag - rhs.imag};
  return value;
}

TerlanNdArrayScalar scalar_scale(TerlanNdArrayScalar value,
                                 long double scale) {
  const TerlanNdArrayScalar output = {value.real * scale,
                                      value.imag * scale};
  return output;
}

TerlanNdArrayStatus array_load_int64(TerlanNdArrayHandle array,
                                     int64_t logical_index,
                                     int64_t *out_value) {
  TerlanNdArrayScalar value;
  TerlanNdArrayStatus status = require_live(array);
  if (status != TERLAN_NDARRAY_OK || out_value == NULL ||
      logical_index < 0 || logical_index >= array->numel ||
      !dtype_is_integer(array->dtype)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  status = dtype_load(array_const_element_pointer(array, logical_index),
                      array->dtype, &value);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (value.imag != 0.0L || value.real < (long double)INT64_MIN ||
      value.real > (long double)INT64_MAX) {
    return TERLAN_NDARRAY_CAST_OVERFLOW;
  }
  *out_value = (int64_t)value.real;
  return TERLAN_NDARRAY_OK;
}

float dtype_half_to_float(uint16_t half) {
  const uint32_t sign = (uint32_t)(half & 0x8000U) << 16;
  uint32_t exponent = (half >> 10) & 0x1fU;
  uint32_t fraction = half & 0x03ffU;
  uint32_t bits;
  float value;
  if (exponent == 0) {
    if (fraction == 0) {
      bits = sign;
    } else {
      exponent = 127U - 15U + 1U;
      while ((fraction & 0x0400U) == 0) {
        fraction <<= 1;
        --exponent;
      }
      fraction &= 0x03ffU;
      bits = sign | (exponent << 23) | (fraction << 13);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (fraction << 13);
  } else {
    bits = sign | ((exponent + (127U - 15U)) << 23) | (fraction << 13);
  }
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static uint64_t round_shift_to_even(uint64_t value, uint32_t shift) {
  const uint64_t truncated = value >> shift;
  const uint64_t mask = (UINT64_C(1) << shift) - 1U;
  const uint64_t remainder = value & mask;
  const uint64_t halfway = UINT64_C(1) << (shift - 1U);
  return truncated +
         (remainder > halfway ||
          (remainder == halfway && (truncated & UINT64_C(1)) != 0));
}

uint16_t dtype_double_to_half(double value) {
  uint64_t bits;
  uint16_t sign;
  uint32_t exponent_bits;
  uint64_t fraction;
  int32_t exponent;
  uint64_t rounded;
  memcpy(&bits, &value, sizeof(bits));
  sign = (uint16_t)((bits >> 48) & UINT64_C(0x8000));
  exponent_bits = (uint32_t)((bits >> 52) & UINT64_C(0x7ff));
  fraction = bits & UINT64_C(0x000fffffffffffff);
  if (exponent_bits == 0x7ffU) {
    const uint16_t payload = (uint16_t)(fraction >> 42);
    return (uint16_t)(sign | UINT16_C(0x7c00) |
                      (fraction == 0 ? 0 : (payload == 0 ? 1 : payload)));
  }
  if (exponent_bits == 0) {
    return sign;
  }
  exponent = (int32_t)exponent_bits - 1023;
  if (exponent > 15) {
    return (uint16_t)(sign | UINT16_C(0x7c00));
  }
  if (exponent < -25) {
    return sign;
  }
  if (exponent < -14) {
    const uint64_t significand = fraction | (UINT64_C(1) << 52);
    rounded = round_shift_to_even(significand, (uint32_t)(28 - exponent));
    return (uint16_t)(sign | (uint16_t)rounded);
  }
  rounded = round_shift_to_even(fraction, 42);
  if (rounded == UINT64_C(0x400)) {
    rounded = ((uint64_t)(exponent + 16) << 10);
  } else {
    rounded |= (uint64_t)(exponent + 15) << 10;
  }
  return (uint16_t)(sign |
                    (rounded >= UINT64_C(0x7c00) ? UINT64_C(0x7c00)
                                                  : rounded));
}

uint16_t dtype_float_to_half(float value) {
  uint32_t bits;
  uint32_t fraction;
  memcpy(&bits, &value, sizeof(bits));
  fraction = bits & 0x7fffffU;
  if (((bits >> 23) & 0xffU) == 0xffU) {
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000U);
    const uint16_t payload = (uint16_t)(fraction >> 13);
    return (uint16_t)(sign | UINT16_C(0x7c00) |
                      (fraction == 0 ? 0 : (payload == 0 ? 1 : payload)));
  }
  return dtype_double_to_half((double)value);
}

double dtype_half_to_double(uint16_t half) {
  const uint16_t exponent = (uint16_t)((half >> 10) & 0x1fU);
  const uint16_t fraction = (uint16_t)(half & 0x03ffU);
  if (exponent == 0x1fU) {
    const uint64_t bits = ((uint64_t)(half & 0x8000U) << 48) |
                          (UINT64_C(0x7ff) << 52) |
                          ((uint64_t)fraction << 42);
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
  }
  return (double)dtype_half_to_float(half);
}

TerlanNdArrayStatus dtype_load(const void *pointer, int32_t dtype,
                               TerlanNdArrayScalar *out_value) {
  if (pointer == NULL || out_value == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  out_value->imag = 0.0L;
  switch (dtype) {
  case TERLAN_NDARRAY_DTYPE_BOOL:
    out_value->real = *(const uint8_t *)pointer != 0 ? 1.0L : 0.0L;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_INT8:
    out_value->real = *(const int8_t *)pointer;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_INT16:
    out_value->real = *(const int16_t *)pointer;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_INT32:
    out_value->real = *(const int32_t *)pointer;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_INT64:
    out_value->real = *(const int64_t *)pointer;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_UINT8:
    out_value->real = *(const uint8_t *)pointer;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_UINT16:
    out_value->real = *(const uint16_t *)pointer;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_UINT32:
    out_value->real = *(const uint32_t *)pointer;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_UINT64:
    out_value->real = *(const uint64_t *)pointer;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_FLOAT16:
    out_value->real = dtype_half_to_double(*(const uint16_t *)pointer);
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_FLOAT32:
    out_value->real = *(const float *)pointer;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_FLOAT64:
    out_value->real = *(const double *)pointer;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_COMPLEX64: {
    const TerlanComplex32 *value = pointer;
    out_value->real = value->real;
    out_value->imag = value->imag;
    return TERLAN_NDARRAY_OK;
  }
  case TERLAN_NDARRAY_DTYPE_COMPLEX128: {
    const TerlanComplex64 *value = pointer;
    out_value->real = value->real;
    out_value->imag = value->imag;
    return TERLAN_NDARRAY_OK;
  }
  default:
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
}

static int scalar_is_integral(TerlanNdArrayScalar value) {
  return value.imag == 0.0L && isfinite(value.real) &&
         truncl(value.real) == value.real;
}

#define STORE_INTEGER(c_type, minimum, maximum)                               \
  do {                                                                         \
    if (!scalar_is_integral(value) || value.real < (long double)(minimum) ||   \
        value.real > (long double)(maximum)) {                                 \
      return TERLAN_NDARRAY_CAST_OVERFLOW;                                     \
    }                                                                          \
    *(c_type *)pointer = (c_type)value.real;                                    \
    return TERLAN_NDARRAY_OK;                                                   \
  } while (0)

TerlanNdArrayStatus dtype_store(void *pointer, int32_t dtype,
                                TerlanNdArrayScalar value) {
  if (pointer == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  switch (dtype) {
  case TERLAN_NDARRAY_DTYPE_BOOL:
    if (value.imag != 0.0L || isnan(value.real)) {
      return TERLAN_NDARRAY_CAST_OVERFLOW;
    }
    *(uint8_t *)pointer = value.real != 0.0L ? 1 : 0;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_INT8:
    STORE_INTEGER(int8_t, INT8_MIN, INT8_MAX);
  case TERLAN_NDARRAY_DTYPE_INT16:
    STORE_INTEGER(int16_t, INT16_MIN, INT16_MAX);
  case TERLAN_NDARRAY_DTYPE_INT32:
    STORE_INTEGER(int32_t, INT32_MIN, INT32_MAX);
  case TERLAN_NDARRAY_DTYPE_INT64:
    STORE_INTEGER(int64_t, INT64_MIN, INT64_MAX);
  case TERLAN_NDARRAY_DTYPE_UINT8:
    STORE_INTEGER(uint8_t, 0, UINT8_MAX);
  case TERLAN_NDARRAY_DTYPE_UINT16:
    STORE_INTEGER(uint16_t, 0, UINT16_MAX);
  case TERLAN_NDARRAY_DTYPE_UINT32:
    STORE_INTEGER(uint32_t, 0, UINT32_MAX);
  case TERLAN_NDARRAY_DTYPE_UINT64:
    STORE_INTEGER(uint64_t, 0, UINT64_MAX);
  case TERLAN_NDARRAY_DTYPE_FLOAT16:
    if (value.imag != 0.0L) {
      return TERLAN_NDARRAY_CAST_OVERFLOW;
    }
    *(uint16_t *)pointer = dtype_double_to_half((double)value.real);
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_FLOAT32:
    if (value.imag != 0.0L ||
        (isfinite(value.real) && fabsl(value.real) > FLT_MAX)) {
      return TERLAN_NDARRAY_CAST_OVERFLOW;
    }
    *(float *)pointer = (float)value.real;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_FLOAT64:
    if (value.imag != 0.0L ||
        (isfinite(value.real) && fabsl(value.real) > DBL_MAX)) {
      return TERLAN_NDARRAY_CAST_OVERFLOW;
    }
    *(double *)pointer = (double)value.real;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_COMPLEX64: {
    TerlanComplex32 *output = pointer;
    if ((isfinite(value.real) && fabsl(value.real) > FLT_MAX) ||
        (isfinite(value.imag) && fabsl(value.imag) > FLT_MAX)) {
      return TERLAN_NDARRAY_CAST_OVERFLOW;
    }
    output->real = (float)value.real;
    output->imag = (float)value.imag;
    return TERLAN_NDARRAY_OK;
  }
  case TERLAN_NDARRAY_DTYPE_COMPLEX128: {
    TerlanComplex64 *output = pointer;
    if ((isfinite(value.real) && fabsl(value.real) > DBL_MAX) ||
        (isfinite(value.imag) && fabsl(value.imag) > DBL_MAX)) {
      return TERLAN_NDARRAY_CAST_OVERFLOW;
    }
    output->real = (double)value.real;
    output->imag = (double)value.imag;
    return TERLAN_NDARRAY_OK;
  }
  default:
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
}

static long double wrapped_unsigned(long double value, int bits) {
  const long double modulus = ldexpl(1.0L, bits);
  long double wrapped;
  if (!isfinite(value)) {
    return 0.0L;
  }
  wrapped = fmodl(truncl(value), modulus);
  return wrapped < 0.0L ? wrapped + modulus : wrapped;
}

static long double wrapped_signed(long double value, int bits) {
  const long double modulus = ldexpl(1.0L, bits);
  const long double boundary = ldexpl(1.0L, bits - 1);
  const long double wrapped = wrapped_unsigned(value, bits);
  return wrapped >= boundary ? wrapped - modulus : wrapped;
}

static TerlanNdArrayStatus dtype_store_unsafe(void *pointer, int32_t dtype,
                                              TerlanNdArrayScalar value) {
  if (pointer == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  switch (dtype) {
  case TERLAN_NDARRAY_DTYPE_BOOL:
    *(uint8_t *)pointer = value.real != 0.0L || value.imag != 0.0L ? 1 : 0;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_INT8:
    *(int8_t *)pointer = (int8_t)wrapped_signed(value.real, 8);
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_INT16:
    *(int16_t *)pointer = (int16_t)wrapped_signed(value.real, 16);
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_INT32:
    *(int32_t *)pointer = (int32_t)wrapped_signed(value.real, 32);
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_INT64:
    *(int64_t *)pointer = (int64_t)wrapped_signed(value.real, 64);
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_UINT8:
    *(uint8_t *)pointer = (uint8_t)wrapped_unsigned(value.real, 8);
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_UINT16:
    *(uint16_t *)pointer = (uint16_t)wrapped_unsigned(value.real, 16);
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_UINT32:
    *(uint32_t *)pointer = (uint32_t)wrapped_unsigned(value.real, 32);
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_UINT64:
    *(uint64_t *)pointer = (uint64_t)wrapped_unsigned(value.real, 64);
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_FLOAT16:
    *(uint16_t *)pointer = dtype_double_to_half((double)value.real);
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_FLOAT32:
    *(float *)pointer = (float)value.real;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_FLOAT64:
    *(double *)pointer = (double)value.real;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_COMPLEX64: {
    TerlanComplex32 *output = pointer;
    output->real = (float)value.real;
    output->imag = (float)value.imag;
    return TERLAN_NDARRAY_OK;
  }
  case TERLAN_NDARRAY_DTYPE_COMPLEX128: {
    TerlanComplex64 *output = pointer;
    output->real = (double)value.real;
    output->imag = (double)value.imag;
    return TERLAN_NDARRAY_OK;
  }
  default:
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
}

TerlanNdArrayStatus dtype_store_cast(void *pointer, int32_t dtype,
                                     TerlanNdArrayScalar value,
                                     int32_t casting) {
  if (casting == TERLAN_NDARRAY_CAST_UNSAFE ||
      casting == TERLAN_NDARRAY_CAST_SAME_KIND) {
    return dtype_store_unsafe(pointer, dtype, value);
  }
  return dtype_store(pointer, dtype, value);
}

TerlanNdArrayStatus dtype_convert(void *destination, int32_t destination_dtype,
                                  const void *source, int32_t source_dtype,
                                  int32_t casting) {
  TerlanNdArrayScalar value;
  if (destination == NULL || source == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (source_dtype == TERLAN_NDARRAY_DTYPE_FLOAT16 &&
      destination_dtype == TERLAN_NDARRAY_DTYPE_FLOAT32) {
    *(float *)destination = dtype_half_to_float(*(const uint16_t *)source);
    return TERLAN_NDARRAY_OK;
  }
  if (source_dtype == TERLAN_NDARRAY_DTYPE_FLOAT16 &&
      destination_dtype == TERLAN_NDARRAY_DTYPE_FLOAT64) {
    *(double *)destination = dtype_half_to_double(*(const uint16_t *)source);
    return TERLAN_NDARRAY_OK;
  }
  if (source_dtype == TERLAN_NDARRAY_DTYPE_FLOAT32 &&
      destination_dtype == TERLAN_NDARRAY_DTYPE_FLOAT16) {
    *(uint16_t *)destination = dtype_float_to_half(*(const float *)source);
    return TERLAN_NDARRAY_OK;
  }
  if (source_dtype == TERLAN_NDARRAY_DTYPE_FLOAT64 &&
      destination_dtype == TERLAN_NDARRAY_DTYPE_FLOAT16) {
    *(uint16_t *)destination = dtype_double_to_half(*(const double *)source);
    return TERLAN_NDARRAY_OK;
  }
  if (dtype_load(source, source_dtype, &value) != TERLAN_NDARRAY_OK) {
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  return dtype_store_cast(destination, destination_dtype, value, casting);
}

static TerlanNdArrayStatus create_from_scalars(
    const double *real_values, const double *imag_values, int64_t value_count,
    const int64_t *shape, int64_t rank, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  const size_t element_size = dtype_size(dtype);
  void *storage = NULL;
  int64_t index;
  TerlanNdArrayStatus status;
  if (out_array == NULL || value_count < 0 || element_size == 0 ||
      (value_count > 0 && real_values == NULL) ||
      (uint64_t)value_count > SIZE_MAX / element_size) {
    return element_size == 0 ? TERLAN_NDARRAY_UNSUPPORTED_DTYPE
                             : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (value_count > 0) {
    storage = checked_malloc((size_t)value_count * element_size);
    if (storage == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (index = 0; index < value_count; ++index) {
    TerlanNdArrayScalar scalar = {real_values[index],
                                  imag_values == NULL ? 0.0L
                                                      : imag_values[index]};
    status = dtype_store((uint8_t *)storage + (size_t)index * element_size,
                         dtype, scalar);
    if (status != TERLAN_NDARRAY_OK) {
      free(storage);
      return status;
    }
  }
  status = create_array(storage, value_count, shape, rank, element_size, dtype,
                        out_array);
  free(storage);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_from_ints_as(
    const int64_t *values, int64_t value_count, const int64_t *shape,
    int64_t rank, int32_t dtype, TerlanNdArrayHandle *out_array) {
  const size_t element_size = dtype_size(dtype);
  void *storage = NULL;
  int64_t index;
  TerlanNdArrayStatus status;
  if (value_count < 0 || (value_count > 0 && values == NULL) ||
      element_size == 0 || (uint64_t)value_count > SIZE_MAX / element_size) {
    return element_size == 0 ? TERLAN_NDARRAY_UNSUPPORTED_DTYPE
                             : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (value_count > 0) {
    storage = checked_malloc((size_t)value_count * element_size);
    if (storage == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    for (index = 0; index < value_count; ++index) {
      TerlanNdArrayScalar scalar = {(long double)values[index], 0.0L};
      status = dtype_store((uint8_t *)storage + (size_t)index * element_size,
                           dtype, scalar);
      if (status != TERLAN_NDARRAY_OK) {
        free(storage);
        return status;
      }
    }
  }
  status = create_array(storage, value_count, shape, rank, element_size, dtype,
                        out_array);
  free(storage);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_from_uints(
    const uint64_t *values, int64_t value_count, const int64_t *shape,
    int64_t rank, TerlanNdArrayHandle *out_array) {
  if (value_count < 0 || (value_count > 0 && values == NULL)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  return create_array(values, value_count, shape, rank, sizeof(*values),
                      TERLAN_NDARRAY_DTYPE_UINT64, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_from_floats_as(
    const double *values, int64_t value_count, const int64_t *shape,
    int64_t rank, int32_t dtype, TerlanNdArrayHandle *out_array) {
  return create_from_scalars(values, NULL, value_count, shape, rank, dtype,
                             out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_from_complex(
    const double *real_values, int64_t real_count, const double *imag_values,
    int64_t imag_count, const int64_t *shape, int64_t rank, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  if (real_count != imag_count || !dtype_is_complex(dtype)) {
    return real_count != imag_count ? TERLAN_NDARRAY_ELEMENT_COUNT_MISMATCH
                                    : TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  return create_from_scalars(real_values, imag_values, real_count, shape, rank,
                             dtype, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_astype(
    TerlanNdArrayHandle array, int32_t dtype, TerlanNdArrayHandle *out_array) {
  return copy_array_ordered(array, dtype, TERLAN_NDARRAY_ORDER_KEEP,
                            TERLAN_NDARRAY_CAST_UNSAFE, out_array);
}

static int order_is_compatible(TerlanNdArrayHandle array, int32_t order) {
  if (order == TERLAN_NDARRAY_ORDER_KEEP) {
    return 1;
  }
  if (order == TERLAN_NDARRAY_ORDER_ANY) {
    order = array_is_fortran_contiguous(array) && !array_is_contiguous(array)
                ? TERLAN_NDARRAY_ORDER_FORTRAN
                : TERLAN_NDARRAY_ORDER_C;
  }
  if (order == TERLAN_NDARRAY_ORDER_C) {
    return array_is_contiguous(array);
  }
  if (order == TERLAN_NDARRAY_ORDER_FORTRAN) {
    return array_is_fortran_contiguous(array);
  }
  return 0;
}

TerlanNdArrayStatus terlan_ndarray_v1_astype_ordered(
    TerlanNdArrayHandle array, int32_t dtype, int32_t order, int32_t copy,
    int32_t casting, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  if (status != TERLAN_NDARRAY_OK || out_array == NULL ||
      (copy != 0 && copy != 1) || order < TERLAN_NDARRAY_ORDER_C ||
      order > TERLAN_NDARRAY_ORDER_ANY ||
      casting < TERLAN_NDARRAY_CAST_NO ||
      casting > TERLAN_NDARRAY_CAST_UNSAFE) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (!dtype_can_cast(array->dtype, dtype, casting)) {
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  if (copy == 0 && array->dtype == dtype && order_is_compatible(array, order)) {
    return create_view(array, array->shape, array->strides, array->rank, 0,
                       array->writable, out_array);
  }
  return copy_array_ordered(array, dtype, order, casting, out_array);
}

static TerlanNdArrayStatus read_component(TerlanNdArrayHandle array,
                                           int component, double **out_values,
                                           size_t *out_length) {
  TerlanNdArrayStatus status = require_live(array);
  double *values = NULL;
  int64_t index;
  if (status != TERLAN_NDARRAY_OK || out_values == NULL ||
      out_length == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_values = NULL;
  *out_length = 0;
  if (array->numel > 0) {
    values = checked_malloc((size_t)array->numel * sizeof(*values));
    if (values == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (index = 0; index < array->numel; ++index) {
    TerlanNdArrayScalar scalar;
    status = dtype_load(array_const_element_pointer(array, index), array->dtype,
                        &scalar);
    if (status != TERLAN_NDARRAY_OK) {
      free(values);
      return status;
    }
    values[index] = (double)(component == 0 ? scalar.real : scalar.imag);
  }
  *out_values = values;
  *out_length = (size_t)array->numel;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_to_floats_cast(
    TerlanNdArrayHandle array, double **out_values, size_t *out_length) {
  if (array != NULL && dtype_is_complex(array->dtype)) {
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  return read_component(array, 0, out_values, out_length);
}

TerlanNdArrayStatus terlan_ndarray_v1_complex_reals(
    TerlanNdArrayHandle array, double **out_values, size_t *out_length) {
  if (array != NULL && !dtype_is_complex(array->dtype)) {
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  return read_component(array, 0, out_values, out_length);
}

TerlanNdArrayStatus terlan_ndarray_v1_complex_imags(
    TerlanNdArrayHandle array, double **out_values, size_t *out_length) {
  if (array != NULL && !dtype_is_complex(array->dtype)) {
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  return read_component(array, 1, out_values, out_length);
}

TerlanNdArrayStatus terlan_ndarray_v1_to_ints_cast(
    TerlanNdArrayHandle array, int64_t **out_values, size_t *out_length) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t *values = NULL;
  int64_t index;
  if (status != TERLAN_NDARRAY_OK || out_values == NULL ||
      out_length == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_values = NULL;
  *out_length = 0;
  if (array->numel > 0) {
    values = checked_malloc((size_t)array->numel * sizeof(*values));
    if (values == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (index = 0; index < array->numel; ++index) {
    TerlanNdArrayScalar scalar;
    status = dtype_load(array_const_element_pointer(array, index), array->dtype,
                        &scalar);
    if (status == TERLAN_NDARRAY_OK && scalar_is_integral(scalar) &&
        scalar.real >= (long double)INT64_MIN &&
        scalar.real <= (long double)INT64_MAX) {
      values[index] = (int64_t)scalar.real;
    } else {
      free(values);
      return status != TERLAN_NDARRAY_OK ? status
                                         : TERLAN_NDARRAY_CAST_OVERFLOW;
    }
  }
  *out_values = values;
  *out_length = (size_t)array->numel;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_to_uints_cast(
    TerlanNdArrayHandle array, uint64_t **out_values, size_t *out_length) {
  TerlanNdArrayStatus status = require_live(array);
  uint64_t *values = NULL;
  int64_t index;
  if (status != TERLAN_NDARRAY_OK || out_values == NULL ||
      out_length == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_values = NULL;
  *out_length = 0;
  if (array->numel > 0) {
    values = checked_malloc((size_t)array->numel * sizeof(*values));
    if (values == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (index = 0; index < array->numel; ++index) {
    const void *element = array_const_element_pointer(array, index);
    if (array->dtype == TERLAN_NDARRAY_DTYPE_UINT64) {
      values[index] = *(const uint64_t *)element;
    } else {
      TerlanNdArrayScalar scalar;
      status = dtype_load(element, array->dtype, &scalar);
      if (status == TERLAN_NDARRAY_OK && scalar_is_integral(scalar) &&
          scalar.real >= 0.0L && scalar.real <= (long double)UINT64_MAX) {
        values[index] = (uint64_t)scalar.real;
      } else {
        free(values);
        return status != TERLAN_NDARRAY_OK ? status
                                           : TERLAN_NDARRAY_CAST_OVERFLOW;
      }
    }
  }
  *out_values = values;
  *out_length = (size_t)array->numel;
  return TERLAN_NDARRAY_OK;
}

void terlan_ndarray_v1_delete_uint_values(uint64_t *values) { free(values); }
