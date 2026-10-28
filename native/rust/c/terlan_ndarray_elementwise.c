#include "terlan_ndarray_internal.h"

#include <complex.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct BroadcastPlan {
  int64_t rank;
  int64_t numel;
  int64_t *shape;
  int64_t *lhs_strides;
  int64_t *rhs_strides;
} BroadcastPlan;

typedef enum BinaryOperation {
  BINARY_ADD,
  BINARY_SUBTRACT,
  BINARY_MULTIPLY,
  BINARY_TRUE_DIVIDE,
  BINARY_FLOOR_DIVIDE,
  BINARY_REMAINDER,
  BINARY_POWER,
  BINARY_MINIMUM,
  BINARY_MAXIMUM,
  BINARY_FMIN,
  BINARY_FMAX,
  BINARY_EQUAL,
  BINARY_NOT_EQUAL,
  BINARY_LESS,
  BINARY_LESS_EQUAL,
  BINARY_GREATER,
  BINARY_GREATER_EQUAL,
  BINARY_LOGICAL_AND,
  BINARY_LOGICAL_OR,
  BINARY_LOGICAL_XOR,
  BINARY_BITWISE_AND,
  BINARY_BITWISE_OR,
  BINARY_BITWISE_XOR,
  BINARY_LEFT_SHIFT,
  BINARY_RIGHT_SHIFT,
  BINARY_GCD,
  BINARY_LCM,
  BINARY_HYPOT,
  BINARY_ARCTAN2,
  BINARY_LOGADDEXP,
  BINARY_LOGADDEXP2,
  BINARY_COPYSIGN,
  BINARY_NEXTAFTER
} BinaryOperation;

static void destroy_plan(BroadcastPlan *plan) {
  if (plan != NULL) {
    free(plan->rhs_strides);
    free(plan->lhs_strides);
    free(plan->shape);
    plan->rhs_strides = NULL;
    plan->lhs_strides = NULL;
    plan->shape = NULL;
  }
}

static TerlanNdArrayStatus broadcast_dimension(int64_t lhs, int64_t rhs,
                                               int64_t *out_size) {
  if (out_size == NULL || lhs < 0 || rhs < 0) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (lhs == rhs) {
    *out_size = lhs;
  } else if (lhs == 1) {
    *out_size = rhs;
  } else if (rhs == 1) {
    *out_size = lhs;
  } else {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus create_plan(TerlanNdArrayHandle lhs,
                                        TerlanNdArrayHandle rhs,
                                        BroadcastPlan *out_plan) {
  int64_t output_axis;
  if (out_plan == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  out_plan->rank = lhs->rank > rhs->rank ? lhs->rank : rhs->rank;
  out_plan->numel = 1;
  out_plan->shape = NULL;
  out_plan->lhs_strides = NULL;
  out_plan->rhs_strides = NULL;
  if (out_plan->rank > 0) {
    const size_t size = (size_t)out_plan->rank * sizeof(int64_t);
    out_plan->shape = checked_malloc(size);
    out_plan->lhs_strides = checked_malloc(size);
    out_plan->rhs_strides = checked_malloc(size);
    if (out_plan->shape == NULL || out_plan->lhs_strides == NULL ||
        out_plan->rhs_strides == NULL) {
      destroy_plan(out_plan);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (output_axis = 0; output_axis < out_plan->rank; ++output_axis) {
    const int64_t lhs_axis = output_axis - (out_plan->rank - lhs->rank);
    const int64_t rhs_axis = output_axis - (out_plan->rank - rhs->rank);
    const int64_t lhs_size = lhs_axis < 0 ? 1 : lhs->shape[lhs_axis];
    const int64_t rhs_size = rhs_axis < 0 ? 1 : rhs->shape[rhs_axis];
    int64_t output_size;
    if (broadcast_dimension(lhs_size, rhs_size, &output_size) !=
        TERLAN_NDARRAY_OK) {
      destroy_plan(out_plan);
      return TERLAN_NDARRAY_SHAPE_MISMATCH;
    }
    out_plan->shape[output_axis] = output_size;
    out_plan->lhs_strides[output_axis] =
        lhs_axis < 0 || lhs_size == 1 ? 0 : lhs->strides[lhs_axis];
    out_plan->rhs_strides[output_axis] =
        rhs_axis < 0 || rhs_size == 1 ? 0 : rhs->strides[rhs_axis];
    if (output_size == 0) {
      out_plan->numel = 0;
    } else if (out_plan->numel != 0) {
      if (out_plan->numel > INT64_MAX / output_size) {
        destroy_plan(out_plan);
        return TERLAN_NDARRAY_SIZE_OVERFLOW;
      }
      out_plan->numel *= output_size;
    }
  }
  return TERLAN_NDARRAY_OK;
}

static void plan_offsets(const BroadcastPlan *plan, int64_t logical_index,
                         int64_t *lhs_offset, int64_t *rhs_offset) {
  int64_t axis;
  *lhs_offset = 0;
  *rhs_offset = 0;
  for (axis = plan->rank; axis > 0; --axis) {
    const int64_t index = axis - 1;
    const int64_t coordinate = logical_index % plan->shape[index];
    logical_index /= plan->shape[index];
    *lhs_offset += coordinate * plan->lhs_strides[index];
    *rhs_offset += coordinate * plan->rhs_strides[index];
  }
}

static TerlanNdArrayStatus create_result(const BroadcastPlan *plan,
                                          int32_t dtype,
                                          TerlanNdArrayHandle *out_array) {
  return create_empty_array(plan->shape, plan->rank, dtype, out_array);
}

static int scalar_truth(TerlanNdArrayScalar value) {
  return value.real != 0.0L || value.imag != 0.0L;
}

static int scalar_has_nan(TerlanNdArrayScalar value) {
  return isnan(value.real) || isnan(value.imag);
}

static int scalar_less(TerlanNdArrayScalar lhs, TerlanNdArrayScalar rhs) {
  return lhs.real < rhs.real ||
         (lhs.real == rhs.real && lhs.imag < rhs.imag);
}

static TerlanNdArrayStatus calculate(BinaryOperation operation,
                                      TerlanNdArrayScalar lhs,
                                      TerlanNdArrayScalar rhs,
                                      int inputs_complex, int32_t output_dtype,
                                      TerlanNdArrayScalar *out_value) {
  const long double complex left = CMPLXL(lhs.real, lhs.imag);
  const long double complex right = CMPLXL(rhs.real, rhs.imag);
  long double complex result = 0.0L;
  out_value->real = 0.0L;
  out_value->imag = 0.0L;
  if (operation >= BINARY_BITWISE_AND && operation <= BINARY_LCM) {
    uint64_t left_integer;
    uint64_t right_integer;
    uint64_t integer_result;
    const int output_signed = dtype_is_signed(output_dtype);
    if (inputs_complex || (!dtype_is_integer(output_dtype) &&
                           !dtype_is_bool(output_dtype))) {
      return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
    }
    left_integer = output_signed ? (uint64_t)(int64_t)lhs.real
                                 : (uint64_t)lhs.real;
    right_integer = output_signed ? (uint64_t)(int64_t)rhs.real
                                  : (uint64_t)rhs.real;
    if (operation == BINARY_BITWISE_AND) {
      integer_result = left_integer & right_integer;
    } else if (operation == BINARY_BITWISE_OR) {
      integer_result = left_integer | right_integer;
    } else if (operation == BINARY_BITWISE_XOR) {
      integer_result = left_integer ^ right_integer;
    } else if (operation == BINARY_LEFT_SHIFT ||
               operation == BINARY_RIGHT_SHIFT) {
      const uint32_t bits = dtype_integer_bits(output_dtype);
      if (rhs.real < 0.0L) {
        return TERLAN_NDARRAY_DOMAIN_ERROR;
      }
      if (rhs.real >= (long double)bits) {
        integer_result = operation == BINARY_RIGHT_SHIFT && output_signed &&
                                 lhs.real < 0.0L
                             ? UINT64_MAX
                             : UINT64_C(0);
      } else {
        const uint32_t shift = (uint32_t)rhs.real;
        integer_result = operation == BINARY_LEFT_SHIFT
                             ? left_integer << shift
                             : output_signed
                                   ? (uint64_t)((int64_t)left_integer >> shift)
                                   : left_integer >> shift;
      }
      integer_result = dtype_truncate_integer(integer_result, output_dtype);
    } else {
      const uint64_t lhs_magnitude =
          output_signed ? scalar_integer_magnitude(lhs.real) : left_integer;
      const uint64_t rhs_magnitude =
          output_signed ? scalar_integer_magnitude(rhs.real) : right_integer;
      const uint64_t divisor = scalar_unsigned_gcd(lhs_magnitude, rhs_magnitude);
      if (operation == BINARY_GCD) {
        integer_result = divisor;
      } else if (divisor == 0) {
        integer_result = 0;
      } else if (lhs_magnitude / divisor > UINT64_MAX / rhs_magnitude) {
        return TERLAN_NDARRAY_ARITHMETIC_OVERFLOW;
      } else {
        integer_result = lhs_magnitude / divisor * rhs_magnitude;
      }
      if (output_signed && integer_result > INT64_MAX) {
        return TERLAN_NDARRAY_ARITHMETIC_OVERFLOW;
      }
    }
    out_value->real = output_signed
                          ? (long double)dtype_signed_integer(integer_result,
                                                              output_dtype)
                          : (long double)integer_result;
    return TERLAN_NDARRAY_OK;
  }
  if (operation >= BINARY_HYPOT) {
    if (inputs_complex) {
      return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
    }
    if (operation == BINARY_HYPOT) {
      out_value->real = hypotl(lhs.real, rhs.real);
    } else if (operation == BINARY_ARCTAN2) {
      out_value->real = atan2l(lhs.real, rhs.real);
    } else if (operation == BINARY_LOGADDEXP) {
      const long double maximum = fmaxl(lhs.real, rhs.real);
      out_value->real = isnan(lhs.real) || isnan(rhs.real) ? NAN
                            : isinf(maximum) ? maximum
                                             : maximum + log1pl(expl(-fabsl(lhs.real - rhs.real)));
    } else if (operation == BINARY_LOGADDEXP2) {
      const long double maximum = fmaxl(lhs.real, rhs.real);
      out_value->real = isnan(lhs.real) || isnan(rhs.real) ? NAN
                            : isinf(maximum) ? maximum
                                             : maximum + log2l(1.0L + exp2l(-fabsl(lhs.real - rhs.real)));
    } else if (operation == BINARY_COPYSIGN) {
      out_value->real = copysignl(lhs.real, rhs.real);
    } else if (output_dtype == TERLAN_NDARRAY_DTYPE_FLOAT16) {
      uint16_t bits = dtype_double_to_half((double)lhs.real);
      if (isnan(lhs.real) || isnan(rhs.real)) {
        out_value->real = NAN;
      } else if (lhs.real == rhs.real) {
        out_value->real = rhs.real;
      } else if (lhs.real == 0.0L) {
        bits = signbit(rhs.real) ? UINT16_C(0x8001) : UINT16_C(0x0001);
        out_value->real = dtype_half_to_double(bits);
      } else {
        if (lhs.real < rhs.real) {
          bits = signbit(lhs.real) ? (uint16_t)(bits - 1U)
                                   : (uint16_t)(bits + 1U);
        } else {
          bits = signbit(lhs.real) ? (uint16_t)(bits + 1U)
                                   : (uint16_t)(bits - 1U);
        }
        out_value->real = dtype_half_to_double(bits);
      }
    } else if (output_dtype == TERLAN_NDARRAY_DTYPE_FLOAT32) {
      out_value->real = nextafterf((float)lhs.real, (float)rhs.real);
    } else if (output_dtype == TERLAN_NDARRAY_DTYPE_FLOAT64) {
      out_value->real = nextafter((double)lhs.real, (double)rhs.real);
    } else {
      out_value->real = nextafterl(lhs.real, rhs.real);
    }
    return TERLAN_NDARRAY_OK;
  }
  if (!inputs_complex) {
    if (dtype_is_integer(output_dtype) &&
        (operation == BINARY_ADD || operation == BINARY_SUBTRACT ||
         operation == BINARY_MULTIPLY || operation == BINARY_POWER)) {
      const TerlanNdArrayIntegerBinaryOperation integer_operation =
          operation == BINARY_ADD
              ? TERLAN_NDARRAY_INTEGER_ADD
              : operation == BINARY_SUBTRACT
                    ? TERLAN_NDARRAY_INTEGER_SUBTRACT
                    : operation == BINARY_MULTIPLY
                          ? TERLAN_NDARRAY_INTEGER_MULTIPLY
                          : TERLAN_NDARRAY_INTEGER_POWER;
      return scalar_wrapping_integer_binary(integer_operation, lhs.real,
                                             rhs.real, output_dtype, out_value);
    }
    switch (operation) {
    case BINARY_ADD:
      out_value->real = lhs.real + rhs.real;
      return TERLAN_NDARRAY_OK;
    case BINARY_SUBTRACT:
      out_value->real = lhs.real - rhs.real;
      return TERLAN_NDARRAY_OK;
    case BINARY_MULTIPLY:
      out_value->real = lhs.real * rhs.real;
      return TERLAN_NDARRAY_OK;
    case BINARY_TRUE_DIVIDE:
      if (rhs.real == 0.0L) {
        return TERLAN_NDARRAY_DIVIDE_BY_ZERO;
      }
      out_value->real = lhs.real / rhs.real;
      return TERLAN_NDARRAY_OK;
    case BINARY_POWER:
      if ((dtype_is_integer(output_dtype) || dtype_is_bool(output_dtype)) &&
          rhs.real < 0.0L) {
        return TERLAN_NDARRAY_DOMAIN_ERROR;
      }
      out_value->real = powl(lhs.real, rhs.real);
      return TERLAN_NDARRAY_OK;
    default:
      break;
    }
  }
  switch (operation) {
  case BINARY_ADD:
    result = left + right;
    break;
  case BINARY_SUBTRACT:
    result = left - right;
    break;
  case BINARY_MULTIPLY:
    result = left * right;
    break;
  case BINARY_TRUE_DIVIDE:
    if (rhs.real == 0.0L && rhs.imag == 0.0L) {
      if (isinf(lhs.real) || isinf(lhs.imag)) {
        *out_value = lhs;
      } else if (isnan(lhs.real) || isnan(lhs.imag) ||
                 (lhs.real == 0.0L && lhs.imag == 0.0L)) {
        out_value->real = NAN;
        out_value->imag = NAN;
      } else {
        out_value->real = copysignl(INFINITY, lhs.real);
        out_value->imag = copysignl(INFINITY, lhs.imag);
      }
      return TERLAN_NDARRAY_OK;
    }
    result = left / right;
    break;
  case BINARY_POWER:
    scalar_complex_power(lhs, rhs, out_value);
    return TERLAN_NDARRAY_OK;
  case BINARY_MINIMUM:
  case BINARY_MAXIMUM:
  case BINARY_FMIN:
  case BINARY_FMAX: {
    const int lhs_nan = scalar_has_nan(lhs);
    const int rhs_nan = scalar_has_nan(rhs);
    const int ignore_nan = operation == BINARY_FMIN || operation == BINARY_FMAX;
    const int select_minimum =
        operation == BINARY_MINIMUM || operation == BINARY_FMIN;
    if (lhs_nan || rhs_nan) {
      if (!ignore_nan || (lhs_nan && rhs_nan)) {
        *out_value = lhs_nan ? lhs : rhs;
      } else {
        *out_value = lhs_nan ? rhs : lhs;
      }
    } else {
      const int left_is_less = scalar_less(lhs, rhs);
      *out_value = select_minimum
                       ? (left_is_less ? lhs : rhs)
                       : (scalar_less(rhs, lhs) ? lhs : rhs);
    }
    return TERLAN_NDARRAY_OK;
  }
  case BINARY_FLOOR_DIVIDE:
  case BINARY_REMAINDER: {
    long double quotient;
    long double remainder;
    if (inputs_complex) {
      return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
    }
    if (rhs.real == 0.0L) {
      if (!dtype_is_float(output_dtype)) {
        return TERLAN_NDARRAY_DIVIDE_BY_ZERO;
      }
      out_value->real = operation == BINARY_FLOOR_DIVIDE
                            ? lhs.real / rhs.real
                            : fmodl(lhs.real, rhs.real);
      return TERLAN_NDARRAY_OK;
    }
    if (dtype_is_integer(output_dtype) && dtype_is_signed(output_dtype) &&
        rhs.real == -1.0L &&
        lhs.real == -ldexpl(1.0L, (int)(dtype_size(output_dtype) * 8U - 1U))) {
      out_value->real = operation == BINARY_FLOOR_DIVIDE ? lhs.real : 0.0L;
      return TERLAN_NDARRAY_OK;
    }
    scalar_floating_floor_divmod(lhs.real, rhs.real, &quotient, &remainder);
    out_value->real = operation == BINARY_FLOOR_DIVIDE ? quotient : remainder;
    return TERLAN_NDARRAY_OK;
  }
  case BINARY_EQUAL:
    out_value->real = lhs.real == rhs.real && lhs.imag == rhs.imag;
    return TERLAN_NDARRAY_OK;
  case BINARY_NOT_EQUAL:
    out_value->real = lhs.real != rhs.real || lhs.imag != rhs.imag;
    return TERLAN_NDARRAY_OK;
  case BINARY_LESS:
  case BINARY_LESS_EQUAL:
  case BINARY_GREATER:
  case BINARY_GREATER_EQUAL:
    if (inputs_complex) {
      const int equal = lhs.real == rhs.real && lhs.imag == rhs.imag;
      const int less = scalar_less(lhs, rhs);
      const int unordered = scalar_has_nan(lhs) || scalar_has_nan(rhs);
      out_value->real =
          unordered ? 0.0L
                    : operation == BINARY_LESS          ? less
                    : operation == BINARY_LESS_EQUAL    ? less || equal
                    : operation == BINARY_GREATER       ? !less && !equal
                                                        : !less;
      return TERLAN_NDARRAY_OK;
    }
    out_value->real = operation == BINARY_LESS
                          ? lhs.real < rhs.real
                          : operation == BINARY_LESS_EQUAL
                                ? lhs.real <= rhs.real
                                : operation == BINARY_GREATER
                                      ? lhs.real > rhs.real
                                      : lhs.real >= rhs.real;
    return TERLAN_NDARRAY_OK;
  case BINARY_LOGICAL_AND:
    out_value->real = scalar_truth(lhs) && scalar_truth(rhs);
    return TERLAN_NDARRAY_OK;
  case BINARY_LOGICAL_OR:
    out_value->real = scalar_truth(lhs) || scalar_truth(rhs);
    return TERLAN_NDARRAY_OK;
  case BINARY_LOGICAL_XOR:
    out_value->real = scalar_truth(lhs) != scalar_truth(rhs);
    return TERLAN_NDARRAY_OK;
  case BINARY_BITWISE_AND:
  case BINARY_BITWISE_OR:
  case BINARY_BITWISE_XOR:
  case BINARY_LEFT_SHIFT:
  case BINARY_RIGHT_SHIFT:
  case BINARY_GCD:
  case BINARY_LCM:
  case BINARY_HYPOT:
  case BINARY_ARCTAN2:
  case BINARY_LOGADDEXP:
  case BINARY_LOGADDEXP2:
  case BINARY_COPYSIGN:
  case BINARY_NEXTAFTER:
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  out_value->real = creall(result);
  out_value->imag = cimagl(result);
  return TERLAN_NDARRAY_OK;
}
static int operation_returns_bool(BinaryOperation operation) {
  return operation >= BINARY_EQUAL && operation <= BINARY_LOGICAL_XOR;
}
static int operation_requires_integer(BinaryOperation operation) {
  return operation >= BINARY_BITWISE_AND && operation <= BINARY_LCM;
}
static int operation_returns_float(BinaryOperation operation) {
  return operation >= BINARY_HYPOT;
}
static TerlanNdArrayStatus binary_operation(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs,
    BinaryOperation operation, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(lhs);
  BroadcastPlan plan;
  TerlanNdArrayHandle result = NULL;
  int32_t output_dtype;
  int64_t index;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(rhs);
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = create_plan(lhs, rhs, &plan);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  output_dtype = operation_returns_bool(operation)
                     ? TERLAN_NDARRAY_DTYPE_BOOL
                     : dtype_promote(lhs->dtype, rhs->dtype);
  if (operation == BINARY_SUBTRACT && dtype_is_bool(output_dtype)) {
    destroy_plan(&plan);
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  if (operation_requires_integer(operation) &&
      ((!dtype_is_integer(lhs->dtype) && !dtype_is_bool(lhs->dtype)) ||
       (!dtype_is_integer(rhs->dtype) && !dtype_is_bool(rhs->dtype)))) {
    destroy_plan(&plan);
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  if (operation_returns_float(operation) &&
      (dtype_is_integer(output_dtype) || dtype_is_bool(output_dtype))) {
    output_dtype = TERLAN_NDARRAY_DTYPE_FLOAT64;
  }
  if (operation == BINARY_TRUE_DIVIDE &&
      (dtype_is_bool(output_dtype) || dtype_is_integer(output_dtype))) {
    output_dtype = TERLAN_NDARRAY_DTYPE_FLOAT64;
  }
  status = create_result(&plan, output_dtype, &result);
  if (status != TERLAN_NDARRAY_OK) {
    destroy_plan(&plan);
    return status;
  }
  for (index = 0; index < plan.numel; ++index) {
    int64_t lhs_offset;
    int64_t rhs_offset;
    TerlanNdArrayScalar left;
    TerlanNdArrayScalar right;
    TerlanNdArrayScalar output;
    plan_offsets(&plan, index, &lhs_offset, &rhs_offset);
    status = dtype_load((const uint8_t *)lhs->data +
                            lhs_offset * (int64_t)dtype_size(lhs->dtype),
                        lhs->dtype, &left);
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_load((const uint8_t *)rhs->data +
                              rhs_offset * (int64_t)dtype_size(rhs->dtype),
                          rhs->dtype, &right);
    }
    if (status == TERLAN_NDARRAY_OK) {
      status = calculate(operation, left, right,
                         dtype_is_complex(lhs->dtype) ||
                             dtype_is_complex(rhs->dtype),
                         output_dtype, &output);
    }
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_store(array_element_pointer(result, index), output_dtype,
                           output);
    }
    if (status != TERLAN_NDARRAY_OK) {
      destroy_plan(&plan);
      (void)terlan_ndarray_v1_delete(result);
      return status == TERLAN_NDARRAY_CAST_OVERFLOW
                 ? TERLAN_NDARRAY_ARITHMETIC_OVERFLOW
                 : status;
    }
  }
  destroy_plan(&plan);
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}
TerlanNdArrayStatus terlan_ndarray_v1_broadcast_to(
    TerlanNdArrayHandle array, const int64_t *shape, int64_t rank,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t *strides = NULL;
  int64_t output_axis;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL || rank < array->rank ||
      (rank > 0 && shape == NULL)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (rank > 0) {
    strides = checked_malloc((size_t)rank * sizeof(*strides));
    if (strides == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (output_axis = 0; output_axis < rank; ++output_axis) {
    const int64_t source_axis = output_axis - (rank - array->rank);
    const int64_t source_size =
        source_axis < 0 ? 1 : array->shape[source_axis];
    if (shape[output_axis] < 0 ||
        (source_size != shape[output_axis] && source_size != 1)) {
      free(strides);
      return shape[output_axis] < 0 ? TERLAN_NDARRAY_NEGATIVE_DIMENSION
                                    : TERLAN_NDARRAY_SHAPE_MISMATCH;
    }
    strides[output_axis] =
        source_axis < 0 || source_size == 1 ? 0
                                             : array->strides[source_axis];
  }
  status = create_view(array, shape, strides, rank, 0, 0, out_array);
  free(strides);
  return status;
}
#define BINARY_WRAPPER(name, operation)                                       \
  TerlanNdArrayStatus name(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs,  \
                           TerlanNdArrayHandle *out_array) {                   \
    return binary_operation(lhs, rhs, operation, out_array);                  \
  }
BINARY_WRAPPER(terlan_ndarray_v1_add, BINARY_ADD)
BINARY_WRAPPER(terlan_ndarray_v1_subtract, BINARY_SUBTRACT)
BINARY_WRAPPER(terlan_ndarray_v1_multiply, BINARY_MULTIPLY)
BINARY_WRAPPER(terlan_ndarray_v1_true_divide, BINARY_TRUE_DIVIDE)
BINARY_WRAPPER(terlan_ndarray_v1_floor_divide, BINARY_FLOOR_DIVIDE)
BINARY_WRAPPER(terlan_ndarray_v1_remainder, BINARY_REMAINDER)
BINARY_WRAPPER(terlan_ndarray_v1_power, BINARY_POWER)
BINARY_WRAPPER(terlan_ndarray_v1_elementwise_minimum, BINARY_MINIMUM)
BINARY_WRAPPER(terlan_ndarray_v1_elementwise_maximum, BINARY_MAXIMUM)
BINARY_WRAPPER(terlan_ndarray_v1_fmin, BINARY_FMIN)
BINARY_WRAPPER(terlan_ndarray_v1_fmax, BINARY_FMAX)
BINARY_WRAPPER(terlan_ndarray_v1_equal, BINARY_EQUAL)
BINARY_WRAPPER(terlan_ndarray_v1_not_equal, BINARY_NOT_EQUAL)
BINARY_WRAPPER(terlan_ndarray_v1_less, BINARY_LESS)
BINARY_WRAPPER(terlan_ndarray_v1_less_equal, BINARY_LESS_EQUAL)
BINARY_WRAPPER(terlan_ndarray_v1_greater, BINARY_GREATER)
BINARY_WRAPPER(terlan_ndarray_v1_greater_equal, BINARY_GREATER_EQUAL)
BINARY_WRAPPER(terlan_ndarray_v1_logical_and, BINARY_LOGICAL_AND)
BINARY_WRAPPER(terlan_ndarray_v1_logical_or, BINARY_LOGICAL_OR)
BINARY_WRAPPER(terlan_ndarray_v1_logical_xor, BINARY_LOGICAL_XOR)
BINARY_WRAPPER(terlan_ndarray_v1_bitwise_and, BINARY_BITWISE_AND)
BINARY_WRAPPER(terlan_ndarray_v1_bitwise_or, BINARY_BITWISE_OR)
BINARY_WRAPPER(terlan_ndarray_v1_bitwise_xor, BINARY_BITWISE_XOR)
BINARY_WRAPPER(terlan_ndarray_v1_left_shift, BINARY_LEFT_SHIFT)
BINARY_WRAPPER(terlan_ndarray_v1_right_shift, BINARY_RIGHT_SHIFT)
BINARY_WRAPPER(terlan_ndarray_v1_gcd, BINARY_GCD)
BINARY_WRAPPER(terlan_ndarray_v1_lcm, BINARY_LCM)
BINARY_WRAPPER(terlan_ndarray_v1_hypot, BINARY_HYPOT)
BINARY_WRAPPER(terlan_ndarray_v1_arctan2, BINARY_ARCTAN2)
BINARY_WRAPPER(terlan_ndarray_v1_logaddexp, BINARY_LOGADDEXP)
BINARY_WRAPPER(terlan_ndarray_v1_logaddexp2, BINARY_LOGADDEXP2)
BINARY_WRAPPER(terlan_ndarray_v1_copysign, BINARY_COPYSIGN)
BINARY_WRAPPER(terlan_ndarray_v1_nextafter, BINARY_NEXTAFTER)

TerlanNdArrayStatus terlan_ndarray_v1_is_close(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, double relative_tolerance,
    double absolute_tolerance, int32_t equal_nan,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(lhs);
  BroadcastPlan plan;
  TerlanNdArrayHandle result = NULL;
  int64_t index;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(rhs);
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL ||
      !isfinite(relative_tolerance) || !isfinite(absolute_tolerance) ||
      relative_tolerance < 0.0 || absolute_tolerance < 0.0 ||
      (equal_nan != 0 && equal_nan != 1)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = create_plan(lhs, rhs, &plan);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  status = create_result(&plan, TERLAN_NDARRAY_DTYPE_BOOL, &result);
  if (status != TERLAN_NDARRAY_OK) {
    destroy_plan(&plan);
    return status;
  }
  for (index = 0; index < plan.numel; ++index) {
    int64_t lhs_offset;
    int64_t rhs_offset;
    TerlanNdArrayScalar left;
    TerlanNdArrayScalar right;
    TerlanNdArrayScalar output = {0.0L, 0.0L};
    const long double complex zero = 0.0L;
    plan_offsets(&plan, index, &lhs_offset, &rhs_offset);
    (void)dtype_load((const uint8_t *)lhs->data +
                         lhs_offset * (int64_t)dtype_size(lhs->dtype),
                     lhs->dtype, &left);
    (void)dtype_load((const uint8_t *)rhs->data +
                         rhs_offset * (int64_t)dtype_size(rhs->dtype),
                     rhs->dtype, &right);
    if (scalar_has_nan(left) || scalar_has_nan(right)) {
      output.real = equal_nan != 0 && scalar_has_nan(left) &&
                    scalar_has_nan(right);
    } else {
      const long double complex difference =
          (left.real - right.real) + (left.imag - right.imag) * I;
      const long double complex reference = right.real + right.imag * I;
      output.real = cabsl(difference + zero) <=
                    absolute_tolerance + relative_tolerance * cabsl(reference);
    }
    (void)dtype_store(array_element_pointer(result, index), result->dtype,
                      output);
  }
  destroy_plan(&plan);
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}
TerlanNdArrayStatus terlan_ndarray_v1_clip(
    TerlanNdArrayHandle array, TerlanNdArrayHandle minimum,
    TerlanNdArrayHandle maximum, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayHandle lower_bounded = NULL;
  TerlanNdArrayScalar bound;
  TerlanNdArrayStatus status = require_live(array);
  int ignore_minimum = 0;
  int ignore_maximum = 0;
  if (out_array == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(minimum);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(maximum);
  }
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (minimum->rank == 0 &&
      dtype_load(array_const_element_pointer(minimum, 0), minimum->dtype,
                 &bound) == TERLAN_NDARRAY_OK) {
    ignore_minimum = scalar_has_nan(bound);
  }
  if (maximum->rank == 0 &&
      dtype_load(array_const_element_pointer(maximum, 0), maximum->dtype,
                 &bound) == TERLAN_NDARRAY_OK) {
    ignore_maximum = scalar_has_nan(bound);
  }
  if (ignore_minimum && ignore_maximum) {
    return copy_array(array, out_array);
  }
  if (ignore_minimum) {
    return terlan_ndarray_v1_elementwise_minimum(array, maximum, out_array);
  }
  if (ignore_maximum) {
    return terlan_ndarray_v1_elementwise_maximum(array, minimum, out_array);
  }
  status = terlan_ndarray_v1_elementwise_maximum(array, minimum,
                                                 &lower_bounded);
  if (status == TERLAN_NDARRAY_OK) {
    status = terlan_ndarray_v1_elementwise_minimum(lower_bounded, maximum,
                                                   out_array);
  }
  if (lower_bounded != NULL) {
    (void)terlan_ndarray_v1_delete(lower_bounded);
  }
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_clip_lower(
    TerlanNdArrayHandle array, TerlanNdArrayHandle minimum,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayScalar bound;
  TerlanNdArrayStatus status = require_live(array);
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(minimum);
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (minimum->rank == 0 &&
      dtype_load(array_const_element_pointer(minimum, 0), minimum->dtype,
                 &bound) == TERLAN_NDARRAY_OK &&
      scalar_has_nan(bound)) {
    return copy_array(array, out_array);
  }
  return terlan_ndarray_v1_elementwise_maximum(array, minimum, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_clip_upper(
    TerlanNdArrayHandle array, TerlanNdArrayHandle maximum,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayScalar bound;
  TerlanNdArrayStatus status = require_live(array);
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(maximum);
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (maximum->rank == 0 &&
      dtype_load(array_const_element_pointer(maximum, 0), maximum->dtype,
                 &bound) == TERLAN_NDARRAY_OK &&
      scalar_has_nan(bound)) {
    return copy_array(array, out_array);
  }
  return terlan_ndarray_v1_elementwise_minimum(array, maximum, out_array);
}

static TerlanNdArrayStatus finish_clip_into(
    TerlanNdArrayHandle destination, TerlanNdArrayStatus clip_status,
    TerlanNdArrayHandle result) {
  return array_finish_into(destination, clip_status, result,
                           TERLAN_NDARRAY_CAST_UNSAFE);
}

TerlanNdArrayStatus terlan_ndarray_v1_clip_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    TerlanNdArrayHandle minimum, TerlanNdArrayHandle maximum) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status =
      terlan_ndarray_v1_clip(array, minimum, maximum, &result);
  return finish_clip_into(destination, status, result);
}

TerlanNdArrayStatus terlan_ndarray_v1_clip_lower_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    TerlanNdArrayHandle minimum) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status =
      terlan_ndarray_v1_clip_lower(array, minimum, &result);
  return finish_clip_into(destination, status, result);
}

TerlanNdArrayStatus terlan_ndarray_v1_clip_upper_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    TerlanNdArrayHandle maximum) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status =
      terlan_ndarray_v1_clip_upper(array, maximum, &result);
  return finish_clip_into(destination, status, result);
}

TerlanNdArrayStatus terlan_ndarray_v1_clip_inplace(
    TerlanNdArrayHandle array, TerlanNdArrayHandle minimum,
    TerlanNdArrayHandle maximum) {
  return terlan_ndarray_v1_clip_into(array, array, minimum, maximum);
}

TerlanNdArrayStatus terlan_ndarray_v1_clip_lower_inplace(
    TerlanNdArrayHandle array, TerlanNdArrayHandle minimum) {
  return terlan_ndarray_v1_clip_lower_into(array, array, minimum);
}

TerlanNdArrayStatus terlan_ndarray_v1_clip_upper_inplace(
    TerlanNdArrayHandle array, TerlanNdArrayHandle maximum) {
  return terlan_ndarray_v1_clip_upper_into(array, array, maximum);
}

static TerlanNdArrayStatus create_where_plan(
    TerlanNdArrayHandle condition, TerlanNdArrayHandle when_true,
    TerlanNdArrayHandle when_false, BroadcastPlan *out_plan,
    int64_t **out_condition_strides) {
  int64_t output_axis;
  if (out_plan == NULL || out_condition_strides == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  out_plan->rank = when_true->rank > when_false->rank ? when_true->rank
                                                       : when_false->rank;
  if (condition->rank > out_plan->rank) {
    out_plan->rank = condition->rank;
  }
  out_plan->numel = 1;
  out_plan->shape = NULL;
  out_plan->lhs_strides = NULL;
  out_plan->rhs_strides = NULL;
  *out_condition_strides = NULL;
  if (out_plan->rank > 0) {
    const size_t size = (size_t)out_plan->rank * sizeof(int64_t);
    out_plan->shape = checked_malloc(size);
    out_plan->lhs_strides = checked_malloc(size);
    out_plan->rhs_strides = checked_malloc(size);
    *out_condition_strides = checked_malloc(size);
    if (out_plan->shape == NULL || out_plan->lhs_strides == NULL ||
        out_plan->rhs_strides == NULL || *out_condition_strides == NULL) {
      free(*out_condition_strides);
      *out_condition_strides = NULL;
      destroy_plan(out_plan);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (output_axis = 0; output_axis < out_plan->rank; ++output_axis) {
    const int64_t true_axis =
        output_axis - (out_plan->rank - when_true->rank);
    const int64_t false_axis =
        output_axis - (out_plan->rank - when_false->rank);
    const int64_t condition_axis =
        output_axis - (out_plan->rank - condition->rank);
    const int64_t true_size =
        true_axis < 0 ? 1 : when_true->shape[true_axis];
    const int64_t false_size =
        false_axis < 0 ? 1 : when_false->shape[false_axis];
    const int64_t condition_size =
        condition_axis < 0 ? 1 : condition->shape[condition_axis];
    int64_t branch_size;
    int64_t output_size;
    TerlanNdArrayStatus status =
        broadcast_dimension(true_size, false_size, &branch_size);
    if (status == TERLAN_NDARRAY_OK) {
      status = broadcast_dimension(branch_size, condition_size, &output_size);
    }
    if (status != TERLAN_NDARRAY_OK) {
      free(*out_condition_strides);
      *out_condition_strides = NULL;
      destroy_plan(out_plan);
      return status;
    }
    out_plan->shape[output_axis] = output_size;
    out_plan->lhs_strides[output_axis] =
        true_axis < 0 || true_size == 1 ? 0 : when_true->strides[true_axis];
    out_plan->rhs_strides[output_axis] =
        false_axis < 0 || false_size == 1 ? 0 : when_false->strides[false_axis];
    (*out_condition_strides)[output_axis] =
        condition_axis < 0 || condition_size == 1
            ? 0
            : condition->strides[condition_axis];
    if (output_size == 0) {
      out_plan->numel = 0;
    } else if (out_plan->numel != 0) {
      if (out_plan->numel > INT64_MAX / output_size) {
        free(*out_condition_strides);
        *out_condition_strides = NULL;
        destroy_plan(out_plan);
        return TERLAN_NDARRAY_SIZE_OVERFLOW;
      }
      out_plan->numel *= output_size;
    }
  }
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_where(
    TerlanNdArrayHandle condition, TerlanNdArrayHandle when_true,
    TerlanNdArrayHandle when_false, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(condition);
  BroadcastPlan plan;
  int64_t *condition_strides = NULL;
  TerlanNdArrayHandle result = NULL;
  int32_t output_dtype;
  int64_t index;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(when_true);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(when_false);
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = create_where_plan(condition, when_true, when_false, &plan,
                             &condition_strides);
  if (status != TERLAN_NDARRAY_OK) {
    destroy_plan(&plan);
    return status;
  }
  output_dtype = dtype_promote(when_true->dtype, when_false->dtype);
  status = create_result(&plan, output_dtype, &result);
  if (status != TERLAN_NDARRAY_OK) {
    free(condition_strides);
    destroy_plan(&plan);
    return status;
  }
  for (index = 0; index < plan.numel; ++index) {
    int64_t lhs_offset;
    int64_t rhs_offset;
    int64_t condition_offset = 0;
    int64_t remaining = index;
    int64_t axis;
    TerlanNdArrayScalar predicate;
    TerlanNdArrayScalar selected;
    plan_offsets(&plan, index, &lhs_offset, &rhs_offset);
    for (axis = plan.rank; axis > 0; --axis) {
      const int64_t output_axis = axis - 1;
      const int64_t coordinate = remaining % plan.shape[output_axis];
      remaining /= plan.shape[output_axis];
      condition_offset += coordinate * condition_strides[output_axis];
    }
    status = dtype_load((const uint8_t *)condition->data +
                            condition_offset *
                                (int64_t)dtype_size(condition->dtype),
                        condition->dtype, &predicate);
    if (status == TERLAN_NDARRAY_OK && scalar_truth(predicate)) {
      status = dtype_load((const uint8_t *)when_true->data +
                              lhs_offset *
                                  (int64_t)dtype_size(when_true->dtype),
                          when_true->dtype, &selected);
    } else if (status == TERLAN_NDARRAY_OK) {
      status = dtype_load((const uint8_t *)when_false->data +
                              rhs_offset *
                                  (int64_t)dtype_size(when_false->dtype),
                          when_false->dtype, &selected);
    }
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_store(array_element_pointer(result, index), output_dtype,
                           selected);
    }
    if (status != TERLAN_NDARRAY_OK) {
      (void)terlan_ndarray_v1_delete(result);
      free(condition_strides);
      destroy_plan(&plan);
      return status;
    }
  }
  free(condition_strides);
  destroy_plan(&plan);
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}
