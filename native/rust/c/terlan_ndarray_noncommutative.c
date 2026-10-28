#include "terlan_ndarray_internal.h"

#include <complex.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum NoncommutativeOperation {
  NONCOMMUTATIVE_SUBTRACT,
  NONCOMMUTATIVE_DIVIDE
} NoncommutativeOperation;

static TerlanNdArrayStatus normalize_noncommutative_axis(
    TerlanNdArrayHandle array, int64_t axis, int64_t *out_axis) {
  if (array->rank == 0 || out_axis == NULL) {
    return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
  }
  axis = axis < 0 ? axis + array->rank : axis;
  if (axis < 0 || axis >= array->rank) {
    return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
  }
  *out_axis = axis;
  return TERLAN_NDARRAY_OK;
}

static int32_t noncommutative_dtype(TerlanNdArrayHandle array,
                                    NoncommutativeOperation operation) {
  return operation == NONCOMMUTATIVE_DIVIDE
             ? dtype_inexact_or_float64(array->dtype)
             : array->dtype;
}

static TerlanNdArrayStatus combine_noncommutative(
    TerlanNdArrayScalar accumulator, TerlanNdArrayScalar operand,
    int32_t dtype, NoncommutativeOperation operation,
    TerlanNdArrayScalar *out_value) {
  if (operation == NONCOMMUTATIVE_SUBTRACT && dtype_is_integer(dtype)) {
    return scalar_wrapping_integer_binary(TERLAN_NDARRAY_INTEGER_SUBTRACT,
                                          accumulator.real, operand.real,
                                          dtype, out_value);
  }
  if (operation == NONCOMMUTATIVE_SUBTRACT) {
    out_value->real = accumulator.real - operand.real;
    out_value->imag = accumulator.imag - operand.imag;
  } else if (dtype_is_complex(dtype)) {
    const long double complex lhs =
        CMPLXL(accumulator.real, accumulator.imag);
    const long double complex rhs = CMPLXL(operand.real, operand.imag);
    const long double complex result = lhs / rhs;
    out_value->real = creall(result);
    out_value->imag = cimagl(result);
  } else {
    out_value->real = accumulator.real / operand.real;
    out_value->imag = 0.0L;
  }
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus create_noncommutative_output(
    TerlanNdArrayHandle array, int64_t axis, int accumulate, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  int64_t *shape = NULL;
  int64_t source_axis;
  int64_t output_axis = 0;
  const int64_t rank = accumulate ? array->rank : array->rank - 1;
  TerlanNdArrayStatus status;
  if (rank > 0) {
    shape = checked_malloc((size_t)rank * sizeof(*shape));
    if (shape == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (source_axis = 0; source_axis < array->rank; ++source_axis) {
    if (accumulate || source_axis != axis) {
      shape[output_axis++] = array->shape[source_axis];
    }
  }
  status = create_empty_array(shape, rank, dtype, out_array);
  free(shape);
  return status;
}

static TerlanNdArrayStatus noncommutative_operation(
    TerlanNdArrayHandle array, int64_t requested_axis,
    NoncommutativeOperation operation, int accumulate,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle output = NULL;
  TerlanNdArrayScalar accumulator;
  TerlanNdArrayScalar operand;
  int64_t axis = 0;
  int64_t inner = 1;
  int64_t outer;
  int64_t outer_index;
  int64_t inner_index;
  int64_t axis_index;
  int32_t output_dtype;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = normalize_noncommutative_axis(array, requested_axis, &axis);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (array->shape[axis] == 0) {
    return TERLAN_NDARRAY_DOMAIN_ERROR;
  }
  if (operation == NONCOMMUTATIVE_SUBTRACT && dtype_is_bool(array->dtype)) {
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  output_dtype = noncommutative_dtype(array, operation);
  for (axis_index = axis + 1; axis_index < array->rank; ++axis_index) {
    inner *= array->shape[axis_index];
  }
  outer = array->numel / (array->shape[axis] * inner);
  status = create_noncommutative_output(array, axis, accumulate, output_dtype,
                                        &output);
  for (outer_index = 0; status == TERLAN_NDARRAY_OK && outer_index < outer;
       ++outer_index) {
    for (inner_index = 0;
         status == TERLAN_NDARRAY_OK && inner_index < inner; ++inner_index) {
      const int64_t first =
          outer_index * array->shape[axis] * inner + inner_index;
      status = dtype_load(array_const_element_pointer(array, first),
                          array->dtype, &accumulator);
      if (status == TERLAN_NDARRAY_OK && accumulate) {
        status = dtype_store(array_element_pointer(output, first), output_dtype,
                             accumulator);
      }
      for (axis_index = 1;
           status == TERLAN_NDARRAY_OK && axis_index < array->shape[axis];
           ++axis_index) {
        const int64_t source = first + axis_index * inner;
        status = dtype_load(array_const_element_pointer(array, source),
                            array->dtype, &operand);
        if (status == TERLAN_NDARRAY_OK) {
          status = combine_noncommutative(accumulator, operand, output_dtype,
                                          operation, &accumulator);
        }
        if (status == TERLAN_NDARRAY_OK && accumulate) {
          status = dtype_store(array_element_pointer(output, source),
                               output_dtype, accumulator);
        }
      }
      if (status == TERLAN_NDARRAY_OK && !accumulate) {
        status = dtype_store(
            array_element_pointer(output, outer_index * inner + inner_index),
            output_dtype, accumulator);
      }
    }
  }
  if (status != TERLAN_NDARRAY_OK) {
    if (output != NULL) {
      (void)terlan_ndarray_v1_delete(output);
    }
    return status;
  }
  *out_array = output;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_subtract_reduce(
    TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array) {
  return noncommutative_operation(array, axis, NONCOMMUTATIVE_SUBTRACT, 0,
                                  out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_subtract_accumulate(
    TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array) {
  return noncommutative_operation(array, axis, NONCOMMUTATIVE_SUBTRACT, 1,
                                  out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_divide_reduce(
    TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array) {
  return noncommutative_operation(array, axis, NONCOMMUTATIVE_DIVIDE, 0,
                                  out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_divide_accumulate(
    TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array) {
  return noncommutative_operation(array, axis, NONCOMMUTATIVE_DIVIDE, 1,
                                  out_array);
}
