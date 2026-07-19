#include "terlan_ndarray_internal.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int same_shape(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs) {
  if (lhs->rank != rhs->rank) {
    return 0;
  }
  if (lhs->rank == 0) {
    return 1;
  }
  return memcmp(lhs->shape, rhs->shape,
                (size_t)lhs->rank * sizeof(*lhs->shape)) == 0;
}

static TerlanNdArrayStatus normalize_axis(int64_t axis, int64_t rank,
                                          int64_t *out_axis) {
  if (out_axis == NULL || rank <= 0) {
    return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
  }
  if (axis < 0) {
    if (axis < -rank) {
      return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
    }
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
  }
  *out_axis = axis;
  return TERLAN_NDARRAY_OK;
}

static int checked_add_int64(int64_t lhs, int64_t rhs, int64_t *out_value) {
  if ((rhs > 0 && lhs > INT64_MAX - rhs) ||
      (rhs < 0 && lhs < INT64_MIN - rhs)) {
    return 0;
  }
  *out_value = lhs + rhs;
  return 1;
}

static int checked_subtract_int64(int64_t lhs, int64_t rhs,
                                  int64_t *out_value) {
  if ((rhs > 0 && lhs < INT64_MIN + rhs) ||
      (rhs < 0 && lhs > INT64_MAX + rhs)) {
    return 0;
  }
  *out_value = lhs - rhs;
  return 1;
}

static int checked_multiply_int64(int64_t lhs, int64_t rhs,
                                  int64_t *out_value) {
  if (lhs == 0 || rhs == 0) {
    *out_value = 0;
    return 1;
  }
  if ((lhs == -1 && rhs == INT64_MIN) ||
      (rhs == -1 && lhs == INT64_MIN)) {
    return 0;
  }
  if ((lhs > 0 && rhs > 0 && lhs > INT64_MAX / rhs) ||
      (lhs > 0 && rhs < 0 && rhs < INT64_MIN / lhs) ||
      (lhs < 0 && rhs > 0 && lhs < INT64_MIN / rhs) ||
      (lhs < 0 && rhs < 0 && rhs < INT64_MAX / lhs)) {
    return 0;
  }
  *out_value = lhs * rhs;
  return 1;
}

TerlanNdArrayStatus terlan_ndarray_v1_reshape(
    TerlanNdArrayHandle array, const int64_t *shape, int64_t rank,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  return create_array(array->data, array->numel, shape, rank,
                      dtype_size(array->dtype), array->dtype, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_transpose(
    TerlanNdArrayHandle array, int64_t axis_a, int64_t axis_b,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle result = NULL;
  int64_t normalized_a = 0;
  int64_t normalized_b = 0;
  int64_t *shape = NULL;
  size_t dimensions_size;
  size_t element_size;
  int64_t output_index;

  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (array->rank == 0) {
    if (axis_a != 0 || axis_b != 0) {
      return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
    }
    return create_array(array->data, array->numel, NULL, 0,
                        dtype_size(array->dtype), array->dtype, out_array);
  }
  status = normalize_axis(axis_a, array->rank, &normalized_a);
  if (status == TERLAN_NDARRAY_OK) {
    status = normalize_axis(axis_b, array->rank, &normalized_b);
  }
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  dimensions_size = (size_t)array->rank * sizeof(*shape);
  shape = checked_malloc(dimensions_size);
  if (shape == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  memcpy(shape, array->shape, dimensions_size);
  shape[normalized_a] = array->shape[normalized_b];
  shape[normalized_b] = array->shape[normalized_a];
  element_size = dtype_size(array->dtype);
  status = create_array(array->data, array->numel, shape, array->rank,
                        element_size, array->dtype, &result);
  free(shape);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (output_index = 0; output_index < result->numel; ++output_index) {
    int64_t source_index = 0;
    int64_t dimension;
    for (dimension = 0; dimension < result->rank; ++dimension) {
      const int64_t coordinate =
          (output_index / result->strides[dimension]) % result->shape[dimension];
      int64_t source_dimension = dimension;
      if (dimension == normalized_a) {
        source_dimension = normalized_b;
      } else if (dimension == normalized_b) {
        source_dimension = normalized_a;
      }
      source_index += coordinate * array->strides[source_dimension];
    }
    memcpy((uint8_t *)result->data + (size_t)output_index * element_size,
           (const uint8_t *)array->data + (size_t)source_index * element_size,
           element_size);
  }
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}
typedef enum TerlanNdArrayBinaryOperation {
  TERLAN_NDARRAY_BINARY_ADD,
  TERLAN_NDARRAY_BINARY_SUBTRACT,
  TERLAN_NDARRAY_BINARY_MULTIPLY
} TerlanNdArrayBinaryOperation;

static TerlanNdArrayStatus binary_operation(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs,
    TerlanNdArrayBinaryOperation operation, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(lhs);
  TerlanNdArrayHandle result = NULL;
  int64_t index;

  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(rhs);
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (lhs->dtype != rhs->dtype) {
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  if (!same_shape(lhs, rhs)) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  if (lhs->dtype == TERLAN_NDARRAY_DTYPE_BOOL) {
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  status = create_array(lhs->data, lhs->numel, lhs->shape, lhs->rank,
                        dtype_size(lhs->dtype), lhs->dtype, &result);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (lhs->dtype == TERLAN_NDARRAY_DTYPE_INT64) {
    const int64_t *left = lhs->data;
    const int64_t *right = rhs->data;
    int64_t *output = result->data;
    for (index = 0; index < lhs->numel; ++index) {
      int valid;
      switch (operation) {
      case TERLAN_NDARRAY_BINARY_ADD:
        valid = checked_add_int64(left[index], right[index], &output[index]);
        break;
      case TERLAN_NDARRAY_BINARY_SUBTRACT:
        valid =
            checked_subtract_int64(left[index], right[index], &output[index]);
        break;
      case TERLAN_NDARRAY_BINARY_MULTIPLY:
        valid =
            checked_multiply_int64(left[index], right[index], &output[index]);
        break;
      default:
        valid = 0;
        break;
      }
      if (!valid) {
        (void)terlan_ndarray_v1_delete(result);
        return TERLAN_NDARRAY_ARITHMETIC_OVERFLOW;
      }
    }
  } else if (lhs->dtype == TERLAN_NDARRAY_DTYPE_FLOAT64) {
    const double *left = lhs->data;
    const double *right = rhs->data;
    double *output = result->data;
    for (index = 0; index < lhs->numel; ++index) {
      switch (operation) {
      case TERLAN_NDARRAY_BINARY_ADD:
        output[index] = left[index] + right[index];
        break;
      case TERLAN_NDARRAY_BINARY_SUBTRACT:
        output[index] = left[index] - right[index];
        break;
      case TERLAN_NDARRAY_BINARY_MULTIPLY:
        output[index] = left[index] * right[index];
        break;
      }
    }
  } else {
    (void)terlan_ndarray_v1_delete(result);
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_add(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs,
    TerlanNdArrayHandle *out_array) {
  return binary_operation(lhs, rhs, TERLAN_NDARRAY_BINARY_ADD, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_subtract(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs,
    TerlanNdArrayHandle *out_array) {
  return binary_operation(lhs, rhs, TERLAN_NDARRAY_BINARY_SUBTRACT, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_multiply(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs,
    TerlanNdArrayHandle *out_array) {
  return binary_operation(lhs, rhs, TERLAN_NDARRAY_BINARY_MULTIPLY, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_sum(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  uint8_t *reduced = NULL;
  int64_t *output_shape = NULL;
  int64_t normalized_count = axis_count;
  int64_t output_rank;
  int64_t axis_index;
  int64_t input_index;
  TerlanNdArrayHandle result = NULL;

  if (status != TERLAN_NDARRAY_OK || out_array == NULL || axis_count < 0 ||
      (axis_count > 0 && axes == NULL) || (keep_dims != 0 && keep_dims != 1)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (array->dtype == TERLAN_NDARRAY_DTYPE_BOOL) {
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  if (axis_count == 0) {
    normalized_count = array->rank;
  }
  if (array->rank > 0) {
    reduced = checked_calloc((size_t)array->rank, sizeof(*reduced));
    if (reduced == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (axis_index = 0; axis_index < normalized_count; ++axis_index) {
    int64_t normalized_axis;
    status = axis_count == 0
                 ? TERLAN_NDARRAY_OK
                 : normalize_axis(axes[axis_index], array->rank,
                                  &normalized_axis);
    if (axis_count == 0) {
      normalized_axis = axis_index;
    }
    if (status != TERLAN_NDARRAY_OK) {
      free(reduced);
      return status;
    }
    if (reduced[normalized_axis] != 0) {
      free(reduced);
      return TERLAN_NDARRAY_DUPLICATE_AXIS;
    }
    reduced[normalized_axis] = 1;
  }
  output_rank = keep_dims != 0 ? array->rank : array->rank - normalized_count;
  if (output_rank > 0) {
    int64_t output_dimension = 0;
    output_shape = checked_malloc((size_t)output_rank * sizeof(*output_shape));
    if (output_shape == NULL) {
      free(reduced);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    for (axis_index = 0; axis_index < array->rank; ++axis_index) {
      if (keep_dims != 0) {
        output_shape[output_dimension++] =
            reduced[axis_index] != 0 ? 1 : array->shape[axis_index];
      } else if (reduced[axis_index] == 0) {
        output_shape[output_dimension++] = array->shape[axis_index];
      }
    }
  }
  {
    int64_t output_numel = 1;
    int64_t dimension;
    void *zero_values = NULL;
    for (dimension = 0; dimension < output_rank; ++dimension) {
      if (output_shape[dimension] != 0 &&
          output_numel > INT64_MAX / output_shape[dimension]) {
        free(output_shape);
        free(reduced);
        return TERLAN_NDARRAY_SIZE_OVERFLOW;
      }
      output_numel *= output_shape[dimension];
    }
    if (output_numel > 0) {
      const size_t element_size = dtype_size(array->dtype);
      if ((uint64_t)output_numel > SIZE_MAX / element_size) {
        free(output_shape);
        free(reduced);
        return TERLAN_NDARRAY_SIZE_OVERFLOW;
      }
      zero_values = checked_calloc((size_t)output_numel, element_size);
      if (zero_values == NULL) {
        free(output_shape);
        free(reduced);
        return TERLAN_NDARRAY_ALLOCATION_FAILED;
      }
    }
    status = create_array(zero_values, output_numel, output_shape, output_rank,
                          dtype_size(array->dtype), array->dtype, &result);
    free(zero_values);
  }
  free(output_shape);
  if (status != TERLAN_NDARRAY_OK) {
    free(reduced);
    return status;
  }
  if (result->byte_count > 0) {
    memset(result->data, 0, (size_t)result->byte_count);
  }
  for (input_index = 0; input_index < array->numel; ++input_index) {
    int64_t result_index = 0;
    int64_t result_dimension = 0;
    int64_t dimension;
    for (dimension = 0; dimension < array->rank; ++dimension) {
      const int64_t coordinate =
          (input_index / array->strides[dimension]) % array->shape[dimension];
      if (keep_dims != 0) {
        if (reduced[dimension] == 0) {
          result_index += coordinate * result->strides[dimension];
        }
      } else if (reduced[dimension] == 0) {
        result_index += coordinate * result->strides[result_dimension++];
      }
    }
    if (array->dtype == TERLAN_NDARRAY_DTYPE_INT64) {
      const int64_t *input = array->data;
      int64_t *output = result->data;
      if (!checked_add_int64(output[result_index], input[input_index],
                             &output[result_index])) {
        free(reduced);
        (void)terlan_ndarray_v1_delete(result);
        return TERLAN_NDARRAY_ARITHMETIC_OVERFLOW;
      }
    } else {
      const double *input = array->data;
      double *output = result->data;
      output[result_index] += input[input_index];
    }
  }
  free(reduced);
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}
