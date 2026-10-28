#include "terlan_ndarray_internal.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static TerlanNdArrayStatus normalize_axis_index(int64_t value, int64_t size,
                                                 int64_t *out_value) {
  if (out_value == NULL || size < 0) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (value < 0) {
    value += size;
  }
  if (value < 0 || value >= size) {
    return TERLAN_NDARRAY_INDEX_OUT_OF_RANGE;
  }
  *out_value = value;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus normalize_axis_number(int64_t value, int64_t rank,
                                                  int64_t *out_value) {
  if (rank <= 0) {
    return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
  }
  if (value < 0) {
    value += rank;
  }
  if (value < 0 || value >= rank) {
    return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
  }
  *out_value = value;
  return TERLAN_NDARRAY_OK;
}

static int checked_add(int64_t lhs, int64_t rhs, int64_t *out_value) {
  if ((rhs > 0 && lhs > INT64_MAX - rhs) ||
      (rhs < 0 && lhs < INT64_MIN - rhs)) {
    return 0;
  }
  *out_value = lhs + rhs;
  return 1;
}

static int checked_multiply(int64_t lhs, int64_t rhs, int64_t *out_value) {
  if (lhs == 0 || rhs == 0) {
    *out_value = 0;
    return 1;
  }
  if ((lhs == -1 && rhs == INT64_MIN) ||
      (rhs == -1 && lhs == INT64_MIN)) {
    return 0;
  }
  if (lhs > 0) {
    if ((rhs > 0 && lhs > INT64_MAX / rhs) ||
        (rhs < 0 && rhs < INT64_MIN / lhs)) {
      return 0;
    }
  } else if ((rhs > 0 && lhs < INT64_MIN / rhs) ||
             (rhs < 0 && lhs < INT64_MAX / rhs)) {
    return 0;
  }
  *out_value = lhs * rhs;
  return 1;
}

static TerlanNdArrayStatus indexed_pointer(TerlanNdArrayHandle array,
                                           const int64_t *indices,
                                           int64_t index_count,
                                           void **out_pointer) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t offset = 0;
  int64_t dimension;
  if (status != TERLAN_NDARRAY_OK || out_pointer == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (index_count != array->rank) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  if (index_count > 0 && indices == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  for (dimension = 0; dimension < array->rank; ++dimension) {
    int64_t normalized;
    int64_t contribution;
    status = normalize_axis_index(indices[dimension], array->shape[dimension],
                                  &normalized);
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
    if (!checked_multiply(normalized, array->strides[dimension],
                          &contribution) ||
        !checked_add(offset, contribution, &offset)) {
      return TERLAN_NDARRAY_SIZE_OVERFLOW;
    }
  }
  *out_pointer = (uint8_t *)array->data +
                 offset * (int64_t)dtype_size(array->dtype);
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_copy(TerlanNdArrayHandle array,
                                            TerlanNdArrayHandle *out_array) {
  return copy_array_ordered(array, array == NULL ? 0 : array->dtype,
                            TERLAN_NDARRAY_ORDER_KEEP,
                            TERLAN_NDARRAY_CAST_NO, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_copy_ordered(
    TerlanNdArrayHandle array, int32_t order,
    TerlanNdArrayHandle *out_array) {
  return copy_array_ordered(array, array == NULL ? 0 : array->dtype, order,
                            TERLAN_NDARRAY_CAST_NO, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_contiguous(
    TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array) {
  return copy_array(array, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_index_axis(
    TerlanNdArrayHandle array, int64_t axis, int64_t index,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t normalized_axis;
  int64_t normalized_index;
  int64_t *shape = NULL;
  int64_t *strides = NULL;
  int64_t source_dimension;
  int64_t output_dimension = 0;
  int64_t output_rank;

  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = normalize_axis_number(axis, array->rank, &normalized_axis);
  if (status == TERLAN_NDARRAY_OK) {
    status = normalize_axis_index(index, array->shape[normalized_axis],
                                  &normalized_index);
  }
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  output_rank = array->rank - 1;
  if (output_rank > 0) {
    shape = checked_malloc((size_t)output_rank * sizeof(*shape));
    strides = checked_malloc((size_t)output_rank * sizeof(*strides));
    if (shape == NULL || strides == NULL) {
      free(strides);
      free(shape);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    for (source_dimension = 0; source_dimension < array->rank;
         ++source_dimension) {
      if (source_dimension != normalized_axis) {
        shape[output_dimension] = array->shape[source_dimension];
        strides[output_dimension++] = array->strides[source_dimension];
      }
    }
  }
  status = create_view(array, shape, strides, output_rank,
                       normalized_index * array->strides[normalized_axis],
                       array->writable, out_array);
  free(strides);
  free(shape);
  return status;
}

static int64_t clamp(int64_t value, int64_t minimum, int64_t maximum) {
  if (value < minimum) {
    return minimum;
  }
  return value > maximum ? maximum : value;
}

static void normalize_slice(int64_t length, int64_t raw_start,
                            int64_t raw_stop, int64_t step, int64_t *out_start,
                            int64_t *out_count) {
  int64_t start;
  int64_t stop;
  if (step > 0) {
    start = raw_start == INT64_MIN ? 0 : raw_start;
    stop = raw_stop == INT64_MIN ? length : raw_stop;
    if (start < 0) {
      start += length;
    }
    if (stop < 0) {
      stop += length;
    }
    start = clamp(start, 0, length);
    stop = clamp(stop, 0, length);
    *out_count = stop <= start ? 0 : 1 + (stop - start - 1) / step;
  } else {
    start = raw_start == INT64_MIN ? length - 1 : raw_start;
    stop = raw_stop == INT64_MIN ? -1 : raw_stop;
    if (raw_start != INT64_MIN && start < 0) {
      start += length;
    }
    if (raw_stop != INT64_MIN && stop < 0) {
      stop += length;
    }
    start = clamp(start, -1, length - 1);
    stop = clamp(stop, -1, length - 1);
    *out_count = stop >= start ? 0 : 1 + (start - stop - 1) / -step;
  }
  *out_start = *out_count == 0 ? 0 : start;
}

TerlanNdArrayStatus terlan_ndarray_v1_slice_axis(
    TerlanNdArrayHandle array, int64_t axis, int64_t start, int64_t stop,
    int64_t step, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t normalized_axis;
  int64_t normalized_start;
  int64_t count;
  int64_t slice_stride;
  int64_t relative_offset;
  int64_t *shape = NULL;
  int64_t *strides = NULL;
  size_t dimensions_size;

  if (status != TERLAN_NDARRAY_OK || out_array == NULL || step == 0 ||
      step == INT64_MIN) {
    return status != TERLAN_NDARRAY_OK
               ? status
               : (step == 0 || step == INT64_MIN
                      ? TERLAN_NDARRAY_INVALID_SLICE
                      : TERLAN_NDARRAY_INVALID_ARGUMENT);
  }
  *out_array = NULL;
  status = normalize_axis_number(axis, array->rank, &normalized_axis);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  normalize_slice(array->shape[normalized_axis], start, stop, step,
                  &normalized_start, &count);
  dimensions_size = (size_t)array->rank * sizeof(*shape);
  shape = checked_malloc(dimensions_size);
  strides = checked_malloc(dimensions_size);
  if (shape == NULL || strides == NULL) {
    free(strides);
    free(shape);
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  memcpy(shape, array->shape, dimensions_size);
  memcpy(strides, array->strides, dimensions_size);
  shape[normalized_axis] = count;
  if (!checked_multiply(strides[normalized_axis], step, &slice_stride) ||
      !checked_multiply(normalized_start, array->strides[normalized_axis],
                        &relative_offset)) {
    free(strides);
    free(shape);
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  strides[normalized_axis] = slice_stride;
  status = create_view(array, shape, strides, array->rank, relative_offset,
                       array->writable, out_array);
  free(strides);
  free(shape);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_permute(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t *shape = NULL;
  int64_t *strides = NULL;
  uint8_t *seen = NULL;
  int64_t output_axis;

  if (status != TERLAN_NDARRAY_OK || out_array == NULL ||
      axis_count != (array == NULL ? -1 : array->rank) ||
      (axis_count > 0 && axes == NULL)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (axis_count > 0) {
    shape = checked_malloc((size_t)axis_count * sizeof(*shape));
    strides = checked_malloc((size_t)axis_count * sizeof(*strides));
    seen = checked_calloc((size_t)axis_count, sizeof(*seen));
    if (shape == NULL || strides == NULL || seen == NULL) {
      free(seen);
      free(strides);
      free(shape);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (output_axis = 0; output_axis < axis_count; ++output_axis) {
    int64_t source_axis;
    status = normalize_axis_number(axes[output_axis], array->rank, &source_axis);
    if (status != TERLAN_NDARRAY_OK || seen[source_axis] != 0) {
      status = status != TERLAN_NDARRAY_OK ? status
                                           : TERLAN_NDARRAY_DUPLICATE_AXIS;
      break;
    }
    seen[source_axis] = 1;
    shape[output_axis] = array->shape[source_axis];
    strides[output_axis] = array->strides[source_axis];
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = create_view(array, shape, strides, axis_count, 0, array->writable,
                         out_array);
  }
  free(seen);
  free(strides);
  free(shape);
  return status;
}

static TerlanNdArrayStatus normalize_take_index(int64_t index, int64_t count,
                                                int32_t mode,
                                                int64_t *out_index) {
  if (out_index == NULL || mode < 0 || mode > 2 || count < 0) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (count == 0) {
    return TERLAN_NDARRAY_INDEX_OUT_OF_RANGE;
  }
  if (mode == 1) {
    index %= count;
    if (index < 0) {
      index += count;
    }
  } else if (mode == 2) {
    if (index < 0) {
      index = 0;
    } else if (index >= count) {
      index = count - 1;
    }
  } else {
    TerlanNdArrayStatus status = normalize_axis_index(index, count, &index);
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
  }
  *out_index = index;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_take_mode(
    TerlanNdArrayHandle array, const int64_t *indices, int64_t index_count,
    int64_t axis, int32_t mode, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle result = NULL;
  int64_t *shape = NULL;
  int64_t normalized_axis;
  int64_t outer = 1;
  int64_t inner = 1;
  int64_t dimension;
  int64_t outer_index;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL || index_count < 0 ||
      (index_count > 0 && indices == NULL) || mode < 0 || mode > 2) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = normalize_axis_number(axis, array->rank, &normalized_axis);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  shape = checked_malloc((size_t)array->rank * sizeof(*shape));
  if (shape == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  memcpy(shape, array->shape, (size_t)array->rank * sizeof(*shape));
  shape[normalized_axis] = index_count;
  for (dimension = 0; dimension < normalized_axis; ++dimension) {
    outer *= array->shape[dimension];
  }
  for (dimension = normalized_axis + 1; dimension < array->rank; ++dimension) {
    inner *= array->shape[dimension];
  }
  status = create_empty_array(shape, array->rank, array->dtype, &result);
  free(shape);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (outer_index = 0; outer_index < outer; ++outer_index) {
    int64_t selected_index;
    for (selected_index = 0; selected_index < index_count; ++selected_index) {
      int64_t source_axis_index = indices[selected_index];
      int64_t inner_index;
      status = normalize_take_index(source_axis_index,
                                    array->shape[normalized_axis], mode,
                                    &source_axis_index);
      if (status != TERLAN_NDARRAY_OK) {
        (void)terlan_ndarray_v1_delete(result);
        return status;
      }
      for (inner_index = 0; inner_index < inner; ++inner_index) {
        const int64_t source_logical =
            (outer_index * array->shape[normalized_axis] + source_axis_index) *
                inner +
            inner_index;
        const int64_t output_logical =
            (outer_index * index_count + selected_index) * inner + inner_index;
        memcpy(array_element_pointer(result, output_logical),
               array_const_element_pointer(array, source_logical),
               dtype_size(array->dtype));
      }
    }
  }
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_take(
    TerlanNdArrayHandle array, const int64_t *indices, int64_t index_count,
    int64_t axis, TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_v1_take_mode(array, indices, index_count, axis, 0,
                                    out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_take_indices_mode(
    TerlanNdArrayHandle array, TerlanNdArrayHandle indices, int64_t axis,
    int32_t mode, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(indices);
  TerlanNdArrayHandle flattened = NULL;
  int64_t *index_values = NULL;
  int64_t *output_shape = NULL;
  int64_t normalized_axis;
  int64_t output_rank;
  int64_t input_axis;
  int64_t output_axis = 0;
  int64_t index;
  if (out_array != NULL) {
    *out_array = NULL;
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL ||
      !dtype_is_integer(indices->dtype)) {
    return status != TERLAN_NDARRAY_OK
               ? status
               : (out_array == NULL ? TERLAN_NDARRAY_INVALID_ARGUMENT
                                    : TERLAN_NDARRAY_DTYPE_MISMATCH);
  }
  status = require_live(array);
  if (status == TERLAN_NDARRAY_OK) {
    status = normalize_axis_number(axis, array->rank, &normalized_axis);
  }
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (!checked_add(array->rank - 1, indices->rank, &output_rank) ||
      (uint64_t)indices->numel > SIZE_MAX / sizeof(*index_values) ||
      (uint64_t)output_rank > SIZE_MAX / sizeof(*output_shape)) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  if (indices->numel > 0) {
    index_values = checked_malloc((size_t)indices->numel *
                                  sizeof(*index_values));
    if (index_values == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (index = 0; index < indices->numel; ++index) {
    status = array_load_int64(indices, index, &index_values[index]);
    if (status != TERLAN_NDARRAY_OK) {
      free(index_values);
      return status == TERLAN_NDARRAY_CAST_OVERFLOW
                 ? TERLAN_NDARRAY_INDEX_OUT_OF_RANGE
                 : status;
    }
  }
  status = terlan_ndarray_v1_take_mode(array, index_values, indices->numel,
                                       normalized_axis, mode, &flattened);
  free(index_values);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (output_rank > 0) {
    output_shape = checked_malloc((size_t)output_rank * sizeof(*output_shape));
    if (output_shape == NULL) {
      (void)terlan_ndarray_v1_delete(flattened);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (input_axis = 0; input_axis < normalized_axis; ++input_axis) {
    output_shape[output_axis++] = array->shape[input_axis];
  }
  for (input_axis = 0; input_axis < indices->rank; ++input_axis) {
    output_shape[output_axis++] = indices->shape[input_axis];
  }
  for (input_axis = normalized_axis + 1; input_axis < array->rank;
       ++input_axis) {
    output_shape[output_axis++] = array->shape[input_axis];
  }
  status = terlan_ndarray_v1_reshape(flattened, output_shape, output_rank,
                                     out_array);
  free(output_shape);
  (void)terlan_ndarray_v1_delete(flattened);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_take_indices(
    TerlanNdArrayHandle array, TerlanNdArrayHandle indices, int64_t axis,
    TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_v1_take_indices_mode(array, indices, axis, 0,
                                             out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_take_indices_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    TerlanNdArrayHandle indices, int64_t axis, int32_t mode) {
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayStatus status = terlan_ndarray_v1_take_indices_mode(
      array, indices, axis, mode, &result);
  return array_finish_into(destination, status, result,
                           TERLAN_NDARRAY_CAST_NO);
}

static TerlanNdArrayStatus count_nonzero_values(TerlanNdArrayHandle array,
                                                int64_t *out_count) {
  int64_t count = 0;
  int64_t index;
  for (index = 0; index < array->numel; ++index) {
    TerlanNdArrayScalar value;
    TerlanNdArrayStatus status =
        dtype_load(array_const_element_pointer(array, index), array->dtype,
                   &value);
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
    if (value.real != 0.0L || value.imag != 0.0L) {
      ++count;
    }
  }
  *out_count = count;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_count_nonzero(
    TerlanNdArrayHandle array, int64_t *out_count) {
  TerlanNdArrayStatus status = require_live(array);
  if (status != TERLAN_NDARRAY_OK || out_count == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  return count_nonzero_values(array, out_count);
}

TerlanNdArrayStatus terlan_ndarray_v1_flatnonzero(
    TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle result = NULL;
  int64_t count = 0;
  int64_t output_index = 0;
  int64_t logical_index;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = count_nonzero_values(array, &count);
  if (status == TERLAN_NDARRAY_OK) {
    status = create_empty_array(&count, 1, TERLAN_NDARRAY_DTYPE_INT64, &result);
  }
  for (logical_index = 0;
       status == TERLAN_NDARRAY_OK && logical_index < array->numel;
       ++logical_index) {
    TerlanNdArrayScalar value;
    status = dtype_load(array_const_element_pointer(array, logical_index),
                        array->dtype, &value);
    if (status == TERLAN_NDARRAY_OK &&
        (value.real != 0.0L || value.imag != 0.0L)) {
      ((int64_t *)result->data)[output_index++] = logical_index;
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

TerlanNdArrayStatus terlan_ndarray_v1_argwhere(
    TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle result = NULL;
  int64_t count = 0;
  int64_t output_shape[2];
  int64_t output_row = 0;
  int64_t logical_index;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = count_nonzero_values(array, &count);
  output_shape[0] = count;
  output_shape[1] = array->rank;
  if (status == TERLAN_NDARRAY_OK) {
    status = create_empty_array(output_shape, 2, TERLAN_NDARRAY_DTYPE_INT64,
                                &result);
  }
  for (logical_index = 0;
       status == TERLAN_NDARRAY_OK && logical_index < array->numel;
       ++logical_index) {
    TerlanNdArrayScalar value;
    status = dtype_load(array_const_element_pointer(array, logical_index),
                        array->dtype, &value);
    if (status == TERLAN_NDARRAY_OK &&
        (value.real != 0.0L || value.imag != 0.0L)) {
      int64_t remaining = logical_index;
      int64_t dimension;
      for (dimension = array->rank; dimension > 0; --dimension) {
        const int64_t axis = dimension - 1;
        ((int64_t *)result->data)[output_row * array->rank + axis] =
            remaining % array->shape[axis];
        remaining /= array->shape[axis];
      }
      ++output_row;
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

TerlanNdArrayStatus terlan_ndarray_v1_nonzero(
    TerlanNdArrayHandle array, uint64_t **out_values, size_t *out_length) {
  TerlanNdArrayStatus status;
  uint64_t *values = NULL;
  int64_t count = 0;
  int64_t coordinate_rank;
  int64_t dimension;
  int64_t logical_index;
  int64_t output_index = 0;
  if (out_values == NULL || out_length == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_values = NULL;
  *out_length = 0;
  status = require_live(array);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  coordinate_rank = array->rank == 0 ? 1 : array->rank;
  if ((uint64_t)coordinate_rank > SIZE_MAX / sizeof(uint64_t)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  status = count_nonzero_values(array, &count);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  values = checked_calloc((size_t)coordinate_rank, sizeof(uint64_t));
  if (values == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  for (dimension = 0; dimension < coordinate_rank; ++dimension) {
    TerlanNdArrayHandle coordinates = NULL;
    status = create_empty_array(&count, 1, TERLAN_NDARRAY_DTYPE_INT64,
                                &coordinates);
    if (status != TERLAN_NDARRAY_OK) {
      break;
    }
    values[dimension] = (uint64_t)(uintptr_t)coordinates;
  }
  for (logical_index = 0;
       status == TERLAN_NDARRAY_OK && logical_index < array->numel;
       ++logical_index) {
    TerlanNdArrayScalar value;
    status = dtype_load(array_const_element_pointer(array, logical_index),
                        array->dtype, &value);
    if (status == TERLAN_NDARRAY_OK &&
        (value.real != 0.0L || value.imag != 0.0L)) {
      if (array->rank == 0) {
        TerlanNdArrayHandle coordinates =
            (TerlanNdArrayHandle)(uintptr_t)values[0];
        ((int64_t *)coordinates->data)[output_index++] = 0;
        continue;
      }
      int64_t remaining = logical_index;
      for (dimension = array->rank; dimension > 0; --dimension) {
        const int64_t axis = dimension - 1;
        TerlanNdArrayHandle coordinates =
            (TerlanNdArrayHandle)(uintptr_t)values[axis];
        ((int64_t *)coordinates->data)[output_index] =
            remaining % array->shape[axis];
        remaining /= array->shape[axis];
      }
      ++output_index;
    }
  }
  if (status != TERLAN_NDARRAY_OK) {
    for (dimension = 0; dimension < coordinate_rank; ++dimension) {
      if (values[dimension] != 0) {
        (void)terlan_ndarray_v1_delete(
            (TerlanNdArrayHandle)(uintptr_t)values[dimension]);
      }
    }
    free(values);
    return status;
  }
  *out_values = values;
  *out_length = (size_t)coordinate_rank;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_item_bool(TerlanNdArrayHandle array,
                                                 bool *out_value) {
  TerlanNdArrayScalar value;
  TerlanNdArrayStatus status;
  if (out_value == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  status = require_live(array);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (array->numel != 1) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  status = dtype_load(array_const_element_pointer(array, 0), array->dtype,
                      &value);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  *out_value = value.real != 0.0L || value.imag != 0.0L;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_item_int(TerlanNdArrayHandle array,
                                                int64_t *out_value) {
  TerlanNdArrayScalar value;
  TerlanNdArrayStatus status;
  if (out_value == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  status = require_live(array);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (array->numel != 1) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  status = dtype_load(array_const_element_pointer(array, 0), array->dtype,
                      &value);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (value.imag != 0.0L) {
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  if (!isfinite(value.real) || value.real < (long double)INT64_MIN ||
      value.real > (long double)INT64_MAX) {
    return TERLAN_NDARRAY_CAST_OVERFLOW;
  }
  *out_value = (int64_t)value.real;
  return TERLAN_NDARRAY_OK;
}

#define TERLAN_NDARRAY_GETTER(name, dtype_value, c_type)                       \
  TerlanNdArrayStatus name(TerlanNdArrayHandle array, const int64_t *indices, \
                           int64_t index_count, c_type *out_value) {           \
    void *pointer = NULL;                                                      \
    TerlanNdArrayStatus status;                                                \
    if (out_value == NULL) {                                                   \
      return TERLAN_NDARRAY_INVALID_ARGUMENT;                                  \
    }                                                                          \
    status = indexed_pointer(array, indices, index_count, &pointer);           \
    if (status != TERLAN_NDARRAY_OK) {                                         \
      return status;                                                           \
    }                                                                          \
    if (array->dtype != dtype_value) {                                         \
      return TERLAN_NDARRAY_DTYPE_MISMATCH;                                    \
    }                                                                          \
    *out_value = *(const c_type *)pointer;                                     \
    return TERLAN_NDARRAY_OK;                                                  \
  }

TERLAN_NDARRAY_GETTER(terlan_ndarray_v1_get_int, TERLAN_NDARRAY_DTYPE_INT64,
                      int64_t)
TERLAN_NDARRAY_GETTER(terlan_ndarray_v1_get_float,
                      TERLAN_NDARRAY_DTYPE_FLOAT64, double)

#define TERLAN_NDARRAY_SETTER(name, dtype_value, c_type, validate)             \
  TerlanNdArrayStatus name(TerlanNdArrayHandle array, const int64_t *indices, \
                           int64_t index_count, c_type value) {                \
    void *pointer = NULL;                                                      \
    TerlanNdArrayStatus status =                                               \
        indexed_pointer(array, indices, index_count, &pointer);                \
    if (status != TERLAN_NDARRAY_OK) {                                         \
      return status;                                                           \
    }                                                                          \
    if (array->dtype != dtype_value) {                                         \
      return TERLAN_NDARRAY_DTYPE_MISMATCH;                                    \
    }                                                                          \
    if (array->writable == 0) {                                                \
      return TERLAN_NDARRAY_READ_ONLY;                                         \
    }                                                                          \
    if (!(validate)) {                                                         \
      return TERLAN_NDARRAY_INVALID_BOOL;                                      \
    }                                                                          \
    *(c_type *)pointer = value;                                                \
    return TERLAN_NDARRAY_OK;                                                  \
  }

TERLAN_NDARRAY_SETTER(terlan_ndarray_v1_set_int, TERLAN_NDARRAY_DTYPE_INT64,
                      int64_t, 1)
TERLAN_NDARRAY_SETTER(terlan_ndarray_v1_set_float,
                      TERLAN_NDARRAY_DTYPE_FLOAT64, double, 1)

TerlanNdArrayStatus terlan_ndarray_v1_get_bool(TerlanNdArrayHandle array,
                                                const int64_t *indices,
                                                int64_t index_count,
                                                bool *out_value) {
  void *pointer = NULL;
  TerlanNdArrayStatus status;
  if (out_value == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  status = indexed_pointer(array, indices, index_count, &pointer);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (array->dtype != TERLAN_NDARRAY_DTYPE_BOOL) {
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  *out_value = *(const uint8_t *)pointer != 0;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_set_bool(TerlanNdArrayHandle array,
                                                const int64_t *indices,
                                                int64_t index_count,
                                                bool value) {
  void *pointer = NULL;
  TerlanNdArrayStatus status =
      indexed_pointer(array, indices, index_count, &pointer);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (array->dtype != TERLAN_NDARRAY_DTYPE_BOOL) {
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  if (array->writable == 0) {
    return TERLAN_NDARRAY_READ_ONLY;
  }
  *(uint8_t *)pointer = value ? 1 : 0;
  return TERLAN_NDARRAY_OK;
}
