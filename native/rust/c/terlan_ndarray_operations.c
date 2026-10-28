#include "terlan_ndarray_internal.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

static TerlanNdArrayStatus reshape_layout(const int64_t *shape, int64_t rank,
                                          int32_t order, int64_t *strides,
                                          int64_t *out_count) {
  int64_t count = 1;
  int64_t position;
  if (rank < 0 || (rank > 0 && (shape == NULL || strides == NULL)) ||
      out_count == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  for (position = 0; position < rank; ++position) {
    const int64_t axis = order == TERLAN_NDARRAY_ORDER_FORTRAN
                             ? position
                             : rank - position - 1;
    if (shape[axis] < 0) {
      return TERLAN_NDARRAY_NEGATIVE_DIMENSION;
    }
    strides[axis] = count;
    if (shape[axis] != 0 && count > INT64_MAX / shape[axis]) {
      return TERLAN_NDARRAY_SIZE_OVERFLOW;
    }
    count *= shape[axis];
  }
  *out_count = count;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus reshape_vector_layout(
    TerlanNdArrayHandle array, const int64_t *shape, int64_t rank,
    int32_t order, int64_t *strides) {
  int64_t stride = array->strides[0];
  int64_t position;
  for (position = 0; position < rank; ++position) {
    const int64_t axis = order == TERLAN_NDARRAY_ORDER_FORTRAN
                             ? position
                             : rank - position - 1;
    strides[axis] = stride;
    if (shape[axis] != 0 && stride != 0) {
      if (stride > 0 && stride > INT64_MAX / shape[axis]) {
        return TERLAN_NDARRAY_SIZE_OVERFLOW;
      }
      if (stride < 0 && stride < INT64_MIN / shape[axis]) {
        return TERLAN_NDARRAY_SIZE_OVERFLOW;
      }
      stride *= shape[axis];
    }
  }
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_reshape_ordered(
    TerlanNdArrayHandle array, const int64_t *shape, int64_t rank,
    int32_t order, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle output = NULL;
  int64_t *strides = NULL;
  int64_t *source_axes = NULL;
  int64_t *output_axes = NULL;
  int64_t count = 0;
  int64_t ordered_index;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL || rank < 0 ||
      (rank > 0 && shape == NULL)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = array_normalize_order(array, order, 0, &order);
  if (rank > 0) {
    strides = checked_malloc((size_t)rank * sizeof(*strides));
    output_axes = checked_malloc((size_t)rank * sizeof(*output_axes));
  }
  if (array->rank > 0) {
    source_axes =
        checked_malloc((size_t)array->rank * sizeof(*source_axes));
  }
  if (status == TERLAN_NDARRAY_OK &&
      ((rank > 0 && (strides == NULL || output_axes == NULL)) ||
       (array->rank > 0 && source_axes == NULL))) {
    status = TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = reshape_layout(shape, rank, order, strides, &count);
  }
  if (status == TERLAN_NDARRAY_OK && count != array->numel) {
    status = TERLAN_NDARRAY_ELEMENT_COUNT_MISMATCH;
  }
  if (status == TERLAN_NDARRAY_OK && array->rank == 1 && rank > 0) {
    status = reshape_vector_layout(array, shape, rank, order, strides);
    if (status == TERLAN_NDARRAY_OK) {
      status = create_view(array, shape, strides, rank, 0, array->writable,
                           out_array);
    }
    goto cleanup;
  }
  if (status == TERLAN_NDARRAY_OK &&
      array_order_is_contiguous(array, order)) {
    status = create_view(array, shape, strides, rank, 0, array->writable,
                         out_array);
    goto cleanup;
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = create_empty_array(shape, rank, array->dtype, &output);
  }
  if (status == TERLAN_NDARRAY_OK && rank > 0) {
    memcpy(output->strides, strides,
           (size_t)rank * sizeof(*output->strides));
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = array_order_axes(array, order, source_axes);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = array_order_axes(output, order, output_axes);
  }
  for (ordered_index = 0;
       status == TERLAN_NDARRAY_OK && ordered_index < array->numel;
       ++ordered_index) {
    const int64_t source_logical =
        array_order_logical_index(array, source_axes, ordered_index);
    const int64_t output_logical =
        array_order_logical_index(output, output_axes, ordered_index);
    memcpy(array_element_pointer(output, output_logical),
           array_const_element_pointer(array, source_logical),
           dtype_size(array->dtype));
  }
  if (status == TERLAN_NDARRAY_OK) {
    *out_array = output;
    output = NULL;
  }

cleanup:
  if (output != NULL) {
    (void)terlan_ndarray_v1_delete(output);
  }
  free(output_axes);
  free(source_axes);
  free(strides);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_reshape(
    TerlanNdArrayHandle array, const int64_t *shape, int64_t rank,
    TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_v1_reshape_ordered(
      array, shape, rank, TERLAN_NDARRAY_ORDER_C, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_transpose(
    TerlanNdArrayHandle array, int64_t axis_a, int64_t axis_b,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t normalized_a = 0;
  int64_t normalized_b = 0;
  int64_t *shape = NULL;
  int64_t *strides = NULL;
  size_t dimensions_size;

  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (array->rank == 0) {
    if (axis_a != 0 || axis_b != 0) {
      return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
    }
    return create_view(array, NULL, NULL, 0, 0, array->writable, out_array);
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
  strides = checked_malloc(dimensions_size);
  if (shape == NULL || strides == NULL) {
    free(strides);
    free(shape);
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  memcpy(shape, array->shape, dimensions_size);
  memcpy(strides, array->strides, dimensions_size);
  shape[normalized_a] = array->shape[normalized_b];
  shape[normalized_b] = array->shape[normalized_a];
  strides[normalized_a] = array->strides[normalized_b];
  strides[normalized_b] = array->strides[normalized_a];
  status = create_view(array, shape, strides, array->rank, 0, array->writable,
                       out_array);
  free(strides);
  free(shape);
  return status;
}
