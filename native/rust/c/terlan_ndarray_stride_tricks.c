#include "terlan_ndarray_internal.h"

#include <limits.h>
#include <stdlib.h>

TerlanNdArrayStatus terlan_ndarray_v1_as_strided(
    TerlanNdArrayHandle array, const int64_t *shape, const int64_t *strides,
    int64_t shape_count, int64_t stride_count,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  if (status != TERLAN_NDARRAY_OK || out_array == NULL || shape_count < 0 ||
      stride_count < 0 || shape_count != stride_count ||
      (shape_count > 0 && (shape == NULL || strides == NULL))) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  return create_view(array, shape, strides, shape_count, 0, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_sliding_window_view(
    TerlanNdArrayHandle array, const int64_t *window_shape,
    int64_t window_count, const int64_t *axes, int64_t axis_count,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t output_rank;
  int64_t *output_shape = NULL;
  int64_t *output_strides = NULL;
  int64_t index;

  if (status != TERLAN_NDARRAY_OK || out_array == NULL || window_count < 0 ||
      axis_count < 0 || (window_count > 0 && window_shape == NULL) ||
      (axis_count > 0 && axes == NULL)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if ((axis_count == 0 && window_count != array->rank) ||
      (axis_count > 0 && window_count != axis_count)) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  if (array->rank > INT64_MAX - window_count) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  output_rank = array->rank + window_count;
  if ((uint64_t)output_rank > SIZE_MAX / sizeof(*output_shape)) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  if (output_rank > 0) {
    output_shape = checked_malloc((size_t)output_rank * sizeof(*output_shape));
    output_strides =
        checked_malloc((size_t)output_rank * sizeof(*output_strides));
    if (output_shape == NULL || output_strides == NULL) {
      free(output_shape);
      free(output_strides);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (index = 0; index < array->rank; ++index) {
    output_shape[index] = array->shape[index];
    output_strides[index] = array->strides[index];
  }
  for (index = 0; index < window_count; ++index) {
    const int64_t window = window_shape[index];
    int64_t axis = axis_count == 0 ? index : axes[index];
    if (axis < 0) {
      axis += array->rank;
    }
    if (axis < 0 || axis >= array->rank) {
      status = TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
      goto cleanup;
    }
    if (window < 0) {
      status = TERLAN_NDARRAY_NEGATIVE_DIMENSION;
      goto cleanup;
    }
    if (window > output_shape[axis]) {
      status = TERLAN_NDARRAY_SHAPE_MISMATCH;
      goto cleanup;
    }
    if (window == 0) {
      if (output_shape[axis] == INT64_MAX) {
        status = TERLAN_NDARRAY_SIZE_OVERFLOW;
        goto cleanup;
      }
      output_shape[axis] += 1;
    } else {
      output_shape[axis] -= window - 1;
    }
    output_shape[array->rank + index] = window;
    output_strides[array->rank + index] = array->strides[axis];
  }
  status = create_view(array, output_shape, output_strides, output_rank, 0, 0,
                       out_array);

cleanup:
  free(output_shape);
  free(output_strides);
  return status;
}
