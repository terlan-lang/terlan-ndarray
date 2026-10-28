#include "terlan_ndarray_internal.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static uint64_t stride_magnitude(int64_t stride) {
  return stride < 0 ? (uint64_t)(-(stride + 1)) + 1U : (uint64_t)stride;
}

TerlanNdArrayStatus array_normalize_order(TerlanNdArrayHandle array,
                                          int32_t order, int allow_keep,
                                          int32_t *out_order) {
  TerlanNdArrayStatus status = require_live(array);
  if (status != TERLAN_NDARRAY_OK || out_order == NULL ||
      order < TERLAN_NDARRAY_ORDER_C || order > TERLAN_NDARRAY_ORDER_ANY ||
      (allow_keep == 0 && order == TERLAN_NDARRAY_ORDER_KEEP)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (order == TERLAN_NDARRAY_ORDER_ANY) {
    order = array_is_fortran_contiguous(array) && !array_is_contiguous(array)
                ? TERLAN_NDARRAY_ORDER_FORTRAN
                : TERLAN_NDARRAY_ORDER_C;
  }
  *out_order = order;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus array_order_axes(TerlanNdArrayHandle array, int32_t order,
                                     int64_t *out_axes) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t position;
  if (status != TERLAN_NDARRAY_OK ||
      (array->rank > 0 && out_axes == NULL) ||
      order < TERLAN_NDARRAY_ORDER_C || order > TERLAN_NDARRAY_ORDER_KEEP) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  for (position = 0; position < array->rank; ++position) {
    out_axes[position] = order == TERLAN_NDARRAY_ORDER_C
                             ? array->rank - position - 1
                             : position;
  }
  if (order == TERLAN_NDARRAY_ORDER_KEEP) {
    for (position = 1; position < array->rank; ++position) {
      const int64_t candidate = out_axes[position];
      int64_t insertion = position;
      while (insertion > 0 &&
             (stride_magnitude(array->strides[candidate]) <
                  stride_magnitude(array->strides[out_axes[insertion - 1]]) ||
              (stride_magnitude(array->strides[candidate]) ==
                   stride_magnitude(array->strides[out_axes[insertion - 1]]) &&
               candidate < out_axes[insertion - 1]))) {
        out_axes[insertion] = out_axes[insertion - 1];
        --insertion;
      }
      out_axes[insertion] = candidate;
    }
  }
  return TERLAN_NDARRAY_OK;
}

int64_t array_order_logical_index(TerlanNdArrayHandle array,
                                  const int64_t *axes,
                                  int64_t ordered_index) {
  int64_t logical_index = 0;
  int64_t position;
  for (position = 0; position < array->rank; ++position) {
    const int64_t axis = axes[position];
    const int64_t coordinate = ordered_index % array->shape[axis];
    int64_t multiplier = 1;
    int64_t trailing;
    ordered_index /= array->shape[axis];
    for (trailing = axis + 1; trailing < array->rank; ++trailing) {
      multiplier *= array->shape[trailing];
    }
    logical_index += coordinate * multiplier;
  }
  return logical_index;
}

int array_order_is_contiguous(TerlanNdArrayHandle array, int32_t order) {
  int64_t *axes = NULL;
  int64_t expected = 1;
  int64_t position;
  if (array == NULL || array->numel == 0) {
    return 1;
  }
  if (order == TERLAN_NDARRAY_ORDER_C) {
    return array_is_contiguous(array);
  }
  if (order == TERLAN_NDARRAY_ORDER_FORTRAN) {
    return array_is_fortran_contiguous(array);
  }
  if (order != TERLAN_NDARRAY_ORDER_KEEP) {
    return 0;
  }
  if (array->rank > 0) {
    axes = checked_malloc((size_t)array->rank * sizeof(*axes));
    if (axes == NULL ||
        array_order_axes(array, TERLAN_NDARRAY_ORDER_KEEP, axes) !=
            TERLAN_NDARRAY_OK) {
      free(axes);
      return 0;
    }
  }
  for (position = 0; position < array->rank; ++position) {
    const int64_t axis = axes[position];
    if (array->shape[axis] > 1 &&
        (array->strides[axis] != expected || array->strides[axis] < 0)) {
      free(axes);
      return 0;
    }
    if (array->shape[axis] > 1) {
      expected *= array->shape[axis];
    }
  }
  free(axes);
  return 1;
}

TerlanNdArrayStatus array_ordered_strides(TerlanNdArrayHandle source,
                                          int32_t order,
                                          int64_t *out_strides) {
  int64_t stride = 1;
  int64_t position;
  TerlanNdArrayStatus status =
      array_normalize_order(source, order, 1, &order);
  if (status != TERLAN_NDARRAY_OK ||
      (source->rank > 0 && out_strides == NULL)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (source->rank == 0) {
    return TERLAN_NDARRAY_OK;
  }
  if (order == TERLAN_NDARRAY_ORDER_C ||
      order == TERLAN_NDARRAY_ORDER_FORTRAN) {
    for (position = 0; position < source->rank; ++position) {
      const int64_t axis = order == TERLAN_NDARRAY_ORDER_FORTRAN
                               ? position
                               : source->rank - position - 1;
      out_strides[axis] = stride;
      if (source->shape[axis] > 1) {
        stride *= source->shape[axis];
      }
    }
    return TERLAN_NDARRAY_OK;
  }

  {
    uint8_t *selected = checked_calloc((size_t)source->rank, sizeof(*selected));
    if (selected == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    for (position = 0; position < source->rank; ++position) {
      int64_t candidate = -1;
      int64_t axis;
      for (axis = 0; axis < source->rank; ++axis) {
        if (selected[axis] != 0 || source->shape[axis] <= 1) {
          continue;
        }
        if (candidate < 0 ||
            stride_magnitude(source->strides[axis]) <
                stride_magnitude(source->strides[candidate])) {
          candidate = axis;
        }
      }
      if (candidate < 0) {
        break;
      }
      selected[candidate] = 1;
      out_strides[candidate] = stride;
      stride *= source->shape[candidate];
    }
    for (position = 0; position < source->rank; ++position) {
      if (selected[position] == 0) {
        out_strides[position] = stride;
      }
    }
    free(selected);
  }
  return TERLAN_NDARRAY_OK;
}
