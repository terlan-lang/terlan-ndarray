#include "terlan_ndarray_internal.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

TerlanNdArrayStatus terlan_ndarray_v1_resize_in_place(
    TerlanNdArrayHandle array, const int64_t *shape, int64_t rank) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t *new_shape = NULL;
  int64_t *new_strides = NULL;
  int64_t new_numel = 0;
  int64_t new_byte_count = 0;
  void *new_data = NULL;
  size_t dimensions_size;
  size_t copied_bytes;
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (rank < 0 || (rank > 0 && shape == NULL)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if ((uint64_t)rank > SIZE_MAX / sizeof(*new_shape)) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  dimensions_size = (size_t)rank * sizeof(*new_shape);
  if (rank > 0) {
    new_shape = checked_malloc(dimensions_size);
    new_strides = checked_malloc(dimensions_size);
    if (new_shape == NULL || new_strides == NULL) {
      free(new_shape);
      free(new_strides);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    memcpy(new_shape, shape, dimensions_size);
  }
  status = array_checked_layout(new_shape, rank, dtype_size(array->dtype),
                                &new_numel, &new_byte_count, new_strides);
  if (status != TERLAN_NDARRAY_OK) {
    free(new_shape);
    free(new_strides);
    return status;
  }
  if (array->writable == 0) {
    free(new_shape);
    free(new_strides);
    return TERLAN_NDARRAY_READ_ONLY;
  }
  if ((array->numel != 0 || new_numel != 0) &&
      (array->storage_offset != 0 || !array_is_contiguous(array) ||
       array->storage->external_release != NULL ||
       atomic_load_explicit(&array->storage->reference_count,
                            memory_order_acquire) != 1)) {
    free(new_shape);
    free(new_strides);
    return TERLAN_NDARRAY_UNSUPPORTED_LAYOUT;
  }
  if (new_byte_count > 0) {
    new_data = checked_calloc((size_t)new_byte_count, 1);
    if (new_data == NULL) {
      free(new_shape);
      free(new_strides);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    copied_bytes = (size_t)(array->byte_count < new_byte_count
                                ? array->byte_count
                                : new_byte_count);
    if (copied_bytes > 0) {
      memcpy(new_data, array->data, copied_bytes);
    }
  }
  if (array->numel != 0 || new_numel != 0) {
    free(array->storage->data);
    array->storage->data = new_data;
    array->storage->byte_count = new_byte_count;
    array->storage_offset = 0;
    array->data = new_data;
  } else {
    free(new_data);
  }
  free(array->shape);
  free(array->strides);
  array->shape = new_shape;
  array->strides = new_strides;
  array->rank = rank;
  array->numel = new_numel;
  array->byte_count = new_byte_count;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus normalized_axis(int64_t axis, int64_t rank,
                                            int allow_end,
                                            int64_t *out_axis) {
  const int64_t upper = rank + (allow_end != 0 ? 1 : 0);
  if (out_axis == NULL || rank < 0) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= upper) {
    return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
  }
  *out_axis = axis;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus reshape_at_least(TerlanNdArrayHandle array,
                                             int64_t minimum_rank,
                                             TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t shape[3] = {1, 1, 1};
  if (status != TERLAN_NDARRAY_OK || out_array == NULL || minimum_rank < 1 ||
      minimum_rank > 3) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (array->rank >= minimum_rank) {
    return create_view(array, array->shape, array->strides, array->rank, 0,
                       array->writable, out_array);
  }
  if (minimum_rank == 1) {
    shape[0] = array->numel;
  } else if (minimum_rank == 2) {
    shape[0] = 1;
    shape[1] = array->numel;
  } else if (array->rank == 1) {
    shape[1] = array->shape[0];
  } else if (array->rank == 2) {
    shape[0] = array->shape[0];
    shape[1] = array->shape[1];
  }
  return terlan_ndarray_v1_reshape(array, shape, minimum_rank, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_atleast_1d(
    TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array) {
  return reshape_at_least(array, 1, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_atleast_2d(
    TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array) {
  return reshape_at_least(array, 2, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_atleast_3d(
    TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array) {
  return reshape_at_least(array, 3, out_array);
}

static TerlanNdArrayStatus flatten_in_order(TerlanNdArrayHandle array,
                                            int32_t order, int copy,
                                            TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle output = NULL;
  int64_t *axes = NULL;
  const int64_t shape[] = {array == NULL ? 0 : array->numel};
  const int64_t strides[] = {1};
  int64_t ordered_index;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL ||
      (copy != 0 && copy != 1)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = array_normalize_order(array, order, 1, &order);
  if (status == TERLAN_NDARRAY_OK && copy == 0 &&
      array_order_is_contiguous(array, order)) {
    return create_view(array, shape, strides, 1, 0, array->writable,
                       out_array);
  }
  if (array->rank > 0) {
    axes = checked_malloc((size_t)array->rank * sizeof(*axes));
    if (axes == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = array_order_axes(array, order, axes);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = create_empty_array(shape, 1, array->dtype, &output);
  }
  for (ordered_index = 0;
       status == TERLAN_NDARRAY_OK && ordered_index < array->numel;
       ++ordered_index) {
    const int64_t source_logical =
        array_order_logical_index(array, axes, ordered_index);
    memcpy(array_element_pointer(output, ordered_index),
           array_const_element_pointer(array, source_logical),
           dtype_size(array->dtype));
  }
  free(axes);
  if (status != TERLAN_NDARRAY_OK) {
    if (output != NULL) {
      (void)terlan_ndarray_v1_delete(output);
    }
    return status;
  }
  *out_array = output;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_ravel_ordered(
    TerlanNdArrayHandle array, int32_t order,
    TerlanNdArrayHandle *out_array) {
  return flatten_in_order(array, order, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_ravel(
    TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_v1_ravel_ordered(array, TERLAN_NDARRAY_ORDER_C,
                                         out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_flatten_ordered(
    TerlanNdArrayHandle array, int32_t order,
    TerlanNdArrayHandle *out_array) {
  return flatten_in_order(array, order, 1, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_flatten(
    TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_v1_flatten_ordered(array, TERLAN_NDARRAY_ORDER_C,
                                           out_array);
}

static TerlanNdArrayStatus permuted_view(TerlanNdArrayHandle array,
                                         const int64_t *order,
                                         TerlanNdArrayHandle *out_array) {
  int64_t *shape = NULL;
  int64_t *strides = NULL;
  TerlanNdArrayStatus status;
  int64_t axis;
  if (array->rank > 0) {
    shape = checked_malloc((size_t)array->rank * sizeof(*shape));
    strides = checked_malloc((size_t)array->rank * sizeof(*strides));
    if (shape == NULL || strides == NULL) {
      free(strides);
      free(shape);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (axis = 0; axis < array->rank; ++axis) {
    shape[axis] = array->shape[order[axis]];
    strides[axis] = array->strides[order[axis]];
  }
  status = create_view(array, shape, strides, array->rank, 0, array->writable,
                       out_array);
  free(strides);
  free(shape);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_moveaxis(
    TerlanNdArrayHandle array, const int64_t *sources, int64_t source_count,
    const int64_t *destinations, int64_t destination_count,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t *source_axes = NULL;
  int64_t *destination_axes = NULL;
  int64_t *order = NULL;
  uint8_t *selected = NULL;
  int64_t output_count = 0;
  int64_t axis;
  if (out_array == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (status != TERLAN_NDARRAY_OK || source_count < 0 ||
      source_count != destination_count ||
      (uint64_t)source_count > SIZE_MAX / sizeof(*source_axes) ||
      (source_count > 0 && (sources == NULL || destinations == NULL))) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (array->rank > 0) {
    order = checked_malloc((size_t)array->rank * sizeof(*order));
    selected = checked_calloc((size_t)array->rank, 1);
  }
  if (source_count > 0) {
    source_axes = checked_malloc((size_t)source_count * sizeof(*source_axes));
    destination_axes =
        checked_malloc((size_t)source_count * sizeof(*destination_axes));
  }
  if ((array->rank > 0 && (order == NULL || selected == NULL)) ||
      (source_count > 0 &&
       (source_axes == NULL || destination_axes == NULL))) {
    status = TERLAN_NDARRAY_ALLOCATION_FAILED;
    goto cleanup;
  }
  for (axis = 0; axis < source_count; ++axis) {
    status = normalized_axis(sources[axis], array->rank, 0,
                             &source_axes[axis]);
    if (status == TERLAN_NDARRAY_OK) {
      status = normalized_axis(destinations[axis], array->rank, 0,
                               &destination_axes[axis]);
    }
    if (status != TERLAN_NDARRAY_OK) {
      goto cleanup;
    }
    if (selected[source_axes[axis]] != 0) {
      status = TERLAN_NDARRAY_DUPLICATE_AXIS;
      goto cleanup;
    }
    selected[source_axes[axis]] = 1;
  }
  for (axis = 0; axis < source_count; ++axis) {
    int64_t other;
    for (other = axis + 1; other < source_count; ++other) {
      if (destination_axes[axis] == destination_axes[other]) {
        status = TERLAN_NDARRAY_DUPLICATE_AXIS;
        goto cleanup;
      }
    }
  }
  for (axis = 0; axis < array->rank; ++axis) {
    if (selected[axis] == 0) {
      order[output_count++] = axis;
    }
  }
  while (output_count < array->rank) {
    int64_t best = -1;
    int64_t best_destination = array->rank;
    int64_t index;
    for (index = 0; index < source_count; ++index) {
      if (source_axes[index] >= 0 &&
          destination_axes[index] < best_destination) {
        best = index;
        best_destination = destination_axes[index];
      }
    }
    for (index = output_count; index > best_destination; --index) {
      order[index] = order[index - 1];
    }
    order[best_destination] = source_axes[best];
    source_axes[best] = -1;
    ++output_count;
  }
  status = permuted_view(array, order, out_array);

cleanup:
  free(selected);
  free(order);
  free(destination_axes);
  free(source_axes);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_rollaxis(
    TerlanNdArrayHandle array, int64_t axis, int64_t start,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t normalized;
  int64_t insertion;
  int64_t source[1];
  int64_t destination[1];
  if (out_array == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  status = normalized_axis(axis, array->rank, 0, &normalized);
  if (status == TERLAN_NDARRAY_OK) {
    status = normalized_axis(start, array->rank, 1, &insertion);
  }
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (normalized < insertion) {
    --insertion;
  }
  source[0] = normalized;
  destination[0] = insertion;
  return terlan_ndarray_v1_moveaxis(array, source, 1, destination, 1,
                                    out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_flipud(
    TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array) {
  const int64_t axis = 0;
  return terlan_ndarray_v1_flip(array, &axis, 1, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_fliplr(
    TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array) {
  const int64_t axis = 1;
  return terlan_ndarray_v1_flip(array, &axis, 1, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_rot90(
    TerlanNdArrayHandle array, int64_t rotations, int64_t axis_a,
    int64_t axis_b, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle intermediate = NULL;
  int64_t normalized_a;
  int64_t normalized_b;
  int64_t selected[2];
  int64_t quarter_turns = rotations % 4;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (quarter_turns < 0) {
    quarter_turns += 4;
  }
  status = normalized_axis(axis_a, array->rank, 0, &normalized_a);
  if (status == TERLAN_NDARRAY_OK) {
    status = normalized_axis(axis_b, array->rank, 0, &normalized_b);
  }
  if (status != TERLAN_NDARRAY_OK || normalized_a == normalized_b) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_DUPLICATE_AXIS;
  }
  if (quarter_turns == 0) {
    return create_view(array, array->shape, array->strides, array->rank, 0,
                       array->writable, out_array);
  }
  if (quarter_turns == 2) {
    selected[0] = normalized_a;
    selected[1] = normalized_b;
    return terlan_ndarray_v1_flip(array, selected, 2, out_array);
  }
  status = terlan_ndarray_v1_transpose(array, normalized_a, normalized_b,
                                       &intermediate);
  if (status == TERLAN_NDARRAY_OK) {
    selected[0] = quarter_turns == 1 ? normalized_a : normalized_b;
    status = terlan_ndarray_v1_flip(intermediate, selected, 1, out_array);
  }
  if (intermediate != NULL) {
    (void)terlan_ndarray_v1_delete(intermediate);
  }
  return status;
}
