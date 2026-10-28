#include "terlan_ndarray_internal.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static TerlanNdArrayStatus normalize_axis(int64_t axis, int64_t rank,
                                          int allow_end,
                                          int64_t *out_axis) {
  const int64_t upper = rank + (allow_end != 0 ? 1 : 0);
  if (axis < 0) {
    axis += upper;
  }
  if (axis < 0 || axis >= upper || out_axis == NULL) {
    return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
  }
  *out_axis = axis;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_expand_dims(
    TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t normalized;
  int64_t *shape = NULL;
  int64_t *strides = NULL;
  int64_t output_axis;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = normalize_axis(axis, array->rank, 1, &normalized);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  shape = checked_malloc((size_t)(array->rank + 1) * sizeof(*shape));
  strides = checked_malloc((size_t)(array->rank + 1) * sizeof(*strides));
  if (shape == NULL || strides == NULL) {
    free(strides);
    free(shape);
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  for (output_axis = 0; output_axis < array->rank + 1; ++output_axis) {
    if (output_axis == normalized) {
      shape[output_axis] = 1;
      strides[output_axis] =
          output_axis >= array->rank
              ? 1
              : array->strides[output_axis] * array->shape[output_axis];
    } else {
      const int64_t source_axis = output_axis > normalized ? output_axis - 1
                                                            : output_axis;
      shape[output_axis] = array->shape[source_axis];
      strides[output_axis] = array->strides[source_axis];
    }
  }
  status = create_view(array, shape, strides, array->rank + 1, 0,
                       array->writable, out_array);
  free(strides);
  free(shape);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_squeeze(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  uint8_t *remove = NULL;
  int64_t remove_count = 0;
  int64_t *shape = NULL;
  int64_t *strides = NULL;
  int64_t axis;
  int64_t output_axis = 0;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL || axis_count < 0 ||
      (axis_count > 0 && axes == NULL)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (array->rank > 0) {
    remove = checked_calloc((size_t)array->rank, 1);
    if (remove == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  if (axis_count == 0) {
    for (axis = 0; axis < array->rank; ++axis) {
      if (array->shape[axis] == 1) {
        remove[axis] = 1;
        ++remove_count;
      }
    }
  } else {
    for (axis = 0; axis < axis_count; ++axis) {
      int64_t normalized;
      int duplicate;
      int singleton;
      status = normalize_axis(axes[axis], array->rank, 0, &normalized);
      if (status != TERLAN_NDARRAY_OK) {
        free(remove);
        return status;
      }
      duplicate = remove[normalized] != 0;
      singleton = array->shape[normalized] == 1;
      if (duplicate || !singleton) {
        free(remove);
        return duplicate ? TERLAN_NDARRAY_DUPLICATE_AXIS
                         : TERLAN_NDARRAY_SHAPE_MISMATCH;
      }
      remove[normalized] = 1;
      ++remove_count;
    }
  }
  if (array->rank - remove_count > 0) {
    shape = checked_malloc((size_t)(array->rank - remove_count) *
                           sizeof(*shape));
    strides = checked_malloc((size_t)(array->rank - remove_count) *
                             sizeof(*strides));
    if (shape == NULL || strides == NULL) {
      free(strides);
      free(shape);
      free(remove);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (axis = 0; axis < array->rank; ++axis) {
    if (remove[axis] == 0) {
      shape[output_axis] = array->shape[axis];
      strides[output_axis++] = array->strides[axis];
    }
  }
  status = create_view(array, shape, strides, array->rank - remove_count, 0,
                       array->writable, out_array);
  free(strides);
  free(shape);
  free(remove);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_flip(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t *strides = NULL;
  uint8_t *seen = NULL;
  int64_t relative_offset = 0;
  int64_t index;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL || axis_count < 0 ||
      (axis_count > 0 && axes == NULL)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (array->rank > 0) {
    strides = checked_malloc((size_t)array->rank * sizeof(*strides));
    seen = checked_calloc((size_t)array->rank, 1);
    if (strides == NULL || seen == NULL) {
      free(seen);
      free(strides);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    memcpy(strides, array->strides,
           (size_t)array->rank * sizeof(*strides));
  }
  for (index = 0; index < axis_count; ++index) {
    int64_t axis = axes[index];
    status = normalize_axis(axis, array->rank, 0, &axis);
    if (status != TERLAN_NDARRAY_OK || seen[axis] != 0) {
      free(seen);
      free(strides);
      return status != TERLAN_NDARRAY_OK ? status
                                         : TERLAN_NDARRAY_DUPLICATE_AXIS;
    }
    seen[axis] = 1;
    relative_offset += (array->shape[axis] - 1) * strides[axis];
    strides[axis] = -strides[axis];
  }
  status = create_view(array, array->shape, strides, array->rank,
                       relative_offset, array->writable, out_array);
  free(seen);
  free(strides);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_flip_all(
    TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t *axes = NULL;
  int64_t axis;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (array->rank > 0) {
    axes = checked_malloc((size_t)array->rank * sizeof(*axes));
    if (axes == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (axis = 0; axis < array->rank; ++axis) {
    axes[axis] = axis;
  }
  status = terlan_ndarray_v1_flip(array, axes, array->rank, out_array);
  free(axes);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_roll(
    TerlanNdArrayHandle array, int64_t shift, int64_t axis,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle result = NULL;
  int64_t normalized;
  int64_t output_index;
  const size_t element_size =
      status == TERLAN_NDARRAY_OK ? dtype_size(array->dtype) : 0;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = normalize_axis(axis, array->rank, 0, &normalized);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  status = create_empty_array(array->shape, array->rank, array->dtype, &result);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (array->shape[normalized] > 0) {
    shift %= array->shape[normalized];
    if (shift < 0) {
      shift += array->shape[normalized];
    }
  }
  for (output_index = 0; output_index < array->numel; ++output_index) {
    int64_t remaining = output_index;
    int64_t input_logical = 0;
    int64_t canonical_stride = 1;
    int64_t input_coordinates[64];
    int64_t dimension;
    if (array->rank > 64) {
      (void)terlan_ndarray_v1_delete(result);
      return TERLAN_NDARRAY_UNSUPPORTED_LAYOUT;
    }
    for (dimension = array->rank; dimension > 0; --dimension) {
      const int64_t index = dimension - 1;
      input_coordinates[index] = remaining % array->shape[index];
      remaining /= array->shape[index];
    }
    input_coordinates[normalized] =
        (input_coordinates[normalized] - shift + array->shape[normalized]) %
        array->shape[normalized];
    for (dimension = array->rank; dimension > 0; --dimension) {
      const int64_t index = dimension - 1;
      input_logical += input_coordinates[index] * canonical_stride;
      canonical_stride *= array->shape[index];
    }
    memcpy(array_element_pointer(result, output_index),
           array_const_element_pointer(array, input_logical), element_size);
  }
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_roll_flat(
    TerlanNdArrayHandle array, int64_t shift,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle flattened = NULL;
  TerlanNdArrayHandle rolled = NULL;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = terlan_ndarray_v1_ravel(array, &flattened);
  if (status == TERLAN_NDARRAY_OK) {
    status = terlan_ndarray_v1_roll(flattened, shift, 0, &rolled);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = terlan_ndarray_v1_reshape(rolled, array->shape, array->rank,
                                      out_array);
  }
  if (rolled != NULL) {
    (void)terlan_ndarray_v1_delete(rolled);
  }
  if (flattened != NULL) {
    (void)terlan_ndarray_v1_delete(flattened);
  }
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_roll_axes(
    TerlanNdArrayHandle array, const int64_t *shifts, int64_t shift_count,
    const int64_t *axes, int64_t axis_count,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle current = NULL;
  int64_t index;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL || shift_count < 0 ||
      axis_count < 0 || (shift_count > 0 && shifts == NULL) ||
      (axis_count > 0 && axes == NULL) ||
      (axis_count > 0 && shift_count != 1 && shift_count != axis_count) ||
      (axis_count == 0 && shift_count > 1)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = copy_array(array, &current);
  for (index = 0; status == TERLAN_NDARRAY_OK && index < axis_count; ++index) {
    TerlanNdArrayHandle next = NULL;
    const int64_t shift = shifts[shift_count == 1 ? 0 : index];
    status = terlan_ndarray_v1_roll(current, shift, axes[index], &next);
    (void)terlan_ndarray_v1_delete(current);
    current = next;
  }
  if (status != TERLAN_NDARRAY_OK) {
    if (current != NULL) {
      (void)terlan_ndarray_v1_delete(current);
    }
    return status;
  }
  *out_array = current;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_roll_axes_by(
    TerlanNdArrayHandle array, int64_t shift, const int64_t *axes,
    int64_t axis_count, TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_v1_roll_axes(array, &shift, 1, axes, axis_count,
                                    out_array);
}

static TerlanNdArrayStatus repeat_counts_impl(
    TerlanNdArrayHandle array, const int64_t *repeats, int64_t repeat_count,
    int64_t axis, int flatten, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t normalized = 0;
  int64_t source_axis_size;
  int64_t output_axis_size = 0;
  int64_t *shape = NULL;
  TerlanNdArrayHandle result = NULL;
  int64_t outer = 1;
  int64_t inner = 1;
  int64_t outer_index;
  int64_t axis_index;
  int64_t dimension;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL || repeat_count < 0 ||
      (repeat_count > 0 && repeats == NULL)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (flatten) {
    source_axis_size = array->numel;
  } else {
    status = normalize_axis(axis, array->rank, 0, &normalized);
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
    source_axis_size = array->shape[normalized];
    for (dimension = 0; dimension < normalized; ++dimension) {
      outer *= array->shape[dimension];
    }
    for (dimension = normalized + 1; dimension < array->rank; ++dimension) {
      inner *= array->shape[dimension];
    }
  }
  if (repeat_count != 1 && repeat_count != source_axis_size) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  for (axis_index = 0; axis_index < source_axis_size; ++axis_index) {
    const int64_t count = repeats[repeat_count == 1 ? 0 : axis_index];
    if (count < 0) {
      return TERLAN_NDARRAY_INVALID_ARGUMENT;
    }
    if (output_axis_size > INT64_MAX - count) {
      return TERLAN_NDARRAY_SIZE_OVERFLOW;
    }
    output_axis_size += count;
  }
  shape = checked_malloc((size_t)(flatten ? 1 : array->rank) * sizeof(*shape));
  if (shape == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  if (flatten) {
    shape[0] = output_axis_size;
  } else {
    memcpy(shape, array->shape, (size_t)array->rank * sizeof(*shape));
    shape[normalized] = output_axis_size;
  }
  status = create_empty_array(shape, flatten ? 1 : array->rank, array->dtype,
                              &result);
  free(shape);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  {
    int64_t output_index = 0;
    for (outer_index = 0; outer_index < outer; ++outer_index) {
      for (axis_index = 0; axis_index < source_axis_size; ++axis_index) {
        const int64_t count = repeats[repeat_count == 1 ? 0 : axis_index];
        int64_t repetition;
        for (repetition = 0; repetition < count; ++repetition) {
          int64_t inner_index;
          for (inner_index = 0; inner_index < inner; ++inner_index) {
            const int64_t source_logical =
                (outer_index * source_axis_size + axis_index) * inner +
                inner_index;
            memcpy(array_element_pointer(result, output_index++),
                   array_const_element_pointer(array, source_logical),
                   dtype_size(array->dtype));
          }
        }
      }
    }
  }
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_repeat(
    TerlanNdArrayHandle array, int64_t repeats, int64_t axis,
    TerlanNdArrayHandle *out_array) {
  return repeat_counts_impl(array, &repeats, 1, axis, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_repeat_counts(
    TerlanNdArrayHandle array, const int64_t *repeats, int64_t repeat_count,
    int64_t axis, TerlanNdArrayHandle *out_array) {
  return repeat_counts_impl(array, repeats, repeat_count, axis, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_repeat_flat(
    TerlanNdArrayHandle array, const int64_t *repeats, int64_t repeat_count,
    TerlanNdArrayHandle *out_array) {
  return repeat_counts_impl(array, repeats, repeat_count, 0, 1, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_tile(
    TerlanNdArrayHandle array, const int64_t *repetitions,
    int64_t repetition_count, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t output_rank;
  int64_t *shape = NULL;
  TerlanNdArrayHandle result = NULL;
  int64_t output_index;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL ||
      repetition_count < 0 ||
      (repetition_count > 0 && repetitions == NULL)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  output_rank = array->rank > repetition_count ? array->rank : repetition_count;
  if (output_rank > 0) {
    int64_t axis;
    shape = checked_malloc((size_t)output_rank * sizeof(*shape));
    if (shape == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    for (axis = 0; axis < output_rank; ++axis) {
      const int64_t source_axis = axis - (output_rank - array->rank);
      const int64_t repeat_axis = axis - (output_rank - repetition_count);
      const int64_t source_size =
          source_axis < 0 ? 1 : array->shape[source_axis];
      const int64_t repeat = repeat_axis < 0 ? 1 : repetitions[repeat_axis];
      if (repeat < 0 || (repeat != 0 && source_size > INT64_MAX / repeat)) {
        free(shape);
        return repeat < 0 ? TERLAN_NDARRAY_INVALID_ARGUMENT
                          : TERLAN_NDARRAY_SIZE_OVERFLOW;
      }
      shape[axis] = source_size * repeat;
    }
  }
  status = create_empty_array(shape, output_rank, array->dtype, &result);
  free(shape);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (output_index = 0; output_index < result->numel; ++output_index) {
    int64_t remaining = output_index;
    int64_t source_logical = 0;
    int64_t source_stride = 1;
    int64_t axis;
    for (axis = output_rank; axis > 0; --axis) {
      const int64_t output_axis = axis - 1;
      const int64_t source_axis = output_axis - (output_rank - array->rank);
      const int64_t coordinate = remaining % result->shape[output_axis];
      remaining /= result->shape[output_axis];
      if (source_axis >= 0) {
        source_logical +=
            (coordinate % array->shape[source_axis]) * source_stride;
        source_stride *= array->shape[source_axis];
      }
    }
    memcpy(array_element_pointer(result, output_index),
           array_const_element_pointer(array, source_logical),
           dtype_size(array->dtype));
  }
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}

static int shapes_match_except(TerlanNdArrayHandle lhs,
                               TerlanNdArrayHandle rhs, int64_t axis) {
  int64_t index;
  if (lhs->rank != rhs->rank) {
    return 0;
  }
  for (index = 0; index < lhs->rank; ++index) {
    if (index != axis && lhs->shape[index] != rhs->shape[index]) {
      return 0;
    }
  }
  return 1;
}

TerlanNdArrayStatus terlan_ndarray_concatenate_pair(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, int64_t axis,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(lhs);
  int64_t normalized;
  int64_t *shape = NULL;
  TerlanNdArrayHandle result = NULL;
  int32_t output_dtype;
  int64_t output_index;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(rhs);
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = normalize_axis(axis, lhs->rank, 0, &normalized);
  if (status != TERLAN_NDARRAY_OK || !shapes_match_except(lhs, rhs, normalized)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  if (lhs->shape[normalized] > INT64_MAX - rhs->shape[normalized]) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  output_dtype = dtype_promote(lhs->dtype, rhs->dtype);
  if (output_dtype == 0) {
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  shape = checked_malloc((size_t)lhs->rank * sizeof(*shape));
  if (shape == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  memcpy(shape, lhs->shape, (size_t)lhs->rank * sizeof(*shape));
  shape[normalized] += rhs->shape[normalized];
  status = create_empty_array(shape, lhs->rank, output_dtype, &result);
  free(shape);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (output_index = 0; output_index < result->numel; ++output_index) {
    int64_t remaining = output_index;
    int64_t source_logical = 0;
    int64_t stride = 1;
    int64_t dimension;
    TerlanNdArrayHandle source = lhs;
    for (dimension = result->rank; dimension > 0; --dimension) {
      const int64_t index = dimension - 1;
      int64_t coordinate = remaining % result->shape[index];
      remaining /= result->shape[index];
      if (index == normalized && coordinate >= lhs->shape[index]) {
        source = rhs;
        coordinate -= lhs->shape[index];
      }
      source_logical += coordinate * stride;
      stride *= source->shape[index];
    }
    TerlanNdArrayScalar value;
    status = dtype_load(array_const_element_pointer(source, source_logical),
                        source->dtype, &value);
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_store(array_element_pointer(result, output_index),
                           output_dtype, value);
    }
    if (status != TERLAN_NDARRAY_OK) {
      (void)terlan_ndarray_v1_delete(result);
      return status;
    }
  }
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}

static void release_split_values(uint64_t *values, size_t count) {
  size_t index;
  if (values == NULL) {
    return;
  }
  for (index = 0; index < count; ++index) {
    if (values[index] != 0) {
      (void)terlan_ndarray_v1_delete(
          (TerlanNdArrayHandle)(uintptr_t)values[index]);
    }
  }
  free(values);
}

void terlan_ndarray_v1_delete_handle_values(uint64_t *values) { free(values); }

static TerlanNdArrayStatus create_split_views(
    TerlanNdArrayHandle array, int64_t axis, const int64_t *stops,
    size_t output_count, uint64_t **out_values, size_t *out_length) {
  uint64_t *values;
  int64_t start = 0;
  size_t index;
  TerlanNdArrayStatus status;
  if (out_values == NULL || out_length == NULL || output_count == 0 ||
      output_count > SIZE_MAX / sizeof(uint64_t)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_values = NULL;
  *out_length = 0;
  values = checked_calloc(output_count, sizeof(uint64_t));
  if (values == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  for (index = 0; index < output_count; ++index) {
    TerlanNdArrayHandle view = NULL;
    status = terlan_ndarray_v1_slice_axis(array, axis, start, stops[index], 1,
                                          &view);
    if (status != TERLAN_NDARRAY_OK) {
      release_split_values(values, index);
      return status;
    }
    values[index] = (uint64_t)(uintptr_t)view;
    start = stops[index];
  }
  *out_values = values;
  *out_length = output_count;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_split(
    TerlanNdArrayHandle array, int64_t sections, int64_t axis,
    uint64_t **out_values, size_t *out_length) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t normalized;
  int64_t *stops = NULL;
  int64_t section;
  if (status != TERLAN_NDARRAY_OK || out_values == NULL ||
      out_length == NULL || sections <= 0 ||
      (uint64_t)sections > SIZE_MAX / sizeof(int64_t)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_values = NULL;
  *out_length = 0;
  status = normalize_axis(axis, array->rank, 0, &normalized);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (array->shape[normalized] % sections != 0) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  stops = checked_malloc((size_t)sections * sizeof(int64_t));
  if (stops == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  for (section = 0; section < sections; ++section) {
    stops[section] =
        (section + 1) * (array->shape[normalized] / sections);
  }
  status = create_split_views(array, normalized, stops, (size_t)sections,
                              out_values, out_length);
  free(stops);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_array_split(
    TerlanNdArrayHandle array, int64_t sections, int64_t axis,
    uint64_t **out_values, size_t *out_length) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t normalized;
  int64_t *stops = NULL;
  int64_t quotient;
  int64_t remainder;
  int64_t stop = 0;
  int64_t section;
  if (status != TERLAN_NDARRAY_OK || out_values == NULL ||
      out_length == NULL || sections <= 0 ||
      (uint64_t)sections > SIZE_MAX / sizeof(int64_t)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_values = NULL;
  *out_length = 0;
  status = normalize_axis(axis, array->rank, 0, &normalized);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  stops = checked_malloc((size_t)sections * sizeof(int64_t));
  if (stops == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  quotient = array->shape[normalized] / sections;
  remainder = array->shape[normalized] % sections;
  for (section = 0; section < sections; ++section) {
    stop += quotient + (section < remainder ? 1 : 0);
    stops[section] = stop;
  }
  status = create_split_views(array, normalized, stops, (size_t)sections,
                              out_values, out_length);
  free(stops);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_split_at(
    TerlanNdArrayHandle array, const int64_t *indices, int64_t index_count,
    int64_t axis, uint64_t **out_values, size_t *out_length) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t normalized;
  int64_t *stops = NULL;
  int64_t index;
  size_t output_count;
  if (status != TERLAN_NDARRAY_OK || out_values == NULL ||
      out_length == NULL || index_count < 0 ||
      (index_count > 0 && indices == NULL) ||
      (uint64_t)index_count >= SIZE_MAX / sizeof(int64_t)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_values = NULL;
  *out_length = 0;
  status = normalize_axis(axis, array->rank, 0, &normalized);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  output_count = (size_t)index_count + 1;
  stops = checked_malloc(output_count * sizeof(int64_t));
  if (stops == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  for (index = 0; index < index_count; ++index) {
    int64_t normalized_index = indices[index];
    if (normalized_index < 0) {
      normalized_index += array->shape[normalized];
    }
    if (normalized_index < 0) {
      normalized_index = 0;
    } else if (normalized_index > array->shape[normalized]) {
      normalized_index = array->shape[normalized];
    }
    stops[index] = normalized_index;
  }
  stops[index_count] = array->shape[normalized];
  status = create_split_views(array, normalized, stops, output_count,
                              out_values, out_length);
  free(stops);
  return status;
}
