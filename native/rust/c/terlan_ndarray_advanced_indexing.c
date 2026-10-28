#include "terlan_ndarray_internal.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static TerlanNdArrayStatus normalize_advanced_axis(int64_t axis, int64_t rank,
                                                    int64_t *out_axis) {
  if (axis < 0) {
    axis += rank;
  }
  if (out_axis == NULL || axis < 0 || axis >= rank) {
    return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
  }
  *out_axis = axis;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus normalize_advanced_index(int64_t index,
                                                     int64_t length,
                                                     int64_t *out_index) {
  if (index < 0) {
    index += length;
  }
  if (out_index == NULL || index < 0 || index >= length) {
    return TERLAN_NDARRAY_INDEX_OUT_OF_RANGE;
  }
  *out_index = index;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus load_advanced_index(TerlanNdArrayHandle indexes,
                                                int64_t logical,
                                                int64_t *out_index) {
  TerlanNdArrayStatus status;
  if (!dtype_is_integer(indexes->dtype)) {
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  status = array_load_int64(indexes, logical, out_index);
  return status == TERLAN_NDARRAY_CAST_OVERFLOW
             ? TERLAN_NDARRAY_INDEX_OUT_OF_RANGE
             : status;
}

static TerlanNdArrayStatus broadcast_iteration_shape(
    TerlanNdArrayHandle array, TerlanNdArrayHandle indices, int64_t axis,
    int64_t **out_shape) {
  int64_t *shape;
  int64_t dimension;
  if (out_shape == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_shape = NULL;
  if (array->rank != indices->rank) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  shape = checked_malloc((size_t)array->rank * sizeof(*shape));
  if (shape == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  for (dimension = 0; dimension < array->rank; ++dimension) {
    const int64_t array_dimension = array->shape[dimension];
    const int64_t index_dimension = indices->shape[dimension];
    if (dimension == axis) {
      shape[dimension] = index_dimension;
    } else if (array_dimension == index_dimension || index_dimension == 1) {
      shape[dimension] = array_dimension;
    } else if (array_dimension == 1) {
      shape[dimension] = index_dimension;
    } else {
      free(shape);
      return TERLAN_NDARRAY_SHAPE_MISMATCH;
    }
  }
  *out_shape = shape;
  return TERLAN_NDARRAY_OK;
}

static int64_t broadcast_logical(int64_t logical,
                                 const int64_t *iteration_shape,
                                 int64_t iteration_rank,
                                 TerlanNdArrayHandle target) {
  return array_broadcast_index(target, iteration_shape, iteration_rank,
                               logical);
}

static int64_t axis_replaced_broadcast_logical(
    int64_t logical, const int64_t *iteration_shape,
    TerlanNdArrayHandle source, int64_t axis, int64_t replacement) {
  int64_t source_logical = 0;
  int64_t stride = 1;
  int64_t dimension;
  for (dimension = source->rank; dimension > 0; --dimension) {
    const int64_t current = dimension - 1;
    int64_t coordinate = logical % iteration_shape[current];
    logical /= iteration_shape[current];
    if (current == axis) {
      coordinate = replacement;
    } else if (source->shape[current] == 1) {
      coordinate = 0;
    }
    source_logical += coordinate * stride;
    stride *= source->shape[current];
  }
  return source_logical;
}

static int broadcasts_to(TerlanNdArrayHandle value,
                         const int64_t *target_shape, int64_t target_rank) {
  return array_broadcast_compatible(value, target_shape, target_rank);
}

static TerlanNdArrayStatus shape_element_count(const int64_t *shape,
                                               int64_t rank,
                                               int64_t *out_count) {
  int64_t count = 1;
  int64_t dimension;
  if (out_count == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  for (dimension = 0; dimension < rank; ++dimension) {
    if (shape[dimension] != 0 && count > INT64_MAX / shape[dimension]) {
      return TERLAN_NDARRAY_SIZE_OVERFLOW;
    }
    count *= shape[dimension];
  }
  *out_count = count;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus create_index_output(
    const int64_t *shape, int64_t rank, int32_t dtype,
    TerlanNdArrayHandle layout_source, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status;
  TerlanNdArrayHandle backing = NULL;
  int64_t *reversed_shape = NULL;
  int64_t *reversed_axes = NULL;
  int64_t axis;
  const int preserve_fortran =
      rank > 1 && array_is_fortran_contiguous(layout_source) &&
      !array_is_contiguous(layout_source);
  if (!preserve_fortran) {
    return create_empty_array(shape, rank, dtype, out_array);
  }
  if ((uint64_t)rank > SIZE_MAX / sizeof(*reversed_shape)) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  reversed_shape = checked_malloc((size_t)rank * sizeof(*reversed_shape));
  reversed_axes = checked_malloc((size_t)rank * sizeof(*reversed_axes));
  if (reversed_shape == NULL || reversed_axes == NULL) {
    free(reversed_axes);
    free(reversed_shape);
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  for (axis = 0; axis < rank; ++axis) {
    reversed_shape[axis] = shape[rank - axis - 1];
    reversed_axes[axis] = rank - axis - 1;
  }
  status = create_empty_array(reversed_shape, rank, dtype, &backing);
  if (status == TERLAN_NDARRAY_OK) {
    status = terlan_ndarray_v1_permute(backing, reversed_axes, rank,
                                       out_array);
  }
  if (backing != NULL) {
    (void)terlan_ndarray_v1_delete(backing);
  }
  free(reversed_axes);
  free(reversed_shape);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_take_along_axis(
    TerlanNdArrayHandle array, TerlanNdArrayHandle indices, int64_t axis,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle output = NULL;
  int64_t normalized_axis;
  int64_t *iteration_shape = NULL;
  int64_t logical;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(indices);
  }
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
  status = normalize_advanced_axis(axis, array->rank, &normalized_axis);
  if (status == TERLAN_NDARRAY_OK) {
    status = broadcast_iteration_shape(array, indices, normalized_axis,
                                       &iteration_shape);
  }
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  status = create_index_output(iteration_shape, indices->rank, array->dtype,
                               indices, &output);
  for (logical = 0; status == TERLAN_NDARRAY_OK && logical < output->numel;
       ++logical) {
    int64_t selected;
    const int64_t index_logical = broadcast_logical(
        logical, iteration_shape, indices->rank, indices);
    status = load_advanced_index(indices, index_logical, &selected);
    if (status == TERLAN_NDARRAY_OK) {
      status = normalize_advanced_index(
          selected, array->shape[normalized_axis], &selected);
    }
    if (status == TERLAN_NDARRAY_OK) {
      const int64_t source_logical = axis_replaced_broadcast_logical(
          logical, iteration_shape, array, normalized_axis, selected);
      memcpy(array_element_pointer(output, logical),
             array_const_element_pointer(array, source_logical),
             dtype_size(array->dtype));
    }
  }
  if (status != TERLAN_NDARRAY_OK) {
    if (output != NULL) {
      (void)terlan_ndarray_v1_delete(output);
    }
    free(iteration_shape);
    return status;
  }
  free(iteration_shape);
  *out_array = output;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_put(
    TerlanNdArrayHandle array, const int64_t *indices, int64_t index_count,
    TerlanNdArrayHandle values) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t index;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(values);
  }
  if (status != TERLAN_NDARRAY_OK || index_count < 0 ||
      (index_count > 0 && indices == NULL) || array->writable == 0 ||
      array->dtype != values->dtype ||
      (values->numel != 1 && values->numel != index_count)) {
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
    if (array->writable == 0) {
      return TERLAN_NDARRAY_READ_ONLY;
    }
    if (array->dtype != values->dtype) {
      return TERLAN_NDARRAY_DTYPE_MISMATCH;
    }
    return values->numel != 1 && values->numel != index_count
               ? TERLAN_NDARRAY_SHAPE_MISMATCH
               : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  for (index = 0; index < index_count; ++index) {
    int64_t target;
    status = normalize_advanced_index(indices[index], array->numel, &target);
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
  }
  for (index = 0; index < index_count; ++index) {
    int64_t target;
    (void)normalize_advanced_index(indices[index], array->numel, &target);
    memcpy(array_element_pointer(array, target),
           array_const_element_pointer(values, values->numel == 1 ? 0 : index),
           dtype_size(array->dtype));
  }
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_put_along_axis(
    TerlanNdArrayHandle array, TerlanNdArrayHandle indices,
    TerlanNdArrayHandle values, int64_t axis) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t normalized_axis;
  int64_t *iteration_shape = NULL;
  int64_t iteration_count = 0;
  unsigned char *staged_values = NULL;
  size_t element_size;
  int64_t logical;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(indices);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(values);
  }
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  status = normalize_advanced_axis(axis, array->rank, &normalized_axis);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (array->writable == 0) {
    return TERLAN_NDARRAY_READ_ONLY;
  }
  if (!dtype_is_integer(indices->dtype) || values->dtype != array->dtype) {
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  status = broadcast_iteration_shape(array, indices, normalized_axis,
                                     &iteration_shape);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (!broadcasts_to(values, iteration_shape, indices->rank)) {
    free(iteration_shape);
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  status = shape_element_count(iteration_shape, indices->rank,
                               &iteration_count);
  element_size = dtype_size(array->dtype);
  if (status != TERLAN_NDARRAY_OK ||
      (uint64_t)iteration_count > SIZE_MAX / element_size) {
    free(iteration_shape);
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  if (iteration_count > 0) {
    staged_values = checked_malloc((size_t)iteration_count * element_size);
    if (staged_values == NULL) {
      free(iteration_shape);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (logical = 0; logical < iteration_count; ++logical) {
    int64_t selected;
    const int64_t index_logical = broadcast_logical(
        logical, iteration_shape, indices->rank, indices);
    const int64_t value_logical = broadcast_logical(
        logical, iteration_shape, indices->rank, values);
    status = load_advanced_index(indices, index_logical, &selected);
    if (status == TERLAN_NDARRAY_OK) {
      status = normalize_advanced_index(
          selected, array->shape[normalized_axis], &selected);
    }
    if (status != TERLAN_NDARRAY_OK) {
      free(staged_values);
      free(iteration_shape);
      return status;
    }
    memcpy(staged_values + (size_t)logical * element_size,
           array_const_element_pointer(values, value_logical), element_size);
  }
  for (logical = 0; logical < iteration_count; ++logical) {
    int64_t selected;
    const int64_t index_logical = broadcast_logical(
        logical, iteration_shape, indices->rank, indices);
    (void)load_advanced_index(indices, index_logical, &selected);
    (void)normalize_advanced_index(selected, array->shape[normalized_axis],
                                   &selected);
    memcpy(array_element_pointer(
               array, axis_replaced_broadcast_logical(
                          logical, iteration_shape, array, normalized_axis,
                          selected)),
           staged_values + (size_t)logical * element_size, element_size);
  }
  free(staged_values);
  free(iteration_shape);
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_compress(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask, int64_t axis,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t normalized_axis;
  int64_t *indices = NULL;
  int64_t count = 0;
  int64_t index;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(mask);
  }
  if (out_array != NULL) {
    *out_array = NULL;
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL ||
      mask->dtype != TERLAN_NDARRAY_DTYPE_BOOL || mask->rank != 1) {
    return status != TERLAN_NDARRAY_OK
               ? status
               : (out_array == NULL ? TERLAN_NDARRAY_INVALID_ARGUMENT
                                    : TERLAN_NDARRAY_DTYPE_MISMATCH);
  }
  status = normalize_advanced_axis(axis, array->rank, &normalized_axis);
  if (status != TERLAN_NDARRAY_OK ||
      mask->numel > array->shape[normalized_axis]) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  if (mask->numel > 0) {
    indices = checked_malloc((size_t)mask->numel * sizeof(*indices));
    if (indices == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (index = 0; index < mask->numel; ++index) {
    if (*(const uint8_t *)array_const_element_pointer(mask, index) != 0) {
      indices[count++] = index;
    }
  }
  status = terlan_ndarray_v1_take(array, indices, count, normalized_axis,
                                  out_array);
  free(indices);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_compress_flat(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle result = NULL;
  int64_t selected_count = 0;
  int64_t source_index;
  int64_t output_index = 0;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(mask);
  }
  if (out_array != NULL) {
    *out_array = NULL;
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (mask->dtype != TERLAN_NDARRAY_DTYPE_BOOL || mask->rank != 1) {
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  if (mask->numel > array->numel) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  for (source_index = 0; source_index < mask->numel; ++source_index) {
    if (*(const uint8_t *)array_const_element_pointer(mask, source_index) != 0) {
      ++selected_count;
    }
  }
  status = create_empty_array(&selected_count, 1, array->dtype, &result);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (source_index = 0; source_index < mask->numel; ++source_index) {
    if (*(const uint8_t *)array_const_element_pointer(mask, source_index) != 0) {
      memcpy(array_element_pointer(result, output_index++),
             array_const_element_pointer(array, source_index),
             dtype_size(array->dtype));
    }
  }
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_compress_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    TerlanNdArrayHandle mask, int64_t axis) {
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayStatus status =
      terlan_ndarray_v1_compress(array, mask, axis, &result);
  return array_finish_into(destination, status, result,
                           TERLAN_NDARRAY_CAST_NO);
}

TerlanNdArrayStatus terlan_ndarray_v1_compress_flat_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    TerlanNdArrayHandle mask) {
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayStatus status =
      terlan_ndarray_v1_compress_flat(array, mask, &result);
  return array_finish_into(destination, status, result,
                           TERLAN_NDARRAY_CAST_NO);
}

static TerlanNdArrayStatus require_prefix_mask(TerlanNdArrayHandle array,
                                               TerlanNdArrayHandle mask) {
  int64_t dimension;
  if (mask->dtype != TERLAN_NDARRAY_DTYPE_BOOL) {
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  if (mask->rank > array->rank) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  for (dimension = 0; dimension < mask->rank; ++dimension) {
    if (mask->shape[dimension] != array->shape[dimension]) {
      return TERLAN_NDARRAY_SHAPE_MISMATCH;
    }
  }
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus mask_selection_layout(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask,
    int64_t **out_shape, int64_t *out_rank, int64_t *out_suffix_count) {
  TerlanNdArrayStatus status = require_prefix_mask(array, mask);
  int64_t suffix_rank;
  int64_t *shape;
  int64_t index;
  int64_t selected = 0;
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  suffix_rank = array->rank - mask->rank;
  if (out_shape == NULL || out_rank == NULL || out_suffix_count == NULL ||
      (uint64_t)(suffix_rank + 1) > SIZE_MAX / sizeof(*shape)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  shape = checked_malloc((size_t)(suffix_rank + 1) * sizeof(*shape));
  if (shape == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  for (index = 0; index < mask->numel; ++index) {
    selected +=
        *(const uint8_t *)array_const_element_pointer(mask, index) != 0;
  }
  shape[0] = selected;
  if (suffix_rank > 0) {
    memcpy(shape + 1, array->shape + mask->rank,
           (size_t)suffix_rank * sizeof(*shape));
  }
  status = shape_element_count(
      suffix_rank == 0 ? NULL : array->shape + mask->rank, suffix_rank,
      out_suffix_count);
  if (status != TERLAN_NDARRAY_OK) {
    free(shape);
    return status;
  }
  *out_shape = shape;
  *out_rank = suffix_rank + 1;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_mask_select(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle output = NULL;
  int64_t *output_shape = NULL;
  int64_t output_rank = 0;
  int64_t suffix_count = 0;
  int64_t mask_index;
  int64_t output_index = 0;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(mask);
  }
  if (out_array != NULL) {
    *out_array = NULL;
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  status = mask_selection_layout(array, mask, &output_shape, &output_rank,
                                 &suffix_count);
  if (status == TERLAN_NDARRAY_OK) {
    status = create_empty_array(output_shape, output_rank, array->dtype,
                                &output);
  }
  for (mask_index = 0;
       status == TERLAN_NDARRAY_OK && mask_index < mask->numel;
       ++mask_index) {
    if (*(const uint8_t *)array_const_element_pointer(mask, mask_index) != 0) {
      int64_t suffix_index;
      for (suffix_index = 0; suffix_index < suffix_count; ++suffix_index) {
        memcpy(array_element_pointer(output, output_index++),
               array_const_element_pointer(
                   array, mask_index * suffix_count + suffix_index),
               dtype_size(array->dtype));
      }
    }
  }
  free(output_shape);
  if (status != TERLAN_NDARRAY_OK) {
    if (output != NULL) {
      (void)terlan_ndarray_v1_delete(output);
    }
    return status;
  }
  *out_array = output;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_mask_assign(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask,
    TerlanNdArrayHandle values) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t *selection_shape = NULL;
  int64_t selection_rank = 0;
  int64_t suffix_count = 0;
  int64_t selected_elements = 0;
  int64_t *targets = NULL;
  unsigned char *staged_values = NULL;
  size_t element_size;
  int64_t mask_index;
  int64_t selected_index = 0;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(mask);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(values);
  }
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  status = mask_selection_layout(array, mask, &selection_shape,
                                 &selection_rank, &suffix_count);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (array->writable == 0) {
    free(selection_shape);
    return TERLAN_NDARRAY_READ_ONLY;
  }
  if (values->dtype != array->dtype) {
    free(selection_shape);
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  if (!array_broadcast_compatible(values, selection_shape, selection_rank)) {
    free(selection_shape);
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  status = shape_element_count(selection_shape, selection_rank,
                               &selected_elements);
  element_size = dtype_size(array->dtype);
  if (status != TERLAN_NDARRAY_OK ||
      (uint64_t)selected_elements > SIZE_MAX / sizeof(*targets) ||
      (uint64_t)selected_elements > SIZE_MAX / element_size) {
    free(selection_shape);
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  if (selected_elements == 0) {
    free(selection_shape);
    return TERLAN_NDARRAY_OK;
  }
  targets = checked_malloc((size_t)selected_elements * sizeof(*targets));
  staged_values = checked_malloc((size_t)selected_elements * element_size);
  if (targets == NULL || staged_values == NULL) {
    free(staged_values);
    free(targets);
    free(selection_shape);
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  for (mask_index = 0; mask_index < mask->numel; ++mask_index) {
    if (*(const uint8_t *)array_const_element_pointer(mask, mask_index) != 0) {
      int64_t suffix_index;
      for (suffix_index = 0; suffix_index < suffix_count; ++suffix_index) {
        targets[selected_index] = mask_index * suffix_count + suffix_index;
        memcpy(staged_values + (size_t)selected_index * element_size,
               array_const_element_pointer(
                   values, array_broadcast_index(values, selection_shape,
                                                 selection_rank,
                                                 selected_index)),
               element_size);
        ++selected_index;
      }
    }
  }
  for (selected_index = 0; selected_index < selected_elements;
       ++selected_index) {
    memcpy(array_element_pointer(array, targets[selected_index]),
           staged_values + (size_t)selected_index * element_size,
           element_size);
  }
  free(staged_values);
  free(targets);
  free(selection_shape);
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus array_coordinate_targets(
    TerlanNdArrayHandle array, TerlanNdArrayHandle coordinates,
    int64_t **out_targets, int64_t *out_count) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t *targets = NULL;
  int64_t count = 0;
  int64_t row;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(coordinates);
  }
  if (out_targets == NULL || out_count == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_targets = NULL;
  *out_count = 0;
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (coordinates->rank < 1 ||
      coordinates->shape[coordinates->rank - 1] != array->rank) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  if (!dtype_is_integer(coordinates->dtype)) {
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  status = shape_element_count(coordinates->shape, coordinates->rank - 1,
                               &count);
  if (status != TERLAN_NDARRAY_OK || count == 0) {
    *out_count = count;
    return status;
  }
  if ((uint64_t)count > SIZE_MAX / sizeof(*targets)) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  targets = checked_malloc((size_t)count * sizeof(*targets));
  if (targets == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  for (row = 0; row < count; ++row) {
    int64_t logical = 0;
    int64_t dimension;
    for (dimension = 0; dimension < array->rank; ++dimension) {
      int64_t coordinate;
      status = load_advanced_index(
          coordinates, row * array->rank + dimension, &coordinate);
      if (status == TERLAN_NDARRAY_OK) {
        status = normalize_advanced_index(coordinate, array->shape[dimension],
                                          &coordinate);
      }
      if (status != TERLAN_NDARRAY_OK) {
        free(targets);
        return status;
      }
      logical = logical * array->shape[dimension] + coordinate;
    }
    targets[row] = logical;
  }
  *out_targets = targets;
  *out_count = count;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_gather_coordinates(
    TerlanNdArrayHandle array, TerlanNdArrayHandle coordinates,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status;
  TerlanNdArrayHandle output = NULL;
  int64_t *targets = NULL;
  int64_t output_count = 0;
  int64_t row;
  if (out_array == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = array_coordinate_targets(array, coordinates, &targets,
                                    &output_count);
  if (status == TERLAN_NDARRAY_OK) {
    status = create_index_output(coordinates->shape, coordinates->rank - 1,
                                 array->dtype, coordinates, &output);
  }
  for (row = 0; status == TERLAN_NDARRAY_OK && row < output_count; ++row) {
    memcpy(array_element_pointer(output, row),
           array_const_element_pointer(array, targets[row]),
           dtype_size(array->dtype));
  }
  free(targets);
  if (status != TERLAN_NDARRAY_OK) {
    if (output != NULL) {
      (void)terlan_ndarray_v1_delete(output);
    }
    return status;
  }
  *out_array = output;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_put_coordinates(
    TerlanNdArrayHandle array, TerlanNdArrayHandle coordinates,
    TerlanNdArrayHandle values) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t *targets = NULL;
  uint8_t *copied_values = NULL;
  size_t element_size;
  int64_t selection_rank;
  int64_t count;
  int64_t row;
  int flatten_values;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(values);
  }
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (array->writable == 0) {
    return TERLAN_NDARRAY_READ_ONLY;
  }
  status = array_coordinate_targets(array, coordinates, &targets, &count);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  selection_rank = coordinates->rank - 1;
  if (array->dtype != values->dtype) {
    free(targets);
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  flatten_values = values->numel == count;
  if (!flatten_values &&
      !array_broadcast_compatible(values, coordinates->shape,
                                  selection_rank)) {
    free(targets);
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  if (count == 0) {
    free(targets);
    return TERLAN_NDARRAY_OK;
  }
  element_size = dtype_size(array->dtype);
  if ((uint64_t)count > SIZE_MAX / element_size) {
    free(targets);
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  copied_values = checked_malloc((size_t)count * element_size);
  if (copied_values == NULL) {
    free(targets);
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  for (row = 0; row < count; ++row) {
    const int64_t value_index =
        flatten_values
            ? row
            : array_broadcast_index(values, coordinates->shape,
                                    selection_rank, row);
    memcpy(copied_values + (size_t)row * element_size,
           array_const_element_pointer(values, value_index),
           element_size);
  }
  for (row = 0; row < count; ++row) {
    memcpy(array_element_pointer(array, targets[row]),
           copied_values + (size_t)row * element_size,
           element_size);
  }
  free(copied_values);
  free(targets);
  return TERLAN_NDARRAY_OK;
}
