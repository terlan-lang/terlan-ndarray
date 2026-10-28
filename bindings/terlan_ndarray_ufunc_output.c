#include "terlan_ndarray_internal.h"

#include <stdint.h>
#include <stdlib.h>

typedef TerlanNdArrayStatus (*UnaryArrayOperation)(
    TerlanNdArrayHandle, TerlanNdArrayHandle *);
typedef TerlanNdArrayStatus (*BinaryArrayOperation)(
    TerlanNdArrayHandle, TerlanNdArrayHandle, TerlanNdArrayHandle *);

static TerlanNdArrayStatus unary_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    UnaryArrayOperation operation) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status = operation(array, &result);
  return array_finish_into_broadcast(destination, status, result,
                                     TERLAN_NDARRAY_CAST_SAME_KIND);
}

static TerlanNdArrayStatus binary_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle lhs,
    TerlanNdArrayHandle rhs, BinaryArrayOperation operation) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status = operation(lhs, rhs, &result);
  return array_finish_into_broadcast(destination, status, result,
                                     TERLAN_NDARRAY_CAST_SAME_KIND);
}

static TerlanNdArrayStatus unary_where_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    TerlanNdArrayHandle mask, UnaryArrayOperation operation) {
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayStatus status = operation(array, &result);
  if (status == TERLAN_NDARRAY_OK) {
    status = terlan_ndarray_v1_copyto_where(
        destination, result, mask, TERLAN_NDARRAY_CAST_SAME_KIND);
  }
  if (result != NULL) {
    (void)terlan_ndarray_v1_delete(result);
  }
  return status;
}

static TerlanNdArrayStatus binary_where_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle lhs,
    TerlanNdArrayHandle rhs, TerlanNdArrayHandle mask,
    BinaryArrayOperation operation) {
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayStatus status = operation(lhs, rhs, &result);
  if (status == TERLAN_NDARRAY_OK) {
    status = terlan_ndarray_v1_copyto_where(
        destination, result, mask, TERLAN_NDARRAY_CAST_SAME_KIND);
  }
  if (result != NULL) {
    (void)terlan_ndarray_v1_delete(result);
  }
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_add_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle lhs,
    TerlanNdArrayHandle rhs) {
  return binary_into(destination, lhs, rhs, terlan_ndarray_v1_add);
}

TerlanNdArrayStatus terlan_ndarray_v1_add_where_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle lhs,
    TerlanNdArrayHandle rhs, TerlanNdArrayHandle mask) {
  return binary_where_into(destination, lhs, rhs, mask,
                           terlan_ndarray_v1_add);
}

TerlanNdArrayStatus terlan_ndarray_v1_absolute_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array) {
  return unary_into(destination, array, terlan_ndarray_v1_absolute);
}

TerlanNdArrayStatus terlan_ndarray_v1_logical_not_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array) {
  return unary_into(destination, array, terlan_ndarray_v1_logical_not);
}

TerlanNdArrayStatus terlan_ndarray_v1_logical_and_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle lhs,
    TerlanNdArrayHandle rhs) {
  return binary_into(destination, lhs, rhs, terlan_ndarray_v1_logical_and);
}

TerlanNdArrayStatus terlan_ndarray_v1_logical_or_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle lhs,
    TerlanNdArrayHandle rhs) {
  return binary_into(destination, lhs, rhs, terlan_ndarray_v1_logical_or);
}

TerlanNdArrayStatus terlan_ndarray_v1_logical_xor_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle lhs,
    TerlanNdArrayHandle rhs) {
  return binary_into(destination, lhs, rhs, terlan_ndarray_v1_logical_xor);
}

TerlanNdArrayStatus terlan_ndarray_v1_invert_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array) {
  return unary_into(destination, array, terlan_ndarray_v1_bitwise_not);
}

TerlanNdArrayStatus terlan_ndarray_v1_invert_where_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    TerlanNdArrayHandle mask) {
  return unary_where_into(destination, array, mask,
                          terlan_ndarray_v1_bitwise_not);
}

TerlanNdArrayStatus terlan_ndarray_v1_reciprocal_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array) {
  return unary_into(destination, array, terlan_ndarray_v1_reciprocal);
}

TerlanNdArrayStatus terlan_ndarray_v1_reciprocal_where_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    TerlanNdArrayHandle mask) {
  return unary_where_into(destination, array, mask,
                          terlan_ndarray_v1_reciprocal);
}

static TerlanNdArrayStatus copied_indexes(
    TerlanNdArrayHandle indexes, int64_t bound, int64_t **out_indexes,
    int64_t *out_count) {
  TerlanNdArrayStatus status = require_live(indexes);
  int64_t *values = NULL;
  int64_t index;
  if (status != TERLAN_NDARRAY_OK || out_indexes == NULL ||
      out_count == NULL || bound < 0 || indexes->rank != 1 ||
      !dtype_is_integer(indexes->dtype)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_indexes = NULL;
  *out_count = 0;
  if (indexes->numel > 0) {
    if ((uint64_t)indexes->numel > SIZE_MAX / sizeof(*values)) {
      return TERLAN_NDARRAY_SIZE_OVERFLOW;
    }
    values = checked_malloc((size_t)indexes->numel * sizeof(*values));
    if (values == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (index = 0; index < indexes->numel; ++index) {
    status = array_load_int64(indexes, index, &values[index]);
    if (status != TERLAN_NDARRAY_OK || values[index] < 0 ||
        values[index] >= bound) {
      free(values);
      return status != TERLAN_NDARRAY_OK ? status
                                         : TERLAN_NDARRAY_INDEX_OUT_OF_RANGE;
    }
  }
  *out_indexes = values;
  *out_count = indexes->numel;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus invert_pointer(void *pointer, int32_t dtype) {
  if (pointer == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  switch (dtype) {
  case TERLAN_NDARRAY_DTYPE_BOOL:
    *(uint8_t *)pointer = *(uint8_t *)pointer == 0;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_INT8:
    *(int8_t *)pointer = (int8_t)~*(uint8_t *)pointer;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_INT16:
    *(int16_t *)pointer = (int16_t)~*(uint16_t *)pointer;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_INT32:
    *(int32_t *)pointer = (int32_t)~*(uint32_t *)pointer;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_INT64:
    *(int64_t *)pointer = (int64_t)~*(uint64_t *)pointer;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_UINT8:
    *(uint8_t *)pointer = (uint8_t)~*(uint8_t *)pointer;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_UINT16:
    *(uint16_t *)pointer = (uint16_t)~*(uint16_t *)pointer;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_UINT32:
    *(uint32_t *)pointer = ~*(uint32_t *)pointer;
    return TERLAN_NDARRAY_OK;
  case TERLAN_NDARRAY_DTYPE_UINT64:
    *(uint64_t *)pointer = ~*(uint64_t *)pointer;
    return TERLAN_NDARRAY_OK;
  default:
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
}

TerlanNdArrayStatus terlan_ndarray_v1_bitwise_not(
    TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle result = NULL;
  int64_t index;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (array->dtype != TERLAN_NDARRAY_DTYPE_BOOL &&
      !dtype_is_integer(array->dtype)) {
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  status = create_empty_array(array->shape, array->rank, array->dtype,
                              &result);
  for (index = 0; status == TERLAN_NDARRAY_OK && index < array->numel;
       ++index) {
    const size_t element_size = dtype_size(array->dtype);
    void *destination = array_element_pointer(result, index);
    const void *source = array_const_element_pointer(array, index);
    for (size_t byte = 0; byte < element_size; ++byte) {
      ((uint8_t *)destination)[byte] = ((const uint8_t *)source)[byte];
    }
    status = invert_pointer(destination, array->dtype);
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

TerlanNdArrayStatus terlan_ndarray_v1_invert_at(
    TerlanNdArrayHandle array, TerlanNdArrayHandle indexes) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t *stable_indexes = NULL;
  int64_t index_count = 0;
  int64_t index;
  if (status != TERLAN_NDARRAY_OK || array->rank != 1 ||
      array->writable == 0) {
    return status != TERLAN_NDARRAY_OK
               ? status
               : array->writable == 0 ? TERLAN_NDARRAY_READ_ONLY
                                      : TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  status = copied_indexes(indexes, array->shape[0], &stable_indexes,
                          &index_count);
  for (index = 0; status == TERLAN_NDARRAY_OK && index < index_count;
       ++index) {
    status = invert_pointer(array_element_pointer(array, stable_indexes[index]),
                            array->dtype);
  }
  free(stable_indexes);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_add_at(
    TerlanNdArrayHandle array, TerlanNdArrayHandle indexes,
    TerlanNdArrayHandle values) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle stable_values = NULL;
  int64_t *stable_indexes = NULL;
  int64_t index_count = 0;
  int64_t index;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(values);
  }
  if (status != TERLAN_NDARRAY_OK || array->rank != 1 ||
      array->writable == 0) {
    return status != TERLAN_NDARRAY_OK
               ? status
               : array->writable == 0 ? TERLAN_NDARRAY_READ_ONLY
                                      : TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  status = copied_indexes(indexes, array->shape[0], &stable_indexes,
                          &index_count);
  if (status == TERLAN_NDARRAY_OK &&
      (!array_broadcast_compatible(values, indexes->shape, indexes->rank) ||
       !dtype_can_cast(values->dtype, array->dtype,
                       TERLAN_NDARRAY_CAST_SAME_KIND))) {
    status = TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = copy_array(values, &stable_values);
  }
  for (index = 0; status == TERLAN_NDARRAY_OK && index < index_count;
       ++index) {
    TerlanNdArrayScalar current;
    TerlanNdArrayScalar addition;
    const int64_t value_index = array_broadcast_index(
        stable_values, indexes->shape, indexes->rank, index);
    void *destination = array_element_pointer(array, stable_indexes[index]);
    status = dtype_load(destination, array->dtype, &current);
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_load(array_const_element_pointer(stable_values,
                                                       value_index),
                          stable_values->dtype, &addition);
    }
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_store_cast(destination, array->dtype,
                                scalar_add(current, addition),
                                TERLAN_NDARRAY_CAST_SAME_KIND);
    }
  }
  if (stable_values != NULL) {
    (void)terlan_ndarray_v1_delete(stable_values);
  }
  free(stable_indexes);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_add_at_coordinates(
    TerlanNdArrayHandle array, TerlanNdArrayHandle coordinates,
    TerlanNdArrayHandle values) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle stable_values = NULL;
  TerlanNdArrayHandle working = NULL;
  int64_t *targets = NULL;
  int64_t count = 0;
  int64_t selection_rank;
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
  flatten_values = values->numel == count;
  if (!dtype_can_cast(values->dtype, array->dtype,
                      TERLAN_NDARRAY_CAST_SAME_KIND)) {
    status = TERLAN_NDARRAY_DTYPE_MISMATCH;
  } else if (!flatten_values &&
             !array_broadcast_compatible(values, coordinates->shape,
                                         selection_rank)) {
    status = TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = copy_array(values, &stable_values);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = copy_array(array, &working);
  }
  for (row = 0; status == TERLAN_NDARRAY_OK && row < count; ++row) {
    TerlanNdArrayScalar current;
    TerlanNdArrayScalar addition;
    const int64_t value_index =
        flatten_values
            ? row
            : array_broadcast_index(stable_values, coordinates->shape,
                                    selection_rank, row);
    void *destination = array_element_pointer(working, targets[row]);
    status = dtype_load(destination, working->dtype, &current);
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_load(array_const_element_pointer(stable_values,
                                                       value_index),
                          stable_values->dtype, &addition);
    }
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_store_cast(destination, working->dtype,
                                scalar_add(current, addition),
                                TERLAN_NDARRAY_CAST_SAME_KIND);
    }
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = array_write_exact(array, working, TERLAN_NDARRAY_CAST_NO);
  }
  if (working != NULL) {
    (void)terlan_ndarray_v1_delete(working);
  }
  if (stable_values != NULL) {
    (void)terlan_ndarray_v1_delete(stable_values);
  }
  free(targets);
  return status;
}

static TerlanNdArrayStatus normalized_axis(int64_t axis, int64_t rank,
                                           int64_t *out_axis) {
  if (out_axis == NULL || rank <= 0) {
    return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
  }
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
  }
  *out_axis = axis;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_add_reduceat(
    TerlanNdArrayHandle array, TerlanNdArrayHandle indexes, int64_t axis,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle result = NULL;
  int64_t *stable_indexes = NULL;
  int64_t index_count = 0;
  int64_t *output_shape = NULL;
  int64_t output_index;
  int64_t normalized = 0;
  if (status == TERLAN_NDARRAY_OK) {
    status = normalized_axis(axis, array->rank, &normalized);
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = copied_indexes(indexes, array->shape[normalized], &stable_indexes,
                          &index_count);
  if (status == TERLAN_NDARRAY_OK && array->rank > 0) {
    output_shape = checked_malloc((size_t)array->rank * sizeof(*output_shape));
    if (output_shape == NULL) {
      status = TERLAN_NDARRAY_ALLOCATION_FAILED;
    } else {
      for (int64_t dimension = 0; dimension < array->rank; ++dimension) {
        output_shape[dimension] = dimension == normalized
                                      ? index_count
                                      : array->shape[dimension];
      }
      status = create_empty_array(output_shape, array->rank, array->dtype,
                                  &result);
    }
  }
  for (output_index = 0;
       status == TERLAN_NDARRAY_OK && output_index < result->numel;
       ++output_index) {
    int64_t remaining = output_index;
    int64_t segment = 0;
    int64_t source_offset = 0;
    int64_t end;
    TerlanNdArrayScalar sum = {0.0L, 0.0L};
    for (int64_t dimension = array->rank; dimension > 0; --dimension) {
      const int64_t current = dimension - 1;
      const int64_t coordinate = remaining % output_shape[current];
      remaining /= output_shape[current];
      if (current == normalized) {
        segment = coordinate;
      } else {
        source_offset += coordinate * array->strides[current];
      }
    }
    end = segment + 1 >= index_count
              ? array->shape[normalized]
              : stable_indexes[segment + 1] > stable_indexes[segment]
                    ? stable_indexes[segment + 1]
                    : stable_indexes[segment] + 1;
    for (int64_t position = stable_indexes[segment]; position < end;
         ++position) {
      TerlanNdArrayScalar value;
      const void *pointer = (const uint8_t *)array->data +
                            (source_offset + position *
                                                 array->strides[normalized]) *
                                (int64_t)dtype_size(array->dtype);
      status = dtype_load(pointer, array->dtype, &value);
      if (status != TERLAN_NDARRAY_OK) {
        break;
      }
      sum = scalar_add(sum, value);
    }
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_store_cast(array_element_pointer(result, output_index),
                                result->dtype, sum,
                                TERLAN_NDARRAY_CAST_SAME_KIND);
    }
  }
  free(output_shape);
  free(stable_indexes);
  if (status != TERLAN_NDARRAY_OK) {
    if (result != NULL) {
      (void)terlan_ndarray_v1_delete(result);
    }
    return status;
  }
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_add_reduceat_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    TerlanNdArrayHandle indexes, int64_t axis) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status =
      terlan_ndarray_v1_add_reduceat(array, indexes, axis, &result);
  return array_finish_into(destination, status, result,
                           TERLAN_NDARRAY_CAST_SAME_KIND);
}
