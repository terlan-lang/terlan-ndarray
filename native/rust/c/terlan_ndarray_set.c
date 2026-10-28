#include "terlan_ndarray_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct TerlanNdArraySetEntry {
  TerlanNdArrayScalar value;
  int64_t source_index;
} TerlanNdArraySetEntry;

static int compare_set_entries(const void *lhs_pointer,
                               const void *rhs_pointer) {
  const TerlanNdArraySetEntry *lhs = lhs_pointer;
  const TerlanNdArraySetEntry *rhs = rhs_pointer;
  int order = scalar_order_compare(lhs->value, rhs->value);
  if (order == 0) {
    order = lhs->source_index < rhs->source_index
                ? -1
                : lhs->source_index > rhs->source_index;
  }
  return order;
}

static TerlanNdArrayStatus sorted_entries(TerlanNdArrayHandle array,
                                          TerlanNdArraySetEntry **out_entries) {
  TerlanNdArraySetEntry *entries = NULL;
  int64_t index;
  if (out_entries == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_entries = NULL;
  if (array->numel > 0) {
    if ((uint64_t)array->numel > SIZE_MAX / sizeof(*entries)) {
      return TERLAN_NDARRAY_SIZE_OVERFLOW;
    }
    entries = checked_malloc((size_t)array->numel * sizeof(*entries));
    if (entries == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (index = 0; index < array->numel; ++index) {
    TerlanNdArrayStatus status =
        dtype_load(array_const_element_pointer(array, index), array->dtype,
                   &entries[index].value);
    if (status != TERLAN_NDARRAY_OK) {
      free(entries);
      return status;
    }
    entries[index].source_index = index;
  }
  if (array->numel > 1) {
    qsort(entries, (size_t)array->numel, sizeof(*entries),
          compare_set_entries);
  }
  *out_entries = entries;
  return TERLAN_NDARRAY_OK;
}

static int64_t unique_count(const TerlanNdArraySetEntry *entries,
                            int64_t length) {
  int64_t count = 0;
  int64_t index;
  for (index = 0; index < length; ++index) {
    if (index == 0 ||
        scalar_order_compare(entries[index - 1].value, entries[index].value) !=
            0) {
      ++count;
    }
  }
  return count;
}

static TerlanNdArrayStatus arrays_for_unique(
    TerlanNdArrayHandle source, const TerlanNdArraySetEntry *entries,
    int64_t count, TerlanNdArrayHandle *out_unique,
    TerlanNdArrayHandle *out_indices, TerlanNdArrayHandle *out_inverse,
    TerlanNdArrayHandle *out_counts) {
  TerlanNdArrayStatus status;
  int64_t sorted_index;
  int64_t unique_index = -1;
  status = create_empty_array(&count, 1, source->dtype, out_unique);
  if (status == TERLAN_NDARRAY_OK) {
    status = create_empty_array(&count, 1, TERLAN_NDARRAY_DTYPE_INT64,
                                out_indices);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = create_empty_array(source->shape, source->rank,
                                TERLAN_NDARRAY_DTYPE_INT64, out_inverse);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = create_empty_array(&count, 1, TERLAN_NDARRAY_DTYPE_INT64,
                                out_counts);
  }
  for (sorted_index = 0;
       status == TERLAN_NDARRAY_OK && sorted_index < source->numel;
       ++sorted_index) {
    if (sorted_index == 0 ||
        scalar_order_compare(entries[sorted_index - 1].value,
                             entries[sorted_index].value) != 0) {
      ++unique_index;
      status = dtype_store(array_element_pointer(*out_unique, unique_index),
                           source->dtype, entries[sorted_index].value);
      ((int64_t *)(*out_indices)->data)[unique_index] =
          entries[sorted_index].source_index;
    } else if (entries[sorted_index].source_index <
               ((int64_t *)(*out_indices)->data)[unique_index]) {
      ((int64_t *)(*out_indices)->data)[unique_index] =
          entries[sorted_index].source_index;
    }
    ((int64_t *)(*out_inverse)->data)[entries[sorted_index].source_index] =
        unique_index;
    ++((int64_t *)(*out_counts)->data)[unique_index];
  }
  return status;
}

static void delete_if_live(TerlanNdArrayHandle array) {
  if (array != NULL) {
    (void)terlan_ndarray_v1_delete(array);
  }
}

TerlanNdArrayStatus terlan_ndarray_v1_unique_all(
    TerlanNdArrayHandle array, uint64_t **out_values, size_t *out_length) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArraySetEntry *entries = NULL;
  TerlanNdArrayHandle outputs[4] = {NULL, NULL, NULL, NULL};
  uint64_t *values = NULL;
  int64_t count;
  size_t index;
  if (status != TERLAN_NDARRAY_OK || out_values == NULL ||
      out_length == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_values = NULL;
  *out_length = 0;
  status = sorted_entries(array, &entries);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  count = unique_count(entries, array->numel);
  status = arrays_for_unique(array, entries, count, &outputs[0], &outputs[1],
                             &outputs[2], &outputs[3]);
  free(entries);
  if (status == TERLAN_NDARRAY_OK) {
    values = checked_malloc(sizeof(outputs));
    if (values == NULL) {
      status = TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  if (status != TERLAN_NDARRAY_OK) {
    for (index = 0; index < 4; ++index) {
      delete_if_live(outputs[index]);
    }
    return status;
  }
  for (index = 0; index < 4; ++index) {
    values[index] = (uint64_t)(uintptr_t)outputs[index];
  }
  *out_values = values;
  *out_length = 4;
  return TERLAN_NDARRAY_OK;
}

typedef enum TerlanNdArraySetOperation {
  TERLAN_NDARRAY_SET_UNION,
  TERLAN_NDARRAY_SET_INTERSECTION,
  TERLAN_NDARRAY_SET_DIFFERENCE,
  TERLAN_NDARRAY_SET_XOR
} TerlanNdArraySetOperation;

static int set_operation_keeps(TerlanNdArraySetOperation operation,
                               int in_lhs, int in_rhs) {
  switch (operation) {
  case TERLAN_NDARRAY_SET_UNION:
    return in_lhs || in_rhs;
  case TERLAN_NDARRAY_SET_INTERSECTION:
    return in_lhs && in_rhs;
  case TERLAN_NDARRAY_SET_DIFFERENCE:
    return in_lhs && !in_rhs;
  case TERLAN_NDARRAY_SET_XOR:
    return in_lhs != in_rhs;
  }
  return 0;
}

static int64_t next_distinct(const TerlanNdArraySetEntry *entries,
                             int64_t length, int64_t index) {
  const TerlanNdArrayScalar value = entries[index].value;
  do {
    ++index;
  } while (index < length &&
           scalar_order_compare(value, entries[index].value) == 0);
  return index;
}

static TerlanNdArrayStatus set_operation(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs,
    TerlanNdArraySetOperation operation, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(lhs);
  TerlanNdArraySetEntry *left = NULL;
  TerlanNdArraySetEntry *right = NULL;
  TerlanNdArrayScalar *selected = NULL;
  TerlanNdArrayHandle output = NULL;
  int64_t left_index = 0;
  int64_t right_index = 0;
  int64_t selected_count = 0;
  int64_t output_index;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(rhs);
  }
  if (out_array != NULL) {
    *out_array = NULL;
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL ||
      lhs->dtype != rhs->dtype) {
    return status != TERLAN_NDARRAY_OK
               ? status
               : (out_array == NULL ? TERLAN_NDARRAY_INVALID_ARGUMENT
                                    : TERLAN_NDARRAY_DTYPE_MISMATCH);
  }
  status = sorted_entries(lhs, &left);
  if (status == TERLAN_NDARRAY_OK) {
    status = sorted_entries(rhs, &right);
  }
  if (status == TERLAN_NDARRAY_OK && lhs->numel + rhs->numel > 0) {
    const uint64_t capacity = (uint64_t)lhs->numel + (uint64_t)rhs->numel;
    if (capacity > SIZE_MAX / sizeof(*selected)) {
      status = TERLAN_NDARRAY_SIZE_OVERFLOW;
    } else {
      selected = checked_malloc((size_t)capacity * sizeof(*selected));
      if (selected == NULL) {
        status = TERLAN_NDARRAY_ALLOCATION_FAILED;
      }
    }
  }
  while (status == TERLAN_NDARRAY_OK &&
         (left_index < lhs->numel || right_index < rhs->numel)) {
    int in_left;
    int in_right;
    TerlanNdArrayScalar value;
    if (right_index >= rhs->numel ||
        (left_index < lhs->numel &&
         scalar_order_compare(left[left_index].value,
                              right[right_index].value) < 0)) {
      value = left[left_index].value;
      in_left = 1;
      in_right = 0;
    } else if (left_index >= lhs->numel ||
               scalar_order_compare(left[left_index].value,
                                    right[right_index].value) > 0) {
      value = right[right_index].value;
      in_left = 0;
      in_right = 1;
    } else {
      value = left[left_index].value;
      in_left = 1;
      in_right = 1;
    }
    if (set_operation_keeps(operation, in_left, in_right)) {
      selected[selected_count++] = value;
    }
    if (in_left) {
      left_index = next_distinct(left, lhs->numel, left_index);
    }
    if (in_right) {
      right_index = next_distinct(right, rhs->numel, right_index);
    }
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = create_empty_array(&selected_count, 1, lhs->dtype, &output);
  }
  for (output_index = 0;
       status == TERLAN_NDARRAY_OK && output_index < selected_count;
       ++output_index) {
    status = dtype_store(array_element_pointer(output, output_index),
                         output->dtype, selected[output_index]);
  }
  free(selected);
  free(right);
  free(left);
  if (status != TERLAN_NDARRAY_OK) {
    delete_if_live(output);
    return status;
  }
  *out_array = output;
  return TERLAN_NDARRAY_OK;
}

#define TERLAN_NDARRAY_SET_FUNCTION(name, operation)                           \
  TerlanNdArrayStatus name(TerlanNdArrayHandle lhs,                            \
                           TerlanNdArrayHandle rhs,                            \
                           TerlanNdArrayHandle *out_array) {                   \
    return set_operation(lhs, rhs, operation, out_array);                     \
  }

TERLAN_NDARRAY_SET_FUNCTION(terlan_ndarray_v1_union,
                            TERLAN_NDARRAY_SET_UNION)
TERLAN_NDARRAY_SET_FUNCTION(terlan_ndarray_v1_intersect,
                            TERLAN_NDARRAY_SET_INTERSECTION)
TERLAN_NDARRAY_SET_FUNCTION(terlan_ndarray_v1_set_difference,
                            TERLAN_NDARRAY_SET_DIFFERENCE)
TERLAN_NDARRAY_SET_FUNCTION(terlan_ndarray_v1_set_xor,
                            TERLAN_NDARRAY_SET_XOR)

static TerlanNdArrayStatus membership(
    TerlanNdArrayHandle elements, TerlanNdArrayHandle test_elements,
    int flatten, int invert, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(elements);
  TerlanNdArraySetEntry *test = NULL;
  TerlanNdArrayHandle output = NULL;
  int64_t index;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(test_elements);
  }
  if (out_array != NULL) {
    *out_array = NULL;
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL ||
      elements->dtype != test_elements->dtype) {
    return status != TERLAN_NDARRAY_OK
               ? status
               : (out_array == NULL ? TERLAN_NDARRAY_INVALID_ARGUMENT
                                    : TERLAN_NDARRAY_DTYPE_MISMATCH);
  }
  status = sorted_entries(test_elements, &test);
  if (status == TERLAN_NDARRAY_OK) {
    const int64_t flat_shape[] = {elements->numel};
    status = create_empty_array(flatten ? flat_shape : elements->shape,
                                flatten ? 1 : elements->rank,
                                TERLAN_NDARRAY_DTYPE_BOOL, &output);
  }
  for (index = 0; status == TERLAN_NDARRAY_OK && index < elements->numel;
       ++index) {
    TerlanNdArrayScalar value;
    int64_t low = 0;
    int64_t high = test_elements->numel;
    status = dtype_load(array_const_element_pointer(elements, index),
                        elements->dtype, &value);
    while (status == TERLAN_NDARRAY_OK && low < high) {
      const int64_t middle = low + (high - low) / 2;
      const int order = scalar_order_compare(test[middle].value, value);
      if (order < 0) {
        low = middle + 1;
      } else {
        high = middle;
      }
    }
    if (status == TERLAN_NDARRAY_OK) {
      const int found = low < test_elements->numel &&
                        scalar_order_compare(test[low].value, value) == 0;
      ((uint8_t *)output->data)[index] = invert ? !found : found;
    }
  }
  free(test);
  if (status != TERLAN_NDARRAY_OK) {
    delete_if_live(output);
    return status;
  }
  *out_array = output;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_is_in(
    TerlanNdArrayHandle elements, TerlanNdArrayHandle test_elements,
    TerlanNdArrayHandle *out_array) {
  return membership(elements, test_elements, 0, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_in1d(
    TerlanNdArrayHandle elements, TerlanNdArrayHandle test_elements,
    int32_t invert, TerlanNdArrayHandle *out_array) {
  return membership(elements, test_elements, 1, invert != 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_set_difference_unique(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status;
  TerlanNdArrayHandle mask = NULL;
  TerlanNdArrayHandle output = NULL;
  int64_t selected_count = 0;
  int64_t source_index;
  int64_t output_index = 0;
  if (out_array == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = membership(lhs, rhs, 1, 1, &mask);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (source_index = 0; source_index < lhs->numel; ++source_index) {
    selected_count += ((const uint8_t *)mask->data)[source_index] != 0;
  }
  status = create_empty_array(&selected_count, 1, lhs->dtype, &output);
  for (source_index = 0;
       status == TERLAN_NDARRAY_OK && source_index < lhs->numel;
       ++source_index) {
    if (((const uint8_t *)mask->data)[source_index] != 0) {
      TerlanNdArrayScalar value;
      status = dtype_load(array_const_element_pointer(lhs, source_index),
                          lhs->dtype, &value);
      if (status == TERLAN_NDARRAY_OK) {
        status = dtype_store(array_element_pointer(output, output_index++),
                             output->dtype, value);
      }
    }
  }
  delete_if_live(mask);
  if (status != TERLAN_NDARRAY_OK) {
    delete_if_live(output);
    return status;
  }
  *out_array = output;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_intersect_all(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, uint64_t **out_values,
    size_t *out_length) {
  TerlanNdArrayStatus status = require_live(lhs);
  TerlanNdArrayHandle outputs[3] = {NULL, NULL, NULL};
  uint64_t *values = NULL;
  int64_t count = 0;
  int64_t unique_index;
  size_t output_index;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(rhs);
  }
  if (out_values != NULL) {
    *out_values = NULL;
  }
  if (out_length != NULL) {
    *out_length = 0;
  }
  if (status != TERLAN_NDARRAY_OK || out_values == NULL ||
      out_length == NULL || lhs->dtype != rhs->dtype) {
    return status != TERLAN_NDARRAY_OK
               ? status
               : (out_values == NULL || out_length == NULL
                      ? TERLAN_NDARRAY_INVALID_ARGUMENT
                      : TERLAN_NDARRAY_DTYPE_MISMATCH);
  }
  status = set_operation(lhs, rhs, TERLAN_NDARRAY_SET_INTERSECTION,
                         &outputs[0]);
  if (status == TERLAN_NDARRAY_OK) {
    count = outputs[0]->numel;
    status = create_empty_array(&count, 1, TERLAN_NDARRAY_DTYPE_INT64,
                                &outputs[1]);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = create_empty_array(&count, 1, TERLAN_NDARRAY_DTYPE_INT64,
                                &outputs[2]);
  }
  for (unique_index = 0;
       status == TERLAN_NDARRAY_OK && unique_index < count; ++unique_index) {
    TerlanNdArrayScalar needle;
    TerlanNdArrayHandle sources[2] = {lhs, rhs};
    int source;
    status = dtype_load(array_const_element_pointer(outputs[0], unique_index),
                        outputs[0]->dtype, &needle);
    for (source = 0; status == TERLAN_NDARRAY_OK && source < 2; ++source) {
      int64_t index;
      for (index = 0; index < sources[source]->numel; ++index) {
        TerlanNdArrayScalar candidate;
        status = dtype_load(array_const_element_pointer(sources[source], index),
                            sources[source]->dtype, &candidate);
        if (status != TERLAN_NDARRAY_OK ||
            scalar_order_compare(candidate, needle) == 0) {
          break;
        }
      }
      if (status == TERLAN_NDARRAY_OK) {
        ((int64_t *)outputs[source + 1]->data)[unique_index] = index;
      }
    }
  }
  if (status == TERLAN_NDARRAY_OK) {
    values = checked_malloc(sizeof(outputs));
    if (values == NULL) {
      status = TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  if (status != TERLAN_NDARRAY_OK) {
    for (output_index = 0; output_index < 3; ++output_index) {
      delete_if_live(outputs[output_index]);
    }
    return status;
  }
  for (output_index = 0; output_index < 3; ++output_index) {
    values[output_index] = (uint64_t)(uintptr_t)outputs[output_index];
  }
  *out_values = values;
  *out_length = 3;
  return TERLAN_NDARRAY_OK;
}

typedef struct TerlanNdArrayAxisSetEntry {
  TerlanNdArrayScalar *values;
  int64_t value_count;
  int64_t source_index;
} TerlanNdArrayAxisSetEntry;

static int compare_axis_set_entries(const void *lhs_pointer,
                                    const void *rhs_pointer) {
  const TerlanNdArrayAxisSetEntry *lhs = lhs_pointer;
  const TerlanNdArrayAxisSetEntry *rhs = rhs_pointer;
  int64_t index;
  for (index = 0; index < lhs->value_count; ++index) {
    const int order = scalar_order_compare(lhs->values[index],
                                           rhs->values[index]);
    if (order != 0) {
      return order;
    }
  }
  return lhs->source_index < rhs->source_index
             ? -1
             : lhs->source_index > rhs->source_index;
}

static int axis_entries_equal(const TerlanNdArrayAxisSetEntry *lhs,
                              const TerlanNdArrayAxisSetEntry *rhs) {
  int64_t index;
  for (index = 0; index < lhs->value_count; ++index) {
    if (scalar_order_compare(lhs->values[index], rhs->values[index]) != 0) {
      return 0;
    }
  }
  return 1;
}

static void free_axis_entries(TerlanNdArrayAxisSetEntry *entries,
                              int64_t count) {
  int64_t index;
  for (index = 0; index < count; ++index) {
    free(entries[index].values);
  }
  free(entries);
}

TerlanNdArrayStatus terlan_ndarray_v1_unique_axis_all(
    TerlanNdArrayHandle array, int64_t axis, uint64_t **out_values,
    size_t *out_length) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayAxisSetEntry *entries = NULL;
  TerlanNdArrayHandle outputs[4] = {NULL, NULL, NULL, NULL};
  int64_t *output_shape = NULL;
  uint64_t *values = NULL;
  int64_t axis_count;
  int64_t inner = 1;
  int64_t outer = 1;
  int64_t item_width;
  int64_t unique_count_value = 0;
  int64_t source_axis;
  int64_t sorted_index;
  int64_t unique_index = -1;
  int64_t dimension;
  size_t output_index;
  if (out_values != NULL) {
    *out_values = NULL;
  }
  if (out_length != NULL) {
    *out_length = 0;
  }
  if (status != TERLAN_NDARRAY_OK || out_values == NULL ||
      out_length == NULL || array->rank == 0) {
    return status != TERLAN_NDARRAY_OK
               ? status
               : (out_values == NULL || out_length == NULL
                      ? TERLAN_NDARRAY_INVALID_ARGUMENT
                      : TERLAN_NDARRAY_AXIS_OUT_OF_RANGE);
  }
  if (axis < 0) {
    axis += array->rank;
  }
  if (axis < 0 || axis >= array->rank) {
    return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
  }
  axis_count = array->shape[axis];
  for (dimension = 0; dimension < axis; ++dimension) {
    outer *= array->shape[dimension];
  }
  for (dimension = axis + 1; dimension < array->rank; ++dimension) {
    inner *= array->shape[dimension];
  }
  item_width = outer * inner;
  if (axis_count > 0) {
    if ((uint64_t)axis_count > SIZE_MAX / sizeof(*entries)) {
      return TERLAN_NDARRAY_SIZE_OVERFLOW;
    }
    entries = checked_calloc((size_t)axis_count, sizeof(*entries));
    if (entries == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (source_axis = 0;
       status == TERLAN_NDARRAY_OK && source_axis < axis_count; ++source_axis) {
    int64_t rest_index;
    entries[source_axis].source_index = source_axis;
    entries[source_axis].value_count = item_width;
    if (item_width > 0) {
      if ((uint64_t)item_width > SIZE_MAX / sizeof(*entries[source_axis].values)) {
        status = TERLAN_NDARRAY_SIZE_OVERFLOW;
        break;
      }
      entries[source_axis].values =
          checked_malloc((size_t)item_width * sizeof(*entries[source_axis].values));
      if (entries[source_axis].values == NULL) {
        status = TERLAN_NDARRAY_ALLOCATION_FAILED;
        break;
      }
    }
    for (rest_index = 0; rest_index < item_width; ++rest_index) {
      const int64_t outer_index = rest_index / inner;
      const int64_t inner_index = rest_index % inner;
      const int64_t logical_index =
          (outer_index * axis_count + source_axis) * inner + inner_index;
      status = dtype_load(array_const_element_pointer(array, logical_index),
                          array->dtype, &entries[source_axis].values[rest_index]);
      if (status != TERLAN_NDARRAY_OK) {
        break;
      }
    }
  }
  if (status == TERLAN_NDARRAY_OK && axis_count > 1) {
    qsort(entries, (size_t)axis_count, sizeof(*entries),
          compare_axis_set_entries);
  }
  for (sorted_index = 0;
       status == TERLAN_NDARRAY_OK && sorted_index < axis_count; ++sorted_index) {
    if (sorted_index == 0 ||
        !axis_entries_equal(&entries[sorted_index - 1], &entries[sorted_index])) {
      ++unique_count_value;
    }
  }
  if (status == TERLAN_NDARRAY_OK) {
    output_shape = checked_malloc((size_t)array->rank * sizeof(*output_shape));
    if (output_shape == NULL) {
      status = TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  if (status == TERLAN_NDARRAY_OK) {
    memcpy(output_shape, array->shape, (size_t)array->rank * sizeof(*output_shape));
    output_shape[axis] = unique_count_value;
    status = create_empty_array(output_shape, array->rank, array->dtype,
                                &outputs[0]);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = create_empty_array(&unique_count_value, 1,
                                TERLAN_NDARRAY_DTYPE_INT64, &outputs[1]);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = create_empty_array(&axis_count, 1, TERLAN_NDARRAY_DTYPE_INT64,
                                &outputs[2]);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = create_empty_array(&unique_count_value, 1,
                                TERLAN_NDARRAY_DTYPE_INT64, &outputs[3]);
  }
  for (sorted_index = 0;
       status == TERLAN_NDARRAY_OK && sorted_index < axis_count; ++sorted_index) {
    const int is_new = sorted_index == 0 ||
                       !axis_entries_equal(&entries[sorted_index - 1],
                                           &entries[sorted_index]);
    int64_t rest_index;
    if (is_new) {
      ++unique_index;
      ((int64_t *)outputs[1]->data)[unique_index] =
          entries[sorted_index].source_index;
      for (rest_index = 0; rest_index < item_width; ++rest_index) {
        const int64_t outer_index = rest_index / inner;
        const int64_t inner_index = rest_index % inner;
        const int64_t destination =
            (outer_index * unique_count_value + unique_index) * inner +
            inner_index;
        status = dtype_store(array_element_pointer(outputs[0], destination),
                             outputs[0]->dtype,
                             entries[sorted_index].values[rest_index]);
        if (status != TERLAN_NDARRAY_OK) {
          break;
        }
      }
    }
    ((int64_t *)outputs[2]->data)[entries[sorted_index].source_index] =
        unique_index;
    ++((int64_t *)outputs[3]->data)[unique_index];
  }
  free(output_shape);
  free_axis_entries(entries, axis_count);
  if (status == TERLAN_NDARRAY_OK) {
    values = checked_malloc(sizeof(outputs));
    if (values == NULL) {
      status = TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  if (status != TERLAN_NDARRAY_OK) {
    for (output_index = 0; output_index < 4; ++output_index) {
      delete_if_live(outputs[output_index]);
    }
    return status;
  }
  for (output_index = 0; output_index < 4; ++output_index) {
    values[output_index] = (uint64_t)(uintptr_t)outputs[output_index];
  }
  *out_values = values;
  *out_length = 4;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus ediff1d(
    TerlanNdArrayHandle array, TerlanNdArrayHandle begin,
    TerlanNdArrayHandle end, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle output = NULL;
  const int64_t difference_count = array == NULL || array->numel == 0
                                       ? 0
                                       : array->numel - 1;
  const int64_t begin_count = begin == NULL ? 0 : begin->numel;
  const int64_t end_count = end == NULL ? 0 : end->numel;
  int64_t output_count;
  int64_t output_index = 0;
  int64_t index;
  if (status == TERLAN_NDARRAY_OK && begin != NULL) {
    status = require_live(begin);
  }
  if (status == TERLAN_NDARRAY_OK && end != NULL) {
    status = require_live(end);
  }
  if (out_array != NULL) {
    *out_array = NULL;
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL ||
      (begin != NULL && begin->dtype != array->dtype) ||
      (end != NULL && end->dtype != array->dtype)) {
    return status != TERLAN_NDARRAY_OK
               ? status
               : (out_array == NULL ? TERLAN_NDARRAY_INVALID_ARGUMENT
                                    : TERLAN_NDARRAY_DTYPE_MISMATCH);
  }
  if (begin_count > INT64_MAX - difference_count ||
      begin_count + difference_count > INT64_MAX - end_count) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  output_count = begin_count + difference_count + end_count;
  status = create_empty_array(&output_count, 1, array->dtype, &output);
  for (index = 0; status == TERLAN_NDARRAY_OK && index < begin_count; ++index) {
    TerlanNdArrayScalar value;
    status = dtype_load(array_const_element_pointer(begin, index), begin->dtype,
                        &value);
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_store(array_element_pointer(output, output_index++),
                           output->dtype, value);
    }
  }
  for (index = 0;
       status == TERLAN_NDARRAY_OK && index < difference_count; ++index) {
    TerlanNdArrayScalar left;
    TerlanNdArrayScalar right;
    status = dtype_load(array_const_element_pointer(array, index + 1),
                        array->dtype, &left);
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_load(array_const_element_pointer(array, index),
                          array->dtype, &right);
    }
    if (status == TERLAN_NDARRAY_OK) {
      left.real -= right.real;
      left.imag -= right.imag;
      status = dtype_store(array_element_pointer(output, output_index++),
                           output->dtype, left);
    }
  }
  for (index = 0; status == TERLAN_NDARRAY_OK && index < end_count; ++index) {
    TerlanNdArrayScalar value;
    status = dtype_load(array_const_element_pointer(end, index), end->dtype,
                        &value);
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_store(array_element_pointer(output, output_index++),
                           output->dtype, value);
    }
  }
  if (status != TERLAN_NDARRAY_OK) {
    delete_if_live(output);
    return status;
  }
  *out_array = output;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_ediff1d(
    TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array) {
  return ediff1d(array, NULL, NULL, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_ediff1d_edges(
    TerlanNdArrayHandle array, TerlanNdArrayHandle begin,
    TerlanNdArrayHandle end, TerlanNdArrayHandle *out_array) {
  return ediff1d(array, begin, end, out_array);
}
