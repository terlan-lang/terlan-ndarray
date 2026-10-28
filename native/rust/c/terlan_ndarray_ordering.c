#include "terlan_ndarray_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct TerlanNdArrayOrderEntry {
  TerlanNdArrayScalar value;
  int64_t index;
} TerlanNdArrayOrderEntry;

static TerlanNdArrayStatus normalize_order_axis(int64_t axis, int64_t rank,
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

static int compare_order_entries(const void *lhs_pointer,
                                 const void *rhs_pointer) {
  const TerlanNdArrayOrderEntry *lhs = lhs_pointer;
  const TerlanNdArrayOrderEntry *rhs = rhs_pointer;
  int result = scalar_order_compare(lhs->value, rhs->value);
  if (result == 0) {
    result = lhs->index < rhs->index ? -1 : lhs->index > rhs->index;
  }
  return result;
}

static int compare_order_entries_unstable(const void *lhs_pointer,
                                          const void *rhs_pointer) {
  const TerlanNdArrayOrderEntry *lhs = lhs_pointer;
  const TerlanNdArrayOrderEntry *rhs = rhs_pointer;
  return scalar_order_compare(lhs->value, rhs->value);
}

static TerlanNdArrayStatus checked_axis_geometry(TerlanNdArrayHandle array,
                                                 int64_t axis,
                                                 int64_t *out_outer,
                                                 int64_t *out_inner) {
  int64_t outer = 1;
  int64_t inner = 1;
  int64_t dimension;
  for (dimension = 0; dimension < axis; ++dimension) {
    if (array->shape[dimension] != 0 &&
        outer > INT64_MAX / array->shape[dimension]) {
      return TERLAN_NDARRAY_SIZE_OVERFLOW;
    }
    outer *= array->shape[dimension];
  }
  for (dimension = axis + 1; dimension < array->rank; ++dimension) {
    if (array->shape[dimension] != 0 &&
        inner > INT64_MAX / array->shape[dimension]) {
      return TERLAN_NDARRAY_SIZE_OVERFLOW;
    }
    inner *= array->shape[dimension];
  }
  *out_outer = outer;
  *out_inner = inner;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus order_array(TerlanNdArrayHandle array, int64_t axis,
                                       int return_indices, int stable,
                                       TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayOrderEntry *entries = NULL;
  int64_t normalized;
  int64_t outer;
  int64_t inner;
  int64_t outer_index;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = normalize_order_axis(axis, array->rank, &normalized);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  status = checked_axis_geometry(array, normalized, &outer, &inner);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (array->shape[normalized] > 0) {
    entries = checked_malloc((size_t)array->shape[normalized] *
                             sizeof(*entries));
    if (entries == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  status = create_empty_array(array->shape, array->rank,
                              return_indices ? TERLAN_NDARRAY_DTYPE_INT64
                                             : array->dtype,
                              &result);
  if (status != TERLAN_NDARRAY_OK) {
    free(entries);
    return status;
  }
  for (outer_index = 0; outer_index < outer; ++outer_index) {
    int64_t inner_index;
    for (inner_index = 0; inner_index < inner; ++inner_index) {
      int64_t axis_index;
      for (axis_index = 0; axis_index < array->shape[normalized];
           ++axis_index) {
        const int64_t logical_index =
            (outer_index * array->shape[normalized] + axis_index) * inner +
            inner_index;
        entries[axis_index].index = axis_index;
        status = dtype_load(array_const_element_pointer(array, logical_index),
                            array->dtype, &entries[axis_index].value);
        if (status != TERLAN_NDARRAY_OK) {
          goto fail;
        }
      }
      if (array->shape[normalized] > 1) {
        qsort(entries, (size_t)array->shape[normalized], sizeof(*entries),
              stable != 0 ? compare_order_entries
                          : compare_order_entries_unstable);
      }
      for (axis_index = 0; axis_index < array->shape[normalized];
           ++axis_index) {
        const int64_t logical_index =
            (outer_index * array->shape[normalized] + axis_index) * inner +
            inner_index;
        if (return_indices) {
          memcpy(array_element_pointer(result, logical_index),
                 &entries[axis_index].index, sizeof(entries[axis_index].index));
        } else {
          status = dtype_store(array_element_pointer(result, logical_index),
                               result->dtype, entries[axis_index].value);
          if (status != TERLAN_NDARRAY_OK) {
            goto fail;
          }
        }
      }
    }
  }
  free(entries);
  *out_array = result;
  return TERLAN_NDARRAY_OK;

fail:
  free(entries);
  (void)terlan_ndarray_v1_delete(result);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_sort(
    TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array) {
  return order_array(array, axis, 0, 1, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_argsort(
    TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array) {
  return order_array(array, axis, 1, 1, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_sort_unstable(
    TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array) {
  return order_array(array, axis, 0, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_argsort_unstable(
    TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array) {
  return order_array(array, axis, 1, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_sort_flat(
    TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayHandle flattened = NULL;
  TerlanNdArrayStatus status = terlan_ndarray_v1_ravel(array, &flattened);
  if (out_array == NULL) {
    if (flattened != NULL) {
      (void)terlan_ndarray_v1_delete(flattened);
    }
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (status == TERLAN_NDARRAY_OK) {
    status = order_array(flattened, 0, 0, 1, out_array);
  }
  if (flattened != NULL) {
    (void)terlan_ndarray_v1_delete(flattened);
  }
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_argsort_flat(
    TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayHandle flattened = NULL;
  TerlanNdArrayStatus status = terlan_ndarray_v1_ravel(array, &flattened);
  if (out_array == NULL) {
    if (flattened != NULL) {
      (void)terlan_ndarray_v1_delete(flattened);
    }
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (status == TERLAN_NDARRAY_OK) {
    status = order_array(flattened, 0, 1, 1, out_array);
  }
  if (flattened != NULL) {
    (void)terlan_ndarray_v1_delete(flattened);
  }
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_sort_inplace(
    TerlanNdArrayHandle array, int64_t axis) {
  TerlanNdArrayHandle sorted = NULL;
  TerlanNdArrayStatus status = order_array(array, axis, 0, 1, &sorted);
  if (status == TERLAN_NDARRAY_OK) {
    status = terlan_ndarray_v1_copyto(array, sorted);
  }
  if (sorted != NULL) {
    (void)terlan_ndarray_v1_delete(sorted);
  }
  return status;
}

static TerlanNdArrayStatus partition_array(TerlanNdArrayHandle array,
                                           TerlanNdArrayHandle kths,
                                           int64_t scalar_kth, int64_t axis,
                                           int return_indices,
                                           TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t normalized_axis;
  int64_t kth_count = kths == NULL ? 1 : kths->numel;
  int64_t kth_index;
  if (out_array != NULL) {
    *out_array = NULL;
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  status = normalize_order_axis(axis, array->rank, &normalized_axis);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (kths != NULL) {
    status = require_live(kths);
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
    if (!dtype_is_integer(kths->dtype)) {
      return TERLAN_NDARRAY_DTYPE_MISMATCH;
    }
  }
  for (kth_index = 0; kth_index < kth_count; ++kth_index) {
    int64_t kth = scalar_kth;
    if (kths != NULL) {
      status = array_load_int64(kths, kth_index, &kth);
      if (status != TERLAN_NDARRAY_OK) {
        return status;
      }
    }
    if (kth < 0) {
      kth += array->shape[normalized_axis];
    }
    if (array->numel != 0 &&
        (kth < 0 || kth >= array->shape[normalized_axis])) {
      return TERLAN_NDARRAY_INDEX_OUT_OF_RANGE;
    }
    if (array->numel == 0 && kth != 0) {
      return TERLAN_NDARRAY_INDEX_OUT_OF_RANGE;
    }
  }
  /* A complete ordering satisfies the stronger partition postcondition. */
  return order_array(array, normalized_axis, return_indices, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_partition(
    TerlanNdArrayHandle array, int64_t kth, int64_t axis,
    TerlanNdArrayHandle *out_array) {
  return partition_array(array, NULL, kth, axis, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_argpartition(
    TerlanNdArrayHandle array, int64_t kth, int64_t axis,
    TerlanNdArrayHandle *out_array) {
  return partition_array(array, NULL, kth, axis, 1, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_partition_kths(
    TerlanNdArrayHandle array, TerlanNdArrayHandle kths, int64_t axis,
    TerlanNdArrayHandle *out_array) {
  return partition_array(array, kths, 0, axis, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_argpartition_kths(
    TerlanNdArrayHandle array, TerlanNdArrayHandle kths, int64_t axis,
    TerlanNdArrayHandle *out_array) {
  return partition_array(array, kths, 0, axis, 1, out_array);
}

static TerlanNdArrayStatus partition_flattened(
    TerlanNdArrayHandle array, TerlanNdArrayHandle kths, int64_t scalar_kth,
    int return_indices, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayHandle flattened = NULL;
  TerlanNdArrayStatus status;
  if (out_array == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = terlan_ndarray_v1_ravel(array, &flattened);
  if (status == TERLAN_NDARRAY_OK) {
    status = partition_array(flattened, kths, scalar_kth, 0, return_indices,
                             out_array);
  }
  if (flattened != NULL) {
    (void)terlan_ndarray_v1_delete(flattened);
  }
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_partition_flat(
    TerlanNdArrayHandle array, int64_t kth, TerlanNdArrayHandle *out_array) {
  return partition_flattened(array, NULL, kth, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_argpartition_flat(
    TerlanNdArrayHandle array, int64_t kth, TerlanNdArrayHandle *out_array) {
  return partition_flattened(array, NULL, kth, 1, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_partition_kths_flat(
    TerlanNdArrayHandle array, TerlanNdArrayHandle kths,
    TerlanNdArrayHandle *out_array) {
  return partition_flattened(array, kths, 0, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_argpartition_kths_flat(
    TerlanNdArrayHandle array, TerlanNdArrayHandle kths,
    TerlanNdArrayHandle *out_array) {
  return partition_flattened(array, kths, 0, 1, out_array);
}

static TerlanNdArrayStatus partition_inplace(
    TerlanNdArrayHandle array, TerlanNdArrayHandle kths, int64_t scalar_kth,
    int64_t axis) {
  TerlanNdArrayHandle partitioned = NULL;
  TerlanNdArrayStatus status =
      partition_array(array, kths, scalar_kth, axis, 0, &partitioned);
  if (status == TERLAN_NDARRAY_OK) {
    status = terlan_ndarray_v1_copyto(array, partitioned);
  }
  if (partitioned != NULL) {
    (void)terlan_ndarray_v1_delete(partitioned);
  }
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_partition_inplace(
    TerlanNdArrayHandle array, int64_t kth, int64_t axis) {
  return partition_inplace(array, NULL, kth, axis);
}

TerlanNdArrayStatus terlan_ndarray_v1_partition_kths_inplace(
    TerlanNdArrayHandle array, TerlanNdArrayHandle kths, int64_t axis) {
  return partition_inplace(array, kths, 0, axis);
}

static TerlanNdArrayStatus searchsorted_array(
    TerlanNdArrayHandle sorted, TerlanNdArrayHandle values,
    TerlanNdArrayHandle sorter, int32_t right,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(sorted);
  TerlanNdArrayHandle result = NULL;
  int64_t value_index;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(values);
  }
  if (status == TERLAN_NDARRAY_OK && sorter != NULL) {
    status = require_live(sorter);
  }
  if (out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (status != TERLAN_NDARRAY_OK || sorted->rank != 1 ||
      (sorter != NULL &&
       (sorter->rank != 1 || !dtype_is_integer(sorter->dtype) ||
        sorter->numel != sorted->numel)) ||
      (right != 0 && right != 1)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  status = create_empty_array(values->shape, values->rank,
                              TERLAN_NDARRAY_DTYPE_INT64, &result);
  if (status == TERLAN_NDARRAY_OK && sorter != NULL) {
    uint8_t *seen = checked_calloc((size_t)sorter->numel, 1);
    int64_t index;
    if (seen == NULL && sorter->numel > 0) {
      status = TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    for (index = 0; status == TERLAN_NDARRAY_OK && index < sorter->numel;
         ++index) {
      int64_t selected;
      status = array_load_int64(sorter, index, &selected);
      if (status == TERLAN_NDARRAY_OK &&
          (selected < 0 || selected >= sorted->numel || seen[selected] != 0)) {
        status = TERLAN_NDARRAY_INVALID_ARGUMENT;
      } else if (status == TERLAN_NDARRAY_OK) {
        seen[selected] = 1;
      }
    }
    free(seen);
  }
  for (value_index = 0;
       status == TERLAN_NDARRAY_OK && value_index < values->numel;
       ++value_index) {
    TerlanNdArrayScalar value;
    int64_t low = 0;
    int64_t high = sorted->numel;
    status = dtype_load(array_const_element_pointer(values, value_index),
                        values->dtype, &value);
    while (status == TERLAN_NDARRAY_OK && low < high) {
      const int64_t middle = low + (high - low) / 2;
      int64_t source_index = middle;
      TerlanNdArrayScalar candidate;
      int order;
      if (sorter != NULL) {
        status = array_load_int64(sorter, middle, &source_index);
        if (status != TERLAN_NDARRAY_OK) {
          break;
        }
      }
      status = dtype_load(array_const_element_pointer(sorted, source_index),
                          sorted->dtype, &candidate);
      if (status != TERLAN_NDARRAY_OK) {
        break;
      }
      order = scalar_order_compare(candidate, value);
      if (order < 0 || (right != 0 && order == 0)) {
        low = middle + 1;
      } else {
        high = middle;
      }
    }
    if (status == TERLAN_NDARRAY_OK) {
      ((int64_t *)result->data)[value_index] = low;
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

TerlanNdArrayStatus terlan_ndarray_v1_searchsorted(
    TerlanNdArrayHandle sorted, TerlanNdArrayHandle values, int32_t right,
    TerlanNdArrayHandle *out_array) {
  return searchsorted_array(sorted, values, NULL, right, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_searchsorted_with_sorter(
    TerlanNdArrayHandle array, TerlanNdArrayHandle values,
    TerlanNdArrayHandle sorter, int32_t right,
    TerlanNdArrayHandle *out_array) {
  return searchsorted_array(array, values, sorter, right, out_array);
}

static TerlanNdArrayStatus lexicographic_compare(
    TerlanNdArrayHandle keys, int64_t key_size, int64_t lhs_flat,
    int64_t rhs_flat, int64_t lhs_index, int64_t rhs_index, int *out_order) {
  int64_t key;
  for (key = keys->shape[0]; key > 0; --key) {
    TerlanNdArrayScalar lhs_value;
    TerlanNdArrayScalar rhs_value;
    TerlanNdArrayStatus status = dtype_load(
        array_const_element_pointer(keys, (key - 1) * key_size + lhs_flat),
        keys->dtype, &lhs_value);
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_load(
          array_const_element_pointer(keys, (key - 1) * key_size + rhs_flat),
          keys->dtype, &rhs_value);
    }
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
    *out_order = scalar_order_compare(lhs_value, rhs_value);
    if (*out_order != 0) {
      return TERLAN_NDARRAY_OK;
    }
  }
  *out_order = lhs_index < rhs_index ? -1 : lhs_index > rhs_index;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_lexsort_axis(
    TerlanNdArrayHandle keys, int64_t axis,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(keys);
  TerlanNdArrayHandle output = NULL;
  int64_t normalized;
  int64_t outer = 0;
  int64_t inner = 0;
  int64_t outer_index;
  if (out_array != NULL) {
    *out_array = NULL;
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL || keys->rank < 2 ||
      keys->shape[0] == 0) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  status = normalize_order_axis(axis, keys->rank - 1, &normalized);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  status = create_empty_array(keys->shape + 1, keys->rank - 1,
                              TERLAN_NDARRAY_DTYPE_INT64, &output);
  if (status == TERLAN_NDARRAY_OK) {
    status = checked_axis_geometry(output, normalized, &outer, &inner);
  }
  for (outer_index = 0; status == TERLAN_NDARRAY_OK && outer_index < outer;
       ++outer_index) {
    int64_t inner_index;
    for (inner_index = 0; status == TERLAN_NDARRAY_OK && inner_index < inner;
         ++inner_index) {
      const int64_t axis_size = output->shape[normalized];
      const int64_t base = outer_index * axis_size * inner + inner_index;
      int64_t index;
      for (index = 0; index < axis_size; ++index) {
        ((int64_t *)output->data)[base + index * inner] = index;
      }
      for (index = 1; status == TERLAN_NDARRAY_OK && index < axis_size;
           ++index) {
        const int64_t selected =
            ((int64_t *)output->data)[base + index * inner];
        int64_t position = index;
        while (position > 0) {
          const int64_t previous =
              ((int64_t *)output->data)[base + (position - 1) * inner];
          int order;
          status = lexicographic_compare(
              keys, output->numel, base + previous * inner,
              base + selected * inner, previous, selected, &order);
          if (status != TERLAN_NDARRAY_OK || order <= 0) {
            break;
          }
          ((int64_t *)output->data)[base + position * inner] = previous;
          --position;
        }
        ((int64_t *)output->data)[base + position * inner] = selected;
      }
    }
  }
  if (status != TERLAN_NDARRAY_OK) {
    if (output != NULL) {
      (void)terlan_ndarray_v1_delete(output);
    }
    return status;
  }
  *out_array = output;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_lexsort_keys(
    TerlanNdArrayHandle keys, TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_v1_lexsort_axis(keys, -1, out_array);
}
