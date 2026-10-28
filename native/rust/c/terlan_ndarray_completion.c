#include "terlan_ndarray_internal.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int compare_addresses(const void *lhs, const void *rhs) {
  const uintptr_t left = *(const uintptr_t *)lhs;
  const uintptr_t right = *(const uintptr_t *)rhs;
  return (left > right) - (left < right);
}

static TerlanNdArrayStatus require_mutation_inputs(
    TerlanNdArrayHandle array, TerlanNdArrayHandle values) {
  TerlanNdArrayStatus status = require_live(array);
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(values);
  }
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (array->writable == 0) {
    return TERLAN_NDARRAY_READ_ONLY;
  }
  return array->dtype == values->dtype ? TERLAN_NDARRAY_OK
                                       : TERLAN_NDARRAY_DTYPE_MISMATCH;
}

static TerlanNdArrayStatus copyto_impl(TerlanNdArrayHandle destination,
                                       TerlanNdArrayHandle source,
                                       TerlanNdArrayHandle mask,
                                       int32_t casting) {
  TerlanNdArrayStatus status = require_live(destination);
  TerlanNdArrayHandle stable_source = source;
  TerlanNdArrayHandle stable_mask = mask;
  int64_t index;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(source);
  }
  if (status == TERLAN_NDARRAY_OK && mask != NULL) {
    status = require_live(mask);
  }
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (destination->writable == 0) {
    return TERLAN_NDARRAY_READ_ONLY;
  }
  if (!dtype_can_cast(source->dtype, destination->dtype, casting)) {
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  if (!array_broadcast_compatible(source, destination->shape,
                                  destination->rank) ||
      (mask != NULL && !array_broadcast_compatible(
                           mask, destination->shape, destination->rank))) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  if (source->storage == destination->storage) {
    status = copy_array(source, &stable_source);
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
  }
  if (mask != NULL && mask->storage == destination->storage) {
    status = copy_array(mask, &stable_mask);
    if (status != TERLAN_NDARRAY_OK) {
      if (stable_source != source) {
        (void)terlan_ndarray_v1_delete(stable_source);
      }
      return status;
    }
  }
  for (index = 0; status == TERLAN_NDARRAY_OK && index < destination->numel;
       ++index) {
    int selected = 1;
    if (stable_mask != NULL) {
      TerlanNdArrayScalar mask_value;
      status = dtype_load(
          array_const_element_pointer(
              stable_mask,
              array_broadcast_index(stable_mask, destination->shape,
                                    destination->rank, index)),
          stable_mask->dtype, &mask_value);
      selected = status == TERLAN_NDARRAY_OK &&
                 (mask_value.real != 0.0L || mask_value.imag != 0.0L);
    }
    if (selected && status == TERLAN_NDARRAY_OK) {
      TerlanNdArrayScalar value;
      status = dtype_load(
          array_const_element_pointer(
              stable_source,
              array_broadcast_index(stable_source, destination->shape,
                                    destination->rank, index)),
          stable_source->dtype, &value);
      if (status == TERLAN_NDARRAY_OK) {
        status = dtype_store_cast(array_element_pointer(destination, index),
                                  destination->dtype, value, casting);
      }
    }
  }
  if (stable_source != source) {
    (void)terlan_ndarray_v1_delete(stable_source);
  }
  if (stable_mask != mask) {
    (void)terlan_ndarray_v1_delete(stable_mask);
  }
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_copyto(TerlanNdArrayHandle destination,
                                              TerlanNdArrayHandle source) {
  return copyto_impl(destination, source, NULL,
                     TERLAN_NDARRAY_CAST_SAME_KIND);
}

TerlanNdArrayStatus terlan_ndarray_v1_copyto_casting(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle source,
    int32_t casting) {
  return copyto_impl(destination, source, NULL, casting);
}

TerlanNdArrayStatus terlan_ndarray_v1_copyto_where(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle source,
    TerlanNdArrayHandle mask, int32_t casting) {
  if (mask == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  return copyto_impl(destination, source, mask, casting);
}

static TerlanNdArrayStatus require_mask_mutation(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask,
    TerlanNdArrayHandle values) {
  TerlanNdArrayStatus status = require_mutation_inputs(array, values);
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(mask);
  }
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (mask->dtype != TERLAN_NDARRAY_DTYPE_BOOL ||
      mask->numel != array->numel || values->numel == 0) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus stabilize_mutation_inputs(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask,
    TerlanNdArrayHandle values, TerlanNdArrayHandle *out_mask,
    TerlanNdArrayHandle *out_values) {
  TerlanNdArrayStatus status = TERLAN_NDARRAY_OK;
  *out_mask = mask;
  *out_values = values;
  if (array->storage == mask->storage) {
    status = copy_array(mask, out_mask);
  }
  if (status == TERLAN_NDARRAY_OK && array->storage == values->storage) {
    status = copy_array(values, out_values);
  }
  if (status != TERLAN_NDARRAY_OK && *out_mask != mask) {
    (void)terlan_ndarray_v1_delete(*out_mask);
    *out_mask = mask;
  }
  return status;
}

static void release_stable_mutation_inputs(TerlanNdArrayHandle original_mask,
                                           TerlanNdArrayHandle original_values,
                                           TerlanNdArrayHandle stable_mask,
                                           TerlanNdArrayHandle stable_values) {
  if (stable_mask != original_mask) {
    (void)terlan_ndarray_v1_delete(stable_mask);
  }
  if (stable_values != original_values) {
    (void)terlan_ndarray_v1_delete(stable_values);
  }
}

TerlanNdArrayStatus terlan_ndarray_v1_place(TerlanNdArrayHandle array,
                                             TerlanNdArrayHandle mask,
                                             TerlanNdArrayHandle values) {
  TerlanNdArrayStatus status = require_mask_mutation(array, mask, values);
  TerlanNdArrayHandle stable_mask;
  TerlanNdArrayHandle stable_values;
  int64_t index;
  int64_t selected = 0;
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  status = stabilize_mutation_inputs(array, mask, values, &stable_mask,
                                     &stable_values);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (index = 0; index < array->numel; ++index) {
    if (*(const uint8_t *)array_const_element_pointer(stable_mask, index) != 0) {
      memcpy(array_element_pointer(array, index),
             array_const_element_pointer(stable_values,
                                         selected++ % stable_values->numel),
             dtype_size(array->dtype));
    }
  }
  release_stable_mutation_inputs(mask, values, stable_mask, stable_values);
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_putmask(TerlanNdArrayHandle array,
                                               TerlanNdArrayHandle mask,
                                               TerlanNdArrayHandle values) {
  TerlanNdArrayStatus status = require_mask_mutation(array, mask, values);
  TerlanNdArrayHandle stable_mask;
  TerlanNdArrayHandle stable_values;
  int64_t index;
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  status = stabilize_mutation_inputs(array, mask, values, &stable_mask,
                                     &stable_values);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (index = 0; index < array->numel; ++index) {
    if (*(const uint8_t *)array_const_element_pointer(stable_mask, index) != 0) {
      memcpy(array_element_pointer(array, index),
             array_const_element_pointer(stable_values,
                                         index % stable_values->numel),
             dtype_size(array->dtype));
    }
  }
  release_stable_mutation_inputs(mask, values, stable_mask, stable_values);
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_may_share_memory(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, bool *out_value) {
  TerlanNdArrayStatus status = require_live(lhs);
  uintptr_t lhs_min = UINTPTR_MAX;
  uintptr_t lhs_max = 0;
  uintptr_t rhs_min = UINTPTR_MAX;
  uintptr_t rhs_max = 0;
  int64_t index;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(rhs);
  }
  if (status != TERLAN_NDARRAY_OK || out_value == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_value = false;
  if (lhs->storage != rhs->storage || lhs->numel == 0 || rhs->numel == 0) {
    return TERLAN_NDARRAY_OK;
  }
  for (index = 0; index < lhs->numel; ++index) {
    const uintptr_t address =
        (uintptr_t)array_const_element_pointer(lhs, index);
    lhs_min = address < lhs_min ? address : lhs_min;
    lhs_max = address > lhs_max ? address : lhs_max;
  }
  for (index = 0; index < rhs->numel; ++index) {
    const uintptr_t address =
        (uintptr_t)array_const_element_pointer(rhs, index);
    rhs_min = address < rhs_min ? address : rhs_min;
    rhs_max = address > rhs_max ? address : rhs_max;
  }
  *out_value = lhs_min <= rhs_max && rhs_min <= lhs_max;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_shares_memory(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, bool *out_value) {
  TerlanNdArrayStatus status = require_live(lhs);
  TerlanNdArrayHandle smaller;
  TerlanNdArrayHandle larger;
  uintptr_t *addresses;
  int64_t index;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(rhs);
  }
  if (status != TERLAN_NDARRAY_OK || out_value == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_value = false;
  if (lhs->storage != rhs->storage || lhs->numel == 0 || rhs->numel == 0) {
    return TERLAN_NDARRAY_OK;
  }
  smaller = lhs->numel <= rhs->numel ? lhs : rhs;
  larger = smaller == lhs ? rhs : lhs;
  if ((uint64_t)smaller->numel > SIZE_MAX / sizeof(*addresses)) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  addresses = checked_malloc((size_t)smaller->numel * sizeof(*addresses));
  if (addresses == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  for (index = 0; index < smaller->numel; ++index) {
    addresses[index] =
        (uintptr_t)array_const_element_pointer(smaller, index);
  }
  qsort(addresses, (size_t)smaller->numel, sizeof(*addresses),
        compare_addresses);
  for (index = 0; index < larger->numel; ++index) {
    const uintptr_t address =
        (uintptr_t)array_const_element_pointer(larger, index);
    if (bsearch(&address, addresses, (size_t)smaller->numel,
                sizeof(*addresses), compare_addresses) != NULL) {
      *out_value = true;
      break;
    }
  }
  free(addresses);
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus exact_overlap_with_work(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, int64_t max_work,
    bool *out_value) {
  int64_t lhs_index;
  int64_t rhs_index;
  int64_t work = 0;
  if (max_work < -1) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_value = false;
  if (lhs->storage != rhs->storage || lhs->numel == 0 || rhs->numel == 0) {
    return TERLAN_NDARRAY_OK;
  }
  for (lhs_index = 0; lhs_index < lhs->numel; ++lhs_index) {
    const void *lhs_address = array_const_element_pointer(lhs, lhs_index);
    for (rhs_index = 0; rhs_index < rhs->numel; ++rhs_index) {
      if (max_work >= 0 && work >= max_work) {
        return TERLAN_NDARRAY_TOO_HARD;
      }
      ++work;
      if (lhs_address == array_const_element_pointer(rhs, rhs_index)) {
        *out_value = true;
        return TERLAN_NDARRAY_OK;
      }
    }
  }
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_may_share_memory_with_work(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, int64_t max_work,
    bool *out_value) {
  TerlanNdArrayStatus status;
  if (out_value == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_value = false;
  status = require_live(lhs);
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(rhs);
  }
  if (status != TERLAN_NDARRAY_OK || max_work < -1) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (max_work == 0) {
    return terlan_ndarray_v1_may_share_memory(lhs, rhs, out_value);
  }
  status = exact_overlap_with_work(lhs, rhs, max_work, out_value);
  if (status == TERLAN_NDARRAY_TOO_HARD) {
    *out_value = true;
    return TERLAN_NDARRAY_OK;
  }
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_shares_memory_with_work(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, int64_t max_work,
    bool *out_value) {
  TerlanNdArrayStatus status;
  if (out_value == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_value = false;
  status = require_live(lhs);
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(rhs);
  }
  if (status != TERLAN_NDARRAY_OK || max_work < -1) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (max_work == -1) {
    return terlan_ndarray_v1_shares_memory(lhs, rhs, out_value);
  }
  return exact_overlap_with_work(lhs, rhs, max_work, out_value);
}

TerlanNdArrayStatus terlan_ndarray_v1_internal_overlap(
    TerlanNdArrayHandle array, bool *out_value) {
  TerlanNdArrayStatus status = require_live(array);
  uintptr_t *addresses;
  int64_t index;
  if (status != TERLAN_NDARRAY_OK || out_value == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_value = false;
  if (array->numel <= 1) {
    return TERLAN_NDARRAY_OK;
  }
  if ((uint64_t)array->numel > SIZE_MAX / sizeof(*addresses)) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  addresses = checked_malloc((size_t)array->numel * sizeof(*addresses));
  if (addresses == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  for (index = 0; index < array->numel; ++index) {
    addresses[index] = (uintptr_t)array_const_element_pointer(array, index);
  }
  qsort(addresses, (size_t)array->numel, sizeof(*addresses),
        compare_addresses);
  for (index = 1; index < array->numel; ++index) {
    if (addresses[index - 1] == addresses[index]) {
      *out_value = true;
      break;
    }
  }
  free(addresses);
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayHandle identity_array(uint64_t identity) {
  return (TerlanNdArrayHandle)(uintptr_t)identity;
}

static TerlanNdArrayStatus broadcast_shape(
    TerlanNdArrayHandle indexes, const uint64_t *choices, int64_t choice_count,
    int64_t **out_shape, int64_t *out_rank) {
  int64_t rank = indexes->rank;
  int64_t choice_index;
  int64_t *shape;
  for (choice_index = 0; choice_index < choice_count; ++choice_index) {
    TerlanNdArrayHandle choice = identity_array(choices[choice_index]);
    TerlanNdArrayStatus status = require_live(choice);
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
    if (choice->rank > rank) {
      rank = choice->rank;
    }
  }
  shape = rank == 0 ? NULL : checked_malloc((size_t)rank * sizeof(*shape));
  if (rank > 0 && shape == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  for (int64_t axis = 0; axis < rank; ++axis) {
    shape[axis] = 1;
  }
  for (choice_index = -1; choice_index < choice_count; ++choice_index) {
    TerlanNdArrayHandle array =
        choice_index < 0 ? indexes : identity_array(choices[choice_index]);
    int64_t offset = rank - array->rank;
    for (int64_t axis = 0; axis < array->rank; ++axis) {
      int64_t output_axis = offset + axis;
      int64_t dimension = array->shape[axis];
      if (shape[output_axis] == 1) {
        shape[output_axis] = dimension;
      } else if (dimension != 1 && dimension != shape[output_axis]) {
        free(shape);
        return TERLAN_NDARRAY_SHAPE_MISMATCH;
      }
    }
  }
  *out_shape = shape;
  *out_rank = rank;
  return TERLAN_NDARRAY_OK;
}

static int64_t broadcast_index(TerlanNdArrayHandle array,
                               const int64_t *output_shape,
                               int64_t output_rank,
                               int64_t output_index) {
  int64_t source_index = 0;
  int64_t source_stride = 1;
  int64_t offset = output_rank - array->rank;
  for (int64_t axis = output_rank; axis-- > 0;) {
    int64_t coordinate = output_index % output_shape[axis];
    int64_t source_axis = axis - offset;
    output_index /= output_shape[axis];
    if (source_axis >= 0) {
      if (array->shape[source_axis] != 1) {
        source_index += coordinate * source_stride;
      }
      source_stride *= array->shape[source_axis];
    }
  }
  return source_index;
}

static TerlanNdArrayStatus choice_index(TerlanNdArrayScalar scalar,
                                        int64_t count, int32_t mode,
                                        int64_t *out_index) {
  int64_t index;
  if (scalar.imag != 0.0L || !isfinite((double)scalar.real) ||
      scalar.real < INT64_MIN || scalar.real > INT64_MAX ||
      truncl(scalar.real) != scalar.real) {
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  index = (int64_t)scalar.real;
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
  } else if (index < 0 || index >= count) {
    return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
  }
  *out_index = index;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_choose(
    TerlanNdArrayHandle indexes, const uint64_t *choices,
    int64_t choice_count, int32_t mode, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(indexes);
  int64_t *shape = NULL;
  int64_t rank = 0;
  int32_t dtype;
  int64_t output_index;
  if (out_array != NULL) {
    *out_array = NULL;
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL || choice_count <= 0 ||
      choices == NULL || !dtype_is_integer(indexes->dtype) || mode < 0 ||
      mode > 2) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  status = broadcast_shape(indexes, choices, choice_count, &shape, &rank);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  dtype = identity_array(choices[0])->dtype;
  for (int64_t index = 1; index < choice_count; ++index) {
    dtype = dtype_promote(dtype, identity_array(choices[index])->dtype);
    if (dtype == 0) {
      free(shape);
      return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
    }
  }
  status = create_empty_array(shape, rank, dtype, out_array);
  if (status != TERLAN_NDARRAY_OK) {
    free(shape);
    return status;
  }
  for (output_index = 0; output_index < (*out_array)->numel; ++output_index) {
    TerlanNdArrayScalar selected;
    TerlanNdArrayScalar value;
    int64_t selected_index;
    TerlanNdArrayHandle choice;
    status = dtype_load(array_const_element_pointer(
                            indexes, broadcast_index(indexes, shape, rank,
                                                     output_index)),
                        indexes->dtype, &selected);
    if (status == TERLAN_NDARRAY_OK) {
      status = choice_index(selected, choice_count, mode, &selected_index);
    }
    if (status != TERLAN_NDARRAY_OK) {
      break;
    }
    choice = identity_array(choices[selected_index]);
    status = dtype_load(array_const_element_pointer(
                            choice, broadcast_index(choice, shape, rank,
                                                    output_index)),
                        choice->dtype, &value);
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_store(array_element_pointer(*out_array, output_index),
                           dtype, value);
    }
    if (status != TERLAN_NDARRAY_OK) {
      break;
    }
  }
  free(shape);
  if (status != TERLAN_NDARRAY_OK) {
    (void)terlan_ndarray_v1_delete(*out_array);
    *out_array = NULL;
  }
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_choose_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle indexes,
    const uint64_t *choices, int64_t choice_count, int32_t mode) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status = terlan_ndarray_v1_choose(
      indexes, choices, choice_count, mode, &result);
  return array_finish_into(destination, status, result,
                           TERLAN_NDARRAY_CAST_UNSAFE);
}

TerlanNdArrayStatus terlan_ndarray_v1_fill(
    TerlanNdArrayHandle array, TerlanNdArrayHandle value) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayScalar scalar;
  int64_t index;
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  status = require_live(value);
  if (status != TERLAN_NDARRAY_OK || value->numel != 1) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (array->writable == 0) {
    return TERLAN_NDARRAY_READ_ONLY;
  }
  status = dtype_load(array_const_element_pointer(value, 0), value->dtype,
                      &scalar);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (index = 0; index < array->numel; ++index) {
    status = dtype_store_cast(array_element_pointer(array, index),
                              array->dtype, scalar,
                              TERLAN_NDARRAY_CAST_UNSAFE);
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
  }
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_resize(
    TerlanNdArrayHandle array, const int64_t *shape, int64_t rank,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t index;
  const TerlanNdArrayScalar zero = {0.0L, 0.0L};
  if (out_array != NULL) {
    *out_array = NULL;
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  status = create_empty_array(shape, rank, array->dtype, out_array);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (index = 0; index < (*out_array)->numel; ++index) {
    TerlanNdArrayScalar value = zero;
    if (array->numel > 0) {
      status = dtype_load(
          array_const_element_pointer(array, index % array->numel),
          array->dtype, &value);
    }
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_store(array_element_pointer(*out_array, index),
                           array->dtype, value);
    }
    if (status != TERLAN_NDARRAY_OK) {
      (void)terlan_ndarray_v1_delete(*out_array);
      *out_array = NULL;
      return status;
    }
  }
  return TERLAN_NDARRAY_OK;
}
