#include "terlan_ndarray_internal.h"

#include <limits.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static _Atomic int64_t live_count = 0;
static _Atomic int64_t allocation_fail_after = -1;
static _Atomic int64_t allocation_attempt = 0;

static void storage_release(TerlanNdArrayStorage *storage);

static int allocation_must_fail(void) {
  const int64_t fail_after = atomic_load_explicit(
      &allocation_fail_after, memory_order_relaxed);
  int64_t attempt;

  if (fail_after < 0) {
    return 0;
  }
  attempt = atomic_fetch_add_explicit(&allocation_attempt, 1,
                                      memory_order_relaxed);
  return attempt >= fail_after;
}

void *checked_malloc(size_t size) {
  if (allocation_must_fail()) {
    return NULL;
  }
  return malloc(size);
}

void *checked_calloc(size_t count, size_t size) {
  if (allocation_must_fail()) {
    return NULL;
  }
  return calloc(count, size);
}

TerlanNdArrayStatus array_checked_layout(
    const int64_t *shape, int64_t rank, size_t element_size,
    int64_t *out_numel, int64_t *out_byte_count, int64_t *strides) {
  int64_t numel = 1;
  int64_t stride = 1;
  int64_t index;

  if (out_numel == NULL || out_byte_count == NULL || rank < 0 ||
      (rank > 0 && shape == NULL) || element_size == 0) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  for (index = 0; index < rank; ++index) {
    const int64_t dimension = shape[index];
    if (dimension < 0) {
      return TERLAN_NDARRAY_NEGATIVE_DIMENSION;
    }
    if (dimension == 0) {
      numel = 0;
    } else if (numel != 0) {
      if (numel > INT64_MAX / dimension) {
        return TERLAN_NDARRAY_SIZE_OVERFLOW;
      }
      numel *= dimension;
    }
  }
  for (index = rank; index > 0; --index) {
    const int64_t dimension = shape[index - 1];
    if (strides != NULL) {
      strides[index - 1] = stride;
    }
    if (dimension == 0) {
      stride = 0;
    } else if (stride != 0) {
      if (stride > INT64_MAX / dimension) {
        return TERLAN_NDARRAY_SIZE_OVERFLOW;
      }
      stride *= dimension;
    }
  }
  if ((uint64_t)numel > (uint64_t)INT64_MAX / element_size ||
      (uint64_t)numel > SIZE_MAX / element_size) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  *out_numel = numel;
  *out_byte_count = numel * (int64_t)element_size;
  return TERLAN_NDARRAY_OK;
}

static void destroy_partial(TerlanNdArray *array) {
  if (array == NULL) {
    return;
  }
  storage_release(array->storage);
  free(array->strides);
  free(array->shape);
  free(array);
}

TerlanNdArrayStatus require_live(TerlanNdArrayHandle array) {
  if (array == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (array->state != TERLAN_NDARRAY_STATE_LIVE) {
    return TERLAN_NDARRAY_DISPOSED;
  }
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStorage *storage_create(const void *values,
                                             int64_t byte_count) {
  TerlanNdArrayStorage *storage = checked_calloc(1, sizeof(*storage));
  if (storage == NULL) {
    return NULL;
  }
  if (byte_count > 0) {
    storage->data = checked_malloc((size_t)byte_count);
    if (storage->data == NULL) {
      free(storage);
      return NULL;
    }
    memcpy(storage->data, values, (size_t)byte_count);
  }
  storage->byte_count = byte_count;
  atomic_init(&storage->reference_count, 1);
  return storage;
}

static void storage_retain(TerlanNdArrayStorage *storage) {
  if (storage != NULL) {
    (void)atomic_fetch_add_explicit(&storage->reference_count, 1,
                                    memory_order_relaxed);
  }
}

static void storage_release(TerlanNdArrayStorage *storage) {
  if (storage != NULL &&
      atomic_fetch_sub_explicit(&storage->reference_count, 1,
                                memory_order_acq_rel) == 1) {
    if (storage->external_release != NULL) {
      storage->external_release(storage->external_context);
    } else {
      free(storage->data);
    }
    free(storage);
  }
}

TerlanNdArrayStatus terlan_ndarray_v1_create_external(
    void *data, int64_t byte_count, const int64_t *shape, int64_t rank,
    int32_t dtype, int32_t writable, void *context,
    void (*release)(void *context), int32_t (*flush)(void *context),
    TerlanNdArrayHandle *out_array) {
  TerlanNdArray *array = NULL;
  TerlanNdArrayStorage *storage = NULL;
  TerlanNdArrayStatus status;
  int64_t expected_bytes = 0;
  size_t dimensions_size;
  const size_t element_size = dtype_size(dtype);

  if (out_array == NULL || byte_count < 0 || rank < 0 ||
      (rank > 0 && shape == NULL) || (byte_count > 0 && data == NULL) ||
      element_size == 0 || (writable != 0 && writable != 1) ||
      release == NULL) {
    return element_size == 0 ? TERLAN_NDARRAY_UNSUPPORTED_DTYPE
                             : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if ((uint64_t)rank > SIZE_MAX / sizeof(*array->shape)) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  array = checked_calloc(1, sizeof(*array));
  storage = checked_calloc(1, sizeof(*storage));
  if (array == NULL || storage == NULL) {
    free(array);
    free(storage);
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  dimensions_size = (size_t)rank * sizeof(*array->shape);
  if (rank > 0) {
    array->shape = checked_malloc(dimensions_size);
    array->strides = checked_malloc(dimensions_size);
    if (array->shape == NULL || array->strides == NULL) {
      free(storage);
      destroy_partial(array);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    memcpy(array->shape, shape, dimensions_size);
  }
  status = array_checked_layout(array->shape, rank, element_size, &array->numel,
                          &expected_bytes, array->strides);
  if (status != TERLAN_NDARRAY_OK || expected_bytes != byte_count) {
    free(storage);
    destroy_partial(array);
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_ELEMENT_COUNT_MISMATCH;
  }
  atomic_init(&storage->reference_count, 1);
  storage->byte_count = byte_count;
  storage->data = data;
  storage->external_context = context;
  storage->external_release = release;
  storage->external_flush = flush;
  array->storage = storage;
  array->data = data;
  array->byte_count = byte_count;
  array->dtype = dtype;
  array->device_type = TERLAN_NDARRAY_DEVICE_CPU;
  array->rank = rank;
  array->writable = writable;
  array->state = TERLAN_NDARRAY_STATE_LIVE;
  atomic_fetch_add_explicit(&live_count, 1, memory_order_relaxed);
  *out_array = array;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_create_external_ordered(
    void *data, int64_t byte_count, const int64_t *shape, int64_t rank,
    int32_t dtype, int32_t writable, int32_t order, void *context,
    void (*release)(void *context), int32_t (*flush)(void *context),
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status;
  int64_t axis;
  int64_t stride = 1;
  if (order != TERLAN_NDARRAY_ORDER_C &&
      order != TERLAN_NDARRAY_ORDER_FORTRAN) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  status = terlan_ndarray_v1_create_external(
      data, byte_count, shape, rank, dtype, writable, context, release, flush,
      out_array);
  if (status != TERLAN_NDARRAY_OK || order == TERLAN_NDARRAY_ORDER_C) {
    return status;
  }
  for (axis = 0; axis < rank; ++axis) {
    (*out_array)->strides[axis] = stride;
    if (shape[axis] > 1) {
      stride *= shape[axis];
    }
  }
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_flush(TerlanNdArrayHandle array) {
  TerlanNdArrayStatus status = require_live(array);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  return array->storage->external_flush == NULL
             ? TERLAN_NDARRAY_OK
             : array->storage->external_flush(array->storage->external_context);
}

static int checked_add_offset(int64_t value, int64_t delta,
                              int64_t *out_value) {
  if ((delta > 0 && value > INT64_MAX - delta) ||
      (delta < 0 && value < INT64_MIN - delta)) {
    return 0;
  }
  *out_value = value + delta;
  return 1;
}

static TerlanNdArrayStatus validate_view_bounds(TerlanNdArrayHandle source,
                                                 const int64_t *shape,
                                                 const int64_t *strides,
                                                 int64_t rank,
                                                 int64_t storage_offset,
                                                 int64_t numel) {
  const size_t element_size = dtype_size(source->dtype);
  const int64_t storage_elements =
      element_size == 0 ? 0 : source->storage->byte_count / (int64_t)element_size;
  int64_t minimum = storage_offset;
  int64_t maximum = storage_offset;
  int64_t dimension;

  if (numel == 0) {
    return storage_offset >= 0 && storage_offset <= storage_elements
               ? TERLAN_NDARRAY_OK
               : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  for (dimension = 0; dimension < rank; ++dimension) {
    int64_t extent;
    if (shape[dimension] <= 1) {
      continue;
    }
    if ((strides[dimension] > 0 &&
         shape[dimension] - 1 > INT64_MAX / strides[dimension]) ||
        (strides[dimension] < 0 &&
         strides[dimension] < INT64_MIN / (shape[dimension] - 1))) {
      return TERLAN_NDARRAY_SIZE_OVERFLOW;
    }
    extent = (shape[dimension] - 1) * strides[dimension];
    if (!checked_add_offset(extent < 0 ? minimum : maximum, extent,
                            extent < 0 ? &minimum : &maximum)) {
      return TERLAN_NDARRAY_SIZE_OVERFLOW;
    }
  }
  return minimum >= 0 && maximum >= minimum && maximum < storage_elements
             ? TERLAN_NDARRAY_OK
             : TERLAN_NDARRAY_INVALID_ARGUMENT;
}


TerlanNdArrayStatus create_array(const void *values, int64_t value_count,
                                 const int64_t *shape, int64_t rank,
                                 size_t element_size, int32_t dtype,
                                 TerlanNdArrayHandle *out_array) {
  TerlanNdArray *array;
  TerlanNdArrayStatus status;
  size_t dimensions_size;

  if (out_array == NULL || value_count < 0 ||
      (value_count > 0 && values == NULL) || rank < 0 ||
      (rank > 0 && shape == NULL)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if ((uint64_t)rank > SIZE_MAX / sizeof(*array->shape)) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  dimensions_size = (size_t)rank * sizeof(*array->shape);
  array = checked_calloc(1, sizeof(*array));
  if (array == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  if (rank > 0) {
    array->shape = checked_malloc(dimensions_size);
    if (array->shape == NULL) {
      destroy_partial(array);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    array->strides = checked_malloc(dimensions_size);
    if (array->strides == NULL) {
      destroy_partial(array);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    memcpy(array->shape, shape, dimensions_size);
  }
  status = array_checked_layout(array->shape, rank, element_size, &array->numel,
                          &array->byte_count, array->strides);
  if (status != TERLAN_NDARRAY_OK) {
    destroy_partial(array);
    return status;
  }
  if (array->numel != value_count) {
    destroy_partial(array);
    return TERLAN_NDARRAY_ELEMENT_COUNT_MISMATCH;
  }
  if (array->byte_count > 0) {
    array->storage = storage_create(values, array->byte_count);
  } else {
    array->storage = storage_create(NULL, 0);
  }
  if (array->storage == NULL) {
    destroy_partial(array);
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  array->data = array->storage->data;
  array->storage_offset = 0;
  array->writable = 1;
  array->dtype = dtype;
  array->device_type = TERLAN_NDARRAY_DEVICE_CPU;
  array->rank = rank;
  array->state = TERLAN_NDARRAY_STATE_LIVE;
  atomic_fetch_add_explicit(&live_count, 1, memory_order_relaxed);
  *out_array = array;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus create_empty_array(const int64_t *shape, int64_t rank,
                                       int32_t dtype,
                                       TerlanNdArrayHandle *out_array) {
  const size_t element_size = dtype_size(dtype);
  int64_t numel = 0;
  int64_t byte_count = 0;
  void *zeroes = NULL;
  TerlanNdArrayStatus status;
  if (element_size == 0 || out_array == NULL) {
    return element_size == 0 ? TERLAN_NDARRAY_UNSUPPORTED_DTYPE
                             : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  status = array_checked_layout(shape, rank, element_size, &numel, &byte_count,
                                NULL);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (byte_count > 0) {
    zeroes = checked_calloc((size_t)numel, element_size);
    if (zeroes == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  status = create_array(zeroes, numel, shape, rank, element_size, dtype,
                        out_array);
  free(zeroes);
  return status;
}

TerlanNdArrayStatus create_view(TerlanNdArrayHandle source,
                                const int64_t *shape, const int64_t *strides,
                                int64_t rank, int64_t relative_offset,
                                int32_t writable,
                                TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(source);
  TerlanNdArray *array = NULL;
  int64_t storage_offset;
  size_t dimensions_size;
  const size_t element_size =
      status == TERLAN_NDARRAY_OK ? dtype_size(source->dtype) : 0;

  if (status != TERLAN_NDARRAY_OK || out_array == NULL || rank < 0 ||
      (rank > 0 && (shape == NULL || strides == NULL)) ||
      (writable != 0 && writable != 1)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (!checked_add_offset(source->storage_offset, relative_offset,
                          &storage_offset)) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  if ((uint64_t)rank > SIZE_MAX / sizeof(*array->shape)) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  dimensions_size = (size_t)rank * sizeof(*array->shape);
  array = checked_calloc(1, sizeof(*array));
  if (array == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  if (rank > 0) {
    array->shape = checked_malloc(dimensions_size);
    array->strides = checked_malloc(dimensions_size);
    if (array->shape == NULL || array->strides == NULL) {
      destroy_partial(array);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    memcpy(array->shape, shape, dimensions_size);
    memcpy(array->strides, strides, dimensions_size);
  }
  status = array_checked_layout(array->shape, rank, element_size, &array->numel,
                          &array->byte_count, NULL);
  if (status == TERLAN_NDARRAY_OK) {
    status = validate_view_bounds(source, array->shape, array->strides, rank,
                                  storage_offset, array->numel);
  }
  if (status != TERLAN_NDARRAY_OK) {
    destroy_partial(array);
    return status;
  }
  storage_retain(source->storage);
  array->storage = source->storage;
  array->storage_offset = storage_offset;
  array->data = array->storage->data == NULL
                    ? NULL
                    : (uint8_t *)array->storage->data +
                          (size_t)storage_offset * element_size;
  array->dtype = source->dtype;
  array->device_type = source->device_type;
  array->rank = rank;
  array->writable = source->writable != 0 && writable != 0;
  array->state = TERLAN_NDARRAY_STATE_LIVE;
  atomic_fetch_add_explicit(&live_count, 1, memory_order_relaxed);
  *out_array = array;
  return TERLAN_NDARRAY_OK;
}

static int array_is_order_contiguous(TerlanNdArrayHandle array,
                                     int fortran_order) {
  int64_t expected = 1;
  int64_t position;
  if (array == NULL || array->numel == 0) {
    return 1;
  }
  for (position = 0; position < array->rank; ++position) {
    const int64_t index = fortran_order != 0 ? position
                                             : array->rank - position - 1;
    if (array->shape[index] > 1 && array->strides[index] != expected) {
      return 0;
    }
    if (array->shape[index] > 1) {
      expected *= array->shape[index];
    }
  }
  return 1;
}

int array_is_contiguous(TerlanNdArrayHandle array) {
  return array_is_order_contiguous(array, 0);
}

int array_is_fortran_contiguous(TerlanNdArrayHandle array) {
  return array_is_order_contiguous(array, 1);
}

void *array_element_pointer(TerlanNdArrayHandle array, int64_t logical_index) {
  int64_t remaining = logical_index;
  int64_t element_offset = 0;
  int64_t dimension;
  if (array == NULL || logical_index < 0 || logical_index >= array->numel) {
    return NULL;
  }
  for (dimension = array->rank; dimension > 0; --dimension) {
    const int64_t index = dimension - 1;
    const int64_t coordinate = remaining % array->shape[index];
    remaining /= array->shape[index];
    element_offset += coordinate * array->strides[index];
  }
  return (uint8_t *)array->data +
         element_offset * (int64_t)dtype_size(array->dtype);
}

const void *array_const_element_pointer(TerlanNdArrayHandle array,
                                        int64_t logical_index) {
  return array_element_pointer(array, logical_index);
}

int array_broadcast_compatible(TerlanNdArrayHandle source,
                               const int64_t *destination_shape,
                               int64_t destination_rank) {
  int64_t source_axis = source->rank - 1;
  int64_t destination_axis = destination_rank - 1;
  if (destination_rank < 0 ||
      (destination_rank > 0 && destination_shape == NULL)) {
    return 0;
  }
  for (; source_axis >= 0 && destination_axis >= 0;
       --source_axis, --destination_axis) {
    if (source->shape[source_axis] != 1 &&
        source->shape[source_axis] != destination_shape[destination_axis]) {
      return 0;
    }
  }
  for (; source_axis >= 0; --source_axis) {
    if (source->shape[source_axis] != 1) {
      return 0;
    }
  }
  return 1;
}

int64_t array_broadcast_index(TerlanNdArrayHandle source,
                              const int64_t *destination_shape,
                              int64_t destination_rank,
                              int64_t destination_index) {
  int64_t source_index = 0;
  int64_t source_multiplier = 1;
  int64_t destination_axis;
  for (destination_axis = destination_rank; destination_axis-- > 0;) {
    const int64_t dimension = destination_shape[destination_axis];
    const int64_t coordinate = destination_index % dimension;
    const int64_t source_axis =
        destination_axis - (destination_rank - source->rank);
    destination_index /= dimension;
    if (source_axis >= 0) {
      if (source->shape[source_axis] != 1) {
        source_index += coordinate * source_multiplier;
      }
      source_multiplier *= source->shape[source_axis];
    }
  }
  return source_index;
}

int array_same_shape(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs) {
  int64_t axis;
  if (lhs->rank != rhs->rank) {
    return 0;
  }
  for (axis = 0; axis < lhs->rank; ++axis) {
    if (lhs->shape[axis] != rhs->shape[axis]) {
      return 0;
    }
  }
  return 1;
}

TerlanNdArrayStatus copy_array(TerlanNdArrayHandle source,
                               TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(source);
  void *values = NULL;
  int64_t index;
  const size_t element_size =
      status == TERLAN_NDARRAY_OK ? dtype_size(source->dtype) : 0;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (source->byte_count > 0) {
    values = checked_malloc((size_t)source->byte_count);
    if (values == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    for (index = 0; index < source->numel; ++index) {
      memcpy((uint8_t *)values + (size_t)index * element_size,
             array_const_element_pointer(source, index), element_size);
    }
  }
  status = create_array(values, source->numel, source->shape, source->rank,
                        element_size, source->dtype, out_array);
  free(values);
  return status;
}

TerlanNdArrayStatus copy_array_ordered(TerlanNdArrayHandle source,
                                       int32_t dtype, int32_t order,
                                       int32_t casting,
                                       TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(source);
  TerlanNdArrayHandle output = NULL;
  int64_t *strides = NULL;
  int64_t index;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL ||
      dtype_size(dtype) == 0) {
    return status != TERLAN_NDARRAY_OK
               ? status
               : (dtype_size(dtype) == 0 ? TERLAN_NDARRAY_UNSUPPORTED_DTYPE
                                         : TERLAN_NDARRAY_INVALID_ARGUMENT);
  }
  *out_array = NULL;
  if (!dtype_can_cast(source->dtype, dtype, casting)) {
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  if (source->rank > 0) {
    strides = checked_malloc((size_t)source->rank * sizeof(*strides));
    if (strides == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  status = array_ordered_strides(source, order, strides);
  if (status == TERLAN_NDARRAY_OK) {
    status = create_empty_array(source->shape, source->rank, dtype, &output);
  }
  if (status == TERLAN_NDARRAY_OK && source->rank > 0) {
    memcpy(output->strides, strides,
           (size_t)source->rank * sizeof(*output->strides));
  }
  free(strides);
  for (index = 0; status == TERLAN_NDARRAY_OK && index < source->numel;
       ++index) {
    status = dtype_convert(array_element_pointer(output, index), dtype,
                           array_const_element_pointer(source, index),
                           source->dtype, casting);
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

static TerlanNdArrayStatus copy_values(TerlanNdArrayHandle array,
                                       int32_t expected_dtype, void **out_values,
                                       size_t *out_length) {
  TerlanNdArrayStatus status = require_live(array);
  void *values;

  if (status != TERLAN_NDARRAY_OK || out_values == NULL || out_length == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_values = NULL;
  *out_length = 0;
  if (array->dtype != expected_dtype) {
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  if (array->byte_count == 0) {
    return TERLAN_NDARRAY_OK;
  }
  values = checked_malloc((size_t)array->byte_count);
  if (values == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  if (array_is_contiguous(array)) {
    memcpy(values, array->data, (size_t)array->byte_count);
  } else {
    size_t index;
    const size_t element_size = dtype_size(array->dtype);
    for (index = 0; index < (size_t)array->numel; ++index) {
      memcpy((uint8_t *)values + index * element_size,
             array_const_element_pointer(array, (int64_t)index), element_size);
    }
  }
  *out_values = values;
  *out_length = (size_t)array->numel;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_from_bools(
    const uint8_t *values, int64_t value_count, const int64_t *shape,
    int64_t rank, TerlanNdArrayHandle *out_array) {
  int64_t index;
  if (value_count > 0 && values == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  for (index = 0; index < value_count; ++index) {
    if (values[index] > 1) {
      return TERLAN_NDARRAY_INVALID_BOOL;
    }
  }
  return create_array(values, value_count, shape, rank, sizeof(*values),
                      TERLAN_NDARRAY_DTYPE_BOOL, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_from_ints(
    const int64_t *values, int64_t value_count, const int64_t *shape,
    int64_t rank, TerlanNdArrayHandle *out_array) {
  return create_array(values, value_count, shape, rank, sizeof(*values),
                      TERLAN_NDARRAY_DTYPE_INT64, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_from_floats(
    const double *values, int64_t value_count, const int64_t *shape,
    int64_t rank, TerlanNdArrayHandle *out_array) {
  return create_array(values, value_count, shape, rank, sizeof(*values),
                      TERLAN_NDARRAY_DTYPE_FLOAT64, out_array);
}

#define TERLAN_NDARRAY_SCALAR_ACCESSOR(name, field, output_type)                \
  TerlanNdArrayStatus name(TerlanNdArrayHandle array, output_type *out_value) { \
    TerlanNdArrayStatus status = require_live(array);                           \
    if (status != TERLAN_NDARRAY_OK || out_value == NULL) {                     \
      return status != TERLAN_NDARRAY_OK ? status                              \
                                         : TERLAN_NDARRAY_INVALID_ARGUMENT;     \
    }                                                                           \
    *out_value = array->field;                                                   \
    return TERLAN_NDARRAY_OK;                                                    \
  }

TERLAN_NDARRAY_SCALAR_ACCESSOR(terlan_ndarray_v1_rank, rank, int64_t)
TERLAN_NDARRAY_SCALAR_ACCESSOR(terlan_ndarray_v1_numel, numel, int64_t)
TERLAN_NDARRAY_SCALAR_ACCESSOR(terlan_ndarray_v1_byte_count, byte_count, int64_t)
TERLAN_NDARRAY_SCALAR_ACCESSOR(terlan_ndarray_v1_dtype, dtype, int32_t)
TERLAN_NDARRAY_SCALAR_ACCESSOR(terlan_ndarray_v1_device_type, device_type,
                               int32_t)

static TerlanNdArrayStatus borrowed_int64(TerlanNdArrayHandle array,
                                           int64_t *values,
                                           int64_t **out_values) {
  TerlanNdArrayStatus status = require_live(array);
  if (status != TERLAN_NDARRAY_OK || out_values == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_values = values;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_shape(TerlanNdArrayHandle array,
                                             int64_t **out_shape) {
  return borrowed_int64(array, array == NULL ? NULL : array->shape, out_shape);
}

TerlanNdArrayStatus terlan_ndarray_v1_strides(TerlanNdArrayHandle array,
                                               int64_t **out_strides) {
  return borrowed_int64(array, array == NULL ? NULL : array->strides,
                        out_strides);
}

TerlanNdArrayStatus terlan_ndarray_v1_is_contiguous(
    TerlanNdArrayHandle array, int32_t *out_contiguous) {
  TerlanNdArrayStatus status = require_live(array);
  if (status != TERLAN_NDARRAY_OK || out_contiguous == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_contiguous = array_is_contiguous(array);
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_is_fortran_contiguous(
    TerlanNdArrayHandle array, int32_t *out_contiguous) {
  TerlanNdArrayStatus status = require_live(array);
  if (status != TERLAN_NDARRAY_OK || out_contiguous == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_contiguous = array_is_fortran_contiguous(array);
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_to_bools(
    TerlanNdArrayHandle array, uint8_t **out_values, size_t *out_length) {
  void *copy = NULL;
  TerlanNdArrayStatus status;
  if (out_values == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  status = copy_values(array, TERLAN_NDARRAY_DTYPE_BOOL, &copy, out_length);
  *out_values = copy;
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_to_ints(
    TerlanNdArrayHandle array, int64_t **out_values, size_t *out_length) {
  void *copy = NULL;
  TerlanNdArrayStatus status;
  if (out_values == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  status = copy_values(array, TERLAN_NDARRAY_DTYPE_INT64, &copy, out_length);
  *out_values = copy;
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_to_floats(
    TerlanNdArrayHandle array, double **out_values, size_t *out_length) {
  void *copy = NULL;
  TerlanNdArrayStatus status;
  if (out_values == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  status = copy_values(array, TERLAN_NDARRAY_DTYPE_FLOAT64, &copy, out_length);
  *out_values = copy;
  return status;
}

void terlan_ndarray_v1_delete_bool_values(uint8_t *values) { free(values); }

void terlan_ndarray_v1_delete_int_values(int64_t *values) { free(values); }

void terlan_ndarray_v1_delete_float_values(double *values) { free(values); }

TerlanNdArrayStatus terlan_ndarray_v1_delete(TerlanNdArrayHandle array) {
  TerlanNdArrayStatus status = require_live(array);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  array->state = TERLAN_NDARRAY_STATE_DISPOSED;
  storage_release(array->storage);
  free(array->strides);
  free(array->shape);
  atomic_fetch_sub_explicit(&live_count, 1, memory_order_relaxed);
  free(array);
  return TERLAN_NDARRAY_OK;
}

int64_t terlan_ndarray_v1_live_count(void) {
  return atomic_load_explicit(&live_count, memory_order_relaxed);
}

void terlan_ndarray_test_fail_allocation_after(int64_t successful_allocations) {
  atomic_store_explicit(&allocation_attempt, 0, memory_order_relaxed);
  atomic_store_explicit(&allocation_fail_after, successful_allocations,
                        memory_order_relaxed);
}

void terlan_ndarray_test_reset_allocation_failures(void) {
  atomic_store_explicit(&allocation_fail_after, -1, memory_order_relaxed);
  atomic_store_explicit(&allocation_attempt, 0, memory_order_relaxed);
}
