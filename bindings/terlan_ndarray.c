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

static TerlanNdArrayStatus checked_layout(const int64_t *shape, int64_t rank,
                                           size_t element_size,
                                           int64_t *out_numel,
                                           int64_t *out_byte_count,
                                           int64_t *strides) {
  int64_t numel = 1;
  int64_t stride = 1;
  int64_t index;

  if (out_numel == NULL || out_byte_count == NULL || rank < 0 ||
      (rank > 0 && (shape == NULL || strides == NULL)) || element_size == 0) {
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
    strides[index - 1] = stride;
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
  free(array->data);
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

size_t dtype_size(int32_t dtype) {
  switch (dtype) {
  case TERLAN_NDARRAY_DTYPE_BOOL:
    return sizeof(uint8_t);
  case TERLAN_NDARRAY_DTYPE_INT64:
    return sizeof(int64_t);
  case TERLAN_NDARRAY_DTYPE_FLOAT64:
    return sizeof(double);
  default:
    return 0;
  }
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
  status = checked_layout(array->shape, rank, element_size, &array->numel,
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
    array->data = checked_malloc((size_t)array->byte_count);
    if (array->data == NULL) {
      destroy_partial(array);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    memcpy(array->data, values, (size_t)array->byte_count);
  }
  array->dtype = dtype;
  array->device_type = TERLAN_NDARRAY_DEVICE_CPU;
  array->rank = rank;
  array->state = TERLAN_NDARRAY_STATE_LIVE;
  atomic_fetch_add_explicit(&live_count, 1, memory_order_relaxed);
  *out_array = array;
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
  memcpy(values, array->data, (size_t)array->byte_count);
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
  *out_contiguous = 1;
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
  free(array->data);
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
