#include "include/terlan_ndarray.h"

#include <limits.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct TerlanNdArray {
  int32_t dtype;
  int64_t rank;
  int64_t numel;
  int64_t *shape;
  void *data;
};

static _Atomic int64_t live_count = 0;

static TerlanNdArrayStatus checked_numel(const int64_t *shape, int64_t rank,
                                         int64_t *out_numel) {
  int64_t result = 1;
  int64_t index;

  if (out_numel == NULL || rank < 0 || (rank > 0 && shape == NULL)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  for (index = 0; index < rank; ++index) {
    const int64_t dimension = shape[index];
    if (dimension < 0) {
      return TERLAN_NDARRAY_NEGATIVE_DIMENSION;
    }
    if (dimension == 0) {
      result = 0;
      continue;
    }
    if (result > INT64_MAX / dimension) {
      return TERLAN_NDARRAY_SIZE_OVERFLOW;
    }
    result *= dimension;
  }
  *out_numel = result;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus create_array(const void *values, int64_t value_count,
                                        const int64_t *shape, int64_t rank,
                                        size_t element_size, int32_t dtype,
                                        TerlanNdArrayHandle *out_array) {
  TerlanNdArray *array;
  TerlanNdArrayStatus status;
  int64_t numel;
  size_t byte_count;

  if (out_array == NULL || value_count < 0 ||
      (value_count > 0 && values == NULL)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = checked_numel(shape, rank, &numel);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (numel != value_count) {
    return TERLAN_NDARRAY_ELEMENT_COUNT_MISMATCH;
  }
  if ((uint64_t)numel > SIZE_MAX / element_size) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  byte_count = (size_t)numel * element_size;
  array = calloc(1, sizeof(*array));
  if (array == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  if (rank > 0) {
    if ((uint64_t)rank > SIZE_MAX / sizeof(*array->shape)) {
      free(array);
      return TERLAN_NDARRAY_SIZE_OVERFLOW;
    }
    array->shape = malloc((size_t)rank * sizeof(*array->shape));
    if (array->shape == NULL) {
      free(array);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    memcpy(array->shape, shape, (size_t)rank * sizeof(*array->shape));
  }
  if (byte_count > 0) {
    array->data = malloc(byte_count);
    if (array->data == NULL) {
      free(array->shape);
      free(array);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    memcpy(array->data, values, byte_count);
  }
  array->dtype = dtype;
  array->rank = rank;
  array->numel = numel;
  atomic_fetch_add_explicit(&live_count, 1, memory_order_relaxed);
  *out_array = array;
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

TerlanNdArrayStatus terlan_ndarray_v1_rank(TerlanNdArrayHandle array,
                                            int64_t *out_rank) {
  if (array == NULL || out_rank == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_rank = array->rank;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_shape(TerlanNdArrayHandle array,
                                             int64_t **out_shape) {
  if (array == NULL || out_shape == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_shape = array->shape;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_numel(TerlanNdArrayHandle array,
                                             int64_t *out_numel) {
  if (array == NULL || out_numel == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_numel = array->numel;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_dtype(TerlanNdArrayHandle array,
                                             int32_t *out_dtype) {
  if (array == NULL || out_dtype == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_dtype = array->dtype;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_is_contiguous(
    TerlanNdArrayHandle array, int32_t *out_contiguous) {
  if (array == NULL || out_contiguous == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_contiguous = 1;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_delete(TerlanNdArrayHandle array) {
  if (array == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  free(array->data);
  free(array->shape);
  free(array);
  atomic_fetch_sub_explicit(&live_count, 1, memory_order_relaxed);
  return TERLAN_NDARRAY_OK;
}

int64_t terlan_ndarray_v1_live_count(void) {
  return atomic_load_explicit(&live_count, memory_order_relaxed);
}
