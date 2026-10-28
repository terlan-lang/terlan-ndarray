#include "terlan_ndarray_internal.h"

#include <complex.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static TerlanNdArrayStatus require_output(TerlanNdArrayHandle array,
                                          TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_empty_like(
    TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_output(array, out_array);
  return status == TERLAN_NDARRAY_OK
             ? terlan_ndarray_v1_empty(array->shape, array->rank, array->dtype,
                                       out_array)
             : status;
}

TerlanNdArrayStatus terlan_ndarray_v1_zeros_like(
    TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_output(array, out_array);
  return status == TERLAN_NDARRAY_OK
             ? terlan_ndarray_v1_zeros(array->shape, array->rank, array->dtype,
                                       out_array)
             : status;
}

TerlanNdArrayStatus terlan_ndarray_v1_ones_like(
    TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_output(array, out_array);
  return status == TERLAN_NDARRAY_OK
             ? terlan_ndarray_v1_ones(array->shape, array->rank, array->dtype,
                                      out_array)
             : status;
}

TerlanNdArrayStatus terlan_ndarray_v1_full_like_int(
    TerlanNdArrayHandle array, int64_t value, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_output(array, out_array);
  return status == TERLAN_NDARRAY_OK
             ? terlan_ndarray_v1_full_int(array->shape, array->rank, value,
                                          array->dtype, out_array)
             : status;
}

TerlanNdArrayStatus terlan_ndarray_v1_full_like_float(
    TerlanNdArrayHandle array, double value, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_output(array, out_array);
  return status == TERLAN_NDARRAY_OK
             ? terlan_ndarray_v1_full_float(array->shape, array->rank, value,
                                            array->dtype, out_array)
             : status;
}

static TerlanNdArrayStatus spaced_powers(
    double start, double stop, int64_t count, int32_t endpoint,
    long double base, int32_t dtype, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status;
  long double divisor;
  int64_t index;
  if (out_array == NULL || count < 0 || (endpoint != 0 && endpoint != 1) ||
      !isfinite(start) || !isfinite(stop) || !isfinite(base) || base <= 0.0L) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = create_empty_array(&count, 1, dtype, out_array);
  if (status != TERLAN_NDARRAY_OK || count == 0) {
    return status;
  }
  divisor = endpoint != 0 && count > 1 ? count - 1 : count;
  for (index = 0; index < count; ++index) {
    const long double exponent =
        count == 1 ? start
                   : (long double)start +
                         ((long double)stop - start) * index / divisor;
    const TerlanNdArrayScalar value = {powl(base, exponent), 0.0L};
    TerlanNdArrayScalar stored = value;
    if (dtype_is_integer(dtype)) {
      stored.real = truncl(stored.real);
    }
    status =
        dtype_store(array_element_pointer(*out_array, index), dtype, stored);
    if (status != TERLAN_NDARRAY_OK) {
      (void)terlan_ndarray_v1_delete(*out_array);
      *out_array = NULL;
      return status;
    }
  }
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_logspace(
    double start, double stop, int64_t count, int32_t endpoint, double base,
    int32_t dtype, TerlanNdArrayHandle *out_array) {
  return spaced_powers(start, stop, count, endpoint, base, dtype, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_geomspace(
    double start, double stop, int64_t count, int32_t endpoint, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  const int negative = start < 0.0 && stop < 0.0;
  TerlanNdArrayStatus status;
  int64_t index;
  if (out_array == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (count < 0 || (endpoint != 0 && endpoint != 1)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (start == 0.0 || stop == 0.0 || !isfinite(start) || !isfinite(stop)) {
    return TERLAN_NDARRAY_DOMAIN_ERROR;
  }
  if ((start < 0.0) != (stop < 0.0)) {
    status = create_empty_array(&count, 1, dtype, out_array);
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
    for (index = 0; index < count; ++index) {
      TerlanNdArrayScalar value = {NAN, 0.0L};
      if (index == 0) {
        value.real = start;
      } else if (endpoint != 0 && count > 1 && index == count - 1) {
        value.real = stop;
      }
      status = dtype_store(array_element_pointer(*out_array, index), dtype,
                           value);
      if (status != TERLAN_NDARRAY_OK) {
        (void)terlan_ndarray_v1_delete(*out_array);
        *out_array = NULL;
        return status;
      }
    }
    return TERLAN_NDARRAY_OK;
  }
  status = spaced_powers(log10(fabs(start)), log10(fabs(stop)), count,
                         endpoint, 10.0L, dtype, out_array);
  if (status != TERLAN_NDARRAY_OK || !negative) {
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
  } else {
    for (index = 0; index < (*out_array)->numel; ++index) {
      TerlanNdArrayScalar value;
      status = dtype_load(array_const_element_pointer(*out_array, index), dtype,
                          &value);
      value.real = -value.real;
      if (status == TERLAN_NDARRAY_OK) {
        status = dtype_store(array_element_pointer(*out_array, index), dtype,
                             value);
      }
      if (status != TERLAN_NDARRAY_OK) {
        (void)terlan_ndarray_v1_delete(*out_array);
        *out_array = NULL;
        return status;
      }
    }
  }
  if ((*out_array)->numel > 0) {
    TerlanNdArrayScalar value = {start, 0.0L};
    status = dtype_store(array_element_pointer(*out_array, 0), dtype, value);
    if (status == TERLAN_NDARRAY_OK && endpoint != 0 && count > 1) {
      value.real = stop;
      status = dtype_store(array_element_pointer(*out_array, count - 1), dtype,
                           value);
    }
    if (status != TERLAN_NDARRAY_OK) {
      (void)terlan_ndarray_v1_delete(*out_array);
      *out_array = NULL;
      return status;
    }
  }
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_tri(
    int64_t rows, int64_t columns, int64_t diagonal, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  int64_t shape[2];
  const TerlanNdArrayScalar one = {1.0L, 0.0L};
  TerlanNdArrayStatus status;
  int64_t row;
  int64_t column;
  if (out_array == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (columns == -1) {
    columns = rows;
  }
  if (rows < 0 || columns < 0) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  shape[0] = rows;
  shape[1] = columns;
  status = terlan_ndarray_v1_zeros(shape, 2, dtype, out_array);
  for (row = 0; status == TERLAN_NDARRAY_OK && row < rows; ++row) {
    for (column = 0; column < columns && column - row <= diagonal; ++column) {
      status = dtype_store(array_element_pointer(*out_array,
                                                  row * columns + column),
                           dtype, one);
    }
  }
  if (status != TERLAN_NDARRAY_OK && *out_array != NULL) {
    (void)terlan_ndarray_v1_delete(*out_array);
    *out_array = NULL;
  }
  return status;
}

static TerlanNdArrayStatus triangular_copy(TerlanNdArrayHandle array,
                                           int64_t diagonal, int upper,
                                           TerlanNdArrayHandle *out_array) {
  const TerlanNdArrayScalar zero = {0.0L, 0.0L};
  TerlanNdArrayStatus status = require_output(array, out_array);
  int64_t index;
  int64_t rows;
  int64_t columns;
  if (status != TERLAN_NDARRAY_OK || array->rank < 2) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  rows = array->shape[array->rank - 2];
  columns = array->shape[array->rank - 1];
  status = create_empty_array(array->shape, array->rank, array->dtype,
                              out_array);
  for (index = 0; status == TERLAN_NDARRAY_OK && index < array->numel; ++index) {
    const int64_t row = columns == 0 ? 0 : (index / columns) % rows;
    const int64_t column = columns == 0 ? 0 : index % columns;
    const int keep = upper != 0 ? column - row >= diagonal
                                : column - row <= diagonal;
    TerlanNdArrayScalar value = zero;
    if (keep) {
      status = dtype_load(array_const_element_pointer(array, index),
                          array->dtype, &value);
    }
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_store(array_element_pointer(*out_array, index),
                           array->dtype, value);
    }
  }
  if (status != TERLAN_NDARRAY_OK && *out_array != NULL) {
    (void)terlan_ndarray_v1_delete(*out_array);
    *out_array = NULL;
  }
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_tril(
    TerlanNdArrayHandle array, int64_t diagonal,
    TerlanNdArrayHandle *out_array) {
  return triangular_copy(array, diagonal, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_triu(
    TerlanNdArrayHandle array, int64_t diagonal,
    TerlanNdArrayHandle *out_array) {
  return triangular_copy(array, diagonal, 1, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_diagflat(
    TerlanNdArrayHandle array, int64_t diagonal,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_output(array, out_array);
  int64_t offset;
  int64_t side;
  int64_t shape[2];
  int64_t index;
  if (status != TERLAN_NDARRAY_OK || diagonal == INT64_MIN) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  offset = diagonal < 0 ? -diagonal : diagonal;
  if (array->numel > INT64_MAX - offset) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  side = array->numel + offset;
  shape[0] = side;
  shape[1] = side;
  status = terlan_ndarray_v1_zeros(shape, 2, array->dtype, out_array);
  for (index = 0; status == TERLAN_NDARRAY_OK && index < array->numel; ++index) {
    const int64_t row = diagonal < 0 ? index + offset : index;
    const int64_t column = diagonal > 0 ? index + offset : index;
    TerlanNdArrayScalar value;
    status = dtype_load(array_const_element_pointer(array, index),
                        array->dtype, &value);
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_store(array_element_pointer(*out_array,
                                                  row * side + column),
                           array->dtype, value);
    }
  }
  if (status != TERLAN_NDARRAY_OK && *out_array != NULL) {
    (void)terlan_ndarray_v1_delete(*out_array);
    *out_array = NULL;
  }
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_indices(
    const int64_t *shape, int64_t rank, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  int64_t *output_shape = NULL;
  TerlanNdArrayStatus status;
  int64_t index;
  int64_t axis;
  if (out_array == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (rank < 0 || (rank > 0 && shape == NULL) ||
      dtype == TERLAN_NDARRAY_DTYPE_BOOL || dtype_size(dtype) == 0 ||
      (uint64_t)rank >= SIZE_MAX / sizeof(*output_shape)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  output_shape = checked_malloc((size_t)(rank + 1) * sizeof(*output_shape));
  if (output_shape == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  output_shape[0] = rank;
  for (axis = 0; axis < rank; ++axis) {
    output_shape[axis + 1] = shape[axis];
  }
  status = create_empty_array(output_shape, rank + 1, dtype, out_array);
  free(output_shape);
  for (index = 0; status == TERLAN_NDARRAY_OK && index < (*out_array)->numel;
       ++index) {
    const int64_t plane = (*out_array)->numel / rank;
    const int64_t source_index = index % plane;
    const int64_t selected_axis = index / plane;
    int64_t stride = 1;
    int64_t coordinate = 0;
    for (axis = rank; axis > selected_axis + 1; --axis) {
      stride *= shape[axis - 1];
    }
    if (shape[selected_axis] > 0) {
      coordinate = (source_index / stride) % shape[selected_axis];
    }
    status = dtype_store(array_element_pointer(*out_array, index), dtype,
                         (TerlanNdArrayScalar){coordinate, 0.0L});
  }
  if (status != TERLAN_NDARRAY_OK && out_array != NULL && *out_array != NULL) {
    (void)terlan_ndarray_v1_delete(*out_array);
    *out_array = NULL;
  }
  return status;
}

static TerlanNdArrayScalar scalar_power(TerlanNdArrayScalar base,
                                        int64_t exponent) {
  long double complex result = 1.0L;
  long double complex factor = base.real + base.imag * I;
  while (exponent > 0) {
    if ((exponent & 1) != 0) {
      result *= factor;
    }
    factor *= factor;
    exponent >>= 1;
  }
  return (TerlanNdArrayScalar){creall(result), cimagl(result)};
}

TerlanNdArrayStatus terlan_ndarray_v1_vander(
    TerlanNdArrayHandle array, int64_t columns, int32_t increasing,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_output(array, out_array);
  int32_t output_dtype;
  int64_t shape[2];
  int64_t row;
  int64_t column;
  if (status != TERLAN_NDARRAY_OK || array->rank != 1 || columns < -1 ||
      (increasing != 0 && increasing != 1)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  shape[0] = array->shape[0];
  shape[1] = columns == -1 ? array->shape[0] : columns;
  output_dtype = dtype_promote(array->dtype, TERLAN_NDARRAY_DTYPE_INT64);
  if (output_dtype == 0) {
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  status = create_empty_array(shape, 2, output_dtype, out_array);
  for (row = 0; status == TERLAN_NDARRAY_OK && row < shape[0]; ++row) {
    TerlanNdArrayScalar base;
    status = dtype_load(array_const_element_pointer(array, row), array->dtype,
                        &base);
    for (column = 0; status == TERLAN_NDARRAY_OK && column < shape[1];
         ++column) {
      const int64_t exponent = increasing != 0 ? column
                                               : shape[1] - column - 1;
      status = dtype_store(array_element_pointer(*out_array,
                                                  row * shape[1] + column),
                           output_dtype, scalar_power(base, exponent));
    }
  }
  if (status != TERLAN_NDARRAY_OK && *out_array != NULL) {
    (void)terlan_ndarray_v1_delete(*out_array);
    *out_array = NULL;
  }
  return status;
}
