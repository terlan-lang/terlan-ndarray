#include "terlan_ndarray_internal.h"

#include <complex.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static TerlanNdArrayStatus fill_array(const int64_t *shape, int64_t rank,
                                      TerlanNdArrayScalar value, int32_t dtype,
                                      TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status =
      create_empty_array(shape, rank, dtype, out_array);
  int64_t index;
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (index = 0; index < (*out_array)->numel; ++index) {
    status = dtype_store(array_element_pointer(*out_array, index), dtype, value);
    if (status != TERLAN_NDARRAY_OK) {
      (void)terlan_ndarray_v1_delete(*out_array);
      *out_array = NULL;
      return status;
    }
  }
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus apply_creation_order(TerlanNdArrayHandle array,
                                                int32_t order) {
  int64_t axis;
  int64_t stride = 1;
  if (order == TERLAN_NDARRAY_ORDER_C) {
    return TERLAN_NDARRAY_OK;
  }
  if (order != TERLAN_NDARRAY_ORDER_FORTRAN) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  for (axis = 0; axis < array->rank; ++axis) {
    array->strides[axis] = stride;
    if (array->shape[axis] != 0) {
      stride *= array->shape[axis];
    }
  }
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus broadcast_pair_shape(
    TerlanNdArrayHandle start, TerlanNdArrayHandle stop, int64_t **out_shape,
    int64_t *out_rank, int64_t *out_numel) {
  int64_t rank;
  int64_t dimension;
  int64_t numel = 1;
  int64_t *shape = NULL;
  TerlanNdArrayStatus status;
  if (out_shape == NULL || out_rank == NULL || out_numel == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_shape = NULL;
  *out_rank = 0;
  *out_numel = 0;
  status = require_live(start);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  status = require_live(stop);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  rank = start->rank > stop->rank ? start->rank : stop->rank;
  if (rank > 0) {
    shape = checked_calloc((size_t)rank, sizeof(*shape));
    if (shape == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (dimension = 0; dimension < rank; ++dimension) {
    const int64_t start_dimension = dimension - (rank - start->rank);
    const int64_t stop_dimension = dimension - (rank - stop->rank);
    const int64_t start_size =
        start_dimension < 0 ? 1 : start->shape[start_dimension];
    const int64_t stop_size =
        stop_dimension < 0 ? 1 : stop->shape[stop_dimension];
    const int64_t size = start_size > stop_size ? start_size : stop_size;
    if (start_size != stop_size && start_size != 1 && stop_size != 1) {
      free(shape);
      return TERLAN_NDARRAY_SHAPE_MISMATCH;
    }
    if (size != 0 && numel > INT64_MAX / size) {
      free(shape);
      return TERLAN_NDARRAY_SIZE_OVERFLOW;
    }
    shape[dimension] = size;
    numel *= size;
  }
  *out_shape = shape;
  *out_rank = rank;
  *out_numel = numel;
  return TERLAN_NDARRAY_OK;
}

static int64_t broadcast_input_index(TerlanNdArrayHandle input,
                                     const int64_t *broadcast_shape,
                                     int64_t broadcast_rank,
                                     int64_t broadcast_index) {
  int64_t input_index = 0;
  int64_t input_stride = 1;
  int64_t dimension;
  for (dimension = broadcast_rank; dimension-- > 0;) {
    const int64_t size = broadcast_shape[dimension];
    const int64_t coordinate = size == 0 ? 0 : broadcast_index % size;
    const int64_t input_dimension =
        dimension - (broadcast_rank - input->rank);
    if (size != 0) {
      broadcast_index /= size;
    }
    if (input_dimension >= 0) {
      const int64_t input_coordinate =
          input->shape[input_dimension] == 1 ? 0 : coordinate;
      input_index += input_coordinate * input_stride;
      input_stride *= input->shape[input_dimension];
    }
  }
  return input_index;
}

static TerlanNdArrayScalar linspace_sample(TerlanNdArrayScalar start,
                                           TerlanNdArrayScalar stop,
                                           int64_t sample, int64_t count,
                                           int32_t endpoint) {
  TerlanNdArrayScalar value = start;
  const int64_t divisor = endpoint != 0 && count > 1 ? count - 1 : count;
  if (endpoint != 0 && count > 1 && sample == count - 1) {
    return stop;
  }
  if (count > 1 && divisor > 0) {
    const long double real_delta = stop.real - start.real;
    const long double imag_delta = stop.imag - start.imag;
    const long double real_step = real_delta / divisor;
    const long double imag_step = imag_delta / divisor;
    value.real += real_step == 0.0L && real_delta != 0.0L
                      ? ((long double)sample / divisor) * real_delta
                      : sample * real_step;
    value.imag += imag_step == 0.0L && imag_delta != 0.0L
                      ? ((long double)sample / divisor) * imag_delta
                      : sample * imag_step;
  }
  return value;
}

TerlanNdArrayStatus terlan_ndarray_v1_zeros(
    const int64_t *shape, int64_t rank, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  const TerlanNdArrayScalar zero = {0.0L, 0.0L};
  return fill_array(shape, rank, zero, dtype, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_zeros_ordered(
    const int64_t *shape, int64_t rank, int32_t dtype, int32_t order,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status =
      terlan_ndarray_v1_zeros(shape, rank, dtype, out_array);
  if (status == TERLAN_NDARRAY_OK) {
    status = apply_creation_order(*out_array, order);
  }
  if (status != TERLAN_NDARRAY_OK && out_array != NULL && *out_array != NULL) {
    (void)terlan_ndarray_v1_delete(*out_array);
    *out_array = NULL;
  }
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_empty(
    const int64_t *shape, int64_t rank, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  /* Native storage crossing the package boundary must never be uninitialized. */
  return terlan_ndarray_v1_zeros(shape, rank, dtype, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_ones(
    const int64_t *shape, int64_t rank, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  const TerlanNdArrayScalar one = {1.0L, 0.0L};
  return fill_array(shape, rank, one, dtype, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_full_int(
    const int64_t *shape, int64_t rank, int64_t value, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  const TerlanNdArrayScalar scalar = {(long double)value, 0.0L};
  return fill_array(shape, rank, scalar, dtype, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_full_float(
    const int64_t *shape, int64_t rank, double value, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  const TerlanNdArrayScalar scalar = {(long double)value, 0.0L};
  return fill_array(shape, rank, scalar, dtype, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_arange(
    double start, double stop, double step, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  long double span;
  long double raw_count;
  int64_t count;
  int64_t index;
  TerlanNdArrayStatus status;
  if (out_array == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (step == 0.0 || !isfinite(start) || !isfinite(stop) || isnan(step)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  span = (long double)stop - (long double)start;
  if (isinf(step)) {
    raw_count = ((span > 0.0L && step > 0.0) ||
                 (span < 0.0L && step < 0.0))
                    ? 1.0L
                    : 0.0L;
  } else {
    raw_count = span / (long double)step;
  }
  if ((raw_count > 0.0L && !isfinite(raw_count)) || raw_count > INT64_MAX) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  count = raw_count <= 0.0L ? 0 : (int64_t)ceill(raw_count);
  status = create_empty_array(&count, 1, dtype, out_array);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (index = 0; index < count; ++index) {
    const TerlanNdArrayScalar value = {
        index == 0
            ? (long double)start
            : (long double)start + (long double)index * (long double)step,
        0.0L};
    status = dtype_store(array_element_pointer(*out_array, index), dtype, value);
    if (status != TERLAN_NDARRAY_OK) {
      (void)terlan_ndarray_v1_delete(*out_array);
      *out_array = NULL;
      return status;
    }
  }
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_arange_float_stop(
    double stop, TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_v1_arange(0.0, stop, 1.0,
                                  TERLAN_NDARRAY_DTYPE_FLOAT64, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_arange_float_interval(
    double start, double stop, TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_v1_arange(start, stop, 1.0,
                                  TERLAN_NDARRAY_DTYPE_FLOAT64, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_arange_float_step(
    double start, double stop, double step, TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_v1_arange(start, stop, step,
                                  TERLAN_NDARRAY_DTYPE_FLOAT64, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_arange_int(
    int64_t start, int64_t stop, int64_t step,
    TerlanNdArrayHandle *out_array) {
  uint64_t distance = 0;
  uint64_t magnitude = 0;
  uint64_t unsigned_count = 0;
  int64_t count = 0;
  int64_t index;
  int64_t value;
  TerlanNdArrayStatus status;
  if (out_array == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (step == 0) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if ((step > 0 && start < stop) || (step < 0 && start > stop)) {
    distance = step > 0 ? (uint64_t)stop - (uint64_t)start
                        : (uint64_t)start - (uint64_t)stop;
    magnitude = step > 0 ? (uint64_t)step : 0U - (uint64_t)step;
    unsigned_count = distance / magnitude + (distance % magnitude != 0U);
    if (unsigned_count > (uint64_t)INT64_MAX) {
      return TERLAN_NDARRAY_SIZE_OVERFLOW;
    }
    count = (int64_t)unsigned_count;
  }
  status = create_empty_array(&count, 1, TERLAN_NDARRAY_DTYPE_INT64,
                              out_array);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  value = start;
  for (index = 0; index < count; ++index) {
    const TerlanNdArrayScalar scalar = {(long double)value, 0.0L};
    status = dtype_store(array_element_pointer(*out_array, index),
                         TERLAN_NDARRAY_DTYPE_INT64, scalar);
    if (status != TERLAN_NDARRAY_OK) {
      (void)terlan_ndarray_v1_delete(*out_array);
      *out_array = NULL;
      return status;
    }
    if (index + 1 < count) {
      value += step;
    }
  }
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_arange_int_stop(
    int64_t stop, TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_v1_arange_int(0, stop, 1, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_arange_int_interval(
    int64_t start, int64_t stop, TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_v1_arange_int(start, stop, 1, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_linspace(
    double start, double stop, int64_t count, int32_t endpoint, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status;
  int64_t index;
  long double divisor;
  if (out_array == NULL || count < 0 ||
      (endpoint != 0 && endpoint != 1) || !isfinite(start) || !isfinite(stop)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = create_empty_array(&count, 1, dtype, out_array);
  if (status != TERLAN_NDARRAY_OK || count == 0) {
    return status;
  }
  divisor = endpoint != 0 && count > 1 ? count - 1 : count;
  for (index = 0; index < count; ++index) {
    TerlanNdArrayScalar value;
    value.real = count == 1
                     ? start
                     : (long double)start +
                           ((long double)stop - start) * index / divisor;
    value.imag = 0.0L;
    if (dtype_is_integer(dtype)) {
      value.real = floorl(value.real);
    }
    status = dtype_store(array_element_pointer(*out_array, index), dtype, value);
    if (status != TERLAN_NDARRAY_OK) {
      (void)terlan_ndarray_v1_delete(*out_array);
      *out_array = NULL;
      return status;
    }
  }
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_linspace_arrays(
    TerlanNdArrayHandle start, TerlanNdArrayHandle stop, int64_t count,
    int32_t endpoint, int64_t axis, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  int64_t *broadcast_shape = NULL;
  int64_t *output_shape = NULL;
  int64_t broadcast_rank = 0;
  int64_t broadcast_numel = 0;
  int64_t output_rank;
  int64_t normalized_axis;
  int64_t suffix_size = 1;
  int64_t dimension;
  int64_t broadcast_index;
  TerlanNdArrayStatus status;
  if (out_array == NULL || count < 0 || (endpoint != 0 && endpoint != 1)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = broadcast_pair_shape(start, stop, &broadcast_shape, &broadcast_rank,
                                &broadcast_numel);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  output_rank = broadcast_rank + 1;
  normalized_axis = axis < 0 ? axis + output_rank : axis;
  if (normalized_axis < 0 || normalized_axis >= output_rank) {
    free(broadcast_shape);
    return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
  }
  output_shape = checked_calloc((size_t)output_rank, sizeof(*output_shape));
  if (output_shape == NULL) {
    free(broadcast_shape);
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  for (dimension = 0; dimension < output_rank; ++dimension) {
    output_shape[dimension] =
        dimension == normalized_axis
            ? count
            : broadcast_shape[dimension < normalized_axis ? dimension
                                                           : dimension - 1];
  }
  for (dimension = normalized_axis; dimension < broadcast_rank; ++dimension) {
    if (broadcast_shape[dimension] != 0 &&
        suffix_size > INT64_MAX / broadcast_shape[dimension]) {
      free(output_shape);
      free(broadcast_shape);
      return TERLAN_NDARRAY_SIZE_OVERFLOW;
    }
    suffix_size *= broadcast_shape[dimension];
  }
  status = create_empty_array(output_shape, output_rank, dtype, out_array);
  free(output_shape);
  if (status != TERLAN_NDARRAY_OK) {
    free(broadcast_shape);
    return status;
  }
  for (broadcast_index = 0; broadcast_index < broadcast_numel;
       ++broadcast_index) {
    const int64_t start_index = broadcast_input_index(
        start, broadcast_shape, broadcast_rank, broadcast_index);
    const int64_t stop_index = broadcast_input_index(
        stop, broadcast_shape, broadcast_rank, broadcast_index);
    TerlanNdArrayScalar start_value;
    TerlanNdArrayScalar stop_value;
    int64_t sample;
    status = dtype_load(array_const_element_pointer(start, start_index),
                        start->dtype, &start_value);
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_load(array_const_element_pointer(stop, stop_index),
                          stop->dtype, &stop_value);
    }
    if (status != TERLAN_NDARRAY_OK) {
      break;
    }
    for (sample = 0; sample < count; ++sample) {
      const int64_t prefix =
          suffix_size == 0 ? 0 : broadcast_index / suffix_size;
      const int64_t suffix =
          suffix_size == 0 ? 0 : broadcast_index % suffix_size;
      const int64_t output_index =
          (prefix * count + sample) * suffix_size + suffix;
      TerlanNdArrayScalar value =
          linspace_sample(start_value, stop_value, sample, count, endpoint);
      if (dtype_is_integer(dtype)) {
        value.real = floorl(value.real);
      }
      status = dtype_store(array_element_pointer(*out_array, output_index),
                           dtype, value);
      if (status != TERLAN_NDARRAY_OK) {
        break;
      }
    }
    if (status != TERLAN_NDARRAY_OK) {
      break;
    }
  }
  free(broadcast_shape);
  if (status != TERLAN_NDARRAY_OK) {
    (void)terlan_ndarray_v1_delete(*out_array);
    *out_array = NULL;
  }
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_linspace_steps(
    TerlanNdArrayHandle start, TerlanNdArrayHandle stop, int64_t count,
    int32_t endpoint, TerlanNdArrayHandle *out_array) {
  int64_t *shape = NULL;
  int64_t rank = 0;
  int64_t numel = 0;
  int64_t index;
  int32_t dtype;
  TerlanNdArrayStatus status;
  if (out_array == NULL || count < 0 || (endpoint != 0 && endpoint != 1)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = broadcast_pair_shape(start, stop, &shape, &rank, &numel);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  dtype = dtype_is_complex(start->dtype) || dtype_is_complex(stop->dtype)
              ? TERLAN_NDARRAY_DTYPE_COMPLEX128
              : TERLAN_NDARRAY_DTYPE_FLOAT64;
  status = create_empty_array(shape, rank, dtype, out_array);
  if (status != TERLAN_NDARRAY_OK) {
    free(shape);
    return status;
  }
  for (index = 0; index < numel; ++index) {
    const int64_t start_index =
        broadcast_input_index(start, shape, rank, index);
    const int64_t stop_index = broadcast_input_index(stop, shape, rank, index);
    const int64_t divisor = endpoint != 0 ? count - 1 : count;
    TerlanNdArrayScalar start_value;
    TerlanNdArrayScalar stop_value;
    TerlanNdArrayScalar step;
    status = dtype_load(array_const_element_pointer(start, start_index),
                        start->dtype, &start_value);
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_load(array_const_element_pointer(stop, stop_index),
                          stop->dtype, &stop_value);
    }
    if (status != TERLAN_NDARRAY_OK) {
      break;
    }
    if (divisor <= 0) {
      step.real = NAN;
      step.imag = 0.0L;
    } else {
      step.real = (stop_value.real - start_value.real) / divisor;
      step.imag = (stop_value.imag - start_value.imag) / divisor;
    }
    status = dtype_store(array_element_pointer(*out_array, index), dtype, step);
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

static TerlanNdArrayScalar complex_power_sample(
    TerlanNdArrayScalar exponent, long double base) {
  const long double complex value = cexpl(
      logl(base) * CMPLXL(exponent.real, exponent.imag));
  const TerlanNdArrayScalar result = {creall(value), cimagl(value)};
  return result;
}

static TerlanNdArrayScalar geometric_sample(
    TerlanNdArrayScalar start, TerlanNdArrayScalar stop, int64_t sample,
    int64_t count, int32_t endpoint, int complex_domain) {
  long double complex sign = 1.0L;
  long double complex start_value = CMPLXL(start.real, start.imag);
  long double complex stop_value = CMPLXL(stop.real, stop.imag);
  TerlanNdArrayScalar start_log;
  TerlanNdArrayScalar stop_log;
  TerlanNdArrayScalar exponent;
  long double complex value;

  if (sample == 0) {
    return start;
  }
  if (endpoint != 0 && count > 1 && sample == count - 1) {
    return stop;
  }
  if (!complex_domain && start.imag == 0.0L && stop.imag == 0.0L &&
      ((start.real < 0.0L) != (stop.real < 0.0L))) {
    const TerlanNdArrayScalar invalid = {NAN, 0.0L};
    return invalid;
  }
  if (start.real == 0.0L && stop.real == 0.0L) {
    start_value = CMPLXL(start.imag, 0.0L);
    stop_value = CMPLXL(stop.imag, 0.0L);
    sign = I;
  }
  if (cimagl(start_value) == 0.0L && cimagl(stop_value) == 0.0L &&
      creall(start_value) < 0.0L && creall(stop_value) < 0.0L) {
    start_value = -start_value;
    stop_value = -stop_value;
    sign = -sign;
  }
  value = clogl(start_value);
  start_log.real = creall(value);
  start_log.imag = cimagl(value);
  value = clogl(stop_value);
  stop_log.real = creall(value);
  stop_log.imag = cimagl(value);
  exponent = linspace_sample(start_log, stop_log, sample, count, endpoint);
  value = sign * cexpl(CMPLXL(exponent.real, exponent.imag));
  {
    const TerlanNdArrayScalar result = {creall(value), cimagl(value)};
    return result;
  }
}

static TerlanNdArrayStatus logarithmic_array_samples(
    TerlanNdArrayHandle start, TerlanNdArrayHandle stop, int64_t count,
    int32_t endpoint, int64_t axis, long double base, int32_t dtype,
    int geometric, TerlanNdArrayHandle *out_array) {
  int64_t *broadcast_shape = NULL;
  int64_t *output_shape = NULL;
  int64_t broadcast_rank = 0;
  int64_t broadcast_numel = 0;
  int64_t output_rank;
  int64_t normalized_axis;
  int64_t suffix_size = 1;
  int64_t dimension;
  int64_t broadcast_index;
  int complex_domain;
  TerlanNdArrayStatus status;
  if (out_array == NULL || count < 0 || (endpoint != 0 && endpoint != 1) ||
      (!geometric && (!isfinite(base) || base <= 0.0L))) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = broadcast_pair_shape(start, stop, &broadcast_shape, &broadcast_rank,
                                &broadcast_numel);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  output_rank = broadcast_rank + 1;
  normalized_axis = axis < 0 ? axis + output_rank : axis;
  if (normalized_axis < 0 || normalized_axis >= output_rank) {
    free(broadcast_shape);
    return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
  }
  output_shape = checked_calloc((size_t)output_rank, sizeof(*output_shape));
  if (output_shape == NULL) {
    free(broadcast_shape);
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  for (dimension = 0; dimension < output_rank; ++dimension) {
    output_shape[dimension] =
        dimension == normalized_axis
            ? count
            : broadcast_shape[dimension < normalized_axis ? dimension
                                                           : dimension - 1];
  }
  for (dimension = normalized_axis; dimension < broadcast_rank; ++dimension) {
    if (broadcast_shape[dimension] != 0 &&
        suffix_size > INT64_MAX / broadcast_shape[dimension]) {
      free(output_shape);
      free(broadcast_shape);
      return TERLAN_NDARRAY_SIZE_OVERFLOW;
    }
    suffix_size *= broadcast_shape[dimension];
  }
  status = create_empty_array(output_shape, output_rank, dtype, out_array);
  free(output_shape);
  if (status != TERLAN_NDARRAY_OK) {
    free(broadcast_shape);
    return status;
  }
  complex_domain = dtype_is_complex(start->dtype) ||
                   dtype_is_complex(stop->dtype) || dtype_is_complex(dtype);
  for (broadcast_index = 0; broadcast_index < broadcast_numel;
       ++broadcast_index) {
    const int64_t start_index = broadcast_input_index(
        start, broadcast_shape, broadcast_rank, broadcast_index);
    const int64_t stop_index = broadcast_input_index(
        stop, broadcast_shape, broadcast_rank, broadcast_index);
    TerlanNdArrayScalar start_value;
    TerlanNdArrayScalar stop_value;
    int64_t sample;
    status = dtype_load(array_const_element_pointer(start, start_index),
                        start->dtype, &start_value);
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_load(array_const_element_pointer(stop, stop_index),
                          stop->dtype, &stop_value);
    }
    if (status != TERLAN_NDARRAY_OK) {
      break;
    }
    if (geometric &&
        ((start_value.real == 0.0L && start_value.imag == 0.0L) ||
         (stop_value.real == 0.0L && stop_value.imag == 0.0L))) {
      status = TERLAN_NDARRAY_DOMAIN_ERROR;
      break;
    }
    for (sample = 0; sample < count; ++sample) {
      const int64_t prefix =
          suffix_size == 0 ? 0 : broadcast_index / suffix_size;
      const int64_t suffix =
          suffix_size == 0 ? 0 : broadcast_index % suffix_size;
      const int64_t output_index =
          (prefix * count + sample) * suffix_size + suffix;
      TerlanNdArrayScalar value =
          geometric
              ? geometric_sample(start_value, stop_value, sample, count,
                                 endpoint, complex_domain)
              : complex_power_sample(
                    linspace_sample(start_value, stop_value, sample, count,
                                    endpoint),
                    base);
      if (dtype_is_integer(dtype)) {
        value.real = truncl(value.real);
      }
      status = dtype_store(array_element_pointer(*out_array, output_index),
                           dtype, value);
      if (status != TERLAN_NDARRAY_OK) {
        break;
      }
    }
    if (status != TERLAN_NDARRAY_OK) {
      break;
    }
  }
  free(broadcast_shape);
  if (status != TERLAN_NDARRAY_OK) {
    (void)terlan_ndarray_v1_delete(*out_array);
    *out_array = NULL;
  }
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_logspace_arrays(
    TerlanNdArrayHandle start, TerlanNdArrayHandle stop, int64_t count,
    int32_t endpoint, int64_t axis, double base, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  return logarithmic_array_samples(start, stop, count, endpoint, axis, base,
                                   dtype, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_geomspace_arrays(
    TerlanNdArrayHandle start, TerlanNdArrayHandle stop, int64_t count,
    int32_t endpoint, int64_t axis, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  return logarithmic_array_samples(start, stop, count, endpoint, axis, 10.0L,
                                   dtype, 1, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_eye(
    int64_t rows, int64_t columns, int64_t diagonal, int32_t dtype,
    int32_t order, TerlanNdArrayHandle *out_array) {
  int64_t shape[2];
  TerlanNdArrayStatus status;
  int64_t row;
  if (out_array == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (columns == -1) {
    columns = rows;
  }
  if (rows < 0 || columns < 0 || (order != 0 && order != 1)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  shape[0] = rows;
  shape[1] = columns;
  status = terlan_ndarray_v1_zeros(shape, 2, dtype, out_array);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (order == 1) {
    (*out_array)->strides[0] = 1;
    (*out_array)->strides[1] = rows;
  }
  for (row = 0; row < rows; ++row) {
    const int64_t column = row + diagonal;
    if (column >= 0 && column < columns) {
      const TerlanNdArrayScalar one = {1.0L, 0.0L};
      status = dtype_store(array_element_pointer(*out_array,
                                                  row * columns + column),
                           dtype, one);
      if (status != TERLAN_NDARRAY_OK) {
        (void)terlan_ndarray_v1_delete(*out_array);
        *out_array = NULL;
        return status;
      }
    }
  }
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_identity(
    int64_t rows, int64_t columns, int64_t diagonal, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_v1_eye(rows, columns, diagonal, dtype, 0, out_array);
}

static TerlanNdArrayStatus normalized_axis(int64_t value, int64_t rank,
                                            int64_t *out_axis) {
  if (value < 0) {
    value += rank;
  }
  if (value < 0 || value >= rank || out_axis == NULL) {
    return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
  }
  *out_axis = value;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_diagonal(
    TerlanNdArrayHandle array, int64_t offset, int64_t axis_a, int64_t axis_b,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t a;
  int64_t b;
  int64_t diagonal_length;
  int64_t relative_offset;
  int64_t *shape = NULL;
  int64_t *strides = NULL;
  int64_t source_axis;
  int64_t output_axis = 0;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL || array->rank < 2 ||
      offset == INT64_MIN) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = normalized_axis(axis_a, array->rank, &a);
  if (status == TERLAN_NDARRAY_OK) {
    status = normalized_axis(axis_b, array->rank, &b);
  }
  if (status != TERLAN_NDARRAY_OK || a == b) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_DUPLICATE_AXIS;
  }
  if (offset >= 0) {
    diagonal_length = offset >= array->shape[b]
                          ? 0
                          : array->shape[b] - offset;
    if (diagonal_length > array->shape[a]) {
      diagonal_length = array->shape[a];
    }
    relative_offset = offset * array->strides[b];
  } else {
    const int64_t row_offset = -offset;
    diagonal_length = row_offset >= array->shape[a]
                          ? 0
                          : array->shape[a] - row_offset;
    if (diagonal_length > array->shape[b]) {
      diagonal_length = array->shape[b];
    }
    relative_offset = row_offset * array->strides[a];
  }
  shape = checked_malloc((size_t)(array->rank - 1) * sizeof(*shape));
  strides = checked_malloc((size_t)(array->rank - 1) * sizeof(*strides));
  if (shape == NULL || strides == NULL) {
    free(strides);
    free(shape);
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  for (source_axis = 0; source_axis < array->rank; ++source_axis) {
    if (source_axis != a && source_axis != b) {
      shape[output_axis] = array->shape[source_axis];
      strides[output_axis++] = array->strides[source_axis];
    }
  }
  shape[output_axis] = diagonal_length;
  strides[output_axis] = array->strides[a] + array->strides[b];
  status = create_view(array, shape, strides, array->rank - 1, relative_offset,
                       array->writable, out_array);
  free(strides);
  free(shape);
  return status;
}
