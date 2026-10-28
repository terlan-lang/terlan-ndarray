#include "terlan_ndarray_internal.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum PaddingMode {
  PADDING_CONSTANT = 0,
  PADDING_EDGE = 1,
  PADDING_REFLECT = 2,
  PADDING_SYMMETRIC = 3,
  PADDING_WRAP = 4,
  PADDING_LINEAR_RAMP = 5,
  PADDING_REFLECT_ODD = 6,
  PADDING_SYMMETRIC_ODD = 7,
  PADDING_MINIMUM = 8,
  PADDING_MAXIMUM = 9,
  PADDING_MEAN = 10,
  PADDING_MEDIAN = 11
};

static int64_t logical_index(const int64_t *coordinates,
                             const int64_t *shape, int64_t rank);

static int64_t positive_modulo(int64_t value, int64_t modulus) {
  const int64_t remainder = value % modulus;
  return remainder < 0 ? remainder + modulus : remainder;
}

static int64_t mapped_coordinate(int64_t coordinate, int64_t length,
                                 enum PaddingMode mode) {
  int64_t period;
  int64_t position;
  if (mode == PADDING_EDGE || length == 1) {
    return coordinate < 0 ? 0 : length - 1;
  }
  if (mode == PADDING_WRAP) {
    return positive_modulo(coordinate, length);
  }
  if (mode == PADDING_REFLECT || mode == PADDING_REFLECT_ODD) {
    period = 2 * (length - 1);
    position = positive_modulo(coordinate, period);
    return position < length ? position : period - position;
  }
  period = 2 * length;
  position = positive_modulo(coordinate, period);
  return position < length ? position : period - position - 1;
}

static TerlanNdArrayStatus odd_reflected_value(
    TerlanNdArrayHandle array, int64_t axis, int64_t coordinate,
    enum PaddingMode mode, int64_t *coordinates,
    TerlanNdArrayScalar *out_value) {
  const int include_edge = mode == PADDING_SYMMETRIC_ODD;
  const int64_t length = array->shape[axis];
  const int64_t half_period = include_edge ? length : length - 1;
  const int64_t period = 2 * half_period;
  int64_t quotient = coordinate / period;
  int64_t position = coordinate % period;
  int64_t mapped;
  TerlanNdArrayScalar edge;
  TerlanNdArrayStatus status;
  if (length == 1) {
    coordinates[axis] = 0;
    return dtype_load(array_const_element_pointer(
                          array, logical_index(coordinates, array->shape,
                                               array->rank)),
                      array->dtype, out_value);
  }
  if (position < 0) {
    position += period;
    --quotient;
  }
  if (include_edge) {
    mapped = position < length ? position : period - position - 1;
  } else {
    mapped = position <= half_period ? position : period - position;
  }
  coordinates[axis] = mapped;
  status = dtype_load(array_const_element_pointer(
                          array, logical_index(coordinates, array->shape,
                                               array->rank)),
                      array->dtype, out_value);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  coordinates[axis] = length - 1;
  status = dtype_load(array_const_element_pointer(
                          array, logical_index(coordinates, array->shape,
                                               array->rank)),
                      array->dtype, &edge);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (position >= length) {
    out_value->real = 2.0L * edge.real - out_value->real;
    out_value->imag = 2.0L * edge.imag - out_value->imag;
  }
  coordinates[axis] = 0;
  status = dtype_load(array_const_element_pointer(
                          array, logical_index(coordinates, array->shape,
                                               array->rank)),
                      array->dtype, &edge);
  if (status == TERLAN_NDARRAY_OK) {
    TerlanNdArrayScalar high;
    coordinates[axis] = length - 1;
    status = dtype_load(array_const_element_pointer(
                            array, logical_index(coordinates, array->shape,
                                                 array->rank)),
                        array->dtype, &high);
    if (status == TERLAN_NDARRAY_OK) {
      const long double shifts = 2.0L * (long double)quotient;
      out_value->real += shifts * (high.real - edge.real);
      out_value->imag += shifts * (high.imag - edge.imag);
    }
  }
  return status;
}

static int64_t logical_index(const int64_t *coordinates,
                             const int64_t *shape, int64_t rank) {
  int64_t index = 0;
  int64_t axis;
  for (axis = 0; axis < rank; ++axis) {
    index = index * shape[axis] + coordinates[axis];
  }
  return index;
}

static void decode_index(int64_t index, const int64_t *shape, int64_t rank,
                         int64_t *coordinates) {
  int64_t axis;
  for (axis = rank; axis > 0; --axis) {
    const int64_t dimension = shape[axis - 1];
    coordinates[axis - 1] = dimension == 0 ? 0 : index % dimension;
    if (dimension != 0) {
      index /= dimension;
    }
  }
}

static void sort_scalars(TerlanNdArrayScalar *values, int64_t count) {
  int64_t index;
  for (index = 1; index < count; ++index) {
    TerlanNdArrayScalar value = values[index];
    int64_t position = index;
    while (position > 0 &&
           scalar_order_compare(value, values[position - 1]) < 0) {
      values[position] = values[position - 1];
      --position;
    }
    values[position] = value;
  }
}

static TerlanNdArrayStatus statistical_value(
    TerlanNdArrayHandle array, int64_t axis, int64_t source_coordinate,
    int64_t length, enum PaddingMode mode, int64_t *coordinates,
    TerlanNdArrayScalar *scratch, TerlanNdArrayScalar *out_value) {
  int64_t index;
  TerlanNdArrayStatus status;
  TerlanNdArrayScalar value = {0.0L, 0.0L};
  const int64_t start = source_coordinate < 0 ? 0 : array->shape[axis] - length;
  if (length == 0) {
    out_value->real = NAN;
    out_value->imag = 0.0L;
    return TERLAN_NDARRAY_OK;
  }
  for (index = 0; index < length; ++index) {
    coordinates[axis] = start + index;
    status = dtype_load(array_const_element_pointer(
                            array, logical_index(coordinates, array->shape,
                                                 array->rank)),
                        array->dtype, &scratch[index]);
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
    value.real += scratch[index].real;
    value.imag += scratch[index].imag;
  }
  if (mode == PADDING_MEAN) {
    value.real /= (long double)length;
    value.imag /= (long double)length;
  } else {
    sort_scalars(scratch, length);
    if (mode == PADDING_MINIMUM) {
      value = scratch[0];
    } else if (mode == PADDING_MAXIMUM) {
      value = scratch[length - 1];
    } else if ((length & 1) != 0) {
      value = scratch[length / 2];
    } else {
      value.real = (scratch[length / 2 - 1].real +
                    scratch[length / 2].real) /
                   2.0L;
      value.imag = (scratch[length / 2 - 1].imag +
                    scratch[length / 2].imag) /
                   2.0L;
    }
  }
  *out_value = value;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus pad_axis(
    TerlanNdArrayHandle array, int64_t axis, int64_t before, int64_t after,
    enum PaddingMode mode, TerlanNdArrayScalar before_value,
    TerlanNdArrayScalar after_value,
    int64_t before_length, int64_t after_length,
    TerlanNdArrayHandle *out_array) {
  int64_t *shape = NULL;
  int64_t *coordinates = NULL;
  TerlanNdArrayScalar *scratch = NULL;
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayStatus status;
  int64_t output_index;
  const int64_t source_length = array->shape[axis];
  if (source_length > INT64_MAX - before ||
      source_length + before > INT64_MAX - after) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  if (source_length == 0 && (before > 0 || after > 0) &&
      mode != PADDING_CONSTANT) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if ((mode == PADDING_REFLECT || mode == PADDING_REFLECT_ODD) &&
      source_length > 1 && source_length - 1 > INT64_MAX / 2) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  if ((mode == PADDING_SYMMETRIC || mode == PADDING_SYMMETRIC_ODD) &&
      source_length > INT64_MAX / 2) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  if (before_length > source_length) {
    before_length = source_length;
  }
  if (after_length > source_length) {
    after_length = source_length;
  }
  shape = checked_malloc((size_t)array->rank * sizeof(*shape));
  coordinates = checked_malloc((size_t)array->rank * sizeof(*coordinates));
  if (shape == NULL || coordinates == NULL) {
    free(coordinates);
    free(shape);
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  memcpy(shape, array->shape, (size_t)array->rank * sizeof(*shape));
  shape[axis] += before + after;
  if (mode >= PADDING_MINIMUM && (before_length > 0 || after_length > 0)) {
    const int64_t capacity = before_length > after_length
                                 ? before_length
                                 : after_length;
    scratch = checked_malloc((size_t)capacity * sizeof(*scratch));
    if (scratch == NULL) {
      free(coordinates);
      free(shape);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  status = create_empty_array(shape, array->rank, array->dtype, &result);
  if (status != TERLAN_NDARRAY_OK) {
    free(scratch);
    free(coordinates);
    free(shape);
    return status;
  }
  for (output_index = 0; output_index < result->numel; ++output_index) {
    TerlanNdArrayScalar value;
    int64_t source_coordinate;
    decode_index(output_index, result->shape, result->rank, coordinates);
    source_coordinate = coordinates[axis] - before;
    if (source_coordinate >= 0 && source_coordinate < source_length) {
      coordinates[axis] = source_coordinate;
      status = dtype_load(array_const_element_pointer(
                              array, logical_index(coordinates, array->shape,
                                                   array->rank)),
                          array->dtype, &value);
    } else if (mode == PADDING_CONSTANT) {
      value = source_coordinate < 0 ? before_value : after_value;
      status = TERLAN_NDARRAY_OK;
    } else if (mode == PADDING_LINEAR_RAMP) {
      const int64_t edge_coordinate = source_coordinate < 0 ? 0
                                                            : source_length - 1;
      TerlanNdArrayScalar edge;
      const TerlanNdArrayScalar end = source_coordinate < 0 ? before_value
                                                             : after_value;
      long double fraction;
      coordinates[axis] = edge_coordinate;
      status = dtype_load(array_const_element_pointer(
                              array, logical_index(coordinates, array->shape,
                                                   array->rank)),
                          array->dtype, &edge);
      if (status == TERLAN_NDARRAY_OK) {
        if (source_coordinate < 0) {
          fraction = (long double)(source_coordinate + before) /
                     (long double)before;
          value.real = end.real + (edge.real - end.real) * fraction;
          value.imag = end.imag + (edge.imag - end.imag) * fraction;
        } else {
          fraction = (long double)(source_coordinate - source_length + 1) /
                     (long double)after;
          value.real = edge.real + (end.real - edge.real) * fraction;
          value.imag = edge.imag + (end.imag - edge.imag) * fraction;
        }
        if (dtype_is_integer(array->dtype)) {
          value.real = floorl(value.real);
        }
      }
    } else if (mode >= PADDING_MINIMUM) {
      const int64_t length = source_coordinate < 0 ? before_length
                                                   : after_length;
      status = statistical_value(array, axis, source_coordinate, length, mode,
                                 coordinates, scratch, &value);
    } else if (mode == PADDING_REFLECT_ODD ||
               mode == PADDING_SYMMETRIC_ODD) {
      status = odd_reflected_value(array, axis, source_coordinate, mode,
                                   coordinates, &value);
    } else {
      coordinates[axis] =
          mapped_coordinate(source_coordinate, source_length, mode);
      status = dtype_load(array_const_element_pointer(
                              array, logical_index(coordinates, array->shape,
                                                   array->rank)),
                          array->dtype, &value);
    }
    if (status == TERLAN_NDARRAY_OK &&
        (mode == PADDING_MEAN || mode == PADDING_MEDIAN) &&
        dtype_is_integer(result->dtype)) {
      value.real = nearbyintl(value.real);
    }
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_store(array_element_pointer(result, output_index),
                           result->dtype, value);
    }
    if (status != TERLAN_NDARRAY_OK) {
      (void)terlan_ndarray_v1_delete(result);
      free(scratch);
      free(coordinates);
      free(shape);
      return status;
    }
  }
  free(scratch);
  free(coordinates);
  free(shape);
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus validate_padding(
    TerlanNdArrayHandle array, const int64_t *before, int64_t before_count,
    const int64_t *after, int64_t after_count,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  int64_t axis;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL ||
      !((array->rank == 0 && before_count == 0) ||
        (array->rank > 0 && (before_count == 1 || before_count == array->rank))) ||
      !((array->rank == 0 && after_count == 0) ||
        (array->rank > 0 && (after_count == 1 || after_count == array->rank))) ||
      (array->rank > 0 && (before == NULL || after == NULL))) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  for (axis = 0; axis < array->rank; ++axis) {
    const int64_t before_value = before[before_count == 1 ? 0 : axis];
    const int64_t after_value = after[after_count == 1 ? 0 : axis];
    if (before_value < 0 || after_value < 0) {
      return TERLAN_NDARRAY_INVALID_ARGUMENT;
    }
  }
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus pad_all(
    TerlanNdArrayHandle array, const int64_t *before, int64_t before_count,
    const int64_t *after, int64_t after_count,
    const double *before_values, const double *after_values,
    int64_t value_count, TerlanNdArrayHandle before_value_array,
    TerlanNdArrayHandle after_value_array, const int64_t *before_lengths,
    int64_t before_length_count, const int64_t *after_lengths,
    int64_t after_length_count, enum PaddingMode mode,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayHandle current = NULL;
  int64_t axis;
  TerlanNdArrayStatus status = copy_array(array, &current);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (axis = 0; axis < array->rank; ++axis) {
    TerlanNdArrayHandle next = NULL;
    const int64_t value_index = value_count == 1 ? 0 : axis;
    const int64_t before_index = before_count == 1 ? 0 : axis;
    const int64_t after_index = after_count == 1 ? 0 : axis;
    TerlanNdArrayScalar before_value = {0.0L, 0.0L};
    TerlanNdArrayScalar after_value = {0.0L, 0.0L};
    const int64_t before_length = before_lengths == NULL
                                      ? current->shape[axis]
                                      : before_lengths[before_length_count == 1
                                                           ? 0
                                                           : axis];
    const int64_t after_length = after_lengths == NULL
                                     ? current->shape[axis]
                                     : after_lengths[after_length_count == 1
                                                         ? 0
                                                         : axis];
    if (before_value_array != NULL) {
      status = dtype_load(array_const_element_pointer(before_value_array,
                                                      value_index),
                          before_value_array->dtype, &before_value);
      if (status == TERLAN_NDARRAY_OK) {
        status = dtype_load(array_const_element_pointer(after_value_array,
                                                        value_index),
                            after_value_array->dtype, &after_value);
      }
    } else {
      before_value.real = before_values == NULL ? 0.0L
                                                 : before_values[value_index];
      after_value.real = after_values == NULL ? 0.0L
                                               : after_values[value_index];
      status = TERLAN_NDARRAY_OK;
    }
    if (status != TERLAN_NDARRAY_OK) {
      (void)terlan_ndarray_v1_delete(current);
      return status;
    }
    status = pad_axis(current, axis, before[before_index], after[after_index], mode,
                      before_value, after_value, before_length, after_length,
                      &next);
    (void)terlan_ndarray_v1_delete(current);
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
    current = next;
  }
  *out_array = current;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_pad_constant(
    TerlanNdArrayHandle array, const int64_t *before, int64_t before_count,
    const int64_t *after, int64_t after_count, const double *before_values,
    int64_t before_value_count, const double *after_values,
    int64_t after_value_count, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = validate_padding(
      array, before, before_count, after, after_count, out_array);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (before_value_count != after_value_count ||
      (before_value_count != 1 && before_value_count != array->rank) ||
      before_values == NULL || after_values == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  return pad_all(array, before, before_count, after, after_count, before_values,
                 after_values, before_value_count, NULL, NULL, NULL, 0, NULL, 0,
                 PADDING_CONSTANT, out_array);
}

static TerlanNdArrayStatus pad_array_values(
    TerlanNdArrayHandle array, const int64_t *before, int64_t before_count,
    const int64_t *after, int64_t after_count,
    TerlanNdArrayHandle before_values, TerlanNdArrayHandle after_values,
    enum PaddingMode mode, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = validate_padding(
      array, before, before_count, after, after_count, out_array);
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(before_values);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(after_values);
  }
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (before_values->numel != after_values->numel ||
      (before_values->numel != 1 && before_values->numel != array->rank)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  return pad_all(array, before, before_count, after, after_count, NULL, NULL,
                 before_values->numel, before_values, after_values, NULL, 0,
                 NULL, 0, mode, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_pad_constant_values(
    TerlanNdArrayHandle array, const int64_t *before, int64_t before_count,
    const int64_t *after, int64_t after_count,
    TerlanNdArrayHandle before_values, TerlanNdArrayHandle after_values,
    TerlanNdArrayHandle *out_array) {
  return pad_array_values(array, before, before_count, after, after_count,
                          before_values, after_values, PADDING_CONSTANT,
                          out_array);
}

static TerlanNdArrayStatus pad_mapped(
    TerlanNdArrayHandle array, const int64_t *before, int64_t before_count,
    const int64_t *after, int64_t after_count, enum PaddingMode mode,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = validate_padding(
      array, before, before_count, after, after_count, out_array);
  return status == TERLAN_NDARRAY_OK
             ? pad_all(array, before, before_count, after, after_count, NULL,
                       NULL, 0, NULL, NULL, NULL, 0, NULL, 0, mode, out_array)
             : status;
}

#define DEFINE_MAPPED_PADDING(name, mode)                                      \
  TerlanNdArrayStatus terlan_ndarray_v1_pad_##name(                            \
      TerlanNdArrayHandle array, const int64_t *before, int64_t before_count,  \
      const int64_t *after, int64_t after_count,                               \
      TerlanNdArrayHandle *out_array) {                                        \
    return pad_mapped(array, before, before_count, after, after_count, mode,    \
                      out_array);                                               \
  }

DEFINE_MAPPED_PADDING(edge, PADDING_EDGE)
DEFINE_MAPPED_PADDING(reflect, PADDING_REFLECT)
DEFINE_MAPPED_PADDING(symmetric, PADDING_SYMMETRIC)
DEFINE_MAPPED_PADDING(wrap, PADDING_WRAP)
DEFINE_MAPPED_PADDING(reflect_odd, PADDING_REFLECT_ODD)
DEFINE_MAPPED_PADDING(symmetric_odd, PADDING_SYMMETRIC_ODD)

TerlanNdArrayStatus terlan_ndarray_v1_pad_linear_ramp(
    TerlanNdArrayHandle array, const int64_t *before, int64_t before_count,
    const int64_t *after, int64_t after_count, const double *before_values,
    int64_t before_value_count, const double *after_values,
    int64_t after_value_count, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = validate_padding(
      array, before, before_count, after, after_count, out_array);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (before_value_count != after_value_count ||
      (before_value_count != 1 && before_value_count != array->rank) ||
      before_values == NULL || after_values == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  return pad_all(array, before, before_count, after, after_count, before_values,
                 after_values, before_value_count, NULL, NULL, NULL, 0, NULL, 0,
                 PADDING_LINEAR_RAMP, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_pad_linear_ramp_values(
    TerlanNdArrayHandle array, const int64_t *before, int64_t before_count,
    const int64_t *after, int64_t after_count,
    TerlanNdArrayHandle before_values, TerlanNdArrayHandle after_values,
    TerlanNdArrayHandle *out_array) {
  return pad_array_values(array, before, before_count, after, after_count,
                          before_values, after_values, PADDING_LINEAR_RAMP,
                          out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_pad_empty(
    TerlanNdArrayHandle array, const int64_t *before, int64_t before_count,
    const int64_t *after, int64_t after_count,
    TerlanNdArrayHandle *out_array) {
  static const double zero = 0.0;
  TerlanNdArrayStatus status = validate_padding(
      array, before, before_count, after, after_count, out_array);
  return status == TERLAN_NDARRAY_OK
             ? pad_all(array, before, before_count, after, after_count, &zero,
                       &zero, 1, NULL, NULL, NULL, 0, NULL, 0, PADDING_CONSTANT,
                       out_array)
             : status;
}

TerlanNdArrayStatus terlan_ndarray_v1_pad_statistical(
    TerlanNdArrayHandle array, const int64_t *before, int64_t before_count,
    const int64_t *after, int64_t after_count,
    const int64_t *before_lengths, int64_t before_length_count,
    const int64_t *after_lengths, int64_t after_length_count, int64_t mode,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = validate_padding(
      array, before, before_count, after, after_count, out_array);
  enum PaddingMode padding_mode;
  int64_t axis;
  if (status != TERLAN_NDARRAY_OK ||
      !((before_length_count == 0) ||
        (array->rank > 0 &&
         (before_length_count == 1 || before_length_count == array->rank))) ||
      !((after_length_count == 0) ||
        (array->rank > 0 &&
         (after_length_count == 1 || after_length_count == array->rank))) ||
      (before_length_count > 0 && before_lengths == NULL) ||
      (after_length_count > 0 && after_lengths == NULL) ||
      mode < 0 || mode > 3) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (before_length_count == 0) {
    before_lengths = NULL;
  }
  if (after_length_count == 0) {
    after_lengths = NULL;
  }
  for (axis = 0; axis < array->rank; ++axis) {
    const int64_t before_length = before_lengths == NULL
                                      ? array->shape[axis]
                                      : before_lengths[before_length_count == 1
                                                           ? 0
                                                           : axis];
    const int64_t after_length = after_lengths == NULL
                                     ? array->shape[axis]
                                     : after_lengths[after_length_count == 1
                                                         ? 0
                                                         : axis];
    if (before_length < 0 || after_length < 0 ||
        ((mode == 0 || mode == 1) &&
         ((before_length_count > 0 && before_length == 0) ||
          (after_length_count > 0 && after_length == 0)))) {
      return TERLAN_NDARRAY_INVALID_ARGUMENT;
    }
  }
  padding_mode = (enum PaddingMode)(PADDING_MINIMUM + mode);
  return pad_all(array, before, before_count, after, after_count, NULL, NULL, 0,
                 NULL, NULL, before_lengths, before_length_count, after_lengths,
                 after_length_count, padding_mode, out_array);
}

#define DEFINE_STATISTICAL_PADDING(name, mode)                                \
  TerlanNdArrayStatus terlan_ndarray_v1_pad_##name(                            \
      TerlanNdArrayHandle array, const int64_t *before, int64_t before_count,  \
      const int64_t *after, int64_t after_count,                               \
      const int64_t *before_lengths, int64_t before_length_count,              \
      const int64_t *after_lengths, int64_t after_length_count,                \
      TerlanNdArrayHandle *out_array) {                                        \
    return terlan_ndarray_v1_pad_statistical(                                  \
        array, before, before_count, after, after_count, before_lengths,        \
        before_length_count, after_lengths, after_length_count, mode,           \
        out_array);                                                            \
  }

DEFINE_STATISTICAL_PADDING(minimum, 0)
DEFINE_STATISTICAL_PADDING(maximum, 1)
DEFINE_STATISTICAL_PADDING(mean, 2)
DEFINE_STATISTICAL_PADDING(median, 3)
