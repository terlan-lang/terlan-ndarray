#include "terlan_ndarray_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static TerlanNdArrayStatus validate_histogram_range(
    TerlanNdArrayHandle array, int64_t bins, double minimum, double maximum,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  if (status != TERLAN_NDARRAY_OK || out_array == NULL || bins <= 0 ||
      !isfinite(minimum) || !isfinite(maximum) || !(maximum > minimum) ||
      dtype_is_complex(array->dtype)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  return TERLAN_NDARRAY_OK;
}

static int64_t histogram_bin(long double value, int64_t bins,
                             long double minimum, long double maximum) {
  int64_t bin;
  if (isnan(value) || value < minimum || value > maximum) {
    return -1;
  }
  if (value == maximum) {
    return bins - 1;
  }
  bin = (int64_t)floorl((value - minimum) * bins / (maximum - minimum));
  return bin >= 0 && bin < bins ? bin : -1;
}

TerlanNdArrayStatus terlan_ndarray_v1_histogram(
    TerlanNdArrayHandle array, int64_t bins, double minimum, double maximum,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status =
      validate_histogram_range(array, bins, minimum, maximum, out_array);
  TerlanNdArrayHandle result = NULL;
  int64_t index;
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  status = create_empty_array(&bins, 1, TERLAN_NDARRAY_DTYPE_INT64, &result);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (index = 0; index < array->numel; ++index) {
    TerlanNdArrayScalar value;
    int64_t bin;
    int64_t count;
    status = dtype_load(array_const_element_pointer(array, index), array->dtype,
                        &value);
    if (status != TERLAN_NDARRAY_OK) {
      goto fail;
    }
    bin = histogram_bin(value.real, bins, minimum, maximum);
    if (bin < 0) {
      continue;
    }
    (void)dtype_load(array_const_element_pointer(result, bin), result->dtype,
                     &value);
    if (value.real >= INT64_MAX) {
      status = TERLAN_NDARRAY_ARITHMETIC_OVERFLOW;
      goto fail;
    }
    count = (int64_t)value.real + 1;
    value.real = count;
    (void)dtype_store(array_element_pointer(result, bin), result->dtype, value);
  }
  *out_array = result;
  return TERLAN_NDARRAY_OK;

fail:
  (void)terlan_ndarray_v1_delete(result);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_histogram_bin_edges(
    TerlanNdArrayHandle array, int64_t bins, double minimum, double maximum,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status =
      validate_histogram_range(array, bins, minimum, maximum, out_array);
  TerlanNdArrayHandle result = NULL;
  int64_t length;
  int64_t index;
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (bins == INT64_MAX) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  length = bins + 1;
  status = create_empty_array(&length, 1, TERLAN_NDARRAY_DTYPE_FLOAT64,
                              &result);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (index = 0; index <= bins; ++index) {
    TerlanNdArrayScalar value = {
        minimum + (maximum - minimum) * (long double)index / bins, 0.0L};
    (void)dtype_store(array_element_pointer(result, index), result->dtype,
                      value);
  }
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus load_range_vector(TerlanNdArrayHandle array,
                                              int64_t dimensions,
                                              long double *out_values) {
  int64_t index;
  TerlanNdArrayStatus status = require_live(array);
  if (status != TERLAN_NDARRAY_OK || array->rank != 1 ||
      array->shape[0] != dimensions || dtype_is_complex(array->dtype)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  for (index = 0; index < dimensions; ++index) {
    TerlanNdArrayScalar value;
    status = dtype_load(array_const_element_pointer(array, index), array->dtype,
                        &value);
    if (status != TERLAN_NDARRAY_OK || !isfinite(value.real)) {
      return status != TERLAN_NDARRAY_OK ? status
                                         : TERLAN_NDARRAY_DOMAIN_ERROR;
    }
    out_values[index] = value.real;
  }
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_histogramdd(
    TerlanNdArrayHandle samples, int64_t bins, TerlanNdArrayHandle minima,
    TerlanNdArrayHandle maxima, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(samples);
  TerlanNdArrayHandle result = NULL;
  int64_t *shape = NULL;
  long double *lower = NULL;
  long double *upper = NULL;
  int64_t dimensions;
  int64_t sample;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL || samples->rank != 2 ||
      bins <= 0 || dtype_is_complex(samples->dtype)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  dimensions = samples->shape[1];
  if (dimensions <= 0) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  shape = checked_malloc((size_t)dimensions * sizeof(*shape));
  lower = checked_malloc((size_t)dimensions * sizeof(*lower));
  upper = checked_malloc((size_t)dimensions * sizeof(*upper));
  if (shape == NULL || lower == NULL || upper == NULL) {
    status = TERLAN_NDARRAY_ALLOCATION_FAILED;
    goto cleanup;
  }
  status = load_range_vector(minima, dimensions, lower);
  if (status == TERLAN_NDARRAY_OK) {
    status = load_range_vector(maxima, dimensions, upper);
  }
  if (status != TERLAN_NDARRAY_OK) {
    goto cleanup;
  }
  for (sample = 0; sample < dimensions; ++sample) {
    if (!(upper[sample] > lower[sample])) {
      status = TERLAN_NDARRAY_INVALID_ARGUMENT;
      goto cleanup;
    }
    shape[sample] = bins;
  }
  status = create_empty_array(shape, dimensions, TERLAN_NDARRAY_DTYPE_INT64,
                              &result);
  if (status != TERLAN_NDARRAY_OK) {
    goto cleanup;
  }
  for (sample = 0; sample < samples->shape[0]; ++sample) {
    int64_t coordinate = 0;
    int64_t dimension;
    int admitted = 1;
    for (dimension = 0; dimension < dimensions; ++dimension) {
      TerlanNdArrayScalar value;
      int64_t bin;
      (void)dtype_load(array_const_element_pointer(
                           samples, sample * dimensions + dimension),
                       samples->dtype, &value);
      bin = histogram_bin(value.real, bins, lower[dimension], upper[dimension]);
      if (bin < 0) {
        admitted = 0;
        break;
      }
      coordinate = coordinate * bins + bin;
    }
    if (admitted != 0) {
      TerlanNdArrayScalar count;
      (void)dtype_load(array_const_element_pointer(result, coordinate),
                       result->dtype, &count);
      count.real += 1.0L;
      (void)dtype_store(array_element_pointer(result, coordinate),
                        result->dtype, count);
    }
  }
  *out_array = result;
  result = NULL;

cleanup:
  if (result != NULL) {
    (void)terlan_ndarray_v1_delete(result);
  }
  free(upper);
  free(lower);
  free(shape);
  return status;
}

static TerlanNdArrayStatus bincount_array(
    TerlanNdArrayHandle values, TerlanNdArrayHandle weights,
    int64_t minimum_length, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(values);
  TerlanNdArrayHandle result = NULL;
  int64_t length = minimum_length;
  int64_t index;
  if (status == TERLAN_NDARRAY_OK && weights != NULL) {
    status = require_live(weights);
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL ||
      !dtype_is_integer(values->dtype) || values->rank != 1 ||
      (weights != NULL &&
       (weights->rank != 1 || weights->numel != values->numel ||
        dtype_is_complex(weights->dtype))) ||
      minimum_length < 0) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  for (index = 0; index < values->numel; ++index) {
    TerlanNdArrayScalar value;
    (void)dtype_load(array_const_element_pointer(values, index), values->dtype,
                     &value);
    if (value.real < 0.0L || value.real > INT64_MAX - 1) {
      return TERLAN_NDARRAY_DOMAIN_ERROR;
    }
    if ((int64_t)value.real + 1 > length) {
      length = (int64_t)value.real + 1;
    }
  }
  status = create_empty_array(
      &length, 1,
      weights == NULL ? TERLAN_NDARRAY_DTYPE_INT64
                      : TERLAN_NDARRAY_DTYPE_FLOAT64,
      &result);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (index = 0; index < values->numel; ++index) {
    TerlanNdArrayScalar value;
    TerlanNdArrayScalar weight;
    TerlanNdArrayScalar count;
    int64_t bin;
    (void)dtype_load(array_const_element_pointer(values, index), values->dtype,
                     &value);
    if (weights == NULL) {
      weight.real = 1.0L;
      weight.imag = 0.0L;
    } else {
      (void)dtype_load(array_const_element_pointer(weights, index),
                       weights->dtype, &weight);
    }
    bin = (int64_t)value.real;
    (void)dtype_load(array_const_element_pointer(result, bin), result->dtype,
                     &count);
    count.real += weight.real;
    status = dtype_store(array_element_pointer(result, bin), result->dtype,
                         count);
    if (status != TERLAN_NDARRAY_OK) {
      goto fail;
    }
  }
  *out_array = result;
  return TERLAN_NDARRAY_OK;

fail:
  (void)terlan_ndarray_v1_delete(result);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_bincount(
    TerlanNdArrayHandle values, TerlanNdArrayHandle weights,
    int64_t minimum_length, TerlanNdArrayHandle *out_array) {
  return bincount_array(values, weights, minimum_length, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_bincount_unweighted(
    TerlanNdArrayHandle values, int64_t minimum_length,
    TerlanNdArrayHandle *out_array) {
  return bincount_array(values, NULL, minimum_length, out_array);
}

static TerlanNdArrayStatus digitize_direction(TerlanNdArrayHandle bins,
                                              int *out_direction) {
  int direction = 0;
  int64_t index;
  for (index = 1; index < bins->numel; ++index) {
    TerlanNdArrayScalar previous;
    TerlanNdArrayScalar current;
    (void)dtype_load(array_const_element_pointer(bins, index - 1), bins->dtype,
                     &previous);
    (void)dtype_load(array_const_element_pointer(bins, index), bins->dtype,
                     &current);
    if (current.real > previous.real) {
      if (direction < 0) {
        return TERLAN_NDARRAY_INVALID_ARGUMENT;
      }
      direction = 1;
    } else if (current.real < previous.real) {
      if (direction > 0) {
        return TERLAN_NDARRAY_INVALID_ARGUMENT;
      }
      direction = -1;
    }
  }
  *out_direction = direction == 0 ? 1 : direction;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_digitize(
    TerlanNdArrayHandle values, TerlanNdArrayHandle bins, int32_t right,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(values);
  TerlanNdArrayHandle result = NULL;
  int direction = 1;
  int64_t index;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(bins);
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL || bins->rank != 1 ||
      dtype_is_complex(values->dtype) || dtype_is_complex(bins->dtype) ||
      (right != 0 && right != 1)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = digitize_direction(bins, &direction);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  status = create_empty_array(values->shape, values->rank,
                              TERLAN_NDARRAY_DTYPE_INT64, &result);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (index = 0; index < values->numel; ++index) {
    TerlanNdArrayScalar value;
    int64_t low = 0;
    int64_t high = bins->numel;
    (void)dtype_load(array_const_element_pointer(values, index), values->dtype,
                     &value);
    while (low < high) {
      const int64_t middle = low + (high - low) / 2;
      TerlanNdArrayScalar edge;
      const int64_t edge_index =
          direction > 0 ? middle : bins->numel - middle - 1;
      (void)dtype_load(array_const_element_pointer(bins, edge_index),
                       bins->dtype, &edge);
      if (right != 0 ? value.real <= edge.real : value.real < edge.real) {
        high = middle;
      } else {
        low = middle + 1;
      }
    }
    value.real = direction > 0 ? low : bins->numel - low;
    value.imag = 0.0L;
    (void)dtype_store(array_element_pointer(result, index), result->dtype,
                      value);
  }
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}
