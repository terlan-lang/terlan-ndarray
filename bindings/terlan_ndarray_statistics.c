#include "terlan_ndarray_internal.h"

#include <complex.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

enum {
  QUANTILE_LINEAR = 0,
  QUANTILE_LOWER = 1,
  QUANTILE_HIGHER = 2,
  QUANTILE_MIDPOINT = 3,
  QUANTILE_NEAREST = 4
};

static int compare_scalars(const void *lhs, const void *rhs) {
  return scalar_order_compare(*(const TerlanNdArrayScalar *)lhs,
                              *(const TerlanNdArrayScalar *)rhs);
}

static int32_t statistic_dtype(int32_t dtype) {
  if (dtype == TERLAN_NDARRAY_DTYPE_COMPLEX64 ||
      dtype == TERLAN_NDARRAY_DTYPE_COMPLEX128) {
    return TERLAN_NDARRAY_DTYPE_COMPLEX128;
  }
  return TERLAN_NDARRAY_DTYPE_FLOAT64;
}

static int32_t median_dtype(int32_t dtype) {
  if (dtype_is_float(dtype) || dtype_is_complex(dtype)) {
    return dtype;
  }
  return TERLAN_NDARRAY_DTYPE_FLOAT64;
}

static int32_t quantile_dtype(int32_t dtype, int32_t interpolation,
                              const double *quantiles, int64_t quantile_count,
                              int64_t reduced_count) {
  int64_t index;
  if (interpolation == QUANTILE_LOWER ||
      interpolation == QUANTILE_HIGHER ||
      interpolation == QUANTILE_NEAREST) {
    return dtype;
  }
  if (!dtype_is_bool(dtype)) {
    return statistic_dtype(dtype);
  }
  for (index = 0; index < quantile_count; ++index) {
    const long double position = quantiles[index] * (reduced_count - 1);
    if (floorl(position) != position) {
      return TERLAN_NDARRAY_DTYPE_FLOAT64;
    }
  }
  return dtype;
}

static TerlanNdArrayScalar interpolate_quantile(
    const TerlanNdArrayScalar *values, int64_t count, long double quantile,
    int32_t interpolation) {
  const long double position = quantile * (count - 1);
  const int64_t lower = (int64_t)floorl(position);
  const int64_t higher = (int64_t)ceill(position);
  const long double fraction = position - lower;
  TerlanNdArrayScalar result;
  if (lower == higher || fraction == 0.0L) {
    return values[lower];
  }
  if (interpolation == QUANTILE_LOWER) {
    return values[lower];
  }
  if (interpolation == QUANTILE_HIGHER) {
    return values[higher];
  }
  if (interpolation == QUANTILE_NEAREST) {
    return values[(int64_t)nearbyintl(position)];
  }
  if (interpolation == QUANTILE_MIDPOINT) {
    result.real = (values[lower].real + values[higher].real) / 2.0L;
    result.imag = (values[lower].imag + values[higher].imag) / 2.0L;
    return result;
  }
  result.real = values[lower].real * (1.0L - fraction) +
                values[higher].real * fraction;
  result.imag = values[lower].imag * (1.0L - fraction) +
                values[higher].imag * fraction;
  return result;
}

static int scalar_has_nan(TerlanNdArrayScalar value) {
  return isnan(value.real) || isnan(value.imag);
}

static int64_t quantile_group_capacity(
    TerlanNdArrayHandle array, const TerlanNdArrayReductionPlan *plan) {
  int64_t capacity = 1;
  int64_t axis;
  for (axis = 0; axis < array->rank; ++axis) {
    if (plan->reduced[axis] != 0) {
      capacity *= array->shape[axis];
    }
  }
  return capacity;
}

static TerlanNdArrayStatus quantile_output_shape(
    const TerlanNdArrayReductionPlan *plan, int64_t quantile_count,
    int prepend_quantiles, int64_t **out_shape, int64_t *out_rank) {
  int64_t index;
  *out_rank = plan->output_rank + (prepend_quantiles != 0 ? 1 : 0);
  *out_shape = NULL;
  if (*out_rank == 0) {
    return TERLAN_NDARRAY_OK;
  }
  *out_shape = checked_malloc((size_t)*out_rank * sizeof(**out_shape));
  if (*out_shape == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  if (prepend_quantiles != 0) {
    (*out_shape)[0] = quantile_count;
  }
  for (index = 0; index < plan->output_rank; ++index) {
    (*out_shape)[index + (prepend_quantiles != 0 ? 1 : 0)] =
        plan->output_shape[index];
  }
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus quantile_array_many(
    TerlanNdArrayHandle array, const double *quantiles, int64_t quantile_count,
    long double scale, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int32_t interpolation, int ignore_nan,
    int empty_is_nan, int preserve_inexact_dtype, int prepend_quantiles,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayReductionPlan plan;
  TerlanNdArrayScalar *groups = NULL;
  int64_t *counts = NULL;
  uint8_t *has_nan = NULL;
  double *normalized_quantiles = NULL;
  int64_t *output_shape = NULL;
  TerlanNdArrayHandle result = NULL;
  int64_t output_rank = 0;
  int64_t group_capacity = 0;
  int64_t index;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL || quantile_count < 0 ||
      (quantile_count > 0 && quantiles == NULL) || !(scale > 0.0L) ||
      interpolation < QUANTILE_LINEAR || interpolation > QUANTILE_NEAREST) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (quantile_count > 0) {
    normalized_quantiles =
        checked_malloc((size_t)quantile_count * sizeof(*normalized_quantiles));
    if (normalized_quantiles == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (index = 0; index < quantile_count; ++index) {
    const double normalized = (double)((long double)quantiles[index] / scale);
    if (!isfinite(normalized) || normalized < 0.0 || normalized > 1.0) {
      status = TERLAN_NDARRAY_INVALID_ARGUMENT;
      goto cleanup;
    }
    normalized_quantiles[index] = normalized;
  }
  status = reduction_plan_create(array, axes, axis_count, keep_dims, &plan);
  if (status != TERLAN_NDARRAY_OK) {
    goto cleanup;
  }
  group_capacity = quantile_group_capacity(array, &plan);
  status = quantile_output_shape(&plan, quantile_count, prepend_quantiles,
                                 &output_shape, &output_rank);
  if (status != TERLAN_NDARRAY_OK) {
    goto cleanup_plan;
  }
  if (array->numel > 0) {
    groups = checked_malloc((size_t)array->numel * sizeof(*groups));
  }
  if (plan.output_numel > 0) {
    counts = checked_calloc((size_t)plan.output_numel, sizeof(*counts));
    has_nan = checked_calloc((size_t)plan.output_numel, sizeof(*has_nan));
  }
  if ((array->numel > 0 && groups == NULL) ||
      (plan.output_numel > 0 && (counts == NULL || has_nan == NULL))) {
    status = TERLAN_NDARRAY_ALLOCATION_FAILED;
    goto cleanup_plan;
  }
  for (index = 0; index < array->numel; ++index) {
    TerlanNdArrayScalar value;
    int64_t output_index;
    int64_t reduced_index;
    reduction_plan_indexes(array, &plan, index, &output_index, &reduced_index);
    status = dtype_load(array_const_element_pointer(array, index), array->dtype,
                        &value);
    if (status != TERLAN_NDARRAY_OK) {
      goto cleanup_plan;
    }
    if (scalar_has_nan(value)) {
      has_nan[output_index] = 1;
      if (ignore_nan != 0) {
        continue;
      }
    }
    groups[output_index * group_capacity +
           counts[output_index]++] = value;
    (void)reduced_index;
  }
  status = create_empty_array(
      output_shape, output_rank,
      preserve_inexact_dtype != 0
          ? median_dtype(array->dtype)
          : quantile_dtype(array->dtype, interpolation, normalized_quantiles,
                           quantile_count, group_capacity),
      &result);
  if (status != TERLAN_NDARRAY_OK) {
    goto cleanup_plan;
  }
  for (index = 0; index < plan.output_numel; ++index) {
    TerlanNdArrayScalar *group = NULL;
    int64_t quantile_index;
    if (counts[index] == 0 && ignore_nan == 0 && empty_is_nan == 0 &&
        quantile_count > 0) {
      status = TERLAN_NDARRAY_DOMAIN_ERROR;
      goto cleanup_plan;
    }
    if (counts[index] > 1 && (ignore_nan != 0 || has_nan[index] == 0)) {
      group = groups + index * group_capacity;
      qsort(group, (size_t)counts[index], sizeof(*group), compare_scalars);
    } else if (counts[index] > 0) {
      group = groups + index * group_capacity;
    }
    for (quantile_index = 0; quantile_index < quantile_count;
         ++quantile_index) {
      TerlanNdArrayScalar value;
      const int64_t output_index =
          (prepend_quantiles != 0 ? quantile_index * plan.output_numel : 0) +
          index;
      if (counts[index] == 0 || (ignore_nan == 0 && has_nan[index] != 0)) {
        value.real = NAN;
        value.imag = dtype_is_complex(result->dtype) ? NAN : 0.0L;
      } else {
        value = interpolate_quantile(group, counts[index],
                                     normalized_quantiles[quantile_index],
                                     interpolation);
      }
      status = dtype_store(array_element_pointer(result, output_index),
                           result->dtype, value);
      if (status != TERLAN_NDARRAY_OK) {
        goto cleanup_plan;
      }
    }
  }
  *out_array = result;
  result = NULL;

cleanup_plan:
  if (result != NULL) {
    (void)terlan_ndarray_v1_delete(result);
  }
  free(has_nan);
  free(counts);
  free(groups);
  free(output_shape);
  reduction_plan_destroy(&plan);
cleanup:
  free(normalized_quantiles);
  return status;
}

static TerlanNdArrayStatus quantile_array(
    TerlanNdArrayHandle array, double quantile, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, int32_t interpolation,
    int ignore_nan, int empty_is_nan, int preserve_inexact_dtype,
    TerlanNdArrayHandle *out_array) {
  return quantile_array_many(array, &quantile, 1, 1.0L, axes, axis_count,
                             keep_dims, interpolation, ignore_nan,
                             empty_is_nan, preserve_inexact_dtype, 0,
                             out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_median(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return quantile_array(array, 0.5, axes, axis_count, keep_dims,
                        QUANTILE_LINEAR, 0, 1, 1, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_quantile(
    TerlanNdArrayHandle array, double quantile, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, int32_t interpolation,
    TerlanNdArrayHandle *out_array) {
  return quantile_array(array, quantile, axes, axis_count, keep_dims,
                        interpolation, 0, 0, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_percentile(
    TerlanNdArrayHandle array, double percentile, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, int32_t interpolation,
    TerlanNdArrayHandle *out_array) {
  return quantile_array_many(array, &percentile, 1, 100.0L, axes, axis_count,
                             keep_dims, interpolation, 0, 0, 0, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_quantiles(
    TerlanNdArrayHandle array, const double *quantiles, int64_t quantile_count,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims,
    int32_t interpolation, TerlanNdArrayHandle *out_array) {
  return quantile_array_many(array, quantiles, quantile_count, 1.0L, axes,
                             axis_count, keep_dims, interpolation, 0, 0, 0, 1,
                             out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_percentiles(
    TerlanNdArrayHandle array, const double *percentiles,
    int64_t percentile_count, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int32_t interpolation,
    TerlanNdArrayHandle *out_array) {
  return quantile_array_many(array, percentiles, percentile_count, 100.0L,
                             axes, axis_count, keep_dims, interpolation, 0, 0,
                             0, 1, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_nanquantile_axes(
    TerlanNdArrayHandle array, double quantile, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, int32_t interpolation,
    TerlanNdArrayHandle *out_array) {
  return quantile_array(array, quantile, axes, axis_count, keep_dims,
                        interpolation, 1, 1, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_nanpercentile_axes(
    TerlanNdArrayHandle array, double percentile, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, int32_t interpolation,
    TerlanNdArrayHandle *out_array) {
  return quantile_array_many(array, &percentile, 1, 100.0L, axes, axis_count,
                             keep_dims, interpolation, 1, 1, 0, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_nanquantiles(
    TerlanNdArrayHandle array, const double *quantiles, int64_t quantile_count,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims,
    int32_t interpolation, TerlanNdArrayHandle *out_array) {
  return quantile_array_many(array, quantiles, quantile_count, 1.0L, axes,
                             axis_count, keep_dims, interpolation, 1, 1, 0, 1,
                             out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_nanpercentiles(
    TerlanNdArrayHandle array, const double *percentiles,
    int64_t percentile_count, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int32_t interpolation,
    TerlanNdArrayHandle *out_array) {
  return quantile_array_many(array, percentiles, percentile_count, 100.0L,
                             axes, axis_count, keep_dims, interpolation, 1, 1,
                             0, 1, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_nanmedian_axes(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return quantile_array(array, 0.5, axes, axis_count, keep_dims,
                        QUANTILE_LINEAR, 1, 1, 1, out_array);
}

static TerlanNdArrayStatus quantile_finish_into(
    TerlanNdArrayHandle destination, TerlanNdArrayStatus status,
    TerlanNdArrayHandle result) {
  return array_finish_into(destination, status, result,
                           TERLAN_NDARRAY_CAST_UNSAFE);
}

TerlanNdArrayStatus terlan_ndarray_v1_median_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status = terlan_ndarray_v1_median(
      array, axes, axis_count, keep_dims, &result);
  return quantile_finish_into(destination, status, result);
}

TerlanNdArrayStatus terlan_ndarray_v1_nanmedian_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status = terlan_ndarray_v1_nanmedian_axes(
      array, axes, axis_count, keep_dims, &result);
  return quantile_finish_into(destination, status, result);
}

TerlanNdArrayStatus terlan_ndarray_v1_quantile_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    double quantile, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int32_t interpolation) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status = terlan_ndarray_v1_quantile(
      array, quantile, axes, axis_count, keep_dims, interpolation, &result);
  return quantile_finish_into(destination, status, result);
}

TerlanNdArrayStatus terlan_ndarray_v1_percentile_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    double percentile, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int32_t interpolation) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status = terlan_ndarray_v1_percentile(
      array, percentile, axes, axis_count, keep_dims, interpolation, &result);
  return quantile_finish_into(destination, status, result);
}

TerlanNdArrayStatus terlan_ndarray_v1_quantiles_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    const double *quantiles, int64_t quantile_count, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, int32_t interpolation) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status = terlan_ndarray_v1_quantiles(
      array, quantiles, quantile_count, axes, axis_count, keep_dims,
      interpolation, &result);
  return quantile_finish_into(destination, status, result);
}

TerlanNdArrayStatus terlan_ndarray_v1_percentiles_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    const double *percentiles, int64_t percentile_count, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, int32_t interpolation) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status = terlan_ndarray_v1_percentiles(
      array, percentiles, percentile_count, axes, axis_count, keep_dims,
      interpolation, &result);
  return quantile_finish_into(destination, status, result);
}

TerlanNdArrayStatus terlan_ndarray_v1_nanquantile_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    double quantile, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int32_t interpolation) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status = terlan_ndarray_v1_nanquantile_axes(
      array, quantile, axes, axis_count, keep_dims, interpolation, &result);
  return quantile_finish_into(destination, status, result);
}

TerlanNdArrayStatus terlan_ndarray_v1_nanpercentile_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    double percentile, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int32_t interpolation) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status = terlan_ndarray_v1_nanpercentile_axes(
      array, percentile, axes, axis_count, keep_dims, interpolation, &result);
  return quantile_finish_into(destination, status, result);
}

TerlanNdArrayStatus terlan_ndarray_v1_nanquantiles_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    const double *quantiles, int64_t quantile_count, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, int32_t interpolation) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status = terlan_ndarray_v1_nanquantiles(
      array, quantiles, quantile_count, axes, axis_count, keep_dims,
      interpolation, &result);
  return quantile_finish_into(destination, status, result);
}

TerlanNdArrayStatus terlan_ndarray_v1_nanpercentiles_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    const double *percentiles, int64_t percentile_count, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, int32_t interpolation) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status = terlan_ndarray_v1_nanpercentiles(
      array, percentiles, percentile_count, axes, axis_count, keep_dims,
      interpolation, &result);
  return quantile_finish_into(destination, status, result);
}

TerlanNdArrayStatus terlan_ndarray_v1_peak_to_peak(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status;
  TerlanNdArrayHandle minimum = NULL;
  TerlanNdArrayHandle maximum = NULL;
  if (out_array == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = terlan_ndarray_v1_minimum(array, axes, axis_count, keep_dims,
                                     &minimum);
  if (status == TERLAN_NDARRAY_OK) {
    status = terlan_ndarray_v1_maximum(array, axes, axis_count, keep_dims,
                                       &maximum);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = terlan_ndarray_v1_subtract(maximum, minimum, out_array);
  }
  if (maximum != NULL) {
    (void)terlan_ndarray_v1_delete(maximum);
  }
  if (minimum != NULL) {
    (void)terlan_ndarray_v1_delete(minimum);
  }
  return status;
}

static int64_t axis_coordinate(TerlanNdArrayHandle array, int64_t logical,
                               int64_t axis) {
  int64_t inner = 1;
  int64_t dimension;
  for (dimension = axis + 1; dimension < array->rank; ++dimension) {
    inner *= array->shape[dimension];
  }
  return (logical / inner) % array->shape[axis];
}

static int32_t average_dtype(TerlanNdArrayHandle array,
                             TerlanNdArrayHandle weights) {
  int32_t dtype = weights == NULL
                      ? array->dtype
                      : dtype_promote(array->dtype, weights->dtype);
  if (dtype_is_integer(array->dtype) || dtype_is_bool(array->dtype)) {
    dtype = dtype_promote(dtype, TERLAN_NDARRAY_DTYPE_FLOAT64);
  }
  return dtype;
}

static TerlanNdArrayStatus weighted_average_result(
    TerlanNdArrayHandle array, TerlanNdArrayHandle weights,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims,
    int return_weight_sum, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayReductionPlan plan;
  TerlanNdArrayScalar *sums = NULL;
  TerlanNdArrayScalar *weight_sums = NULL;
  TerlanNdArrayHandle result = NULL;
  int64_t vector_axis = -1;
  int vector_weights = 0;
  int64_t index;
  if (status == TERLAN_NDARRAY_OK && weights != NULL) {
    status = require_live(weights);
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (weights != NULL && !array_same_shape(array, weights)) {
    if (axis_count != 1 || axes == NULL || weights->rank != 1) {
      return TERLAN_NDARRAY_SHAPE_MISMATCH;
    }
    vector_axis = axes[0] < 0 ? axes[0] + array->rank : axes[0];
    if (vector_axis < 0 || vector_axis >= array->rank) {
      return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
    }
    if (weights->shape[0] != array->shape[vector_axis]) {
      return TERLAN_NDARRAY_SHAPE_MISMATCH;
    }
    vector_weights = 1;
  }
  status = reduction_plan_create(array, axes, axis_count, keep_dims, &plan);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (plan.output_numel > 0) {
    sums = checked_calloc((size_t)plan.output_numel, sizeof(*sums));
    weight_sums = checked_calloc((size_t)plan.output_numel,
                                 sizeof(*weight_sums));
  }
  if (plan.output_numel > 0 && (sums == NULL || weight_sums == NULL)) {
    status = TERLAN_NDARRAY_ALLOCATION_FAILED;
    goto cleanup;
  }
  for (index = 0; index < array->numel; ++index) {
    TerlanNdArrayScalar value;
    TerlanNdArrayScalar weight;
    int64_t output_index;
    int64_t reduced_index;
    const int64_t weight_index =
        vector_weights != 0
            ? axis_coordinate(array, index, vector_axis)
            : index;
    reduction_plan_indexes(array, &plan, index, &output_index, &reduced_index);
    status = dtype_load(array_const_element_pointer(array, index), array->dtype,
                        &value);
    if (status == TERLAN_NDARRAY_OK && weights != NULL) {
      status = dtype_load(array_const_element_pointer(weights, weight_index),
                          weights->dtype, &weight);
    } else if (status == TERLAN_NDARRAY_OK) {
      weight.real = 1.0L;
      weight.imag = 0.0L;
    }
    if (status != TERLAN_NDARRAY_OK) {
      goto cleanup;
    }
    value = scalar_multiply(value, weight);
    sums[output_index].real += value.real;
    sums[output_index].imag += value.imag;
    weight_sums[output_index].real += weight.real;
    weight_sums[output_index].imag += weight.imag;
    (void)reduced_index;
  }
  status = create_empty_array(plan.output_shape, plan.output_rank,
                              average_dtype(array, weights), &result);
  if (status != TERLAN_NDARRAY_OK) {
    goto cleanup;
  }
  for (index = 0; index < plan.output_numel; ++index) {
    TerlanNdArrayScalar value = weight_sums[index];
    if (return_weight_sum == 0) {
      if (weight_sums[index].real == 0.0L &&
          weight_sums[index].imag == 0.0L) {
        if (weights != NULL) {
          status = TERLAN_NDARRAY_DOMAIN_ERROR;
          goto cleanup;
        }
        value.real = NAN;
        value.imag = dtype_is_complex(result->dtype) ? NAN : 0.0L;
      } else {
        value = scalar_divide(sums[index], weight_sums[index]);
      }
    }
    status = dtype_store(array_element_pointer(result, index), result->dtype,
                         value);
    if (status != TERLAN_NDARRAY_OK) {
      goto cleanup;
    }
  }
  *out_array = result;
  result = NULL;

cleanup:
  if (result != NULL) {
    (void)terlan_ndarray_v1_delete(result);
  }
  free(weight_sums);
  free(sums);
  reduction_plan_destroy(&plan);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_weighted_average_axes(
    TerlanNdArrayHandle array, TerlanNdArrayHandle weights,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims,
    TerlanNdArrayHandle *out_array) {
  return weighted_average_result(array, weights, axes, axis_count, keep_dims,
                                 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_average_weight_sum(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return weighted_average_result(array, NULL, axes, axis_count, keep_dims, 1,
                                 out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_weighted_average_weight_sum(
    TerlanNdArrayHandle array, TerlanNdArrayHandle weights,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims,
    TerlanNdArrayHandle *out_array) {
  return weighted_average_result(array, weights, axes, axis_count, keep_dims,
                                 1, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_weighted_average(
    TerlanNdArrayHandle array, TerlanNdArrayHandle weights, int64_t axis,
    int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_v1_weighted_average_axes(
      array, weights, &axis, 1, keep_dims, out_array);
}
