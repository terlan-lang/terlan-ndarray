#include "terlan_ndarray_internal.h"

#include <complex.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

void reduction_plan_destroy(TerlanNdArrayReductionPlan *plan) {
  if (plan != NULL) {
    free(plan->reduced);
    free(plan->output_shape);
    plan->reduced = NULL;
    plan->output_shape = NULL;
  }
}

TerlanNdArrayStatus reduction_plan_create(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayReductionPlan *out_plan) {
  int64_t reduced_count = axis_count == 0 ? array->rank : axis_count;
  int64_t axis_index;
  int64_t output_dimension = 0;
  if (out_plan == NULL || axis_count < 0 ||
      (axis_count > 0 && axes == NULL) ||
      (keep_dims != 0 && keep_dims != 1)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (array->rank == 0 && axis_count > 0) {
    if (axis_count != 1 || (axes[0] != 0 && axes[0] != -1)) {
      return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
    }
    reduced_count = 0;
  }
  out_plan->output_shape = NULL;
  out_plan->reduced = NULL;
  out_plan->output_rank = keep_dims != 0 ? array->rank
                                         : array->rank - reduced_count;
  out_plan->output_numel = 1;
  if (array->rank > 0) {
    out_plan->reduced = checked_calloc((size_t)array->rank, 1);
    if (out_plan->reduced == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (axis_index = 0; axis_index < reduced_count; ++axis_index) {
    int64_t axis = axis_count == 0 ? axis_index : axes[axis_index];
    if (axis < 0) {
      axis += array->rank;
    }
    if (axis < 0 || axis >= array->rank) {
      reduction_plan_destroy(out_plan);
      return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
    }
    if (out_plan->reduced[axis] != 0) {
      reduction_plan_destroy(out_plan);
      return TERLAN_NDARRAY_DUPLICATE_AXIS;
    }
    out_plan->reduced[axis] = 1;
  }
  if (out_plan->output_rank > 0) {
    out_plan->output_shape =
        checked_malloc((size_t)out_plan->output_rank * sizeof(int64_t));
    if (out_plan->output_shape == NULL) {
      reduction_plan_destroy(out_plan);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    for (axis_index = 0; axis_index < array->rank; ++axis_index) {
      if (keep_dims != 0) {
        out_plan->output_shape[output_dimension++] =
            out_plan->reduced[axis_index] != 0 ? 1
                                               : array->shape[axis_index];
      } else if (out_plan->reduced[axis_index] == 0) {
        out_plan->output_shape[output_dimension++] = array->shape[axis_index];
      }
    }
  }
  for (axis_index = 0; axis_index < out_plan->output_rank; ++axis_index) {
    const int64_t dimension = out_plan->output_shape[axis_index];
    if (dimension == 0) {
      out_plan->output_numel = 0;
    } else if (out_plan->output_numel != 0) {
      if (out_plan->output_numel > INT64_MAX / dimension) {
        reduction_plan_destroy(out_plan);
        return TERLAN_NDARRAY_SIZE_OVERFLOW;
      }
      out_plan->output_numel *= dimension;
    }
  }
  return TERLAN_NDARRAY_OK;
}

void reduction_plan_indexes(TerlanNdArrayHandle array,
                            const TerlanNdArrayReductionPlan *plan,
                            int64_t logical_index, int64_t *out_output,
                            int64_t *out_reduced) {
  int64_t output = 0;
  int64_t reduced = 0;
  int64_t output_stride = 1;
  int64_t reduced_stride = 1;
  int64_t axis;
  for (axis = array->rank; axis > 0; --axis) {
    const int64_t index = axis - 1;
    const int64_t coordinate = logical_index % array->shape[index];
    logical_index /= array->shape[index];
    if (plan->reduced[index] != 0) {
      reduced += coordinate * reduced_stride;
      reduced_stride *= array->shape[index];
    } else {
      output += coordinate * output_stride;
      output_stride *= array->shape[index];
    }
  }
  *out_output = output;
  *out_reduced = reduced;
}

static int32_t accumulator_dtype(int32_t input, ReductionOperation operation) {
  if (operation == REDUCE_ANY || operation == REDUCE_ALL ||
      operation == REDUCE_LOGICAL_XOR) {
    return TERLAN_NDARRAY_DTYPE_BOOL;
  }
  if (operation == REDUCE_ARGMIN || operation == REDUCE_ARGMAX) {
    return TERLAN_NDARRAY_DTYPE_INT64;
  }
  if (operation == REDUCE_COUNT_NONZERO) {
    return TERLAN_NDARRAY_DTYPE_INT64;
  }
  if (operation == REDUCE_MEAN || operation == REDUCE_VARIANCE ||
      operation == REDUCE_STANDARD_DEVIATION) {
    if (dtype_is_complex(input)) {
      return operation == REDUCE_MEAN
                 ? input
                 : (input == TERLAN_NDARRAY_DTYPE_COMPLEX64
                        ? TERLAN_NDARRAY_DTYPE_FLOAT32
                        : TERLAN_NDARRAY_DTYPE_FLOAT64);
    }
    return dtype_is_integer(input) || dtype_is_bool(input)
               ? TERLAN_NDARRAY_DTYPE_FLOAT64
               : input;
  }
  if (operation == REDUCE_SUM || operation == REDUCE_PRODUCT) {
    if (input == TERLAN_NDARRAY_DTYPE_BOOL ||
        input == TERLAN_NDARRAY_DTYPE_INT8 ||
        input == TERLAN_NDARRAY_DTYPE_INT16 ||
        input == TERLAN_NDARRAY_DTYPE_INT32) {
      return TERLAN_NDARRAY_DTYPE_INT64;
    }
    if (input == TERLAN_NDARRAY_DTYPE_UINT8 ||
        input == TERLAN_NDARRAY_DTYPE_UINT16 ||
        input == TERLAN_NDARRAY_DTYPE_UINT32) {
      return TERLAN_NDARRAY_DTYPE_UINT64;
    }
  }
  return input;
}

static int reduction_is_bitwise(ReductionOperation operation) {
  return operation == REDUCE_BITWISE_AND ||
         operation == REDUCE_BITWISE_OR ||
         operation == REDUCE_BITWISE_XOR;
}

static uint64_t reduction_integer_bits(TerlanNdArrayScalar value,
                                       int32_t dtype) {
  if (dtype_is_signed(dtype)) {
    return dtype_truncate_integer((uint64_t)(int64_t)value.real, dtype);
  }
  return dtype_truncate_integer((uint64_t)value.real, dtype);
}

static TerlanNdArrayScalar reduction_integer_value(uint64_t bits,
                                                   int32_t dtype) {
  TerlanNdArrayScalar value = {0.0L, 0.0L};
  bits = dtype_truncate_integer(bits, dtype);
  value.real = dtype_is_signed(dtype)
                   ? (long double)dtype_signed_integer(bits, dtype)
                   : (long double)bits;
  return value;
}

static int scalar_truth(TerlanNdArrayScalar value) {
  return value.real != 0.0L || value.imag != 0.0L;
}

static int scalar_compare(TerlanNdArrayScalar lhs,
                          TerlanNdArrayScalar rhs) {
  if (lhs.real < rhs.real) {
    return -1;
  }
  if (lhs.real > rhs.real) {
    return 1;
  }
  return lhs.imag < rhs.imag ? -1 : lhs.imag > rhs.imag ? 1 : 0;
}

static TerlanNdArrayStatus reduction_selects(
    TerlanNdArrayHandle mask, TerlanNdArrayHandle array, int64_t input_index,
    int *out_selected) {
  TerlanNdArrayScalar value;
  TerlanNdArrayStatus status;
  if (mask == NULL) {
    *out_selected = 1;
    return TERLAN_NDARRAY_OK;
  }
  status = dtype_load(
      array_const_element_pointer(
          mask, array_broadcast_index(mask, array->shape, array->rank,
                                      input_index)),
      mask->dtype, &value);
  if (status == TERLAN_NDARRAY_OK) {
    *out_selected = scalar_truth(value);
  }
  return status;
}

TerlanNdArrayStatus terlan_ndarray_reduce_configured(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int64_t correction, ReductionOperation operation,
    int32_t skip_nan, TerlanNdArrayHandle mask, TerlanNdArrayHandle initial,
    int32_t requested_dtype, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayReductionPlan plan;
  TerlanNdArrayScalar *accumulators = NULL;
  int64_t *counts = NULL;
  int64_t *arguments = NULL;
  uint8_t *initialized = NULL;
  TerlanNdArrayHandle result = NULL;
  int32_t output_dtype = 0;
  int64_t input_index;
  int64_t output_index;
  if (status == TERLAN_NDARRAY_OK && mask != NULL) {
    status = require_live(mask);
  }
  if (status == TERLAN_NDARRAY_OK && initial != NULL) {
    status = require_live(initial);
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL || correction < 0) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (mask != NULL &&
      !array_broadcast_compatible(mask, array->shape, array->rank)) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  if (initial != NULL && initial->numel != 1) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  if (reduction_is_bitwise(operation) &&
      !dtype_is_integer(array->dtype) && !dtype_is_bool(array->dtype)) {
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  output_dtype = requested_dtype == 0
                     ? accumulator_dtype(array->dtype, operation)
                     : requested_dtype;
  if (dtype_size(output_dtype) == 0) {
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  *out_array = NULL;
  status = reduction_plan_create(array, axes, axis_count, keep_dims, &plan);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (plan.output_numel > 0) {
    accumulators = checked_calloc((size_t)plan.output_numel,
                                  sizeof(*accumulators));
    counts = checked_calloc((size_t)plan.output_numel, sizeof(*counts));
    arguments = checked_calloc((size_t)plan.output_numel, sizeof(*arguments));
    initialized = checked_calloc((size_t)plan.output_numel, 1);
    if (accumulators == NULL || counts == NULL || arguments == NULL ||
        initialized == NULL) {
      status = TERLAN_NDARRAY_ALLOCATION_FAILED;
      goto cleanup;
    }
    if (operation == REDUCE_PRODUCT || operation == REDUCE_ALL ||
        operation == REDUCE_BITWISE_AND) {
      for (output_index = 0; output_index < plan.output_numel; ++output_index) {
        if (operation == REDUCE_BITWISE_AND) {
          accumulators[output_index] = reduction_integer_value(
              UINT64_MAX, output_dtype);
        } else {
          accumulators[output_index].real = 1.0L;
        }
      }
    }
    if (initial != NULL) {
      TerlanNdArrayScalar initial_value;
      status = dtype_load(array_const_element_pointer(initial, 0),
                          initial->dtype, &initial_value);
      if (status != TERLAN_NDARRAY_OK) {
        goto cleanup;
      }
      for (output_index = 0; output_index < plan.output_numel; ++output_index) {
        accumulators[output_index] = initial_value;
        initialized[output_index] = 1;
      }
    }
  }
  for (input_index = 0; input_index < array->numel; ++input_index) {
    TerlanNdArrayScalar value;
    int64_t reduced_index;
    int selected;
    reduction_plan_indexes(array, &plan, input_index, &output_index,
                           &reduced_index);
    status = reduction_selects(mask, array, input_index, &selected);
    if (status != TERLAN_NDARRAY_OK) {
      goto cleanup;
    }
    if (!selected) {
      continue;
    }
    status = dtype_load(array_const_element_pointer(array, input_index),
                        array->dtype, &value);
    if (status != TERLAN_NDARRAY_OK) {
      goto cleanup;
    }
    if (skip_nan != 0 && (isnan(value.real) || isnan(value.imag))) {
      continue;
    }
    ++counts[output_index];
    switch (operation) {
    case REDUCE_SUM:
    case REDUCE_MEAN:
    case REDUCE_VARIANCE:
    case REDUCE_STANDARD_DEVIATION:
      accumulators[output_index].real += value.real;
      accumulators[output_index].imag += value.imag;
      break;
    case REDUCE_PRODUCT: {
      const long double complex current =
          accumulators[output_index].real +
          accumulators[output_index].imag * I;
      const long double complex product =
          current * (value.real + value.imag * I);
      accumulators[output_index].real = creall(product);
      accumulators[output_index].imag = cimagl(product);
      break;
    }
    case REDUCE_ANY:
      accumulators[output_index].real =
          scalar_truth(accumulators[output_index]) || scalar_truth(value);
      break;
    case REDUCE_COUNT_NONZERO:
      accumulators[output_index].real += scalar_truth(value) ? 1.0L : 0.0L;
      break;
    case REDUCE_ALL:
      accumulators[output_index].real =
          scalar_truth(accumulators[output_index]) && scalar_truth(value);
      break;
    case REDUCE_LOGICAL_XOR:
      accumulators[output_index].real =
          scalar_truth(accumulators[output_index]) != scalar_truth(value);
      break;
    case REDUCE_BITWISE_AND:
    case REDUCE_BITWISE_OR:
    case REDUCE_BITWISE_XOR: {
      const uint64_t current =
          reduction_integer_bits(accumulators[output_index], output_dtype);
      const uint64_t operand = reduction_integer_bits(value, array->dtype);
      const uint64_t reduced =
          operation == REDUCE_BITWISE_AND
              ? current & operand
              : operation == REDUCE_BITWISE_OR ? current | operand
                                                : current ^ operand;
      accumulators[output_index] =
          reduction_integer_value(reduced, output_dtype);
      break;
    }
    case REDUCE_MINIMUM:
    case REDUCE_MAXIMUM:
    case REDUCE_ARGMIN:
    case REDUCE_ARGMAX: {
      const int value_nan = isnan(value.real) || isnan(value.imag);
      const int accumulator_nan =
          initialized[output_index] != 0 &&
          (isnan(accumulators[output_index].real) ||
           isnan(accumulators[output_index].imag));
      const int comparison =
          initialized[output_index] == 0 || value_nan || accumulator_nan
              ? 0
              : scalar_compare(value, accumulators[output_index]);
      const int replace =
          initialized[output_index] == 0 ||
          (value_nan && !accumulator_nan) ||
          (!value_nan && !accumulator_nan &&
           ((operation == REDUCE_MINIMUM || operation == REDUCE_ARGMIN)
                ? comparison < 0
                : comparison > 0));
      if (replace) {
        accumulators[output_index] = value;
        arguments[output_index] = reduced_index;
        initialized[output_index] = 1;
      }
      break;
    }
    }
  }
  if (operation == REDUCE_VARIANCE ||
      operation == REDUCE_STANDARD_DEVIATION) {
    TerlanNdArrayScalar *squared = checked_calloc(
        (size_t)plan.output_numel, sizeof(*squared));
    if (squared == NULL && plan.output_numel > 0) {
      status = TERLAN_NDARRAY_ALLOCATION_FAILED;
      goto cleanup;
    }
    for (input_index = 0; input_index < array->numel; ++input_index) {
      TerlanNdArrayScalar value;
      int64_t reduced_index;
      long double delta_real;
      long double delta_imag;
      int selected;
      reduction_plan_indexes(array, &plan, input_index, &output_index,
                             &reduced_index);
      (void)reduced_index;
      status = reduction_selects(mask, array, input_index, &selected);
      if (status != TERLAN_NDARRAY_OK) {
        free(squared);
        goto cleanup;
      }
      if (!selected) {
        continue;
      }
      status = dtype_load(array_const_element_pointer(array, input_index),
                          array->dtype, &value);
      if (status != TERLAN_NDARRAY_OK) {
        free(squared);
        goto cleanup;
      }
      if (skip_nan != 0 && (isnan(value.real) || isnan(value.imag))) {
        continue;
      }
      delta_real = value.real -
                   accumulators[output_index].real / counts[output_index];
      delta_imag = value.imag -
                   accumulators[output_index].imag / counts[output_index];
      squared[output_index].real +=
          delta_real * delta_real + delta_imag * delta_imag;
    }
    free(accumulators);
    accumulators = squared;
  }
  status = create_empty_array(plan.output_shape, plan.output_rank, output_dtype,
                              &result);
  if (status != TERLAN_NDARRAY_OK) {
    goto cleanup;
  }
  for (output_index = 0; output_index < plan.output_numel; ++output_index) {
    TerlanNdArrayScalar value = accumulators[output_index];
    if ((operation == REDUCE_MINIMUM || operation == REDUCE_MAXIMUM ||
         operation == REDUCE_ARGMIN || operation == REDUCE_ARGMAX) &&
        counts[output_index] == 0 && initialized[output_index] == 0) {
      if (skip_nan != 0 &&
          (operation == REDUCE_MINIMUM || operation == REDUCE_MAXIMUM) &&
          array->numel > 0 &&
          (dtype_is_float(output_dtype) || dtype_is_complex(output_dtype))) {
        value.real = NAN;
        value.imag = dtype_is_complex(output_dtype) ? NAN : 0.0L;
      } else {
        status = TERLAN_NDARRAY_DOMAIN_ERROR;
        goto cleanup;
      }
    }
    if (operation == REDUCE_MEAN) {
      if (counts[output_index] == 0) {
        value.real = NAN;
        value.imag = 0.0L;
      } else {
        value.real /= counts[output_index];
        value.imag /= counts[output_index];
      }
    } else if (operation == REDUCE_VARIANCE ||
               operation == REDUCE_STANDARD_DEVIATION) {
      if (counts[output_index] <= correction) {
        value.real = NAN;
        value.imag = 0.0L;
      } else {
        value.real /= counts[output_index] - correction;
        if (operation == REDUCE_STANDARD_DEVIATION) {
          value.real = sqrtl(value.real);
        }
      }
    } else if (operation == REDUCE_ARGMIN || operation == REDUCE_ARGMAX) {
      value.real = arguments[output_index];
      value.imag = 0.0L;
    }
    status = dtype_store(array_element_pointer(result, output_index),
                         output_dtype, value);
    if (status != TERLAN_NDARRAY_OK) {
      status = status == TERLAN_NDARRAY_CAST_OVERFLOW
                   ? TERLAN_NDARRAY_ARITHMETIC_OVERFLOW
                   : status;
      goto cleanup;
    }
  }
  *out_array = result;
  result = NULL;

cleanup:
  if (result != NULL) {
    (void)terlan_ndarray_v1_delete(result);
  }
  free(initialized);
  free(arguments);
  free(counts);
  free(accumulators);
  reduction_plan_destroy(&plan);
  return status;
}

static TerlanNdArrayStatus reduce_array(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int64_t correction, ReductionOperation operation,
    int32_t skip_nan, TerlanNdArrayHandle mask, int32_t requested_dtype,
    TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_reduce_configured(
      array, axes, axis_count, keep_dims, correction, operation, skip_nan,
      mask, NULL, requested_dtype, out_array);
}

#define REDUCTION_WRAPPER(name, operation)                                    \
  TerlanNdArrayStatus name(TerlanNdArrayHandle array, const int64_t *axes,    \
                           int64_t axis_count, int32_t keep_dims,              \
                           TerlanNdArrayHandle *out_array) {                   \
    return reduce_array(array, axes, axis_count, keep_dims, 0, operation, 0,  \
                        NULL, 0, out_array);                                   \
  }

REDUCTION_WRAPPER(terlan_ndarray_v1_sum, REDUCE_SUM)
REDUCTION_WRAPPER(terlan_ndarray_v1_product, REDUCE_PRODUCT)
REDUCTION_WRAPPER(terlan_ndarray_v1_mean, REDUCE_MEAN)
REDUCTION_WRAPPER(terlan_ndarray_v1_minimum, REDUCE_MINIMUM)
REDUCTION_WRAPPER(terlan_ndarray_v1_maximum, REDUCE_MAXIMUM)
REDUCTION_WRAPPER(terlan_ndarray_v1_any, REDUCE_ANY)
REDUCTION_WRAPPER(terlan_ndarray_v1_all, REDUCE_ALL)
REDUCTION_WRAPPER(terlan_ndarray_v1_logical_xor_reduce, REDUCE_LOGICAL_XOR)
REDUCTION_WRAPPER(terlan_ndarray_v1_argmin, REDUCE_ARGMIN)
REDUCTION_WRAPPER(terlan_ndarray_v1_argmax, REDUCE_ARGMAX)

TerlanNdArrayStatus terlan_ndarray_v1_count_nonzero_axes(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status;
  TerlanNdArrayHandle result = NULL;
  int64_t logical_index;
  if (out_array == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = require_live(array);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (axis_count < 0 || (keep_dims != 0 && keep_dims != 1)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (axis_count != 0) {
    return reduce_array(array, axes, axis_count, keep_dims, 0,
                        REDUCE_COUNT_NONZERO, 0, NULL, 0, out_array);
  }
  status = create_empty_array(array->shape, array->rank,
                              TERLAN_NDARRAY_DTYPE_INT64, &result);
  for (logical_index = 0;
       status == TERLAN_NDARRAY_OK && logical_index < array->numel;
       ++logical_index) {
    TerlanNdArrayScalar source;
    TerlanNdArrayScalar count = {0.0L, 0.0L};
    status = dtype_load(array_const_element_pointer(array, logical_index),
                        array->dtype, &source);
    if (status == TERLAN_NDARRAY_OK) {
      count.real = scalar_truth(source) ? 1.0L : 0.0L;
      status = dtype_store(array_element_pointer(result, logical_index),
                           TERLAN_NDARRAY_DTYPE_INT64, count);
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

TerlanNdArrayStatus terlan_ndarray_v1_any_where(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return reduce_array(array, axes, axis_count, keep_dims, 0, REDUCE_ANY, 0,
                      mask, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_all_where(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return reduce_array(array, axes, axis_count, keep_dims, 0, REDUCE_ALL, 0,
                      mask, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_variance(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int64_t correction, TerlanNdArrayHandle *out_array) {
  return reduce_array(array, axes, axis_count, keep_dims, correction,
                      REDUCE_VARIANCE, 0, NULL, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_standard_deviation(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int64_t correction, TerlanNdArrayHandle *out_array) {
  return reduce_array(array, axes, axis_count, keep_dims, correction,
                      REDUCE_STANDARD_DEVIATION, 0, NULL, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_mean_where(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return reduce_array(array, axes, axis_count, keep_dims, 0, REDUCE_MEAN, 0,
                      mask, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_mean_as(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int32_t dtype, TerlanNdArrayHandle *out_array) {
  return reduce_array(array, axes, axis_count, keep_dims, 0, REDUCE_MEAN, 0,
                      NULL, dtype, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_variance_where(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, int64_t correction,
    TerlanNdArrayHandle *out_array) {
  return reduce_array(array, axes, axis_count, keep_dims, correction,
                      REDUCE_VARIANCE, 0, mask, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_variance_as(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int64_t correction, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  return reduce_array(array, axes, axis_count, keep_dims, correction,
                      REDUCE_VARIANCE, 0, NULL, dtype, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_standard_deviation_where(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, int64_t correction,
    TerlanNdArrayHandle *out_array) {
  return reduce_array(array, axes, axis_count, keep_dims, correction,
                      REDUCE_STANDARD_DEVIATION, 0, mask, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_standard_deviation_as(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int64_t correction, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  return reduce_array(array, axes, axis_count, keep_dims, correction,
                      REDUCE_STANDARD_DEVIATION, 0, NULL, dtype, out_array);
}

static TerlanNdArrayStatus finish_reduction_into(
    TerlanNdArrayHandle destination, TerlanNdArrayStatus reduction_status,
    TerlanNdArrayHandle result) {
  return array_finish_into(destination, reduction_status, result,
                           TERLAN_NDARRAY_CAST_UNSAFE);
}

static TerlanNdArrayStatus finish_argument_reduction_into(
    TerlanNdArrayHandle destination, TerlanNdArrayStatus reduction_status,
    TerlanNdArrayHandle result) {
  TerlanNdArrayStatus status = require_live(destination);
  if (status != TERLAN_NDARRAY_OK) {
    if (result != NULL) {
      (void)terlan_ndarray_v1_delete(result);
    }
    return status;
  }
  if (!dtype_is_integer(destination->dtype) &&
      destination->dtype != TERLAN_NDARRAY_DTYPE_BOOL) {
    if (result != NULL) {
      (void)terlan_ndarray_v1_delete(result);
    }
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  return finish_reduction_into(destination, reduction_status, result);
}

TerlanNdArrayStatus terlan_ndarray_v1_argmin_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status =
      terlan_ndarray_v1_argmin(array, axes, axis_count, keep_dims, &result);
  return finish_argument_reduction_into(destination, status, result);
}

TerlanNdArrayStatus terlan_ndarray_v1_argmax_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status =
      terlan_ndarray_v1_argmax(array, axes, axis_count, keep_dims, &result);
  return finish_argument_reduction_into(destination, status, result);
}

TerlanNdArrayStatus terlan_ndarray_v1_mean_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status =
      terlan_ndarray_v1_mean(array, axes, axis_count, keep_dims, &result);
  return finish_reduction_into(destination, status, result);
}

TerlanNdArrayStatus terlan_ndarray_v1_variance_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims,
    int64_t correction) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status = terlan_ndarray_v1_variance(
      array, axes, axis_count, keep_dims, correction, &result);
  return finish_reduction_into(destination, status, result);
}

TerlanNdArrayStatus terlan_ndarray_v1_standard_deviation_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims,
    int64_t correction) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status = terlan_ndarray_v1_standard_deviation(
      array, axes, axis_count, keep_dims, correction, &result);
  return finish_reduction_into(destination, status, result);
}

#define NAN_REDUCTION_WRAPPER(name, operation)                                \
  TerlanNdArrayStatus name(TerlanNdArrayHandle array, const int64_t *axes,    \
                           int64_t axis_count, int32_t keep_dims,              \
                           TerlanNdArrayHandle *out_array) {                   \
    return reduce_array(array, axes, axis_count, keep_dims, 0, operation, 1,  \
                        NULL, 0, out_array);                                   \
  }

NAN_REDUCTION_WRAPPER(terlan_ndarray_v1_nan_sum, REDUCE_SUM)
NAN_REDUCTION_WRAPPER(terlan_ndarray_v1_nan_product, REDUCE_PRODUCT)
NAN_REDUCTION_WRAPPER(terlan_ndarray_v1_nan_mean, REDUCE_MEAN)
NAN_REDUCTION_WRAPPER(terlan_ndarray_v1_nan_minimum, REDUCE_MINIMUM)
NAN_REDUCTION_WRAPPER(terlan_ndarray_v1_nan_maximum, REDUCE_MAXIMUM)

TerlanNdArrayStatus terlan_ndarray_v1_nan_variance(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int64_t correction, TerlanNdArrayHandle *out_array) {
  return reduce_array(array, axes, axis_count, keep_dims, correction,
                      REDUCE_VARIANCE, 1, NULL, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_nan_standard_deviation(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int64_t correction, TerlanNdArrayHandle *out_array) {
  return reduce_array(array, axes, axis_count, keep_dims, correction,
                      REDUCE_STANDARD_DEVIATION, 1, NULL, 0, out_array);
}

static TerlanNdArrayStatus nan_reduction_as(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int64_t correction, ReductionOperation operation,
    int32_t requested_dtype, TerlanNdArrayHandle *out_array) {
  const TerlanNdArrayStatus status = require_live(array);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if ((operation == REDUCE_MEAN || operation == REDUCE_VARIANCE ||
       operation == REDUCE_STANDARD_DEVIATION) &&
      !dtype_is_float(requested_dtype) && !dtype_is_complex(requested_dtype)) {
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  return reduce_array(array, axes, axis_count, keep_dims, correction, operation,
                      1, NULL, requested_dtype, out_array);
}

#define NAN_REDUCTION_AS_WRAPPER(name, operation)                             \
  TerlanNdArrayStatus name(TerlanNdArrayHandle array, const int64_t *axes,    \
                           int64_t axis_count, int32_t keep_dims,              \
                           int32_t requested_dtype,                            \
                           TerlanNdArrayHandle *out_array) {                   \
    return nan_reduction_as(array, axes, axis_count, keep_dims, 0, operation, \
                            requested_dtype, out_array);                       \
  }

NAN_REDUCTION_AS_WRAPPER(terlan_ndarray_v1_nan_sum_as, REDUCE_SUM)
NAN_REDUCTION_AS_WRAPPER(terlan_ndarray_v1_nan_product_as, REDUCE_PRODUCT)
NAN_REDUCTION_AS_WRAPPER(terlan_ndarray_v1_nan_mean_as, REDUCE_MEAN)

TerlanNdArrayStatus terlan_ndarray_v1_nan_variance_as(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int64_t correction, int32_t requested_dtype,
    TerlanNdArrayHandle *out_array) {
  return nan_reduction_as(array, axes, axis_count, keep_dims, correction,
                          REDUCE_VARIANCE, requested_dtype, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_nan_standard_deviation_as(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int64_t correction, int32_t requested_dtype,
    TerlanNdArrayHandle *out_array) {
  return nan_reduction_as(array, axes, axis_count, keep_dims, correction,
                          REDUCE_STANDARD_DEVIATION, requested_dtype,
                          out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_nan_argmin(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return reduce_array(array, axes, axis_count, keep_dims, 0, REDUCE_ARGMIN, 1,
                      NULL, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_nan_argmax(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return reduce_array(array, axes, axis_count, keep_dims, 0, REDUCE_ARGMAX, 1,
                      NULL, 0, out_array);
}

static TerlanNdArrayStatus nan_reduction_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims,
    int64_t correction, ReductionOperation operation) {
  TerlanNdArrayStatus status = require_live(destination);
  TerlanNdArrayHandle result = NULL;
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if ((operation == REDUCE_MEAN || operation == REDUCE_VARIANCE ||
       operation == REDUCE_STANDARD_DEVIATION) &&
      !dtype_is_float(destination->dtype) &&
      !dtype_is_complex(destination->dtype)) {
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  status = reduce_array(array, axes, axis_count, keep_dims, correction,
                        operation, 1, NULL, 0, &result);
  return array_finish_into(destination, status, result,
                           TERLAN_NDARRAY_CAST_UNSAFE);
}

#define NAN_REDUCTION_INTO_WRAPPER(name, operation)                           \
  TerlanNdArrayStatus name(                                                   \
      TerlanNdArrayHandle destination, TerlanNdArrayHandle array,             \
      const int64_t *axes, int64_t axis_count, int32_t keep_dims) {           \
    return nan_reduction_into(destination, array, axes, axis_count, keep_dims, \
                              0, operation);                                  \
  }

NAN_REDUCTION_INTO_WRAPPER(terlan_ndarray_v1_nan_sum_into, REDUCE_SUM)
NAN_REDUCTION_INTO_WRAPPER(terlan_ndarray_v1_nan_product_into,
                           REDUCE_PRODUCT)
NAN_REDUCTION_INTO_WRAPPER(terlan_ndarray_v1_nan_mean_into, REDUCE_MEAN)
NAN_REDUCTION_INTO_WRAPPER(terlan_ndarray_v1_nan_minimum_into,
                           REDUCE_MINIMUM)
NAN_REDUCTION_INTO_WRAPPER(terlan_ndarray_v1_nan_maximum_into,
                           REDUCE_MAXIMUM)

TerlanNdArrayStatus terlan_ndarray_v1_nan_variance_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims,
    int64_t correction) {
  return nan_reduction_into(destination, array, axes, axis_count, keep_dims,
                            correction, REDUCE_VARIANCE);
}

TerlanNdArrayStatus terlan_ndarray_v1_nan_standard_deviation_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims,
    int64_t correction) {
  return nan_reduction_into(destination, array, axes, axis_count, keep_dims,
                            correction, REDUCE_STANDARD_DEVIATION);
}

static TerlanNdArrayStatus cumulative_array(
    TerlanNdArrayHandle array, int64_t axis, int product,
    int32_t requested_dtype,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle result = NULL;
  int64_t outer = 1;
  int64_t inner = 1;
  int64_t dimension;
  int64_t outer_index;
  const int32_t output_dtype = status == TERLAN_NDARRAY_OK
                                   ? (requested_dtype == 0
                                          ? accumulator_dtype(
                                                array->dtype,
                                                product != 0 ? REDUCE_PRODUCT
                                                             : REDUCE_SUM)
                                          : requested_dtype)
                                   : 0;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (axis < 0) {
    axis += array->rank;
  }
  if (axis < 0 || axis >= array->rank) {
    return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
  }
  for (dimension = 0; dimension < axis; ++dimension) {
    outer *= array->shape[dimension];
  }
  for (dimension = axis + 1; dimension < array->rank; ++dimension) {
    inner *= array->shape[dimension];
  }
  status = create_empty_array(array->shape, array->rank, output_dtype,
                              &result);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (outer_index = 0; outer_index < outer; ++outer_index) {
    int64_t inner_index;
    for (inner_index = 0; inner_index < inner; ++inner_index) {
      TerlanNdArrayScalar accumulator = {product != 0 ? 1.0L : 0.0L, 0.0L};
      int64_t axis_index;
      for (axis_index = 0; axis_index < array->shape[axis]; ++axis_index) {
        const int64_t logical_index =
            (outer_index * array->shape[axis] + axis_index) * inner +
            inner_index;
        TerlanNdArrayScalar value;
        status = dtype_load(array_const_element_pointer(array, logical_index),
                            array->dtype, &value);
        if (status != TERLAN_NDARRAY_OK) {
          goto fail;
        }
        if (product != 0) {
          const long double complex accumulated =
              (accumulator.real + accumulator.imag * I) *
              (value.real + value.imag * I);
          accumulator.real = creall(accumulated);
          accumulator.imag = cimagl(accumulated);
        } else {
          accumulator.real += value.real;
          accumulator.imag += value.imag;
        }
        status = dtype_store(array_element_pointer(result, logical_index),
                             output_dtype, accumulator);
        if (status != TERLAN_NDARRAY_OK) {
          status = status == TERLAN_NDARRAY_CAST_OVERFLOW
                       ? TERLAN_NDARRAY_ARITHMETIC_OVERFLOW
                       : status;
          goto fail;
        }
      }
    }
  }
  *out_array = result;
  return TERLAN_NDARRAY_OK;

fail:
  (void)terlan_ndarray_v1_delete(result);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_cumulative_sum(
    TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array) {
  return cumulative_array(array, axis, 0, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_cumulative_product(
    TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array) {
  return cumulative_array(array, axis, 1, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_cumulative_sum_as(
    TerlanNdArrayHandle array, int64_t axis, int32_t requested_dtype,
    TerlanNdArrayHandle *out_array) {
  return cumulative_array(array, axis, 0, requested_dtype, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_cumulative_product_as(
    TerlanNdArrayHandle array, int64_t axis, int32_t requested_dtype,
    TerlanNdArrayHandle *out_array) {
  return cumulative_array(array, axis, 1, requested_dtype, out_array);
}
