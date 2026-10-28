#include "terlan_ndarray_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct CovarianceInput {
  TerlanNdArrayHandle first;
  TerlanNdArrayHandle second;
  int32_t row_variables;
  int64_t first_variables;
  int64_t second_variables;
  int64_t observations;
} CovarianceInput;

static TerlanNdArrayStatus covariance_shape(
    TerlanNdArrayHandle array, int32_t row_variables, int64_t *out_variables,
    int64_t *out_observations) {
  if (array->rank > 2 || (row_variables != 0 && row_variables != 1)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (array->rank == 0) {
    *out_variables = 1;
    *out_observations = 1;
  } else if (array->rank == 1) {
    *out_variables = 1;
    *out_observations = array->shape[0];
  } else if (row_variables != 0 || array->shape[0] == 1) {
    *out_variables = array->shape[0];
    *out_observations = array->shape[1];
  } else {
    *out_variables = array->shape[1];
    *out_observations = array->shape[0];
  }
  return TERLAN_NDARRAY_OK;
}

static int64_t covariance_index(TerlanNdArrayHandle array,
                                int32_t row_variables, int64_t variable,
                                int64_t observation) {
  if (array->rank == 0) {
    return 0;
  }
  if (array->rank == 1) {
    return observation;
  }
  if (row_variables != 0 || array->shape[0] == 1) {
    return variable * array->shape[1] + observation;
  }
  return observation * array->shape[1] + variable;
}

static TerlanNdArrayStatus covariance_input_create(
    TerlanNdArrayHandle first, TerlanNdArrayHandle second,
    int32_t row_variables, CovarianceInput *out_input) {
  TerlanNdArrayStatus status = require_live(first);
  int64_t second_observations = 0;
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (second != NULL) {
    status = require_live(second);
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
  }
  out_input->first = first;
  out_input->second = second;
  out_input->row_variables = row_variables;
  out_input->second_variables = 0;
  status = covariance_shape(first, row_variables,
                            &out_input->first_variables,
                            &out_input->observations);
  if (status != TERLAN_NDARRAY_OK || out_input->first_variables == 0 ||
      second == NULL) {
    return status;
  }
  status = covariance_shape(second, row_variables,
                            &out_input->second_variables,
                            &second_observations);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  return second_observations == out_input->observations
             ? TERLAN_NDARRAY_OK
             : TERLAN_NDARRAY_SHAPE_MISMATCH;
}

static int64_t covariance_variable_count(const CovarianceInput *input) {
  return input->first_variables + input->second_variables;
}

static TerlanNdArrayStatus covariance_load(
    const CovarianceInput *input, int64_t variable, int64_t observation,
    TerlanNdArrayScalar *out_value) {
  TerlanNdArrayHandle array = input->first;
  if (variable >= input->first_variables) {
    array = input->second;
    variable -= input->first_variables;
  }
  return dtype_load(array_const_element_pointer(
                        array, covariance_index(array, input->row_variables,
                                                variable, observation)),
                    array->dtype, out_value);
}

static int32_t covariance_default_dtype(const CovarianceInput *input) {
  if (dtype_is_complex(input->first->dtype) ||
      (input->second != NULL && dtype_is_complex(input->second->dtype))) {
    return TERLAN_NDARRAY_DTYPE_COMPLEX128;
  }
  return TERLAN_NDARRAY_DTYPE_FLOAT64;
}

static TerlanNdArrayStatus covariance_result_dtype(
    const CovarianceInput *input, int32_t requested, int32_t *out_dtype) {
  if (requested == 0) {
    *out_dtype = covariance_default_dtype(input);
    return TERLAN_NDARRAY_OK;
  }
  if (!dtype_is_float(requested) && !dtype_is_complex(requested)) {
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  if ((dtype_is_complex(input->first->dtype) ||
       (input->second != NULL && dtype_is_complex(input->second->dtype))) &&
      !dtype_is_complex(requested)) {
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  *out_dtype = requested;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus validate_weight(TerlanNdArrayHandle weight,
                                           int64_t observations,
                                           int frequency) {
  int64_t index;
  TerlanNdArrayStatus status;
  if (weight == NULL) {
    return TERLAN_NDARRAY_OK;
  }
  status = require_live(weight);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (weight->rank != 1 || weight->shape[0] != observations ||
      dtype_is_complex(weight->dtype)) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  for (index = 0; index < observations; ++index) {
    TerlanNdArrayScalar value;
    status = dtype_load(array_const_element_pointer(weight, index),
                        weight->dtype, &value);
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
    if (!isfinite(value.real) || value.real < 0.0L) {
      return TERLAN_NDARRAY_DOMAIN_ERROR;
    }
    if (frequency != 0 && floorl(value.real) != value.real) {
      return TERLAN_NDARRAY_INVALID_ARGUMENT;
    }
  }
  return TERLAN_NDARRAY_OK;
}

static long double covariance_weight(TerlanNdArrayHandle frequency_weights,
                                     TerlanNdArrayHandle analytic_weights,
                                     int64_t observation,
                                     long double *out_analytic) {
  TerlanNdArrayScalar frequency = {1.0L, 0.0L};
  TerlanNdArrayScalar analytic = {1.0L, 0.0L};
  if (frequency_weights != NULL) {
    (void)dtype_load(array_const_element_pointer(frequency_weights,
                                                 observation),
                     frequency_weights->dtype, &frequency);
  }
  if (analytic_weights != NULL) {
    (void)dtype_load(array_const_element_pointer(analytic_weights,
                                                 observation),
                     analytic_weights->dtype, &analytic);
  }
  *out_analytic = analytic.real;
  return frequency.real * analytic.real;
}

static TerlanNdArrayStatus covariance_means(
    const CovarianceInput *input, TerlanNdArrayHandle frequency_weights,
    TerlanNdArrayHandle analytic_weights, TerlanNdArrayScalar *means,
    long double *out_weight_sum, long double *out_weight_analytic_sum) {
  const int64_t variables = covariance_variable_count(input);
  int64_t observation;
  for (observation = 0; observation < input->observations; ++observation) {
    long double analytic;
    const long double weight = covariance_weight(
        frequency_weights, analytic_weights, observation, &analytic);
    int64_t variable;
    *out_weight_sum += weight;
    *out_weight_analytic_sum += weight * analytic;
    for (variable = 0; variable < variables; ++variable) {
      TerlanNdArrayScalar value;
      TerlanNdArrayStatus status =
          covariance_load(input, variable, observation, &value);
      if (status != TERLAN_NDARRAY_OK) {
        return status;
      }
      means[variable].real += value.real * weight;
      means[variable].imag += value.imag * weight;
    }
  }
  if (*out_weight_sum == 0.0L &&
      (frequency_weights != NULL || analytic_weights != NULL)) {
    return TERLAN_NDARRAY_DIVIDE_BY_ZERO;
  }
  for (int64_t variable = 0; variable < variables; ++variable) {
    means[variable].real /= *out_weight_sum;
    means[variable].imag /= *out_weight_sum;
  }
  return TERLAN_NDARRAY_OK;
}

static long double covariance_normalization(
    const CovarianceInput *input, TerlanNdArrayHandle frequency_weights,
    TerlanNdArrayHandle analytic_weights, int64_t correction,
    long double weight_sum, long double weight_analytic_sum) {
  if (frequency_weights == NULL && analytic_weights == NULL) {
    return (long double)input->observations - correction;
  }
  if (correction == 0) {
    return weight_sum;
  }
  if (analytic_weights == NULL) {
    return weight_sum - correction;
  }
  return weight_sum - correction * weight_analytic_sum / weight_sum;
}

static TerlanNdArrayStatus covariance_allocate(int64_t variables,
                                               int32_t dtype,
                                               TerlanNdArrayHandle *out_array) {
  const int64_t shape[2] = {variables, variables};
  if (variables == 1) {
    return create_empty_array(NULL, 0, dtype, out_array);
  }
  return create_empty_array(shape, 2, dtype, out_array);
}

static TerlanNdArrayStatus covariance_values(
    const CovarianceInput *input, TerlanNdArrayHandle frequency_weights,
    TerlanNdArrayHandle analytic_weights, const TerlanNdArrayScalar *means,
    long double normalization, TerlanNdArrayHandle result) {
  const int64_t variables = covariance_variable_count(input);
  int64_t row;
  for (row = 0; row < variables; ++row) {
    int64_t column;
    for (column = 0; column < variables; ++column) {
      TerlanNdArrayScalar sum = {0.0L, 0.0L};
      int64_t observation;
      for (observation = 0; observation < input->observations; ++observation) {
        TerlanNdArrayScalar lhs;
        TerlanNdArrayScalar rhs;
        TerlanNdArrayScalar product;
        long double analytic;
        const long double weight = covariance_weight(
            frequency_weights, analytic_weights, observation, &analytic);
        (void)analytic;
        (void)covariance_load(input, row, observation, &lhs);
        (void)covariance_load(input, column, observation, &rhs);
        lhs.real -= means[row].real;
        lhs.imag -= means[row].imag;
        rhs.real -= means[column].real;
        rhs.imag = means[column].imag - rhs.imag;
        product = scalar_multiply(lhs, rhs);
        sum.real += product.real * weight;
        sum.imag += product.imag * weight;
      }
      sum.real /= normalization;
      sum.imag /= normalization;
      if (!dtype_is_complex(result->dtype)) {
        sum.imag = 0.0L;
      }
      {
        TerlanNdArrayStatus status = dtype_store(
            array_element_pointer(result, row * variables + column),
            result->dtype, sum);
        if (status != TERLAN_NDARRAY_OK) {
          return status;
        }
      }
    }
  }
  return TERLAN_NDARRAY_OK;
}

static long double clipped_correlation_component(long double value) {
  if (value < -1.0L) {
    return -1.0L;
  }
  if (value > 1.0L) {
    return 1.0L;
  }
  return value;
}

static TerlanNdArrayStatus normalize_correlation(TerlanNdArrayHandle result,
                                                 int64_t variables) {
  long double *deviations = checked_malloc((size_t)variables *
                                            sizeof(*deviations));
  if (deviations == NULL && variables > 0) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  for (int64_t row = 0; row < variables; ++row) {
    TerlanNdArrayScalar variance;
    (void)dtype_load(array_const_element_pointer(result,
                                                  row * variables + row),
                     result->dtype, &variance);
    deviations[row] = sqrtl(variance.real);
  }
  for (int64_t row = 0; row < variables; ++row) {
    for (int64_t column = 0; column < variables; ++column) {
      TerlanNdArrayScalar value;
      const long double scale = deviations[row] * deviations[column];
      (void)dtype_load(array_const_element_pointer(
                           result, row * variables + column),
                       result->dtype, &value);
      value.real = clipped_correlation_component(value.real / scale);
      value.imag = clipped_correlation_component(value.imag / scale);
      (void)dtype_store(array_element_pointer(result,
                                               row * variables + column),
                        result->dtype, value);
    }
  }
  free(deviations);
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus covariance_compute(
    TerlanNdArrayHandle array, TerlanNdArrayHandle other,
    int32_t row_variables, int64_t correction,
    TerlanNdArrayHandle frequency_weights,
    TerlanNdArrayHandle analytic_weights, int32_t dtype, int correlation,
    TerlanNdArrayHandle *out_array) {
  CovarianceInput input;
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayScalar *means = NULL;
  TerlanNdArrayStatus status;
  int32_t result_dtype;
  int64_t variables;
  long double weight_sum = 0.0L;
  long double weight_analytic_sum = 0.0L;
  long double normalization;
  if (out_array == NULL || correction < 0) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = covariance_input_create(array, other, row_variables, &input);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  status = covariance_result_dtype(&input, dtype, &result_dtype);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (input.first_variables == 0) {
    const int64_t shape[2] = {0, 0};
    return create_empty_array(shape, 2, result_dtype, out_array);
  }
  status = validate_weight(frequency_weights, input.observations, 1);
  if (status == TERLAN_NDARRAY_OK) {
    status = validate_weight(analytic_weights, input.observations, 0);
  }
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  variables = covariance_variable_count(&input);
  means = checked_calloc((size_t)variables, sizeof(*means));
  if (means == NULL && variables > 0) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  status = covariance_means(&input, frequency_weights, analytic_weights,
                            means, &weight_sum, &weight_analytic_sum);
  if (status != TERLAN_NDARRAY_OK) {
    goto cleanup;
  }
  normalization = covariance_normalization(
      &input, frequency_weights, analytic_weights, correction, weight_sum,
      weight_analytic_sum);
  if (normalization < 0.0L) {
    normalization = 0.0L;
  }
  status = covariance_allocate(variables, result_dtype, &result);
  if (status == TERLAN_NDARRAY_OK) {
    status = covariance_values(&input, frequency_weights, analytic_weights,
                               means, normalization, result);
  }
  if (status == TERLAN_NDARRAY_OK && correlation != 0) {
    if (variables == 1) {
      TerlanNdArrayScalar value;
      (void)dtype_load(array_const_element_pointer(result, 0), result->dtype,
                       &value);
      value.real = value.real / value.real;
      value.imag = 0.0L;
      status = dtype_store(array_element_pointer(result, 0), result->dtype,
                           value);
    } else {
      status = normalize_correlation(result, variables);
    }
  }
  if (status == TERLAN_NDARRAY_OK) {
    *out_array = result;
    result = NULL;
  }

cleanup:
  if (result != NULL) {
    (void)terlan_ndarray_v1_delete(result);
  }
  free(means);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_covariance(
    TerlanNdArrayHandle array, int32_t row_variables, int64_t correction,
    TerlanNdArrayHandle *out_array) {
  return covariance_compute(array, NULL, row_variables, correction, NULL,
                            NULL, 0, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_covariance_typed(
    TerlanNdArrayHandle array, int32_t row_variables, int64_t correction,
    int32_t dtype, TerlanNdArrayHandle *out_array) {
  return covariance_compute(array, NULL, row_variables, correction, NULL,
                            NULL, dtype, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_covariance_pair(
    TerlanNdArrayHandle array, TerlanNdArrayHandle other,
    int32_t row_variables, int64_t correction, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  return covariance_compute(array, other, row_variables, correction, NULL,
                            NULL, dtype, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_covariance_frequency_weighted(
    TerlanNdArrayHandle array, TerlanNdArrayHandle frequency_weights,
    int32_t row_variables, int64_t correction, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  return covariance_compute(array, NULL, row_variables, correction,
                            frequency_weights, NULL, dtype, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_covariance_analytic_weighted(
    TerlanNdArrayHandle array, TerlanNdArrayHandle analytic_weights,
    int32_t row_variables, int64_t correction, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  return covariance_compute(array, NULL, row_variables, correction, NULL,
                            analytic_weights, dtype, 0, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_covariance_weighted(
    TerlanNdArrayHandle array, TerlanNdArrayHandle frequency_weights,
    TerlanNdArrayHandle analytic_weights, int32_t row_variables,
    int64_t correction, int32_t dtype, TerlanNdArrayHandle *out_array) {
  return covariance_compute(array, NULL, row_variables, correction,
                            frequency_weights, analytic_weights, dtype, 0,
                            out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_covariance_pair_weighted(
    TerlanNdArrayHandle array, TerlanNdArrayHandle other,
    TerlanNdArrayHandle frequency_weights,
    TerlanNdArrayHandle analytic_weights, int32_t row_variables,
    int64_t correction, int32_t dtype, TerlanNdArrayHandle *out_array) {
  return covariance_compute(array, other, row_variables, correction,
                            frequency_weights, analytic_weights, dtype, 0,
                            out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_correlation(
    TerlanNdArrayHandle array, int32_t row_variables,
    TerlanNdArrayHandle *out_array) {
  return covariance_compute(array, NULL, row_variables, 1, NULL, NULL, 0, 1,
                            out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_correlation_typed(
    TerlanNdArrayHandle array, int32_t row_variables, int32_t dtype,
    TerlanNdArrayHandle *out_array) {
  return covariance_compute(array, NULL, row_variables, 1, NULL, NULL, dtype,
                            1, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_correlation_pair(
    TerlanNdArrayHandle array, TerlanNdArrayHandle other,
    int32_t row_variables, int32_t dtype, TerlanNdArrayHandle *out_array) {
  return covariance_compute(array, other, row_variables, 1, NULL, NULL, dtype,
                            1, out_array);
}
