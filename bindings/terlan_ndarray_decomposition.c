#include "terlan_ndarray_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum DecompositionOperation {
  DECOMPOSE_FREXP,
  DECOMPOSE_MODF
} DecompositionOperation;

static void delete_outputs(TerlanNdArrayHandle first,
                           TerlanNdArrayHandle second) {
  if (second != NULL) {
    (void)terlan_ndarray_v1_delete(second);
  }
  if (first != NULL) {
    (void)terlan_ndarray_v1_delete(first);
  }
}

static TerlanNdArrayStatus publish_outputs(TerlanNdArrayHandle first,
                                           TerlanNdArrayHandle second,
                                           uint64_t **out_values,
                                           size_t *out_length) {
  uint64_t *values = checked_malloc(2 * sizeof(*values));
  if (values == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  values[0] = (uint64_t)(uintptr_t)first;
  values[1] = (uint64_t)(uintptr_t)second;
  *out_values = values;
  *out_length = 2;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus decompose_array(
    TerlanNdArrayHandle array, DecompositionOperation operation,
    uint64_t **out_values, size_t *out_length) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle first = NULL;
  TerlanNdArrayHandle second = NULL;
  int64_t index;
  if (status != TERLAN_NDARRAY_OK || out_values == NULL ||
      out_length == NULL || dtype_is_complex(array->dtype)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_values = NULL;
  *out_length = 0;
  status = create_empty_array(array->shape, array->rank,
                              TERLAN_NDARRAY_DTYPE_FLOAT64, &first);
  if (status == TERLAN_NDARRAY_OK) {
    status = create_empty_array(
        array->shape, array->rank,
        operation == DECOMPOSE_FREXP ? TERLAN_NDARRAY_DTYPE_INT64
                                     : TERLAN_NDARRAY_DTYPE_FLOAT64,
        &second);
  }
  if (status != TERLAN_NDARRAY_OK) {
    delete_outputs(first, second);
    return status;
  }
  for (index = 0; index < array->numel; ++index) {
    TerlanNdArrayScalar input;
    TerlanNdArrayScalar first_value = {0.0L, 0.0L};
    TerlanNdArrayScalar second_value = {0.0L, 0.0L};
    (void)dtype_load(array_const_element_pointer(array, index), array->dtype,
                     &input);
    if (operation == DECOMPOSE_FREXP) {
      int exponent = 0;
      first_value.real = frexpl(input.real, &exponent);
      second_value.real = exponent;
    } else {
      first_value.real = modfl(input.real, &second_value.real);
    }
    status = dtype_store(array_element_pointer(first, index), first->dtype,
                         first_value);
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_store(array_element_pointer(second, index), second->dtype,
                           second_value);
    }
    if (status != TERLAN_NDARRAY_OK) {
      delete_outputs(first, second);
      return status;
    }
  }
  status = publish_outputs(first, second, out_values, out_length);
  if (status != TERLAN_NDARRAY_OK) {
    delete_outputs(first, second);
  }
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_frexp_all(
    TerlanNdArrayHandle array, uint64_t **out_values, size_t *out_length) {
  return decompose_array(array, DECOMPOSE_FREXP, out_values, out_length);
}

TerlanNdArrayStatus terlan_ndarray_v1_modf_all(
    TerlanNdArrayHandle array, uint64_t **out_values, size_t *out_length) {
  return decompose_array(array, DECOMPOSE_MODF, out_values, out_length);
}
