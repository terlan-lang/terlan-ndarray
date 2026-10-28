#include "terlan_ndarray_internal.h"

#include <stdint.h>
#include <string.h>

static TerlanNdArrayStatus array_write_exact_with_policy(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle source,
    int32_t casting, int checked_values) {
  TerlanNdArrayStatus status = require_live(destination);
  TerlanNdArrayHandle staged = NULL;
  int64_t index;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(source);
  }
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (destination->writable == 0) {
    return TERLAN_NDARRAY_READ_ONLY;
  }
  if (!array_same_shape(destination, source)) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  if (!dtype_can_cast(source->dtype, destination->dtype, casting)) {
    return TERLAN_NDARRAY_DTYPE_MISMATCH;
  }
  status = create_empty_array(destination->shape, destination->rank,
                              destination->dtype, &staged);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (index = 0; index < source->numel; ++index) {
    TerlanNdArrayScalar value;
    status = dtype_load(array_const_element_pointer(source, index),
                        source->dtype, &value);
    if (status == TERLAN_NDARRAY_OK) {
      status = checked_values != 0
                   ? dtype_store(array_element_pointer(staged, index),
                                 staged->dtype, value)
                   : dtype_store_cast(array_element_pointer(staged, index),
                                      staged->dtype, value, casting);
    }
    if (status != TERLAN_NDARRAY_OK) {
      (void)terlan_ndarray_v1_delete(staged);
      return status;
    }
  }
  for (index = 0; index < destination->numel; ++index) {
    memcpy(array_element_pointer(destination, index),
           array_const_element_pointer(staged, index),
           dtype_size(destination->dtype));
  }
  (void)terlan_ndarray_v1_delete(staged);
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus array_write_exact(TerlanNdArrayHandle destination,
                                      TerlanNdArrayHandle source,
                                      int32_t casting) {
  return array_write_exact_with_policy(destination, source, casting, 0);
}

TerlanNdArrayStatus array_finish_into(TerlanNdArrayHandle destination,
                                      TerlanNdArrayStatus operation_status,
                                      TerlanNdArrayHandle result,
                                      int32_t casting) {
  TerlanNdArrayStatus status = operation_status;
  if (status == TERLAN_NDARRAY_OK) {
    status = array_write_exact(destination, result, casting);
  }
  if (result != NULL) {
    (void)terlan_ndarray_v1_delete(result);
  }
  return status;
}

TerlanNdArrayStatus array_finish_into_broadcast(
    TerlanNdArrayHandle destination, TerlanNdArrayStatus operation_status,
    TerlanNdArrayHandle result, int32_t casting) {
  TerlanNdArrayStatus status = operation_status;
  TerlanNdArrayHandle broadcast = NULL;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(destination);
  }
  if (status == TERLAN_NDARRAY_OK && destination->rank < result->rank) {
    status = TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = terlan_ndarray_v1_broadcast_to(
        result, destination->shape, destination->rank, &broadcast);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = array_write_exact(destination, broadcast, casting);
  }
  if (broadcast != NULL) {
    (void)terlan_ndarray_v1_delete(broadcast);
  }
  if (result != NULL) {
    (void)terlan_ndarray_v1_delete(result);
  }
  return status;
}

TerlanNdArrayStatus array_finish_into_checked(
    TerlanNdArrayHandle destination, TerlanNdArrayStatus operation_status,
    TerlanNdArrayHandle result, int32_t casting) {
  TerlanNdArrayStatus status = operation_status;
  if (status == TERLAN_NDARRAY_OK) {
    status = array_write_exact_with_policy(destination, result, casting, 1);
  }
  if (result != NULL) {
    (void)terlan_ndarray_v1_delete(result);
  }
  return status;
}
