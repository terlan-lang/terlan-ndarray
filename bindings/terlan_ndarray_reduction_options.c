#include "terlan_ndarray_internal.h"

typedef TerlanNdArrayStatus (*ReductionOperationFunction)(
    TerlanNdArrayHandle, const int64_t *, int64_t, int32_t,
    TerlanNdArrayHandle *);

static TerlanNdArrayStatus reduction_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims,
    ReductionOperationFunction operation) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status =
      operation(array, axes, axis_count, keep_dims, &result);
  return array_finish_into(destination, status, result,
                           TERLAN_NDARRAY_CAST_UNSAFE);
}

#define REDUCTION_INTO_WRAPPER(name, operation)                              \
  TerlanNdArrayStatus name(                                                  \
      TerlanNdArrayHandle destination, TerlanNdArrayHandle array,            \
      const int64_t *axes, int64_t axis_count, int32_t keep_dims) {          \
    return reduction_into(destination, array, axes, axis_count, keep_dims,    \
                          operation);                                        \
  }

REDUCTION_INTO_WRAPPER(terlan_ndarray_v1_sum_into, terlan_ndarray_v1_sum)
REDUCTION_INTO_WRAPPER(terlan_ndarray_v1_product_into,
                       terlan_ndarray_v1_product)
REDUCTION_INTO_WRAPPER(terlan_ndarray_v1_minimum_into,
                       terlan_ndarray_v1_minimum)
REDUCTION_INTO_WRAPPER(terlan_ndarray_v1_maximum_into,
                       terlan_ndarray_v1_maximum)

static TerlanNdArrayStatus reduce_plain(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, ReductionOperation operation, int32_t skip_nan,
    TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_reduce_configured(
      array, axes, axis_count, keep_dims, 0, operation, skip_nan, NULL, NULL, 0,
      out_array);
}

static TerlanNdArrayStatus reduce_initial(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask,
    TerlanNdArrayHandle initial, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, ReductionOperation operation,
    TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_reduce_configured(
      array, axes, axis_count, keep_dims, 0, operation, 0, mask, initial, 0,
      out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_sum_where(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_reduce_configured(
      array, axes, axis_count, keep_dims, 0, REDUCE_SUM, 0, mask, NULL, 0,
      out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_sum_initial(
    TerlanNdArrayHandle array, TerlanNdArrayHandle initial,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims,
    TerlanNdArrayHandle *out_array) {
  return reduce_initial(array, NULL, initial, axes, axis_count, keep_dims,
                        REDUCE_SUM, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_sum_where_initial(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask,
    TerlanNdArrayHandle initial, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return reduce_initial(array, mask, initial, axes, axis_count, keep_dims,
                        REDUCE_SUM, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_minimum_initial(
    TerlanNdArrayHandle array, TerlanNdArrayHandle initial,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims,
    TerlanNdArrayHandle *out_array) {
  return reduce_initial(array, NULL, initial, axes, axis_count, keep_dims,
                        REDUCE_MINIMUM, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_minimum_where(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_reduce_configured(
      array, axes, axis_count, keep_dims, 0, REDUCE_MINIMUM, 0, mask, NULL, 0,
      out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_maximum_where(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_reduce_configured(
      array, axes, axis_count, keep_dims, 0, REDUCE_MAXIMUM, 0, mask, NULL, 0,
      out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_maximum_initial(
    TerlanNdArrayHandle array, TerlanNdArrayHandle initial,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims,
    TerlanNdArrayHandle *out_array) {
  return reduce_initial(array, NULL, initial, axes, axis_count, keep_dims,
                        REDUCE_MAXIMUM, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_minimum_where_initial(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask,
    TerlanNdArrayHandle initial, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return reduce_initial(array, mask, initial, axes, axis_count, keep_dims,
                        REDUCE_MINIMUM, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_maximum_where_initial(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask,
    TerlanNdArrayHandle initial, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return reduce_initial(array, mask, initial, axes, axis_count, keep_dims,
                        REDUCE_MAXIMUM, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_fmin_reduce(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return reduce_plain(array, axes, axis_count, keep_dims, REDUCE_MINIMUM, 1,
                      out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_fmax_reduce(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return reduce_plain(array, axes, axis_count, keep_dims, REDUCE_MAXIMUM, 1,
                      out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_bitwise_and_reduce(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return reduce_plain(array, axes, axis_count, keep_dims, REDUCE_BITWISE_AND, 0,
                      out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_bitwise_or_reduce(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return reduce_plain(array, axes, axis_count, keep_dims, REDUCE_BITWISE_OR, 0,
                      out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_bitwise_xor_reduce(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array) {
  return reduce_plain(array, axes, axis_count, keep_dims, REDUCE_BITWISE_XOR, 0,
                      out_array);
}
