#ifndef TERLAN_NDARRAY_H
#define TERLAN_NDARRAY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t TerlanNdArrayStatus;
typedef struct TerlanNdArray TerlanNdArray;
typedef TerlanNdArray *TerlanNdArrayHandle;

enum {
  TERLAN_NDARRAY_OK = 0,
  TERLAN_NDARRAY_INVALID_ARGUMENT = 100,
  TERLAN_NDARRAY_NEGATIVE_DIMENSION = 101,
  TERLAN_NDARRAY_ELEMENT_COUNT_MISMATCH = 102,
  TERLAN_NDARRAY_SIZE_OVERFLOW = 103,
  TERLAN_NDARRAY_ALLOCATION_FAILED = 104,
  TERLAN_NDARRAY_INVALID_BOOL = 105
};

enum {
  TERLAN_NDARRAY_DTYPE_BOOL = 1,
  TERLAN_NDARRAY_DTYPE_INT64 = 2,
  TERLAN_NDARRAY_DTYPE_FLOAT64 = 3
};

TerlanNdArrayStatus terlan_ndarray_v1_from_bools(
    const uint8_t *values, int64_t value_count, const int64_t *shape,
    int64_t rank, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_from_ints(
    const int64_t *values, int64_t value_count, const int64_t *shape,
    int64_t rank, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_from_floats(
    const double *values, int64_t value_count, const int64_t *shape,
    int64_t rank, TerlanNdArrayHandle *out_array);

TerlanNdArrayStatus terlan_ndarray_v1_rank(TerlanNdArrayHandle array,
                                            int64_t *out_rank);
TerlanNdArrayStatus terlan_ndarray_v1_shape(TerlanNdArrayHandle array,
                                             int64_t **out_shape);
TerlanNdArrayStatus terlan_ndarray_v1_numel(TerlanNdArrayHandle array,
                                             int64_t *out_numel);
TerlanNdArrayStatus terlan_ndarray_v1_dtype(TerlanNdArrayHandle array,
                                             int32_t *out_dtype);
TerlanNdArrayStatus terlan_ndarray_v1_is_contiguous(
    TerlanNdArrayHandle array, int32_t *out_contiguous);
TerlanNdArrayStatus terlan_ndarray_v1_delete(TerlanNdArrayHandle array);
int64_t terlan_ndarray_v1_live_count(void);

#ifdef __cplusplus
}
#endif

#endif

