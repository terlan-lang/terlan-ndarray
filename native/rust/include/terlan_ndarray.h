#ifndef TERLAN_NDARRAY_H
#define TERLAN_NDARRAY_H

#include <stddef.h>
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
  TERLAN_NDARRAY_INVALID_BOOL = 105,
  TERLAN_NDARRAY_DTYPE_MISMATCH = 106,
  TERLAN_NDARRAY_DISPOSED = 107
};

enum {
  TERLAN_NDARRAY_DTYPE_BOOL = 1,
  TERLAN_NDARRAY_DTYPE_INT64 = 2,
  TERLAN_NDARRAY_DTYPE_FLOAT64 = 3
};

enum { TERLAN_NDARRAY_DEVICE_CPU = 1 };

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
TerlanNdArrayStatus terlan_ndarray_v1_strides(TerlanNdArrayHandle array,
                                               int64_t **out_strides);
TerlanNdArrayStatus terlan_ndarray_v1_numel(TerlanNdArrayHandle array,
                                             int64_t *out_numel);
TerlanNdArrayStatus terlan_ndarray_v1_byte_count(TerlanNdArrayHandle array,
                                                  int64_t *out_byte_count);
TerlanNdArrayStatus terlan_ndarray_v1_dtype(TerlanNdArrayHandle array,
                                             int32_t *out_dtype);
TerlanNdArrayStatus terlan_ndarray_v1_is_contiguous(
    TerlanNdArrayHandle array, int32_t *out_contiguous);
TerlanNdArrayStatus terlan_ndarray_v1_device_type(
    TerlanNdArrayHandle array, int32_t *out_device_type);
TerlanNdArrayStatus terlan_ndarray_v1_to_bools(
    TerlanNdArrayHandle array, uint8_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_to_ints(
    TerlanNdArrayHandle array, int64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_to_floats(
    TerlanNdArrayHandle array, double **out_values, size_t *out_length);
void terlan_ndarray_v1_delete_bool_values(uint8_t *values);
void terlan_ndarray_v1_delete_int_values(int64_t *values);
void terlan_ndarray_v1_delete_float_values(double *values);
TerlanNdArrayStatus terlan_ndarray_v1_delete(TerlanNdArrayHandle array);
int64_t terlan_ndarray_v1_live_count(void);

#ifdef __cplusplus
}
#endif

#endif
