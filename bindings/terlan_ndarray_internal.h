#ifndef TERLAN_NDARRAY_INTERNAL_H
#define TERLAN_NDARRAY_INTERNAL_H

#include "include/terlan_ndarray.h"
#if defined(__has_include)
#if __has_include("include/dlpack/dlpack.h")
#include "include/dlpack/dlpack.h"
#else
#include "dlpack.h"
#endif
#else
#include "include/dlpack/dlpack.h"
#endif

#include <stddef.h>
#include <stdatomic.h>
#include <stdint.h>

typedef enum TerlanNdArrayState {
  TERLAN_NDARRAY_STATE_LIVE = 1,
  TERLAN_NDARRAY_STATE_DISPOSED = 2
} TerlanNdArrayState;

typedef struct TerlanNdArrayStorage {
  _Atomic int64_t reference_count;
  int64_t byte_count;
  void *data;
  void *external_context;
  void (*external_release)(void *context);
  int32_t (*external_flush)(void *context);
} TerlanNdArrayStorage;

typedef struct TerlanNdArrayScalar {
  long double real;
  long double imag;
} TerlanNdArrayScalar;

typedef struct TerlanNdArrayReductionPlan {
  int64_t output_rank;
  int64_t output_numel;
  int64_t *output_shape;
  uint8_t *reduced;
} TerlanNdArrayReductionPlan;

typedef struct TerlanNdArrayAxisLayout {
  int64_t axis;
  int64_t outer;
  int64_t inner;
} TerlanNdArrayAxisLayout;

typedef enum TerlanNdArrayOrder {
  TERLAN_NDARRAY_ORDER_C = 0,
  TERLAN_NDARRAY_ORDER_FORTRAN = 1,
  TERLAN_NDARRAY_ORDER_KEEP = 2,
  TERLAN_NDARRAY_ORDER_ANY = 3
} TerlanNdArrayOrder;

typedef enum TerlanNdArrayCasting {
  TERLAN_NDARRAY_CAST_NO = 0,
  TERLAN_NDARRAY_CAST_EQUIVALENT = 1,
  TERLAN_NDARRAY_CAST_SAFE = 2,
  TERLAN_NDARRAY_CAST_SAME_KIND = 3,
  TERLAN_NDARRAY_CAST_UNSAFE = 4
} TerlanNdArrayCasting;

typedef enum TerlanNdArrayIntegerBinaryOperation {
  TERLAN_NDARRAY_INTEGER_ADD,
  TERLAN_NDARRAY_INTEGER_SUBTRACT,
  TERLAN_NDARRAY_INTEGER_MULTIPLY,
  TERLAN_NDARRAY_INTEGER_POWER
} TerlanNdArrayIntegerBinaryOperation;

typedef enum ReductionOperation {
  REDUCE_SUM,
  REDUCE_PRODUCT,
  REDUCE_MEAN,
  REDUCE_MINIMUM,
  REDUCE_MAXIMUM,
  REDUCE_ANY,
  REDUCE_ALL,
  REDUCE_LOGICAL_XOR,
  REDUCE_BITWISE_AND,
  REDUCE_BITWISE_OR,
  REDUCE_BITWISE_XOR,
  REDUCE_ARGMIN,
  REDUCE_ARGMAX,
  REDUCE_COUNT_NONZERO,
  REDUCE_VARIANCE,
  REDUCE_STANDARD_DEVIATION
} ReductionOperation;

struct TerlanNdArray {
  int32_t dtype;
  int32_t device_type;
  int64_t rank;
  int64_t numel;
  int64_t byte_count;
  int64_t *shape;
  int64_t *strides;
  int64_t storage_offset;
  int32_t writable;
  TerlanNdArrayStorage *storage;
  void *data;
  TerlanNdArrayState state;
};

TerlanNdArrayStatus terlan_ndarray_v1_to_dlpack(
    TerlanNdArrayHandle array, DLManagedTensorVersioned **out_tensor);
TerlanNdArrayStatus terlan_ndarray_v1_from_dlpack(
    DLManagedTensorVersioned *tensor, TerlanNdArrayHandle *out_array);
int64_t terlan_ndarray_v1_dlpack_deleter_count(void);

void *checked_malloc(size_t size);
void *checked_calloc(size_t count, size_t size);
TerlanNdArrayStatus array_checked_layout(
    const int64_t *shape, int64_t rank, size_t element_size,
    int64_t *out_numel, int64_t *out_byte_count, int64_t *strides);
TerlanNdArrayStatus require_live(TerlanNdArrayHandle array);
size_t dtype_size(int32_t dtype);
int dtype_is_bool(int32_t dtype);
int dtype_is_integer(int32_t dtype);
int dtype_is_signed(int32_t dtype);
int dtype_is_float(int32_t dtype);
int dtype_is_complex(int32_t dtype);
uint32_t dtype_integer_bits(int32_t dtype);
uint64_t dtype_truncate_integer(uint64_t value, int32_t dtype);
int64_t dtype_signed_integer(uint64_t value, int32_t dtype);
uint64_t scalar_unsigned_gcd(uint64_t lhs, uint64_t rhs);
uint64_t scalar_integer_magnitude(long double value);
void scalar_complex_power(TerlanNdArrayScalar lhs, TerlanNdArrayScalar rhs,
                          TerlanNdArrayScalar *out_value);
TerlanNdArrayStatus scalar_wrapping_integer_binary(
    TerlanNdArrayIntegerBinaryOperation operation, long double lhs,
    long double rhs, int32_t dtype, TerlanNdArrayScalar *out_value);
void scalar_floating_floor_divmod(long double lhs, long double rhs,
                                  long double *out_quotient,
                                  long double *out_remainder);
int32_t dtype_promote(int32_t lhs, int32_t rhs);
int32_t dtype_inexact_or_float64(int32_t dtype);
TerlanNdArrayStatus dtype_load(const void *pointer, int32_t dtype,
                               TerlanNdArrayScalar *out_value);
float dtype_half_to_float(uint16_t half);
double dtype_half_to_double(uint16_t half);
uint16_t dtype_float_to_half(float value);
uint16_t dtype_double_to_half(double value);
TerlanNdArrayStatus dtype_convert(void *destination, int32_t destination_dtype,
                                  const void *source, int32_t source_dtype,
                                  int32_t casting);
TerlanNdArrayStatus dtype_store(void *pointer, int32_t dtype,
                                TerlanNdArrayScalar value);
int dtype_can_cast(int32_t source, int32_t destination, int32_t casting);
TerlanNdArrayStatus dtype_store_cast(void *pointer, int32_t dtype,
                                     TerlanNdArrayScalar value,
                                     int32_t casting);
TerlanNdArrayStatus array_load_int64(TerlanNdArrayHandle array,
                                     int64_t logical_index,
                                     int64_t *out_value);
int scalar_order_compare(TerlanNdArrayScalar lhs, TerlanNdArrayScalar rhs);
TerlanNdArrayScalar scalar_multiply(TerlanNdArrayScalar lhs,
                                    TerlanNdArrayScalar rhs);
TerlanNdArrayScalar scalar_divide(TerlanNdArrayScalar lhs,
                                  TerlanNdArrayScalar rhs);
TerlanNdArrayScalar scalar_add(TerlanNdArrayScalar lhs,
                               TerlanNdArrayScalar rhs);
TerlanNdArrayScalar scalar_subtract(TerlanNdArrayScalar lhs,
                                    TerlanNdArrayScalar rhs);
TerlanNdArrayScalar scalar_scale(TerlanNdArrayScalar value,
                                 long double scale);
int array_is_contiguous(TerlanNdArrayHandle array);
int array_is_fortran_contiguous(TerlanNdArrayHandle array);
TerlanNdArrayStatus array_normalize_order(TerlanNdArrayHandle array,
                                          int32_t order, int allow_keep,
                                          int32_t *out_order);
TerlanNdArrayStatus array_order_axes(TerlanNdArrayHandle array, int32_t order,
                                     int64_t *out_axes);
int64_t array_order_logical_index(TerlanNdArrayHandle array,
                                  const int64_t *axes,
                                  int64_t ordered_index);
int array_order_is_contiguous(TerlanNdArrayHandle array, int32_t order);
TerlanNdArrayStatus array_ordered_strides(TerlanNdArrayHandle source,
                                          int32_t order,
                                          int64_t *out_strides);
void *array_element_pointer(TerlanNdArrayHandle array, int64_t logical_index);
const void *array_const_element_pointer(TerlanNdArrayHandle array,
                                        int64_t logical_index);
int array_broadcast_compatible(TerlanNdArrayHandle source,
                               const int64_t *destination_shape,
                               int64_t destination_rank);
int64_t array_broadcast_index(TerlanNdArrayHandle source,
                              const int64_t *destination_shape,
                              int64_t destination_rank,
                              int64_t destination_index);
TerlanNdArrayStatus array_coordinate_targets(
    TerlanNdArrayHandle array, TerlanNdArrayHandle coordinates,
    int64_t **out_targets, int64_t *out_count);
int array_same_shape(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs);
TerlanNdArrayStatus array_axis_layout(
    TerlanNdArrayHandle array, int64_t axis,
    TerlanNdArrayAxisLayout *out_layout);
TerlanNdArrayStatus array_write_exact(TerlanNdArrayHandle destination,
                                      TerlanNdArrayHandle source,
                                      int32_t casting);
TerlanNdArrayStatus array_finish_into(TerlanNdArrayHandle destination,
                                      TerlanNdArrayStatus operation_status,
                                      TerlanNdArrayHandle result,
                                      int32_t casting);
TerlanNdArrayStatus array_finish_into_broadcast(
    TerlanNdArrayHandle destination, TerlanNdArrayStatus operation_status,
    TerlanNdArrayHandle result, int32_t casting);
TerlanNdArrayStatus array_finish_into_checked(
    TerlanNdArrayHandle destination, TerlanNdArrayStatus operation_status,
    TerlanNdArrayHandle result, int32_t casting);
TerlanNdArrayStatus create_array(const void *values, int64_t value_count,
                                 const int64_t *shape, int64_t rank,
                                 size_t element_size, int32_t dtype,
                                 TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus create_empty_array(const int64_t *shape, int64_t rank,
                                       int32_t dtype,
                                       TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_create_external(
    void *data, int64_t byte_count, const int64_t *shape, int64_t rank,
    int32_t dtype, int32_t writable, void *context,
    void (*release)(void *context), int32_t (*flush)(void *context),
    TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_create_external_ordered(
    void *data, int64_t byte_count, const int64_t *shape, int64_t rank,
    int32_t dtype, int32_t writable, int32_t order, void *context,
    void (*release)(void *context), int32_t (*flush)(void *context),
    TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_flush(TerlanNdArrayHandle array);
TerlanNdArrayStatus create_view(TerlanNdArrayHandle source,
                                const int64_t *shape, const int64_t *strides,
                                int64_t rank, int64_t relative_offset,
                                int32_t writable,
                                TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus copy_array(TerlanNdArrayHandle source,
                               TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus copy_array_ordered(TerlanNdArrayHandle source,
                                       int32_t dtype, int32_t order,
                                       int32_t casting,
                                       TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_concatenate_pair(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, int64_t axis,
    TerlanNdArrayHandle *out_array);

TerlanNdArrayStatus terlan_ndarray_v1_sum_where(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_sum_initial(
    TerlanNdArrayHandle array, TerlanNdArrayHandle initial,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims,
    TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_sum_where_initial(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask,
    TerlanNdArrayHandle initial, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_minimum_initial(
    TerlanNdArrayHandle array, TerlanNdArrayHandle initial,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims,
    TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_maximum_initial(
    TerlanNdArrayHandle array, TerlanNdArrayHandle initial,
    const int64_t *axes, int64_t axis_count, int32_t keep_dims,
    TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_minimum_where_initial(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask,
    TerlanNdArrayHandle initial, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_minimum_where(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_maximum_where(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask, const int64_t *axes,
    int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_maximum_where_initial(
    TerlanNdArrayHandle array, TerlanNdArrayHandle mask,
    TerlanNdArrayHandle initial, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_fmin_reduce(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_fmax_reduce(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_bitwise_and_reduce(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_bitwise_or_reduce(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_bitwise_xor_reduce(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_outer_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle lhs,
    TerlanNdArrayHandle rhs);
TerlanNdArrayStatus terlan_ndarray_reduce_configured(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, int64_t correction, ReductionOperation operation,
    int32_t skip_nan, TerlanNdArrayHandle mask, TerlanNdArrayHandle initial,
    int32_t requested_dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus reduction_plan_create(
    TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count,
    int32_t keep_dims, TerlanNdArrayReductionPlan *out_plan);
void reduction_plan_indexes(TerlanNdArrayHandle array,
                            const TerlanNdArrayReductionPlan *plan,
                            int64_t logical_index, int64_t *out_output,
                            int64_t *out_reduced);
void reduction_plan_destroy(TerlanNdArrayReductionPlan *plan);

void terlan_ndarray_test_fail_allocation_after(int64_t successful_allocations);
void terlan_ndarray_test_reset_allocation_failures(void);

#endif
