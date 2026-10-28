#include "terlan_ndarray_internal.h"

#include <cblas.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct TerlanMatmulPlan {
  int64_t batch_rank;
  int64_t batch_count;
  int64_t rows;
  int64_t shared;
  int64_t columns;
  int64_t output_rank;
  int64_t *output_shape;
  int lhs_vector;
  int rhs_vector;
} TerlanMatmulPlan;

static int checked_multiply(int64_t lhs, int64_t rhs, int64_t *out_value) {
  if (lhs < 0 || rhs < 0 || out_value == NULL ||
      (lhs != 0 && rhs > INT64_MAX / lhs)) {
    return 0;
  }
  *out_value = lhs * rhs;
  return 1;
}

static void scalar_accumulate(TerlanNdArrayScalar *accumulator,
                              TerlanNdArrayScalar value) {
  accumulator->real += value.real;
  accumulator->imag += value.imag;
}

static int64_t batch_offset(TerlanNdArrayHandle array, int64_t trailing_rank,
                            const int64_t *batch_shape, int64_t batch_rank,
                            int64_t batch_index) {
  const int64_t array_batch_rank = array->rank - trailing_rank;
  const int64_t alignment = batch_rank - array_batch_rank;
  int64_t remaining = batch_index;
  int64_t offset = 0;
  int64_t dimension;
  for (dimension = batch_rank; dimension > 0; --dimension) {
    const int64_t output_dimension = dimension - 1;
    const int64_t coordinate = remaining % batch_shape[output_dimension];
    remaining /= batch_shape[output_dimension];
    if (output_dimension >= alignment) {
      const int64_t array_dimension = output_dimension - alignment;
      if (array->shape[array_dimension] != 1) {
        offset += coordinate * array->strides[array_dimension];
      }
    }
  }
  return offset;
}

static const void *element_at_offset(TerlanNdArrayHandle array,
                                     int64_t element_offset) {
  return (const uint8_t *)array->data +
         element_offset * (int64_t)dtype_size(array->dtype);
}

static TerlanNdArrayStatus initialize_matmul_plan(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanMatmulPlan *plan) {
  const int64_t lhs_batch_rank = lhs->rank == 1 ? 0 : lhs->rank - 2;
  const int64_t rhs_batch_rank = rhs->rank == 1 ? 0 : rhs->rank - 2;
  int64_t output_dimension = 0;
  int64_t dimension;
  int64_t batch_count = 1;

  if (plan == NULL || lhs->rank < 1 || rhs->rank < 1) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  plan->lhs_vector = lhs->rank == 1;
  plan->rhs_vector = rhs->rank == 1;
  plan->batch_rank =
      lhs_batch_rank > rhs_batch_rank ? lhs_batch_rank : rhs_batch_rank;
  plan->rows = plan->lhs_vector ? 1 : lhs->shape[lhs->rank - 2];
  plan->shared = lhs->shape[lhs->rank - 1];
  plan->columns = plan->rhs_vector ? 1 : rhs->shape[rhs->rank - 1];
  if (plan->shared !=
      rhs->shape[plan->rhs_vector ? 0 : rhs->rank - 2]) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  plan->output_rank = plan->batch_rank + (plan->lhs_vector ? 0 : 1) +
                      (plan->rhs_vector ? 0 : 1);
  plan->output_shape = NULL;
  if (plan->output_rank > 0) {
    plan->output_shape =
        checked_malloc((size_t)plan->output_rank * sizeof(int64_t));
    if (plan->output_shape == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (dimension = 0; dimension < plan->batch_rank; ++dimension) {
    const int64_t lhs_dimension = dimension - (plan->batch_rank - lhs_batch_rank);
    const int64_t rhs_dimension = dimension - (plan->batch_rank - rhs_batch_rank);
    const int64_t lhs_size = lhs_dimension < 0 ? 1 : lhs->shape[lhs_dimension];
    const int64_t rhs_size = rhs_dimension < 0 ? 1 : rhs->shape[rhs_dimension];
    int64_t output_size;
    if (lhs_size != rhs_size && lhs_size != 1 && rhs_size != 1) {
      free(plan->output_shape);
      plan->output_shape = NULL;
      return TERLAN_NDARRAY_SHAPE_MISMATCH;
    }
    output_size = lhs_size > rhs_size ? lhs_size : rhs_size;
    plan->output_shape[output_dimension++] = output_size;
    if (!checked_multiply(batch_count, output_size, &batch_count)) {
      free(plan->output_shape);
      plan->output_shape = NULL;
      return TERLAN_NDARRAY_SIZE_OVERFLOW;
    }
  }
  if (!plan->lhs_vector) {
    plan->output_shape[output_dimension++] = plan->rows;
  }
  if (!plan->rhs_vector) {
    plan->output_shape[output_dimension] = plan->columns;
  }
  plan->batch_count = batch_count;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus execute_generic_matmul(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs,
    const TerlanMatmulPlan *plan, TerlanNdArrayHandle result) {
  const int64_t lhs_trailing_rank = plan->lhs_vector ? 1 : 2;
  const int64_t rhs_trailing_rank = plan->rhs_vector ? 1 : 2;
  int64_t batch;
  for (batch = 0; batch < plan->batch_count; ++batch) {
    const int64_t lhs_base = batch_offset(
        lhs, lhs_trailing_rank, plan->output_shape, plan->batch_rank, batch);
    const int64_t rhs_base = batch_offset(
        rhs, rhs_trailing_rank, plan->output_shape, plan->batch_rank, batch);
    int64_t row;
    for (row = 0; row < plan->rows; ++row) {
      int64_t column;
      for (column = 0; column < plan->columns; ++column) {
        TerlanNdArrayScalar accumulator = {0.0L, 0.0L};
        int64_t inner;
        for (inner = 0; inner < plan->shared; ++inner) {
          const int64_t lhs_offset =
              lhs_base +
              (plan->lhs_vector
                   ? inner * lhs->strides[0]
                   : row * lhs->strides[lhs->rank - 2] +
                         inner * lhs->strides[lhs->rank - 1]);
          const int64_t rhs_offset =
              rhs_base +
              (plan->rhs_vector
                   ? inner * rhs->strides[0]
                   : inner * rhs->strides[rhs->rank - 2] +
                         column * rhs->strides[rhs->rank - 1]);
          TerlanNdArrayScalar lhs_value;
          TerlanNdArrayScalar rhs_value;
          TerlanNdArrayStatus status =
              dtype_load(element_at_offset(lhs, lhs_offset), lhs->dtype,
                         &lhs_value);
          if (status == TERLAN_NDARRAY_OK) {
            status = dtype_load(element_at_offset(rhs, rhs_offset), rhs->dtype,
                                &rhs_value);
          }
          if (status != TERLAN_NDARRAY_OK) {
            return status;
          }
          scalar_accumulate(&accumulator,
                            scalar_multiply(lhs_value, rhs_value));
        }
        {
          const int64_t output_index =
              (batch * plan->rows + row) * plan->columns + column;
          TerlanNdArrayStatus status =
              dtype_store(array_element_pointer(result, output_index),
                          result->dtype, accumulator);
          if (status != TERLAN_NDARRAY_OK) {
            return status;
          }
        }
      }
    }
  }
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_matmul(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(lhs);
  TerlanMatmulPlan plan = {0, 0, 0, 0, 0, 0, NULL, 0, 0};
  TerlanNdArrayHandle result = NULL;
  int32_t output_dtype;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(rhs);
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  output_dtype = dtype_promote(lhs->dtype, rhs->dtype);
  if (output_dtype == 0) {
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  status = initialize_matmul_plan(lhs, rhs, &plan);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (lhs->rank == 2 && rhs->rank == 2 && output_dtype == TERLAN_NDARRAY_DTYPE_FLOAT64 &&
      lhs->dtype == TERLAN_NDARRAY_DTYPE_FLOAT64 &&
      rhs->dtype == TERLAN_NDARRAY_DTYPE_FLOAT64 &&
      (sizeof(blasint) != sizeof(int32_t) || plan.rows > INT32_MAX ||
       plan.shared > INT32_MAX || plan.columns > INT32_MAX)) {
    free(plan.output_shape);
    return TERLAN_NDARRAY_BLAS_INTEGER_WIDTH;
  }
  status = create_empty_array(plan.output_shape, plan.output_rank, output_dtype,
                              &result);
  if (status == TERLAN_NDARRAY_OK && lhs->rank == 2 && rhs->rank == 2 &&
      output_dtype == TERLAN_NDARRAY_DTYPE_FLOAT64 &&
      lhs->dtype == TERLAN_NDARRAY_DTYPE_FLOAT64 &&
      rhs->dtype == TERLAN_NDARRAY_DTYPE_FLOAT64 && array_is_contiguous(lhs) &&
      array_is_contiguous(rhs) && plan.rows != 0 && plan.columns != 0 &&
      plan.shared != 0) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (blasint)plan.rows,
                (blasint)plan.columns, (blasint)plan.shared, 1.0,
                (const double *)lhs->data, (blasint)plan.shared,
                (const double *)rhs->data, (blasint)plan.columns, 0.0,
                (double *)result->data, (blasint)plan.columns);
  } else if (status == TERLAN_NDARRAY_OK) {
    status = execute_generic_matmul(lhs, rhs, &plan, result);
  }
  free(plan.output_shape);
  if (status != TERLAN_NDARRAY_OK) {
    if (result != NULL) {
      (void)terlan_ndarray_v1_delete(result);
    }
    return status;
  }
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus initialize_dot_shape(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, int64_t **out_shape,
    int64_t *out_rank, int64_t *out_shared) {
  int64_t rank;
  int64_t *shape = NULL;
  int64_t dimension = 0;
  if (out_shape == NULL || out_rank == NULL || out_shared == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (lhs->rank == 0 || rhs->rank == 0) {
    TerlanNdArrayHandle source = lhs->rank == 0 ? rhs : lhs;
    rank = source->rank;
    if (rank > 0) {
      shape = checked_malloc((size_t)rank * sizeof(int64_t));
      if (shape == NULL) {
        return TERLAN_NDARRAY_ALLOCATION_FAILED;
      }
      for (dimension = 0; dimension < rank; ++dimension) {
        shape[dimension] = source->shape[dimension];
      }
    }
    *out_shape = shape;
    *out_rank = rank;
    *out_shared = 0;
    return TERLAN_NDARRAY_OK;
  }
  if (lhs->shape[lhs->rank - 1] !=
      rhs->shape[rhs->rank == 1 ? 0 : rhs->rank - 2]) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  rank = lhs->rank - 1 + (rhs->rank == 1 ? 0 : rhs->rank - 1);
  if (rank > 0) {
    shape = checked_malloc((size_t)rank * sizeof(int64_t));
    if (shape == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (dimension = 0; dimension < lhs->rank - 1; ++dimension) {
    shape[dimension] = lhs->shape[dimension];
  }
  if (rhs->rank > 1) {
    int64_t rhs_dimension;
    int64_t output_dimension = lhs->rank - 1;
    for (rhs_dimension = 0; rhs_dimension < rhs->rank; ++rhs_dimension) {
      if (rhs_dimension != rhs->rank - 2) {
        shape[output_dimension++] = rhs->shape[rhs_dimension];
      }
    }
  }
  *out_shape = shape;
  *out_rank = rank;
  *out_shared = lhs->shape[lhs->rank - 1];
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus execute_dot(TerlanNdArrayHandle lhs,
                                       TerlanNdArrayHandle rhs,
                                       int64_t shared,
                                       TerlanNdArrayHandle result) {
  int64_t output_index;
  if (lhs->rank == 0 || rhs->rank == 0) {
    TerlanNdArrayScalar scalar;
    TerlanNdArrayHandle array;
    TerlanNdArrayStatus status;
    if (lhs->rank == 0) {
      status = dtype_load(array_const_element_pointer(lhs, 0), lhs->dtype,
                          &scalar);
      array = rhs;
    } else {
      status = dtype_load(array_const_element_pointer(rhs, 0), rhs->dtype,
                          &scalar);
      array = lhs;
    }
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
    for (output_index = 0; output_index < result->numel; ++output_index) {
      TerlanNdArrayScalar value;
      status = dtype_load(array_const_element_pointer(array, output_index),
                          array->dtype, &value);
      if (status == TERLAN_NDARRAY_OK) {
        status = dtype_store(array_element_pointer(result, output_index),
                             result->dtype, scalar_multiply(scalar, value));
      }
      if (status != TERLAN_NDARRAY_OK) {
        return status;
      }
    }
    return TERLAN_NDARRAY_OK;
  }
  for (output_index = 0; output_index < result->numel; ++output_index) {
    int64_t remaining = output_index;
    int64_t lhs_base = 0;
    int64_t rhs_base = 0;
    int64_t output_dimension;
    TerlanNdArrayScalar accumulator = {0.0L, 0.0L};
    for (output_dimension = result->rank; output_dimension > 0;
         --output_dimension) {
      const int64_t index = output_dimension - 1;
      const int64_t coordinate = remaining % result->shape[index];
      remaining /= result->shape[index];
      if (index < lhs->rank - 1) {
        lhs_base += coordinate * lhs->strides[index];
      } else if (rhs->rank > 1) {
        const int64_t rhs_output_index = index - (lhs->rank - 1);
        const int64_t rhs_dimension =
            rhs_output_index < rhs->rank - 2 ? rhs_output_index
                                              : rhs_output_index + 1;
        rhs_base += coordinate * rhs->strides[rhs_dimension];
      }
    }
    {
      int64_t inner;
      for (inner = 0; inner < shared; ++inner) {
        TerlanNdArrayScalar lhs_value;
        TerlanNdArrayScalar rhs_value;
        TerlanNdArrayStatus status = dtype_load(
            element_at_offset(lhs, lhs_base + inner * lhs->strides[lhs->rank - 1]),
            lhs->dtype, &lhs_value);
        if (status == TERLAN_NDARRAY_OK) {
          const int64_t rhs_axis = rhs->rank == 1 ? 0 : rhs->rank - 2;
          status = dtype_load(
              element_at_offset(rhs, rhs_base + inner * rhs->strides[rhs_axis]),
              rhs->dtype, &rhs_value);
        }
        if (status != TERLAN_NDARRAY_OK) {
          return status;
        }
        scalar_accumulate(&accumulator, scalar_multiply(lhs_value, rhs_value));
      }
    }
    {
      TerlanNdArrayStatus status =
          dtype_store(array_element_pointer(result, output_index), result->dtype,
                      accumulator);
      if (status != TERLAN_NDARRAY_OK) {
        return status;
      }
    }
  }
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_dot(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(lhs);
  TerlanNdArrayHandle result = NULL;
  int64_t *output_shape = NULL;
  int64_t output_rank = 0;
  int64_t shared = 0;
  int32_t output_dtype;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(rhs);
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  output_dtype = dtype_promote(lhs->dtype, rhs->dtype);
  if (output_dtype == 0) {
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  status = initialize_dot_shape(lhs, rhs, &output_shape, &output_rank, &shared);
  if (status == TERLAN_NDARRAY_OK) {
    status = create_empty_array(output_shape, output_rank, output_dtype, &result);
  }
  free(output_shape);
  if (status == TERLAN_NDARRAY_OK) {
    status = execute_dot(lhs, rhs, shared, result);
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

static TerlanNdArrayStatus create_contraction_result(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, int64_t rhs_axis,
    TerlanNdArrayHandle *out_array) {
  const int64_t output_rank = lhs->rank + rhs->rank - 2;
  int64_t *output_shape = NULL;
  int64_t output_dimension = 0;
  int64_t dimension;
  int32_t output_dtype;
  TerlanNdArrayStatus status;
  if (lhs->shape[lhs->rank - 1] != rhs->shape[rhs_axis]) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  output_dtype = dtype_promote(lhs->dtype, rhs->dtype);
  if (output_dtype == 0) {
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  if (output_rank > 0) {
    output_shape = checked_malloc((size_t)output_rank * sizeof(int64_t));
    if (output_shape == NULL) {
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (dimension = 0; dimension < lhs->rank - 1; ++dimension) {
    output_shape[output_dimension++] = lhs->shape[dimension];
  }
  for (dimension = 0; dimension < rhs->rank; ++dimension) {
    if (dimension != rhs_axis) {
      output_shape[output_dimension++] = rhs->shape[dimension];
    }
  }
  status = create_empty_array(output_shape, output_rank, output_dtype, out_array);
  free(output_shape);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_inner(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(lhs);
  TerlanNdArrayHandle result = NULL;
  int64_t output_index;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(rhs);
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (lhs->rank == 0 || rhs->rank == 0) {
    return terlan_ndarray_v1_multiply(lhs, rhs, out_array);
  }
  status = create_contraction_result(lhs, rhs, rhs->rank - 1, &result);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (output_index = 0; output_index < result->numel; ++output_index) {
    int64_t remaining = output_index;
    int64_t lhs_base = 0;
    int64_t rhs_base = 0;
    int64_t output_dimension;
    TerlanNdArrayScalar accumulator = {0.0L, 0.0L};
    for (output_dimension = result->rank; output_dimension > 0;
         --output_dimension) {
      const int64_t index = output_dimension - 1;
      const int64_t coordinate = remaining % result->shape[index];
      remaining /= result->shape[index];
      if (index < lhs->rank - 1) {
        lhs_base += coordinate * lhs->strides[index];
      } else {
        rhs_base += coordinate * rhs->strides[index - (lhs->rank - 1)];
      }
    }
    {
      int64_t inner;
      for (inner = 0; inner < lhs->shape[lhs->rank - 1]; ++inner) {
        TerlanNdArrayScalar lhs_value;
        TerlanNdArrayScalar rhs_value;
        status = dtype_load(
            element_at_offset(lhs, lhs_base + inner * lhs->strides[lhs->rank - 1]),
            lhs->dtype, &lhs_value);
        if (status == TERLAN_NDARRAY_OK) {
          status = dtype_load(
              element_at_offset(rhs, rhs_base + inner * rhs->strides[rhs->rank - 1]),
              rhs->dtype, &rhs_value);
        }
        if (status != TERLAN_NDARRAY_OK) {
          break;
        }
        scalar_accumulate(&accumulator, scalar_multiply(lhs_value, rhs_value));
      }
    }
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_store_cast(array_element_pointer(result, output_index),
                                result->dtype, accumulator,
                                TERLAN_NDARRAY_CAST_UNSAFE);
    }
    if (status != TERLAN_NDARRAY_OK) {
      (void)terlan_ndarray_v1_delete(result);
      return status;
    }
  }
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus flat_product(TerlanNdArrayHandle lhs,
                                        TerlanNdArrayHandle rhs,
                                        int conjugate_lhs,
                                        TerlanNdArrayScalar *out_value) {
  int64_t index;
  out_value->real = 0.0L;
  out_value->imag = 0.0L;
  for (index = 0; index < lhs->numel; ++index) {
    TerlanNdArrayScalar lhs_value;
    TerlanNdArrayScalar rhs_value;
    TerlanNdArrayStatus status =
        dtype_load(array_const_element_pointer(lhs, index), lhs->dtype,
                   &lhs_value);
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_load(array_const_element_pointer(rhs, index), rhs->dtype,
                          &rhs_value);
    }
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
    if (conjugate_lhs) {
      lhs_value.imag = -lhs_value.imag;
    }
    scalar_accumulate(out_value, scalar_multiply(lhs_value, rhs_value));
  }
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_vdot(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(lhs);
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayScalar value;
  int32_t output_dtype;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(rhs);
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (lhs->numel != rhs->numel) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  output_dtype = dtype_promote(lhs->dtype, rhs->dtype);
  if (output_dtype == 0) {
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  status = create_empty_array(NULL, 0, output_dtype, &result);
  if (status == TERLAN_NDARRAY_OK) {
    status = flat_product(lhs, rhs, 1, &value);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = dtype_store_cast(array_element_pointer(result, 0), result->dtype,
                              value, TERLAN_NDARRAY_CAST_UNSAFE);
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

TerlanNdArrayStatus terlan_ndarray_v1_outer(
    TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(lhs);
  TerlanNdArrayHandle result = NULL;
  const int64_t output_shape[2] = {lhs == NULL ? 0 : lhs->numel,
                                   rhs == NULL ? 0 : rhs->numel};
  int32_t output_dtype;
  int64_t lhs_index;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(rhs);
  }
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  output_dtype = dtype_promote(lhs->dtype, rhs->dtype);
  if (output_dtype == 0) {
    return TERLAN_NDARRAY_UNSUPPORTED_DTYPE;
  }
  status = create_empty_array(output_shape, 2, output_dtype, &result);
  for (lhs_index = 0; status == TERLAN_NDARRAY_OK && lhs_index < lhs->numel;
       ++lhs_index) {
    TerlanNdArrayScalar lhs_value;
    int64_t rhs_index;
    status = dtype_load(array_const_element_pointer(lhs, lhs_index), lhs->dtype,
                        &lhs_value);
    for (rhs_index = 0; status == TERLAN_NDARRAY_OK && rhs_index < rhs->numel;
         ++rhs_index) {
      TerlanNdArrayScalar rhs_value;
      status = dtype_load(array_const_element_pointer(rhs, rhs_index), rhs->dtype,
                          &rhs_value);
      if (status == TERLAN_NDARRAY_OK) {
        status = dtype_store(
            array_element_pointer(result, lhs_index * rhs->numel + rhs_index),
            result->dtype, scalar_multiply(lhs_value, rhs_value));
      }
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

TerlanNdArrayStatus terlan_ndarray_v1_trace(
    TerlanNdArrayHandle array, int64_t offset, int64_t axis_a, int64_t axis_b,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayHandle diagonal = NULL;
  int64_t reduction_axis;
  TerlanNdArrayStatus status;
  if (out_array == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = terlan_ndarray_v1_diagonal(array, offset, axis_a, axis_b, &diagonal);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  reduction_axis = diagonal->rank - 1;
  status = terlan_ndarray_v1_sum(diagonal, &reduction_axis, 1, 0, out_array);
  (void)terlan_ndarray_v1_delete(diagonal);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_matmul_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle lhs,
    TerlanNdArrayHandle rhs) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status =
      terlan_ndarray_v1_matmul(lhs, rhs, &result);
  return array_finish_into(destination, status, result,
                           TERLAN_NDARRAY_CAST_NO);
}

TerlanNdArrayStatus terlan_ndarray_v1_outer_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle lhs,
    TerlanNdArrayHandle rhs) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status =
      terlan_ndarray_v1_outer(lhs, rhs, &result);
  return array_finish_into(destination, status, result,
                           TERLAN_NDARRAY_CAST_SAME_KIND);
}

TerlanNdArrayStatus terlan_ndarray_v1_dot_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle lhs,
    TerlanNdArrayHandle rhs) {
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayStatus status = require_live(destination);
  if (status == TERLAN_NDARRAY_OK && !array_is_contiguous(destination)) {
    status = TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = terlan_ndarray_v1_dot(lhs, rhs, &result);
  }
  return array_finish_into(destination, status, result,
                           TERLAN_NDARRAY_CAST_NO);
}

TerlanNdArrayStatus terlan_ndarray_v1_trace_into(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle array, int64_t offset,
    int64_t axis_a, int64_t axis_b) {
  TerlanNdArrayHandle result = NULL;
  const TerlanNdArrayStatus status = terlan_ndarray_v1_trace(
      array, offset, axis_a, axis_b, &result);
  return array_finish_into(destination, status, result,
                           TERLAN_NDARRAY_CAST_UNSAFE);
}
