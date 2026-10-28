#include "terlan_ndarray_internal.h"

#include <complex.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

TerlanNdArrayStatus array_axis_layout(
    TerlanNdArrayHandle array, int64_t axis,
    TerlanNdArrayAxisLayout *out_layout) {
  int64_t dimension;
  if (axis < 0) {
    axis += array->rank;
  }
  if (axis < 0 || axis >= array->rank) {
    return TERLAN_NDARRAY_AXIS_OUT_OF_RANGE;
  }
  out_layout->axis = axis;
  out_layout->outer = 1;
  out_layout->inner = 1;
  for (dimension = 0; dimension < axis; ++dimension) {
    out_layout->outer *= array->shape[dimension];
  }
  for (dimension = axis + 1; dimension < array->rank; ++dimension) {
    out_layout->inner *= array->shape[dimension];
  }
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_difference(
    TerlanNdArrayHandle array, int64_t order, int64_t axis,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle current = NULL;
  int64_t iteration;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL || order < 0) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = array_axis_layout(array, axis, &(TerlanNdArrayAxisLayout){0});
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  status = copy_array(array, &current);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (iteration = 0; iteration < order; ++iteration) {
    TerlanNdArrayAxisLayout layout;
    TerlanNdArrayHandle next = NULL;
    int64_t *shape = NULL;
    int64_t outer_index;
    status = array_axis_layout(current, axis, &layout);
    if (status != TERLAN_NDARRAY_OK) {
      goto fail;
    }
    if (current->rank > 0) {
      shape = checked_malloc((size_t)current->rank * sizeof(*shape));
      if (shape == NULL) {
        status = TERLAN_NDARRAY_ALLOCATION_FAILED;
        goto fail;
      }
      for (outer_index = 0; outer_index < current->rank; ++outer_index) {
        shape[outer_index] = current->shape[outer_index];
      }
      if (shape[layout.axis] > 0) {
        --shape[layout.axis];
      }
    }
    status = create_empty_array(shape, current->rank, current->dtype, &next);
    free(shape);
    if (status != TERLAN_NDARRAY_OK) {
      goto fail;
    }
    for (outer_index = 0; outer_index < layout.outer; ++outer_index) {
      int64_t axis_index;
      for (axis_index = 0; axis_index < next->shape[layout.axis]; ++axis_index) {
        int64_t inner_index;
        for (inner_index = 0; inner_index < layout.inner; ++inner_index) {
          const int64_t source =
              (outer_index * current->shape[layout.axis] + axis_index) *
                  layout.inner +
              inner_index;
          const int64_t destination =
              (outer_index * next->shape[layout.axis] + axis_index) *
                  layout.inner +
              inner_index;
          TerlanNdArrayScalar lhs;
          TerlanNdArrayScalar rhs;
          (void)dtype_load(array_const_element_pointer(current, source +
                                                                   layout.inner),
                           current->dtype, &lhs);
          (void)dtype_load(array_const_element_pointer(current, source),
                           current->dtype, &rhs);
          status = dtype_store(array_element_pointer(next, destination),
                               next->dtype, scalar_subtract(lhs, rhs));
          if (status != TERLAN_NDARRAY_OK) {
            (void)terlan_ndarray_v1_delete(next);
            goto fail;
          }
        }
      }
    }
    (void)terlan_ndarray_v1_delete(current);
    current = next;
  }
  *out_array = current;
  return TERLAN_NDARRAY_OK;

fail:
  (void)terlan_ndarray_v1_delete(current);
  return status;
}

static TerlanNdArrayStatus difference_with_boundaries(
    TerlanNdArrayHandle array, TerlanNdArrayHandle prepend,
    TerlanNdArrayHandle append, int64_t order, int64_t axis,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayHandle joined = NULL;
  TerlanNdArrayHandle next = NULL;
  TerlanNdArrayStatus status;
  if (out_array == NULL || (prepend == NULL && append == NULL)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = copy_array(array, &joined);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (prepend != NULL) {
    status = terlan_ndarray_concatenate_pair(prepend, joined, axis, &next);
    if (status != TERLAN_NDARRAY_OK) {
      goto cleanup;
    }
    (void)terlan_ndarray_v1_delete(joined);
    joined = next;
    next = NULL;
  }
  if (append != NULL) {
    status = terlan_ndarray_concatenate_pair(joined, append, axis, &next);
    if (status != TERLAN_NDARRAY_OK) {
      goto cleanup;
    }
    (void)terlan_ndarray_v1_delete(joined);
    joined = next;
    next = NULL;
  }
  status = terlan_ndarray_v1_difference(joined, order, axis, out_array);

cleanup:
  if (next != NULL) {
    (void)terlan_ndarray_v1_delete(next);
  }
  if (joined != NULL) {
    (void)terlan_ndarray_v1_delete(joined);
  }
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_difference_prepend(
    TerlanNdArrayHandle array, TerlanNdArrayHandle prepend, int64_t order,
    int64_t axis, TerlanNdArrayHandle *out_array) {
  return difference_with_boundaries(array, prepend, NULL, order, axis,
                                    out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_difference_append(
    TerlanNdArrayHandle array, TerlanNdArrayHandle append, int64_t order,
    int64_t axis, TerlanNdArrayHandle *out_array) {
  return difference_with_boundaries(array, NULL, append, order, axis,
                                    out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_difference_surround(
    TerlanNdArrayHandle array, TerlanNdArrayHandle prepend,
    TerlanNdArrayHandle append, int64_t order, int64_t axis,
    TerlanNdArrayHandle *out_array) {
  return difference_with_boundaries(array, prepend, append, order, axis,
                                    out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_unwrap(
    TerlanNdArrayHandle array, double period, int64_t axis,
    TerlanNdArrayHandle *out_array) {
  return terlan_ndarray_v1_unwrap_discontinuity(array, period, period / 2.0,
                                                axis, out_array);
}

static TerlanNdArrayStatus unwrap_array(
    TerlanNdArrayHandle array, double period, double discontinuity,
    int64_t axis, int32_t output_dtype, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayAxisLayout layout;
  int64_t outer_index;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL ||
      dtype_is_complex(array->dtype) || !isfinite(period) || period <= 0.0 ||
      !isfinite(discontinuity) || discontinuity < 0.0) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = array_axis_layout(array, axis, &layout);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  status = create_empty_array(array->shape, array->rank, output_dtype, &result);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  for (outer_index = 0; outer_index < layout.outer; ++outer_index) {
    int64_t inner_index;
    for (inner_index = 0; inner_index < layout.inner; ++inner_index) {
      long double correction = 0.0L;
      long double previous = 0.0L;
      int64_t axis_index;
      for (axis_index = 0; axis_index < array->shape[layout.axis];
           ++axis_index) {
        const int64_t index =
            (outer_index * array->shape[layout.axis] + axis_index) *
                layout.inner +
            inner_index;
        TerlanNdArrayScalar value;
        (void)dtype_load(array_const_element_pointer(array, index), array->dtype,
                         &value);
        if (axis_index > 0) {
          const long double delta = value.real - previous;
          long double wrapped = fmodl(delta + period / 2.0L, period);
          if (wrapped < 0.0L) {
            wrapped += period;
          }
          wrapped -= period / 2.0L;
          if (wrapped == -period / 2.0L && delta > 0.0L) {
            wrapped = period / 2.0L;
          }
          if (fabsl(delta) >= discontinuity) {
            correction += wrapped - delta;
          }
        }
        previous = value.real;
        value.real += correction;
        (void)dtype_store(array_element_pointer(result, index), result->dtype,
                          value);
      }
    }
  }
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_unwrap_discontinuity(
    TerlanNdArrayHandle array, double period, double discontinuity,
    int64_t axis, TerlanNdArrayHandle *out_array) {
  const int32_t output_dtype = array == NULL
                                   ? TERLAN_NDARRAY_DTYPE_FLOAT64
                                   : dtype_inexact_or_float64(array->dtype);
  return unwrap_array(array, period, discontinuity, axis, output_dtype,
                      out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_unwrap_integer_period(
    TerlanNdArrayHandle array, int64_t period, int64_t discontinuity,
    int64_t axis, TerlanNdArrayHandle *out_array) {
  const int32_t output_dtype =
      array != NULL && (dtype_is_float(array->dtype) ||
                        dtype_is_complex(array->dtype))
          ? array->dtype
          : TERLAN_NDARRAY_DTYPE_INT64;
  if (period <= 0 || discontinuity < 0) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  return unwrap_array(array, (double)period, (double)discontinuity, axis,
                      output_dtype, out_array);
}

typedef struct InterpolationPoint {
  long double coordinate;
  TerlanNdArrayScalar sample;
  int64_t order;
} InterpolationPoint;

static int compare_interpolation_points(const void *lhs_pointer,
                                        const void *rhs_pointer) {
  const InterpolationPoint *lhs = lhs_pointer;
  const InterpolationPoint *rhs = rhs_pointer;
  if (isnan(lhs->coordinate) || isnan(rhs->coordinate)) {
    if (isnan(lhs->coordinate) != isnan(rhs->coordinate)) {
      return isnan(lhs->coordinate) ? 1 : -1;
    }
  } else if (lhs->coordinate < rhs->coordinate) {
    return -1;
  } else if (lhs->coordinate > rhs->coordinate) {
    return 1;
  }
  return lhs->order < rhs->order ? -1 : lhs->order > rhs->order;
}

static long double positive_remainder(long double value, long double period) {
  long double result = fmodl(value, period);
  return result < 0.0L ? result + period : result;
}

static TerlanNdArrayStatus interpolation_points_create(
    TerlanNdArrayHandle coordinates, TerlanNdArrayHandle samples,
    long double period, InterpolationPoint **out_points,
    int64_t *out_count) {
  TerlanNdArrayStatus status = require_live(coordinates);
  InterpolationPoint *points;
  int64_t index;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(samples);
  }
  if (status != TERLAN_NDARRAY_OK || coordinates->rank != 1 ||
      samples->rank != 1 || coordinates->numel != samples->numel ||
      coordinates->numel == 0 || dtype_is_complex(coordinates->dtype)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  points = checked_malloc((size_t)(coordinates->numel + (period > 0.0L ? 2 : 0)) *
                          sizeof(*points));
  if (points == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  for (index = 0; index < coordinates->numel; ++index) {
    TerlanNdArrayScalar coordinate;
    status = dtype_load(array_const_element_pointer(coordinates, index),
                        coordinates->dtype, &coordinate);
    if (status == TERLAN_NDARRAY_OK) {
      status = dtype_load(array_const_element_pointer(samples, index),
                          samples->dtype, &points[index].sample);
    }
    if (status != TERLAN_NDARRAY_OK) {
      free(points);
      return status;
    }
    points[index].coordinate = period > 0.0L
                                   ? positive_remainder(coordinate.real, period)
                                   : coordinate.real;
    points[index].order = index;
  }
  if (period > 0.0L) {
    qsort(points, (size_t)coordinates->numel, sizeof(*points),
          compare_interpolation_points);
    points[coordinates->numel + 1] = points[0];
    points[coordinates->numel + 1].coordinate += period;
    points[coordinates->numel] = points[coordinates->numel - 1];
    points[coordinates->numel].coordinate -= period;
    for (index = coordinates->numel; index > 0; --index) {
      points[index] = points[index - 1];
    }
    points[0] = points[coordinates->numel + 1 - 1];
    points[0].coordinate -= period;
    points[coordinates->numel + 1] = points[1];
    points[coordinates->numel + 1].coordinate += period;
    *out_count = coordinates->numel + 2;
  } else {
    *out_count = coordinates->numel;
  }
  *out_points = points;
  return TERLAN_NDARRAY_OK;
}

static int64_t interpolation_search(long double value,
                                    const InterpolationPoint *points,
                                    int64_t count) {
  int64_t low = 0;
  int64_t high = count;
  if (value > points[count - 1].coordinate) {
    return count;
  }
  if (value < points[0].coordinate) {
    return -1;
  }
  while (low < high) {
    const int64_t middle = low + (high - low) / 2;
    if (value >= points[middle].coordinate) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low - 1;
}

static long double interpolation_component(long double value,
                                           long double x0, long double x1,
                                           long double y0, long double y1) {
  const long double slope = (y1 - y0) / (x1 - x0);
  long double result = slope * (value - x0) + y0;
  if (isnan(result)) {
    result = slope * (value - x1) + y1;
    if (isnan(result) && y0 == y1) {
      result = y0;
    }
  }
  return result;
}

static TerlanNdArrayStatus interpolate_array(
    TerlanNdArrayHandle values, TerlanNdArrayHandle coordinates,
    TerlanNdArrayHandle samples, const TerlanNdArrayScalar *left,
    const TerlanNdArrayScalar *right, long double period,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(values);
  TerlanNdArrayHandle result = NULL;
  InterpolationPoint *points = NULL;
  int64_t point_count = 0;
  TerlanNdArrayScalar left_value;
  TerlanNdArrayScalar right_value;
  int coordinates_have_nan = 0;
  int64_t index;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL ||
      dtype_is_complex(values->dtype) || period < 0.0L ||
      (period > 0.0L && !isfinite(period))) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = interpolation_points_create(coordinates, samples, period, &points,
                                       &point_count);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  left_value = left == NULL ? points[0].sample : *left;
  right_value = right == NULL ? points[point_count - 1].sample : *right;
  for (index = 0; index < point_count; ++index) {
    coordinates_have_nan |= isnan(points[index].coordinate) != 0;
  }
  status = create_empty_array(
      values->shape, values->rank,
      dtype_is_complex(samples->dtype) ? TERLAN_NDARRAY_DTYPE_COMPLEX128
                                       : TERLAN_NDARRAY_DTYPE_FLOAT64,
      &result);
  if (status != TERLAN_NDARRAY_OK) {
    free(points);
    return status;
  }
  for (index = 0; index < values->numel; ++index) {
    TerlanNdArrayScalar query;
    TerlanNdArrayScalar output;
    int64_t position;
    (void)dtype_load(array_const_element_pointer(values, index), values->dtype,
                     &query);
    if (period > 0.0L) {
      query.real = positive_remainder(query.real, period);
    }
    if (isnan(query.real) || coordinates_have_nan != 0) {
      output.real = query.real;
      if (!isnan(output.real)) {
        output.real = NAN;
      }
      output.imag = 0.0L;
    } else {
      position = interpolation_search(query.real, points, point_count);
      if (position < 0) {
        output = left_value;
      } else if (position >= point_count) {
        output = right_value;
      } else if (position == point_count - 1 ||
                 points[position].coordinate == query.real) {
        output = points[position].sample;
      } else {
        output.real = interpolation_component(
            query.real, points[position].coordinate,
            points[position + 1].coordinate, points[position].sample.real,
            points[position + 1].sample.real);
        output.imag = interpolation_component(
            query.real, points[position].coordinate,
            points[position + 1].coordinate, points[position].sample.imag,
            points[position + 1].sample.imag);
      }
    }
    status = dtype_store(array_element_pointer(result, index), result->dtype,
                         output);
    if (status != TERLAN_NDARRAY_OK) {
      (void)terlan_ndarray_v1_delete(result);
      free(points);
      return status;
    }
  }
  free(points);
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_interpolate(
    TerlanNdArrayHandle values, TerlanNdArrayHandle coordinates,
    TerlanNdArrayHandle samples, double left, double right,
    TerlanNdArrayHandle *out_array) {
  const TerlanNdArrayScalar left_value = {left, 0.0L};
  const TerlanNdArrayScalar right_value = {right, 0.0L};
  return interpolate_array(values, coordinates, samples, &left_value,
                           &right_value, 0.0L, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_interpolate_default(
    TerlanNdArrayHandle values, TerlanNdArrayHandle coordinates,
    TerlanNdArrayHandle samples, TerlanNdArrayHandle *out_array) {
  return interpolate_array(values, coordinates, samples, NULL, NULL, 0.0L,
                           out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_interpolate_boundaries(
    TerlanNdArrayHandle values, TerlanNdArrayHandle coordinates,
    TerlanNdArrayHandle samples, TerlanNdArrayHandle left,
    TerlanNdArrayHandle right, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(left);
  TerlanNdArrayScalar left_value;
  TerlanNdArrayScalar right_value;
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(right);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = require_live(samples);
  }
  if (status != TERLAN_NDARRAY_OK || left->rank != 0 || right->rank != 0 ||
      (!dtype_is_complex(samples->dtype) &&
       (dtype_is_complex(left->dtype) || dtype_is_complex(right->dtype)))) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  (void)dtype_load(array_const_element_pointer(left, 0), left->dtype,
                   &left_value);
  (void)dtype_load(array_const_element_pointer(right, 0), right->dtype,
                   &right_value);
  return interpolate_array(values, coordinates, samples, &left_value,
                           &right_value, 0.0L, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_interpolate_periodic(
    TerlanNdArrayHandle values, TerlanNdArrayHandle coordinates,
    TerlanNdArrayHandle samples, double period,
    TerlanNdArrayHandle *out_array) {
  if (period == 0.0 || !isfinite(period)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  return interpolate_array(values, coordinates, samples, NULL, NULL,
                           fabsl(period), out_array);
}

static TerlanNdArrayStatus trapezoid_array(
    TerlanNdArrayHandle array, TerlanNdArrayHandle coordinates,
    long double spacing, int64_t axis, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = require_live(array);
  TerlanNdArrayReductionPlan plan;
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayAxisLayout layout;
  int32_t output_dtype;
  int64_t outer_index;
  if (status != TERLAN_NDARRAY_OK || out_array == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = array_axis_layout(array, axis, &layout);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (coordinates != NULL) {
    status = require_live(coordinates);
    if (status != TERLAN_NDARRAY_OK || dtype_is_complex(coordinates->dtype) ||
        !((coordinates->rank == 1 &&
           coordinates->numel == array->shape[layout.axis]) ||
          (coordinates->rank == array->rank &&
           coordinates->shape[layout.axis] == array->shape[layout.axis] &&
           array_broadcast_compatible(coordinates, array->shape,
                                      array->rank)))) {
      return status != TERLAN_NDARRAY_OK ? status
                                         : TERLAN_NDARRAY_SHAPE_MISMATCH;
    }
  } else if (!isfinite(spacing)) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  status = reduction_plan_create(array, &layout.axis, 1, 0, &plan);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  output_dtype = coordinates == NULL
                     ? array->dtype
                     : dtype_promote(array->dtype, coordinates->dtype);
  output_dtype = dtype_inexact_or_float64(output_dtype);
  status = create_empty_array(plan.output_shape, plan.output_rank, output_dtype,
                              &result);
  if (status != TERLAN_NDARRAY_OK) {
    reduction_plan_destroy(&plan);
    return status;
  }
  for (outer_index = 0; outer_index < layout.outer; ++outer_index) {
    int64_t inner_index;
    for (inner_index = 0; inner_index < layout.inner; ++inner_index) {
      TerlanNdArrayScalar sum = {0.0L, 0.0L};
      int64_t axis_index;
      for (axis_index = 1; axis_index < array->shape[layout.axis];
           ++axis_index) {
        const int64_t previous =
            (outer_index * array->shape[layout.axis] + axis_index - 1) *
                layout.inner +
            inner_index;
        const int64_t current = previous + layout.inner;
        TerlanNdArrayScalar lhs;
        TerlanNdArrayScalar rhs;
        long double width = spacing;
        (void)dtype_load(array_const_element_pointer(array, previous),
                         array->dtype, &lhs);
        (void)dtype_load(array_const_element_pointer(array, current),
                         array->dtype, &rhs);
        if (coordinates != NULL) {
          TerlanNdArrayScalar x0;
          TerlanNdArrayScalar x1;
          const int64_t coordinate_previous =
              coordinates->rank == 1
                  ? axis_index - 1
                  : array_broadcast_index(coordinates, array->shape,
                                          array->rank, previous);
          const int64_t coordinate_current =
              coordinates->rank == 1
                  ? axis_index
                  : array_broadcast_index(coordinates, array->shape,
                                          array->rank, current);
          (void)dtype_load(array_const_element_pointer(coordinates,
                                                       coordinate_previous),
                           coordinates->dtype, &x0);
          (void)dtype_load(array_const_element_pointer(coordinates,
                                                       coordinate_current),
                           coordinates->dtype, &x1);
          width = x1.real - x0.real;
        }
        sum.real += (lhs.real + rhs.real) * width / 2.0L;
        sum.imag += (lhs.imag + rhs.imag) * width / 2.0L;
      }
      status = dtype_store(array_element_pointer(
                               result, outer_index * layout.inner +
                                           inner_index),
                           result->dtype, sum);
      if (status != TERLAN_NDARRAY_OK) {
        (void)terlan_ndarray_v1_delete(result);
        reduction_plan_destroy(&plan);
        return status;
      }
    }
  }
  reduction_plan_destroy(&plan);
  *out_array = result;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_trapezoidal_integral(
    TerlanNdArrayHandle array, TerlanNdArrayHandle coordinates, int64_t axis,
    TerlanNdArrayHandle *out_array) {
  return trapezoid_array(array, coordinates, 0.0L, axis, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_trapezoidal_integral_uniform(
    TerlanNdArrayHandle array, double spacing, int64_t axis,
    TerlanNdArrayHandle *out_array) {
  return trapezoid_array(array, NULL, spacing, axis, out_array);
}
