#include "terlan_ndarray_internal.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct GradientSpacing {
  TerlanNdArrayHandle coordinates;
  long double uniform;
  int is_uniform;
} GradientSpacing;

static TerlanNdArrayStatus gradient_spacing_create(
    TerlanNdArrayHandle coordinates, long double uniform, int64_t axis_length,
    GradientSpacing *out_spacing) {
  TerlanNdArrayStatus status;
  out_spacing->coordinates = NULL;
  out_spacing->uniform = uniform;
  out_spacing->is_uniform = 1;
  if (coordinates == NULL) {
    return TERLAN_NDARRAY_OK;
  }
  status = require_live(coordinates);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (dtype_is_complex(coordinates->dtype) || coordinates->rank > 1) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (coordinates->rank == 0) {
    TerlanNdArrayScalar value;
    status = dtype_load(array_const_element_pointer(coordinates, 0),
                        coordinates->dtype, &value);
    if (status == TERLAN_NDARRAY_OK) {
      out_spacing->uniform = value.real;
    }
    return status;
  }
  if (coordinates->shape[0] != axis_length) {
    return TERLAN_NDARRAY_SHAPE_MISMATCH;
  }
  out_spacing->coordinates = coordinates;
  out_spacing->is_uniform = 0;
  return TERLAN_NDARRAY_OK;
}

static long double gradient_coordinate(const GradientSpacing *spacing,
                                       int64_t index) {
  TerlanNdArrayScalar value;
  (void)dtype_load(array_const_element_pointer(spacing->coordinates, index),
                   spacing->coordinates->dtype, &value);
  return value.real;
}

static long double gradient_delta(const GradientSpacing *spacing,
                                  int64_t interval) {
  if (spacing->is_uniform != 0) {
    return spacing->uniform;
  }
  return gradient_coordinate(spacing, interval + 1) -
         gradient_coordinate(spacing, interval);
}

static int64_t gradient_index(const TerlanNdArrayAxisLayout *layout,
                              int64_t axis_length, int64_t outer,
                              int64_t position, int64_t inner) {
  return (outer * axis_length + position) * layout->inner + inner;
}

static TerlanNdArrayStatus gradient_combination(
    TerlanNdArrayHandle array, TerlanNdArrayHandle result,
    const TerlanNdArrayAxisLayout *layout, int64_t outer, int64_t inner,
    int64_t destination_position, const int64_t *positions,
    const long double *coefficients, int64_t term_count) {
  TerlanNdArrayScalar value = {0.0L, 0.0L};
  const int64_t axis_length = array->shape[layout->axis];
  int64_t term;
  for (term = 0; term < term_count; ++term) {
    TerlanNdArrayScalar source;
    TerlanNdArrayStatus status = dtype_load(
        array_const_element_pointer(
            array, gradient_index(layout, axis_length, outer,
                                  positions[term], inner)),
        array->dtype, &source);
    if (status != TERLAN_NDARRAY_OK) {
      return status;
    }
    value = scalar_add(value, scalar_scale(source, coefficients[term]));
  }
  if (!dtype_is_complex(result->dtype)) {
    value.imag = 0.0L;
  }
  return dtype_store(
      array_element_pointer(
          result, gradient_index(layout, axis_length, outer,
                                 destination_position, inner)),
      result->dtype, value);
}

static TerlanNdArrayStatus gradient_interior(
    TerlanNdArrayHandle array, TerlanNdArrayHandle result,
    const TerlanNdArrayAxisLayout *layout, const GradientSpacing *spacing) {
  const int64_t axis_length = array->shape[layout->axis];
  int64_t outer;
  for (outer = 0; outer < layout->outer; ++outer) {
    int64_t position;
    for (position = 1; position + 1 < axis_length; ++position) {
      long double coefficients[3];
      const int64_t positions[3] = {position - 1, position, position + 1};
      int64_t inner;
      if (spacing->is_uniform != 0) {
        coefficients[0] = -0.5L / spacing->uniform;
        coefficients[1] = 0.0L;
        coefficients[2] = 0.5L / spacing->uniform;
      } else {
        const long double left = gradient_delta(spacing, position - 1);
        const long double right = gradient_delta(spacing, position);
        coefficients[0] = -right / (left * (left + right));
        coefficients[1] = (right - left) / (left * right);
        coefficients[2] = left / (right * (left + right));
      }
      for (inner = 0; inner < layout->inner; ++inner) {
        TerlanNdArrayStatus status = gradient_combination(
            array, result, layout, outer, inner, position, positions,
            coefficients, 3);
        if (status != TERLAN_NDARRAY_OK) {
          return status;
        }
      }
    }
  }
  return TERLAN_NDARRAY_OK;
}

static void gradient_edge_coefficients(const GradientSpacing *spacing,
                                       int edge_order, int trailing,
                                       long double *out_coefficients) {
  const int64_t count = spacing->is_uniform != 0
                            ? 0
                            : spacing->coordinates->shape[0];
  const int64_t first_interval = trailing != 0 ? count - 3 : 0;
  const int64_t second_interval = trailing != 0 ? count - 2 : 1;
  if (edge_order == 1) {
    const long double delta = gradient_delta(
        spacing, trailing != 0 ? second_interval : first_interval);
    out_coefficients[0] = -1.0L / delta;
    out_coefficients[1] = 1.0L / delta;
    return;
  }
  if (spacing->is_uniform != 0) {
    if (trailing == 0) {
      out_coefficients[0] = -1.5L / spacing->uniform;
      out_coefficients[1] = 2.0L / spacing->uniform;
      out_coefficients[2] = -0.5L / spacing->uniform;
    } else {
      out_coefficients[0] = 0.5L / spacing->uniform;
      out_coefficients[1] = -2.0L / spacing->uniform;
      out_coefficients[2] = 1.5L / spacing->uniform;
    }
    return;
  }
  {
    const long double first = gradient_delta(spacing, first_interval);
    const long double second = gradient_delta(spacing, second_interval);
    if (trailing == 0) {
      out_coefficients[0] =
          -(2.0L * first + second) / (first * (first + second));
      out_coefficients[1] = (first + second) / (first * second);
      out_coefficients[2] = -first / (second * (first + second));
    } else {
      out_coefficients[0] = second / (first * (first + second));
      out_coefficients[1] = -(second + first) / (first * second);
      out_coefficients[2] =
          (2.0L * second + first) / (second * (first + second));
    }
  }
}

static TerlanNdArrayStatus gradient_edges(
    TerlanNdArrayHandle array, TerlanNdArrayHandle result,
    const TerlanNdArrayAxisLayout *layout, const GradientSpacing *spacing,
    int edge_order) {
  const int64_t axis_length = array->shape[layout->axis];
  const int64_t leading_positions[3] = {0, 1, 2};
  const int64_t trailing_positions[3] = {axis_length - 3, axis_length - 2,
                                         axis_length - 1};
  long double leading[3] = {0.0L, 0.0L, 0.0L};
  long double trailing[3] = {0.0L, 0.0L, 0.0L};
  int64_t outer;
  gradient_edge_coefficients(spacing, edge_order, 0, leading);
  gradient_edge_coefficients(spacing, edge_order, 1, trailing);
  for (outer = 0; outer < layout->outer; ++outer) {
    int64_t inner;
    for (inner = 0; inner < layout->inner; ++inner) {
      TerlanNdArrayStatus status = gradient_combination(
          array, result, layout, outer, inner, 0, leading_positions, leading,
          edge_order + 1);
      if (status == TERLAN_NDARRAY_OK) {
        status = gradient_combination(
            array, result, layout, outer, inner, axis_length - 1,
            trailing_positions + (edge_order == 1 ? 1 : 0), trailing,
            edge_order + 1);
      }
      if (status != TERLAN_NDARRAY_OK) {
        return status;
      }
    }
  }
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus gradient_axis(
    TerlanNdArrayHandle array, TerlanNdArrayHandle coordinates,
    long double uniform_spacing, int64_t axis, int edge_order,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayAxisLayout layout;
  GradientSpacing spacing;
  TerlanNdArrayHandle result = NULL;
  TerlanNdArrayStatus status = require_live(array);
  if (status != TERLAN_NDARRAY_OK || out_array == NULL ||
      (edge_order != 1 && edge_order != 2)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  status = array_axis_layout(array, axis, &layout);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  if (array->shape[layout.axis] < edge_order + 1) {
    return TERLAN_NDARRAY_DOMAIN_ERROR;
  }
  status = gradient_spacing_create(coordinates, uniform_spacing,
                                   array->shape[layout.axis], &spacing);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  status = create_empty_array(array->shape, array->rank,
                              dtype_inexact_or_float64(array->dtype), &result);
  if (status == TERLAN_NDARRAY_OK) {
    status = gradient_interior(array, result, &layout, &spacing);
  }
  if (status == TERLAN_NDARRAY_OK) {
    status = gradient_edges(array, result, &layout, &spacing, edge_order);
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

static void gradient_delete_outputs(uint64_t *values, size_t count) {
  size_t index;
  if (values == NULL) {
    return;
  }
  for (index = 0; index < count; ++index) {
    if (values[index] != 0) {
      (void)terlan_ndarray_v1_delete(
          (TerlanNdArrayHandle)(uintptr_t)values[index]);
    }
  }
  free(values);
}

static TerlanNdArrayStatus gradient_many(
    TerlanNdArrayHandle array, const uint64_t *spacing_arrays,
    int64_t spacing_array_count, const double *uniform_spacings,
    int64_t uniform_spacing_count, const int64_t *axes, int64_t axis_count,
    int all_axes, int edge_order, uint64_t **out_values,
    size_t *out_length) {
  TerlanNdArrayStatus status = require_live(array);
  uint64_t *values = NULL;
  uint8_t *seen = NULL;
  int64_t output_count = all_axes != 0 ? array->rank : axis_count;
  int64_t index;
  if (status != TERLAN_NDARRAY_OK || out_values == NULL ||
      out_length == NULL || output_count < 0 ||
      (output_count > 0 && all_axes == 0 && axes == NULL) ||
      (edge_order != 1 && edge_order != 2)) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_values = NULL;
  *out_length = 0;
  if (spacing_arrays != NULL &&
      spacing_array_count != output_count && spacing_array_count != 1) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (uniform_spacings != NULL &&
      uniform_spacing_count != output_count && uniform_spacing_count != 1) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (output_count > 0) {
    values = checked_calloc((size_t)output_count, sizeof(*values));
    seen = checked_calloc((size_t)array->rank, sizeof(*seen));
    if (values == NULL || (seen == NULL && array->rank > 0)) {
      free(seen);
      free(values);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
  }
  for (index = 0; index < output_count; ++index) {
    TerlanNdArrayAxisLayout layout;
    TerlanNdArrayHandle spacing = NULL;
    TerlanNdArrayHandle output = NULL;
    const int64_t axis = all_axes != 0 ? index : axes[index];
    long double uniform = 1.0L;
    status = array_axis_layout(array, axis, &layout);
    if (status != TERLAN_NDARRAY_OK || seen[layout.axis] != 0) {
      status = status != TERLAN_NDARRAY_OK ? status
                                           : TERLAN_NDARRAY_DUPLICATE_AXIS;
      goto cleanup;
    }
    seen[layout.axis] = 1;
    if (spacing_arrays != NULL) {
      const int64_t spacing_index = spacing_array_count == 1 ? 0 : index;
      spacing = (TerlanNdArrayHandle)(uintptr_t)spacing_arrays[spacing_index];
      if (spacing == NULL) {
        status = TERLAN_NDARRAY_INVALID_ARGUMENT;
        goto cleanup;
      }
      status = require_live(spacing);
      if (status != TERLAN_NDARRAY_OK) {
        goto cleanup;
      }
      if (spacing_array_count == 1 && output_count > 1 && spacing->rank != 0) {
        status = TERLAN_NDARRAY_INVALID_ARGUMENT;
        goto cleanup;
      }
    } else if (uniform_spacings != NULL) {
      uniform = uniform_spacings[uniform_spacing_count == 1 ? 0 : index];
    }
    status = gradient_axis(array, spacing, uniform, layout.axis, edge_order,
                           &output);
    if (status != TERLAN_NDARRAY_OK) {
      goto cleanup;
    }
    values[index] = (uint64_t)(uintptr_t)output;
  }
  free(seen);
  *out_values = values;
  *out_length = (size_t)output_count;
  return TERLAN_NDARRAY_OK;

cleanup:
  free(seen);
  gradient_delete_outputs(values, (size_t)output_count);
  return status;
}

TerlanNdArrayStatus terlan_ndarray_v1_gradient(
    TerlanNdArrayHandle array, double spacing, int64_t axis,
    TerlanNdArrayHandle *out_array) {
  return gradient_axis(array, NULL, spacing, axis, 1, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_gradient_edge(
    TerlanNdArrayHandle array, double spacing, int64_t axis,
    int32_t edge_order, TerlanNdArrayHandle *out_array) {
  return gradient_axis(array, NULL, spacing, axis, edge_order, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_gradient_coordinates(
    TerlanNdArrayHandle array, TerlanNdArrayHandle coordinates, int64_t axis,
    int32_t edge_order, TerlanNdArrayHandle *out_array) {
  return gradient_axis(array, coordinates, 1.0L, axis, edge_order, out_array);
}

TerlanNdArrayStatus terlan_ndarray_v1_gradient_all(
    TerlanNdArrayHandle array, double spacing, int32_t edge_order,
    uint64_t **out_values, size_t *out_length) {
  return gradient_many(array, NULL, 0, &spacing, 1, NULL, 0, 1, edge_order,
                       out_values, out_length);
}

TerlanNdArrayStatus terlan_ndarray_v1_gradient_uniform_axes(
    TerlanNdArrayHandle array, const double *spacings, int64_t spacing_count,
    const int64_t *axes, int64_t axis_count, int32_t edge_order,
    uint64_t **out_values, size_t *out_length) {
  return gradient_many(array, NULL, 0, spacings, spacing_count, axes,
                       axis_count, 0, edge_order, out_values, out_length);
}

TerlanNdArrayStatus terlan_ndarray_v1_gradient_axes(
    TerlanNdArrayHandle array, const uint64_t *spacings,
    int64_t spacing_count, const int64_t *axes, int64_t axis_count,
    int32_t edge_order, uint64_t **out_values, size_t *out_length) {
  return gradient_many(array, spacings, spacing_count, NULL, 0, axes,
                       axis_count, 0, edge_order, out_values, out_length);
}
