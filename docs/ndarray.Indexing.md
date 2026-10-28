# ndarray.Indexing

Compositional basic indexing and shape transforms.

## `basic_slice`

Builds one shared basic-indexing view; Int minimum denotes an omitted bound and ellipsis_index expands omitted source axes.

- C symbol: `terlan_ndarray_v1_basic_slice`
- NativeBoundary operation: `ndarray.shape.basic_slice`
- Ownership: `opaque_handle`

## `take_indices`

Takes an integer index Array while preserving every index dimension in the output shape.

- C symbol: `terlan_ndarray_v1_take_indices`
- NativeBoundary operation: `ndarray.indexing.take_indices`
- Ownership: `opaque_handle`

## `take_indices_mode`

Takes an integer index Array using raise, wrap, or clip normalization while preserving index dimensions.

- C symbol: `terlan_ndarray_v1_take_indices_mode`
- NativeBoundary operation: `ndarray.indexing.take_indices_mode`
- Ownership: `opaque_handle`

## `take_indices_into`

Atomically writes an Array-indexed take result into an exactly shaped destination, including overlapping source views.

- C symbol: `terlan_ndarray_v1_take_indices_into`
- NativeBoundary operation: `ndarray.indexing.take_indices_into`
- Ownership: `value`

## `compress_into`

Atomically writes an axis-compressed result into an exact same-dtype destination.

- C symbol: `terlan_ndarray_v1_compress_into`
- NativeBoundary operation: `ndarray.indexing.compress_into`
- Ownership: `value`

## `compress_flat_into`

Atomically writes a flattened compressed result into an exact same-dtype destination.

- C symbol: `terlan_ndarray_v1_compress_flat_into`
- NativeBoundary operation: `ndarray.indexing.compress_flat_into`
- Ownership: `value`

## `add_at_coordinates`

Applies sequential unbuffered addition at coordinate rows while staging aliases and committing atomically.

- C symbol: `terlan_ndarray_v1_add_at_coordinates`
- NativeBoundary operation: `ndarray.indexing.add_at_coordinates`
- Ownership: `value`

## `pad_constant`

Pads each axis with explicit leading and trailing constants.

- C symbol: `terlan_ndarray_v1_pad_constant`
- NativeBoundary operation: `ndarray.shape.pad_constant`
- Ownership: `opaque_handle`

## `pad_edge`

Pads each axis by repeating its nearest edge value.

- C symbol: `terlan_ndarray_v1_pad_edge`
- NativeBoundary operation: `ndarray.shape.pad_edge`
- Ownership: `opaque_handle`

## `pad_reflect`

Pads each axis with a reflection that excludes the edge value.

- C symbol: `terlan_ndarray_v1_pad_reflect`
- NativeBoundary operation: `ndarray.shape.pad_reflect`
- Ownership: `opaque_handle`

## `pad_symmetric`

Pads each axis with a reflection that includes the edge value.

- C symbol: `terlan_ndarray_v1_pad_symmetric`
- NativeBoundary operation: `ndarray.shape.pad_symmetric`
- Ownership: `opaque_handle`

## `pad_wrap`

Pads each axis by wrapping values from its opposite end.

- C symbol: `terlan_ndarray_v1_pad_wrap`
- NativeBoundary operation: `ndarray.shape.pad_wrap`
- Ownership: `opaque_handle`

## `pad_reflect_odd`

Pads each axis with odd reflection excluding the edge value.

- C symbol: `terlan_ndarray_v1_pad_reflect_odd`
- NativeBoundary operation: `ndarray.shape.pad_reflect_odd`
- Ownership: `opaque_handle`

## `pad_symmetric_odd`

Pads each axis with odd reflection including the edge value.

- C symbol: `terlan_ndarray_v1_pad_symmetric_odd`
- NativeBoundary operation: `ndarray.shape.pad_symmetric_odd`
- Ownership: `opaque_handle`

## `pad_linear_ramp`

Pads each axis with linear ramps from explicit end values to source edges.

- C symbol: `terlan_ndarray_v1_pad_linear_ramp`
- NativeBoundary operation: `ndarray.shape.pad_linear_ramp`
- Ownership: `opaque_handle`

## `pad_empty`

Allocates padded regions without exposing uninitialized memory and preserves source values.

- C symbol: `terlan_ndarray_v1_pad_empty`
- NativeBoundary operation: `ndarray.shape.pad_empty`
- Ownership: `opaque_handle`

## `pad_minimum`

Pads with edge minimums over optional clipped statistic lengths.

- C symbol: `terlan_ndarray_v1_pad_minimum`
- NativeBoundary operation: `ndarray.shape.pad_minimum`
- Ownership: `opaque_handle`

## `pad_maximum`

Pads with edge maximums over optional clipped statistic lengths.

- C symbol: `terlan_ndarray_v1_pad_maximum`
- NativeBoundary operation: `ndarray.shape.pad_maximum`
- Ownership: `opaque_handle`

## `pad_mean`

Pads with edge means over optional clipped statistic lengths.

- C symbol: `terlan_ndarray_v1_pad_mean`
- NativeBoundary operation: `ndarray.shape.pad_mean`
- Ownership: `opaque_handle`

## `pad_median`

Pads with edge medians over optional clipped statistic lengths.

- C symbol: `terlan_ndarray_v1_pad_median`
- NativeBoundary operation: `ndarray.shape.pad_median`
- Ownership: `opaque_handle`

## `pad_constant_values`

Pads each axis with exact numerical values supplied by scalar or rank-sized arrays.

- C symbol: `terlan_ndarray_v1_pad_constant_values`
- NativeBoundary operation: `ndarray.shape.pad_constant_values`
- Ownership: `opaque_handle`

## `pad_linear_ramp_values`

Pads each axis with linear ramps using exact numerical endpoints from scalar or rank-sized arrays.

- C symbol: `terlan_ndarray_v1_pad_linear_ramp_values`
- NativeBoundary operation: `ndarray.shape.pad_linear_ramp_values`
- Ownership: `opaque_handle`

## `pad_statistical`

Pads with minimum, maximum, mean, or median values selected by mode 0 through 3 over explicit edge lengths.

- C symbol: `terlan_ndarray_v1_pad_statistical`
- NativeBoundary operation: `ndarray.shape.pad_statistical`
- Ownership: `opaque_handle`
