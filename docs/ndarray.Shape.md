# ndarray.Shape

NumPy-compatible rank adaptation, flattening, axis movement, and geometric views.

## `atleast_1d`

Returns a view with at least one dimension.

- C symbol: `terlan_ndarray_v1_atleast_1d`
- NativeBoundary operation: `ndarray.shape.atleast_1d`
- Ownership: `opaque_handle`

## `atleast_2d`

Returns a view with at least two dimensions, promoting vectors to rows.

- C symbol: `terlan_ndarray_v1_atleast_2d`
- NativeBoundary operation: `ndarray.shape.atleast_2d`
- Ownership: `opaque_handle`

## `atleast_3d`

Returns a view with at least three dimensions using NumPy rank-promotion rules.

- C symbol: `terlan_ndarray_v1_atleast_3d`
- NativeBoundary operation: `ndarray.shape.atleast_3d`
- Ownership: `opaque_handle`

## `ravel`

Returns a contiguous rank-one view or materialization in logical row-major order.

- C symbol: `terlan_ndarray_v1_ravel`
- NativeBoundary operation: `ndarray.shape.ravel`
- Ownership: `opaque_handle`

## `ravel`

Ravels using explicit C, Fortran, input-compatible, or physical traversal order.

- C symbol: `terlan_ndarray_v1_ravel_ordered`
- NativeBoundary operation: `ndarray.shape.ravel_ordered`
- Ownership: `opaque_handle`

## `flatten`

Returns an independent rank-one copy in logical row-major order.

- C symbol: `terlan_ndarray_v1_flatten`
- NativeBoundary operation: `ndarray.shape.flatten`
- Ownership: `opaque_handle`

## `flatten`

Flattens into an independent copy using explicit C, Fortran, input-compatible, or physical traversal order.

- C symbol: `terlan_ndarray_v1_flatten_ordered`
- NativeBoundary operation: `ndarray.shape.flatten_ordered`
- Ownership: `opaque_handle`

## `moveaxis`

Moves distinct source axes to distinct destinations while preserving all other axis order.

- C symbol: `terlan_ndarray_v1_moveaxis`
- NativeBoundary operation: `ndarray.shape.moveaxis`
- Ownership: `opaque_handle`

## `rollaxis`

Rolls one axis backward to an insertion position.

- C symbol: `terlan_ndarray_v1_rollaxis`
- NativeBoundary operation: `ndarray.shape.rollaxis`
- Ownership: `opaque_handle`

## `flip_all`

Returns a shared negative-stride view with every axis reversed.

- C symbol: `terlan_ndarray_v1_flip_all`
- NativeBoundary operation: `ndarray.shape.flip_all`
- Ownership: `opaque_handle`

## `flipud`

Returns a shared view reversed along axis zero.

- C symbol: `terlan_ndarray_v1_flipud`
- NativeBoundary operation: `ndarray.shape.flipud`
- Ownership: `opaque_handle`

## `fliplr`

Returns a shared view reversed along axis one.

- C symbol: `terlan_ndarray_v1_fliplr`
- NativeBoundary operation: `ndarray.shape.fliplr`
- Ownership: `opaque_handle`

## `rot90`

Rotates matrix planes counterclockwise by a multiple of 90 degrees.

- C symbol: `terlan_ndarray_v1_rot90`
- NativeBoundary operation: `ndarray.shape.rot90`
- Ownership: `opaque_handle`

## `concatenate_flat`

Flattens every input in logical C order and concatenates the resulting vectors.

- C symbol: `terlan_ndarray_v1_concatenate_flat`
- NativeBoundary operation: `ndarray.shape.concatenate_flat`
- Ownership: `opaque_handle`

## `concatenate_with_dtype`

Concatenates arrays and converts the result under an explicit dtype and casting policy.

- C symbol: `terlan_ndarray_v1_concatenate_with_dtype`
- NativeBoundary operation: `ndarray.shape.concatenate_with_dtype`
- Ownership: `opaque_handle`

## `concatenate_flat_with_dtype`

Flattens and concatenates arrays before explicit dtype conversion.

- C symbol: `terlan_ndarray_v1_concatenate_flat_with_dtype`
- NativeBoundary operation: `ndarray.shape.concatenate_flat_with_dtype`
- Ownership: `opaque_handle`

## `concatenate_into`

Concatenates arrays into an exact-shape destination under an explicit casting policy.

- C symbol: `terlan_ndarray_v1_concatenate_into`
- NativeBoundary operation: `ndarray.shape.concatenate_into`
- Ownership: `opaque_handle`

## `concatenate_flat_into`

Flattens and concatenates arrays into an exact-shape destination.

- C symbol: `terlan_ndarray_v1_concatenate_flat_into`
- NativeBoundary operation: `ndarray.shape.concatenate_flat_into`
- Ownership: `opaque_handle`
