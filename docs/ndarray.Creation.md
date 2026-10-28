# ndarray.Creation

NumPy-compatible array creation and triangular matrix operations.

## `linspace_arrays`

Broadcasts numerical endpoint arrays and inserts evenly spaced samples at the requested axis.

- C symbol: `terlan_ndarray_v1_linspace_arrays`
- NativeBoundary operation: `ndarray.creation.linspace_arrays`
- Ownership: `opaque_handle`

## `linspace_steps`

Returns the broadcast-shaped spacing for array endpoints, including NaN for an undefined spacing.

- C symbol: `terlan_ndarray_v1_linspace_steps`
- NativeBoundary operation: `ndarray.creation.linspace_steps`
- Ownership: `opaque_handle`

## `eye_ordered`

Creates a rectangular diagonal-unit matrix with explicit C or Fortran storage order.

- C symbol: `terlan_ndarray_v1_eye`
- NativeBoundary operation: `ndarray.creation.eye_ordered`
- Ownership: `opaque_handle`

## `empty_like`

Allocates an uninitialized array with the source shape and dtype.

- C symbol: `terlan_ndarray_v1_empty_like`
- NativeBoundary operation: `ndarray.creation.empty_like`
- Ownership: `opaque_handle`

## `zeros_like`

Allocates a zero-filled array with the source shape and dtype.

- C symbol: `terlan_ndarray_v1_zeros_like`
- NativeBoundary operation: `ndarray.creation.zeros_like`
- Ownership: `opaque_handle`

## `ones_like`

Allocates a one-filled array with the source shape and dtype.

- C symbol: `terlan_ndarray_v1_ones_like`
- NativeBoundary operation: `ndarray.creation.ones_like`
- Ownership: `opaque_handle`

## `full_like_int`

Fills a new source-shaped and source-typed array with an integer scalar.

- C symbol: `terlan_ndarray_v1_full_like_int`
- NativeBoundary operation: `ndarray.creation.full_like_int`
- Ownership: `opaque_handle`

## `full_like_float`

Fills a new source-shaped and source-typed array with a floating scalar.

- C symbol: `terlan_ndarray_v1_full_like_float`
- NativeBoundary operation: `ndarray.creation.full_like_float`
- Ownership: `opaque_handle`

## `logspace`

Returns powers of a positive base with evenly spaced exponents.

- C symbol: `terlan_ndarray_v1_logspace`
- NativeBoundary operation: `ndarray.creation.logspace`
- Ownership: `opaque_handle`

## `geomspace`

Returns evenly spaced values on a same-sign geometric progression.

- C symbol: `terlan_ndarray_v1_geomspace`
- NativeBoundary operation: `ndarray.creation.geomspace`
- Ownership: `opaque_handle`

## `logspace_arrays`

Broadcasts exponent arrays and inserts powers of a positive base at the requested axis.

- C symbol: `terlan_ndarray_v1_logspace_arrays`
- NativeBoundary operation: `ndarray.creation.logspace_arrays`
- Ownership: `opaque_handle`

## `geomspace_arrays`

Broadcasts nonzero endpoints and inserts geometrically spaced samples at the requested axis.

- C symbol: `terlan_ndarray_v1_geomspace_arrays`
- NativeBoundary operation: `ndarray.creation.geomspace_arrays`
- Ownership: `opaque_handle`

## `tri`

Creates a matrix with ones at and below an indexed diagonal.

- C symbol: `terlan_ndarray_v1_tri`
- NativeBoundary operation: `ndarray.creation.tri`
- Ownership: `opaque_handle`

## `tril`

Copies an array while zeroing entries above an indexed diagonal in every matrix plane.

- C symbol: `terlan_ndarray_v1_tril`
- NativeBoundary operation: `ndarray.creation.tril`
- Ownership: `opaque_handle`

## `triu`

Copies an array while zeroing entries below an indexed diagonal in every matrix plane.

- C symbol: `terlan_ndarray_v1_triu`
- NativeBoundary operation: `ndarray.creation.triu`
- Ownership: `opaque_handle`

## `diagflat`

Flattens an array and places its values on an indexed diagonal of a new square matrix.

- C symbol: `terlan_ndarray_v1_diagflat`
- NativeBoundary operation: `ndarray.creation.diagflat`
- Ownership: `opaque_handle`

## `indices`

Returns one dense numeric coordinate grid for every dimension of a shape.

- C symbol: `terlan_ndarray_v1_indices`
- NativeBoundary operation: `ndarray.creation.indices`
- Ownership: `opaque_handle`

## `indices_sparse`

Returns one sparse numeric coordinate grid per dimension, with singleton dimensions elsewhere.

- C symbol: `terlan_ndarray_v1_indices_sparse`
- NativeBoundary operation: `ndarray.creation.indices_sparse`
- Ownership: `opaque_handle`

## `vander`

Builds a Vandermonde matrix with explicit column count and exponent order.

- C symbol: `terlan_ndarray_v1_vander`
- NativeBoundary operation: `ndarray.creation.vander`
- Ownership: `opaque_handle`
