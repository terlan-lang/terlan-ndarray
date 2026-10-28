# ndarray.Legendre

Legendre polynomial algebra and numerical methods.

## `leg2poly`

Converts coefficients to the power basis.

- C symbol: `terlan_ndarray_v1_leg2poly`
- NativeBoundary operation: `ndarray.array.leg2poly`
- Ownership: `opaque_handle`

## `poly2leg`

Converts power coefficients to the leg basis.

- C symbol: `terlan_ndarray_v1_poly2leg`
- NativeBoundary operation: `ndarray.array.poly2leg`
- Ownership: `opaque_handle`

## `legadd`

Adds coefficient vectors.

- C symbol: `terlan_ndarray_v1_legadd`
- NativeBoundary operation: `ndarray.array.legadd`
- Ownership: `opaque_handle`

## `legsub`

Subtracts coefficient vectors.

- C symbol: `terlan_ndarray_v1_legsub`
- NativeBoundary operation: `ndarray.array.legsub`
- Ownership: `opaque_handle`

## `legmul`

Multiplies coefficient vectors.

- C symbol: `terlan_ndarray_v1_legmul`
- NativeBoundary operation: `ndarray.array.legmul`
- Ownership: `opaque_handle`

## `legdiv`

Returns quotient and remainder.

- C symbol: `terlan_ndarray_v1_legdiv`
- NativeBoundary operation: `ndarray.array.legdiv`
- Ownership: `opaque_handle`

## `legmulx`

Multiplies coefficients by x.

- C symbol: `terlan_ndarray_v1_legmulx`
- NativeBoundary operation: `ndarray.array.legmulx`
- Ownership: `opaque_handle`

## `legpow`

Raises coefficients to a nonnegative power.

- C symbol: `terlan_ndarray_v1_legpow`
- NativeBoundary operation: `ndarray.array.legpow`
- Ownership: `opaque_handle`

## `legder`

Differentiates coefficients.

- C symbol: `terlan_ndarray_v1_legder`
- NativeBoundary operation: `ndarray.array.legder`
- Ownership: `opaque_handle`

## `legint`

Integrates coefficients.

- C symbol: `terlan_ndarray_v1_legint`
- NativeBoundary operation: `ndarray.array.legint`
- Ownership: `opaque_handle`

## `legfit`

Fits coefficients to samples.

- C symbol: `terlan_ndarray_v1_legfit`
- NativeBoundary operation: `ndarray.array.legfit`
- Ownership: `opaque_handle`

## `legfromroots`

Builds coefficients from roots.

- C symbol: `terlan_ndarray_v1_legfromroots`
- NativeBoundary operation: `ndarray.array.legfromroots`
- Ownership: `opaque_handle`

## `legcompanion`

Builds a companion matrix.

- C symbol: `terlan_ndarray_v1_legcompanion`
- NativeBoundary operation: `ndarray.array.legcompanion`
- Ownership: `opaque_handle`

## `legroots`

Returns polynomial roots.

- C symbol: `terlan_ndarray_v1_legroots`
- NativeBoundary operation: `ndarray.array.legroots`
- Ownership: `opaque_handle`

## `legtrim`

Removes insignificant high-order coefficients.

- C symbol: `terlan_ndarray_v1_legtrim`
- NativeBoundary operation: `ndarray.array.legtrim`
- Ownership: `opaque_handle`

## `legline`

Builds coefficients for offset plus scale times x.

- C symbol: `terlan_ndarray_v1_legline`
- NativeBoundary operation: `ndarray.array.legline`
- Ownership: `opaque_handle`

## `legval`

Evaluates coefficients.

- C symbol: `terlan_ndarray_v1_legval`
- NativeBoundary operation: `ndarray.array.legval`
- Ownership: `opaque_handle`

## `legval2d`

Evaluates a two-dimensional coefficient tensor.

- C symbol: `terlan_ndarray_v1_legval2d`
- NativeBoundary operation: `ndarray.array.legval2d`
- Ownership: `opaque_handle`

## `legval3d`

Evaluates a three-dimensional coefficient tensor.

- C symbol: `terlan_ndarray_v1_legval3d`
- NativeBoundary operation: `ndarray.array.legval3d`
- Ownership: `opaque_handle`

## `leggrid2d`

Evaluates over a two-dimensional Cartesian grid.

- C symbol: `terlan_ndarray_v1_leggrid2d`
- NativeBoundary operation: `ndarray.array.leggrid2d`
- Ownership: `opaque_handle`

## `leggrid3d`

Evaluates over a three-dimensional Cartesian grid.

- C symbol: `terlan_ndarray_v1_leggrid3d`
- NativeBoundary operation: `ndarray.array.leggrid3d`
- Ownership: `opaque_handle`

## `legvander`

Builds a Vandermonde matrix.

- C symbol: `terlan_ndarray_v1_legvander`
- NativeBoundary operation: `ndarray.array.legvander`
- Ownership: `opaque_handle`

## `legvander2d`

Builds a two-dimensional Vandermonde matrix.

- C symbol: `terlan_ndarray_v1_legvander2d`
- NativeBoundary operation: `ndarray.array.legvander2d`
- Ownership: `opaque_handle`

## `legvander3d`

Builds a three-dimensional Vandermonde matrix.

- C symbol: `terlan_ndarray_v1_legvander3d`
- NativeBoundary operation: `ndarray.array.legvander3d`
- Ownership: `opaque_handle`

## `leggauss`

Returns Gaussian quadrature nodes and weights.

- C symbol: `terlan_ndarray_v1_leggauss`
- NativeBoundary operation: `ndarray.array.leggauss`
- Ownership: `opaque_handle`

## `legweight`

Evaluates the orthogonality weight function.

- C symbol: `terlan_ndarray_v1_legweight`
- NativeBoundary operation: `ndarray.array.legweight`
- Ownership: `opaque_handle`

## `legder`

Differentiates Legendre coefficient lanes along an axis with derivative scaling.

- C symbol: `terlan_ndarray_v1_legder_axis`
- NativeBoundary operation: `ndarray.array.legder_axis`
- Ownership: `opaque_handle`

## `legint`

Integrates Legendre coefficient lanes along an axis with constants, a lower bound, and scaling.

- C symbol: `terlan_ndarray_v1_legint_axis`
- NativeBoundary operation: `ndarray.array.legint_axis`
- Ownership: `opaque_handle`

## `legval`

Evaluates Legendre coefficient tensors with Cartesian or broadcast semantics.

- C symbol: `terlan_ndarray_v1_legval_tensor`
- NativeBoundary operation: `ndarray.array.legval_tensor`
- Ownership: `opaque_handle`

## `legfit`

Fits selected Legendre degrees to scalar or matrix-valued samples.

- C symbol: `terlan_ndarray_v1_legfit_degrees`
- NativeBoundary operation: `ndarray.array.legfit_degrees`
- Ownership: `opaque_handle`

## `legfit`

Fits selected Legendre degrees using sample weights.

- C symbol: `terlan_ndarray_v1_legfit_weighted_degrees`
- NativeBoundary operation: `ndarray.array.legfit_weighted_degrees`
- Ownership: `opaque_handle`
