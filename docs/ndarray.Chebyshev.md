# ndarray.Chebyshev

Chebyshev polynomial algebra and numerical methods.

## `cheb2poly`

Converts coefficients to the power basis.

- C symbol: `terlan_ndarray_v1_cheb2poly`
- NativeBoundary operation: `ndarray.array.cheb2poly`
- Ownership: `opaque_handle`

## `poly2cheb`

Converts power coefficients to the cheb basis.

- C symbol: `terlan_ndarray_v1_poly2cheb`
- NativeBoundary operation: `ndarray.array.poly2cheb`
- Ownership: `opaque_handle`

## `chebadd`

Adds coefficient vectors.

- C symbol: `terlan_ndarray_v1_chebadd`
- NativeBoundary operation: `ndarray.array.chebadd`
- Ownership: `opaque_handle`

## `chebsub`

Subtracts coefficient vectors.

- C symbol: `terlan_ndarray_v1_chebsub`
- NativeBoundary operation: `ndarray.array.chebsub`
- Ownership: `opaque_handle`

## `chebmul`

Multiplies coefficient vectors.

- C symbol: `terlan_ndarray_v1_chebmul`
- NativeBoundary operation: `ndarray.array.chebmul`
- Ownership: `opaque_handle`

## `chebdiv`

Returns quotient and remainder.

- C symbol: `terlan_ndarray_v1_chebdiv`
- NativeBoundary operation: `ndarray.array.chebdiv`
- Ownership: `opaque_handle`

## `chebmulx`

Multiplies coefficients by x.

- C symbol: `terlan_ndarray_v1_chebmulx`
- NativeBoundary operation: `ndarray.array.chebmulx`
- Ownership: `opaque_handle`

## `chebpow`

Raises coefficients to a nonnegative power.

- C symbol: `terlan_ndarray_v1_chebpow`
- NativeBoundary operation: `ndarray.array.chebpow`
- Ownership: `opaque_handle`

## `chebder`

Differentiates coefficients.

- C symbol: `terlan_ndarray_v1_chebder`
- NativeBoundary operation: `ndarray.array.chebder`
- Ownership: `opaque_handle`

## `chebint`

Integrates coefficients.

- C symbol: `terlan_ndarray_v1_chebint`
- NativeBoundary operation: `ndarray.array.chebint`
- Ownership: `opaque_handle`

## `chebfit`

Fits coefficients to samples.

- C symbol: `terlan_ndarray_v1_chebfit`
- NativeBoundary operation: `ndarray.array.chebfit`
- Ownership: `opaque_handle`

## `chebfromroots`

Builds coefficients from roots.

- C symbol: `terlan_ndarray_v1_chebfromroots`
- NativeBoundary operation: `ndarray.array.chebfromroots`
- Ownership: `opaque_handle`

## `chebcompanion`

Builds a companion matrix.

- C symbol: `terlan_ndarray_v1_chebcompanion`
- NativeBoundary operation: `ndarray.array.chebcompanion`
- Ownership: `opaque_handle`

## `chebroots`

Returns polynomial roots.

- C symbol: `terlan_ndarray_v1_chebroots`
- NativeBoundary operation: `ndarray.array.chebroots`
- Ownership: `opaque_handle`

## `chebtrim`

Removes insignificant high-order coefficients.

- C symbol: `terlan_ndarray_v1_chebtrim`
- NativeBoundary operation: `ndarray.array.chebtrim`
- Ownership: `opaque_handle`

## `chebline`

Builds coefficients for offset plus scale times x.

- C symbol: `terlan_ndarray_v1_chebline`
- NativeBoundary operation: `ndarray.array.chebline`
- Ownership: `opaque_handle`

## `chebval`

Evaluates coefficients.

- C symbol: `terlan_ndarray_v1_chebval`
- NativeBoundary operation: `ndarray.array.chebval`
- Ownership: `opaque_handle`

## `chebval2d`

Evaluates a two-dimensional coefficient tensor.

- C symbol: `terlan_ndarray_v1_chebval2d`
- NativeBoundary operation: `ndarray.array.chebval2d`
- Ownership: `opaque_handle`

## `chebval3d`

Evaluates a three-dimensional coefficient tensor.

- C symbol: `terlan_ndarray_v1_chebval3d`
- NativeBoundary operation: `ndarray.array.chebval3d`
- Ownership: `opaque_handle`

## `chebgrid2d`

Evaluates over a two-dimensional Cartesian grid.

- C symbol: `terlan_ndarray_v1_chebgrid2d`
- NativeBoundary operation: `ndarray.array.chebgrid2d`
- Ownership: `opaque_handle`

## `chebgrid3d`

Evaluates over a three-dimensional Cartesian grid.

- C symbol: `terlan_ndarray_v1_chebgrid3d`
- NativeBoundary operation: `ndarray.array.chebgrid3d`
- Ownership: `opaque_handle`

## `chebvander`

Builds a Vandermonde matrix.

- C symbol: `terlan_ndarray_v1_chebvander`
- NativeBoundary operation: `ndarray.array.chebvander`
- Ownership: `opaque_handle`

## `chebvander2d`

Builds a two-dimensional Vandermonde matrix.

- C symbol: `terlan_ndarray_v1_chebvander2d`
- NativeBoundary operation: `ndarray.array.chebvander2d`
- Ownership: `opaque_handle`

## `chebvander3d`

Builds a three-dimensional Vandermonde matrix.

- C symbol: `terlan_ndarray_v1_chebvander3d`
- NativeBoundary operation: `ndarray.array.chebvander3d`
- Ownership: `opaque_handle`

## `chebgauss`

Returns Gaussian quadrature nodes and weights.

- C symbol: `terlan_ndarray_v1_chebgauss`
- NativeBoundary operation: `ndarray.array.chebgauss`
- Ownership: `opaque_handle`

## `chebweight`

Evaluates the orthogonality weight function.

- C symbol: `terlan_ndarray_v1_chebweight`
- NativeBoundary operation: `ndarray.array.chebweight`
- Ownership: `opaque_handle`

## `chebpts1`

Returns first-kind Chebyshev points.

- C symbol: `terlan_ndarray_v1_chebpts1`
- NativeBoundary operation: `ndarray.array.chebpts1`
- Ownership: `opaque_handle`

## `chebpts2`

Returns second-kind Chebyshev points.

- C symbol: `terlan_ndarray_v1_chebpts2`
- NativeBoundary operation: `ndarray.array.chebpts2`
- Ownership: `opaque_handle`

## `chebinterpolate`

Interpolates sampled values in the Chebyshev basis.

- C symbol: `terlan_ndarray_v1_chebinterpolate`
- NativeBoundary operation: `ndarray.array.chebinterpolate`
- Ownership: `opaque_handle`

## `chebder`

Differentiates Chebyshev coefficient lanes along an axis with derivative scaling.

- C symbol: `terlan_ndarray_v1_chebder_axis`
- NativeBoundary operation: `ndarray.array.chebder_axis`
- Ownership: `opaque_handle`

## `chebint`

Integrates Chebyshev coefficient lanes along an axis with constants, a lower bound, and scaling.

- C symbol: `terlan_ndarray_v1_chebint_axis`
- NativeBoundary operation: `ndarray.array.chebint_axis`
- Ownership: `opaque_handle`

## `chebval`

Evaluates Chebyshev coefficient tensors with Cartesian or broadcast semantics.

- C symbol: `terlan_ndarray_v1_chebval_tensor`
- NativeBoundary operation: `ndarray.array.chebval_tensor`
- Ownership: `opaque_handle`

## `chebfit`

Fits selected Chebyshev degrees to scalar or matrix-valued samples.

- C symbol: `terlan_ndarray_v1_chebfit_degrees`
- NativeBoundary operation: `ndarray.array.chebfit_degrees`
- Ownership: `opaque_handle`

## `chebfit`

Fits selected Chebyshev degrees using sample weights.

- C symbol: `terlan_ndarray_v1_chebfit_weighted_degrees`
- NativeBoundary operation: `ndarray.array.chebfit_weighted_degrees`
- Ownership: `opaque_handle`
