# ndarray.Polynomial

Polynomial evaluation, fitting, arithmetic, roots, calculus, and basis conversion.

## `polyval`

Evaluates low-to-high power-basis coefficients at every coordinate.

- C symbol: `terlan_ndarray_v1_polyval`
- NativeBoundary operation: `ndarray.array.polyval`
- Ownership: `opaque_handle`

## `polyroots`

Returns all complex roots of a finite nonconstant power-basis polynomial.

- C symbol: `terlan_ndarray_v1_polyroots`
- NativeBoundary operation: `ndarray.array.polyroots`
- Ownership: `opaque_handle`

## `polyfit`

Fits low-to-high power-basis coefficients by finite SVD least squares.

- C symbol: `terlan_ndarray_v1_polyfit`
- NativeBoundary operation: `ndarray.array.polyfit`
- Ownership: `opaque_handle`

## `polyadd`

Adds two low-to-high polynomial coefficient vectors.

- C symbol: `terlan_ndarray_v1_polyadd`
- NativeBoundary operation: `ndarray.array.polyadd`
- Ownership: `opaque_handle`

## `polysub`

Subtracts two low-to-high polynomial coefficient vectors.

- C symbol: `terlan_ndarray_v1_polysub`
- NativeBoundary operation: `ndarray.array.polysub`
- Ownership: `opaque_handle`

## `polymul`

Multiplies two low-to-high polynomial coefficient vectors.

- C symbol: `terlan_ndarray_v1_polymul`
- NativeBoundary operation: `ndarray.array.polymul`
- Ownership: `opaque_handle`

## `polydiv_all`

Returns polynomial quotient and remainder as two owned arrays.

- C symbol: `terlan_ndarray_v1_polydiv_all`
- NativeBoundary operation: `ndarray.array.polydiv_all`
- Ownership: `opaque_handle`

## `polyder`

Differentiates low-to-high polynomial coefficients by a nonnegative order.

- C symbol: `terlan_ndarray_v1_polyder`
- NativeBoundary operation: `ndarray.array.polyder`
- Ownership: `opaque_handle`

## `polyint`

Integrates low-to-high polynomial coefficients with a repeated constant.

- C symbol: `terlan_ndarray_v1_polyint`
- NativeBoundary operation: `ndarray.array.polyint`
- Ownership: `opaque_handle`

## `polycompanion`

Returns the power-basis companion matrix used for root finding.

- C symbol: `terlan_ndarray_v1_polycompanion`
- NativeBoundary operation: `ndarray.array.polycompanion`
- Ownership: `opaque_handle`

## `polyconvert`

Converts coefficients among power, Chebyshev, and Legendre bases over a finite domain and window.

- C symbol: `terlan_ndarray_v1_polyconvert`
- NativeBoundary operation: `ndarray.array.polyconvert`
- Ownership: `opaque_handle`

## `polyfromroots`

Returns low-to-high coefficients for a monic polynomial with the supplied roots.

- C symbol: `terlan_ndarray_v1_polyfromroots`
- NativeBoundary operation: `ndarray.array.polyfromroots`
- Ownership: `opaque_handle`

## `polyline`

Returns coefficients for offset plus scale times x.

- C symbol: `terlan_ndarray_v1_polyline`
- NativeBoundary operation: `ndarray.array.polyline`
- Ownership: `opaque_handle`

## `polymulx`

Multiplies a coefficient vector by x.

- C symbol: `terlan_ndarray_v1_polymulx`
- NativeBoundary operation: `ndarray.array.polymulx`
- Ownership: `opaque_handle`

## `polypow`

Raises a coefficient vector to a nonnegative integer power.

- C symbol: `terlan_ndarray_v1_polypow`
- NativeBoundary operation: `ndarray.array.polypow`
- Ownership: `opaque_handle`

## `polytrim`

Removes redundant leading coefficients under a tolerance.

- C symbol: `terlan_ndarray_v1_polytrim`
- NativeBoundary operation: `ndarray.array.polytrim`
- Ownership: `opaque_handle`

## `polyvalfromroots`

Evaluates a monic polynomial directly from roots.

- C symbol: `terlan_ndarray_v1_polyvalfromroots`
- NativeBoundary operation: `ndarray.array.polyvalfromroots`
- Ownership: `opaque_handle`

## `polyder_axis`

Differentiates tensor coefficients along one axis with linear-variable scaling.

- C symbol: `terlan_ndarray_v1_polyder_axis`
- NativeBoundary operation: `ndarray.array.polyder_axis`
- Ownership: `opaque_handle`

## `polyint_axis`

Integrates tensor coefficients along one axis with per-order constants, a lower bound, and scaling.

- C symbol: `terlan_ndarray_v1_polyint_axis`
- NativeBoundary operation: `ndarray.array.polyint_axis`
- Ownership: `opaque_handle`

## `polyvalfromroots_tensor`

Evaluates root-defined polynomial tensors in Cartesian or broadcast mode.

- C symbol: `terlan_ndarray_v1_polyvalfromroots_tensor`
- NativeBoundary operation: `ndarray.array.polyvalfromroots_tensor`
- Ownership: `opaque_handle`

## `polyfit_degrees`

Fits selected power-basis degrees to vector or matrix observations.

- C symbol: `terlan_ndarray_v1_polyfit_degrees`
- NativeBoundary operation: `ndarray.array.polyfit_degrees`
- Ownership: `opaque_handle`

## `polyfit_weighted_degrees`

Fits selected power-basis degrees with one weight per observation.

- C symbol: `terlan_ndarray_v1_polyfit_weighted_degrees`
- NativeBoundary operation: `ndarray.array.polyfit_weighted_degrees`
- Ownership: `opaque_handle`

## `polyfit_covariance`

Fits contiguous power degrees and returns coefficients plus scaled or unscaled covariance.

- C symbol: `terlan_ndarray_v1_polyfit_covariance`
- NativeBoundary operation: `ndarray.array.polyfit_covariance`
- Ownership: `opaque_handle`

## `polyfit_weighted_covariance`

Fits weighted contiguous power degrees and returns coefficients plus scaled or unscaled covariance.

- C symbol: `terlan_ndarray_v1_polyfit_weighted_covariance`
- NativeBoundary operation: `ndarray.array.polyfit_weighted_covariance`
- Ownership: `opaque_handle`

## `polyvander`

Builds a power Vandermonde matrix with columns from degree zero upward.

- C symbol: `terlan_ndarray_v1_polyvander`
- NativeBoundary operation: `ndarray.array.polyvander`
- Ownership: `opaque_handle`

## `polyval2d`

Evaluates a two-dimensional coefficient grid over broadcast coordinates.

- C symbol: `terlan_ndarray_v1_polyval2d`
- NativeBoundary operation: `ndarray.array.polyval2d`
- Ownership: `opaque_handle`

## `polyval3d`

Evaluates a three-dimensional coefficient grid over broadcast coordinates.

- C symbol: `terlan_ndarray_v1_polyval3d`
- NativeBoundary operation: `ndarray.array.polyval3d`
- Ownership: `opaque_handle`

## `polygrid2d`

Evaluates a two-dimensional coefficient grid over a Cartesian product.

- C symbol: `terlan_ndarray_v1_polygrid2d`
- NativeBoundary operation: `ndarray.array.polygrid2d`
- Ownership: `opaque_handle`

## `polygrid3d`

Evaluates a three-dimensional coefficient grid over a Cartesian product.

- C symbol: `terlan_ndarray_v1_polygrid3d`
- NativeBoundary operation: `ndarray.array.polygrid3d`
- Ownership: `opaque_handle`

## `polyvander2d`

Builds a two-dimensional broadcast power Vandermonde matrix.

- C symbol: `terlan_ndarray_v1_polyvander2d`
- NativeBoundary operation: `ndarray.array.polyvander2d`
- Ownership: `opaque_handle`

## `polyvander3d`

Builds a three-dimensional broadcast power Vandermonde matrix.

- C symbol: `terlan_ndarray_v1_polyvander3d`
- NativeBoundary operation: `ndarray.array.polyvander3d`
- Ownership: `opaque_handle`
