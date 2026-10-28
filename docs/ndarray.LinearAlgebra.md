# ndarray.LinearAlgebra

Matrix decompositions, condition analysis, powers, pseudoinverses, and tensor equation solving.

## `det`

Computes one determinant per square matrix while preserving leading batch dimensions.

- C symbol: `terlan_ndarray_v1_det`
- NativeBoundary operation: `ndarray.array.det`
- Ownership: `opaque_handle`

## `least_squares`

Solves a finite real least-squares system through an SVD pseudoinverse.

- C symbol: `terlan_ndarray_v1_least_squares`
- NativeBoundary operation: `ndarray.array.least_squares`
- Ownership: `opaque_handle`

## `cholesky`

Computes a lower-triangular Cholesky factor for a symmetric positive-definite matrix.

- C symbol: `terlan_ndarray_v1_cholesky`
- NativeBoundary operation: `ndarray.array.cholesky`
- Ownership: `opaque_handle`

## `svd`

Computes singular-value factors and returns U, singular values, and Vh.

- C symbol: `terlan_ndarray_v1_svd`
- NativeBoundary operation: `ndarray.array.svd`
- Ownership: `opaque_handle`

## `singular_values`

Computes singular values without materializing singular vectors.

- C symbol: `terlan_ndarray_v1_singular_values`
- NativeBoundary operation: `ndarray.array.singular_values`
- Ownership: `opaque_handle`

## `symmetric_eigen`

Computes a symmetric eigendecomposition and returns ascending eigenvalues followed by eigenvectors.

- C symbol: `terlan_ndarray_v1_symmetric_eigen`
- NativeBoundary operation: `ndarray.array.symmetric_eigen`
- Ownership: `opaque_handle`

## `cond`

Computes a matrix condition number for a selected spectral, column-sum, or row-sum norm.

- C symbol: `terlan_ndarray_v1_cond`
- NativeBoundary operation: `ndarray.array.cond`
- Ownership: `value`

## `eig`

Computes general complex eigenvalues and right eigenvectors.

- C symbol: `terlan_ndarray_v1_eig`
- NativeBoundary operation: `ndarray.array.eig`
- Ownership: `opaque_handle`

## `condition`

Computes one condition number per matrix while preserving leading batch dimensions.

- C symbol: `terlan_ndarray_v1_condition`
- NativeBoundary operation: `ndarray.array.condition`
- Ownership: `opaque_handle`

## `eigvals`

Computes general complex eigenvalues without materializing eigenvectors.

- C symbol: `terlan_ndarray_v1_eigvals`
- NativeBoundary operation: `ndarray.array.eigvals`
- Ownership: `opaque_handle`

## `eigvalsh`

Computes ascending eigenvalues of a finite Hermitian matrix.

- C symbol: `terlan_ndarray_v1_eigvalsh`
- NativeBoundary operation: `ndarray.array.eigvalsh`
- Ownership: `opaque_handle`

## `matrix_power`

Raises a square matrix to an integer power by checked exponentiation.

- C symbol: `terlan_ndarray_v1_matrix_power`
- NativeBoundary operation: `ndarray.array.matrix_power`
- Ownership: `opaque_handle`

## `multi_dot`

Multiplies a chain of arrays using package matmul semantics.

- C symbol: `terlan_ndarray_v1_multi_dot`
- NativeBoundary operation: `ndarray.array.multi_dot`
- Ownership: `opaque_handle`

## `multi_dot_into`

Multiplies a chain and atomically writes the result into a validated destination.

- C symbol: `terlan_ndarray_v1_multi_dot_into`
- NativeBoundary operation: `ndarray.array.multi_dot_into`
- Ownership: `value`

## `norm`

Computes the default Euclidean vector or Frobenius matrix norm.

- C symbol: `terlan_ndarray_v1_norm`
- NativeBoundary operation: `ndarray.array.norm`
- Ownership: `value`

## `pinv`

Computes an SVD pseudoinverse with an explicit singular-value cutoff.

- C symbol: `terlan_ndarray_v1_pinv`
- NativeBoundary operation: `ndarray.array.pinv`
- Ownership: `opaque_handle`

## `slogdet`

Computes determinant sign and logarithmic absolute determinant as scalar arrays.

- C symbol: `terlan_ndarray_v1_slogdet`
- NativeBoundary operation: `ndarray.array.slogdet`
- Ownership: `opaque_handle`

## `tensorinv`

Computes the inverse of a tensor interpreted as a square matrix split at an index.

- C symbol: `terlan_ndarray_v1_tensorinv`
- NativeBoundary operation: `ndarray.array.tensorinv`
- Ownership: `opaque_handle`

## `tensorsolve`

Solves a tensor equation after optionally moving selected coefficient axes to the end.

- C symbol: `terlan_ndarray_v1_tensorsolve`
- NativeBoundary operation: `ndarray.array.tensorsolve`
- Ownership: `opaque_handle`
