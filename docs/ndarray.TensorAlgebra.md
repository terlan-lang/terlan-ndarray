# ndarray.TensorAlgebra

General tensor contraction, Einstein summation, Kronecker products, and cross products.

## `tensordot`

Contracts two tensors over paired normalized axis lists.

- C symbol: `terlan_ndarray_v1_tensordot`
- NativeBoundary operation: `ndarray.array.tensordot`
- Ownership: `opaque_handle`

## `kron`

Computes a NumPy-compatible rank-padded Kronecker product.

- C symbol: `terlan_ndarray_v1_kron`
- NativeBoundary operation: `ndarray.array.kron`
- Ownership: `opaque_handle`

## `cross`

Computes a three-component vector cross product along one normalized axis.

- C symbol: `terlan_ndarray_v1_cross`
- NativeBoundary operation: `ndarray.array.cross`
- Ownership: `opaque_handle`

## `einsum`

Executes an Einstein summation expression over one or more arrays.

- C symbol: `terlan_ndarray_v1_einsum`
- NativeBoundary operation: `ndarray.array.einsum`
- Ownership: `opaque_handle`

## `einsum_into`

Executes Einstein summation and atomically writes the result into an exactly shaped destination.

- C symbol: `terlan_ndarray_v1_einsum_into`
- NativeBoundary operation: `ndarray.array.einsum_into`
- Ownership: `value`

## `einsum_ordered`

Executes Einstein summation using an explicit C, Fortran, keep, or any output-order policy.

- C symbol: `terlan_ndarray_v1_einsum_ordered`
- NativeBoundary operation: `ndarray.array.einsum_ordered`
- Ownership: `opaque_handle`
