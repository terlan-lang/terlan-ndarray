# ndarray.ArrowArray

Official Apache Arrow C Data exports owned by the ndarray package.

## `length`

Returns the flattened Arrow element count.

- C symbol: `terlan_ndarray_v1_arrow_length`
- NativeBoundary operation: `ndarray.arrow_array.length`
- Ownership: `value`

## `null_count`

Returns zero under the ndarray no-null Arrow policy.

- C symbol: `terlan_ndarray_v1_arrow_null_count`
- NativeBoundary operation: `ndarray.arrow_array.null_count`
- Ownership: `value`

## `dispose`

Releases the Arrow C Data callbacks exactly once.

- C symbol: `terlan_ndarray_v1_arrow_delete`
- NativeBoundary operation: `ndarray.arrow_array.dispose`
- Ownership: `dispose_handle`
