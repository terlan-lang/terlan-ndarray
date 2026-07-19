# ndarray.Array

Generated ABI-v1 bindings for owned contiguous CPU arrays.

## `from_bools`

Copies boolean values and shape into owned CPU storage.

- C symbol: `terlan_ndarray_v1_from_bools`
- NativeBoundary operation: `ndarray.array.from_bools`
- Ownership: `opaque_handle`

## `from_ints`

Copies Int64 values and shape into owned CPU storage.

- C symbol: `terlan_ndarray_v1_from_ints`
- NativeBoundary operation: `ndarray.array.from_ints`
- Ownership: `opaque_handle`

## `from_floats`

Copies Float64 values and shape into owned CPU storage.

- C symbol: `terlan_ndarray_v1_from_floats`
- NativeBoundary operation: `ndarray.array.from_floats`
- Ownership: `opaque_handle`

## `rank`

Returns array rank.

- C symbol: `terlan_ndarray_v1_rank`
- NativeBoundary operation: `ndarray.array.rank`
- Ownership: `borrowed_handle`

## `shape`

Copies array shape into a Terlan list.

- C symbol: `terlan_ndarray_v1_shape`
- NativeBoundary operation: `ndarray.array.shape`
- Ownership: `value`

## `numel`

Returns logical element count.

- C symbol: `terlan_ndarray_v1_numel`
- NativeBoundary operation: `ndarray.array.numel`
- Ownership: `borrowed_handle`

## `strides`

Copies canonical row-major element strides.

- C symbol: `terlan_ndarray_v1_strides`
- NativeBoundary operation: `ndarray.array.strides`
- Ownership: `value`

## `byte_count`

Returns the checked contiguous storage byte count.

- C symbol: `terlan_ndarray_v1_byte_count`
- NativeBoundary operation: `ndarray.array.byte_count`
- Ownership: `borrowed_handle`

## `dtype_code`

Returns the ABI-v1 dtype code.

- C symbol: `terlan_ndarray_v1_dtype`
- NativeBoundary operation: `ndarray.array.dtype_code`
- Ownership: `borrowed_handle`

## `contiguous_code`

Returns one for the ABI-v1 contiguous layout.

- C symbol: `terlan_ndarray_v1_is_contiguous`
- NativeBoundary operation: `ndarray.array.contiguous_code`
- Ownership: `borrowed_handle`

## `device_type_code`

Returns the ABI-v1 CPU device code.

- C symbol: `terlan_ndarray_v1_device_type`
- NativeBoundary operation: `ndarray.array.device_type_code`
- Ownership: `borrowed_handle`

## `to_bools`

Returns an owned copy of Bool values.

- C symbol: `terlan_ndarray_v1_to_bools`
- NativeBoundary operation: `ndarray.array.to_bools`
- Ownership: `value`

## `to_ints`

Returns an owned copy of Int64 values.

- C symbol: `terlan_ndarray_v1_to_ints`
- NativeBoundary operation: `ndarray.array.to_ints`
- Ownership: `value`

## `to_floats`

Returns an owned copy of Float64 values.

- C symbol: `terlan_ndarray_v1_to_floats`
- NativeBoundary operation: `ndarray.array.to_floats`
- Ownership: `value`

## `reshape`

Copies an array into an independently owned contiguous shape.

- C symbol: `terlan_ndarray_v1_reshape`
- NativeBoundary operation: `ndarray.array.reshape`
- Ownership: `opaque_handle`

## `transpose`

Copies an array with two axes exchanged into contiguous storage.

- C symbol: `terlan_ndarray_v1_transpose`
- NativeBoundary operation: `ndarray.array.transpose`
- Ownership: `opaque_handle`

## `add`

Adds exact-shape numeric arrays without broadcasting.

- C symbol: `terlan_ndarray_v1_add`
- NativeBoundary operation: `ndarray.array.add`
- Ownership: `opaque_handle`

## `subtract`

Subtracts exact-shape numeric arrays without broadcasting.

- C symbol: `terlan_ndarray_v1_subtract`
- NativeBoundary operation: `ndarray.array.subtract`
- Ownership: `opaque_handle`

## `multiply`

Multiplies exact-shape numeric arrays without broadcasting.

- C symbol: `terlan_ndarray_v1_multiply`
- NativeBoundary operation: `ndarray.array.multiply`
- Ownership: `opaque_handle`

## `sum`

Reduces normalized axes; an empty axis list reduces all axes.

- C symbol: `terlan_ndarray_v1_sum`
- NativeBoundary operation: `ndarray.array.sum`
- Ownership: `opaque_handle`

## `dispose`

Disposes an owned array exactly once.

- C symbol: `terlan_ndarray_v1_delete`
- NativeBoundary operation: `ndarray.array.dispose`
- Ownership: `dispose_handle`

## `live_count`

Returns live native allocations for lifecycle auditing.

- C symbol: `terlan_ndarray_v1_live_count`
- NativeBoundary operation: `ndarray.array.live_count`
- Ownership: `value`

