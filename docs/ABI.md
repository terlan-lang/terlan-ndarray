# terlan-ndarray ABI Version 1

## Representation

- Handles are opaque and uniquely owned by the generated adapter.
- Shape dimensions, rank, element counts, and Int64 data use signed `int64_t`.
- Bool data uses one-byte `uint8_t` values restricted to `0` and `1`.
- Float64 data uses C `double` and IEEE-754 binary64.
- Arrays are CPU-owned, contiguous, row-major, and have byte offset zero.
- Rank-zero shape `[]` is a scalar containing exactly one element.
- Any zero dimension produces a zero-element array.
- Public constructors copy values and shape before returning.
- Borrowed shape output is copied immediately by the generated adapter.

The native struct layout is private. Only symbols prefixed
`terlan_ndarray_v1_` belong to ABI version 1.

All public functions use the platform C calling convention declared by the
header. ABI v1 does not expose Rust layout, C++ types, variadic functions, or
compiler-specific packed structures.

## Status Codes

| Code | Diagnostic family | Meaning |
| ---: | --- | --- |
| 0 | success | Operation completed |
| 100 | `ndarray.argument.invalid` | Required pointer or output slot is invalid |
| 101 | `ndarray.shape.negative_dimension` | Shape contains a negative dimension |
| 102 | `ndarray.shape.element_count_mismatch` | Shape product differs from value count |
| 103 | `ndarray.shape.overflow` | Element or allocation size overflowed |
| 104 | `ndarray.allocation.failed` | Native allocation failed |
| 105 | `ndarray.bool.invalid_byte` | Bool input contained a value other than 0 or 1 |

The generated Terlan boundary translates statuses into typed errors. Raw
native pointers and status integers are not part of the eventual public
high-level array API.

## Threading And Ownership

ABI v1 resources are thread-confined. Calls must not race on one handle.
Disposal consumes the handle exactly once; stale and double disposal are
rejected by `NativeBoundary` before dispatch.

DLPack, Arrow, BLAS provider selection, views, shared storage, and devices
other than CPU are not part of ABI v1 ND0. They require separately versioned
contracts before exposure.
