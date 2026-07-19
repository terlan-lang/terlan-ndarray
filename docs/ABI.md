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
- Borrowed shape and stride output is copied immediately by the generated
  adapter.
- Typed readback returns independently owned copies that the generated adapter
  releases after conversion to Terlan lists.

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
| 106 | `ndarray.dtype.mismatch` | Typed readback does not match the array dtype |
| 107 | `ndarray.handle.disposed` | Native code received a disposed handle |
| 108 | `ndarray.shape.mismatch` | Elementwise operands have different exact shapes |
| 109 | `ndarray.dtype.unsupported` | The operation does not support the array dtype |
| 110 | `ndarray.axis.out_of_range` | A normalized axis is outside the array rank |
| 111 | `ndarray.axis.duplicate` | A reduction names one normalized axis more than once |
| 112 | `ndarray.arithmetic.overflow` | Int64 arithmetic or reduction overflowed |

The generated Terlan boundary translates statuses into typed errors. Raw
native pointers and status integers are not part of the eventual public
high-level array API.

## ND2 Operation Semantics

- `reshape` preserves the exact element count and copies data into independent
  storage. It does not infer dimensions.
- `transpose` exchanges two normalized axes and emits canonical contiguous
  row-major output. Negative axes count from the end. Scalar transpose accepts
  the identity pair `(0, 0)`; rank-one transpose accepts the only normalized
  axis pair.
- `add`, `subtract`, and `multiply` require identical dtype, rank, and shape.
  Bool arithmetic, implicit conversion, scalar expansion, singleton expansion,
  and broadcasting are rejected.
- Int64 operations are checked and return status 112 instead of wrapping or
  invoking C signed-overflow behavior.
- Float64 operations preserve IEEE-754 behavior, including NaN and infinity.
- `sum` normalizes negative axes, rejects duplicates, and uses a zero identity
  for empty reductions. An empty axis list reduces all axes. `keep_dims` must be
  zero or one.
- Float64 reduction accumulates sequentially in canonical row-major order.
  Int64 reduction rejects overflow.
- Every operation returns independently owned storage. Sources remain valid and
  may be disposed before or after their results.

## Threading And Ownership

ABI v1 resources are thread-confined. Calls must not race on one handle.
Disposal consumes the handle exactly once; stale and double disposal are
rejected by `NativeBoundary` before dispatch. Reused handle-table slots receive
a monotonically increasing generation, so an old Terlan value cannot resolve
to a new native array.

DLPack, Arrow, BLAS provider selection, views, shared storage, broadcasting,
and devices other than CPU are not part of the completed ND2 surface. They
require separately versioned contracts before exposure.
