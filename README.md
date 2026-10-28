# terlan-ndarray

Native, Python-independent n-dimensional arrays for Terlan.

The package owns contiguous row-major CPU arrays behind opaque
`NativeBoundary` handles. Its generated stable C ABI is the shared numerical
interchange foundation for Terlan packages such as Polars and PyTorch; the
compiler contains no ndarray-specific lowering or runtime branch.

## Status

The CPU package implements owned Boolean, signed and unsigned integer,
Float16/32/64, and Complex64/128 arrays with checked layout metadata, copied
readback, generation-safe handles, and deterministic disposal. It provides
views and advanced indexing, NumPy-compatible broadcasting and explicit dtype
conversion, reductions and statistics, linear algebra, FFTs, polynomial
operations, and general tensor contraction. Versioned DLPack exchange supports
independently owned transfer to PyTorch, including Polars-to-ndarray
transformation flows.

The validated 0.0.7 host is 64-bit GNU/Linux on x86-64. Other operating
systems and architectures remain unsupported until their complete package and
native-tooling lanes are green.

The source release gate builds the pinned OpenBLAS dependency into a verified
content-addressed cache. It requires `cc`, `curl`, GNU Make, and `pkg-config`,
but does not require a separately installed OpenBLAS package.

Run the focused ABI, lifecycle, operation, BLAS, DLPack, and quality gates with:

```text
make abi-check
make package-check
make operations-check
make blas-check
make dlpack-check
make quality-check
```

Run the complete deterministic and cross-package closure gate with:

```text
make release-check
```

The gates write checked deterministic ABI, package, BLAS, DLPack, quality, and
integration results under [`reports/`](reports/).

The compiler is expected at `../terlan/target/debug/terlc`; override `TERLC`
when using another build.

See [docs/ABI.md](docs/ABI.md),
[docs/NATIVE_DEPENDENCIES.md](docs/NATIVE_DEPENDENCIES.md), and the workspace
[`ROADMAP_NDARRAY.md`](../docs/roadmap/ROADMAP_NDARRAY.md).
