# terlan-ndarray

Native, Python-independent n-dimensional arrays for Terlan.

The package owns contiguous row-major CPU arrays behind opaque
`NativeBoundary` handles. Its generated stable C ABI is the shared numerical
interchange foundation for Terlan packages such as Polars and PyTorch; the
compiler contains no ndarray-specific lowering or runtime branch.

## Status

`ND0` freezes and validates ABI version 1. `ND1` implements owned Bool, Int64,
and Float64 CPU arrays with checked layout metadata, copied readback,
generation-safe handles, and deterministic disposal. Public numerical
operations, CBLAS, DLPack, and package integrations are implemented by later
roadmap milestones.

Run the ABI and owned-lifecycle gates with:

```text
make abi-check
make package-check
```

The gates write checked deterministic results to
[`reports/ndarray-abi-report.json`](reports/ndarray-abi-report.json) and
[`reports/ndarray-package-report.json`](reports/ndarray-package-report.json).

The compiler is expected at `../terlan/target/debug/terlc`; override `TERLC`
when using another build.

See [docs/ABI.md](docs/ABI.md),
[docs/NATIVE_DEPENDENCIES.md](docs/NATIVE_DEPENDENCIES.md), and the workspace
[`ROADMAP_NDARRAY.md`](../docs/roadmap/ROADMAP_NDARRAY.md).
