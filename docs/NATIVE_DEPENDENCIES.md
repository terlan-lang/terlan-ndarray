# Native Dependency Policy

The machine-readable source of truth is
[`native-dependencies.json`](../native-dependencies.json). Dependency updates
must change the immutable revision and verified SHA-256 together and must pass
`make abi-check`.

## DLPack

ABI v1 is designed against DLPack v1.3 at commit
`84d107bf416c6bab9ae68ad285876600d230490d`. The package pins the upstream
`include/dlpack/dlpack.h` file directly, records its Apache-2.0 license and
SHA-256, and validates its versioned ownership contract through the native and
cross-package DLPack gates.

## CBLAS

The default CPU provider is OpenBLAS v0.3.33 at commit
`62bcfb0dc9f1cfa685fc04135c50e2780c303137`, acquired from the immutable source
archive in the dependency lock. The package resolver requires HTTPS, verifies
the archive SHA-256 before extraction, rejects unsafe archive entries, and
stores the completed build under the content-addressed
`~/.cache/terlan/native/openblas/` cache. It builds a static CBLAS provider with
OpenBLAS dynamic CPU dispatch, so the validated x86-64 artifact is not tied to
the build machine's CPU generation. Ordinary compiler builds do not link BLAS.

Source validation requires `cc`, `curl`, GNU Make, and `pkg-config`. It does not
require an OpenBLAS installation. Repeated runs reuse the verified cache.

The supported CBLAS contract is LP64:

- dimensions and leading dimensions passed to CBLAS are signed 32-bit values;
- every Terlan dimension is range-checked before conversion;
- a provider reporting `USE64BITINT` is ILP64 and is rejected;
- the provider probe must compile against `cblas.h`, verify the CBLAS integer
  width, and record `openblas_get_config()` before any numerical gate passes.

The explicit system override is selected with
`TERLAN_NDARRAY_BLAS_PROVIDER=system`. `TERLAN_NDARRAY_OPENBLAS_ROOT` must name
an installation containing `include/cblas.h` and a supported OpenBLAS library.
The system provider is held to the same LP64 probe and version report as the
default provider; discovery by an unrelated library on the host is forbidden.

The validated 0.0.7 baseline host is `x86_64-unknown-linux-gnu`. macOS,
Windows, AArch64, and other hosts remain unsupported until their acquisition,
linkage, sanitizer, and package-consumer paths pass in release CI. The
machine-readable host list is intentionally narrower than platforms supported
by OpenBLAS itself.
