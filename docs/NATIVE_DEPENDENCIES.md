# Native Dependency Policy

The machine-readable source of truth is
[`native-dependencies.json`](../native-dependencies.json). Dependency updates
must change the immutable revision and verified SHA-256 together and must pass
`make abi-check`.

## DLPack

ABI v1 is designed against DLPack v1.3 at commit
`84d107bf416c6bab9ae68ad285876600d230490d`. The package pins the upstream
`include/dlpack/dlpack.h` file directly and records its Apache-2.0 license and
SHA-256. ND0 does not compile or expose DLPack; ND4 will acquire this exact
header and implement the versioned ownership contract.

## CBLAS

The default CPU provider is OpenBLAS v0.3.33 at commit
`62bcfb0dc9f1cfa685fc04135c50e2780c303137`, acquired from the immutable source
archive in the dependency lock. Ordinary compiler builds never download or
link it. ND3 owns cached acquisition, source building, and linkage.

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

The baseline host set is Linux GNU, macOS, and Windows MSVC on x86-64 and
AArch64. Other hosts must fail with a stable unsupported-host diagnostic until
their acquisition and linkage path is tested explicitly.
