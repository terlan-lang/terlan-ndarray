TERLC ?= ../terlan/target/debug/terlc
POLARS ?= ../terlan-polars
PYTORCH ?= ../terlan-pytorch
LIBTORCH ?= $(abspath $(PYTORCH))/vendor/libtorch-2.13.0+cpu/libtorch
MANIFEST := bindings/ndarray-c.json
GENERATED := generated

.PHONY: generate prepare-openblas abi-check package-check terlan-grouped-binding-check terlan-function-reference-check operations-check indexing-view-check broadcasting-promotion-check elementwise-reduction-check shape-composition-check creation-random-check linear-algebra-check serialization-arrow-check nd15-numpy-fixture-check nd18-numpy-fixture-check indexing-ordering-set-check statistics-numerical-check fft-check polynomial-numerical-algebra-check numpy-api-inventory-report numpy-api-inventory-check numpy-test-parity-report numpy-test-parity-check blas-check dlpack-check quality-check release-check feature-complete-release-check clean

generate:
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	rm -rf "$(GENERATED)"
	"$(TERLC)" bind c --manifest "$(MANIFEST)" --out "$(GENERATED)"

prepare-openblas:
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	"$(TERLC)" run scripts/PrepareOpenBlas.terls

abi-check: export RUSTFLAGS := -D warnings
abi-check: generate
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	"$(TERLC)" run scripts/CheckAbi.terls -- --terlc "$(TERLC)" --manifest "$(MANIFEST)"

package-check: export RUSTFLAGS := -D warnings
package-check: abi-check terlan-grouped-binding-check terlan-function-reference-check
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	"$(TERLC)" run scripts/CheckPackage.terls -- --terlc "$(TERLC)"

terlan-grouped-binding-check:
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	@"$(TERLC)" lint --only TL0009 src
	@"$(TERLC)" lint --only TL0009 tests

terlan-function-reference-check:
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	@"$(TERLC)" lint --only TL0010 src
	@"$(TERLC)" lint --only TL0010 tests

operations-check: package-check

indexing-view-check: package-check

broadcasting-promotion-check: package-check quality-check

elementwise-reduction-check: package-check quality-check

shape-composition-check: package-check quality-check

creation-random-check: package-check quality-check

linear-algebra-check: package-check quality-check

serialization-arrow-check: package-check quality-check

nd15-numpy-fixture-check:
	"$(TERLC)" run scripts/CheckNumpyFixtures.terls -- --nd15

indexing-ordering-set-check: nd15-numpy-fixture-check package-check quality-check

statistics-numerical-check: package-check quality-check

fft-check: package-check quality-check

nd18-numpy-fixture-check:
	"$(TERLC)" run scripts/CheckNumpyFixtures.terls -- --nd18

polynomial-numerical-algebra-check: nd18-numpy-fixture-check package-check quality-check

numpy-api-inventory-report:
	"$(TERLC)" run scripts/CheckNumpyApi.terls -- --allow-missing

numpy-api-inventory-check:
	"$(TERLC)" run scripts/CheckNumpyApi.terls

numpy-test-parity-report:
	"$(TERLC)" run scripts/CheckNumpyTestParity.terls -- --allow-uncovered

numpy-test-parity-check:
	"$(TERLC)" run scripts/CheckNumpyTestParity.terls

blas-check: generate terlan-grouped-binding-check terlan-function-reference-check
	"$(TERLC)" run scripts/CheckBlas.terls

dlpack-check: generate terlan-grouped-binding-check terlan-function-reference-check
	"$(TERLC)" run scripts/CheckDlpack.terls -- --terlc "$(TERLC)"

quality-check: generate
	"$(TERLC)" run scripts/CheckQuality.terls -- --terlc "$(TERLC)"

release-check:
	@test -f "$(POLARS)/terlan.toml" || { echo "missing polars package: $(POLARS)" >&2; exit 1; }
	@test -f "$(PYTORCH)/terlan.toml" || { echo "missing pytorch package: $(PYTORCH)" >&2; exit 1; }
	@test -f "$(LIBTORCH)/build-version" || { echo "missing released LibTorch distribution: $(LIBTORCH)" >&2; exit 1; }
	"$(TERLC)" run scripts/CheckReleaseReports.terls -- \
		--terlc "$(abspath $(TERLC))" --polars "$(abspath $(POLARS))" \
		--pytorch "$(abspath $(PYTORCH))" --libtorch "$(abspath $(LIBTORCH))"

feature-complete-release-check: numpy-api-inventory-check release-check
	"$(TERLC)" run scripts/CheckFeatureComplete.terls -- \
		--cuda "$(abspath ../terlan-cuda)" \
		--pytorch "$(abspath $(PYTORCH))" \
		--polars "$(abspath $(POLARS))" \
		--opencv "$(abspath ../terlan-opencv)"

clean:
	rm -rf "$(GENERATED)"
