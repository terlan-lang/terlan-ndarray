TERLC ?= ../terlan/target/debug/terlc
MANIFEST := bindings/ndarray-c.json
GENERATED := generated

.PHONY: generate abi-check package-check operations-check clean

generate:
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	rm -rf "$(GENERATED)"
	"$(TERLC)" bind c --manifest "$(MANIFEST)" --out "$(GENERATED)"

abi-check: export RUSTFLAGS := -D warnings
abi-check:
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	python3 scripts/check-abi.py --terlc "$(TERLC)" --manifest "$(MANIFEST)"

package-check: export RUSTFLAGS := -D warnings
package-check:
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	python3 scripts/check-package.py --terlc "$(TERLC)"

operations-check: package-check

clean:
	rm -rf "$(GENERATED)"
