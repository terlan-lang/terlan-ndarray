TERLC ?= ../terlan/target/debug/terlc
MANIFEST := bindings/ndarray-c.json
GENERATED := generated

.PHONY: generate abi-check clean

generate:
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	rm -rf "$(GENERATED)"
	"$(TERLC)" bind c --manifest "$(MANIFEST)" --out "$(GENERATED)"

abi-check: export RUSTFLAGS := -D warnings
abi-check:
	@test -x "$(TERLC)" || { echo "missing terlc: $(TERLC)" >&2; exit 1; }
	python3 scripts/check-abi.py --terlc "$(TERLC)" --manifest "$(MANIFEST)"

clean:
	rm -rf "$(GENERATED)"
