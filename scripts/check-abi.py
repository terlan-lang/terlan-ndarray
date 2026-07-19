#!/usr/bin/env python3

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def run(command, *, cwd, env=None, expect_success=True):
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if not expect_success and completed.returncode == 0:
        raise RuntimeError(f"command unexpectedly passed: {' '.join(command)}")
    if completed.returncode != 0 and expect_success:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def tree_hashes(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def symbol(manifest, symbol_id):
    return next(
        item for item in manifest["c_metadata"]["symbols"] if item["id"] == symbol_id
    )


def remove_parameter(manifest, symbol_id, name):
    target = symbol(manifest, symbol_id)
    target["parameters"] = [
        parameter for parameter in target["parameters"] if parameter["name"] != name
    ]


def mutate_missing_length(manifest):
    remove_parameter(manifest, "function.from_floats", "rank")


def mutate_mutable_borrow(manifest):
    symbol(manifest, "function.from_ints")["parameters"][0]["c_type"] = "int64_t *"


def mutate_ambiguous_ownership(manifest):
    symbol(manifest, "function.from_ints")["parameters"][0]["ownership"] = (
        "transfer_full"
    )


def mutate_missing_destructor(manifest):
    del symbol(manifest, "record.ndarray")["destructor_symbol"]


def mutate_unsupported_scalar_width(manifest):
    symbol(manifest, "function.from_floats")["parameters"][0]["c_type"] = (
        "const float *"
    )


def mutate_duplicate_symbol(manifest):
    manifest["c_metadata"]["symbols"].append(
        copy.deepcopy(symbol(manifest, "function.rank"))
    )


def mutate_missing_abi_version(manifest):
    manifest["c_metadata"]["abi_version"] = 0


ADVERSARIAL_CASES = (
    ("missing_length", "native_bindgen.c_input_array_contract", mutate_missing_length),
    ("mutable_borrow", "native_bindgen.c_input_array_contract", mutate_mutable_borrow),
    (
        "ambiguous_ownership",
        "native_bindgen.c_input_array_contract",
        mutate_ambiguous_ownership,
    ),
    ("missing_destructor", "native_bindgen.c_missing_destructor", mutate_missing_destructor),
    (
        "unsupported_scalar_width",
        "native_bindgen.c_input_array_contract",
        mutate_unsupported_scalar_width,
    ),
    (
        "duplicate_symbol",
        "duplicate structured C symbol id",
        mutate_duplicate_symbol,
    ),
    ("missing_abi_version", "native_bindgen.c_abi_version_missing", mutate_missing_abi_version),
)

EXPECTED_DEPENDENCIES = {
    "dlpack": {
        "license": "Apache-2.0",
        "release": "v1.3",
        "revision": "84d107bf416c6bab9ae68ad285876600d230490d",
        "sha256": "62052bef24bd69bb7b52e95428f9e99bd7951783850c36c5f2a41e71fd63c2fc",
    },
    "openblas": {
        "cblas_integer_model": "LP64",
        "default_provider": "pinned_source",
        "license": "BSD-3-Clause",
        "release": "v0.3.33",
        "revision": "62bcfb0dc9f1cfa685fc04135c50e2780c303137",
        "sha256": "91c894079dad77b2b51bee58f6f999dbd9db0419dd6a06dff93004767461ab86",
    },
}


def check_symbol_prefixes(manifest):
    bad = [
        item["c_name"]
        for item in manifest["c_metadata"]["symbols"]
        if item["kind"] == "function"
        and item["status"] == "bind"
        and not item["c_name"].startswith("terlan_ndarray_v1_")
    ]
    if bad:
        raise RuntimeError(f"ABI-v1 symbols lack versioned prefix: {bad}")


def check_header_symbols(repository, manifest):
    header = (repository / "bindings/include/terlan_ndarray.h").read_text()
    declared = set(re.findall(r"\b(terlan_ndarray_v1_[a-z0-9_]+)\s*\(", header))
    bound = {
        item["c_name"]
        for item in manifest["c_metadata"]["symbols"]
        if item["kind"] == "function" and item["status"] == "bind"
    }
    if declared != bound:
        raise RuntimeError(
            f"header/metadata symbol mismatch: missing={sorted(bound - declared)}, "
            f"extra={sorted(declared - bound)}"
        )


def check_compiler_independence(repository):
    source_root = repository.parent / "terlan/crates/terlan/src"
    pattern = re.compile(
        r"terlan[-_]?ndarray|ndarray\.|dlpack|openblas|cblas", re.IGNORECASE
    )
    matches = []
    for path in sorted(source_root.rglob("*.rs")):
        for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if pattern.search(line):
                matches.append(f"{path.relative_to(source_root)}:{line_number}")
    if matches:
        raise RuntimeError(
            "ndarray-specific compiler source is forbidden: " + ", ".join(matches[:20])
        )


def check_dependency_lock(repository):
    lock_path = repository / "native-dependencies.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("schema") != "terlan.ndarray.native-dependencies.v1":
        raise RuntimeError("native dependency lock has the wrong schema")
    dependencies = lock.get("dependencies", {})
    for name, expected in EXPECTED_DEPENDENCIES.items():
        dependency = dependencies.get(name, {})
        for field, value in expected.items():
            if dependency.get(field) != value:
                raise RuntimeError(
                    f"native dependency {name}.{field} must equal {value!r}"
                )
    dlpack = dependencies["dlpack"]
    if dlpack["revision"] not in dlpack["source_url"]:
        raise RuntimeError("DLPack source URL must contain its immutable revision")
    openblas = dependencies["openblas"]
    if openblas["revision"] not in openblas["archive_url"]:
        raise RuntimeError("OpenBLAS archive URL must contain its immutable revision")
    system = openblas.get("system_override", {})
    if system.get("provider_env") != "TERLAN_NDARRAY_BLAS_PROVIDER=system":
        raise RuntimeError("OpenBLAS system override must be explicit")
    if system.get("root_env") != "TERLAN_NDARRAY_OPENBLAS_ROOT":
        raise RuntimeError("OpenBLAS system root environment variable is not frozen")
    if not lock.get("supported_hosts"):
        raise RuntimeError("native dependency lock must list supported hosts")
    return lock


def materialize(repository, generated):
    for name in ("src", "native", ".terlan"):
        destination = repository / name
        if destination.exists():
            shutil.rmtree(destination)
        source = generated / name
        if source.exists():
            shutil.copytree(source, destination)
    shutil.copy2(generated / "terlan.toml", repository / "terlan.toml")
    for name in ("native-binding-manifest.json", "skipped-symbols.json"):
        shutil.copy2(generated / "bindings" / name, repository / "bindings" / name)
    generated_docs = generated / "docs"
    if generated_docs.exists():
        for path in generated_docs.iterdir():
            if path.is_file():
                shutil.copy2(path, repository / "docs" / path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terlc", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    terlc = Path(args.terlc).resolve()
    manifest_path = (repository / args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    dependency_lock = check_dependency_lock(repository)
    check_symbol_prefixes(manifest)
    check_header_symbols(repository, manifest)
    check_compiler_independence(repository)

    with tempfile.TemporaryDirectory(prefix="terlan_ndarray_abi_") as temporary:
        temporary = Path(temporary)
        first = temporary / "first"
        second = temporary / "second"
        run(
            [str(terlc), "bind", "c", "--manifest", str(manifest_path), "--out", str(first)],
            cwd=repository,
        )
        run(
            [str(terlc), "bind", "c", "--manifest", str(manifest_path), "--out", str(second)],
            cwd=repository,
        )
        first_hashes = tree_hashes(first)
        second_hashes = tree_hashes(second)
        if first_hashes != second_hashes:
            raise RuntimeError("C ABI generation was not byte-deterministic")

        for name, diagnostic, mutate in ADVERSARIAL_CASES:
            variant = copy.deepcopy(manifest)
            mutate(variant)
            case_root = temporary / name
            case_root.mkdir()
            shutil.copytree(repository / "bindings/include", case_root / "include")
            for source in manifest["c_metadata"].get("sources", []):
                shutil.copy2(repository / "bindings" / source, case_root / source)
            for header in manifest["c_metadata"].get("headers", []):
                shutil.copy2(repository / "bindings" / header, case_root / header)
            variant_path = case_root / "ndarray-c.json"
            variant_path.write_text(json.dumps(variant, indent=2, sort_keys=True) + "\n")
            output = case_root / "output"
            completed = run(
                [str(terlc), "bind", "c", "--manifest", str(variant_path), "--out", str(output)],
                cwd=repository,
                expect_success=False,
            )
            combined = completed.stdout + completed.stderr
            if diagnostic not in combined:
                raise RuntimeError(
                    f"{name} produced the wrong diagnostic; expected {diagnostic!r}\n{combined}"
                )

        native_sources = [
            str(repository / "bindings" / source)
            for source in manifest["c_metadata"].get("sources", [])
        ]
        for test_source in (
            "abi_test.c",
            "operations_test.c",
            "operations_property_test.c",
        ):
            native_test = temporary / f"ndarray-{Path(test_source).stem}"
            run(
                [
                    os.environ.get("CC", "cc"),
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Wpedantic",
                    "-Werror",
                    *native_sources,
                    str(repository / "tests" / test_source),
                    "-lm",
                    "-o",
                    str(native_test),
                ],
                cwd=repository,
            )
            run([str(native_test)], cwd=repository)

        cargo_environment = os.environ.copy()
        cargo_environment["CARGO_TARGET_DIR"] = str(temporary / "cargo-target")
        run(
            [
                "cargo",
                "test",
                "--offline",
                "--manifest-path",
                str(first / "native/rust/Cargo.toml"),
            ],
            cwd=repository,
            env=cargo_environment,
        )
        run(
            [
                "cargo",
                "build",
                "--offline",
                "--manifest-path",
                str(first / "native/rust/Cargo.toml"),
                "--bin",
                "native-boundary-helper",
            ],
            cwd=repository,
            env=cargo_environment,
        )
        test_destination = first / "tests/ndarray/AbiTest.terl"
        test_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / "tests/ndarray/AbiTest.terl", test_destination)
        terlan_environment = os.environ.copy()
        terlan_environment["TERLAN_SAFE_NATIVE_PATH"] = str(
            temporary / "cargo-target/debug/native-boundary-helper"
        )
        run([str(terlc), "test", str(first / "tests")], cwd=repository, env=terlan_environment)

        generated = repository / "generated"
        if generated.exists():
            shutil.rmtree(generated)
        shutil.copytree(first, generated)
        report = {
            "abi_version": 1,
            "adversarial_cases": [name for name, _, _ in ADVERSARIAL_CASES],
            "dtypes": ["bool8", "float64", "int64"],
            "generated_files": first_hashes,
            "native_dependencies": dependency_lock,
            "package": "terlan-ndarray",
            "schema": "terlan.ndarray.abi-report.v1",
            "status": "passed",
        }
        report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        (generated / "ndarray-abi-report.json").write_text(report_text)
        reports = repository / "reports"
        reports.mkdir(exist_ok=True)
        (reports / "ndarray-abi-report.json").write_text(report_text)
        materialize(repository, generated)

    print(
        f"terlan-ndarray ABI v1 passed: {len(first_hashes)} generated files, "
        f"{len(ADVERSARIAL_CASES)} adversarial metadata cases"
    )


if __name__ == "__main__":
    main()
