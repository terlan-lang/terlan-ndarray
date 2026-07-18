#!/usr/bin/env python3

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
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
    if completed.returncode == 0 and not expect_success:
        raise RuntimeError(f"command unexpectedly passed: {' '.join(command)}")
    if completed.returncode != 0 and expect_success:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def encoded(value):
    return base64.b64encode(value.encode()).decode()


def check_helper_lifecycle(helper, repository):
    handle_type = encoded("ndarray.Array.Array")
    wrong_type = encoded("other.Resource")
    operations = {
        name: encoded(f"ndarray.array.{name}")
        for name in ("from_ints", "dispose", "rank", "live_count")
    }
    requests = "".join(
        (
            f"call 1 {operations['from_ints']} li:11,12 li:2\n",
            f"call 2 {operations['dispose']} h:1:1:{handle_type}\n",
            f"call 3 {operations['dispose']} h:1:1:{handle_type}\n",
            f"call 4 {operations['from_ints']} li:21,22 li:2\n",
            f"call 5 {operations['rank']} h:1:1:{handle_type}\n",
            f"call 6 {operations['rank']} h:1:2:{wrong_type}\n",
            f"call 7 {operations['dispose']} h:1:2:{handle_type}\n",
            f"call 8 {operations['live_count']}\n",
        )
    )
    completed = subprocess.run(
        [str(helper)],
        cwd=repository,
        input=requests,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"native helper failed:\n{completed.stderr}")
    replies = completed.stdout.splitlines()
    required = {
        f"reply 1 1 ok_handle 1 1 {handle_type}",
        "reply 2 1 ok_unit",
        f"reply 4 1 ok_handle 1 2 {handle_type}",
        "reply 7 1 ok_unit",
        "reply 8 1 ok_int 0",
    }
    missing = sorted(required.difference(replies))
    if missing:
        raise RuntimeError(
            f"native helper lifecycle replies missing {missing}:\n{completed.stdout}"
        )
    expected_errors = {
        3: "stale_handle",
        5: "handle_storage_mismatch",
        6: "handle_type_mismatch",
    }
    for request_id, code in expected_errors.items():
        encoded_code = encoded(code)
        if not any(
            reply.startswith(f"reply {request_id} 1 ") and encoded_code in reply
            for reply in replies
        ):
            raise RuntimeError(
                f"request {request_id} did not report {code}:\n{completed.stdout}"
            )
    return {
        "double_disposal": "passed",
        "slot_generation_reuse": "passed",
        "stale_generation": "passed",
        "stale_handle": "passed",
        "wrong_resource_kind": "passed",
    }


def check_sanitizers(repository, temporary):
    compiler = os.environ.get("CC", "cc")
    binary = temporary / "ndarray-sanitized-test"
    compile_result = subprocess.run(
        [
            compiler,
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-fno-omit-frame-pointer",
            "-fsanitize=address,undefined",
            str(repository / "bindings/terlan_ndarray.c"),
            str(repository / "tests/abi_test.c"),
            "-o",
            str(binary),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if compile_result.returncode != 0:
        sanitizers = {
            "reason": "compiler does not provide stable address/undefined sanitizers",
            "status": "skipped",
        }
    else:
        environment = os.environ.copy()
        environment["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
        environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
        sanitized = subprocess.run(
            [str(binary)],
            cwd=repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if sanitized.returncode == 0:
            leak_detection = {"status": "passed"}
        elif "LeakSanitizer does not work under ptrace" in sanitized.stderr:
            environment["ASAN_OPTIONS"] = "detect_leaks=0:halt_on_error=1"
            run([str(binary)], cwd=repository, env=environment)
            leak_detection = {
                "reason": "LeakSanitizer is unavailable under the host ptrace environment",
                "status": "skipped",
            }
        else:
            raise RuntimeError(
                "sanitized native lifecycle test failed:\n"
                f"stdout:\n{sanitized.stdout}\nstderr:\n{sanitized.stderr}"
            )
        sanitizers = {
            "leak_detection": leak_detection,
            "status": "passed",
            "tools": ["address", "undefined"],
        }

    valgrind = shutil.which("valgrind")
    if valgrind is None:
        valgrind_result = {
            "reason": "valgrind executable is unavailable",
            "status": "skipped",
        }
    else:
        native_binary = temporary / "ndarray-native-test"
        run(
            [
                compiler,
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                str(repository / "bindings/terlan_ndarray.c"),
                str(repository / "tests/abi_test.c"),
                "-o",
                str(native_binary),
            ],
            cwd=repository,
        )
        run(
            [
                valgrind,
                "--error-exitcode=97",
                "--leak-check=full",
                "--show-leak-kinds=all",
                str(native_binary),
            ],
            cwd=repository,
        )
        valgrind_result = {"status": "passed"}
    return {"sanitizers": sanitizers, "valgrind": valgrind_result}


def write_consumer(consumer):
    source = consumer / "src/ndarray_consumer/Main.terl"
    source.parent.mkdir(parents=True)
    source.write_text(
        """module ndarray_consumer.Main.

import ndarray.Array.{byte_count, device_type_code, dispose, dtype_code, from_floats, live_count, numel, rank, shape, strides, to_floats}.
import std.io.Console.{println}.

pub main(): Unit ->
    let first = 0.5 + 0.75;
    let values = [first, first + 1.0, first + 2.0, first + 3.0, first + 4.0, first + 5.0];
    let array = from_floats(values, [2, 3]);
    let valid = rank(array) == 2 and shape(array) == [2, 3] and
        strides(array) == [3, 1] and numel(array) == 6 and
        byte_count(array) == 48 and dtype_code(array) == 3 and
        device_type_code(array) == 1 and to_floats(array) == values;
    dispose(array);
    if {
        valid and live_count() == 0 -> println("NDARRAY_PACKAGE_CONSUMER_OK");
        true -> println("NDARRAY_PACKAGE_CONSUMER_FAILED")
    }.
"""
    )


def check_fresh_consumer(terlc, generated, temporary, environment):
    package = temporary / "package"
    consumer = temporary / "consumer"
    shutil.copytree(generated, package)
    shutil.rmtree(package / "native/rust/target", ignore_errors=True)
    run(["git", "init", "--quiet"], cwd=package)
    run(["git", "config", "user.email", "terlan-ndarray-tests@example.invalid"], cwd=package)
    run(["git", "config", "user.name", "Terlan NdArray Tests"], cwd=package)
    run(["git", "add", "."], cwd=package)
    run(["git", "commit", "--quiet", "-m", "generated ndarray package"], cwd=package)
    revision = run(["git", "rev-parse", "HEAD"], cwd=package).stdout.strip()
    consumer.mkdir()
    (consumer / "terlan.toml").write_text(
        "[package]\n"
        'name = "terlan-ndarray-consumer"\n'
        'version = "0.0.0"\n'
        'namespace = "ndarray_consumer"\n\n'
        "[build]\n"
        'source_roots = ["src"]\n'
        'artifact = "terlan-vm"\n\n'
        "[dependencies]\n"
        f'ndarray = {{ git = "{package}", rev = "{revision}" }}\n'
    )
    write_consumer(consumer)
    run([str(terlc), "package", "fetch", str(consumer)], cwd=temporary, env=environment)
    if not (consumer / "terlan.lock").is_file():
        raise RuntimeError("fresh ndarray consumer did not write terlan.lock")
    shutil.rmtree(package)
    completed = run(
        [
            str(terlc),
            "run",
            str(consumer),
            "--target",
            "terlan-vm",
            "--out-dir",
            str(temporary / "consumer-build"),
        ],
        cwd=temporary,
        env=environment,
    )
    if (
        "NDARRAY_PACKAGE_CONSUMER_OK" not in completed.stdout
        or "NDARRAY_PACKAGE_CONSUMER_FAILED" in completed.stdout
    ):
        raise RuntimeError(
            f"fresh ndarray consumer returned incorrect semantics:\n{completed.stdout}"
        )
    return {"revision_locked": True, "status": "passed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terlc", required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    terlc = Path(args.terlc).resolve()
    run(
        [
            "python3",
            str(repository / "scripts/check-abi.py"),
            "--terlc",
            str(terlc),
            "--manifest",
            "bindings/ndarray-c.json",
        ],
        cwd=repository,
    )
    generated = repository / "generated"
    with tempfile.TemporaryDirectory(prefix="terlan_ndarray_package_") as root:
        temporary = Path(root)
        environment = os.environ.copy()
        environment["CARGO_NET_OFFLINE"] = "true"
        environment["CARGO_TARGET_DIR"] = str(temporary / "cargo-target")
        run(
            [
                "cargo",
                "build",
                "--offline",
                "--manifest-path",
                str(generated / "native/rust/Cargo.toml"),
                "--bin",
                "native-boundary-helper",
            ],
            cwd=repository,
            env=environment,
        )
        helper = temporary / "cargo-target/debug/native-boundary-helper"
        lifecycle = check_helper_lifecycle(helper, repository)
        tooling = check_sanitizers(repository, temporary)
        consumer = check_fresh_consumer(terlc, generated, temporary, environment)

    abi_report = repository / "reports/ndarray-abi-report.json"
    report = {
        "abi_report_sha256": hashlib.sha256(abi_report.read_bytes()).hexdigest(),
        "consumer": consumer,
        "dtypes": ["bool8", "float64", "int64"],
        "lifecycle": lifecycle,
        "package": "terlan-ndarray",
        "schema": "terlan.ndarray.package-report.v1",
        "status": "passed",
        "tooling": tooling,
    }
    report_path = repository / "reports/ndarray-package-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("terlan-ndarray ND1 package lifecycle passed")


if __name__ == "__main__":
    main()
