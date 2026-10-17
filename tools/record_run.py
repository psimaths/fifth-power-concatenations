#!/usr/bin/env python3
"""Execute one full search once and record its provenance; never restart it."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_run import file_hash, read_records, validate_records

ROOT = Path(__file__).resolve().parents[1]


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def search_command(executable, device):
    if type(device) is not int or device < 0:
        raise ValueError("device must be a nonnegative integer")
    return [str(executable), "--device", str(device)]


def build_command(executable, architecture):
    if not re.fullmatch(r"native|sm_[0-9]+[a-z]?", architecture):
        raise ValueError("invalid CUDA architecture")
    compiler = shutil.which("nvcc")
    if compiler is None:
        raise ValueError("nvcc is not on PATH")
    return [compiler, "-O3", "-std=c++17", "-Xcompiler=-fopenmp",
            "-arch=" + architecture, str(ROOT / "src/search.cu"), "-o", str(executable)]


def probe(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return dict(command=command, exit_code=result.returncode,
                    stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(command=command, error=str(error))


def source_hashes():
    paths = [ROOT / "Makefile"]
    for folder in ("src", "tools", "tests"):
        paths.extend(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new output directory; existing paths are refused")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--architecture", default="native")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    executable = output / "search"
    command = search_command(executable, args.device)
    compilation = build_command(executable, args.architecture)
    before = source_hashes()
    with (output / "build.log").open("wb") as build_log:
        built = subprocess.run(compilation, cwd=ROOT, stdout=build_log, stderr=subprocess.STDOUT)
    build = dict(command=compilation, exit_code=built.returncode, source_sha256=before,
                 compiler=probe([compilation[0], "--version"]), host_compiler=probe(["g++", "--version"]),
                 log_sha256=file_hash(output / "build.log"))
    if built.returncode or before != source_hashes():
        (output / "receipt.json").write_text(json.dumps(dict(status="build_failed", build=build), indent=2) + "\n")
        return 1
    build["executable_sha256"] = file_hash(executable)
    (output / "build.json").write_text(json.dumps(build, indent=2) + "\n")
    receipt = dict(schema_version=1, status="started", started_utc=utc(),
                   build_manifest_sha256=file_hash(output / "build.json"),
                   command=command, executable_sha256=file_hash(executable), source_sha256=before,
                   invocation_count=1, resume_supported=False,
                   platform=platform.platform(), python=sys.version,
                   environment={k: os.environ.get(k) for k in
                                ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES")},
                   evidence_limit="Local execution record, not cryptographic proof of honest execution or formal correctness.")
    for name, cmd in (("gpu", ["nvidia-smi", "-q"]), ("compiler", ["nvcc", "--version"]),
                      ("host_compiler", ["g++", "--version"]), ("cpu", ["lscpu"])):
        receipt[name] = probe(cmd)
    (output / "started.json").write_text(json.dumps(receipt, indent=2) + "\n")
    started = time.monotonic()
    with (output / "stdout.jsonl").open("wb") as stdout, (output / "stderr.txt").open("wb") as stderr:
        proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
        try:
            receipt["process_id"] = proc.pid
            (output / "started.json").write_text(json.dumps(receipt, indent=2) + "\n")
            receipt["exit_code"] = proc.wait()
        except (KeyboardInterrupt, OSError) as error:
            proc.terminate()
            receipt["exit_code"] = proc.wait()
            receipt["interrupted"] = True
            receipt["execution_error"] = str(error)
    receipt.update(ended_utc=utc(), elapsed_seconds=time.monotonic() - started,
                   stdout_sha256=file_hash(output / "stdout.jsonl"),
                   stderr_sha256=file_hash(output / "stderr.txt"))
    try:
        if receipt["exit_code"] != 0 or receipt.get("interrupted"):
            raise ValueError("search did not exit normally")
        if before != source_hashes() or receipt["executable_sha256"] != file_hash(executable):
            raise ValueError("code or executable changed during execution")
        result = validate_records(read_records(output / "stdout.jsonl"))
        if result["profile"] != "full":
            raise ValueError("not the full domain")
        if (output / "stderr.txt").stat().st_size:
            raise ValueError("search emitted stderr; inspect before accepting this run")
        receipt.update(status="complete", result=result)
    except (ValueError, OSError) as error:
        receipt.update(status="failed", validation_error=str(error))
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "exit_code", "elapsed_seconds")}))
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
