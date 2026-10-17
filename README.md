# Fifth powers equal to their concatenation

This program searches for ordered tuples of five five-digit positive integers
satisfying

    a^5 + b^5 + c^5 + d^5 + e^5 = concat(a, b, c, d, e).

Order matters. Repeated values are allowed. Leading-zero blocks and numbers of
other digit lengths are not part of the problem.

The search uses exact integer arithmetic. One invocation searches the entire
domain on one NVIDIA GPU, processing ten disjoint parts sequentially. There are
no checkpoints, resume options, saved lookup tables, or imported search results.
An interrupted invocation is incomplete and must be started again from the
beginning.

## Build and test

Requirements: 64-bit Linux, an NVIDIA CUDA-capable GPU, the CUDA development
toolkit, a compatible C++17 host compiler with OpenMP, Make, and Python 3.9 or
newer. Python uses only its standard library.

Allow at least 8 GB of available host RAM and 7 GB of available GPU memory;
16 GB of host RAM is recommended. The lookup table occupies 6.24 GB on both
host and device. Plans are generated in bounded memory, not stored on disk.

    make CUDA_ARCH=sm_120
    make test
    make test-gpu

`sm_120` targets the RTX 5090/5080 family and requires a supporting toolkit
(CUDA 12.8 or newer). For another supported GPU, use its architecture or build
on the target machine with `make CUDA_ARCH=native`. Use `make -B` after changing
compiler flags: Make does not track changes to command-line variables.

`make test` checks host arithmetic, interval generation, table construction, and
the evidence tools. `make test-gpu` runs a small diagnostic
search and compares all ten parts with an independent Python-integer
implementation. The diagnostic is not a search of the original domain.

## Full computation

Build and test first, then run:

    OMP_NUM_THREADS=8 python3 tools/record_run.py run-output --architecture sm_120

The output directory must not already exist. The recorder first compiles a new
executable directly from the sources and records the exact build command and
output; it does not reuse `build/search`. It then invokes the full
search executable exactly once. It does not retry or resume. The GPU defaults
to device 0; use `--device N` on the recording command to select another device.
The GPU is used by one process; OpenMP parallelizes host interval preparation.

Output files:

- `build.json`, `build.log`, `search`: clean build identity, compiler output,
  and the executable used for this run.

- `stdout.jsonl`: raw enumeration log, including every batch's row interval,
  counters, and any solutions.
- `stderr.txt`: raw diagnostics; an accepted clean run requires this to be empty.
- `started.json`: command, input-code hashes, executable hash, hardware and
  compiler information recorded before execution.
- `receipt.json`: exit status, elapsed time, output hashes and checked totals.
  Its status is `complete` only after successful full-domain validation.

Validate a raw log independently of the recording command:

    python3 tools/verify_run.py run-output/stdout.jsonl

The executable can also be run directly:

    ./build/search > search.jsonl 2> search.err

It supports only `--device`, `--threads 128|256|512`, `--tiny`, and `--help`.
Do not use `--tiny` for the full computation. No option selects a partial range.

## Algorithm and coverage

Write `B = 100000` and `F_j(x) = x^5 - B^j x`. The equation becomes

    F_4(a) + F_3(b) + F_2(c) + F_1(d) + F_0(e) = 0.

Monotonicity and exact exclusion inequalities reduce the original five-digit
domain to

    10000 <= a <= 99999
    10000 <= b <= 88240
    10000 <= c, d, e <= 88239.

The program checks those inequalities at startup. It then partitions `d` by
its last digit. For each part it builds a collision-preserving table of all
`(d,e)` pairs. Exact magnitude bounds restrict the possible `c` interval for
each `(a,b)`. Modulo-30 and modulo-13 conditions reject impossible candidates;
the remaining candidates receive an exact 96-bit test and host 128-bit check.
Python independently rechecks every reported solution.

See [MATHEMATICS.md](MATHEMATICS.md) for the completeness argument, integer
bounds, and limitations of the execution evidence.

## Source layout

- `src/core.hpp`: exact arithmetic, domain bounds, pair-table construction,
  interval generation.
- `src/search.cu`: CUDA traversal, independent startup checks, and the complete
  ten-part execution loop.
- `tests/`: independent small-domain oracle and evidence-tool tests.
- `tools/record_run.py`: one-shot execution and provenance capture.
- `tools/verify_run.py`: strict log/coverage validation and integer verification
  of reported solutions.

No cloud-provider code or credentials are required to build or run the search.

## Result and evidence status

The recorded full computation found exactly one solution:
`(74416, 50589, 59222, 49043, 82475)`. Direct substitution verifies the equality

    74416^5 + 50589^5 + 59222^5 + 49043^5 + 82475^5
    = 7441650589592224904382475.

The [run receipt](results/single-gpu-20261016/receipt.json) records one full
invocation on one RTX 5090, from 2026-10-16 09:59:26 to 18:15:43 UTC
(8.27 hours), with exit code zero and empty stderr. All ten parts completed:
421,434,803,679,310 queries in total. Source, build and raw-output hashes were
checked after download. No restart or checkpoint resume was used.

The [evidence directory](results/single-gpu-20261016/) contains the receipt,
build command and compiler log, startup metadata, raw JSONL output, and stderr.
Neither the executable nor the log verifier requires exactly one solution;
that count is the result of the computation.

A completed exhaustive run, together with the coverage argument, supports a
computational uniqueness claim in the stated domain. A receipt is an execution
record, not a formal proof of compiler/GPU correctness or a third-party
cryptographic attestation. The equality witness alone proves existence, not
uniqueness.
