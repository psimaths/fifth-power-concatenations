#!/usr/bin/env python3
"""Validate exhaustive coverage and reported equalities, not the expected answer."""
import argparse
import hashlib
import json
import math
from pathlib import Path

COUNTERS = ("queries", "candidates", "checksum", "passed13", "solutions_count")
CONFIGS = {
    "tiny": dict(ahi=10019, bhi=10019, chi=10039, ne=40, nd=4, modulus=1009),
    "full": dict(ahi=99999, bhi=88240, chi=88239, ne=78240, nd=7824, modulus=948423849),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(record, key, minimum=0, maximum=None):
    value = record.get(key)
    require(type(value) is int and value >= minimum, f"invalid integer {key}")
    require(maximum is None or value <= maximum, f"out-of-range {key}")
    return value


def check_solution(values, part, config, first, stop):
    require(type(values) is list and len(values) == 5 and all(type(x) is int for x in values),
            "malformed solution")
    a, b, c, d, e = values
    limits = (config["ahi"], config["bhi"], config["chi"], 10000 + 10 * config["nd"] - 1,
              10000 + config["ne"] - 1)
    require(all(10000 <= x <= hi for x, hi in zip(values, limits)), "solution outside domain")
    require(d % 10 == part, "solution in wrong shard")
    row = (a - 10000) * (config["bhi"] - 10000 + 1) + b - 10000
    require(first <= row < stop, "solution outside batch")
    require(sum(x**5 for x in values) == int("".join(map(str, values))), "false solution")


def validate_records(records):
    records = iter(records)
    start = next(records, None)
    require(isinstance(start, dict) and start.get("kind") == "start", "missing start")
    require(start.get("schema_version") == 1, "unsupported schema")
    profile = start.get("profile")
    require(profile in CONFIGS, "unknown profile")
    config = CONFIGS[profile]
    observed_config = start.get("config")
    require(isinstance(observed_config, dict)
            and all(observed_config.get(k) == v for k, v in config.items()),
            "unexpected search domain")
    if "chunk_a" in observed_config:
        require(integer(observed_config, "chunk_a") == (8 if profile == "tiny" else 256),
                "unexpected plan chunk size")
    if "batch_rows" in observed_config:
        require(integer(observed_config, "batch_rows") == (400 if profile == "tiny" else 1048576),
                "unexpected batch size")
    rows = (config["ahi"] - 9999) * (config["bhi"] - 9999)
    require(integer(start, "rows_per_part", 1) == rows, "wrong row count")
    require(start.get("self_tests") == "passed", "startup checks missing")
    require(isinstance(start.get("device"), str) and start["device"], "missing device")
    integer(start, "device_index")
    summaries, all_solutions = [], []
    total_queries, per_part_queries = 0, None
    for part in range(10):
        entry = next(records, None)
        require(isinstance(entry, dict) and entry.get("kind") == "shard_start"
                and entry.get("part") == part, "missing or reordered shard start")
        cursor, sums, hits = 0, dict.fromkeys(COUNTERS, 0), []
        while True:
            entry = next(records, None)
            require(isinstance(entry, dict), "truncated shard")
            require(entry.get("part") == part, "wrong shard")
            if entry.get("kind") == "shard_complete":
                break
            require(entry.get("kind") == "batch", "unexpected event within shard")
            first = integer(entry, "first_pair")
            stop = integer(entry, "next_pair", 1, rows)
            require(first == cursor and stop > first, "noncontiguous coverage")
            require(stop - first <= min(rows, 1048576), "batch exceeds capacity")
            for key in COUNTERS:
                value = integer(entry, key, maximum=(1 << 64) - 1)
                sums[key] += value
            sums["checksum"] %= 1 << 64
            require(entry["queries"] <= (stop - first) * ((config["chi"] - 9999) // 10),
                    "impossible query count")
            require(entry["solutions_count"] <= entry["passed13"] <= entry["candidates"],
                    "inconsistent filter counters")
            seconds = entry.get("device_seconds")
            require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0,
                    "invalid device time")
            solutions = entry.get("solutions")
            require(type(solutions) is list and len(solutions) == entry["solutions_count"],
                    "incomplete solution list")
            for values in solutions:
                check_solution(values, part, config, first, stop)
                hits.append(values)
            cursor = stop
        require(cursor == rows and integer(entry, "rows", 1) == rows, "incomplete shard")
        require(sums["queries"] > 0, "empty search is not full coverage")
        for key in COUNTERS:
            require(integer(entry, key) == sums[key], f"shard {part}: {key} mismatch")
        require(entry.get("solutions") == hits, "shard solution list mismatch")
        if per_part_queries is None:
            per_part_queries = sums["queries"]
        require(sums["queries"] == per_part_queries, "shards disagree on planned query count")
        if profile == "tiny":
            require(sums["queries"] == 1600, "tiny coverage count wrong")
        total_queries += sums["queries"]
        all_solutions.extend(hits)
        summaries.append(dict(part=part, rows=rows, **sums))
    end = next(records, None)
    require(isinstance(end, dict) and end.get("kind") == "complete", "missing completion")
    require(integer(end, "parts") == 10 and integer(end, "queries") == total_queries,
            "completion totals wrong")
    require(integer(end, "solutions_count") == len(all_solutions)
            and end.get("solutions") == all_solutions, "completion solutions wrong")
    require(len({tuple(x) for x in all_solutions}) == len(all_solutions), "duplicate solutions")
    require(next(records, None) is None, "records after completion")
    return dict(profile=profile, device=start["device"], device_index=start["device_index"],
                queries=total_queries, queries_per_part=per_part_queries,
                shards=summaries, solutions_count=len(all_solutions), solutions=all_solutions)


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def read_records(path):
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            require(line.endswith("\n") and line.strip(), "incomplete or empty log line")
            yield json.loads(line, object_pairs_hook=no_duplicates)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for data in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_records(read_records(args.log)), indent=2))


if __name__ == "__main__":
    main()
