#!/usr/bin/env python3
"""Compare a real tiny GPU run with an independent integer enumeration."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from verify_run import read_records, validate_records
from reference import expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    result = validate_records(read_records(args.log))
    if result["profile"] != "tiny":
        raise ValueError("This oracle checks only the diagnostic profile")
    for shard in result["shards"]:
        oracle = expected(shard["part"])
        for key in ("queries", "candidates", "checksum", "passed13", "solutions_count"):
            if shard[key] != oracle[key]:
                raise ValueError(f"shard {shard['part']}: {key} differs from integer oracle")
    print("All ten diagnostic shards match independent Python integer enumeration.")


if __name__ == "__main__":
    main()
