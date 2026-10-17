"""Tests use fabricated tiny logs only to exercise the log validator."""
import copy
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    records = [dict(kind="start", schema_version=1, profile="tiny", rows_per_part=400,
                    device="test fixture, not hardware evidence", device_index=0,
                    self_tests="passed", config=dict(ahi=10019, bhi=10019, chi=10039,
                                                     ne=40, nd=4, modulus=1009))]
    for part in range(10):
        counts = dict(queries=1600, candidates=0, checksum=0, passed13=0,
                      solutions_count=0, solutions=[])
        records += [dict(kind="shard_start", part=part),
                    dict(kind="batch", part=part, first_pair=0, next_pair=400,
                         device_seconds=0.01, **counts),
                    dict(kind="shard_complete", part=part, rows=400, **counts)]
    records.append(dict(kind="complete", parts=10, queries=16000,
                        solutions_count=0, solutions=[]))
    return records


class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = None
        path = ROOT / "tools/verify_run.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location("verify_run", path)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def test_complete_tiny_log_is_accepted(self):
        self.assertIsNotNone(self.module, "run verifier has not been implemented")
        result = self.module.validate_records(fixture())
        self.assertEqual(result["queries"], 16000)
        self.assertEqual(result["solutions"], [])
        self.assertEqual(result["profile"], "tiny")
        records = fixture()
        records[0]["config"].update(chunk_a=8, batch_rows=400)
        self.assertEqual(self.module.validate_records(records)["queries"], 16000)

    def test_incomplete_or_modified_log_is_rejected(self):
        self.assertIsNotNone(self.module, "run verifier has not been implemented")
        bad = []
        bad.append(fixture()[:-1])
        records = fixture(); records.insert(1, copy.deepcopy(records[0])); bad.append(records)
        records = fixture(); records[2]["first_pair"] = 1; bad.append(records)
        records = fixture(); records[2]["queries"] = True; bad.append(records)
        records = fixture(); records[3]["checksum"] = 1; bad.append(records)
        records = fixture(); records[2]["solutions"] = [[10000] * 5]; records[2]["solutions_count"] = 1; bad.append(records)
        records = fixture(); records[-1]["queries"] -= 1; bad.append(records)
        records = fixture(); records.append(dict(kind="complete")); bad.append(records)
        for records in bad:
            with self.subTest(records=records):
                with self.assertRaises(ValueError):
                    self.module.validate_records(records)

    def test_full_domain_cannot_claim_zero_work(self):
        records = fixture()
        records[0].update(profile="full", rows_per_part=7041690000,
                          config=dict(ahi=99999, bhi=88240, chi=88239,
                                      ne=78240, nd=7824, modulus=948423849))
        for record in records:
            if record["kind"] == "batch":
                record.update(next_pair=7041690000, queries=0)
            elif record["kind"] == "shard_complete":
                record.update(rows=7041690000, queries=0)
            elif record["kind"] == "complete":
                record["queries"] = 0
        with self.assertRaises(ValueError):
            self.module.validate_records(records)


if __name__ == "__main__":
    unittest.main()
