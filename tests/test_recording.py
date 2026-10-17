import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RecordingTests(unittest.TestCase):
    def test_full_command_has_no_resume_or_partial_controls(self):
        path = ROOT / "tools/record_run.py"
        self.assertTrue(path.exists(), "receipt runner not implemented")
        spec = importlib.util.spec_from_file_location("record_run", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.search_command(Path("/work/search"), 0),
                         ["/work/search", "--device", "0"])
        with self.assertRaises(ValueError):
            module.search_command(Path("/work/search"), -1)

    def test_build_command_names_sources_flags_and_fresh_output(self):
        path = ROOT / "tools/record_run.py"
        spec = importlib.util.spec_from_file_location("record_run", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, "build_command"), "clean build is not yet recorded")
        command = module.build_command(Path("/work/new/search"), "sm_120")
        self.assertIn("-arch=sm_120", command)
        self.assertEqual(command[-2:], ["-o", "/work/new/search"])
        self.assertIn(str(ROOT / "src/search.cu"), command)


if __name__ == "__main__":
    unittest.main()
