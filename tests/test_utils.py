import os
import tempfile
import time
import unittest
from src.logging import log_run
from src.io import clear_files, clear_all_data


class UtilsTest(unittest.TestCase):
    def test_log_and_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "history.log")
            start = time.time()
            end = start + 1
            log_run("test", start, end, 10, log_path=log_path)
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)
            self.assertIn("test", lines[0])

            file1 = os.path.join(tmpdir, "f1.txt")
            file2 = os.path.join(tmpdir, "f2.txt")
            open(file1, "w").close()
            open(file2, "w").close()
            clear_files([file1])
            self.assertFalse(os.path.exists(file1))
            clear_all_data(tmpdir, exclude=[os.path.basename(log_path)])
            self.assertFalse(os.path.exists(file2))
            self.assertTrue(os.path.exists(log_path))


if __name__ == "__main__":  # pragma: no cover - manual run
    unittest.main()
