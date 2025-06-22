import os
import pandas as pd
import tempfile
import unittest
from src import preprocess, blocking


class BlockingTest(unittest.TestCase):
    def test_blocking_outputs_and_logging(self):
        data = (
            "record_id,company,domain,phone,address\n"
            "1,Acme Corp,acme.com,123,A\n"
            "2,Acme Co,acme.org,123,B\n"
            "3,Widgets Inc,widgets.com,555,C\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.csv")
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(data)
            cleaned_path = os.path.join(tmpdir, "cleaned.csv")
            audit_path = os.path.join(tmpdir, "audit.csv")
            pairs_path = os.path.join(tmpdir, "pairs.csv")
            log_path = os.path.join(tmpdir, "history.log")

            preprocess.main(
                input_path=input_path,
                output_path=cleaned_path,
                audit_path=audit_path,
                use_openai=False,
                log_path=log_path,
            )

            blocking.main(
                input_path=cleaned_path,
                output_path=pairs_path,
                log_path=log_path,
            )

            pairs_df = pd.read_csv(pairs_path)
            self.assertEqual(len(pairs_df), 3)
            pair_sets = [set(row) for row in pairs_df.values.tolist()]
            self.assertIn({1, 2}, pair_sets)

            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("preprocess", lines[0])
            self.assertIn("blocking", lines[1])


if __name__ == "__main__":
    unittest.main()
