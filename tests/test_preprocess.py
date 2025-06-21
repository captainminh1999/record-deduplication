import os
import pandas as pd
import tempfile
import unittest
from src import preprocess


class PreprocessTest(unittest.TestCase):
    def test_preprocess_basic(self):
        data = (
            "name,phone,address\n"
            "José,123456789,A\n"
            "Jose,123456789,B\n"
            "Jane,555,A\n"
            "Jane,555,B\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.csv")
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(data)
            output_path = os.path.join(tmpdir, "out.csv")
            audit_path = os.path.join(tmpdir, "audit.csv")

            preprocess.main(
                input_path=input_path,
                output_path=output_path,
                audit_path=audit_path,
                use_openai=False,
            )

            df = pd.read_csv(output_path)
            self.assertEqual(len(df), 2)
            self.assertIn("combined_id", df.columns)

            audit = pd.read_csv(audit_path)
            self.assertEqual(len(audit), 2)


if __name__ == "__main__":
    unittest.main()
