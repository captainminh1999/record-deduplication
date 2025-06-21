import os
import pandas as pd
import tempfile
import unittest
from src import preprocess


class PreprocessTest(unittest.TestCase):
    def test_normalize_company_name(self):
        self.assertEqual(preprocess.normalize_company_name("The ACME, Inc."), "acme")
        self.assertEqual(preprocess.normalize_company_name("PT Astra International Tbk"), "astra international")
        self.assertEqual(preprocess.normalize_company_name("株式会社ソニー"), "sony")
        self.assertEqual(preprocess.normalize_company_name("XYZ Sdn Bhd"), "xyz")

    def test_preprocess_basic(self):
        data = (
            "record_id,company,domain,phone,address\n"
            "1,Acme Inc,acme.com,123,A\n"
            "2,Acme Inc,acme.com,123,B\n"
            "3,Widgets LLC,widgets.com,555,C\n"
            "4,Widgets LLC,widgets.com,555,D\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.csv")
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(data)
            output_path = os.path.join(tmpdir, "out.csv")
            audit_path = os.path.join(tmpdir, "audit.csv")
            log_path = os.path.join(tmpdir, "log.csv")

            preprocess.main(
                input_path=input_path,
                output_path=output_path,
                audit_path=audit_path,
                use_openai=False,
                log_path=log_path,
            )

            df = pd.read_csv(output_path)
            self.assertEqual(len(df), 2)
            self.assertIn("combined_id", df.columns)

            audit = pd.read_csv(audit_path)
            self.assertEqual(len(audit), 2)

            self.assertTrue(os.path.exists(log_path))

    def test_column_normalization(self):
        data = (
            "Record ID,company,domain,phone,address\n"
            "1,Acme Inc,acme.com,123,A\n"
            "2,Widgets LLC,widgets.com,555,B\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.csv")
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(data)
            output_path = os.path.join(tmpdir, "out.csv")
            audit_path = os.path.join(tmpdir, "audit.csv")
            log_path = os.path.join(tmpdir, "log.csv")

            preprocess.main(
                input_path=input_path,
                output_path=output_path,
                audit_path=audit_path,
                use_openai=False,
                log_path=log_path,
            )

            df = pd.read_csv(output_path)
            self.assertEqual(len(df), 2)
            self.assertIn("record_id", df.columns)
            
    def test_preprocess_missing_domain(self):
        data = (
            "record_id,company,phone,address\n"
            "1,Acme Inc,123,A\n"
            "2,Acme Inc,123,B\n"
            "3,Widgets LLC,555,C\n"
            "4,Widgets LLC,555,D\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.csv")
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(data)
            output_path = os.path.join(tmpdir, "out.csv")
            audit_path = os.path.join(tmpdir, "audit.csv")
            log_path = os.path.join(tmpdir, "log.csv")

            preprocess.main(
                input_path=input_path,
                output_path=output_path,
                audit_path=audit_path,
                use_openai=False,
                log_path=log_path,
            )

            df = pd.read_csv(output_path)
            self.assertEqual(len(df), 2)
            self.assertIn("domain_clean", df.columns)
            self.assertTrue(df["domain_clean"].isna().all())

            audit = pd.read_csv(audit_path)
            self.assertEqual(len(audit), 2)

            self.assertTrue(os.path.exists(log_path))

    def test_preprocess_missing_phone(self):
        data = (
            "record_id,company,domain,address\n"
            "1,Acme Inc,acme.com,A\n"
            "2,Acme Inc,acme.com,B\n"
            "3,Widgets LLC,widgets.com,C\n"
            "4,Widgets LLC,widgets.com,D\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.csv")
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(data)
            output_path = os.path.join(tmpdir, "out.csv")
            audit_path = os.path.join(tmpdir, "audit.csv")
            log_path = os.path.join(tmpdir, "log.csv")

            preprocess.main(
                input_path=input_path,
                output_path=output_path,
                audit_path=audit_path,
                use_openai=False,
                log_path=log_path,
            )

            df = pd.read_csv(output_path)
            self.assertEqual(len(df), 2)
            self.assertIn("phone_clean", df.columns)
            self.assertTrue(df["phone_clean"].isna().all())

            audit = pd.read_csv(audit_path)
            self.assertEqual(len(audit), 2)

            self.assertTrue(os.path.exists(log_path))


if __name__ == "__main__":
    unittest.main()
