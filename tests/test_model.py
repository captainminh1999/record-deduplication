import os
import pandas as pd
import tempfile
import unittest

from src import model


class ModelTest(unittest.TestCase):
    def test_missing_labels_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            features_path = os.path.join(tmpdir, "features.csv")
            model_path = os.path.join(tmpdir, "model.joblib")
            dupes_path = os.path.join(tmpdir, "dupes.csv")

            # minimal features file
            pd.DataFrame({
                "record_id_1": [1],
                "record_id_2": [2],
                "feat": [0.5],
            }).to_csv(features_path, index=False)

            with self.assertRaises(FileNotFoundError) as cm:
                model.main(
                    features_path=features_path,
                    labels_path=os.path.join(tmpdir, "labels.csv"),
                    model_path=model_path,
                    duplicates_path=dupes_path,
                )

            self.assertIn("requires labeled data", str(cm.exception))

    def test_missing_features_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            labels_path = os.path.join(tmpdir, "labels.csv")
            pd.DataFrame({
                "record_id_1": [1],
                "record_id_2": [2],
                "label": [1],
            }).to_csv(labels_path, index=False)

            with self.assertRaises(FileNotFoundError):
                model.main(
                    features_path=os.path.join(tmpdir, "features.csv"),
                    labels_path=labels_path,
                    model_path=os.path.join(tmpdir, "m.joblib"),
                    duplicates_path=os.path.join(tmpdir, "d.csv"),
                )


if __name__ == "__main__":
    unittest.main()
