import os
import pandas as pd
import tempfile
import unittest
import joblib

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

            self.assertIn("required similarity columns", str(cm.exception))

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

    def test_train_and_output_high_confidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            features_path = os.path.join(tmpdir, "features.csv")
            labels_path = os.path.join(tmpdir, "labels.csv")
            model_path = os.path.join(tmpdir, "model.joblib")
            dupes_path = os.path.join(tmpdir, "high_confidence.csv")

            pd.DataFrame(
                {
                    "record_id_1": [1, 2],
                    "record_id_2": [2, 3],
                    "feat": [0, 10],
                }
            ).to_csv(features_path, index=False)

            pd.DataFrame(
                {
                    "record_id_1": [1, 2],
                    "record_id_2": [2, 3],
                    "label": [0, 1],
                }
            ).to_csv(labels_path, index=False)

            scored = model.main(
                features_path=features_path,
                labels_path=labels_path,
                model_path=model_path,
                duplicates_path=dupes_path,
            )

            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(dupes_path))
            self.assertIn("prob", scored.columns)
            dupes = pd.read_csv(dupes_path)
            self.assertGreaterEqual(len(dupes), 1)

    def test_model_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            features_path = os.path.join(tmpdir, "features.csv")
            labels_path = os.path.join(tmpdir, "labels.csv")
            model_path = os.path.join(tmpdir, "model_dir", "model.joblib")
            dupes_path = os.path.join(tmpdir, "dupes_dir", "high.csv")

            pd.DataFrame(
                {
                    "record_id_1": [1, 2],
                    "record_id_2": [2, 3],
                    "feat": [0, 1],
                }
            ).to_csv(features_path, index=False)

            pd.DataFrame(
                {
                    "record_id_1": [1, 2],
                    "record_id_2": [2, 3],
                    "label": [0, 1],
                }
            ).to_csv(labels_path, index=False)

            model.main(
                features_path=features_path,
                labels_path=labels_path,
                model_path=model_path,
                duplicates_path=dupes_path,
            )

            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(dupes_path))

    def test_model_handles_non_numeric_features(self):
        """Model training should not fail if features contain text values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features_path = os.path.join(tmpdir, "features.csv")
            labels_path = os.path.join(tmpdir, "labels.csv")
            model_path = os.path.join(tmpdir, "model.joblib")
            dupes_path = os.path.join(tmpdir, "dupes.csv")

            pd.DataFrame(
                {
                    "record_id_1": [1, 2],
                    "record_id_2": [2, 3],
                    "feat": ["bluehex", 1.0],
                }
            ).to_csv(features_path, index=False)

            pd.DataFrame(
                {
                    "record_id_1": [1, 2],
                    "record_id_2": [2, 3],
                    "label": [0, 1],
                }
            ).to_csv(labels_path, index=False)

            scored = model.main(
                features_path=features_path,
                labels_path=labels_path,
                model_path=model_path,
                duplicates_path=dupes_path,
            )

            self.assertIn("prob", scored.columns)

    def test_model_requires_two_classes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            features_path = os.path.join(tmpdir, "features.csv")
            labels_path = os.path.join(tmpdir, "labels.csv")

            pd.DataFrame(
                {
                    "record_id_1": [1, 2],
                    "record_id_2": [2, 3],
                    "feat": [0.1, 0.2],
                }
            ).to_csv(features_path, index=False)

            pd.DataFrame(
                {
                    "record_id_1": [1, 2],
                    "record_id_2": [2, 3],
                    "label": [0, 0],
                }
            ).to_csv(labels_path, index=False)

            with self.assertRaises(ValueError):
                model.main(
                    features_path=features_path,
                    labels_path=labels_path,
                    model_path=os.path.join(tmpdir, "m.joblib"),
                    duplicates_path=os.path.join(tmpdir, "d.csv"),
                )

    def test_unsupervised_fallback(self):
        """Model should train using heuristic labels if labels.csv is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features_path = os.path.join(tmpdir, "features.csv")
            model_path = os.path.join(tmpdir, "m.joblib")
            dupes_path = os.path.join(tmpdir, "d.csv")

            pd.DataFrame(
                {
                    "record_id_1": [1, 2],
                    "record_id_2": [2, 3],
                    "company_sim": [1.0, 0.0],
                    "domain_sim": [1.0, 0.0],
                    "phone_exact": [1, 0],
                }
            ).to_csv(features_path, index=False)

            scored = model.main(
                features_path=features_path,
                labels_path=os.path.join(tmpdir, "labels.csv"),
                model_path=model_path,
                duplicates_path=dupes_path,
            )

            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(dupes_path))
            self.assertIn("prob", scored.columns)

    def test_labels_with_extra_columns(self):
        """Labels file may contain extra feature columns which should be ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features_path = os.path.join(tmpdir, "features.csv")
            labels_path = os.path.join(tmpdir, "labels.csv")
            model_path = os.path.join(tmpdir, "model.joblib")
            dupes_path = os.path.join(tmpdir, "dupes.csv")

            # features file with a single feature column
            pd.DataFrame(
                {
                    "record_id_1": [1, 2],
                    "record_id_2": [2, 3],
                    "feat": [0, 1],
                }
            ).to_csv(features_path, index=False)

            # labels file mistakenly containing the same feature column and an extra
            # column. Only the ``label`` column should be used.
            pd.DataFrame(
                {
                    "record_id_1": [1, 2],
                    "record_id_2": [2, 3],
                    "feat": [0, 1],
                    "extra": [9, 9],
                    "label": [0, 1],
                }
            ).to_csv(labels_path, index=False)

            scored = model.main(
                features_path=features_path,
                labels_path=labels_path,
                model_path=model_path,
                duplicates_path=dupes_path,
            )

            # The model should have been trained with only one feature
            mdl = joblib.load(model_path)
            self.assertEqual(mdl.coef_.shape[1], 1)
            self.assertIn("prob", scored.columns)


if __name__ == "__main__":
    unittest.main()
