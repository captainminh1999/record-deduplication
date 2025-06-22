import os
import pandas as pd
import tempfile
import unittest
from src import similarity


class SimilarityTest(unittest.TestCase):
    def test_similarity_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cleaned_path = os.path.join(tmpdir, "cleaned.csv")
            pairs_path = os.path.join(tmpdir, "pairs.csv")
            features_path = os.path.join(tmpdir, "features.csv")

            df = pd.DataFrame(
                {
                    "record_id": [1, 2],
                    "company_clean": ["acme", "acme"],
                    "domain_clean": ["acme.com", "acme.com"],
                    "phone_clean": ["123", "123"],
                    "address_clean": ["A St", "B St"],
                }
            )
            df.to_csv(cleaned_path, index=False)

            similarity.main(
                cleaned_path=cleaned_path,
                pairs_path=pairs_path,
                features_path=features_path,
            )

            self.assertTrue(os.path.exists(pairs_path))
            self.assertTrue(os.path.exists(features_path))

            feats = pd.read_csv(features_path)
            self.assertEqual(len(feats), 1)
            self.assertIn("company_sim", feats.columns)
            self.assertIn("address_sim", feats.columns)
            self.assertEqual(feats.loc[0, "phone_exact"], 1)

    def test_similarity_without_address(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cleaned_path = os.path.join(tmpdir, "cleaned.csv")
            pairs_path = os.path.join(tmpdir, "pairs.csv")
            features_path = os.path.join(tmpdir, "features.csv")

            df = pd.DataFrame(
                {
                    "record_id": [1, 2],
                    "company_clean": ["acme", "acme"],
                    "domain_clean": ["acme.com", "acme.com"],
                    "phone_clean": ["123", "123"],
                }
            )
            df.to_csv(cleaned_path, index=False)

            similarity.main(
                cleaned_path=cleaned_path,
                pairs_path=pairs_path,
                features_path=features_path,
            )

            feats = pd.read_csv(features_path)
            self.assertEqual(len(feats), 1)
            self.assertNotIn("address_sim", feats.columns)


if __name__ == "__main__":
    unittest.main()
