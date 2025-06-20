"""Step 4 of the 10-step deduplication pipeline: model training.

This stage fits a logistic regression model on labelled candidate pairs
and predicts duplicate probabilities for all pairs.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression


def main(features_path: str = "data/features.csv", labels_path: str = "data/labels.csv") -> None:
    """Train a model and score candidate pairs.

    TODO:
        * read features and label data from CSV files
        * fit :class:`sklearn.linear_model.LogisticRegression`
        * add a ``prob`` column to the feature table
        * store high-confidence duplicates to ``data/dupes_high_conf.csv``
    """
    pass


if __name__ == "__main__":  # pragma: no cover - sanity run
    print("\u23e9 started model")
    main()
