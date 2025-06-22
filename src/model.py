"""Step 4 of the 10-step deduplication pipeline: model training.

This stage fits a logistic regression model on labelled candidate pairs and
predicts duplicate probabilities for all pairs.
"""

from __future__ import annotations

import click
import joblib
import os
import pandas as pd
from pandas import ExcelWriter  # noqa: F401 - imported for future report steps
from sklearn.linear_model import LogisticRegression


def main(
    features_path: str = "data/outputs/features.csv",
    labels_path: str = "data/outputs/labels.csv",
    model_path: str = "data/outputs/model.joblib",
    duplicates_path: str = "data/outputs/high_confidence.csv",
) -> pd.DataFrame:
    """Train a model and score candidate pairs."""

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not os.path.exists(labels_path):
        msg = (
            f"Labels file not found: {labels_path}. "
            "Model training requires labeled data."
        )
        print(msg)
        raise FileNotFoundError(msg)

    # Load similarity features and labels then merge on the pair identifiers.
    feat_df = pd.read_csv(features_path)
    label_df = pd.read_csv(labels_path)
    train_df = feat_df.merge(label_df, on=["record_id_1", "record_id_2"], how="inner")

    # Split into X/y. All similarity columns are used as features and ``label``
    # is the target variable.
    X = train_df.drop(columns=["record_id_1", "record_id_2", "label"])
    y = train_df["label"]

    # Instantiate a basic logistic regression model. Regularisation strength ``C``
    # and solver can be tuned later for better performance.
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Persist the trained model for later reuse.
    joblib.dump(model, model_path)

    # Score all candidate pairs using the trained model. ``predict_proba`` returns
    # probabilities for the negative and positive classes; ``[:, 1]`` selects the
    # duplicate probability.
    X_all = feat_df.drop(columns=["record_id_1", "record_id_2"])
    probs = model.predict_proba(X_all)[:, 1]
    scored_df = feat_df.copy()
    scored_df["prob"] = probs

    # Filter pairs with probability above the high-confidence threshold. The
    # threshold can be adjusted later as needed.
    high_conf = scored_df[scored_df["prob"] >= 0.9]
    high_conf.to_csv(duplicates_path, index=False)

    return scored_df


@click.command()
@click.option("--features-path", default="data/outputs/features.csv", show_default=True)
@click.option("--labels-path", default="data/outputs/labels.csv", show_default=True)
@click.option("--model-path", default="data/outputs/model.joblib", show_default=True)
@click.option("--duplicates-path", default="data/outputs/high_confidence.csv", show_default=True)
@click.option("--cleaned-path", default="data/outputs/cleaned.csv", show_default=True)
@click.option("--report-path", default="data/outputs/manual_review.xlsx", show_default=True)
def cli(
    features_path: str,
    labels_path: str,
    model_path: str,
    duplicates_path: str,
    cleaned_path: str,  # noqa: ARG001 - reserved for future reporting step
    report_path: str,  # noqa: ARG001 - reserved for future reporting step
) -> None:
    """CLI wrapper for :func:`main`."""

    main(features_path, labels_path, model_path, duplicates_path)


if __name__ == "__main__":  # pragma: no cover - sanity run
    print("\u23e9 started model")
    cli()
