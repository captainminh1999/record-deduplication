"""Step 4 of 6: Model Training (Duplicate Scoring)

Trains a logistic regression model to score candidate pairs as duplicates or not, using similarity features. Supports supervised and unsupervised modes. See README for details.
"""

from __future__ import annotations

import click
import joblib
import os
import time
import pandas as pd
from pandas import ExcelWriter  # noqa: F401 - imported for future report steps
from sklearn.linear_model import LogisticRegression
from .utils import log_run, LOG_PATH
import json


def _heuristic_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return pseudo labels based on simple similarity thresholds."""
    required = {"company_sim", "domain_sim"}
    missing = required.difference(df.columns)
    if missing:
        raise FileNotFoundError(
            "Labels file not found and required similarity columns are missing"
        )

    pos_mask = (df.get("company_sim", 0) >= 0.99) & (df.get("domain_sim", 0) >= 0.99)
    if "phone_exact" in df.columns:
        pos_mask &= df["phone_exact"] == 1
    if "address_sim" in df.columns:
        pos_mask &= df["address_sim"] >= 0.99

    neg_mask = (df.get("company_sim", 1) <= 0.1) & (df.get("domain_sim", 1) <= 0.1)
    if "phone_exact" in df.columns:
        neg_mask &= df["phone_exact"] == 0
    if "address_sim" in df.columns:
        neg_mask &= df["address_sim"] <= 0.1

    pos_df = df[pos_mask]
    neg_df = df[neg_mask]

    if pos_df.empty or neg_df.empty:
        raise FileNotFoundError(
            "Labels file not found and insufficient heuristic examples for training. "
            "Ensure `features.csv` contains at least one near-duplicate pair and one clearly different pair, "
            "or provide a labels.csv with examples."
        )

    n = min(len(pos_df), len(neg_df))
    labeled = pd.concat(
        [
            pos_df.head(n)[["record_id_1", "record_id_2"]].assign(label=1),
            neg_df.head(n)[["record_id_1", "record_id_2"]].assign(label=0),
        ],
        ignore_index=True,
    )
    return labeled


def main(
    features_path: str = "data/outputs/features.csv",
    labels_path: str = "data/outputs/labels.csv",
    model_path: str = "data/outputs/model.joblib",
    duplicates_path: str = "data/outputs/high_confidence.csv",
    log_path: str = LOG_PATH,
) -> pd.DataFrame:
    """Train a model and score candidate pairs."""

    print("🚀 Starting model training and duplicate scoring...")
    
    start_time = time.time()

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")

    feat_df = pd.read_csv(features_path)
    input_pairs = len(feat_df)

    # Track label statistics
    if os.path.exists(labels_path):
        label_df = pd.read_csv(labels_path)
        label_source = "manual"
        # Ignore any extra columns in labels.csv to avoid duplicate feature
        # columns after merging with ``features.csv``.
        label_df = label_df[["record_id_1", "record_id_2", "label"]]
    else:
        print(
            f"Labels file not found: {labels_path}. "
            "Using heuristic positive/negative pairs for training."
        )
        label_df = _heuristic_labels(feat_df)
        label_source = "heuristic"

    label_stats = {
        "source": label_source,
        "total_labels": len(label_df),
        "positive_labels": int(label_df["label"].sum()),
        "negative_labels": int(len(label_df) - label_df["label"].sum()),
        "label_ratio": float(label_df["label"].mean())
    }

    train_df = feat_df.merge(label_df, on=["record_id_1", "record_id_2"], how="inner")
    train_pairs = len(train_df)

    # Split into X/y. All similarity columns are used as features and ``label``
    # is the target variable.
    feature_cols = [col for col in feat_df.columns 
                   if col.endswith("_sim") or col.endswith("_exact")]
    X = train_df[feature_cols]
    y = train_df["label"]

    if y.nunique() < 2:
        raise ValueError(
            "Labels file must contain at least two classes for training"
        )

    # Convert any non-numeric feature columns to numeric. If strings are
    # encountered (e.g. due to bad input like "bluehex"), ``to_numeric`` will
    # produce ``NaN`` which we replace with ``0`` so the model can still train.
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Instantiate and train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Track model coefficients
    model_stats = {
        "features": feature_cols,
        "coefficients": dict(zip(feature_cols, model.coef_[0].tolist())),
        "intercept": float(model.intercept_[0])
    }

    # Persist the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # Score all pairs
    X_all = (
        feat_df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )
    probs = model.predict_proba(X_all)[:, 1]
    scored_df = feat_df.copy()
    scored_df["prob"] = probs

    # Track prediction statistics
    prob_stats = {
        "mean_prob": float(probs.mean()),
        "prob_dist": {
            "p90": float(pd.Series(probs).quantile(0.9)),
            "p95": float(pd.Series(probs).quantile(0.95)),
            "p99": float(pd.Series(probs).quantile(0.99))
        }
    }

    # Filter high-confidence pairs
    high_conf = scored_df[scored_df["prob"] >= 0.9]
    high_conf_pairs = len(high_conf)
    os.makedirs(os.path.dirname(duplicates_path), exist_ok=True)
    high_conf.to_csv(duplicates_path, index=False)

    # Print comprehensive terminal output
    print(f"\n🎯 Model Training Complete!")
    print(f"─" * 50)
    print(f"📊 Data Overview:")
    print(f"  • Input pairs:           {input_pairs:,}")
    print(f"  • Training pairs:        {train_pairs:,}")
    print(f"  • Features used:         {len(feature_cols)} ({', '.join(feature_cols)})")
    
    print(f"\n🏷️  Training Labels ({label_source}):")
    print(f"  • Total labels:          {label_stats['total_labels']:,}")
    print(f"  • Positive (duplicates): {label_stats['positive_labels']:,} ({label_stats['label_ratio']:.1%})")
    print(f"  • Negative (unique):     {label_stats['negative_labels']:,}")
    
    print(f"\n🤖 Model Performance:")
    print(f"  • Mean probability:      {prob_stats['mean_prob']:.3f}")
    print(f"  • 90th percentile:       {prob_stats['prob_dist']['p90']:.3f}")
    print(f"  • 95th percentile:       {prob_stats['prob_dist']['p95']:.3f}")
    print(f"  • 99th percentile:       {prob_stats['prob_dist']['p99']:.3f}")
    
    print(f"\n📈 Results:")
    print(f"  • High-confidence pairs: {high_conf_pairs:,} (≥90% probability)")
    if high_conf_pairs > 0:
        print(f"  • Success! Found {high_conf_pairs:,} likely duplicate pairs")
        print(f"  • Wrote results to: {duplicates_path}")
    else:
        print(f"  • No high-confidence duplicates found (try lowering threshold)")
    
    print(f"\n💾 Files Created:")
    print(f"  • Model: {model_path}")
    print(f"  • High-confidence pairs: {duplicates_path}")
    
    if high_conf_pairs > 0:
        print(f"\n✅ Next step: Run reporting to create Excel review file")
        print(f"   Command: python -m src.reporting")
    else:
        print(f"\n💡 Suggestions:")
        print(f"   • Lower confidence threshold in code (currently 0.9)")
        print(f"   • Add more training examples to labels.csv")
        print(f"   • Review features.csv for data quality issues")

    end_time = time.time()
    stats = {
        "input_pairs": input_pairs,
        "training_pairs": train_pairs,
        "high_confidence_pairs": high_conf_pairs,
        "label_stats": label_stats,
        "model_stats": model_stats,
        "prediction_stats": prob_stats
    }
    log_run("model", start_time, end_time, len(scored_df), additional_info=json.dumps(stats), log_path=log_path)

    return scored_df


@click.command()
@click.option("--features-path", default="data/outputs/features.csv", show_default=True)
@click.option("--labels-path", default="data/outputs/labels.csv", show_default=True)
@click.option("--model-path", default="data/outputs/model.joblib", show_default=True)
@click.option("--duplicates-path", default="data/outputs/high_confidence.csv", show_default=True)
@click.option("--cleaned-path", default="data/outputs/cleaned.csv", show_default=True)
@click.option("--report-path", default="data/outputs/manual_review.xlsx", show_default=True)
@click.option("--log-path", default=LOG_PATH, show_default=True)
def cli(
    features_path: str,
    labels_path: str,
    model_path: str,
    duplicates_path: str,
    cleaned_path: str,  # noqa: ARG001 - reserved for future reporting step
    report_path: str,  # noqa: ARG001 - reserved for future reporting step
    log_path: str,
) -> None:
    """CLI wrapper for :func:`main`."""

    main(features_path, labels_path, model_path, duplicates_path, log_path)


if __name__ == "__main__":  # pragma: no cover - sanity run
    print("\u23e9 started model")
    cli()
