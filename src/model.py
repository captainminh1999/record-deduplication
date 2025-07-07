"""Step 4 of 6: Model Training (Duplicate Scoring)

Legacy interface that bridges to the new modular architecture.

Trains a logistic regression model to score candidate pairs as duplicates or not, using similarity features. Supports supervised and unsupervised modes. See README for details.
"""

from __future__ import annotations

import click
import time
import json
import pandas as pd

from .core.model_engine import ModelEngine, ModelConfig
from .formatters.model_formatter import ModelFormatter
from .utils import log_run, LOG_PATH


def main(
    features_path: str = "data/outputs/features.csv",
    labels_path: str = "data/outputs/labels.csv",
    model_path: str = "data/outputs/model.joblib",
    duplicates_path: str = "data/outputs/high_confidence.csv",
    log_path: str = LOG_PATH,
    confidence_threshold: float = 0.9,
) -> pd.DataFrame:
    """Train a model and score candidate pairs.
    
    This legacy function bridges to the new modular architecture.
    """
    
    start_time = time.time()
    
    # Initialize components
    engine = ModelEngine()
    formatter = ModelFormatter()
    
    # Print progress
    print(formatter.format_progress("Starting model training and duplicate scoring..."))
    
    # Train model and score
    scored_df, stats = engine.train_and_score(
        features_path=features_path,
        labels_path=labels_path,
        model_path=model_path,
        duplicates_path=duplicates_path,
        confidence_threshold=confidence_threshold
    )
    
    # Format and display results
    print(formatter.format_training_complete())
    print(formatter.format_separator())
    
    # Display data overview
    for line in formatter.format_data_overview(stats):
        print(line)
    
    # Display label info (simplified approach)
    print(f"\n🏷️  Training Labels ({stats.get('label_stats', {}).get('source', 'unknown')}):")
    label_stats = stats.get('label_stats', {})
    print(f"  • Total labels:          {label_stats.get('total_labels', 0):,}")
    print(f"  • Positive (duplicates): {label_stats.get('positive_labels', 0):,} ({label_stats.get('label_ratio', 0):.1%})")
    print(f"  • Negative (unique):     {label_stats.get('negative_labels', 0):,}")
    
    # Display model performance
    for line in formatter.format_model_performance(stats):
        print(line)
    
    # Display results
    for line in formatter.format_results(stats):
        print(line)
    
    # Display files created
    print(f"\n💾 Files Created:")
    print(f"  • Model: {model_path}")
    print(f"  • High-confidence pairs: {duplicates_path}")
    
    # Display next steps
    high_confidence_pairs = stats.get('high_confidence_pairs', 0)
    if high_confidence_pairs > 0:
        print(f"\n✅ Next step: Run reporting to create Excel review file")
        print(f"   Command: python -m src.reporting")
    else:
        print(f"\n💡 Suggestions:")
        print(f"   • Lower confidence threshold (currently {confidence_threshold})")
        print(f"   • Add more training examples to labels.csv")
        print(f"   • Review features.csv for data quality issues")
    
    # Log the run
    end_time = time.time()
    log_run("model", start_time, end_time, len(scored_df), 
            additional_info=json.dumps(stats), log_path=log_path)
    
    return scored_df


@click.command()
@click.option("--features-path", default="data/outputs/features.csv", show_default=True)
@click.option("--labels-path", default="data/outputs/labels.csv", show_default=True)
@click.option("--model-path", default="data/outputs/model.joblib", show_default=True)
@click.option("--duplicates-path", default="data/outputs/high_confidence.csv", show_default=True)
@click.option("--confidence-threshold", type=float, default=0.9, show_default=True)
@click.option("--log-path", default=LOG_PATH, show_default=True)
def cli(
    features_path: str,
    labels_path: str,
    model_path: str,
    duplicates_path: str,
    confidence_threshold: float,
    log_path: str,
) -> None:
    """CLI wrapper for :func:`main`."""

    main(features_path, labels_path, model_path, duplicates_path, log_path, confidence_threshold)


if __name__ == "__main__":  # pragma: no cover - sanity run
    print("\u23e9 started model")
    cli()
