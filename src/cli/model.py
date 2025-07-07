"""Model CLI Module

Command-line interface for the model training step of the record deduplication pipeline.
This module orchestrates the model engine and formatter to provide a clean CLI experience.
"""

import time
import json
import click

from ..core.model_engine import ModelEngine
from ..formatters.model_formatter import ModelFormatter
from ..utils import log_run, LOG_PATH


@click.command()
@click.option("--features-path", default="data/outputs/features.csv", show_default=True,
              help="Path to similarity features CSV file")
@click.option("--labels-path", default="data/outputs/labels.csv", show_default=True,
              help="Path to training labels CSV file (optional - will use heuristics if missing)")
@click.option("--model-path", default="data/outputs/model.joblib", show_default=True,
              help="Path to save trained model")
@click.option("--duplicates-path", default="data/outputs/high_confidence.csv", show_default=True,
              help="Path to save high-confidence duplicate pairs")
@click.option("--confidence-threshold", type=float, default=0.9, show_default=True,
              help="Minimum probability threshold for high-confidence pairs")
@click.option("--log-path", default=LOG_PATH, show_default=True,
              help="Path to log file")
@click.option("--show-coefficients", is_flag=True, default=False,
              help="Show model coefficients for debugging")
def model(
    features_path: str,
    labels_path: str,
    model_path: str,
    duplicates_path: str,
    confidence_threshold: float,
    log_path: str,
    show_coefficients: bool,
) -> None:
    """
    Train a logistic regression model to score candidate pairs as duplicates.
    
    This step trains a machine learning model using similarity features and
    either provided labels or heuristic examples. The model then scores all
    pairs and identifies high-confidence duplicates for review.
    
    Examples:
    
        # Basic model training with default settings
        python -m src.cli.model
        
        # Custom confidence threshold
        python -m src.cli.model --confidence-threshold 0.8
        
        # Show model coefficients for debugging
        python -m src.cli.model --show-coefficients
        
        # Use custom paths
        python -m src.cli.model --features-path data/my_features.csv --labels-path data/my_labels.csv
    """
    start_time = time.time()
    formatter = ModelFormatter()
    
    try:
        # Initialize model engine
        engine = ModelEngine()
        
        # Display progress
        print(formatter.format_progress("Starting model training and duplicate scoring..."))
        
        # Train model and score pairs
        scored_df, stats = engine.train_and_score(
            features_path=features_path,
            labels_path=labels_path,
            model_path=model_path,
            duplicates_path=duplicates_path,
            confidence_threshold=confidence_threshold
        )
        
        # Display comprehensive results
        results_output = formatter.format_comprehensive_results(stats, model_path, duplicates_path)
        print(results_output)
        
        # Show model coefficients if requested
        if show_coefficients:
            coef_output = formatter.format_model_coefficients(stats)
            if coef_output:
                print("\n" + "\n".join(coef_output))
        
        # Log the run
        summary_stats = engine.get_summary_stats()
        end_time = time.time()
        log_run(
            "model",
            start_time,
            end_time,
            len(scored_df),
            additional_info=json.dumps(summary_stats),
            log_path=log_path,
        )
        
    except FileNotFoundError as e:
        error_msg = f"Input file not found: {e}"
        print(formatter.format_error(error_msg))
        raise click.ClickException(error_msg)
    
    except ValueError as e:
        error_msg = f"Invalid training data: {e}"
        print(formatter.format_error(error_msg))
        raise click.ClickException(error_msg)
    
    except Exception as e:
        error_msg = f"Model training failed: {e}"
        print(formatter.format_error(error_msg))
        raise click.ClickException(error_msg)


if __name__ == "__main__":
    model()
