"""
Legacy wrapper for OpenAI deduplication functionality.

This module bridges to the new modular architecture while maintaining backward compatibility.
For new code, use: src.cli.openai_deduplication or src.core.openai_engine
"""

from __future__ import annotations

import os
import sys
import time
import json
from typing import Any

import click
import pandas as pd

# Import the new modular components
from .core.openai_engine import OpenAIEngine
from .core.openai_types import OpenAIConfig
from .formatters.openai_formatter import OpenAIFormatter
from .utils import log_run, LOG_PATH

# Default chat model used across this module
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"


def main(
    features_path: str = "data/outputs/features.csv",
    cleaned_path: str = "data/outputs/cleaned.csv",
    output_path: str = "data/outputs/unique_records.csv",
    analysis_path: str = "data/outputs/deduplication_analysis.json",
    similarity_threshold: float = 0.6,
    confidence_threshold: float = 0.7,
    sample_size: int | None = None,
    batch_size: int = 10,
    max_workers: int = 10,
    openai_model: str = DEFAULT_MODEL,
    log_path: str = LOG_PATH,
) -> None:
    """Perform AI-powered record deduplication."""

    print("🤖 Starting AI-powered deduplication...")
    start_time = time.time()

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned data file not found: {cleaned_path}")

    # Load data
    features_df = pd.read_csv(features_path)
    cleaned_df = pd.read_csv(cleaned_path)

    # Initialize the new modular components
    engine = OpenAIEngine()
    formatter = OpenAIFormatter()
    
    # Configure operation
    config = OpenAIConfig(
        model=openai_model,
        similarity_threshold=similarity_threshold,
        confidence_threshold=confidence_threshold,
        sample_size=sample_size,
        batch_size=batch_size,
        max_workers=max_workers
    )
    
    print(formatter.format_progress("Analyzing record pairs with AI..."))
    
    # Run deduplication using modular engine
    result = engine.deduplicate_records(features_df, cleaned_df, config)
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.unique_records.to_csv(output_path, index=False)
    
    os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
    with open(analysis_path, "w", encoding="utf-8") as fh:
        json.dump(result.analysis, fh, indent=2)
    
    # Display results using formatter
    formatter.format_deduplication_results(result.analysis)
    
    print(f"\n💾 Files Created:")
    print(f"  • Unique records:        {output_path}")
    print(f"  • Analysis:              {analysis_path}")
    
    end_time = time.time()
    log_run("openai_deduplication", start_time, end_time, len(result.unique_records), log_path=log_path)


@click.command()
@click.option("--features-path", default="data/outputs/features.csv", show_default=True,
              help="Path to similarity features CSV file")
@click.option("--cleaned-path", default="data/outputs/cleaned.csv", show_default=True,
              help="Path to cleaned records CSV file")
@click.option("--output-path", default="data/outputs/unique_records.csv", show_default=True,
              help="Path to save unique/deduplicated records")
@click.option("--analysis-path", default="data/outputs/deduplication_analysis.json", show_default=True,
              help="Path to save detailed analysis results")
@click.option("--similarity-threshold", default=0.6, show_default=True, type=float,
              help="Minimum similarity score to consider for AI analysis")
@click.option("--confidence-threshold", default=0.7, show_default=True, type=float,
              help="Minimum AI confidence to merge records")
@click.option("--sample-size", default=None, type=int,
              help="Limit analysis to this many pairs (for testing)")
@click.option("--batch-size", default=10, show_default=True,
              help="Number of pairs per batch for parallel processing")
@click.option("--max-workers", default=10, show_default=True,
              help="Number of parallel workers for API calls")
@click.option("--openai-model", default=DEFAULT_MODEL, show_default=True,
              help="OpenAI model to use for analysis")
@click.option("--log-path", default=LOG_PATH, show_default=True,
              help="Path to log file")
def cli_main(
    features_path: str,
    cleaned_path: str,
    output_path: str,
    analysis_path: str,
    similarity_threshold: float,
    confidence_threshold: float,
    sample_size: int,
    batch_size: int,
    max_workers: int,
    openai_model: str,
    log_path: str,
) -> None:
    """CLI entry point for AI deduplication (legacy compatibility)."""
    main(
        features_path=features_path,
        cleaned_path=cleaned_path,
        output_path=output_path,
        analysis_path=analysis_path,
        similarity_threshold=similarity_threshold,
        confidence_threshold=confidence_threshold,
        sample_size=sample_size,
        batch_size=batch_size,
        max_workers=max_workers,
        openai_model=openai_model,
        log_path=log_path,
    )


if __name__ == "__main__":
    cli_main()
