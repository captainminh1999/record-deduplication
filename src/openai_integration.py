"""
Legacy wrapper for OpenAI cluster review functionality.

This module bridges to the new modular architecture while maintaining backward compatibility.
For new code, use: src.cli.openai_cluster_review or src.core.openai_engine
"""

from __future__ import annotations

import os
import sys
import time
import json
from typing import Any, Iterable

import click
import pandas as pd

# Import the new modular components
from .core.openai_engine import OpenAIEngine
from .core.openai_types import OpenAIConfig
from .formatters.openai_formatter import OpenAIFormatter
from .utils import log_run, LOG_PATH

# Default chat model used across this module
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"


def translate_to_english(texts, model: str = DEFAULT_MODEL) -> list[str]:
    """Bridge function for translating company names to English."""
    # Initialize the new modular components
    engine = OpenAIEngine()
    
    # Configure operation
    config = OpenAIConfig(model=model)
    
    # Convert to list if needed
    if not isinstance(texts, (list, tuple)):
        texts = list(texts) if isinstance(texts, Iterable) else [texts]
    
    # Run translation using modular engine
    result = engine.translate_to_english(texts, config)
    
    return result.translations


def main(
    clusters_path: str = "data/outputs/clusters.csv",
    review_path: str = "data/outputs/gpt_review.json",
    openai_model: str = DEFAULT_MODEL,
    log_path: str = LOG_PATH,
    max_workers: int = 10,
    exclude_clusters: tuple = (),
    exclude_noise: bool = True,
    min_cluster_size: int = 2,
    max_cluster_size: int | None = None,
    sample_large_clusters: int | None = None,
) -> None:
    """Review DBSCAN clusters with GPT for validation."""

    print("🤖 Starting GPT cluster analysis...")
    start_time = time.time()

    if not os.path.exists(clusters_path):
        raise FileNotFoundError(f"Clusters file not found: {clusters_path}")

    clusters_df = pd.read_csv(clusters_path)
    # Ensure record_id is a column, not an index
    if "record_id" not in clusters_df.columns and clusters_df.index.name == "record_id":
        clusters_df = clusters_df.reset_index()

    # Initialize the new modular components
    engine = OpenAIEngine()
    formatter = OpenAIFormatter()
    
    # Configure operation
    config = OpenAIConfig(
        model=openai_model,
        max_workers=max_workers,
        exclude_clusters=tuple(exclude_clusters),
        exclude_noise=exclude_noise,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        sample_large_clusters=sample_large_clusters
    )
    
    print(formatter.format_progress("Analyzing clusters with AI..."))
    
    # Run cluster review using modular engine
    result = engine.review_clusters(clusters_df, config)
    
    # Save results
    os.makedirs(os.path.dirname(review_path), exist_ok=True)
    with open(review_path, "w", encoding="utf-8") as fh:
        json.dump(result.cluster_reviews, fh, indent=2)
    
    # Display results using formatter
    formatter.format_cluster_review_results({"cluster_reviews": result.cluster_reviews, "stats": result.stats})
    
    print(f"\n💾 Files Created:")
    print(f"  • GPT review:            {review_path}")
    
    end_time = time.time()
    log_run("openai_integration", start_time, end_time, len(result.cluster_reviews), log_path=log_path)


@click.command()
@click.option("--clusters-path", default="data/outputs/clusters.csv", show_default=True,
              help="Path to clusters CSV file")
@click.option("--review-path", default="data/outputs/gpt_review.json", show_default=True,
              help="Path to save cluster review results")
@click.option("--openai-model", default=DEFAULT_MODEL, show_default=True,
              help="OpenAI model to use for analysis")
@click.option("--max-workers", default=10, show_default=True,
              help="Number of parallel OpenAI requests")
@click.option("--exclude-clusters", multiple=True, type=int,
              help="Cluster IDs to exclude from analysis (can be used multiple times)")
@click.option("--exclude-noise/--include-noise", default=True, show_default=True,
              help="Exclude noise cluster (-1)")
@click.option("--min-cluster-size", type=int, default=2, show_default=True,
              help="Minimum cluster size to analyze")
@click.option("--max-cluster-size", type=int, default=None,
              help="Maximum cluster size to analyze")
@click.option("--sample-large-clusters", type=int, default=None,
              help="Sample size for large clusters")
@click.option("--log-path", default=LOG_PATH, show_default=True,
              help="Path to log file")
def cli_main(
    clusters_path: str,
    review_path: str,
    openai_model: str,
    max_workers: int,
    exclude_clusters: tuple,
    exclude_noise: bool,
    min_cluster_size: int,
    max_cluster_size: int,
    sample_large_clusters: int,
    log_path: str,
) -> None:
    """CLI entry point for cluster review (legacy compatibility)."""
    main(
        clusters_path=clusters_path,
        review_path=review_path,
        openai_model=openai_model,
        log_path=log_path,
        max_workers=max_workers,
        exclude_clusters=exclude_clusters,
        exclude_noise=exclude_noise,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        sample_large_clusters=sample_large_clusters,
    )


if __name__ == "__main__":
    cli_main()