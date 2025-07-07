"""Step 6 of 6: Clustering (Optional Grouping)

Clusters records based on similarity features using DBSCAN. In this implementation, domain similarity is given double weight before clustering. See README for details.

This module now serves as a bridge to the new modular architecture.
For new development, use the modular components:
- src.core.clustering_engine (business logic)
- src.formatters.clustering_formatter (terminal output)
- src.cli.clustering (CLI orchestration)
"""

from __future__ import annotations

import time
import json

import click
import pandas as pd

from .core.clustering_engine import ClusteringEngine
from .formatters.clustering_formatter import ClusteringFormatter
from .utils import log_run, LOG_PATH


@click.command()
@click.option("--features-path", default="data/outputs/features.csv", show_default=True)
@click.option("--cleaned-path", default="data/outputs/cleaned.csv", show_default=True)
@click.option("--eps", type=float, default=0.5, show_default=True)
@click.option("--min-samples", type=int, default=2, show_default=True)
@click.option("--output-path", default="data/outputs/clusters.csv", show_default=True)
@click.option(
    "--scale/--no-scale",
    default=False,
    show_default=True,
    help="Standard-scale similarity columns before DBSCAN",
)
@click.option(
    "--agg-path",
    default="data/outputs/agg_features.csv",
    show_default=True,
    help="Where to write aggregated features (with cluster column)",
)
@click.option(
    "--auto-eps/--no-auto-eps",
    default=False,
    show_default=True,
    help="Automatically select eps using k-distance elbow method",
)
def cli(
    features_path: str,
    cleaned_path: str,
    eps: float,
    min_samples: int,
    output_path: str,
    scale: bool,
    agg_path: str,
    auto_eps: bool,
) -> None:
    """CLI wrapper for :func:`main`."""
    main(features_path, cleaned_path, eps, min_samples, output_path, scale, agg_path, auto_eps)


def main(
    features_path: str = "data/outputs/features.csv",
    cleaned_path: str = "data/outputs/cleaned.csv",
    eps: float = 0.004,
    min_samples: int = 3,
    output_path: str = "data/outputs/clusters.csv",
    scale: bool = False,
    agg_path: str = "data/outputs/agg_features.csv",
    auto_eps: bool = False,
) -> pd.DataFrame:
    """Generate DBSCAN clusters from similarity features."""
    
    start_time = time.time()
    formatter = ClusteringFormatter()
    
    try:
        # Initialize clustering engine
        engine = ClusteringEngine()
        
        # Display auto-eps progress if enabled
        if auto_eps:
            print(formatter.format_progress("Auto-selecting optimal clustering parameters..."))
        
        # Perform clustering
        clustered_records, agg_features, stats = engine.cluster_records(
            features_path=features_path,
            cleaned_path=cleaned_path,
            eps=eps,
            min_samples=min_samples,
            scale=scale,
            auto_eps=auto_eps
        )
        
        # Display auto-eps iteration details if available
        if auto_eps and stats.get("iterations"):
            auto_details = formatter.format_auto_eps_details(stats)
            for line in auto_details:
                print(line)
        
        # Save results
        engine.save_results(clustered_records, agg_features, output_path, agg_path)
        
        # Display file output messages
        print(formatter.format_file_output(output_path, len(clustered_records)))
        print(formatter.format_agg_output(agg_path))
        
        # Display comprehensive results
        results_output = formatter.format_comprehensive_results(stats, output_path, agg_path)
        print(results_output)
        
        # Log the run
        simple_stats = engine.get_simple_stats()
        end_time = time.time()
        log_run(
            "clustering",
            start_time,
            end_time,
            len(clustered_records),
            additional_info=json.dumps(simple_stats),
            log_path=LOG_PATH,
        )
        
        return clustered_records
        
    except Exception as e:
        print(formatter.format_error(f"Clustering failed: {e}"))
        raise


if __name__ == "__main__":  # pragma: no cover - sanity run
    cli()
