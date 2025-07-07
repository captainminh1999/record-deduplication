"""Clustering CLI Module

Command-line interface for the clustering step of the record deduplication pipeline.
This module orchestrates the clustering engine and formatter to provide a clean CLI experience.
"""

import time
import json
import click

from ..core.clustering_engine import ClusteringEngine
from ..formatters.clustering_formatter import ClusteringFormatter
from ..utils import log_run, LOG_PATH


@click.command()
@click.option("--features-path", default="data/outputs/features.csv", show_default=True,
              help="Path to similarity features CSV file")
@click.option("--cleaned-path", default="data/outputs/cleaned.csv", show_default=True,
              help="Path to cleaned records CSV file")
@click.option("--eps", type=float, default=0.5, show_default=True,
              help="DBSCAN epsilon parameter (distance threshold)")
@click.option("--min-samples", type=int, default=2, show_default=True,
              help="DBSCAN minimum samples parameter")
@click.option("--output-path", default="data/outputs/clusters.csv", show_default=True,
              help="Path to save clustered records")
@click.option("--scale/--no-scale", default=False, show_default=True,
              help="Standard-scale similarity columns before DBSCAN")
@click.option("--agg-path", default="data/outputs/agg_features.csv", show_default=True,
              help="Where to write aggregated features (with cluster column)")
@click.option("--auto-eps/--no-auto-eps", default=False, show_default=True,
              help="Automatically select eps using k-distance elbow method")
def clustering(
    features_path: str,
    cleaned_path: str,
    eps: float,
    min_samples: int,
    output_path: str,
    scale: bool,
    agg_path: str,
    auto_eps: bool,
) -> None:
    """
    Cluster records based on similarity features using DBSCAN.
    
    This step groups similar records into clusters based on company and domain
    similarity features. It can automatically select optimal parameters using
    k-distance analysis and silhouette scoring.
    
    Examples:
    
        # Basic clustering with default parameters
        python -m src.cli.clustering
        
        # Auto-select optimal parameters with scaling
        python -m src.cli.clustering --auto-eps --scale
        
        # Manual parameter tuning
        python -m src.cli.clustering --eps 0.3 --min-samples 3 --scale
    """
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
        
    except FileNotFoundError as e:
        error_msg = f"Input file not found: {e}"
        print(formatter.format_error(error_msg))
        raise click.ClickException(error_msg)
    
    except ValueError as e:
        error_msg = f"Invalid input data: {e}"
        print(formatter.format_error(error_msg))
        raise click.ClickException(error_msg)
    
    except Exception as e:
        error_msg = f"Clustering failed: {e}"
        print(formatter.format_error(error_msg))
        raise click.ClickException(error_msg)


if __name__ == "__main__":
    clustering()
