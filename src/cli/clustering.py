"""Clustering CLI Module

Command-line interface for the clustering step of the record deduplication pipeline.
This module orchestrates the clustering engine and formatter to provide a clean CLI experience.
"""

import time
import json
import click

from ..core.clustering_engine_v2 import ClusteringEngine
from ..formatters.clustering_formatter import ClusteringFormatter
from ..logging import log_run, LOG_PATH


@click.command()
@click.option("--features-path", default="data/outputs/features.csv", show_default=True,
              help="Path to similarity features CSV file")
@click.option("--cleaned-path", default="data/outputs/cleaned.csv", show_default=True,
              help="Path to cleaned records CSV file")
@click.option("--eps", type=float, default=0.1, show_default=True,
              help="DBSCAN epsilon parameter (distance threshold)")
@click.option("--min-samples", type=int, default=2, show_default=True,
              help="DBSCAN minimum samples parameter")
@click.option("--output-path", default="data/outputs/clusters.csv", show_default=True,
              help="Path to save clustered records")
@click.option("--scale/--no-scale", default=True, show_default=True,
              help="Use enhanced scaling (PowerTransformer + StandardScaler)")
@click.option("--agg-path", default="data/outputs/agg_features.csv", show_default=True,
              help="Where to write aggregated features (with cluster column)")
@click.option("--hierarchical/--no-hierarchical", default=False, show_default=True,
              help="Apply hierarchical clustering to break large clusters")
@click.option("--max-cluster-size", type=int, default=50, show_default=True,
              help="Maximum cluster size before subdivision (continues until all clusters meet this)")
@click.option("--max-depth", type=int, default=20, show_default=True,
              help="Maximum absolute depth limit (safety limit, use adaptive depth)")
@click.option("--adaptive-depth/--no-adaptive-depth", default=True, show_default=True,
              help="Use adaptive depth (continues until max-cluster-size is met)")
@click.option("--timeout", type=int, default=300, show_default=True,
              help="Timeout in seconds for hierarchical clustering")
@click.option("--performance-mode/--no-performance-mode", default=True, show_default=True,
              help="Use fast strategies for very large clusters")
def clustering(
    features_path: str,
    cleaned_path: str,
    eps: float,
    min_samples: int,
    output_path: str,
    scale: bool,
    agg_path: str,
    hierarchical: bool,
    max_cluster_size: int,
    max_depth: int,
    adaptive_depth: bool,
    timeout: int,
    performance_mode: bool,
) -> None:
    """
    Cluster records based on similarity features using DBSCAN with advanced hierarchical subdivision.
    
    This command supports both traditional fixed-depth hierarchical clustering and new adaptive depth
    clustering that continues until all clusters meet the size constraint.
    
    This step groups similar records into clusters based on company and domain
    similarity features. It uses advanced feature engineering and enhanced scaling
    to improve clustering quality, particularly targeting the eps range of 0.01-0.15.
    
    Hierarchical clustering can break large clusters into smaller, more manageable ones.
    
    Examples:
    
        # Basic clustering with enhanced features
        python -m src.cli.clustering
        
        # Adaptive hierarchical clustering (recommended)
        python -m src.cli.clustering --hierarchical --max-cluster-size 10 --adaptive-depth
        
        # Manual parameter tuning for more granular clusters
        python -m src.cli.clustering --eps 0.1 --min-samples 2 --scale
        
        # Hierarchical clustering to break large clusters
        python -m src.cli.clustering --eps 0.15 --hierarchical --max-cluster-size 30
        
        # Fine-tuning for domain-based clustering
        python -m src.cli.clustering --eps 0.05 --min-samples 2 --scale --hierarchical
    """
    start_time = time.time()
    formatter = ClusteringFormatter()
    
    try:
        # Initialize clustering engine
        engine = ClusteringEngine()
        
        # Choose clustering method based on hierarchical flag
        if hierarchical:
            if adaptive_depth:
                # Use the new adaptive hierarchical clusterer
                from src.core.clustering.hierarchical.adaptive_clusterer_v3 import AdaptiveHierarchicalClusterer
                
                # Initialize adaptive clusterer
                adaptive_clusterer = AdaptiveHierarchicalClusterer(timeout_seconds=timeout)
                
                # Set performance mode
                if performance_mode:
                    adaptive_clusterer.set_performance_mode()
                
                # Perform adaptive hierarchical clustering
                clustered_records, agg_features, stats = adaptive_clusterer.cluster_dataset(
                    features_path=features_path,
                    cleaned_path=cleaned_path,
                    eps=eps,
                    min_samples=min_samples,
                    scale=scale,
                    max_cluster_size=max_cluster_size,
                    performance_mode=performance_mode
                )
            else:
                # Use traditional fixed-depth hierarchical clustering
                clustered_records, agg_features, stats = engine.hierarchical_clustering(
                    features_path=features_path,
                    cleaned_path=cleaned_path,
                    eps=eps,
                    min_samples=min_samples,
                    scale=scale,
                    max_cluster_size=max_cluster_size,
                    max_depth=max_depth
                )
        else:
            # Perform clustering with enhanced features
            clustered_records, agg_features, stats = engine.cluster_records(
                features_path=features_path,
                cleaned_path=cleaned_path,
                eps=eps,
                min_samples=min_samples,
                scale=scale
            )
        
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
