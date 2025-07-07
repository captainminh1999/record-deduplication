"""OpenAI Cluster Review CLI Module

Command-line interface for AI-powered cluster review.
This module orchestrates the OpenAI engine and formatter to provide a clean CLI experience.
"""

import time
import json
import click

from ..core.openai_engine import OpenAIEngine, OpenAIConfig
from ..formatters.openai_formatter import OpenAIFormatter
from ..utils import log_run, LOG_PATH

# Default model
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"


@click.command()
@click.option("--clusters-path", default="data/outputs/clusters.csv", show_default=True,
              help="Path to clusters CSV file")
@click.option("--review-path", default="data/outputs/gpt_review.json", show_default=True,
              help="Path to save cluster review results")
@click.option("--model", default=DEFAULT_MODEL, show_default=True,
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
def cluster_review(
    clusters_path: str,
    review_path: str,
    model: str,
    max_workers: int,
    exclude_clusters: tuple,
    exclude_noise: bool,
    min_cluster_size: int,
    max_cluster_size: int,
    sample_large_clusters: int,
    log_path: str,
) -> None:
    """
    Review DBSCAN clusters with AI for validation.
    
    This step uses OpenAI to analyze clusters and determine if they represent
    valid groupings of duplicate records or if they should be split.
    
    Examples:
    
        # Basic cluster review
        python -m src.cli.openai_cluster_review
        
        # Custom model and settings
        python -m src.cli.openai_cluster_review --model gpt-4 --min-cluster-size 3
        
        # Exclude specific clusters
        python -m src.cli.openai_cluster_review --exclude-clusters 5 --exclude-clusters 12
        
        # Sample large clusters
        python -m src.cli.openai_cluster_review --sample-large-clusters 10
    """
    start_time = time.time()
    formatter = OpenAIFormatter()
    
    try:
        # Initialize OpenAI engine
        engine = OpenAIEngine()
        
        # Display progress
        print(formatter.format_progress("Starting AI cluster review..."))
        
        # Load clusters data
        import pandas as pd
        import os
        
        if not os.path.exists(clusters_path):
            raise FileNotFoundError(f"Clusters file not found: {clusters_path}")
        
        clusters_df = pd.read_csv(clusters_path)
        
        # Configure analysis
        config = OpenAIConfig(
            model=model,
            max_workers=max_workers,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            exclude_noise=exclude_noise,
            sample_large_clusters=sample_large_clusters
        )
        
        # Filter excluded clusters
        if exclude_clusters:
            clusters_df = clusters_df[~clusters_df['cluster'].isin(exclude_clusters)]
        
        # Count clusters to analyze
        cluster_groups = clusters_df.groupby('cluster')
        valid_clusters = [
            (cluster_id, group) for cluster_id, group in cluster_groups
            if not (exclude_noise and cluster_id == -1)
            and len(group) >= min_cluster_size
            and (max_cluster_size is None or len(group) <= max_cluster_size)
        ]
        
        print(formatter.format_cluster_review_start(len(valid_clusters)))
        
        # Review clusters with AI
        result = engine.review_clusters(clusters_df, config)
        
        # Save results
        os.makedirs(os.path.dirname(review_path), exist_ok=True)
        with open(review_path, 'w') as f:
            json.dump(result.review_data, f, indent=2)
        
        # Display comprehensive results
        results_output = formatter.format_comprehensive_cluster_results(result.review_data, review_path)
        print(results_output)
        
        # Log the run
        end_time = time.time()
        log_run(
            "openai_cluster_review",
            start_time,
            end_time,
            result.stats.processed_items,
            additional_info=json.dumps({
                "clusters_analyzed": result.review_data.get("clusters_analyzed", 0),
                "valid_clusters": result.review_data.get("valid_clusters", 0),
                "api_calls": result.stats.total_calls,
                "tokens_used": result.stats.total_tokens.get("total", 0),
                "model_used": model
            }),
            log_path=log_path,
        )
        
    except ImportError as e:
        if "openai" in str(e).lower():
            error_msg = "OpenAI package not installed. Install with: pip install openai"
        else:
            error_msg = f"Missing dependency: {e}"
        print(formatter.format_error(error_msg))
        raise click.ClickException(error_msg)
    
    except FileNotFoundError as e:
        error_msg = f"Input file not found: {e}"
        print(formatter.format_error(error_msg))
        raise click.ClickException(error_msg)
    
    except Exception as e:
        error_msg = f"Cluster review failed: {e}"
        print(formatter.format_error(error_msg))
        raise click.ClickException(error_msg)


if __name__ == "__main__":
    cluster_review()
