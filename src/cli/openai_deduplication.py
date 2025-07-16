"""OpenAI Deduplication CLI Module

Command-line interface for AI-powered record deduplication.
This module orchestrates the OpenAI engine and formatter to provide a clean CLI experience.
"""

import time
import json
import click
from tqdm import tqdm

from ..core.openai_engine import OpenAIEngine, OpenAIConfig
from ..formatters.openai_formatter import OpenAIFormatter
from ..logging import log_run, LOG_PATH

# Default model
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"


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
@click.option("--model", default=DEFAULT_MODEL, show_default=True,
              help="OpenAI model to use for analysis")
@click.option("--log-path", default=LOG_PATH, show_default=True,
              help="Path to log file")
def openai_deduplication(
    features_path: str,
    cleaned_path: str,
    output_path: str,
    analysis_path: str,
    similarity_threshold: float,
    confidence_threshold: float,
    sample_size: int,
    batch_size: int,
    max_workers: int,
    model: str,
    log_path: str,
) -> None:
    """
    Perform AI-powered record deduplication using similarity features.
    
    This step uses OpenAI to analyze pairs with similarity scores and make
    intelligent deduplication decisions, generating a list of unique records
    by merging duplicates.
    
    Examples:
    
        # Basic AI deduplication
        python -m src.cli.openai_deduplication
        
        # Custom thresholds and model
        python -m src.cli.openai_deduplication --similarity-threshold 0.7 --confidence-threshold 0.8 --model gpt-4
        
        # Test with sample data
        python -m src.cli.openai_deduplication --sample-size 50 --batch-size 5
        
        # High-performance processing
        python -m src.cli.openai_deduplication --max-workers 20 --batch-size 20
    """
    start_time = time.time()
    formatter = OpenAIFormatter()
    
    try:
        # Initialize OpenAI engine
        engine = OpenAIEngine()
        
        # Display progress
        print(formatter.format_progress("Starting AI-powered deduplication..."))
        
        # Load data
        import pandas as pd
        import os
        
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")
        if not os.path.exists(cleaned_path):
            raise FileNotFoundError(f"Cleaned data file not found: {cleaned_path}")
        
        features_df = pd.read_csv(features_path)
        records_df = pd.read_csv(cleaned_path).set_index('record_id')
        
        # Configure deduplication
        config = OpenAIConfig(
            model=model,
            max_workers=max_workers,
            batch_size=batch_size,
            similarity_threshold=similarity_threshold,
            confidence_threshold=confidence_threshold
        )
        
        # Add sample size to config if provided
        if sample_size:
            config.sample_size = sample_size
        
        # Count candidate pairs
        candidate_pairs = features_df[
            (features_df.get('company_sim', 0) >= similarity_threshold)
        ]
        
        if sample_size:
            candidate_pairs = candidate_pairs.sample(n=min(sample_size, len(candidate_pairs)))
        
        print(f"🔗 Analyzing {len(candidate_pairs)} candidate pairs for merging...")
        print(f"🎯 Using {max_workers} parallel workers with batch size {batch_size}")
        
        # Calculate number of batches for progress bar
        total_pairs = len(candidate_pairs)
        if sample_size:
            total_pairs = min(sample_size, total_pairs)
        num_batches = (total_pairs + batch_size - 1) // batch_size  # Ceiling division
        
        # Initialize progress bar
        progress_bar = tqdm(
            total=num_batches,
            desc="🤖 Processing AI batches",
            unit="batch",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}], {rate_fmt}"
        )
        
        def update_progress(completed: int, total: int):
            """Update the progress bar."""
            progress_bar.n = completed
            progress_bar.refresh()
        
        # Perform AI deduplication with progress tracking
        result = engine.deduplicate_records(
            features_df, 
            records_df, 
            config, 
            sample_size=sample_size, 
            progress_callback=update_progress
        )
        
        # Close progress bar
        progress_bar.close()
        
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
        
        # Reset index for output
        output_df = result.unique_records.copy()
        if 'record_id' not in output_df.columns and output_df.index.name == 'record_id':
            output_df = output_df.reset_index()
        
        output_df.to_csv(output_path, index=False)
        
        with open(analysis_path, 'w') as f:
            json.dump(result.analysis, f, indent=2)
        
        # Calculate actual statistics
        original_count = len(records_df)
        unique_count = len(result.unique_records)
        pairs_analyzed = result.analysis.get('summary', {}).get('total_pairs_analyzed', 0)
        pairs_merged = result.analysis.get('summary', {}).get('pairs_merged', 0)
        reduction_count = original_count - unique_count
        reduction_rate = (reduction_count / original_count * 100) if original_count > 0 else 0
        
        # Display comprehensive results
        print("🎯 AI Deduplication Complete!")
        print("─" * 60)
        print("📊 Deduplication Summary:")
        print(f"  • Original records:      {original_count:,}")
        print(f"  • Unique records:        {unique_count:,}")
        print(f"  • Records merged:        {reduction_count:,}")
        print(f"  • Reduction rate:        {reduction_rate:.1f}%")
        print(f"  • Model used:           {model}")
        
        print("🔗 Merge Analysis:")
        print(f"  • Pairs analyzed:        {pairs_analyzed:,}")
        print(f"  • Confident merges:      {pairs_merged:,}")
        print(f"  • Merge groups created:  {pairs_merged:,}")
        
        print("🔌 API Usage:")
        print(f"  • Total API calls:       {result.stats.total_calls:,}")
        print(f"  • Successful calls:      {result.stats.successful_calls:,}")
        print(f"  • Failed calls:          {result.stats.failed_calls:,}")
        print(f"  • Tokens used:           {result.stats.total_tokens.get('total', 0):,}")
        print(f"    - Prompt tokens:       {result.stats.total_tokens.get('prompt', 0):,}")
        print(f"    - Completion tokens:   {result.stats.total_tokens.get('completion', 0):,}")
        
        # Calculate processing times
        total_processing_time = sum(result.stats.durations) if result.stats.durations else 0
        avg_duration = total_processing_time / len(result.stats.durations) if result.stats.durations else 0
        
        print(f"  • Total processing time: {total_processing_time:.1f}s")
        print(f"  • Average call time:     {avg_duration:.1f}s")
        print(f"  • Total cost estimate:   ${result.stats.total_cost:.4f}")
        
        print("💾 Files Created:")
        print(f"  • Unique records: {output_path}")
        print(f"  • Analysis data: {analysis_path}")
        
        # Log the run
        end_time = time.time()
        log_run(
            "openai_deduplication",
            start_time,
            end_time,
            unique_count,
            additional_info=json.dumps({
                "original_records": original_count,
                "unique_records": unique_count,
                "pairs_analyzed": pairs_analyzed,
                "pairs_merged": pairs_merged,
                "reduction_rate": reduction_rate / 100,
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
        error_msg = f"AI deduplication failed: {e}"
        print(formatter.format_error(error_msg))
        raise click.ClickException(error_msg)


if __name__ == "__main__":
    openai_deduplication()
