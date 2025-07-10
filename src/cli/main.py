"""
Main CLI entry point for the record deduplication pipeline.

This demonstrates the new modular architecture where each pipeline step
has its own CLI module with clean separation of concerns.
"""

import argparse
import sys
from typing import Optional

from .preprocess import PreprocessCLI
from .blocking import BlockingCLI
from .similarity import SimilarityCLI


class MainCLI:
    """Main CLI that orchestrates different pipeline steps."""
    
    def __init__(self):
        self.preprocess_cli = PreprocessCLI()
        self.blocking_cli = BlockingCLI()
        self.similarity_cli = SimilarityCLI()
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser with subcommands."""
        parser = argparse.ArgumentParser(
            description="Record deduplication pipeline",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Available commands:
  preprocess         Clean and prepare data for deduplication
  blocking           Generate candidate pairs for comparison
  similarity         Compute similarity features for pairs
  model              Train model and score duplicate pairs
  clustering         Group similar records into clusters
  reporting          Generate Excel review file
  openai-cluster     Review clusters with OpenAI
  openai-dedup       AI-powered record deduplication
  
Examples:
  python -m src.cli preprocess data/sample_input.csv
  python -m src.cli blocking data/outputs/cleaned.csv
  python -m src.cli similarity data/outputs/cleaned.csv data/outputs/pairs.csv
  python -m src.cli model --confidence-threshold 0.8
  python -m src.cli clustering --auto-eps --scale
  python -m src.cli reporting --include-details
  python -m src.cli openai-cluster --min-cluster-size 3
  python -m src.cli openai-dedup --confidence-threshold 0.8
            """
        )
        
        subparsers = parser.add_subparsers(
            dest="command",
            help="Available commands",
            metavar="COMMAND"
        )
        
        # Preprocess subcommand
        preprocess_parser = subparsers.add_parser(
            "preprocess",
            help="Clean and prepare data for deduplication",
            parents=[self.preprocess_cli.create_parser()],
            add_help=False
        )
        
        # Blocking subcommand
        blocking_parser = subparsers.add_parser(
            "blocking",
            help="Generate candidate pairs for comparison",
            parents=[self.blocking_cli.create_parser()],
            add_help=False
        )
        
        # Similarity subcommand
        similarity_parser = subparsers.add_parser(
            "similarity",
            help="Compute similarity features for pairs",
            parents=[self.similarity_cli.create_parser()],
            add_help=False
        )
        
        # Model subcommand
        model_parser = subparsers.add_parser(
            "model",
            help="Train model and score duplicate pairs"
        )
        model_parser.add_argument("--features-path", default="data/outputs/features.csv")
        model_parser.add_argument("--labels-path", default="data/outputs/labels.csv")
        model_parser.add_argument("--model-path", default="data/outputs/model.joblib")
        model_parser.add_argument("--duplicates-path", default="data/outputs/high_confidence.csv")
        model_parser.add_argument("--confidence-threshold", type=float, default=0.9)
        model_parser.add_argument("--log-path", default="data/run_history.log")
        model_parser.add_argument("--show-coefficients", action="store_true")
        
        # Clustering subcommand
        clustering_parser = subparsers.add_parser(
            "clustering",
            help="Group similar records into clusters"
        )
        clustering_parser.add_argument("--features-path", default="data/outputs/features.csv")
        clustering_parser.add_argument("--cleaned-path", default="data/outputs/cleaned.csv")
        clustering_parser.add_argument("--output-path", default="data/outputs/clusters.csv")
        clustering_parser.add_argument("--eps", type=float, default=0.5)
        clustering_parser.add_argument("--min-samples", type=int, default=2)
        clustering_parser.add_argument("--scale", action="store_true")
        clustering_parser.add_argument("--agg-path", default="data/outputs/agg_features.csv")
        clustering_parser.add_argument("--auto-eps", action="store_true")
        clustering_parser.add_argument("--log-path", default="data/run_history.log")
        
        # Reporting subcommand
        reporting_parser = subparsers.add_parser(
            "reporting",
            help="Generate Excel review file"
        )
        reporting_parser.add_argument("--duplicates-path", default="data/outputs/high_confidence.csv")
        reporting_parser.add_argument("--cleaned-path", default="data/outputs/cleaned.csv")
        reporting_parser.add_argument("--report-path", default="data/outputs/manual_review.xlsx")
        reporting_parser.add_argument("--gpt-review-path", default="data/outputs/gpt_review.json")
        reporting_parser.add_argument("--log-path", default="data/run_history.log")
        reporting_parser.add_argument("--include-details", action="store_true")
        
        # OpenAI Cluster Review subcommand
        openai_cluster_parser = subparsers.add_parser(
            "openai-cluster",
            help="Review clusters with OpenAI"
        )
        openai_cluster_parser.add_argument("--clusters-path", default="data/outputs/clusters.csv")
        openai_cluster_parser.add_argument("--review-path", default="data/outputs/gpt_review.json")
        openai_cluster_parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
        openai_cluster_parser.add_argument("--max-workers", type=int, default=10)
        openai_cluster_parser.add_argument("--exclude-clusters", type=int, nargs="*", default=[])
        openai_cluster_parser.add_argument("--exclude-noise", action="store_true", default=True)
        openai_cluster_parser.add_argument("--min-cluster-size", type=int, default=2)
        openai_cluster_parser.add_argument("--max-cluster-size", type=int, default=None)
        openai_cluster_parser.add_argument("--sample-large-clusters", type=int, default=None)
        
        # OpenAI Deduplication subcommand
        openai_dedup_parser = subparsers.add_parser(
            "openai-dedup",
            help="AI-powered record deduplication"
        )
        openai_dedup_parser.add_argument("--features-path", default="data/outputs/features.csv")
        openai_dedup_parser.add_argument("--cleaned-path", default="data/outputs/cleaned.csv")
        openai_dedup_parser.add_argument("--output-path", default="data/outputs/unique_records.csv")
        openai_dedup_parser.add_argument("--analysis-path", default="data/outputs/deduplication_analysis.json")
        openai_dedup_parser.add_argument("--similarity-threshold", type=float, default=0.6)
        openai_dedup_parser.add_argument("--confidence-threshold", type=float, default=0.7)
        openai_dedup_parser.add_argument("--sample-size", type=int, default=None)
        openai_dedup_parser.add_argument("--batch-size", type=int, default=10)
        openai_dedup_parser.add_argument("--max-workers", type=int, default=10)
        openai_dedup_parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
        
        return parser
    
    def run(self, args: Optional[list] = None) -> int:
        """Run the main CLI."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        if not parsed_args.command:
            parser.print_help()
            return 1
        
        if parsed_args.command == "preprocess":
            # Extract preprocess arguments
            preprocess_args = [
                parsed_args.input_file,
                "--output", parsed_args.output
            ]
            
            if parsed_args.normalize:
                preprocess_args.append("--normalize")
            if parsed_args.deduplicate:
                preprocess_args.append("--deduplicate")
            if parsed_args.quiet:
                preprocess_args.append("--quiet")
            
            return self.preprocess_cli.run(preprocess_args)
        
        if parsed_args.command == "blocking":
            # Extract blocking arguments
            blocking_args = [
                parsed_args.input_file,
                "--output", parsed_args.output
            ]
            
            if parsed_args.quiet:
                blocking_args.append("--quiet")
            
            return self.blocking_cli.run(blocking_args)
        
        if parsed_args.command == "similarity":
            # Extract similarity arguments
            similarity_args = [
                parsed_args.cleaned_file,
                parsed_args.pairs_file,
                "--output", parsed_args.output
            ]
            
            if parsed_args.quiet:
                similarity_args.append("--quiet")
            
            return self.similarity_cli.run(similarity_args)
        
        if parsed_args.command == "model":
            # Import and run model CLI
            from .model import model
            try:
                model([
                    "--features-path", parsed_args.features_path,
                    "--labels-path", parsed_args.labels_path,
                    "--model-path", parsed_args.model_path,
                    "--duplicates-path", parsed_args.duplicates_path,
                    "--confidence-threshold", str(parsed_args.confidence_threshold),
                    "--log-path", parsed_args.log_path
                ] + (["--show-coefficients"] if parsed_args.show_coefficients else []),
                standalone_mode=False)
                return 0
            except SystemExit as e:
                return int(e.code) if e.code else 0
        
        if parsed_args.command == "clustering":
            # Import and run clustering CLI
            from .clustering import clustering
            try:
                clustering([
                    "--features-path", parsed_args.features_path,
                    "--cleaned-path", parsed_args.cleaned_path,
                    "--output-path", parsed_args.output_path,
                    "--eps", str(parsed_args.eps),
                    "--min-samples", str(parsed_args.min_samples),
                    "--agg-path", parsed_args.agg_path
                ] + (["--auto-eps"] if parsed_args.auto_eps else ["--no-auto-eps"]) +
                    (["--scale"] if parsed_args.scale else ["--no-scale"]),
                standalone_mode=False)
                return 0
            except SystemExit as e:
                return int(e.code) if e.code else 0
        
        if parsed_args.command == "reporting":
            # Import and run reporting CLI
            from .reporting import reporting
            try:
                reporting([
                    "--duplicates-path", parsed_args.duplicates_path,
                    "--cleaned-path", parsed_args.cleaned_path,
                    "--report-path", parsed_args.report_path,
                    "--gpt-review-path", parsed_args.gpt_review_path,
                    "--log-path", parsed_args.log_path
                ], standalone_mode=False)
                return 0
            except SystemExit as e:
                return int(e.code) if e.code else 0
        
        if parsed_args.command == "openai-cluster":
            # Import and run OpenAI cluster review CLI
            from .openai_cluster_review import cluster_review
            try:
                args_list = [
                    "--clusters-path", parsed_args.clusters_path,
                    "--review-path", parsed_args.review_path,
                    "--model", parsed_args.model,
                    "--max-workers", str(parsed_args.max_workers),
                    "--min-cluster-size", str(parsed_args.min_cluster_size)
                ]
                
                if parsed_args.exclude_clusters:
                    for cluster_id in parsed_args.exclude_clusters:
                        args_list.extend(["--exclude-clusters", str(cluster_id)])
                
                if not parsed_args.exclude_noise:
                    args_list.append("--no-exclude-noise")
                
                if parsed_args.max_cluster_size is not None:
                    args_list.extend(["--max-cluster-size", str(parsed_args.max_cluster_size)])
                
                if parsed_args.sample_large_clusters is not None:
                    args_list.extend(["--sample-large-clusters", str(parsed_args.sample_large_clusters)])
                
                cluster_review(args_list, standalone_mode=False)
                return 0
            except SystemExit as e:
                return int(e.code) if e.code else 0
        
        if parsed_args.command == "openai-dedup":
            # Import and run OpenAI deduplication CLI
            from .openai_deduplication import openai_deduplication
            try:
                args_list = [
                    "--features-path", parsed_args.features_path,
                    "--cleaned-path", parsed_args.cleaned_path,
                    "--output-path", parsed_args.output_path,
                    "--analysis-path", parsed_args.analysis_path,
                    "--similarity-threshold", str(parsed_args.similarity_threshold),
                    "--confidence-threshold", str(parsed_args.confidence_threshold),
                    "--batch-size", str(parsed_args.batch_size),
                    "--max-workers", str(parsed_args.max_workers),
                    "--model", parsed_args.model
                ]
                
                if parsed_args.sample_size is not None:
                    args_list.extend(["--sample-size", str(parsed_args.sample_size)])
                
                openai_deduplication(args_list, standalone_mode=False)
                return 0
            except SystemExit as e:
                return int(e.code) if e.code else 0
        
        return 1


def main():
    """Entry point for the main CLI."""
    cli = MainCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
