"""
CLI for the similarity step of the record deduplication pipeline.
"""

import argparse
import sys
import time
from typing import Optional

from .base import StandardCLI, CLIArgumentPatterns
from ..core.similarity_engine import SimilarityEngine, SimilarityConfig
from ..formatters.similarity_formatter import SimilarityTerminalFormatter
from ..utils import log_run


class SimilarityCLI(StandardCLI):
    """CLI for the similarity step with clean separation of concerns."""
    
    def __init__(self):
        super().__init__(SimilarityEngine, SimilarityTerminalFormatter)
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser for similarity."""
        parser = argparse.ArgumentParser(
            description="Compute similarity features for candidate pairs",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python -m src.cli.similarity data/outputs/cleaned.csv data/outputs/pairs.csv
  python -m src.cli.similarity data/outputs/cleaned.csv data/outputs/pairs.csv --output data/features.csv
            """
        )
        
        parser.add_argument(
            "cleaned_file",
            help="Path to the cleaned CSV file"
        )
        
        parser.add_argument(
            "pairs_file",
            help="Path to the candidate pairs CSV file"
        )
        
        CLIArgumentPatterns.add_output_argument(parser, "data/outputs/features.csv")
        CLIArgumentPatterns.add_boolean_flag(parser, "quiet", "Suppress progress output")
        
        return parser
    
    def validate_input_file(self, input_file: str, required_extension: str = '.csv') -> bool:
        """Override to handle multiple input files for similarity."""
        # This will be called by the base class for 'input_file' which doesn't exist in similarity
        # So we'll handle validation in process_data instead
        return True
    
    def validate_similarity_inputs(self, cleaned_file: str, pairs_file: str) -> bool:
        """Validate that the input files exist and are readable."""
        from pathlib import Path
        
        for file_path, name in [(cleaned_file, "cleaned"), (pairs_file, "pairs")]:
            path = Path(file_path)
            if not path.exists():
                print(f"Error: {name.capitalize()} file '{file_path}' does not exist.")
                return False
            if not path.is_file():
                print(f"Error: '{file_path}' is not a file.")
                return False
        return True
    
    def process_data(self, parsed_args: argparse.Namespace) -> bool:
        """Process the data according to the parsed arguments."""
        start_time = time.time()
        
        try:
            # Validate similarity-specific inputs
            if not self.validate_similarity_inputs(parsed_args.cleaned_file, parsed_args.pairs_file):
                return False
            
            # Load data
            if not parsed_args.quiet:
                self.formatter.format_start_message(parsed_args.cleaned_file, parsed_args.pairs_file)
            
            cleaned_df = self.file_reader.read_data(parsed_args.cleaned_file)
            pairs_df = self.file_reader.read_data(parsed_args.pairs_file)
            
            # Process data
            if not parsed_args.quiet:
                print("\nComputing similarity features...")
            
            config = SimilarityConfig()
            result = self.engine.process(cleaned_df, pairs_df, config)
            
            # Save results (similarity doesn't use index)
            self.save_results(result.features_df, parsed_args.output, parsed_args.quiet, use_index=False)
            
            # Display results
            if not parsed_args.quiet:
                self.formatter.format_results(
                    result, 
                    parsed_args.cleaned_file, 
                    parsed_args.pairs_file, 
                    parsed_args.output
                )
            
            # Log the run
            end_time = time.time()
            log_run(
                step="similarity",
                start=start_time,
                end=end_time,
                rows=len(result.features_df),
                additional_info=str(result.stats.__dict__).replace("'", '"')
            )
            
            return True
            
        except Exception as e:
            self.handle_error(e, parsed_args.quiet)
            return False


def main():
    """Entry point for the similarity CLI."""
    cli = SimilarityCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
