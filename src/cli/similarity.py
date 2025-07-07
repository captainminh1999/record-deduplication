"""
CLI for the similarity step of the record deduplication pipeline.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..core.similarity_engine import SimilarityEngine, SimilarityConfig
from ..formatters.similarity_formatter import SimilarityTerminalFormatter
from ..io.file_handler import FileReader, FileWriter


class SimilarityCLI:
    """CLI for the similarity step with clean separation of concerns."""
    
    def __init__(self):
        self.engine = SimilarityEngine()
        self.formatter = SimilarityTerminalFormatter()
        self.file_reader = FileReader()
        self.file_writer = FileWriter()
    
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
        
        parser.add_argument(
            "--output",
            default="data/outputs/features.csv",
            help="Output file path (default: data/outputs/features.csv)"
        )
        
        parser.add_argument(
            "--quiet",
            action="store_true",
            help="Suppress progress output"
        )
        
        return parser
    
    def validate_input(self, cleaned_file: str, pairs_file: str) -> bool:
        """Validate that the input files exist and are readable."""
        for file_path, name in [(cleaned_file, "cleaned"), (pairs_file, "pairs")]:
            path = Path(file_path)
            if not path.exists():
                print(f"Error: {name.capitalize()} file '{file_path}' does not exist.")
                return False
            if not path.is_file():
                print(f"Error: '{file_path}' is not a file.")
                return False
        return True
    
    def run(self, args: Optional[list] = None) -> int:
        """Run the similarity CLI."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        # Validate input
        if not self.validate_input(parsed_args.cleaned_file, parsed_args.pairs_file):
            return 1
        
        try:
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
            
            # Save results
            if not parsed_args.quiet:
                print(f"\nSaving results to {parsed_args.output}...")
            
            self.file_writer.write_csv_no_index(result.features_df, parsed_args.output)
            
            # Display results
            if not parsed_args.quiet:
                self.formatter.format_results(
                    result, 
                    parsed_args.cleaned_file, 
                    parsed_args.pairs_file, 
                    parsed_args.output
                )
            
            return 0
            
        except Exception as e:
            if not parsed_args.quiet:
                self.formatter.format_error(e)
                import traceback
                traceback.print_exc()
            return 1


def main():
    """Entry point for the similarity CLI."""
    cli = SimilarityCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
