"""
CLI for the preprocessing step of the record deduplication pipeline.

This module demonstrates the new modular architecture:
- Business logic is handled by PreprocessEngine
- Terminal output is formatted by PreprocessFormatter  
- File I/O is managed by FileHandler
- CLI orchestrates the flow and handles user interaction
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..core.preprocess_engine import PreprocessEngine
from ..formatters.preprocess_formatter import PreprocessTerminalFormatter as PreprocessFormatter
from ..io.file_handler import FileReader, FileWriter


class PreprocessCLI:
    """CLI for the preprocessing step with clean separation of concerns."""
    
    def __init__(self):
        self.engine = PreprocessEngine()
        self.formatter = PreprocessFormatter()
        self.file_reader = FileReader()
        self.file_writer = FileWriter()
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser for preprocessing."""
        parser = argparse.ArgumentParser(
            description="Preprocess records for deduplication",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python -m src.cli.preprocess data/sample_input.csv
  python -m src.cli.preprocess data/sample_input.csv --output data/cleaned.csv
  python -m src.cli.preprocess data/sample_input.csv --normalize --deduplicate
            """
        )
        
        parser.add_argument(
            "input_file",
            help="Path to the input CSV file"
        )
        
        parser.add_argument(
            "--output",
            default="data/outputs/cleaned.csv",
            help="Output file path (default: data/outputs/cleaned.csv)"
        )
        
        parser.add_argument(
            "--normalize",
            action="store_true",
            help="Apply normalization to company names"
        )
        
        parser.add_argument(
            "--deduplicate", 
            action="store_true",
            help="Remove exact duplicates"
        )
        
        parser.add_argument(
            "--quiet",
            action="store_true",
            help="Suppress progress output"
        )
        
        return parser
    
    def validate_input(self, input_file: str) -> bool:
        """Validate that the input file exists and is readable."""
        input_path = Path(input_file)
        
        if not input_path.exists():
            print(f"Error: Input file '{input_file}' does not exist.")
            return False
        
        if not input_path.is_file():
            print(f"Error: '{input_file}' is not a file.")
            return False
        
        if input_path.suffix.lower() != '.csv':
            print(f"Warning: Input file '{input_file}' is not a CSV file.")
            response = input("Continue anyway? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                return False
        
        return True
    
    def run(self, args: Optional[list] = None) -> int:
        """Run the preprocessing CLI."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        # Validate input
        if not self.validate_input(parsed_args.input_file):
            return 1
        
        try:
            # Load data
            if not parsed_args.quiet:
                self.formatter.format_start_message(parsed_args.input_file)
            
            df = self.file_reader.read_data(parsed_args.input_file)
            
            # Process data
            if not parsed_args.quiet:
                print("\nProcessing data...")
            
            from ..core.preprocess_engine import PreprocessConfig
            config = PreprocessConfig(
                use_openai=False,  # Could be added as CLI option
                remove_duplicates=parsed_args.deduplicate
            )
            
            result = self.engine.process(df, config)
            
            # Save results
            if not parsed_args.quiet:
                print(f"\nSaving results to {parsed_args.output}...")
            
            self.file_writer.write_csv(result.cleaned_df, parsed_args.output)
            
            # Display results
            if not parsed_args.quiet:
                audit_path = parsed_args.output.replace('.csv', '_removed.csv')
                if len(result.duplicates_df) > 0:
                    self.file_writer.write_csv(result.duplicates_df, audit_path)
                
                self.formatter.format_results(result, parsed_args.input_file, parsed_args.output, audit_path)
            
            return 0
            
        except FileNotFoundError as e:
            if not parsed_args.quiet:
                self.formatter.format_error(e)
            return 1
        except Exception as e:
            if not parsed_args.quiet:
                self.formatter.format_error(e)
                import traceback
                traceback.print_exc()
            return 1


def main():
    """Entry point for the preprocessing CLI."""
    cli = PreprocessCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
