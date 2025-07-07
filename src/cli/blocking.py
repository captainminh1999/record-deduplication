"""
CLI for the blocking step of the record deduplication pipeline.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..core.blocking_engine import BlockingEngine, BlockingConfig
from ..formatters.blocking_formatter import BlockingTerminalFormatter
from ..io.file_handler import FileReader, FileWriter


class BlockingCLI:
    """CLI for the blocking step with clean separation of concerns."""
    
    def __init__(self):
        self.engine = BlockingEngine()
        self.formatter = BlockingTerminalFormatter()
        self.file_reader = FileReader()
        self.file_writer = FileWriter()
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser for blocking."""
        parser = argparse.ArgumentParser(
            description="Generate candidate pairs for deduplication",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python -m src.cli.blocking data/outputs/cleaned.csv
  python -m src.cli.blocking data/outputs/cleaned.csv --output data/pairs.csv
            """
        )
        
        parser.add_argument(
            "input_file",
            help="Path to the cleaned CSV file"
        )
        
        parser.add_argument(
            "--output",
            default="data/outputs/pairs.csv",
            help="Output file path (default: data/outputs/pairs.csv)"
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
        
        return True
    
    def run(self, args: Optional[list] = None) -> int:
        """Run the blocking CLI."""
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
                print("\nGenerating candidate pairs...")
            
            config = BlockingConfig()
            result = self.engine.process(df, config)
            
            # Save results
            if not parsed_args.quiet:
                print(f"\nSaving results to {parsed_args.output}...")
            
            self.file_writer.write_csv_no_index(result.pairs_df, parsed_args.output)
            
            # Display results
            if not parsed_args.quiet:
                self.formatter.format_results(result, parsed_args.input_file, parsed_args.output)
            
            return 0
            
        except Exception as e:
            if not parsed_args.quiet:
                self.formatter.format_error(e)
                import traceback
                traceback.print_exc()
            return 1


def main():
    """Entry point for the blocking CLI."""
    cli = BlockingCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
