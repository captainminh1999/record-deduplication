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
import time
from pathlib import Path
from typing import Optional

from .base import StandardCLI, CLIArgumentPatterns
from ..core.preprocess_engine import PreprocessEngine, PreprocessConfig
from ..formatters.preprocess_formatter import PreprocessTerminalFormatter as PreprocessFormatter
from ..logging import log_run


class PreprocessCLI(StandardCLI):
    """CLI for the preprocessing step with clean separation of concerns."""
    
    def __init__(self):
        super().__init__(PreprocessEngine, PreprocessFormatter)
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser for preprocessing."""
        parser = self.create_base_parser(
            description="Preprocess records for deduplication",
            examples="""
Examples:
  python -m src.cli.preprocess data/sample_input.csv
  python -m src.cli.preprocess data/sample_input.csv --output data/cleaned.csv
  python -m src.cli.preprocess data/sample_input.csv --normalize --deduplicate
            """
        )
        
        CLIArgumentPatterns.add_output_argument(parser, "data/outputs/cleaned.csv")
        CLIArgumentPatterns.add_boolean_flag(parser, "normalize", "Apply normalization to company names")
        CLIArgumentPatterns.add_boolean_flag(parser, "deduplicate", "Remove exact duplicates")
        
        return parser
    
    def process_data(self, parsed_args: argparse.Namespace) -> bool:
        """Process the data according to the parsed arguments."""
        start_time = time.time()
        
        try:
            # Load data
            df = self.load_data(parsed_args.input_file, parsed_args.quiet)
            
            # Process data
            if not parsed_args.quiet:
                print("\nProcessing data...")
            
            config = PreprocessConfig(
                use_openai=False,  # Could be added as CLI option
                remove_duplicates=parsed_args.deduplicate
            )
            
            result = self.engine.process(df, config)
            
            # Save results
            self.save_results(result.cleaned_df, parsed_args.output, parsed_args.quiet)
            
            # Display results
            if not parsed_args.quiet:
                audit_path = parsed_args.output.replace('.csv', '_removed.csv')
                if len(result.duplicates_df) > 0:
                    self.file_writer.write_csv(result.duplicates_df, audit_path)
                
                self.formatter.format_results(result, parsed_args.input_file, parsed_args.output, audit_path)
            
            # Log the run
            end_time = time.time()
            log_run(
                step="preprocess",
                start=start_time,
                end=end_time,
                rows=len(result.cleaned_df),
                additional_info=str(result.stats.__dict__).replace("'", '"')
            )
            
            return True
            
        except Exception as e:
            self.handle_error(e, parsed_args.quiet)
            return False


def main():
    """Entry point for the preprocessing CLI."""
    cli = PreprocessCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
