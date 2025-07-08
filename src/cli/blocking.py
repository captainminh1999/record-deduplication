"""
CLI for the blocking step of the record deduplication pipeline.
"""

import argparse
import sys
from typing import Optional

from .base import StandardCLI, CLIArgumentPatterns
from ..core.blocking_engine import BlockingEngine, BlockingConfig
from ..formatters.blocking_formatter import BlockingTerminalFormatter


class BlockingCLI(StandardCLI):
    """CLI for the blocking step with clean separation of concerns."""
    
    def __init__(self):
        super().__init__(BlockingEngine, BlockingTerminalFormatter)
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser for blocking."""
        parser = self.create_base_parser(
            description="Generate candidate pairs for deduplication",
            examples="""
Examples:
  python -m src.cli.blocking data/outputs/cleaned.csv
  python -m src.cli.blocking data/outputs/cleaned.csv --output data/pairs.csv
            """
        )
        
        CLIArgumentPatterns.add_output_argument(parser, "data/outputs/pairs.csv")
        
        return parser
    
    def process_data(self, parsed_args: argparse.Namespace) -> bool:
        """Process the data according to the parsed arguments."""
        try:
            # Load data
            df = self.load_data(parsed_args.input_file, parsed_args.quiet)
            
            # Process data
            if not parsed_args.quiet:
                print("\nGenerating candidate pairs...")
            
            config = BlockingConfig()
            result = self.engine.process(df, config)
            
            # Save results (blocking doesn't use index)
            self.save_results(result.pairs_df, parsed_args.output, parsed_args.quiet, use_index=False)
            
            # Display results
            if not parsed_args.quiet:
                self.formatter.format_results(result, parsed_args.input_file, parsed_args.output)
            
            return True
            
        except Exception as e:
            self.handle_error(e, parsed_args.quiet)
            return False
            
            return True
            
        except Exception as e:
            self.handle_error(e, parsed_args.quiet)
            return False


def main():
    """Entry point for the blocking CLI."""
    cli = BlockingCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
