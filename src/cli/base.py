"""
Base classes and patterns for CLI modules.

This module provides common functionality and patterns used across all CLI modules:
- Input validation
- Error handling
- File I/O setup
- Common argument patterns
- Exit code standards
"""

import argparse
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any, Dict, Type, TypeVar, Union
import traceback

from ..io.file_handler import FileReader, FileWriter


# Type variables for generic base classes
EngineType = TypeVar('EngineType')
FormatterType = TypeVar('FormatterType')
ConfigType = TypeVar('ConfigType')


class BaseCLI(ABC):
    """
    Abstract base class for all CLI modules.
    
    Provides common functionality like:
    - File I/O setup
    - Input validation
    - Error handling patterns
    - Exit code management
    """
    
    def __init__(self):
        self.file_reader = FileReader()
        self.file_writer = FileWriter()
        self._setup()
    
    @abstractmethod
    def _setup(self) -> None:
        """Initialize engine, formatter, and other components."""
        pass
    
    @abstractmethod
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser for this CLI module."""
        pass
    
    @abstractmethod
    def process_data(self, parsed_args: argparse.Namespace) -> Any:
        """Process the data according to the parsed arguments."""
        pass
    
    def validate_input_file(self, input_file: str, required_extension: Optional[str] = '.csv') -> bool:
        """
        Validate that the input file exists and is readable.
        
        Args:
            input_file: Path to the input file
            required_extension: Required file extension (None to skip check)
            
        Returns:
            True if file is valid, False otherwise
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            print(f"Error: Input file '{input_file}' does not exist.")
            return False
        
        if not input_path.is_file():
            print(f"Error: '{input_file}' is not a file.")
            return False
        
        if required_extension and input_path.suffix.lower() != required_extension:
            print(f"Warning: Input file '{input_file}' does not have {required_extension} extension.")
            response = input("Continue anyway? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                return False
        
        return True
    
    def validate_output_path(self, output_path: str) -> bool:
        """
        Validate that the output path is writable.
        
        Args:
            output_path: Path where output will be written
            
        Returns:
            True if path is writable, False otherwise
        """
        output_path_obj = Path(output_path)
        
        # Create parent directories if they don't exist
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if parent directory is writable
        if not output_path_obj.parent.is_dir():
            print(f"Error: Cannot create output directory '{output_path_obj.parent}'.")
            return False
        
        return True
    
    def handle_error(self, error: Exception, quiet: bool = False) -> None:
        """
        Handle and format errors consistently across CLI modules.
        
        Args:
            error: The exception that occurred
            quiet: Whether to suppress detailed error output
        """
        if not quiet:
            if hasattr(self, 'formatter') and hasattr(self.formatter, 'format_error'):
                self.formatter.format_error(error)
            else:
                print(f"Error: {error}")
            
            # Print traceback for debugging
            if not isinstance(error, (FileNotFoundError, ValueError)):
                traceback.print_exc()
    
    def run(self, args: Optional[list] = None) -> int:
        """
        Run the CLI with error handling and consistent return codes.
        
        Args:
            args: Command line arguments (None to use sys.argv)
            
        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            parser = self.create_parser()
            parsed_args = parser.parse_args(args)
            
            # Validate inputs if input_file exists
            if hasattr(parsed_args, 'input_file'):
                if not self.validate_input_file(parsed_args.input_file):
                    return 1
            
            # Validate output if output exists  
            if hasattr(parsed_args, 'output'):
                if not self.validate_output_path(parsed_args.output):
                    return 1
            
            # Process the data
            result = self.process_data(parsed_args)
            
            return 0 if result is not False else 1
            
        except KeyboardInterrupt:
            if not getattr(parsed_args, 'quiet', False):
                print("\nOperation cancelled by user.")
            return 1
        except Exception as e:
            quiet = getattr(parsed_args, 'quiet', False) if 'parsed_args' in locals() else False
            self.handle_error(e, quiet)
            return 1


class StandardCLI(BaseCLI):
    """
    Standard CLI pattern for most pipeline steps.
    
    This class handles the common pattern of:
    1. Load data from input file
    2. Process with engine and config
    3. Save results to output file
    4. Display formatted results
    """
    
    def __init__(self, engine_class: Type[EngineType], formatter_class: Type[FormatterType]):
        self.engine_class = engine_class
        self.formatter_class = formatter_class
        self.engine: Any = None
        self.formatter: Any = None
        super().__init__()
    
    def _setup(self) -> None:
        """Initialize engine and formatter."""
        self.engine = self.engine_class()
        self.formatter = self.formatter_class()
    
    def create_base_parser(self, description: str, examples: str) -> argparse.ArgumentParser:
        """
        Create a base parser with common arguments.
        
        Args:
            description: Description for the command
            examples: Examples section for help text
            
        Returns:
            Configured argument parser
        """
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=examples
        )
        
        parser.add_argument(
            "input_file",
            help="Path to the input CSV file"
        )
        
        parser.add_argument(
            "--quiet",
            action="store_true",
            help="Suppress progress output"
        )
        
        return parser
    
    def load_data(self, input_file: str, quiet: bool = False) -> Any:
        """
        Load data from input file with progress messaging.
        
        Args:
            input_file: Path to input file
            quiet: Whether to suppress progress messages
            
        Returns:
            Loaded DataFrame
        """
        if not quiet and hasattr(self.formatter, 'format_start_message'):
            try:
                self.formatter.format_start_message(input_file)
            except TypeError:
                # Some formatters may have different method signatures
                pass
        
        return self.file_reader.read_data(input_file)
    
    def save_results(self, data: Any, output_path: str, quiet: bool = False, use_index: bool = True) -> None:
        """
        Save results to output file with progress messaging.
        
        Args:
            data: Data to save
            output_path: Where to save the data
            quiet: Whether to suppress progress messages
            use_index: Whether to include index in CSV output
        """
        if not quiet:
            print(f"\nSaving results to {output_path}...")
        
        if use_index:
            self.file_writer.write_csv(data, output_path)
        else:
            self.file_writer.write_csv_no_index(data, output_path)


class CLIArgumentPatterns:
    """Common argument patterns used across CLI modules."""
    
    @staticmethod
    def add_output_argument(parser: argparse.ArgumentParser, default_path: str) -> None:
        """Add standard output argument."""
        parser.add_argument(
            "--output",
            default=default_path,
            help=f"Output file path (default: {default_path})"
        )
    
    @staticmethod
    def add_threshold_argument(parser: argparse.ArgumentParser, name: str, default: float, help_text: str) -> None:
        """Add a threshold argument."""
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            default=default,
            type=float,
            help=help_text
        )
    
    @staticmethod
    def add_boolean_flag(parser: argparse.ArgumentParser, name: str, help_text: str) -> None:
        """Add a boolean flag argument."""
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            action="store_true",
            help=help_text
        )


class CLIExitCodes:
    """Standard exit codes for CLI modules."""
    
    SUCCESS = 0
    ERROR = 1
    INVALID_INPUT = 2
    PERMISSION_ERROR = 3
    INTERRUPTED = 130
