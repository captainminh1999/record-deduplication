"""
Backward compatibility module for utility functions.

This module re-exports functionality from the new modular structure:
- File operations: src.io.file_handler
- Logging: src.logging.run_logger  
- Iteration tracking: src.tracking.iteration_tracker

For new code, import directly from the specific modules.
"""

# Re-export from modular structure for backward compatibility
from .io.file_handler import clear_files, clear_all_data
from .logging.run_logger import log_run, LOG_PATH
from .tracking.iteration_tracker import (
    get_current_iteration,
    increment_iteration,
    ITERATION_FILE
)

# Legacy exports for backward compatibility
__all__ = [
    # File operations
    "clear_files",
    "clear_all_data",
    # Logging
    "log_run", 
    "LOG_PATH",
    # Iteration tracking
    "get_current_iteration",
    "increment_iteration",
    "ITERATION_FILE"
]
