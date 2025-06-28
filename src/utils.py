"""
Utility functions for the deduplication pipeline.

Includes file cleanup, logging, and helper functions used across pipeline steps. See README for details.
"""

import os
import glob
from datetime import datetime
from typing import Iterable, Optional
import json

LOG_PATH = "data/run_history.log"
ITERATION_FILE = "data/outputs/iteration_tracker.json"


def get_current_iteration() -> int:
    """Get the current iteration number from the tracker file."""
    if os.path.exists(ITERATION_FILE):
        with open(ITERATION_FILE, "r") as f:
            try:
                data = json.load(f)
                return data.get("current_iteration", 1)
            except json.JSONDecodeError:
                return 1
    return 1


def increment_iteration() -> int:
    """Increment the iteration counter and return the new value."""
    current = get_current_iteration()
    next_iteration = current + 1
    os.makedirs(os.path.dirname(ITERATION_FILE), exist_ok=True)
    with open(ITERATION_FILE, "w") as f:
        json.dump({"current_iteration": next_iteration}, f)
    return next_iteration


def clear_files(paths: Iterable[str]) -> None:
    """Remove files if they exist."""
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def clear_all_data(data_dir: str = "data/outputs", exclude: Iterable[str] | None = None) -> None:
    """Delete all files in ``data_dir`` except those in ``exclude``."""
    exclude = set(exclude or [])
    # Add iteration tracker to excluded files
    exclude.add(os.path.basename(ITERATION_FILE))
    for file_path in glob.glob(os.path.join(data_dir, "*")):
        if os.path.basename(file_path) in exclude:
            continue
        if os.path.isfile(file_path):
            os.remove(file_path)


def log_run(
    step: str,
    start: float,
    end: float,
    rows: int,
    additional_info: Optional[str] = None,
    log_path: str = LOG_PATH,
) -> None:
    """Append a log entry describing a pipeline step."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    duration = end - start
    current_iteration = get_current_iteration()

    # Increment iteration if this is a preprocess step
    if step == "preprocess":
        current_iteration = increment_iteration()

    parts = [
        str(current_iteration),  # Add iteration number
        step,
        datetime.fromtimestamp(start).isoformat(),
        datetime.fromtimestamp(end).isoformat(),
        str(rows),
        f"{duration:.2f}",
    ]

    # Add additional info if provided
    if additional_info:
        parts.append(additional_info)

    line = ",".join(parts)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

