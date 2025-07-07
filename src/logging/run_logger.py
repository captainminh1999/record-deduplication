"""
Pipeline run logging functionality.
"""

import os
from datetime import datetime
from typing import Optional

from ..tracking.iteration_tracker import get_current_iteration, increment_iteration

LOG_PATH = "data/run_history.log"


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
