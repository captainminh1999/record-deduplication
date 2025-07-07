"""
Iteration tracking for pipeline runs.
"""

import os
import json

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
