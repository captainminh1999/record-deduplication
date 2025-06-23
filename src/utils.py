import os
import glob
from datetime import datetime
from typing import Iterable

LOG_PATH = "data/run_history.log"


def clear_files(paths: Iterable[str]) -> None:
    """Remove files if they exist."""
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def clear_all_data(data_dir: str = "data/outputs", exclude: Iterable[str] | None = None) -> None:
    """Delete all files in ``data_dir`` except those in ``exclude``."""
    exclude = set(exclude or [])
    for file_path in glob.glob(os.path.join(data_dir, "*")):
        if os.path.basename(file_path) in exclude:
            continue
        if os.path.isfile(file_path):
            os.remove(file_path)


def log_run(step: str, start: float, end: float, rows: int, log_path: str = LOG_PATH) -> None:
    """Append a log entry describing a pipeline step."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    duration = end - start
    line = ",".join(
        [
            step,
            datetime.fromtimestamp(start).isoformat(),
            datetime.fromtimestamp(end).isoformat(),
            str(rows),
            f"{duration:.2f}",
        ]
    )
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

