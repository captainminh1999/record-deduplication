"""Step 5 of the 10-step deduplication pipeline: reporting.

This module prepares an Excel workbook that lists high-confidence
duplicates side by side for human validation.
"""

from __future__ import annotations

import pandas as pd


def main(dupes_path: str = "data/outputs/dupes_high_conf.csv", cleaned_path: str = "data/outputs/cleaned.csv") -> None:
    """Create a merge suggestion workbook.

    TODO:
        * load the scored duplicates and original cleaned data
        * merge duplicate pairs back to the original rows
        * write the result to ``merge_suggestions.xlsx`` using ``pandas``
    """
    pass


if __name__ == "__main__":  # pragma: no cover - sanity run
    print("\u23e9 started reporting")
    main()
