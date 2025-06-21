"""Step 1 of the 10-step deduplication pipeline: data cleaning.

This module prepares the raw spreadsheet for later stages by normalising
fields and writing the cleaned output to ``data/cleaned.csv``.
"""

from __future__ import annotations

import os
import pandas as pd


def main(input_path: str = "data/your_spreadsheet.csv", output_path: str = "data/cleaned.csv") -> None:
    """Clean the raw spreadsheet and save a CSV.

    TODO:
        * load the source spreadsheet with :func:`pandas.read_csv` or
          ``read_excel``
        * normalise name and phone columns
        * remove exact duplicate rows
        * write the result to ``output_path``
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input spreadsheet not found: {input_path}")

    # TODO: implement data cleaning logic
    pass


if __name__ == "__main__":  # pragma: no cover - sanity run
    print("\u23e9 started preprocessing")
    main()
