"""Step 2 of the 10-step deduplication pipeline: candidate generation.

This stage produces candidate record pairs by blocking on selected
columns to limit the number of comparisons.
"""

from __future__ import annotations

import os
import pandas as pd
import recordlinkage


def main(input_path: str = "data/outputs/cleaned.csv") -> recordlinkage.index.BaseIndex:
    """Build candidate pairs using ``recordlinkage.Index``.

    TODO:
        * read ``input_path`` with :func:`pandas.read_csv`
        * configure blocking on ``phone_clean`` and ``company_clean``
        * return or persist the generated :class:`pandas.MultiIndex`
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cleaned data not found: {input_path}")

    df = pd.read_csv(input_path)
    required_columns = {"phone_clean", "company_clean"}
    missing = required_columns.difference(df.columns)
    if missing:
        cols = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {cols}")

    # TODO: configure blocking using recordlinkage.Index
    return recordlinkage.Index()


if __name__ == "__main__":  # pragma: no cover - sanity run
    print("\u23e9 started blocking")
    main()
