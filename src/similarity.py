"""Step 3 of the 10-step deduplication pipeline: feature building.

Given candidate pairs, this module computes textual and exact-match
similarity metrics that serve as model features.
"""

from __future__ import annotations

import pandas as pd
import recordlinkage
from rapidfuzz import fuzz


def main(df_path: str = "data/cleaned.csv") -> None:
    """Create similarity features between candidate pairs.

    TODO:
        * load the cleaned dataset
        * generate candidate pairs (reuse :mod:`src.blocking` logic)
        * compute string similarities with `recordlinkage.Compare`
        * add address similarity using :func:`rapidfuzz.fuzz.token_set_ratio`
        * persist the resulting feature table
    """
    pass


if __name__ == "__main__":  # pragma: no cover - sanity run
    print("\u23e9 started similarity")
    main()
