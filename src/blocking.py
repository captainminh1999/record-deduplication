"""Step 2 of the 10-step deduplication pipeline: candidate generation.

This stage produces candidate record pairs by blocking on selected
columns to limit the number of comparisons.
"""

from __future__ import annotations

import pandas as pd
import recordlinkage


def main(input_path: str = "data/cleaned.csv") -> None:
    """Build candidate pairs using ``recordlinkage.Index``.

    TODO:
        * read ``input_path`` with :func:`pandas.read_csv`
        * configure blocking on ``phone_clean`` and ``name_clean``
        * return or persist the generated :class:`pandas.MultiIndex`
    """
    pass


if __name__ == "__main__":  # pragma: no cover - sanity run
    print("\u23e9 started blocking")
    main()
