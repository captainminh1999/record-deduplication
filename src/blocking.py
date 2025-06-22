"""Step 2 of the 10-step deduplication pipeline: candidate generation.

This stage produces candidate record pairs by blocking on selected
columns to limit the number of comparisons.
"""

from __future__ import annotations

import os
import time
import pandas as pd
import recordlinkage

from .utils import log_run


def main(
    input_path: str = "data/outputs/cleaned.csv",
    output_path: str = "data/outputs/pairs.csv",
    log_path: str = "data/outputs/run_history.log",
) -> pd.DataFrame:
    """Return a DataFrame of candidate pairs and optionally save to CSV.

    Parameters
    ----------
    input_path:
        Location of the cleaned CSV produced by :mod:`src.preprocess`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``record_id_1`` and ``record_id_2`` columns representing
        candidate record pairs.
    """
    start_time = time.time()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cleaned data not found: {input_path}")

    # Read the cleaned dataset and set record_id as the index for the
    # recordlinkage algorithms.
    df = pd.read_csv(input_path)

    required_columns = {"record_id", "phone_clean", "company_clean", "domain_clean"}
    missing = required_columns.difference(df.columns)
    if missing:
        cols = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {cols}")

    df = df.set_index("record_id")

    # Build exact-match blocks for phone, company and domain.
    blocks = []
    idx = recordlinkage.Index()
    idx.block("phone_clean")
    blocks.append(idx.index(df))

    idx = recordlinkage.Index()
    idx.block("company_clean")
    blocks.append(idx.index(df))

    idx = recordlinkage.Index()
    idx.block("domain_clean")
    blocks.append(idx.index(df))

    # Add a fuzzy block on company name using a sorted-neighbourhood approach.
    # The window size can be tuned later to adjust recall and precision.
    idx = recordlinkage.Index()
    idx.sortedneighbourhood("company_clean", window=5)
    blocks.append(idx.index(df))

    # Union all candidate sets into a single MultiIndex and convert to DataFrame.
    candidates = blocks[0]
    for b in blocks[1:]:
        candidates = candidates.union(b)

    pair_df = candidates.to_frame(index=False)
    pair_df.columns = ["record_id_1", "record_id_2"]

    # Print out the number of pairs generated. Additional blocks and thresholds
    # can be fine-tuned later for better performance.
    print(f"Generated {len(pair_df)} candidate pairs")

    pair_df.to_csv(output_path, index=False)

    end_time = time.time()
    log_run("blocking", start_time, end_time, len(pair_df), log_path=log_path)

    return pair_df


if __name__ == "__main__":  # pragma: no cover - sanity run
    import argparse

    parser = argparse.ArgumentParser(description="Generate candidate pairs")
    parser.add_argument("--input-path", default="data/outputs/cleaned.csv")
    parser.add_argument("--output-path", default="data/outputs/pairs.csv")
    parser.add_argument("--log-path", default="data/outputs/run_history.log")
    args = parser.parse_args()

    print("\u23e9 started blocking")
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        log_path=args.log_path,
    )
