"""Step 2 of 6: Blocking (Candidate Generation)

Generates candidate record pairs for comparison by grouping records on blocking keys (e.g., normalized phone and company name) to limit the number of comparisons. See README for details.
"""

from __future__ import annotations

import os
import time
import pandas as pd
import recordlinkage
import json

from .utils import log_run, LOG_PATH


def generate_candidate_pairs(df: pd.DataFrame) -> pd.MultiIndex:
    """Return a MultiIndex of candidate pairs using simple blocking rules.

    Parameters
    ----------
    df:
        Cleaned DataFrame with ``record_id`` as a column or index and
        normalised ``phone_clean``, ``company_clean`` and ``domain_clean``
        fields.

    Returns
    -------
    pandas.MultiIndex
        Candidate pairs to be compared in the next pipeline step.
    """
    if "record_id" in df.columns:
        df = df.set_index("record_id")

    required = {"phone_clean", "company_clean", "domain_clean"}
    missing = required.difference(df.columns)
    if missing:
        cols = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {cols}")

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

    # A fuzzy company block using sorted neighbourhood. The window size
    # can be tuned later for a different recall/precision trade-off.
    idx = recordlinkage.Index()
    idx.sortedneighbourhood("company_clean", window=5)
    blocks.append(idx.index(df))

    candidates = blocks[0]
    for b in blocks[1:]:
        candidates = candidates.union(b)

    return candidates


def main(
    input_path: str = "data/outputs/cleaned.csv",
    output_path: str = "data/outputs/pairs.csv",
    log_path: str = LOG_PATH,
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
    initial_rows = 0
    block_counts = {
        "phone_block": 0,
        "company_block": 0,
        "domain_block": 0,
        "fuzzy_block": 0,
    }

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cleaned data not found: {input_path}")

    # Read the cleaned dataset
    df = pd.read_csv(input_path)
    initial_rows = len(df)

    required_columns = {"record_id", "phone_clean", "company_clean", "domain_clean"}
    missing = required_columns.difference(df.columns)
    if missing:
        cols = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {cols}")

    # Get block sizes with error handling
    def get_block_size(indexer: recordlinkage.Index, df: pd.DataFrame) -> int:
        try:
            pairs = indexer.index(df)
            return len(pairs) if pairs is not None else 0
        except Exception:
            return 0

    # Phone block
    idx = recordlinkage.Index()
    idx.block("phone_clean")
    block_counts["phone_block"] = get_block_size(idx, df)

    # Company block
    idx = recordlinkage.Index()
    idx.block("company_clean")
    block_counts["company_block"] = get_block_size(idx, df)

    # Domain block
    idx = recordlinkage.Index()
    idx.block("domain_clean")
    block_counts["domain_block"] = get_block_size(idx, df)

    # Fuzzy company block
    idx = recordlinkage.Index()
    idx.sortedneighbourhood("company_clean", window=5)
    block_counts["fuzzy_block"] = get_block_size(idx, df)

    # Generate final candidate pairs
    candidates = generate_candidate_pairs(df)
    pair_df = candidates.to_frame(index=False)
    pair_df.columns = ["record_id_1", "record_id_2"]

    # Print out stats
    print(f"Generated {len(pair_df)} candidate pairs from {initial_rows} records")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pair_df.to_csv(output_path, index=False)

    end_time = time.time()
    total_possible_pairs = initial_rows * (initial_rows - 1) / 2
    stats = {
        "input_rows": initial_rows,
        "output_pairs": len(pair_df),
        "reduction_ratio": 1 - (len(pair_df) / total_possible_pairs if total_possible_pairs > 0 else 0),
        "block_sizes": block_counts
    }
    log_run("blocking", start_time, end_time, len(pair_df), additional_info=json.dumps(stats), log_path=log_path)

    return pair_df


if __name__ == "__main__":  # pragma: no cover - sanity run
    import argparse

    parser = argparse.ArgumentParser(description="Generate candidate pairs")
    parser.add_argument("--input-path", default="data/outputs/cleaned.csv")
    parser.add_argument("--output-path", default="data/outputs/pairs.csv")
    parser.add_argument("--log-path", default=LOG_PATH)
    args = parser.parse_args()

    print("\u23e9 started blocking")
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        log_path=args.log_path,
    )
