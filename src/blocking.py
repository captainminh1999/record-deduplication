"""Utilities for blocking records to limit pairwise comparisons."""

import pandas as pd
import recordlinkage


def main(input_path: str = "data/cleaned.csv"):
    """Generate candidate record pairs using blocking strategies."""
    df = pd.read_csv(input_path, index_col="ID")

    indexer = recordlinkage.Index()
    indexer.block("phone_clean")
    indexer.sortedneighbourhood("name_clean", window=5)

    candidate_pairs = indexer.index(df)
    return candidate_pairs


if __name__ == "__main__":
    pairs = main()
    print(f"Generated {len(pairs)} candidate pairs")
