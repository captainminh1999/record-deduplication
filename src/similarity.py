"""Feature generation for record comparison."""

import pandas as pd
import recordlinkage
from rapidfuzz import fuzz


def main(df_path: str = "data/cleaned.csv") -> pd.DataFrame:
    """Compute similarity features for candidate record pairs."""
    df = pd.read_csv(df_path, index_col="ID")

    # Blocking to get candidate pairs
    indexer = recordlinkage.Index()
    indexer.block("phone_clean")
    indexer.sortedneighbourhood("name_clean", window=5)
    candidate_pairs = indexer.index(df)

    compare = recordlinkage.Compare()
    compare.string("name_clean", "name_clean", method="jarowinkler", label="name_sim")
    compare.exact("phone_clean", "phone_clean", label="phone_match")

    features = compare.compute(candidate_pairs, df)

    features["addr_sim"] = features.apply(
        lambda row: fuzz.token_set_ratio(
            df.loc[row.name[0], "Address"],
            df.loc[row.name[1], "Address"],
        )
        / 100,
        axis=1,
    )

    return features


if __name__ == "__main__":
    feats = main()
    feats.to_csv("data/features.csv")
    print(f"Wrote {len(feats)} feature rows")
