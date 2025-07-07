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

    # Only require company_clean, make others optional
    required = {"company_clean"}
    optional = {"phone_clean", "domain_clean"}
    
    missing_required = required.difference(df.columns)
    if missing_required:
        cols = ", ".join(sorted(missing_required))
        raise KeyError(f"Missing required columns: {cols}")
    
    # Use only available columns for blocking
    available_cols = [col for col in ["phone_clean", "company_clean", "domain_clean"] if col in df.columns]
    
    blocks = []
    for col in available_cols:
        if col == "company_clean":
            # Always do company blocking
            idx = recordlinkage.Index()
            idx.block(col)
            blocks.append(idx.index(df))
            
            # Add fuzzy company blocking
            idx = recordlinkage.Index()
            idx.sortedneighbourhood(col, window=5)
            blocks.append(idx.index(df))
        elif col in df.columns and not df[col].isna().all():
            # Only block on non-empty optional columns
            idx = recordlinkage.Index()
            idx.block(col)
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
    print("🔗 Starting candidate pair generation...")
    
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

    # Only require record_id and company_clean for minimal datasets
    required_columns = {"record_id", "company_clean"}
    missing = required_columns.difference(df.columns)
    if missing:
        cols = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {cols}")

    # Get block sizes with error handling - only for available columns
    def get_block_size(indexer: recordlinkage.Index, df: pd.DataFrame) -> int:
        try:
            pairs = indexer.index(df)
            return len(pairs) if pairs is not None else 0
        except Exception:
            return 0

    # Phone block (only if column exists and has data)
    if "phone_clean" in df.columns and not df["phone_clean"].isna().all():
        idx = recordlinkage.Index()
        idx.block("phone_clean")
        block_counts["phone_block"] = get_block_size(idx, df)

    # Company block (always present for minimal datasets)
    idx = recordlinkage.Index()
    idx.block("company_clean")
    block_counts["company_block"] = get_block_size(idx, df)

    # Domain block (only if column exists and has data)
    if "domain_clean" in df.columns and not df["domain_clean"].isna().all():
        idx = recordlinkage.Index()
        idx.block("domain_clean")
        block_counts["domain_block"] = get_block_size(idx, df)

    # Fuzzy company block (always available)
    idx = recordlinkage.Index()
    idx.sortedneighbourhood("company_clean", window=5)
    block_counts["fuzzy_block"] = get_block_size(idx, df)

    # Generate final candidate pairs
    candidates = generate_candidate_pairs(df)
    pair_df = candidates.to_frame(index=False)
    pair_df.columns = ["record_id_1", "record_id_2"]

    # Print comprehensive terminal output
    print(f"\n🔗 Candidate Pair Generation Complete!")
    print(f"─" * 50)
    print(f"📊 Data Overview:")
    print(f"  • Input records:         {initial_rows:,}")
    print(f"  • Generated pairs:       {len(pair_df):,}")
    print(f"  • Total possible pairs:  {int(initial_rows * (initial_rows - 1) / 2):,}")
    
    reduction_ratio = 1 - (len(pair_df) / (initial_rows * (initial_rows - 1) / 2)) if initial_rows > 1 else 0
    print(f"  • Reduction ratio:       {reduction_ratio:.1%}")
    
    # Show which blocking methods were used
    used_methods = []
    available_cols = []
    if "phone_clean" in df.columns and not df["phone_clean"].isna().all():
        available_cols.append("phone")
        if block_counts["phone_block"] > 0:
            used_methods.append(f"phone ({block_counts['phone_block']:,})")
    if "company_clean" in df.columns:
        available_cols.append("company")
        if block_counts["company_block"] > 0:
            used_methods.append(f"company ({block_counts['company_block']:,})")
        if block_counts["fuzzy_block"] > 0:
            used_methods.append(f"fuzzy company ({block_counts['fuzzy_block']:,})")
    if "domain_clean" in df.columns and not df["domain_clean"].isna().all():
        available_cols.append("domain")
        if block_counts["domain_block"] > 0:
            used_methods.append(f"domain ({block_counts['domain_block']:,})")
    
    print(f"\n🔧 Blocking Strategy:")
    print(f"  • Available fields:      {', '.join(available_cols)}")
    print(f"  • Methods used:          {', '.join(used_methods) if used_methods else 'fuzzy company only'}")
    
    if len(pair_df) == 0:
        print(f"\n⚠️  Warning: No candidate pairs generated!")
        print(f"   • This might indicate no records share blocking keys")
        print(f"   • Consider more lenient blocking or data quality review")
    elif len(pair_df) > 100000:
        print(f"\n⚠️  Warning: Large number of pairs generated ({len(pair_df):,})")
        print(f"   • This might slow down similarity computation")
        print(f"   • Consider more aggressive blocking")
    else:
        print(f"\n✅ Good pair count for similarity analysis")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pair_df.to_csv(output_path, index=False)
    
    print(f"\n💾 Files Created:")
    print(f"  • Candidate pairs:       {output_path}")
    
    print(f"\n✅ Next step: Compute similarity features")
    print(f"   Command: python -m src.similarity")

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
