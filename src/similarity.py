"""Step 3 of 6: Similarity Features

Computes similarity features (e.g., string similarity, exact matches) for each candidate pair, used as input for the model. See README for details.
"""

from __future__ import annotations

import os
import re
import time
import json

import click
import pandas as pd
import recordlinkage
from rapidfuzz import fuzz

from .blocking import generate_candidate_pairs
from .utils import log_run, LOG_PATH


def main(
    cleaned_path: str = "data/outputs/cleaned.csv",
    pairs_path: str = "data/outputs/pairs.csv",
    features_path: str = "data/outputs/features.csv",
    log_path: str = LOG_PATH,
) -> pd.DataFrame:
    """Create similarity features between candidate record pairs."""
    print("📏 Starting similarity feature computation...")
    
    start_time = time.time()
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned data not found: {cleaned_path}")

    df = pd.read_csv(cleaned_path)
    initial_rows = len(df)

    required = {
        "record_id",
        "company_clean",
    }
    optional_cols = ["phone_clean", "domain_clean", "address_clean"]
    address_present = "address_clean" in df.columns
    
    missing = required.difference(df.columns)
    if missing:
        cols = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {cols}")

    df = df.set_index("record_id")

    # Track input pair stats
    if os.path.exists(pairs_path):
        pairs_df = pd.read_csv(pairs_path)
        pairs_index = pd.MultiIndex.from_frame(pairs_df)
        input_pairs = len(pairs_df)
    else:
        pairs_index = generate_candidate_pairs(df)
        input_pairs = len(pairs_index)
        os.makedirs(os.path.dirname(pairs_path), exist_ok=True)
        pairs_index.to_frame(index=False).to_csv(pairs_path, index=False)

    # Compute similarities for available columns
    comp = recordlinkage.Compare()
    
    # Always compute company similarity
    comp.string(
        "company_clean",
        "company_clean",
        method="jarowinkler",
        label="company_sim",
    )
    
    # Only compute domain similarity if column exists and has data
    if "domain_clean" in df.columns and not df["domain_clean"].isna().all():
        comp.string(
            "domain_clean",
            "domain_clean",
            method="jarowinkler",
            label="domain_sim",
        )
    
    # Only compute phone exact match if column exists and has data  
    if "phone_clean" in df.columns and not df["phone_clean"].isna().all():
        comp.exact("phone_clean", "phone_clean", label="phone_exact")

    features = comp.compute(pairs_index, df)

    # Track similarity stats for available features
    similarity_stats = {
        "mean": {},
        "high_similarity": {}
    }
    
    for col in features.columns:
        if col.endswith('_sim') or col.endswith('_exact'):
            similarity_stats["mean"][col] = float(features[col].mean())
            if col.endswith('_sim'):
                similarity_stats["high_similarity"][f"{col}_gt_0.8"] = int((features[col] > 0.8).sum())
            else:  # exact match
                similarity_stats["high_similarity"][f"{col}_matches"] = int(features[col].sum())

    # Define address ratio helper
    def _addr_ratio(a: object, b: object) -> float:
        a_str = str(a) if a is not None and not (isinstance(a, float) and pd.isna(a)) else ""
        b_str = str(b) if b is not None and not (isinstance(b, float) and pd.isna(b)) else ""
        score = fuzz.token_set_ratio(a_str, b_str)
        return score / 100.0

    # Add address similarity if present and has data
    if address_present and not df["address_clean"].isna().all():
        addr_scores = [
            _addr_ratio(df.loc[i1, "address_clean"], df.loc[i2, "address_clean"])
            for i1, i2 in pairs_index
        ]
        features["address_sim"] = addr_scores
        similarity_stats["mean"]["address_sim"] = float(pd.Series(addr_scores).mean())
        similarity_stats["high_similarity"]["address_sim_gt_0.8"] = sum(1 for score in addr_scores if score > 0.8)

    features = features.reset_index()
    os.makedirs(os.path.dirname(features_path), exist_ok=True)

    # Track column stats - only include available columns
    match_cols = ["company_clean"]
    for col in ["domain_clean", "phone_clean", "address_clean"]:
        if col in df.columns:
            match_cols.append(col)
            
    # Find state and country columns if they exist
    state_col = next((c for c in df.columns if re.sub(r"[ _]+", "", c).lower() == "state"), None)
    country_col = next((c for c in df.columns if re.sub(r"[ _]+", "", c).lower() == "countrycode"), None)
    if state_col:
        match_cols.append(state_col)
    if country_col:
        match_cols.append(country_col)

    # Append original values
    left = df.loc[features["record_id_1"], match_cols].reset_index(drop=True)
    left.columns = [f"{c}_1" for c in match_cols]
    right = df.loc[features["record_id_2"], match_cols].reset_index(drop=True)
    right.columns = [f"{c}_2" for c in match_cols]
    match_df = pd.concat([left, right], axis=1)
    features = pd.concat([features, match_df], axis=1)

    features.to_csv(features_path, index=False)

    # Print comprehensive terminal output
    print(f"\n📏 Similarity Feature Computation Complete!")
    print(f"─" * 50)
    print(f"📊 Data Overview:")
    print(f"  • Input records:         {initial_rows:,}")
    print(f"  • Input pairs:           {input_pairs:,}")
    print(f"  • Output features:       {len(features):,}")
    
    # Show computed features
    computed_features = [col for col in features.columns if col.endswith('_sim') or col.endswith('_exact')]
    print(f"\n🔧 Features Computed:")
    print(f"  • Total features:        {len(computed_features)}")
    for feature in computed_features:
        if feature in similarity_stats["mean"]:
            mean_val = similarity_stats["mean"][feature]
            print(f"  • {feature:<15} Mean: {mean_val:.3f}")
    
    # Show high similarity counts
    print(f"\n📈 High Similarity Pairs:")
    for metric, count in similarity_stats["high_similarity"].items():
        if "gt_0.8" in metric:
            feature_name = metric.replace("_gt_0.8", "")
            print(f"  • {feature_name:<15} >80%: {count:,}")
        elif "matches" in metric:
            feature_name = metric.replace("_matches", "")
            print(f"  • {feature_name:<15} Exact: {count:,}")
    
    # Show available vs missing columns
    available_cols = [col for col in match_cols if col in df.columns]
    missing_optional = ["domain_clean", "phone_clean", "address_clean"]
    missing_cols = [col for col in missing_optional if col not in df.columns or df[col].isna().all()]
    
    if missing_cols:
        print(f"\n⚠️  Missing Optional Data:")
        for col in missing_cols:
            clean_name = col.replace("_clean", "")
            print(f"  • {clean_name:<15} (will reduce accuracy)")
    
    print(f"\n💾 Files Created:")
    print(f"  • Feature matrix:        {features_path}")
    
    print(f"\n✅ Next step: Train model or run clustering")
    print(f"   Model:      python -m src.model")
    print(f"   Clustering: python -m src.clustering")

    end_time = time.time()
    stats = {
        "input_records": initial_rows,
        "input_pairs": input_pairs,
        "output_features": len(features),
        "columns_used": match_cols,
        "similarity_metrics": similarity_stats
    }
    log_run("similarity", start_time, end_time, len(features), additional_info=json.dumps(stats), log_path=log_path)
    return features


@click.command()
@click.option("--cleaned-path", default="data/outputs/cleaned.csv", show_default=True)
@click.option("--pairs-path", default="data/outputs/pairs.csv", show_default=True)
@click.option("--features-path", default="data/outputs/features.csv", show_default=True)
@click.option("--log-path", default=LOG_PATH, show_default=True)
def cli(cleaned_path: str, pairs_path: str, features_path: str, log_path: str) -> None:
    """CLI wrapper for :func:`main`."""
    main(cleaned_path, pairs_path, features_path, log_path)


if __name__ == "__main__":  # pragma: no cover - sanity run
    print("\u23e9 started similarity")
    cli()
