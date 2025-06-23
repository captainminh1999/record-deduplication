"""Step 3 of the 10-step deduplication pipeline: feature building.

Given candidate pairs, this module computes textual and exact-match
similarity metrics that serve as model features.
"""

from __future__ import annotations

import os
import re
import time

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
    start_time = time.time()
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned data not found: {cleaned_path}")

    df = pd.read_csv(cleaned_path)

    required = {
        "record_id",
        "phone_clean",
        "company_clean",
        "domain_clean",
    }
    address_present = "address_clean" in df.columns
    missing = required.difference(df.columns)
    if missing:
        cols = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {cols}")

    df = df.set_index("record_id")

    if os.path.exists(pairs_path):
        pairs_df = pd.read_csv(pairs_path)
        pairs_index = pd.MultiIndex.from_frame(pairs_df)
    else:
        pairs_index = generate_candidate_pairs(df)
        os.makedirs(os.path.dirname(pairs_path), exist_ok=True)
        pairs_index.to_frame(index=False).to_csv(pairs_path, index=False)

    comp = recordlinkage.Compare()
    comp.string(
        "company_clean",
        "company_clean",
        method="jarowinkler",
        label="company_sim",
    )
    comp.string(
        "domain_clean",
        "domain_clean",
        method="jarowinkler",
        label="domain_sim",
    )
    comp.exact("phone_clean", "phone_clean", label="phone_exact")

    features = comp.compute(pairs_index, df)

    # RapidFuzz token_set_ratio for addresses. Scale to [0,1] as future
    # thresholds may be tuned later. Additional phonetic or fuzzy-domain
    # checks can be implemented in future iterations.
    def _addr_ratio(a: object, b: object) -> float:
        score = fuzz.token_set_ratio(
            str(a) if pd.notna(a) else "",
            str(b) if pd.notna(b) else "",
        )
        return score / 100.0

    if address_present:
        addr_scores = [
            _addr_ratio(df.loc[i1, "address_clean"], df.loc[i2, "address_clean"])
            for i1, i2 in pairs_index
        ]
        features["address_sim"] = addr_scores

    features = features.reset_index()
    os.makedirs(os.path.dirname(features_path), exist_ok=True)

    # Build columns with the cleaned values for each record and
    # combine them with the similarity features.
    match_cols = ["company_clean", "domain_clean", "phone_clean"]
    if "address_clean" in df.columns:
        match_cols.append("address_clean")
    state_col = next((c for c in df.columns if re.sub(r"[ _]+", "", c).lower() == "state"), None)
    country_col = next((c for c in df.columns if re.sub(r"[ _]+", "", c).lower() == "countrycode"), None)
    if state_col:
        match_cols.append(state_col)
    if country_col:
        match_cols.append(country_col)

    left = df.loc[features["record_id_1"], match_cols].reset_index(drop=True)
    left.columns = [f"{c}_1" for c in match_cols]
    right = df.loc[features["record_id_2"], match_cols].reset_index(drop=True)
    right.columns = [f"{c}_2" for c in match_cols]
    match_df = pd.concat([left, right], axis=1)
    features = pd.concat([features, match_df], axis=1)

    features.to_csv(features_path, index=False)

    print(
        f"Computed {len(features)} feature rows and saved to {features_path}."
    )
    end_time = time.time()
    log_run("similarity", start_time, end_time, len(features), log_path=log_path)
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
