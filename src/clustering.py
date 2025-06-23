"""Cluster records based on pairwise similarity features."""

from __future__ import annotations

import os

import click
import pandas as pd
from sklearn.cluster import DBSCAN


@click.command()
@click.option("--features-path", default="data/outputs/features.csv", show_default=True)
@click.option("--cleaned-path", default="data/outputs/cleaned.csv", show_default=True)
@click.option("--eps", type=float, default=0.5, show_default=True)
@click.option("--min-samples", type=int, default=2, show_default=True)
@click.option("--output-path", default="data/outputs/clusters.csv", show_default=True)
def cli(features_path: str, cleaned_path: str, eps: float, min_samples: int, output_path: str) -> None:
    """CLI wrapper for :func:`main`."""
    main(features_path, cleaned_path, eps, min_samples, output_path)


def main(
    features_path: str = "data/outputs/features.csv",
    cleaned_path: str = "data/outputs/cleaned.csv",
    eps: float = 0.5,
    min_samples: int = 2,
    output_path: str = "data/outputs/clusters.csv",
) -> pd.DataFrame:
    """Generate DBSCAN clusters from similarity features."""

    feats = pd.read_csv(features_path)
    cleaned = pd.read_csv(cleaned_path).set_index("record_id")

    sim_cols = [c for c in ["company_sim", "domain_sim", "address_sim", "phone_exact"] if c in feats.columns]
    if not sim_cols:
        raise ValueError("No similarity columns found in features file")

    left = feats[["record_id_1"] + sim_cols].rename(columns={"record_id_1": "record_id"})
    right = feats[["record_id_2"] + sim_cols].rename(columns={"record_id_2": "record_id"})
    melted = pd.concat([left, right], ignore_index=True)
    melted[sim_cols] = melted[sim_cols].apply(pd.to_numeric, errors="coerce")

    agg = melted.groupby("record_id")[sim_cols].mean()
    agg = agg.reindex(cleaned.index, fill_value=0)

    # Persist the aggregated feature matrix for debugging or later tuning
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    agg.to_csv("data/outputs/agg_features.csv")
    print(f"Wrote aggregated features to data/outputs/agg_features.csv")

    # ``eps`` and ``min_samples`` can be fine-tuned later for different datasets
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(agg[sim_cols])
    agg["cluster"] = labels

    result = (
        agg[["cluster"]]
        .merge(
            cleaned[["domain_clean", "phone_clean", "address_clean"]],
            left_index=True,
            right_index=True,
            how="left",
        )
        .reset_index()
        .rename(columns={"index": "record_id"})
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} rows to {output_path}")

    return result


if __name__ == "__main__":  # pragma: no cover - sanity run
    cli()
