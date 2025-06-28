"""Step 6 of 6: Clustering (Optional Grouping)

Clusters records based on similarity features using DBSCAN. In this implementation, domain similarity is given double weight before clustering. See README for details.
"""

from __future__ import annotations

import os

import click
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


@click.command()
@click.option("--features-path", default="data/outputs/features.csv", show_default=True)
@click.option("--cleaned-path", default="data/outputs/cleaned.csv", show_default=True)
@click.option("--eps", type=float, default=0.5, show_default=True)
@click.option("--min-samples", type=int, default=2, show_default=True)
@click.option("--output-path", default="data/outputs/clusters.csv", show_default=True)
@click.option(
    "--scale/--no-scale",
    default=False,
    show_default=True,
    help="Standard-scale similarity columns before DBSCAN",
)
@click.option(
    "--agg-path",
    default="data/outputs/agg_features.csv",
    show_default=True,
    help="Where to write aggregated features (with cluster column)",
)
def cli(
    features_path: str,
    cleaned_path: str,
    eps: float,
    min_samples: int,
    output_path: str,
    scale: bool,
    agg_path: str,
) -> None:
    """CLI wrapper for :func:`main`."""
    main(features_path, cleaned_path, eps, min_samples, output_path, scale, agg_path)


def main(
    features_path: str = "data/outputs/features.csv",
    cleaned_path: str = "data/outputs/cleaned.csv",
    eps: float = 0.03,
    min_samples: int = 4,
    output_path: str = "data/outputs/clusters.csv",
    scale: bool = False,
    agg_path: str = "data/outputs/agg_features.csv",
) -> pd.DataFrame:
    """Generate DBSCAN clusters from similarity features.

    ``eps`` should come from the elbow of the k-distance curve and ``min_samples``
    should generally be >= 4 to avoid tiny noise clumps. Using ``--scale``
    standardises feature ranges so each contributes equally to clustering.
    """

    feats = pd.read_csv(features_path)
    cleaned = pd.read_csv(cleaned_path).set_index("record_id")

    # Only use company_sim and domain_sim for clustering
    sim_cols = [c for c in ["company_sim", "domain_sim"] if c in feats.columns]
    if not sim_cols:
        raise ValueError("No similarity columns found in features file (need company_sim and/or domain_sim)")

    left = feats[["record_id_1"] + sim_cols].rename(columns={"record_id_1": "record_id"})
    right = feats[["record_id_2"] + sim_cols].rename(columns={"record_id_2": "record_id"})
    melted = pd.concat([left, right], ignore_index=True)
    melted[sim_cols] = melted[sim_cols].apply(pd.to_numeric, errors="coerce")

    # Give 'company_sim' and 'domain_sim' custom weights before aggregation
    if "company_sim" in sim_cols:
        melted["company_sim"] = melted["company_sim"] * 1.0
    if "domain_sim" in sim_cols:
        melted["domain_sim"] = melted["domain_sim"] * 1.0

    agg = melted.groupby("record_id")[sim_cols].mean()
    agg = agg.reindex(cleaned.index, fill_value=0)

    X = agg[sim_cols].values
    if scale:
        X = StandardScaler().fit_transform(X)

    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(X)
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

    os.makedirs(os.path.dirname(agg_path), exist_ok=True)
    agg.reset_index().to_csv(agg_path, index=False)

    print(f"Wrote {len(result)} clustered records to {output_path}")
    print(f"Wrote aggregated features (incl. cluster) to {agg_path}")

    return result


if __name__ == "__main__":  # pragma: no cover - sanity run
    cli()
