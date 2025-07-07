"""Step 6 of 6: Clustering (Optional Grouping)

Clusters records based on similarity features using DBSCAN. In this implementation, domain similarity is given double weight before clustering. See README for details.
"""

from __future__ import annotations

import os
import time
import json

import click
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from .utils import log_run, LOG_PATH


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
@click.option(
    "--auto-eps/--no-auto-eps",
    default=False,
    show_default=True,
    help="Automatically select eps using k-distance elbow method",
)
def cli(
    features_path: str,
    cleaned_path: str,
    eps: float,
    min_samples: int,
    output_path: str,
    scale: bool,
    agg_path: str,
    auto_eps: bool,
) -> None:
    """CLI wrapper for :func:`main`."""
    main(features_path, cleaned_path, eps, min_samples, output_path, scale, agg_path, auto_eps)


def main(
    features_path: str = "data/outputs/features.csv",
    cleaned_path: str = "data/outputs/cleaned.csv",
    eps: float = 0.004,
    min_samples: int = 3,
    output_path: str = "data/outputs/clusters.csv",
    scale: bool = False,
    agg_path: str = "data/outputs/agg_features.csv",
    auto_eps: bool = False,  # New argument to enable auto eps
) -> pd.DataFrame:
    """Generate DBSCAN clusters from similarity features."""
    
    start_time = time.time()
    clustering_stats = {
        "parameters": {
            "eps": eps,
            "min_samples": min_samples,
            "scale": scale,
            "auto_eps": auto_eps,
            "features_used": []
        },
        "data_stats": {
            "input_records": 0,
            "feature_stats": {}
        },
        "iterations": []
    }

    feats = pd.read_csv(features_path)
    cleaned = pd.read_csv(cleaned_path).set_index("record_id")
    clustering_stats["data_stats"]["input_records"] = len(cleaned)

    # Only use company_sim and domain_sim for clustering
    sim_cols = [c for c in ["company_sim", "domain_sim"] if c in feats.columns]
    if not sim_cols:
        raise ValueError("No similarity columns found in features file (need company_sim and/or domain_sim)")
    
    clustering_stats["parameters"]["features_used"] = sim_cols

    left = feats[["record_id_1"] + sim_cols].rename(columns={"record_id_1": "record_id"})
    right = feats[["record_id_2"] + sim_cols].rename(columns={"record_id_2": "record_id"})
    melted = pd.concat([left, right], ignore_index=True)
    melted[sim_cols] = melted[sim_cols].apply(pd.to_numeric, errors="coerce")

    # Track feature statistics before weighting
    for col in sim_cols:
        clustering_stats["data_stats"]["feature_stats"][col] = {
            "mean": float(melted[col].mean()),
            "std": float(melted[col].std()),
            "min": float(melted[col].min()),
            "max": float(melted[col].max()),
            "null_count": int(melted[col].isnull().sum())
        }

    # Give 'company_sim' and 'domain_sim' custom weights before aggregation
    weights = {}
    if "company_sim" in sim_cols:
        weights["company_sim"] = 1.0
        melted["company_sim"] = melted["company_sim"] * weights["company_sim"]
    if "domain_sim" in sim_cols:
        weights["domain_sim"] = 1.0
        melted["domain_sim"] = melted["domain_sim"] * weights["domain_sim"]
    
    clustering_stats["parameters"]["feature_weights"] = weights

    agg = melted.groupby("record_id")[sim_cols].mean()
    agg = agg.reindex(cleaned.index, fill_value=0)

    X = agg[sim_cols].values
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        try:
            # Track scaling parameters if available
            clustering_stats["parameters"]["scaling"] = {
                "mean": scaler.mean_.tolist() if scaler.mean_ is not None else [],
                "scale": scaler.scale_.tolist() if scaler.scale_ is not None else []
            }
        except:
            pass  # Ignore if scaling parameters are not available

    # --- K-distance (elbow) method automation for eps and min_samples ---
    if auto_eps:
        # Try a small range of min_samples and pick the best by Silhouette Score
        from sklearn.metrics import silhouette_score
        import numpy as np
        best_score = -1
        best_params = (eps, min_samples)
        min_samples_range = range(2, 7)  # Try min_samples from 2 to 6
        for ms in min_samples_range:
            neigh = NearestNeighbors(n_neighbors=ms)
            nbrs = neigh.fit(X)
            distances, _ = nbrs.kneighbors(X)
            k_distances = sorted(distances[:, -1])
            try:
                from kneed import KneeLocator
                kneedle = KneeLocator(
                    range(len(k_distances)),
                    k_distances,
                    S=1.0,
                    curve="convex",
                    direction="increasing",
                )
                auto_selected_eps = (
                    k_distances[kneedle.knee]
                    if kneedle.knee is not None
                    else k_distances[int(0.95 * len(k_distances))]
                )
            except ImportError:
                auto_selected_eps = k_distances[int(0.95 * len(k_distances))]
            labels = DBSCAN(eps=auto_selected_eps, min_samples=ms).fit_predict(X)
            # Only score if more than 1 cluster and not all noise
            if len(set(labels)) > 1 and len(set(labels)) < len(X) and -1 in set(labels):
                try:
                    score = silhouette_score(X, labels)
                except Exception:
                    score = -1
                if score > best_score:
                    best_score = score
                    best_params = (auto_selected_eps, ms)
        eps, min_samples = best_params
        print(f"[Auto] Initial eps: {eps:.6f}, min_samples: {min_samples} (Silhouette Score: {best_score:.4f})")

        # --- Iterative Grid Search Refinement ---
        max_iterations = 3  # Maximum number of refinement iterations
        convergence_threshold = 0.01  # Stop if improvement in score is less than this
        search_range_eps = 0.2  # Initial search range for eps (±20%)
        search_range_min_samples = 2  # Initial search range for min_samples (±2)

        current_eps, current_min_samples = eps, min_samples
        current_score = best_score

        for iteration in range(max_iterations):
            # Define grid around current best parameters
            eps_grid = np.linspace(current_eps * (1 - search_range_eps), 
                                  current_eps * (1 + search_range_eps), num=5)
            min_samples_grid = range(max(2, current_min_samples - search_range_min_samples), 
                                    current_min_samples + search_range_min_samples + 1)
            
            # Search the grid
            best_iter_score = current_score
            best_iter_params = (current_eps, current_min_samples)
            
            for e in eps_grid:
                for ms in min_samples_grid:
                    labels = DBSCAN(eps=e, min_samples=ms).fit_predict(X)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters > 1 and n_clusters < len(X):
                        try:
                            score = silhouette_score(X, labels)
                        except Exception:
                            score = -1
                        if score > best_iter_score:
                            best_iter_score = score
                            best_iter_params = (e, ms)
            
            # Check for improvement
            improvement = best_iter_score - current_score
            
            # Log iteration results
            iteration_stats = {
                "iteration": iteration + 1,
                "eps": best_iter_params[0],
                "min_samples": best_iter_params[1],
                "silhouette_score": best_iter_score,
                "improvement": improvement
            }
            clustering_stats["iterations"].append(iteration_stats)
            print(f"[Refinement {iteration+1}] eps: {best_iter_params[0]:.6f}, min_samples: {best_iter_params[1]} "
                  f"(Score: {best_iter_score:.4f}, Improvement: {improvement:.4f})")
            
            # Stop if no significant improvement
            if improvement < convergence_threshold:
                print(f"Converged after {iteration+1} iterations (improvement < {convergence_threshold})")
                break
            
            # Update best parameters and narrow search range for next iteration
            current_eps, current_min_samples = best_iter_params
            current_score = best_iter_score
            search_range_eps *= 0.5  # Reduce search range by half each iteration
            search_range_min_samples = max(1, int(search_range_min_samples * 0.5))

        # Use the final best parameters
        eps, min_samples = current_eps, current_min_samples
        print(f"[Final] Best eps: {eps:.6f}, min_samples: {min_samples} (Silhouette Score: {current_score:.4f})")

    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(X)
    agg["cluster"] = labels

    # Calculate final clustering statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    # Calculate additional cluster metrics
    cluster_sizes = pd.Series(labels[labels != -1]).value_counts().to_dict()
    
    # Try to calculate silhouette score for valid clusterings
    try:
        from sklearn.metrics import silhouette_score
        final_silhouette = float(silhouette_score(X, labels)) if n_clusters > 1 else 0.0
    except Exception:
        final_silhouette = 0.0

    clustering_stats["results"] = {
        "n_clusters": n_clusters,
        "n_noise_points": n_noise,
        "n_total_points": len(labels),
        "noise_ratio": float(n_noise / len(labels)),
        "final_eps": float(eps),
        "final_min_samples": int(min_samples),
        "silhouette_score": final_silhouette,
        "cluster_sizes": cluster_sizes,
        "cluster_stats": {
            "min_size": min(cluster_sizes.values()) if cluster_sizes else 0,
            "max_size": max(cluster_sizes.values()) if cluster_sizes else 0,
            "mean_size": float(sum(cluster_sizes.values()) / len(cluster_sizes)) if cluster_sizes else 0
        }
    }

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

    end_time = time.time()
    log_run(
        "clustering",
        start_time,
        end_time,
        len(result),
        additional_info=json.dumps(clustering_stats, indent=2),
        log_path=LOG_PATH,
    )

    return result


if __name__ == "__main__":  # pragma: no cover - sanity run
    cli()
