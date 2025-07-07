"""Clustering Engine - Business Logic

Core clustering functionality for the record deduplication pipeline.
This module contains the main clustering algorithms and business logic,
separated from CLI and formatting concerns.
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


class ClusteringEngine:
    """Core clustering engine for record deduplication."""
    
    def __init__(self):
        self.scaler = None
        self.clustering_model = None
        self.clustering_stats = {}
    
    def cluster_records(
        self,
        features_path: str,
        cleaned_path: str,
        eps: float = 0.004,
        min_samples: int = 3,
        scale: bool = False,
        auto_eps: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Generate DBSCAN clusters from similarity features.
        
        Args:
            features_path: Path to similarity features CSV
            cleaned_path: Path to cleaned records CSV
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN minimum samples parameter
            scale: Whether to standardize features before clustering
            auto_eps: Whether to automatically select optimal eps
            
        Returns:
            Tuple of (clustered_records, aggregated_features, stats)
        """
        # Initialize clustering statistics
        self.clustering_stats = {
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
        
        # Load data
        feats = pd.read_csv(features_path)
        cleaned = pd.read_csv(cleaned_path).set_index("record_id")
        self.clustering_stats["data_stats"]["input_records"] = len(cleaned)
        
        # Prepare features for clustering
        agg_features, feature_weights = self._prepare_features(feats, cleaned)
        self.clustering_stats["parameters"]["feature_weights"] = feature_weights
        
        # Scale features if requested
        X = agg_features.values
        if scale:
            X, scaling_params = self._scale_features(X)
            self.clustering_stats["parameters"]["scaling"] = scaling_params
        
        # Auto-select parameters if requested
        if auto_eps:
            eps, min_samples = self._auto_select_parameters(X, eps, min_samples)
        
        # Perform clustering
        labels = self._perform_clustering(X, eps, min_samples)
        agg_features["cluster"] = labels
        
        # Calculate clustering statistics
        cluster_stats = self._calculate_cluster_stats(X, labels, eps, min_samples)
        self.clustering_stats["results"] = cluster_stats
        
        # Prepare output
        clustered_records = self._prepare_output(agg_features, cleaned)
        
        return clustered_records, agg_features, self.clustering_stats
    
    def _prepare_features(
        self, 
        feats: pd.DataFrame, 
        cleaned: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Prepare and aggregate similarity features for clustering."""
        # Only use company_sim and domain_sim for clustering
        sim_cols = [c for c in ["company_sim", "domain_sim"] if c in feats.columns]
        if not sim_cols:
            raise ValueError("No similarity columns found in features file (need company_sim and/or domain_sim)")
        
        self.clustering_stats["parameters"]["features_used"] = sim_cols
        
        # Melt features to get all similarity values
        left = feats[["record_id_1"] + sim_cols].rename(columns={"record_id_1": "record_id"})
        right = feats[["record_id_2"] + sim_cols].rename(columns={"record_id_2": "record_id"})
        melted = pd.concat([left, right], ignore_index=True)
        melted[sim_cols] = melted[sim_cols].apply(pd.to_numeric, errors="coerce")
        
        # Track feature statistics before weighting
        for col in sim_cols:
            self.clustering_stats["data_stats"]["feature_stats"][col] = {
                "mean": float(melted[col].mean()),
                "std": float(melted[col].std()),
                "min": float(melted[col].min()),
                "max": float(melted[col].max()),
                "null_count": int(melted[col].isnull().sum())
            }
        
        # Apply feature weights
        weights = {}
        if "company_sim" in sim_cols:
            weights["company_sim"] = 1.0
            melted["company_sim"] = melted["company_sim"] * weights["company_sim"]
        if "domain_sim" in sim_cols:
            weights["domain_sim"] = 1.0
            melted["domain_sim"] = melted["domain_sim"] * weights["domain_sim"]
        
        # Aggregate features by record
        agg = melted.groupby("record_id")[sim_cols].mean()
        agg = agg.reindex(cleaned.index, fill_value=0)
        
        return agg, weights
    
    def _scale_features(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[float]]]:
        """Scale features using StandardScaler."""
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        scaling_params = {}
        try:
            scaling_params = {
                "mean": self.scaler.mean_.tolist() if self.scaler.mean_ is not None else [],
                "scale": self.scaler.scale_.tolist() if self.scaler.scale_ is not None else []
            }
        except:
            pass  # Ignore if scaling parameters are not available
        
        return X_scaled, scaling_params
    
    def _auto_select_parameters(
        self, 
        X: np.ndarray, 
        initial_eps: float, 
        initial_min_samples: int
    ) -> Tuple[float, int]:
        """Automatically select optimal eps and min_samples using k-distance and silhouette score."""
        n_samples = X.shape[0]
        
        # For very small datasets, use simple defaults
        if n_samples < 4:
            return 0.5, 2
        
        best_score = -1
        best_params = (initial_eps, initial_min_samples)
        # Adjust range based on dataset size
        max_min_samples = min(6, n_samples // 2)
        min_samples_range = range(2, max_min_samples + 1)
        
        # Initial parameter search
        for ms in min_samples_range:
            if ms >= n_samples:
                continue
            auto_eps = self._estimate_eps_kdistance(X, min(ms, n_samples - 1))
            labels = DBSCAN(eps=auto_eps, min_samples=ms).fit_predict(X)
            
            # Only score if more than 1 cluster and not all noise
            if len(set(labels)) > 1 and len(set(labels)) < len(X) and -1 in set(labels):
                try:
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_params = (auto_eps, ms)
                except Exception:
                    continue
        
        eps, min_samples = best_params
        
        # Only do refinement for larger datasets
        if n_samples >= 10:
            eps, min_samples = self._refine_parameters(X, eps, min_samples, best_score)
        
        return eps, min_samples
    
    def _estimate_eps_kdistance(self, X: np.ndarray, k: int) -> float:
        """Estimate eps using k-distance elbow method."""
        neigh = NearestNeighbors(n_neighbors=k)
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
            estimated_eps = (
                k_distances[kneedle.knee]
                if kneedle.knee is not None
                else k_distances[int(0.95 * len(k_distances))]
            )
        except ImportError:
            estimated_eps = k_distances[int(0.95 * len(k_distances))]
        
        # Ensure eps is positive and reasonable
        if estimated_eps <= 0 or not np.isfinite(estimated_eps):
            # Fallback to a reasonable default based on data
            estimated_eps = max(0.1, float(np.mean(k_distances)) + float(np.std(k_distances)))
        
        return float(estimated_eps)
    
    def _refine_parameters(
        self, 
        X: np.ndarray, 
        eps: float, 
        min_samples: int, 
        current_score: float
    ) -> Tuple[float, int]:
        """Iteratively refine clustering parameters using grid search."""
        max_iterations = 3
        convergence_threshold = 0.01
        search_range_eps = 0.2
        search_range_min_samples = 2
        
        current_eps, current_min_samples = eps, min_samples
        
        for iteration in range(max_iterations):
            # Define grid around current best parameters
            eps_grid = np.linspace(
                current_eps * (1 - search_range_eps), 
                current_eps * (1 + search_range_eps), 
                num=5
            )
            min_samples_grid = range(
                max(2, current_min_samples - search_range_min_samples), 
                current_min_samples + search_range_min_samples + 1
            )
            
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
                            if score > best_iter_score:
                                best_iter_score = score
                                best_iter_params = (e, ms)
                        except Exception:
                            continue
            
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
            self.clustering_stats["iterations"].append(iteration_stats)
            
            # Stop if no significant improvement
            if improvement < convergence_threshold:
                break
            
            # Update best parameters and narrow search range for next iteration
            current_eps, current_min_samples = best_iter_params
            current_score = best_iter_score
            search_range_eps *= 0.5
            search_range_min_samples = max(1, int(search_range_min_samples * 0.5))
        
        return current_eps, current_min_samples
    
    def _perform_clustering(self, X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
        """Perform DBSCAN clustering."""
        self.clustering_model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = self.clustering_model.fit_predict(X)
        return labels
    
    def _calculate_cluster_stats(
        self, 
        X: np.ndarray, 
        labels: np.ndarray, 
        eps: float, 
        min_samples: int
    ) -> Dict[str, Any]:
        """Calculate comprehensive clustering statistics."""
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        cluster_sizes = pd.Series(labels[labels != -1]).value_counts().to_dict()
        
        # Calculate silhouette score
        try:
            final_silhouette = float(silhouette_score(X, labels)) if n_clusters > 1 else 0.0
        except Exception:
            final_silhouette = 0.0
        
        return {
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
    
    def _prepare_output(self, agg_features: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
        """Prepare final clustered records output."""
        result = (
            agg_features[["cluster"]]
            .merge(
                cleaned[["domain_clean", "phone_clean", "address_clean"]],
                left_index=True,
                right_index=True,
                how="left",
            )
            .reset_index()
            .rename(columns={"index": "record_id"})
        )
        return result
    
    def save_results(
        self, 
        clustered_records: pd.DataFrame, 
        agg_features: pd.DataFrame,
        output_path: str,
        agg_path: str
    ) -> None:
        """Save clustering results to files."""
        # Save clustered records
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        clustered_records.to_csv(output_path, index=False)
        
        # Save aggregated features
        os.makedirs(os.path.dirname(agg_path), exist_ok=True)
        agg_features.reset_index().to_csv(agg_path, index=False)
    
    def get_simple_stats(self) -> Dict[str, Any]:
        """Get simplified statistics for logging."""
        if not self.clustering_stats or "results" not in self.clustering_stats:
            return {}
        
        results = self.clustering_stats["results"]
        params = self.clustering_stats["parameters"]
        
        return {
            "eps": results.get("final_eps", 0.0),
            "min_samples": results.get("final_min_samples", 0),
            "scale": params.get("scale", False),
            "auto_eps": params.get("auto_eps", False),
            "n_clusters": results.get("n_clusters", 0),
            "n_noise": results.get("n_noise_points", 0),
            "silhouette_score": results.get("silhouette_score", 0.0)
        }
