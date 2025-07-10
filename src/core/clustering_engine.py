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
from sklearn.preprocessing import StandardScaler, PowerTransformer
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
        eps: float = 0.1,
        min_samples: int = 2,
        scale: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Generate DBSCAN clusters from similarity features with enhanced feature engineering.
        
        Args:
            features_path: Path to similarity features CSV
            cleaned_path: Path to cleaned records CSV
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN minimum samples parameter
            scale: Whether to use enhanced scaling (PowerTransformer + StandardScaler)
            
        Returns:
            Tuple of (clustered_records, aggregated_features, stats)
        """
        # Initialize clustering statistics
        self.clustering_stats = {
            "parameters": {
                "eps": eps,
                "min_samples": min_samples,
                "scale": scale,
                "enhanced_features": True,
                "features_used": []
            },
            "data_stats": {
                "input_records": 0,
                "feature_stats": {}
            }
        }
        
        # Load data
        feats = pd.read_csv(features_path)
        cleaned = pd.read_csv(cleaned_path).set_index("record_id")
        self.clustering_stats["data_stats"]["input_records"] = len(cleaned)
        
        # Prepare enhanced features for clustering
        agg_features, feature_weights = self._prepare_enhanced_features(feats, cleaned)
        self.clustering_stats["parameters"]["feature_weights"] = feature_weights
        
        # Enhanced scaling if requested
        X = agg_features.values
        if scale:
            X, scaling_params = self._enhanced_scaling(X)
            self.clustering_stats["parameters"]["scaling"] = scaling_params
        
        # Perform clustering
        labels = self._perform_clustering(X, eps, min_samples)
        agg_features["cluster"] = labels
        
        # Calculate clustering statistics
        cluster_stats = self._calculate_cluster_stats(X, labels, eps, min_samples)
        self.clustering_stats["results"] = cluster_stats
        
        # Prepare output
        clustered_records = self._prepare_output(agg_features, cleaned)
        
        return clustered_records, agg_features, self.clustering_stats
    
    def _prepare_enhanced_features(
        self, 
        feats: pd.DataFrame, 
        cleaned: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Prepare enhanced features with advanced feature engineering for clustering."""
        # Base similarity features
        sim_cols = [c for c in ["company_sim", "domain_sim"] if c in feats.columns]
        if not sim_cols:
            raise ValueError("No similarity columns found in features file (need company_sim and/or domain_sim)")
        
        self.clustering_stats["parameters"]["features_used"] = sim_cols
        
        # Melt features to get all similarity values
        left = feats[["record_id_1"] + sim_cols].rename(columns={"record_id_1": "record_id"})
        right = feats[["record_id_2"] + sim_cols].rename(columns={"record_id_2": "record_id"})
        melted = pd.concat([left, right], ignore_index=True)
        melted[sim_cols] = melted[sim_cols].apply(pd.to_numeric, errors="coerce")
        
        # Track feature statistics before engineering
        for col in sim_cols:
            self.clustering_stats["data_stats"]["feature_stats"][col] = {
                "mean": float(melted[col].mean()),
                "std": float(melted[col].std()),
                "min": float(melted[col].min()),
                "max": float(melted[col].max()),
                "null_count": int(melted[col].isnull().sum())
            }
        
        # Enhanced Feature Engineering
        engineered_features = []
        
        # 1. Interaction Features
        if "company_sim" in sim_cols and "domain_sim" in sim_cols:
            melted["company_domain_product"] = melted["company_sim"] * melted["domain_sim"]
            melted["company_domain_sum"] = melted["company_sim"] + melted["domain_sim"]
            melted["company_domain_ratio"] = melted["company_sim"] / (melted["domain_sim"] + 1e-8)
            engineered_features.extend(["company_domain_product", "company_domain_sum", "company_domain_ratio"])
        
        # 2. Non-linear transformations
        for col in sim_cols:
            melted[f"{col}_squared"] = melted[col] ** 2
            melted[f"{col}_sqrt"] = np.sqrt(melted[col] + 1e-8)
            melted[f"{col}_log"] = np.log(melted[col] + 1e-8)
            engineered_features.extend([f"{col}_squared", f"{col}_sqrt", f"{col}_log"])
        
        # 3. Statistical features (variance, skewness indicators)
        if len(sim_cols) > 1:
            melted["sim_variance"] = melted[sim_cols].var(axis=1)
            melted["sim_mean"] = melted[sim_cols].mean(axis=1)
            melted["sim_max"] = melted[sim_cols].max(axis=1)
            melted["sim_min"] = melted[sim_cols].min(axis=1)
            melted["sim_range"] = melted["sim_max"] - melted["sim_min"]
            engineered_features.extend(["sim_variance", "sim_mean", "sim_max", "sim_min", "sim_range"])
        
        # Update features list
        all_features = sim_cols + engineered_features
        self.clustering_stats["parameters"]["features_used"] = all_features
        
        # Handle NaN values from log and sqrt operations
        for col in engineered_features:
            if col in melted.columns:
                melted[col] = melted[col].fillna(0)
        
        # Apply enhanced feature weights - prioritize domain similarity
        weights = {}
        
        # Base features
        if "company_sim" in sim_cols:
            weights["company_sim"] = 0.15  # Reduced weight for company similarity
            melted["company_sim"] = melted["company_sim"] * weights["company_sim"]
        if "domain_sim" in sim_cols:
            weights["domain_sim"] = 2.0  # High weight for domain similarity
            melted["domain_sim"] = melted["domain_sim"] * weights["domain_sim"]
        
        # Interaction features - high weights for domain-related interactions
        if "company_domain_product" in engineered_features:
            weights["company_domain_product"] = 1.5
            melted["company_domain_product"] = melted["company_domain_product"] * weights["company_domain_product"]
        if "company_domain_sum" in engineered_features:
            weights["company_domain_sum"] = 1.0
            melted["company_domain_sum"] = melted["company_domain_sum"] * weights["company_domain_sum"]
        if "company_domain_ratio" in engineered_features:
            weights["company_domain_ratio"] = 0.8
            melted["company_domain_ratio"] = melted["company_domain_ratio"] * weights["company_domain_ratio"]
        
        # Non-linear features - moderate weights
        for col in sim_cols:
            if f"{col}_squared" in engineered_features:
                weights[f"{col}_squared"] = 0.3
                melted[f"{col}_squared"] = melted[f"{col}_squared"] * weights[f"{col}_squared"]
            if f"{col}_sqrt" in engineered_features:
                weights[f"{col}_sqrt"] = 0.4
                melted[f"{col}_sqrt"] = melted[f"{col}_sqrt"] * weights[f"{col}_sqrt"]
            if f"{col}_log" in engineered_features:
                weights[f"{col}_log"] = 0.2
                melted[f"{col}_log"] = melted[f"{col}_log"] * weights[f"{col}_log"]
        
        # Statistical features - lower weights to avoid overfitting
        for stat_feat in ["sim_variance", "sim_mean", "sim_max", "sim_min", "sim_range"]:
            if stat_feat in engineered_features:
                weights[stat_feat] = 0.1
                melted[stat_feat] = melted[stat_feat] * weights[stat_feat]
        
        # Aggregate features by record
        agg = melted.groupby("record_id")[all_features].mean()
        agg = agg.reindex(cleaned.index, fill_value=0)
        
        return agg, weights
    
    def _enhanced_scaling(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Enhanced scaling using PowerTransformer followed by StandardScaler."""
        scaling_params = {}
        
        # Step 1: PowerTransformer to make data more normal
        power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        X_power = power_transformer.fit_transform(X)
        
        # Step 2: StandardScaler for final normalization
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_power)
        
        # Store scaling parameters
        try:
            scaling_params = {
                "power_transformer_lambdas": power_transformer.lambdas_.tolist() if hasattr(power_transformer, 'lambdas_') else [],
                "scaler_mean": self.scaler.mean_.tolist() if self.scaler.mean_ is not None else [],
                "scaler_scale": self.scaler.scale_.tolist() if self.scaler.scale_ is not None else [],
                "method": "PowerTransformer + StandardScaler"
            }
        except:
            scaling_params = {"method": "PowerTransformer + StandardScaler"}
        
        return X_scaled, scaling_params
    
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
        # Include company_clean along with other columns
        columns_to_include = ["company_clean", "domain_clean", "phone_clean", "address_clean"]
        available_columns = [col for col in columns_to_include if col in cleaned.columns]
        
        result = (
            agg_features[["cluster"]]
            .merge(
                cleaned[available_columns],
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
            "enhanced_features": params.get("enhanced_features", False),
            "n_clusters": results.get("n_clusters", 0),
            "n_noise": results.get("n_noise_points", 0),
            "silhouette_score": results.get("silhouette_score", 0.0)
        }
