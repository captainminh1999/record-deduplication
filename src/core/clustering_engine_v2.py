"""Updated Clustering Engine - Business Logic

Core clustering functionality using modular components.
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, PowerTransformer

# Import our modular components
from .clustering import (
    FeatureEngineer,
    HierarchicalClusterer,
    ClusterStatsCalculator,
    AdaptiveEpsCalculator
)


class ClusteringEngine:
    """Core clustering engine for record deduplication using modular components."""
    
    def __init__(self):
        self.scaler = None
        self.clustering_model = None
        self.clustering_stats = {}
        
        # Initialize modular components
        self.feature_engineer = FeatureEngineer()
        self.hierarchical_clusterer = HierarchicalClusterer()
        self.stats_calculator = ClusterStatsCalculator()
    
    def cluster_records(
        self,
        features_path: str,
        cleaned_path: str,
        eps: float = 0.1,
        min_samples: int = 2,
        scale: bool = True,
        hierarchical: bool = False,
        max_cluster_size: int = 50,
        max_depth: int = 3,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Generate DBSCAN clusters from similarity features with enhanced feature engineering.
        
        Args:
            features_path: Path to similarity features CSV
            cleaned_path: Path to cleaned records CSV
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN minimum samples parameter
            scale: Whether to use enhanced scaling (PowerTransformer + StandardScaler)
            hierarchical: Whether to apply hierarchical clustering to break large clusters
            max_cluster_size: Maximum cluster size before subdivision (hierarchical only)
            max_depth: Maximum hierarchical depth (hierarchical only)
            
        Returns:
            Tuple of (clustered_records, agg_features, clustering_stats)
        """
        # Initialize stats tracking
        self.clustering_stats = {
            "parameters": {
                "eps": eps,
                "min_samples": min_samples,
                "scale": scale,
                "hierarchical": hierarchical,
                "max_cluster_size": max_cluster_size if hierarchical else None,
                "max_depth": max_depth if hierarchical else None,
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
        agg_features, feature_weights = self.feature_engineer.prepare_enhanced_features(feats, cleaned)
        self.clustering_stats["parameters"]["feature_weights"] = feature_weights
        
        # Extract feature matrix (exclude record_id)
        feature_cols = [col for col in agg_features.columns if col != "record_id"]
        X = agg_features[feature_cols].values
        
        # Enhanced scaling if requested
        if scale:
            X, scaling_stats = self._enhanced_scaling(X)
            self.clustering_stats["scaling"] = scaling_stats
        
        # Perform clustering
        if hierarchical:
            print(f"🌳 Hierarchical clustering with max_cluster_size={max_cluster_size}, max_depth={max_depth}")
            labels = self._perform_clustering(X, eps, min_samples)
            
            initial_clusters = len(set(labels[labels != -1]))
            print(f"   📊 Initial clustering: {initial_clusters} clusters")
            
            # Apply adaptive hierarchical subdivision with connectivity preservation
            final_labels, hierarchy_stats = self.hierarchical_clusterer.adaptive_hierarchical_clustering(
                X, labels, eps, min_samples, max_cluster_size, max_depth, 
                base_similarity_threshold=2.0, high_similarity_threshold=1.0, use_adaptive_thresholds=True
            )
            
            final_clusters = len(set(final_labels[final_labels != -1]))
            print(f"✅ Hierarchical clustering complete: {initial_clusters} → {final_clusters} clusters")
            
            self.clustering_stats["hierarchical"] = hierarchy_stats
            labels = final_labels
        else:
            labels = self._perform_clustering(X, eps, min_samples)
        
        agg_features["cluster"] = labels
        
        # Calculate clustering statistics using modular component
        cluster_stats = self.stats_calculator.calculate_cluster_stats(X, labels, eps, min_samples)
        self.clustering_stats["results"] = cluster_stats
        
        # Print summary
        print(self.stats_calculator.format_cluster_summary(cluster_stats))
        
        # Prepare output
        clustered_records = self._prepare_output(agg_features, cleaned)
        
        return clustered_records, agg_features, self.clustering_stats
    
    def hierarchical_clustering(
        self,
        features_path: str,
        cleaned_path: str,
        eps: float = 0.15,
        min_samples: int = 2,
        scale: bool = True,
        max_cluster_size: int = 50,
        max_depth: int = 3,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Perform hierarchical DBSCAN clustering to break large clusters into smaller ones.
        
        This is a convenience method that calls cluster_records with hierarchical=True.
        """
        return self.cluster_records(
            features_path=features_path,
            cleaned_path=cleaned_path,
            eps=eps,
            min_samples=min_samples,
            scale=scale,
            hierarchical=True,
            max_cluster_size=max_cluster_size,
            max_depth=max_depth
        )
    
    def _enhanced_scaling(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply enhanced scaling using PowerTransformer + StandardScaler."""
        
        # Step 1: PowerTransformer to handle non-normal distributions
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
    
    def _prepare_output(self, agg_features: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
        """Prepare final output with cluster assignments."""
        # Reset index for cleaned data
        cleaned_reset = cleaned.reset_index()
        
        # Merge cluster assignments with cleaned data
        result = cleaned_reset.merge(
            agg_features[["record_id", "cluster"]], 
            on="record_id", 
            how="left"
        )
        
        # Fill any missing cluster assignments with -1 (noise)
        result["cluster"] = result["cluster"].fillna(-1).astype(int)
        
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
        clustered_records.to_csv(output_path, index=False)
        print(f"Wrote {len(clustered_records)} clustered records to {output_path}")
        
        # Save aggregated features with cluster assignments
        agg_features.to_csv(agg_path, index=False)
        print(f"Wrote aggregated features (incl. cluster) to {agg_path}")
    
    def get_simple_stats(self) -> Dict[str, Any]:
        """Get simplified stats for logging."""
        if "results" not in self.clustering_stats:
            return {}
        
        results = self.clustering_stats["results"]
        params = self.clustering_stats["parameters"]
        
        simple_stats = {
            "eps": float(params["eps"]),
            "min_samples": int(params["min_samples"]),
            "scale": bool(params["scale"]),
            "enhanced_features": bool(params["enhanced_features"]),
            "hierarchical": bool(params["hierarchical"]),
            "max_cluster_size": int(params.get("max_cluster_size", 0)) if params.get("max_cluster_size") else None,
            "n_clusters": int(results["n_clusters"]),
            "n_noise": int(results["n_noise"]),
            "silhouette_score": float(results["silhouette_score"]) if results["silhouette_score"] is not None else 0.0
        }
        
        return simple_stats
