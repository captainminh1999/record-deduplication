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
            max_cluster_size: Maximum cluster size before hierarchical subdivision
            max_depth: Maximum depth for hierarchical clustering
            
        Returns:
            Tuple of (clustered_records, aggregated_features, stats)
        """
        # Initialize clustering statistics
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
        agg_features, feature_weights = self._prepare_enhanced_features(feats, cleaned)
        self.clustering_stats["parameters"]["feature_weights"] = feature_weights
        
        # Enhanced scaling if requested
        X = agg_features.values
        if scale:
            X, scaling_params = self._enhanced_scaling(X)
            self.clustering_stats["parameters"]["scaling"] = scaling_params
        
        # Perform clustering
        labels = self._perform_clustering(X, eps, min_samples)
        
        # Apply hierarchical clustering if requested
        if hierarchical:
            print(f"\n🔄 Applying hierarchical clustering (max size: {max_cluster_size})...")
            initial_clusters = len(set(labels[labels != -1]))
            labels = self._hierarchical_clustering(
                X, labels, eps, min_samples, max_cluster_size, max_depth
            )
            final_clusters = len(set(labels[labels != -1]))
            print(f"✅ Hierarchical clustering complete: {initial_clusters} → {final_clusters} clusters")
        
        agg_features["cluster"] = labels
        
        # Calculate clustering statistics
        cluster_stats = self._calculate_cluster_stats(X, labels, eps, min_samples)
        self.clustering_stats["results"] = cluster_stats
        
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
        
        This method first performs standard DBSCAN clustering, then recursively
        subdivides any clusters larger than max_cluster_size.
        
        Args:
            features_path: Path to similarity features CSV
            cleaned_path: Path to cleaned records CSV  
            eps: Initial DBSCAN epsilon parameter
            min_samples: DBSCAN minimum samples parameter
            scale: Whether to use enhanced scaling
            max_cluster_size: Maximum allowed cluster size before subdivision
            max_depth: Maximum depth of hierarchical subdivisions
            
        Returns:
            Tuple of (clustered_records, agg_features, clustering_stats)
        """
        print(f"🔄 Starting hierarchical clustering...")
        print(f"   📋 Parameters: eps={eps}, min_samples={min_samples}, max_size={max_cluster_size}, max_depth={max_depth}")
        
        # Initialize stats tracking
        self.clustering_stats = {
            "parameters": {
                "eps": eps,
                "min_samples": min_samples,
                "scale": scale,
                "hierarchical": True,
                "max_cluster_size": max_cluster_size,
                "max_depth": max_depth,
                "enhanced_features": True,
                "features_used": []
            },
            "data_stats": {
                "input_records": 0,
                "feature_stats": {}
            }
        }
        
        # Load and prepare data (same as cluster_records)
        agg_features = pd.read_csv(features_path)
        cleaned = pd.read_csv(cleaned_path)
        self.clustering_stats["data_stats"]["input_records"] = len(cleaned)
        
        # Enhanced feature engineering
        agg_features_enhanced, feature_weights = self._prepare_enhanced_features(agg_features, cleaned)
        self.clustering_stats["parameters"]["feature_weights"] = feature_weights
        
        # Convert to numpy array for clustering
        X = agg_features_enhanced.values
        
        # Enhanced scaling
        if scale:
            X, scaling_stats = self._enhanced_scaling(X)
            self.clustering_stats["scaling"] = scaling_stats
        
        # Initial DBSCAN clustering
        print(f"🎯 Phase 1: Initial DBSCAN clustering (eps={eps}, min_samples={min_samples})")
        initial_labels = self._perform_clustering(X, eps, min_samples)
        
        initial_clusters = len(set(initial_labels[initial_labels != -1]))
        print(f"   📊 Initial clustering: {initial_clusters} clusters")
        
        # Apply hierarchical subdivision
        print(f"🌳 Phase 2: Hierarchical subdivision")
        final_labels = self._hierarchical_clustering(
            X, initial_labels, eps, min_samples, max_cluster_size, max_depth
        )
        
        final_clusters = len(set(final_labels[final_labels != -1]))
        print(f"✅ Hierarchical clustering complete: {initial_clusters} → {final_clusters} clusters")
        
        agg_features_enhanced["cluster"] = final_labels
        
        # Calculate clustering statistics
        cluster_stats = self._calculate_cluster_stats(X, final_labels, eps, min_samples)
        self.clustering_stats["results"] = cluster_stats
        self.clustering_stats["hierarchical"] = {
            "initial_clusters": initial_clusters,
            "final_clusters": final_clusters,
            "max_cluster_size": max_cluster_size,
            "max_depth": max_depth
        }
        
        # Prepare output
        clustered_records = self._prepare_output(agg_features_enhanced, cleaned)
        
        return clustered_records, agg_features_enhanced, self.clustering_stats
    
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
            weights["company_sim"] = 0.0001  # Reduced weight for company similarity
            melted["company_sim"] = melted["company_sim"] * weights["company_sim"]
        if "domain_sim" in sim_cols:
            weights["domain_sim"] = 10.0  # High weight for domain similarity
            melted["domain_sim"] = melted["domain_sim"] * weights["domain_sim"]
        
        # Interaction features - high weights for domain-related interactions
        if "company_domain_product" in engineered_features:
            weights["company_domain_product"] = 3.0
            melted["company_domain_product"] = melted["company_domain_product"] * weights["company_domain_product"]
        if "company_domain_sum" in engineered_features:
            weights["company_domain_sum"] = 2.0
            melted["company_domain_sum"] = melted["company_domain_sum"] * weights["company_domain_sum"]
        if "company_domain_ratio" in engineered_features:
            weights["company_domain_ratio"] = 1.5
            melted["company_domain_ratio"] = melted["company_domain_ratio"] * weights["company_domain_ratio"]
        
        # Non-linear features - moderate weights
        for col in sim_cols:
            if f"{col}_squared" in engineered_features:
                weights[f"{col}_squared"] = 20.0
                melted[f"{col}_squared"] = melted[f"{col}_squared"] * weights[f"{col}_squared"]
            if f"{col}_sqrt" in engineered_features:
                weights[f"{col}_sqrt"] = 0.2
                melted[f"{col}_sqrt"] = melted[f"{col}_sqrt"] * weights[f"{col}_sqrt"]
            if f"{col}_log" in engineered_features:
                weights[f"{col}_log"] = 0.0
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
    
    def _hierarchical_clustering(
        self, 
        X: np.ndarray, 
        initial_labels: np.ndarray,
        base_eps: float,
        base_min_samples: int,
        max_cluster_size: int = 50,
        max_depth: int = 3,
        current_depth: int = 0
    ) -> np.ndarray:
        """PCA-based hierarchical clustering - break large clusters into smaller ones using adaptive feature space."""
        
        if current_depth >= max_depth:
            return initial_labels
        
        print(f"🧠 Level {current_depth + 1} PCA-based hierarchical clustering...")
        
        # Get unique clusters (excluding noise -1)
        unique_clusters = np.unique(initial_labels)
        unique_clusters = unique_clusters[unique_clusters != -1]
        
        # Track the next available cluster ID
        next_cluster_id = max(unique_clusters) + 1 if len(unique_clusters) > 0 else 0
        
        new_labels = initial_labels.copy()
        clusters_split = 0
        total_subdivisions = 0
        
        # Import PCA here to avoid module-level import issues
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors
        
        for cluster_id in unique_clusters:
            # Get points in this cluster
            cluster_mask = initial_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            # Only subdivide if cluster is too large
            if cluster_size > max_cluster_size:
                print(f"  📊 Subdividing cluster {cluster_id} (size: {cluster_size}) with PCA-adaptive clustering")
                
                # Extract features for this cluster
                cluster_X = X[cluster_mask]
                
                # Check if all features are identical (zero variance)
                feature_variances = np.var(cluster_X, axis=0)
                has_variance = np.any(feature_variances > 1e-12)
                
                if not has_variance:
                    print(f"    ⚠️  All features identical - adding minimal noise for subdivision")
                    # Add very small random noise to break ties
                    np.random.seed(42 + cluster_id)  # Reproducible per cluster
                    noise_scale = 1e-8
                    cluster_X = cluster_X + np.random.normal(0, noise_scale, cluster_X.shape)
                
                # Step 1: PCA transformation for this cluster's feature space
                try:
                    # Standardize features first
                    cluster_scaler = StandardScaler()
                    cluster_X_scaled = cluster_scaler.fit_transform(cluster_X)
                    
                    # Apply PCA to reduce dimensionality and find principal components
                    # Keep enough components to explain 95% variance (or all if too few samples)
                    max_components = min(cluster_X_scaled.shape[0] - 1, cluster_X_scaled.shape[1])
                    pca = PCA(n_components=min(max_components, 0.95))  # Keep 95% variance or max possible
                    
                    if max_components > 1:
                        cluster_X_pca = pca.fit_transform(cluster_X_scaled)
                        print(f"    🔄 PCA: {cluster_X_scaled.shape[1]} → {pca.n_components_} dimensions ({pca.explained_variance_ratio_.sum():.3f} variance)")
                    else:
                        # Too few samples for meaningful PCA
                        cluster_X_pca = cluster_X_scaled
                        print(f"    🔄 Skipping PCA (too few samples): using original {cluster_X_scaled.shape[1]} dimensions")
                    
                    # Step 2: Calculate adaptive eps for this PCA space
                    if cluster_X_pca.shape[0] > 4:  # Need at least 4 points for 4th NN
                        nn = NearestNeighbors(n_neighbors=min(4, cluster_X_pca.shape[0]))
                        nn.fit(cluster_X_pca)
                        distances, _ = nn.kneighbors(cluster_X_pca)
                        
                        # Use median of 4th (or last available) nearest neighbor distances
                        kth_distances = distances[:, -1]
                        adaptive_eps = np.median(kth_distances)
                        
                        # Apply hierarchical scaling factor
                        level_factor = 0.7 - (current_depth * 0.1)
                        adaptive_eps *= max(0.3, level_factor)
                        
                        print(f"    📏 Adaptive eps in PCA space: {adaptive_eps:.4f}")
                    else:
                        # Fallback for very small clusters
                        adaptive_eps = 0.1
                        print(f"    📏 Fallback eps (small cluster): {adaptive_eps:.4f}")
                    
                    # Step 3: DBSCAN clustering in PCA space
                    sub_min_samples = max(2, min(base_min_samples, cluster_size // 10))  # Adaptive min_samples
                    sub_model = DBSCAN(eps=adaptive_eps, min_samples=sub_min_samples)
                    sub_labels = sub_model.fit_predict(cluster_X_pca)
                    
                    # Analyze results
                    unique_sub_labels = np.unique(sub_labels)
                    unique_sub_labels = unique_sub_labels[unique_sub_labels != -1]
                    n_noise = np.sum(sub_labels == -1)
                    
                    print(f"    📈 PCA-DBSCAN results: {len(unique_sub_labels)} clusters, {n_noise} noise points")
                    
                    if len(unique_sub_labels) > 1:  # Only if we actually split the cluster
                        # Assign new cluster IDs
                        for sub_label in unique_sub_labels:
                            sub_mask = sub_labels == sub_label
                            original_indices = np.where(cluster_mask)[0][sub_mask]
                            new_labels[original_indices] = next_cluster_id
                            next_cluster_id += 1
                            total_subdivisions += 1
                        
                        # Handle noise points from sub-clustering (keep original cluster ID)
                        if n_noise > 0:
                            noise_mask = sub_labels == -1
                            original_noise_indices = np.where(cluster_mask)[0][noise_mask]
                            new_labels[original_noise_indices] = cluster_id  # Keep in original cluster
                        
                        clusters_split += 1
                        sub_cluster_sizes = [np.sum(sub_labels == label) for label in unique_sub_labels]
                        print(f"    ✅ Split into {len(unique_sub_labels)} sub-clusters (sizes: {sorted(sub_cluster_sizes, reverse=True)})")
                    else:
                        print(f"    ⚠️  No subdivision possible even with PCA adaptation")
                        
                except Exception as e:
                    print(f"    ❌ PCA subdivision failed: {str(e)}")
                    print(f"    🔄 Falling back to standard subdivision...")
                    
                    # Fallback to original method
                    sub_eps_factor = 0.7 - (current_depth * 0.1)
                    sub_eps = base_eps * max(0.3, sub_eps_factor)
                    sub_min_samples = max(2, base_min_samples - 1)
                    
                    sub_model = DBSCAN(eps=sub_eps, min_samples=sub_min_samples)
                    sub_labels = sub_model.fit_predict(cluster_X)
                    
                    unique_sub_labels = np.unique(sub_labels)
                    unique_sub_labels = unique_sub_labels[unique_sub_labels != -1]
                    
                    if len(unique_sub_labels) > 1:
                        for sub_label in unique_sub_labels:
                            sub_mask = sub_labels == sub_label
                            original_indices = np.where(cluster_mask)[0][sub_mask]
                            new_labels[original_indices] = next_cluster_id
                            next_cluster_id += 1
                            total_subdivisions += 1
                        
                        clusters_split += 1
                        print(f"    ✅ Fallback split successful")
                
        print(f"  📈 Split {clusters_split} large clusters into {total_subdivisions} sub-clusters at level {current_depth + 1}")
        
        # Recursive call for next level if we made changes
        if clusters_split > 0 and current_depth < max_depth - 1:
            return self._hierarchical_clustering(
                X, new_labels, base_eps, base_min_samples, 
                max_cluster_size, max_depth, current_depth + 1
            )
        else:
            return new_labels
    
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
            "hierarchical": params.get("hierarchical", False),
            "max_cluster_size": params.get("max_cluster_size"),
            "n_clusters": results.get("n_clusters", 0),
            "n_noise": results.get("n_noise_points", 0),
            "silhouette_score": results.get("silhouette_score", 0.0)
        }
