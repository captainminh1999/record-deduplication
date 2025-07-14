"""Adaptive Hierarchical Clusterer V3

High-performance hierarchical clustering with adaptive depth and smart strategy selection.
Completely redesigned for efficiency and effectiveness.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import time
from sklearn.cluster import DBSCAN
from .subdivision_engine_v3 import SubdivisionEngineV3


class AdaptiveHierarchicalClusterer:
    """
    Next-generation hierarchical clusterer with adaptive depth and smart optimization.
    
    Key Features:
    - Adaptive depth: Continues until all clusters meet size constraints
    - Smart strategy selection: Prioritizes efficient methods for large clusters
    - Performance optimization: Built-in timeouts and progress monitoring
    - Simplified architecture: Removes unnecessary complexity
    """
    
    def __init__(self, timeout_seconds: int = 300):
        self.subdivision_engine = SubdivisionEngineV3()
        self.timeout_seconds = timeout_seconds
        self.clustering_stats = {}
    
    def cluster_with_adaptive_depth(
        self,
        X: np.ndarray,
        initial_labels: np.ndarray,
        base_eps: float,
        min_samples: int,
        max_cluster_size: int,
        max_absolute_depth: int = 20,  # Safety limit to prevent infinite loops
        performance_mode: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform hierarchical clustering with adaptive depth.
        
        Args:
            X: Feature matrix
            initial_labels: Initial cluster labels
            base_eps: Base eps value for DBSCAN
            min_samples: Minimum samples for DBSCAN
            max_cluster_size: Maximum cluster size (continues until all clusters meet this)
            max_absolute_depth: Safety limit for maximum depth
            performance_mode: Use fast strategies for very large clusters
            
        Returns:
            Tuple of (final_labels, clustering_stats)
        """
        start_time = time.time()
        current_labels = initial_labels.copy()
        depth = 0
        
        print(f"🚀 Adaptive hierarchical clustering (max_size={max_cluster_size})")
        print(f"   ⏱️  Timeout: {self.timeout_seconds}s, Max depth: {max_absolute_depth}")
        
        stats = {
            "start_time": start_time,
            "max_cluster_size": max_cluster_size,
            "levels": [],
            "total_subdivisions": 0,
            "performance_mode": performance_mode
        }
        
        while depth < max_absolute_depth:
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                print(f"⏱️  Timeout reached ({self.timeout_seconds}s), stopping at depth {depth}")
                break
            
            depth += 1
            print(f"\\n🧠 Level {depth} clustering...")
            
            # Find clusters that need subdivision
            large_clusters = self._find_large_clusters(current_labels, max_cluster_size)
            
            if not large_clusters:
                print(f"✅ All clusters meet size constraint (≤{max_cluster_size})")
                break
            
            print(f"   📊 Processing {len(large_clusters)} large clusters")
            
            # Process clusters with smart prioritization
            level_stats = self._process_level(
                X, current_labels, large_clusters, base_eps, min_samples, 
                depth, performance_mode, elapsed
            )
            
            stats["levels"].append(level_stats)
            stats["total_subdivisions"] += level_stats["subdivisions_made"]
            
            # Update labels
            current_labels = level_stats["updated_labels"]
            
            # Check if we made progress
            if level_stats["subdivisions_made"] == 0:
                print(f"⚠️  No subdivisions made at level {depth}, stopping")
                break
        
        # Final statistics
        stats["final_depth"] = depth
        stats["total_time"] = time.time() - start_time
        stats["final_cluster_count"] = len(set(current_labels[current_labels != -1]))
        
        self.clustering_stats = stats
        return current_labels, stats
    
    def _find_large_clusters(self, labels: np.ndarray, max_size: int) -> list:
        """Find clusters that exceed the maximum size, sorted by size (largest first)."""
        large_clusters = []
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise
                continue
                
            cluster_mask = labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size > max_size:
                large_clusters.append((cluster_id, cluster_mask, cluster_size))
        
        # Sort by size (largest first) for prioritized processing
        large_clusters.sort(key=lambda x: x[2], reverse=True)
        return large_clusters
    
    def _process_level(
        self,
        X: np.ndarray,
        current_labels: np.ndarray,
        large_clusters: list,
        base_eps: float,
        min_samples: int,
        depth: int,
        performance_mode: bool,
        elapsed_time: float
    ) -> Dict[str, Any]:
        """Process all large clusters at this level with smart strategy selection."""
        
        working_labels = current_labels.copy()
        subdivisions_made = 0
        clusters_processed = 0
        strategy_usage = {}
        
        for cluster_id, cluster_mask, cluster_size in large_clusters:
            clusters_processed += 1
            
            print(f"  📊 Subdividing cluster {cluster_id} (size: {cluster_size:,})")
            print(f"      Progress: {clusters_processed}/{len(large_clusters)} clusters")
            
            # Extract cluster data
            cluster_X = X[cluster_mask]
            
            # Smart strategy selection based on size and performance mode
            strategy_hint = self._select_strategy_hint(cluster_size, depth, performance_mode, elapsed_time)
            
            # Perform subdivision
            updated_labels, subdivision_info = self.subdivision_engine.perform_subdivision(
                cluster_X, cluster_mask, cluster_id, working_labels,
                base_eps, min_samples, cluster_size, strategy_hint
            )
            
            if subdivision_info["success"]:
                working_labels = updated_labels
                subdivisions_made += 1
                
                strategy_used = subdivision_info.get("strategy_used", "unknown")
                strategy_usage[strategy_used] = strategy_usage.get(strategy_used, 0) + 1
                
                new_clusters = subdivision_info.get("new_clusters", 0)
                print(f"    ✅ Split into {new_clusters} sub-clusters using {strategy_used}")
            else:
                print(f"    ❌ Subdivision failed for cluster {cluster_id}")
        
        return {
            "updated_labels": working_labels,
            "subdivisions_made": subdivisions_made,
            "clusters_processed": clusters_processed,
            "strategy_usage": strategy_usage,
            "depth": depth
        }
    
    def _select_strategy_hint(
        self, 
        cluster_size: int, 
        depth: int, 
        performance_mode: bool, 
        elapsed_time: float
    ) -> str:
        """Select the most appropriate strategy based on cluster characteristics."""
        
        # For very large clusters or when time is running out, prioritize speed
        if cluster_size > 10000 or elapsed_time > self.timeout_seconds * 0.7:
            return "force"  # Fastest, guaranteed success
        
        if cluster_size > 5000 and performance_mode:
            return "kmeans"  # Fast and effective for large clusters
        
        if cluster_size > 2000 and depth > 3:
            return "aggressive_pca"  # More aggressive for persistent large clusters
        
        if cluster_size > 1000 and depth > 5:
            return "kmeans"  # Switch to K-means for deep levels
        
        # Default to adaptive DBSCAN for smaller clusters and early levels
        return "adaptive_dbscan"
    
    def get_final_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the clustering process."""
        if not self.clustering_stats:
            return {}
        
        stats = self.clustering_stats.copy()
        
        # Calculate strategy usage summary
        total_strategy_usage = {}
        for level in stats.get("levels", []):
            for strategy, count in level.get("strategy_usage", {}).items():
                total_strategy_usage[strategy] = total_strategy_usage.get(strategy, 0) + count
        
        stats["total_strategy_usage"] = total_strategy_usage
        
        return stats

    def cluster_dataset(
        self,
        features_path: str,
        cleaned_path: str,
        eps: float,
        min_samples: int,
        scale: bool,
        max_cluster_size: int,
        max_absolute_depth: int = 20,
        performance_mode: bool = True
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Complete clustering pipeline that loads data and performs adaptive hierarchical clustering.
        
        Args:
            features_path: Path to features CSV file
            cleaned_path: Path to cleaned records CSV file
            eps: Base eps value for DBSCAN
            min_samples: Minimum samples for DBSCAN
            scale: Whether to scale features
            max_cluster_size: Maximum cluster size
            max_absolute_depth: Safety limit for maximum depth
            performance_mode: Use fast strategies for very large clusters
            
        Returns:
            Tuple of (clustered_records, agg_features, stats)
        """
        # Import required modules
        import pandas as pd
        import numpy as np
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler, PowerTransformer
        
        # Load features and records
        feats = pd.read_csv(features_path)
        cleaned = pd.read_csv(cleaned_path).set_index("record_id")
        
        # Prepare enhanced features using simplified logic
        agg_features = self._prepare_features_simple(feats, cleaned)
        
        # Enhanced scaling if requested
        X = agg_features.values
        if scale:
            X = self._scale_features_simple(X)
        
        # Perform initial DBSCAN clustering
        initial_dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        initial_labels = initial_dbscan.fit_predict(X)
        
        # Perform adaptive hierarchical clustering
        final_labels, clustering_stats = self.cluster_with_adaptive_depth(
            X=X,
            initial_labels=initial_labels,
            base_eps=eps,
            min_samples=min_samples,
            max_cluster_size=max_cluster_size,
            max_absolute_depth=max_absolute_depth,
            performance_mode=performance_mode
        )
        
        # Prepare results (same format as original engine)
        # Create clustered records
        cleaned_reset = cleaned.reset_index()
        clustered_records = cleaned_reset.copy()
        clustered_records['cluster_id'] = final_labels
        
        # Create aggregated features with cluster IDs
        agg_features_with_clusters = agg_features.copy()
        agg_features_with_clusters['cluster_id'] = final_labels
        
        return clustered_records, agg_features_with_clusters, clustering_stats

    def set_performance_mode(self):
        """Enable performance mode for faster processing."""
        self.subdivision_engine.set_performance_mode()
    
    def _prepare_features_simple(self, feats: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
        """Simplified feature preparation for clustering."""
        import pandas as pd
        import numpy as np
        
        # Base similarity features
        sim_cols = [c for c in ["company_sim", "domain_sim"] if c in feats.columns]
        if not sim_cols:
            raise ValueError("No similarity columns found in features file")
        
        # Melt features to get all similarity values
        left = feats[["record_id_1"] + sim_cols].rename(columns={"record_id_1": "record_id"})
        right = feats[["record_id_2"] + sim_cols].rename(columns={"record_id_2": "record_id"})
        melted = pd.concat([left, right], ignore_index=True)
        melted[sim_cols] = melted[sim_cols].apply(pd.to_numeric, errors="coerce")
        
        # Ultra-enhanced feature engineering (extremely prioritize domain-based interactions)
        if "company_sim" in sim_cols and "domain_sim" in sim_cols:
            melted["company_domain_product"] = melted["company_sim"] * melted["domain_sim"]
            melted["company_domain_sum"] = melted["company_sim"] + melted["domain_sim"]
            melted["domain_priority"] = melted["domain_sim"] * 10.0  # Ultra domain emphasis
            melted["domain_dominance"] = melted["domain_sim"] ** 2  # Exponential domain weighting
            sim_cols.extend(["company_domain_product", "company_domain_sum", "domain_priority", "domain_dominance"])
        
        # Apply ultra-aggressive weights (extremely prioritize domain similarity)
        if "domain_sim" in melted.columns:
            melted["domain_sim"] = melted["domain_sim"] * 1000.0  # Ultra-high weight for domain (10x boost from 100x)
            
            # Special boost for perfect domain matches, but preserve uniqueness for subdivision
            perfect_domain_mask = melted["domain_sim"] >= 1000.0  # After weighting, perfect match = 1000.0
            # Instead of making all perfect matches identical, add a small unique offset based on record pairs
            # This preserves the high priority while allowing domain-based subdivision
            if perfect_domain_mask.sum() > 0:
                # Add small incremental values (0.001, 0.002, etc.) to preserve uniqueness
                unique_offsets = np.arange(perfect_domain_mask.sum()) * 0.001
                melted.loc[perfect_domain_mask, "domain_sim"] = 4999.0 + unique_offsets
                print(f"    [DOMAIN-BOOST] Applied unique boosts to {perfect_domain_mask.sum()} perfect domain pairs (4999.000-4999.{unique_offsets[-1]:.3f})")
            
            
        if "company_sim" in melted.columns:
            melted["company_sim"] = melted["company_sim"] * 0.1  # Reduce company weight to minimize interference
        
        # Handle NaN values
        melted = melted.fillna(0)
        
        # Aggregate features by record with domain-preserving strategy
        # Use MAX for domain similarity to preserve perfect matches
        # Use MEAN for other similarities to maintain balance
        agg_funcs = {}
        for col in sim_cols:
            if "domain" in col:
                agg_funcs[col] = "max"  # Preserve strongest domain signal
            else:
                agg_funcs[col] = "mean"  # Balance other similarities
        
        agg = melted.groupby("record_id")[sim_cols].agg(agg_funcs)
        agg = agg.reindex(cleaned.index, fill_value=0)
        
        return agg
    
    def _scale_features_simple(self, X: np.ndarray) -> np.ndarray:
        """Simplified feature scaling."""
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled
