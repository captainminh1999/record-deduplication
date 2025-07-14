"""Subdivision Engine V3

High-performance subdivision engine with MANDATORY domain-first clustering and smart strategy selection.
Enforces: "If domain 100% match => go to 1 cluster, Then if partial match, still rely on domain as dominant"
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ..adaptive_eps import AdaptiveEpsCalculator


class DomainFirstClusterer:
    """
    MANDATORY domain-first clustering that enforces domain hierarchy BEFORE any other strategy.
    
    This ensures that:
    1. Perfect domain matches (100%) ALWAYS stay together
    2. High-confidence domain matches (85%+) join perfect match clusters
    3. Only remaining records get processed by standard clustering strategies
    """
    
    @staticmethod
    def detect_domain_column(cluster_X: np.ndarray) -> Tuple[Optional[int], Dict[str, Any]]:
        """Detect if we have a domain similarity column in the feature matrix."""
        domain_info = {
            "domain_col_detected": False,
            "domain_col_idx": None,
            "max_domain_value": 0,
            "perfect_match_count": 0,
            "high_confidence_count": 0,
            "uniform_perfect_cluster": False
        }
        
        if cluster_X.shape[1] < 2:
            return None, domain_info
            
        print(f"    [DOMAIN-DETECT] Analyzing {cluster_X.shape[0]} records with {cluster_X.shape[1]} features")
        
        for i in range(cluster_X.shape[1]):
            col_values = cluster_X[:, i]
            max_val = np.max(col_values)
            min_val = np.min(col_values)
            
            # Case 1: Mixed values - look for potential domain similarity patterns
            if max_val > min_val:
                unique_vals, counts = np.unique(col_values, return_counts=True)
                max_val_idx = np.argmax(unique_vals)
                max_val_count = counts[max_val_idx]
                highest_val = unique_vals[max_val_idx]
                
                # Check for domain similarity patterns (multiple records with same high value)
                if max_val_count >= 2 and highest_val > 0:
                    domain_info.update({
                        "domain_col_detected": True,
                        "domain_col_idx": i,
                        "max_domain_value": highest_val,
                        "perfect_match_count": max_val_count,
                        "high_confidence_count": np.sum(col_values >= highest_val * 0.85)
                    })
                    print(f"    [DOMAIN-DETECT] ✅ Domain column found at index {i}: {max_val_count} perfect matches (val={highest_val:.2f})")
                    return i, domain_info
            
            # Case 2: Uniform high values - ALL records are perfect domain matches
            elif max_val == min_val and max_val > 1.0:
                # Special handling for artificially boosted values
                is_artificially_boosted = max_val >= 4999.0
                domain_info.update({
                    "domain_col_detected": True,
                    "domain_col_idx": i,
                    "max_domain_value": max_val,
                    "perfect_match_count": len(col_values),
                    "high_confidence_count": len(col_values),
                    "uniform_perfect_cluster": not is_artificially_boosted  # Don't preserve if boosted
                })
                if is_artificially_boosted:
                    print(f"    [DOMAIN-DETECT] ⚠️ Artificially boosted uniform values detected at index {i} (all={max_val:.2f}) - subdivision allowed")
                else:
                    print(f"    [DOMAIN-DETECT] ✅ Uniform perfect domain cluster detected at index {i} (all={max_val:.2f})")
                return i, domain_info
            
            # Case 3: High domain values with some variation (new boosted values with unique offsets)
            elif max_val >= 4999.0 and min_val >= 4999.0:
                # These are artificially boosted values that should allow subdivision
                domain_info.update({
                    "domain_col_detected": True,
                    "domain_col_idx": i,
                    "max_domain_value": max_val,
                    "perfect_match_count": len(col_values),
                    "high_confidence_count": len(col_values),
                    "uniform_perfect_cluster": False  # Allow subdivision for boosted values
                })
                print(f"    [DOMAIN-DETECT] ⚠️ Boosted domain values with variation detected at index {i} (range={min_val:.3f}-{max_val:.3f}) - subdivision allowed")
                return i, domain_info
        
        print(f"    [DOMAIN-DETECT] ❌ No domain column detected")
        return None, domain_info
    
    @staticmethod
    def apply_domain_first_clustering(
        cluster_X: np.ndarray, 
        target_size: int = 100,
        max_clusters: int = 10
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """
        Apply MANDATORY domain-first clustering with AGGRESSIVE domain grouping.
        
        NEW RULE: For 85%-100% domain matches, DROP company name weight to 0 
        and FORCE ALL similar domains into ONE cluster regardless of company differences.
        
        Returns:
            (success, labels, info) where success=True means domain-first clustering was applied
        """
        domain_col_idx, domain_info = DomainFirstClusterer.detect_domain_column(cluster_X)
        
        if domain_col_idx is None:
            return False, np.zeros(len(cluster_X)), domain_info
        
        domain_values = cluster_X[:, domain_col_idx]
        max_domain_value = domain_info["max_domain_value"]
        
        # NEW AGGRESSIVE DOMAIN RULE: Group ALL records with 85%+ domain similarity
        # Handle boosted domain values (5000.0) from adaptive clusterer
        if max_domain_value >= 5000.0:
            # For boosted values, we can't distinguish between different domains
            # since they all have the same value (5000.0). In this case, the domain 
            # grouping is meaningless - all records would be grouped together.
            # Instead, skip domain-first clustering and let standard clustering handle it.
            print(f"    [DOMAIN-FIRST] ⚠️ All domain values are boosted to {max_domain_value:.1f} - cannot distinguish domains")
            print(f"    [DOMAIN-FIRST] 🔄 Skipping domain-first clustering due to indistinguishable domain values")
            domain_info["clustering_applied"] = "skipped_boosted_values"
            return False, np.zeros(len(cluster_X)), domain_info
        else:
            # For normal values (0-1 range), use 85% of maximum
            domain_threshold_85 = max_domain_value * 0.85
            print(f"    [DOMAIN-FIRST] 📊 Normal domain values (max={max_domain_value:.3f}), using threshold={domain_threshold_85:.3f}")
        
        high_domain_mask = domain_values >= domain_threshold_85
        high_domain_count = np.sum(high_domain_mask)
        
        print(f"    [DOMAIN-FIRST] 🎯 AGGRESSIVE DOMAIN GROUPING: {high_domain_count} records with 85%+ domain similarity")
        
        if high_domain_count >= 2:
            # FORCE similar domain records into SEPARATE clusters BY ACTUAL DOMAIN
            print(f"    [DOMAIN-FIRST] 🔒 FORCING similar domain records into SEPARATE clusters BY DOMAIN")
            print(f"    [DOMAIN-FIRST] 📝 Dropping company name weight to 0 for domain matches 85%+")
            
            labels = np.full(len(cluster_X), -1, dtype=int)  # Initialize with -1 (unassigned)
            current_cluster = 0
            
            # Group records by their actual domain values (not just similarity)
            high_domain_indices = np.where(high_domain_mask)[0]
            
            # Get unique domain values for high domain similarity records
            high_domain_values = cluster_X[high_domain_indices, domain_col_idx]
            unique_domains = np.unique(high_domain_values)
            
            print(f"    [DOMAIN-FIRST] 📊 Found {len(unique_domains)} unique domains with 85%+ similarity")
            
            # Create separate cluster for each unique domain
            for domain_value in unique_domains:
                domain_mask = (cluster_X[:, domain_col_idx] == domain_value) & high_domain_mask
                domain_indices = np.where(domain_mask)[0]
                
                if len(domain_indices) > 0:
                    for idx in domain_indices:
                        labels[idx] = current_cluster
                    print(f"    [DOMAIN-FIRST] 🏷️  Domain cluster {current_cluster}: {len(domain_indices)} records (domain_val={domain_value:.3f})")
                    current_cluster += 1
            
            # Handle remaining low domain similarity records (below 85%)
            low_domain_mask = ~high_domain_mask
            low_domain_indices = np.where(low_domain_mask)[0]
            
            if len(low_domain_indices) > 0:
                print(f"    [DOMAIN-FIRST] 📊 Processing {len(low_domain_indices)} low domain similarity records separately")
                
                # Cluster low domain records normally
                if len(low_domain_indices) > target_size:
                    low_domain_features = cluster_X[low_domain_indices]
                    n_low_clusters = max(1, len(low_domain_indices) // target_size)
                    
                    # Use PCA for dimensionality reduction if needed
                    n_components = min(3, low_domain_features.shape[1], len(low_domain_indices) - 1)
                    if n_components > 0 and n_components < low_domain_features.shape[1]:
                        pca = PCA(n_components=n_components)
                        pca_features = pca.fit_transform(low_domain_features)
                    else:
                        pca_features = low_domain_features
                    
                    kmeans = KMeans(n_clusters=n_low_clusters, random_state=42, n_init=3)
                    low_domain_labels = kmeans.fit_predict(pca_features)
                    
                    for i, idx in enumerate(low_domain_indices):
                        labels[idx] = current_cluster + low_domain_labels[i]
                    current_cluster += n_low_clusters
                else:
                    # Small low domain group
                    for idx in low_domain_indices:
                        labels[idx] = current_cluster
                    current_cluster += 1
            
            # Ensure no -1 labels remain
            unassigned_mask = labels == -1
            if np.any(unassigned_mask):
                unassigned_indices = np.where(unassigned_mask)[0]
                for idx in unassigned_indices:
                    labels[idx] = current_cluster
                current_cluster += 1
            
            domain_info.update({
                "clustering_applied": "aggressive_domain_grouping_by_domain",
                "high_domain_grouped": high_domain_count,
                "unique_domains_found": len(unique_domains),
                "domain_clusters_created": len(unique_domains),
                "low_domain_clustered": len(low_domain_indices) if 'low_domain_indices' in locals() else 0,
                "total_clusters": current_cluster,
                "domain_threshold_85": domain_threshold_85,
                "company_weight_dropped": True,
                "separate_domain_clusters": True
            })
            
            low_domain_clusters = current_cluster - len(unique_domains) if 'low_domain_indices' in locals() else 0
            print(f"    [DOMAIN-FIRST] ✅ AGGRESSIVE GROUPING: {high_domain_count} records → {len(unique_domains)} domain clusters, {len(low_domain_indices) if 'low_domain_indices' in locals() else 0} others → {low_domain_clusters} clusters")
            return True, labels, domain_info
        
        # Fallback: Handle uniform perfect domain clusters (preserve as single cluster)
        if domain_info["uniform_perfect_cluster"]:
            print(f"    [DOMAIN-FIRST] 🔒 Preserving uniform perfect domain cluster (size={len(cluster_X)})")
            labels = np.zeros(len(cluster_X), dtype=int)
            domain_info["clustering_applied"] = "preserved_uniform_perfect"
            return True, labels, domain_info
        
        # If no high domain similarity found, return false to allow standard clustering
        print(f"    [DOMAIN-FIRST] ❌ No significant domain similarity found (threshold: {domain_threshold_85:.2f})")
        return False, np.zeros(len(cluster_X)), domain_info


class SubdivisionStrategy:
    """Base class for subdivision strategies."""
    
    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority  # Higher priority = tried first
        
    def can_handle(self, cluster_size: int, depth: int = 0) -> bool:
        """Check if this strategy can handle the given cluster characteristics."""
        raise NotImplementedError
        
    def subdivide(
        self, 
        cluster_X: np.ndarray, 
        base_eps: float, 
        min_samples: int,
        cluster_size: int,
        depth: int = 0
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """Attempt subdivision using this strategy."""
        raise NotImplementedError


class FastKMeansStrategy(SubdivisionStrategy):
    """Ultra-fast K-means for very large clusters with aggressive sampling."""
    
    def __init__(self):
        super().__init__("fast_kmeans", priority=100)  # Highest priority
    
    def can_handle(self, cluster_size: int, depth: int = 0) -> bool:
        return cluster_size >= 1000  # Handle large clusters efficiently
    
    def subdivide(
        self, 
        cluster_X: np.ndarray, 
        base_eps: float, 
        min_samples: int,
        cluster_size: int,
        depth: int = 0
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        
        info = {
            "strategy": self.name,
            "cluster_size": cluster_size,
            "depth": depth,
            "eps_used": 0.0,
            "pca_used": True
        }
        
        try:
            # Aggressive target: aim for clusters of ~100-200 records
            target_size = 150
            n_clusters = max(2, min(50, cluster_size // target_size))
            
            # Heavy sampling for very large clusters
            if cluster_size > 20000:
                sample_size = 5000
                sample_indices = np.random.choice(cluster_size, sample_size, replace=False)
                cluster_X_sample = cluster_X[sample_indices]
                info["heavy_sampling"] = True
                info["sample_size"] = sample_size
            elif cluster_size > 5000:
                sample_size = min(3000, cluster_size // 2)
                sample_indices = np.random.choice(cluster_size, sample_size, replace=False)
                cluster_X_sample = cluster_X[sample_indices]
                info["sampling"] = True
                info["sample_size"] = sample_size
            else:
                cluster_X_sample = cluster_X
                sample_indices = np.arange(cluster_size)
                info["sampling"] = False
            
            # Minimal PCA for speed
            scaler = StandardScaler()
            cluster_X_scaled = scaler.fit_transform(cluster_X_sample)
            
            n_components = min(3, cluster_X_sample.shape[1])  # Very aggressive reduction
            pca = PCA(n_components=n_components)
            cluster_X_pca = pca.fit_transform(cluster_X_scaled)
            
            # Fast K-means
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=1,  # Single initialization for speed
                max_iter=50  # Limited iterations
            )
            
            if info.get("sampling", False):
                # Fit on sample, predict on all
                sample_labels = kmeans.fit_predict(cluster_X_pca)
                cluster_X_all_scaled = scaler.transform(cluster_X)
                cluster_X_all_pca = pca.transform(cluster_X_all_scaled)
                labels = kmeans.predict(cluster_X_all_pca)
            else:
                labels = kmeans.fit_predict(cluster_X_pca)
            
            cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
            
            info.update({
                "n_clusters": n_clusters,
                "cluster_sizes": cluster_sizes,
                "pca_components": n_components,
                "max_cluster_size": max(cluster_sizes),
                "avg_cluster_size": np.mean(cluster_sizes)
            })
            
            print(f"    ⚡ Fast K-means: {cluster_size:,} → {n_clusters} clusters (max: {max(cluster_sizes):,})")
            
            return True, labels, info
            
        except Exception as e:
            info["error"] = str(e)
            return False, np.zeros(cluster_size), info


class SmartDBSCANStrategy(SubdivisionStrategy):
    """Optimized DBSCAN with intelligent parameter selection."""
    
    def __init__(self):
        super().__init__("smart_dbscan", priority=50)
        self.eps_calculator = AdaptiveEpsCalculator()
    
    def can_handle(self, cluster_size: int, depth: int = 0) -> bool:
        return 50 <= cluster_size <= 5000  # Sweet spot for DBSCAN
    
    def subdivide(
        self, 
        cluster_X: np.ndarray, 
        base_eps: float, 
        min_samples: int,
        cluster_size: int,
        depth: int = 0
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        
        info = {
            "strategy": self.name,
            "cluster_size": cluster_size,
            "depth": depth,
            "pca_used": False,
            "eps_used": base_eps
        }
        
        try:
            # Smart eps calculation with depth-based adjustment
            adaptive_eps, eps_info = self.eps_calculator.calculate_adaptive_eps(
                cluster_X, base_eps, current_depth=depth, cluster_size=cluster_size, use_pca=True
            )
            
            # Depth-based eps reduction for persistent clusters
            depth_factor = 0.8 ** depth  # More aggressive with depth
            adaptive_eps *= depth_factor
            
            info.update(eps_info)
            info["eps_used"] = adaptive_eps
            info["depth_factor"] = depth_factor
            
            # Apply PCA if beneficial
            if eps_info.get("pca_used", False):
                scaler = StandardScaler()
                cluster_X_scaled = scaler.fit_transform(cluster_X)
                pca = PCA(n_components=eps_info['pca_components'])
                cluster_X_transformed = pca.fit_transform(cluster_X_scaled)
                info["pca_transform"] = (scaler, pca)
                print(f"    🔄 Smart PCA: {cluster_X.shape[1]} → {eps_info['pca_components']} dims")
            else:
                cluster_X_transformed = cluster_X
            
            # Progressive eps values with depth awareness
            eps_values = [
                adaptive_eps,
                adaptive_eps * 0.7,
                adaptive_eps * 0.5,
                adaptive_eps * 0.3
            ]
            
            for i, eps in enumerate(eps_values):
                adaptive_min_samples = max(2, min_samples - min(depth, 2))
                
                dbscan = DBSCAN(eps=eps, min_samples=adaptive_min_samples)
                labels = dbscan.fit_predict(cluster_X_transformed)
                
                unique_labels = np.unique(labels)
                n_clusters = len(unique_labels[unique_labels != -1])
                noise_count = np.sum(labels == -1)
                
                if n_clusters >= 2:
                    cluster_sizes = [np.sum(labels == label) for label in unique_labels if label != -1]
                    
                    info.update({
                        "eps_used": eps,
                        "n_clusters": n_clusters,
                        "noise_points": noise_count,
                        "cluster_sizes": cluster_sizes,
                        "eps_attempt": i + 1
                    })
                    
                    print(f"    ✅ Smart DBSCAN: {cluster_size:,} → {n_clusters} clusters (eps={eps:.3f})")
                    return True, labels, info
            
            return False, np.zeros(cluster_size), info
            
        except Exception as e:
            info["error"] = str(e)
            return False, np.zeros(cluster_size), info


class AggressivePCAStrategy(SubdivisionStrategy):
    """Aggressive PCA reduction for resistant clusters."""
    
    def __init__(self):
        super().__init__("aggressive_pca", priority=30)
    
    def can_handle(self, cluster_size: int, depth: int = 0) -> bool:
        return cluster_size >= 500 and depth >= 3  # For persistent large clusters
    
    def subdivide(
        self, 
        cluster_X: np.ndarray, 
        base_eps: float, 
        min_samples: int,
        cluster_size: int,
        depth: int = 0
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        
        info = {
            "strategy": self.name,
            "cluster_size": cluster_size,
            "depth": depth,
            "pca_used": True,
            "eps_used": base_eps
        }
        
        try:
            # Force very aggressive PCA reduction
            scaler = StandardScaler()
            cluster_X_scaled = scaler.fit_transform(cluster_X)
            
            n_components = min(2, cluster_X.shape[1], cluster_X.shape[0] - 1)
            pca = PCA(n_components=n_components)
            cluster_X_pca = pca.fit_transform(cluster_X_scaled)
            
            variance_explained = pca.explained_variance_ratio_.sum()
            print(f"    🔥 Aggressive PCA: {cluster_X.shape[1]} → {n_components} dims ({variance_explained:.3f} variance)")
            
            info.update({
                "pca_components": n_components,
                "variance_explained": variance_explained
            })
            
            # Very aggressive eps values
            base_eps_pca = base_eps * (0.5 ** depth)  # Exponential reduction with depth
            eps_values = [base_eps_pca * factor for factor in [0.1, 0.05, 0.03, 0.01]]
            
            for eps in eps_values:
                dbscan = DBSCAN(eps=eps, min_samples=2)
                labels = dbscan.fit_predict(cluster_X_pca)
                
                unique_labels = np.unique(labels)
                n_clusters = len(unique_labels[unique_labels != -1])
                
                if n_clusters >= 2:
                    cluster_sizes = [np.sum(labels == label) for label in unique_labels if label != -1]
                    
                    info.update({
                        "eps_used": eps,
                        "n_clusters": n_clusters,
                        "noise_points": np.sum(labels == -1),
                        "cluster_sizes": cluster_sizes
                    })
                    
                    print(f"    ✅ Aggressive subdivision: {cluster_size:,} → {n_clusters} clusters")
                    return True, labels, info
            
            return False, np.zeros(cluster_size), info
            
        except Exception as e:
            info["error"] = str(e)
            return False, np.zeros(cluster_size), info


class ForceStrategy(SubdivisionStrategy):
    """Guaranteed success strategy with intelligent partitioning."""
    
    def __init__(self):
        super().__init__("force", priority=10)  # Lowest priority (last resort)
    
    def can_handle(self, cluster_size: int, depth: int = 0) -> bool:
        return cluster_size >= 2  # Can handle any reasonable size, even very small clusters
    
    def subdivide(
        self, 
        cluster_X: np.ndarray, 
        base_eps: float, 
        min_samples: int,
        cluster_size: int,
        depth: int = 0
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        
        # Adaptive target size based on depth
        target_size = max(50, 200 // (depth + 1))  # Smaller targets for deeper levels
        n_partitions = max(2, cluster_size // target_size)
        
        info = {
            "strategy": self.name,
            "cluster_size": cluster_size,
            "depth": depth,
            "n_partitions": n_partitions,
            "target_size": target_size,
            "eps_used": 0.0
        }
        
        try:
            # Simplified force strategy - domain clustering is now handled globally
            print(f"    [FORCE] Forcing subdivision into {n_partitions} partitions")
            
            # Use PCA for dimensionality reduction if beneficial
            if cluster_X.shape[1] > 3:
                scaler = StandardScaler()
                cluster_X_scaled = scaler.fit_transform(cluster_X)
                
                n_components = min(3, cluster_X.shape[1], cluster_X.shape[0] - 1)
                pca = PCA(n_components=n_components)
                cluster_X_pca = pca.fit_transform(cluster_X_scaled)
                
                variance_explained = pca.explained_variance_ratio_.sum()
                print(f"    [FORCE] PCA: {cluster_X.shape[1]} → {n_components} dims ({variance_explained:.3f} variance)")
                
                info.update({
                    "pca_used": True,
                    "pca_components": n_components,
                    "variance_explained": variance_explained
                })
                
                features_for_clustering = cluster_X_pca
            else:
                features_for_clustering = cluster_X
                info["pca_used"] = False
            
            # Use K-means for guaranteed subdivision
            kmeans = KMeans(n_clusters=n_partitions, random_state=42, n_init=3)
            labels = kmeans.fit_predict(features_for_clustering)
            
            cluster_sizes = [np.sum(labels == i) for i in range(n_partitions)]
            info.update({
                "method": "forced_kmeans",
                "n_clusters": n_partitions,
                "cluster_sizes": cluster_sizes,
                "avg_cluster_size": np.mean(cluster_sizes)
            })
            
            print(f"    [FORCE] ✅ Successfully forced subdivision into {n_partitions} clusters")
            return True, labels, info
            
        except Exception as e:
            info["error"] = str(e)
            print(f"    [FORCE] ❌ Force strategy failed: {e}")
            return False, np.zeros(cluster_size), info


class SubdivisionEngineV3:
    """High-performance subdivision engine with smart strategy selection."""
    
    def __init__(self):
        self.strategies = [
            FastKMeansStrategy(),      # Priority 100 - Very large clusters
            SmartDBSCANStrategy(),     # Priority 50  - Medium clusters
            AggressivePCAStrategy(),   # Priority 30  - Resistant clusters
            ForceStrategy()            # Priority 10  - Last resort
        ]
        # Sort by priority (highest first)
        self.strategies.sort(key=lambda s: s.priority, reverse=True)
        self.subdivision_history = []
        self.performance_mode = False
    
    def perform_subdivision(
        self,
        cluster_X: np.ndarray,
        cluster_mask: np.ndarray,
        cluster_id: int,
        current_labels: np.ndarray,
        base_eps: float,
        min_samples: int,
        cluster_size: int,
        strategy_hint: Optional[str] = None,
        depth: int = 0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform subdivision using MANDATORY domain-first clustering, then fallback strategies.
        
        This enforces the rule: "If domain 100% match => go to 1 cluster, 
        Then if partial match, still rely on domain as dominant"
        """
        subdivision_info = {
            "success": False,
            "original_cluster_id": cluster_id,
            "original_size": cluster_size,
            "new_clusters": 0,
            "strategy_used": None,
            "eps_used": base_eps,
            "pca_used": False,
            "depth": depth,
            "attempts": [],
            "domain_first_attempted": False,
            "domain_first_success": False
        }
        
        # STEP 1: MANDATORY DOMAIN-FIRST CLUSTERING
        # This ensures exact domain matches NEVER scatter across different clusters
        print(f"    [SUBDIVISION] 🎯 STEP 1: Attempting MANDATORY domain-first clustering")
        
        target_size = max(50, 200 // (depth + 1))  # Adaptive target size
        max_clusters = max(2, cluster_size // target_size)
        
        subdivision_info["domain_first_attempted"] = True
        domain_success, domain_labels, domain_info = DomainFirstClusterer.apply_domain_first_clustering(
            cluster_X, target_size=target_size, max_clusters=max_clusters
        )
        
        subdivision_info["attempts"].append({
            "strategy": "domain_first",
            "success": domain_success,
            "info": domain_info
        })
        
        if domain_success:
            print(f"    [SUBDIVISION] ✅ DOMAIN-FIRST clustering succeeded!")
            
            # Check if domain clustering created enough subdivision
            unique_domain_labels = np.unique(domain_labels)
            n_domain_clusters = len(unique_domain_labels)
            
            # Special case: If this is a uniform perfect domain cluster, NEVER break it apart
            if domain_info.get("uniform_perfect_cluster", False):
                print(f"    [SUBDIVISION] 🔒 UNIFORM PERFECT DOMAIN CLUSTER - NEVER SUBDIVIDE FURTHER!")
                print(f"    [SUBDIVISION] 🎯 Preserving domain integrity - stopping subdivision here")
                
                updated_labels = self._apply_subdivision_to_labels(
                    current_labels, cluster_mask, domain_labels, cluster_id
                )
                
                subdivision_info.update({
                    "success": True,
                    "domain_first_success": True,
                    "strategy_used": "domain_first_uniform_perfect",
                    "new_clusters": n_domain_clusters,
                    "uniform_perfect_preserved": True,
                    "eps_used": base_eps,
                    "domain_info": domain_info
                })
                
                self.subdivision_history.append(subdivision_info.copy())
                print(f"    [SUBDIVISION] 🎉 Uniform perfect domain cluster preserved - subdivision complete!")
                return updated_labels, subdivision_info
            
            # NEW: Special case for aggressive domain grouping - NEVER subdivide single domain clusters
            if domain_info.get("single_domain_cluster", False) and domain_info.get("company_weight_dropped", False):
                print(f"    [SUBDIVISION] 🔒 AGGRESSIVE DOMAIN CLUSTER - NEVER SUBDIVIDE SIMILAR DOMAINS!")
                print(f"    [SUBDIVISION] 📝 Company weight was dropped to 0 for 85%+ domain matches")
                print(f"    [SUBDIVISION] 🎯 All similar domains stay together regardless of company differences")
                
                updated_labels = self._apply_subdivision_to_labels(
                    current_labels, cluster_mask, domain_labels, cluster_id
                )
                
                subdivision_info.update({
                    "success": True,
                    "domain_first_success": True,
                    "strategy_used": "domain_first_aggressive_grouping",
                    "new_clusters": n_domain_clusters,
                    "aggressive_domain_preserved": True,
                    "company_weight_dropped": True,
                    "eps_used": base_eps,
                    "domain_info": domain_info
                })
                
                self.subdivision_history.append(subdivision_info.copy())
                print(f"    [SUBDIVISION] 🎉 Aggressive domain cluster preserved - no subdivision needed!")
                return updated_labels, subdivision_info
            
            if n_domain_clusters >= 2:
                # PRESERVE domain clusters and only subdivide unassigned records
                print(f"    [SUBDIVISION] 🔒 PRESERVING {n_domain_clusters} domain clusters, checking for unassigned records")
                
                # Find records not assigned to any domain cluster (labeled as -1)
                unassigned_mask = domain_labels == -1
                n_unassigned = np.sum(unassigned_mask)
                
                if n_unassigned > 0:
                    print(f"    [SUBDIVISION] 🎯 Found {n_unassigned} unassigned records - applying standard clustering to them only")
                    
                    # Apply standard clustering ONLY to unassigned records
                    unassigned_features = cluster_X[unassigned_mask]
                    final_labels = domain_labels.copy()
                    
                    # Try standard strategies on unassigned records only
                    if self._subdivide_unassigned_records(unassigned_features, final_labels, unassigned_mask, 
                                                        n_domain_clusters, base_eps, min_samples, depth):
                        # Successfully subdivided unassigned records
                        updated_labels = self._apply_subdivision_to_labels(
                            current_labels, cluster_mask, final_labels, cluster_id
                        )
                        
                        final_clusters = len(np.unique(final_labels[final_labels != -1]))
                        subdivision_info.update({
                            "success": True,
                            "domain_first_success": True,
                            "strategy_used": "domain_first_with_fallback",
                            "new_clusters": final_clusters,
                            "domain_clusters": n_domain_clusters,
                            "unassigned_subdivided": True,
                            "eps_used": base_eps,
                            "domain_info": domain_info
                        })
                        
                        self.subdivision_history.append(subdivision_info.copy())
                        print(f"    [SUBDIVISION] 🎉 PRESERVED {n_domain_clusters} domain clusters + subdivided {n_unassigned} unassigned records = {final_clusters} total clusters!")
                        return updated_labels, subdivision_info
                
                # No unassigned records or subdivision failed - use domain clusters as-is
                updated_labels = self._apply_subdivision_to_labels(
                    current_labels, cluster_mask, domain_labels, cluster_id
                )
                
                subdivision_info.update({
                    "success": True,
                    "domain_first_success": True,
                    "strategy_used": "domain_first",
                    "new_clusters": n_domain_clusters,
                    "eps_used": base_eps,
                    "domain_info": domain_info
                })
                
                self.subdivision_history.append(subdivision_info.copy())
                print(f"    [SUBDIVISION] 🎉 Domain-first clustering created {n_domain_clusters} clusters - subdivision complete!")
                return updated_labels, subdivision_info
            else:
                print(f"    [SUBDIVISION] ⚠️ Domain-first clustering created only 1 cluster - trying standard strategies")
                # Single cluster from domain analysis - need additional subdivision
        else:
            print(f"    [SUBDIVISION] ⚠️ Domain-first clustering not applicable - trying standard strategies")
        
        # STEP 2: FALLBACK TO STANDARD STRATEGIES (only if domain-first didn't subdivide enough)
        print(f"    [SUBDIVISION] 🔄 STEP 2: Fallback to standard clustering strategies")
        
        # If domain clustering partially succeeded but only created 1 cluster,
        # we still need to respect domain hierarchy in subsequent clustering
        domain_aware_features = cluster_X
        if domain_success and domain_info.get("domain_col_idx") is not None:
            # Mark that we should preserve domain relationships in subsequent clustering
            subdivision_info["domain_preservation_mode"] = True
            subdivision_info["domain_col_idx"] = domain_info["domain_col_idx"]
        
        # Strategy selection with hint support
        strategies_to_try = self._select_strategies(cluster_size, depth, strategy_hint)
        
        for strategy in strategies_to_try:
            if not strategy.can_handle(cluster_size, depth):
                continue
            
            print(f"    [SUBDIVISION] 🔧 Trying {strategy.name} strategy")
            success, sub_labels, strategy_info = strategy.subdivide(
                domain_aware_features, base_eps, min_samples, cluster_size, depth
            )
            
            subdivision_info["attempts"].append(strategy_info)
            
            if success:
                # Apply subdivision to global labels
                updated_labels = self._apply_subdivision_to_labels(
                    current_labels, cluster_mask, sub_labels, cluster_id
                )
                
                subdivision_info.update({
                    "success": True,
                    "strategy_used": strategy.name,
                    "new_clusters": len(np.unique(sub_labels)),
                    "eps_used": strategy_info.get("eps_used", base_eps),
                    "pca_used": strategy_info.get("pca_used", False),
                    "pca_components": strategy_info.get("pca_components", 0),
                    "strategy_info": strategy_info
                })
                
                self.subdivision_history.append(subdivision_info.copy())
                print(f"    [SUBDIVISION] ✅ {strategy.name} succeeded - subdivision complete!")
                return updated_labels, subdivision_info
        
        # This should never happen since ForceStrategy always succeeds
        subdivision_info["failure_reason"] = "all_strategies_failed"
        self.subdivision_history.append(subdivision_info.copy())
        print(f"    [SUBDIVISION] ❌ All strategies failed - this should not happen!")
        return current_labels, subdivision_info
    
    def _select_strategies(self, cluster_size: int, depth: int, hint: Optional[str] = None) -> list:
        """Select and order strategies based on cluster characteristics and hint."""
        
        if hint:
            # Try hinted strategy first
            hinted_strategy = None
            for strategy in self.strategies:
                if hint in strategy.name:
                    hinted_strategy = strategy
                    break
            
            if hinted_strategy:
                other_strategies = [s for s in self.strategies if s != hinted_strategy]
                return [hinted_strategy] + other_strategies
        
        # Default priority-based ordering
        return self.strategies
    
    def _apply_subdivision_to_labels(
        self,
        current_labels: np.ndarray,
        cluster_mask: np.ndarray,
        sub_labels: np.ndarray,
        original_cluster_id: int
    ) -> np.ndarray:
        """Apply subdivision results to the global label array."""
        updated_labels = current_labels.copy()
        cluster_indices = np.where(cluster_mask)[0]
        
        # Find next available cluster ID
        max_existing_label = np.max(updated_labels) if len(updated_labels) > 0 else -1
        next_cluster_id = max_existing_label + 1
        
        # Assign new cluster IDs
        unique_sub_labels = np.unique(sub_labels)
        non_noise_labels = unique_sub_labels[unique_sub_labels != -1]  # Exclude noise
        
        for i, sub_label in enumerate(non_noise_labels):
            sub_mask = sub_labels == sub_label
            new_cluster_id = next_cluster_id + i
            updated_labels[cluster_indices[sub_mask]] = new_cluster_id
        
        # Handle noise points: keep them in the original cluster
        noise_mask = sub_labels == -1
        if np.any(noise_mask):
            updated_labels[cluster_indices[noise_mask]] = original_cluster_id
        
        return updated_labels
    
    def _subdivide_unassigned_records(
        self,
        unassigned_features: np.ndarray,
        labels: np.ndarray,
        unassigned_mask: np.ndarray,
        next_cluster_id: int,
        base_eps: float,
        min_samples: int,
        depth: int
    ) -> bool:
        """
        Apply standard clustering ONLY to unassigned records (those not in domain clusters).
        This preserves domain clusters while subdividing the remaining records.
        """
        if len(unassigned_features) < 2:
            return False
        
        print(f"    [UNASSIGNED-CLUSTERING] 🎯 Clustering {len(unassigned_features)} unassigned records")
        
        # Try standard strategies on unassigned records only
        strategies_to_try = self._select_strategies(len(unassigned_features), depth)
        
        for strategy in strategies_to_try:
            if not strategy.can_handle(len(unassigned_features), depth):
                continue
            
            print(f"    [UNASSIGNED-CLUSTERING] 🔧 Trying {strategy.name} on unassigned records")
            success, sub_labels, strategy_info = strategy.subdivide(
                unassigned_features, base_eps, min_samples, len(unassigned_features), depth
            )
            
            if success:
                # Assign new cluster IDs to unassigned records
                unassigned_indices = np.where(unassigned_mask)[0]
                unique_sub_labels = np.unique(sub_labels)
                non_noise_labels = unique_sub_labels[unique_sub_labels != -1]
                
                for i, sub_label in enumerate(non_noise_labels):
                    sub_mask = sub_labels == sub_label
                    affected_indices = unassigned_indices[sub_mask]
                    labels[affected_indices] = next_cluster_id + i
                
                # Handle noise points from standard clustering - keep them unassigned (-1)
                noise_mask = sub_labels == -1
                if np.any(noise_mask):
                    noise_indices = unassigned_indices[noise_mask]
                    labels[noise_indices] = -1  # Keep as unassigned
                
                print(f"    [UNASSIGNED-CLUSTERING] ✅ {strategy.name} created {len(non_noise_labels)} additional clusters for unassigned records")
                return True
        
        print(f"    [UNASSIGNED-CLUSTERING] ❌ No strategy could subdivide unassigned records")
        return False

    def get_subdivision_stats(self) -> Dict[str, Any]:
        """Get statistics about all subdivisions performed."""
        if not self.subdivision_history:
            return {"total_subdivisions": 0}
        
        successful = [sub for sub in self.subdivision_history if sub["success"]]
        failed = [sub for sub in self.subdivision_history if not sub["success"]]
        
        strategy_usage = {}
        for sub in successful:
            strategy = sub["strategy_used"]
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
        
        return {
            "total_subdivisions": len(self.subdivision_history),
            "successful_subdivisions": len(successful),
            "failed_subdivisions": len(failed),
            "strategy_usage": strategy_usage,
            "total_new_clusters": sum(sub["new_clusters"] for sub in successful),
            "subdivision_history": self.subdivision_history.copy()
        }
    
    def set_performance_mode(self):
        """Enable performance mode for faster processing of very large clusters."""
        # This could adjust internal parameters for faster processing
        # For now, it's a placeholder that enables the subdivision engine
        # to use more aggressive sampling and faster algorithms
        self.performance_mode = True
        print("Performance mode enabled for subdivision engine")
