"""Subdivision Engine V2

Clean, modular subdivision engine with focused strategies.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ..adaptive_eps import AdaptiveEpsCalculator


class SubdivisionStrategy:
    """Base class for subdivision strategies."""
    
    def __init__(self, name: str):
        self.name = name
        
    def can_handle(self, cluster_size: int) -> bool:
        """Check if this strategy can handle the given cluster size."""
        raise NotImplementedError
        
    def subdivide(
        self, 
        cluster_X: np.ndarray, 
        base_eps: float, 
        min_samples: int
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """
        Attempt subdivision using this strategy.
        
        Returns:
            Tuple of (success, labels, info)
        """
        raise NotImplementedError


class AdaptiveDBSCANStrategy(SubdivisionStrategy):
    """Smart DBSCAN with cluster-specific PCA."""
    
    def __init__(self):
        super().__init__("adaptive_dbscan")
        self.eps_calculator = AdaptiveEpsCalculator()
    
    def can_handle(self, cluster_size: int) -> bool:
        return cluster_size >= 50  # Handle medium to large clusters
    
    def subdivide(
        self, 
        cluster_X: np.ndarray, 
        base_eps: float, 
        min_samples: int
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        
        cluster_size = cluster_X.shape[0]
        info = {
            "strategy": self.name,
            "cluster_size": cluster_size,
            "pca_used": False,
            "eps_used": base_eps
        }
        
        try:
            # Use adaptive eps calculator for cluster-specific optimization
            adaptive_eps, eps_info = self.eps_calculator.calculate_adaptive_eps(
                cluster_X, base_eps, current_depth=0, cluster_size=cluster_size, use_pca=True
            )
            
            info.update(eps_info)
            info["eps_used"] = adaptive_eps
            
            # Apply PCA transformation if beneficial
            if eps_info.get("pca_used", False):
                print(f"    🔄 Cluster-specific PCA: {eps_info['original_shape'][1]} → {eps_info['pca_components']} dims")
                scaler = StandardScaler()
                cluster_X_scaled = scaler.fit_transform(cluster_X)
                pca = PCA(n_components=eps_info['pca_components'])
                cluster_X_transformed = pca.fit_transform(cluster_X_scaled)
                info["pca_transform"] = (scaler, pca)
            else:
                cluster_X_transformed = cluster_X
            
            # Try progressive eps values
            eps_values = [
                adaptive_eps,
                adaptive_eps * 0.8,
                adaptive_eps * 0.6,
                base_eps * 0.3
            ]
            
            for eps in eps_values:
                dbscan = DBSCAN(eps=eps, min_samples=max(2, min_samples - 1))
                labels = dbscan.fit_predict(cluster_X_transformed)
                
                unique_labels = np.unique(labels)
                n_clusters = len(unique_labels[unique_labels != -1])
                
                if n_clusters >= 2:
                    info["eps_used"] = eps
                    info["n_clusters"] = n_clusters
                    info["noise_points"] = np.sum(labels == -1)
                    return True, labels, info
            
            return False, np.zeros(cluster_size), info
            
        except Exception as e:
            info["error"] = str(e)
            return False, np.zeros(cluster_size), info


class AggressivePCAStrategy(SubdivisionStrategy):
    """Forced PCA reduction with aggressive DBSCAN."""
    
    def __init__(self):
        super().__init__("aggressive_pca")
    
    def can_handle(self, cluster_size: int) -> bool:
        return cluster_size >= 500  # Lower threshold for very large clusters
    
    def subdivide(
        self, 
        cluster_X: np.ndarray, 
        base_eps: float, 
        min_samples: int
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        
        cluster_size = cluster_X.shape[0]
        info = {
            "strategy": self.name,
            "cluster_size": cluster_size,
            "pca_used": True,
            "eps_used": base_eps  # Will be updated when successful
        }
        
        try:
            # Force aggressive PCA reduction
            scaler = StandardScaler()
            cluster_X_scaled = scaler.fit_transform(cluster_X)
            
            n_components = min(3, cluster_X.shape[1], cluster_X.shape[0] - 1)
            pca = PCA(n_components=n_components)
            cluster_X_pca = pca.fit_transform(cluster_X_scaled)
            
            variance_explained = pca.explained_variance_ratio_.sum()
            print(f"    🔄 Aggressive PCA: {cluster_X.shape[1]} → {n_components} dims, variance: {variance_explained:.3f}")
            
            info.update({
                "pca_components": n_components,
                "variance_explained": variance_explained,
                "pca_transform": (scaler, pca)
            })
            
            # Try very aggressive eps values
            eps_values = [base_eps * 0.05, base_eps * 0.03, base_eps * 0.01]
            
            for eps in eps_values:
                dbscan = DBSCAN(eps=eps, min_samples=2)
                labels = dbscan.fit_predict(cluster_X_pca)
                
                unique_labels = np.unique(labels)
                n_clusters = len(unique_labels[unique_labels != -1])
                
                if n_clusters >= 2:
                    info["eps_used"] = eps
                    info["n_clusters"] = n_clusters
                    info["noise_points"] = np.sum(labels == -1)
                    return True, labels, info
            
            return False, np.zeros(cluster_size), info
            
        except Exception as e:
            info["error"] = str(e)
            return False, np.zeros(cluster_size), info


class KMeansStrategy(SubdivisionStrategy):
    """Optimized K-means with sampling for very large clusters."""
    
    def __init__(self):
        super().__init__("kmeans")
    
    def can_handle(self, cluster_size: int) -> bool:
        return cluster_size >= 200  # Lower threshold for earlier intervention
    
    def subdivide(
        self, 
        cluster_X: np.ndarray, 
        base_eps: float, 
        min_samples: int
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        
        cluster_size = cluster_X.shape[0]
        
        # More aggressive subdivision for very large clusters
        if cluster_size > 10000:
            target_size = 100  # Very aggressive for massive clusters
            max_clusters = 50   # Allow more subdivisions
        elif cluster_size > 5000:
            target_size = 150  # Aggressive for large clusters
            max_clusters = 30
        else:
            target_size = 200  # Normal target
            max_clusters = 15
            
        n_clusters = max(2, min(max_clusters, cluster_size // target_size))
        
        info = {
            "strategy": self.name,
            "cluster_size": cluster_size,
            "n_clusters": n_clusters,
            "pca_used": True,
            "sampled": False,
            "eps_used": 0.0  # K-means doesn't use eps
        }
        
        try:
            # Sample if too large
            if cluster_size > 5000:
                sample_size = 3000
                sample_indices = np.random.choice(cluster_size, sample_size, replace=False)
                cluster_X_sample = cluster_X[sample_indices]
                info["sampled"] = True
                info["sample_size"] = sample_size
            else:
                cluster_X_sample = cluster_X
            
            # Apply PCA for efficiency
            scaler = StandardScaler()
            cluster_X_scaled = scaler.fit_transform(cluster_X_sample)
            
            n_components = min(5, cluster_X_sample.shape[1])
            pca = PCA(n_components=n_components)
            cluster_X_pca = pca.fit_transform(cluster_X_scaled)
            
            info.update({
                "pca_components": n_components,
                "variance_explained": pca.explained_variance_ratio_.sum(),
                "pca_transform": (scaler, pca)
            })
            
            # K-means clustering
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=3,
                max_iter=100
            )
            
            if info["sampled"]:
                # Fit on sample, predict on all
                sample_labels = kmeans.fit_predict(cluster_X_pca)
                cluster_X_all_scaled = scaler.transform(cluster_X)
                cluster_X_all_pca = pca.transform(cluster_X_all_scaled)
                labels = kmeans.predict(cluster_X_all_pca)
            else:
                labels = kmeans.fit_predict(cluster_X_pca)
            
            # K-means always succeeds
            cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
            info["cluster_sizes"] = cluster_sizes
            
            return True, labels, info
            
        except Exception as e:
            info["error"] = str(e)
            return False, np.zeros(cluster_size), info


class ForceStrategy(SubdivisionStrategy):
    """Last resort: intelligent random partitioning."""
    
    def __init__(self):
        super().__init__("force")
    
    def can_handle(self, cluster_size: int) -> bool:
        return cluster_size >= 50  # Can handle any reasonable size
    
    def subdivide(
        self, 
        cluster_X: np.ndarray, 
        base_eps: float, 
        min_samples: int
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        
        cluster_size = cluster_X.shape[0]
        target_size = 50
        n_partitions = max(2, cluster_size // target_size)
        
        info = {
            "strategy": self.name,
            "cluster_size": cluster_size,
            "n_partitions": n_partitions,
            "method": "random",
            "eps_used": 0.0  # Force strategy doesn't use eps
        }
        
        try:
            # Try PCA + K-means first
            scaler = StandardScaler()
            cluster_X_scaled = scaler.fit_transform(cluster_X)
            
            n_components = min(3, cluster_X.shape[1], cluster_X.shape[0] - 1)
            pca = PCA(n_components=n_components)
            cluster_X_pca = pca.fit_transform(cluster_X_scaled)
            
            kmeans = KMeans(n_clusters=n_partitions, random_state=42, n_init=3, max_iter=50)
            labels = kmeans.fit_predict(cluster_X_pca)
            
            info["method"] = "pca_kmeans"
            info["pca_components"] = n_components
            
        except Exception:
            # Fallback to random
            np.random.seed(42)
            labels = np.random.randint(0, n_partitions, cluster_size)
            info["method"] = "random"
        
        partition_sizes = [np.sum(labels == i) for i in np.unique(labels)]
        info["partition_sizes"] = partition_sizes
        
        return True, labels, info


class SubdivisionEngineV2:
    """Clean, modular subdivision engine."""
    
    def __init__(self):
        self.strategies = [
            AdaptiveDBSCANStrategy(),
            AggressivePCAStrategy(),
            KMeansStrategy(),
            ForceStrategy()  # Always succeeds
        ]
        self.subdivision_history = []
    
    def perform_subdivision(
        self,
        cluster_X: np.ndarray,
        cluster_mask: np.ndarray,
        cluster_id: int,
        current_labels: np.ndarray,
        base_eps: float,
        min_samples: int,
        cluster_size: int,
        use_pca: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform subdivision using the best available strategy.
        """
        subdivision_info = {
            "success": False,
            "original_cluster_id": cluster_id,
            "original_size": cluster_size,
            "new_clusters": 0,
            "strategy_used": None,
            "eps_used": base_eps,    # Default eps value
            "pca_used": False,       # Default PCA usage
            "attempts": []
        }
        
        print(f"    🔧 Subdividing cluster {cluster_id} (size: {cluster_size})")
        
        # Try each strategy in order
        for strategy in self.strategies:
            if not strategy.can_handle(cluster_size):
                continue
                
            print(f"    🎯 Trying {strategy.name} strategy...")
            
            success, sub_labels, strategy_info = strategy.subdivide(
                cluster_X, base_eps, min_samples
            )
            
            subdivision_info["attempts"].append(strategy_info)
            
            if success:
                print(f"    ✅ {strategy.name} successful: {cluster_size} → {strategy_info.get('n_clusters', len(np.unique(sub_labels)))} clusters")
                
                # Apply subdivision to global labels
                updated_labels = self._apply_subdivision_to_labels(
                    current_labels, cluster_mask, sub_labels, cluster_id
                )
                
                subdivision_info.update({
                    "success": True,
                    "strategy_used": strategy.name,
                    "new_clusters": len(np.unique(sub_labels)),
                    "eps_used": strategy_info.get("eps_used", base_eps),       # Include eps_used for compatibility
                    "pca_used": strategy_info.get("pca_used", False),         # Include pca_used for compatibility
                    "pca_components": strategy_info.get("pca_components", 0), # Include pca_components for compatibility
                    "eps_info": strategy_info.copy(),                         # Include eps_info for compatibility
                    "strategy_info": strategy_info
                })
                
                self.subdivision_history.append(subdivision_info.copy())
                return updated_labels, subdivision_info
            else:
                print(f"    ❌ {strategy.name} failed: {strategy_info.get('error', 'no subdivision')}")
        
        # This should never happen since ForceStrategy always succeeds
        print(f"    💀 All strategies failed!")
        self.subdivision_history.append(subdivision_info.copy())
        return current_labels, subdivision_info
    
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
