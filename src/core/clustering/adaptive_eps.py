"""Adaptive EPS Calculator

Calculates optimal eps values for hierarchical clustering based on data characteristics.
"""

from typing import Tuple, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class AdaptiveEpsCalculator:
    """Calculates adaptive eps values for DBSCAN clustering."""
    
    def __init__(self):
        self.last_calculation = {}
    
    def calculate_adaptive_eps(
        self,
        X: np.ndarray,
        base_eps: float,
        current_depth: int = 0,
        cluster_size: Optional[int] = None,
        use_pca: bool = True,
        target_cluster_ratio: float = 0.1
    ) -> Tuple[float, dict]:
        """
        Calculate adaptive eps for hierarchical clustering.
        
        Args:
            X: Feature matrix for the cluster
            base_eps: Original eps value
            current_depth: Current hierarchical depth
            cluster_size: Size of the cluster being subdivided
            use_pca: Whether to use PCA transformation
            target_cluster_ratio: Target ratio of points in clusters vs noise
            
        Returns:
            Tuple of (adaptive_eps, calculation_info)
        """
        calculation_info = {
            "method": "adaptive_hierarchy",
            "original_shape": X.shape,
            "pca_used": False,
            "pca_components": 0,
            "variance_explained": 0.0,
            "nn_distance_median": 0.0,
            "depth_factor": 1.0,
            "size_factor": 1.0,
            "final_eps": base_eps
        }
        
        try:
            # Handle edge cases
            if X.shape[0] < 4:
                return base_eps * 0.5, calculation_info
            
            # Check for zero variance (identical features)
            feature_variances = np.var(X, axis=0)
            has_variance = np.any(feature_variances > 1e-12)
            
            if not has_variance:
                print(f"    ⚠️  All features identical - using conservative eps")
                adaptive_eps = base_eps * (0.8 - current_depth * 0.1)
                calculation_info["final_eps"] = max(0.01, adaptive_eps)
                return calculation_info["final_eps"], calculation_info
            
            # Apply PCA if requested and beneficial
            X_transformed = X
            if use_pca and X.shape[1] > 2:
                X_transformed, pca_info = self._apply_pca_transformation(X)
                calculation_info.update(pca_info)
            
            # Calculate nearest neighbor distances
            nn_distance = self._calculate_nearest_neighbor_distance(
                X_transformed, target_cluster_ratio
            )
            calculation_info["nn_distance_median"] = nn_distance
            
            # Apply hierarchical depth factor (less aggressive subdivision)
            depth_factor = max(0.4, 0.9 - current_depth * 0.15)  # More conservative
            calculation_info["depth_factor"] = depth_factor
            
            # Apply cluster size factor (larger clusters need more aggressive subdivision)
            if cluster_size:
                size_factor = min(1.5, 1.0 + np.log10(cluster_size / 1000))
                calculation_info["size_factor"] = size_factor
            else:
                size_factor = 1.0
            
            # Calculate final adaptive eps
            adaptive_eps = nn_distance * depth_factor * size_factor
            
            # Ensure reasonable bounds
            min_eps = base_eps * 0.1
            max_eps = base_eps * 2.0
            adaptive_eps = np.clip(adaptive_eps, min_eps, max_eps)
            
            calculation_info["final_eps"] = adaptive_eps
            self.last_calculation = calculation_info.copy()
            
            return adaptive_eps, calculation_info
            
        except Exception as e:
            print(f"    ❌ Adaptive eps calculation failed: {str(e)}")
            fallback_eps = base_eps * (0.7 - current_depth * 0.1)
            calculation_info["final_eps"] = max(0.01, fallback_eps)
            calculation_info["error"] = str(e)
            return calculation_info["final_eps"], calculation_info
    
    def _apply_pca_transformation(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Apply PCA transformation to reduce dimensionality."""
        pca_info = {
            "pca_used": True,
            "pca_components": 0,
            "variance_explained": 0.0
        }
        
        try:
            # Standardize first
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine optimal number of components
            max_components = min(X_scaled.shape[0] - 1, X_scaled.shape[1])
            
            if max_components > 1:
                # Use PCA to explain 90% variance (more conservative)
                pca = PCA(n_components=min(max_components, 0.90))
                X_pca = pca.fit_transform(X_scaled)
                
                pca_info["pca_components"] = pca.n_components_
                pca_info["variance_explained"] = pca.explained_variance_ratio_.sum()
                
                return X_pca, pca_info
            else:
                # Not enough samples for meaningful PCA
                pca_info["pca_used"] = False
                return X_scaled, pca_info
                
        except Exception as e:
            print(f"    ⚠️  PCA transformation failed: {str(e)}")
            pca_info["pca_used"] = False
            pca_info["error"] = str(e)
            return X, pca_info
    
    def _calculate_nearest_neighbor_distance(
        self, 
        X: np.ndarray, 
        target_cluster_ratio: float = 0.1
    ) -> float:
        """Calculate optimal eps based on nearest neighbor distances."""
        
        try:
            # Use k-th nearest neighbor where k depends on target cluster ratio
            n_samples = X.shape[0]
            k = max(2, min(10, int(n_samples * target_cluster_ratio)))
            
            if n_samples <= k:
                k = max(2, n_samples - 1)
            
            # Calculate k-th nearest neighbor distances
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(X)
            distances, _ = nn.kneighbors(X)
            
            # Use the k-th distance (last column)
            kth_distances = distances[:, -1]
            
            # Use a percentile approach for more stable results
            # Use 60th percentile instead of median for slightly more aggressive clustering
            eps_estimate = np.percentile(kth_distances, 60)
            
            return float(eps_estimate)
            
        except Exception as e:
            print(f"    ⚠️  NN distance calculation failed: {str(e)}")
            return 0.1  # Fallback value
    
    def get_last_calculation_info(self) -> dict:
        """Get information about the last eps calculation."""
        return self.last_calculation.copy()
