"""Hierarchical Clusterer

Advanced hierarchical clustering with PCA-based adaptive subdivision.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.cluster import DBSCAN
from .adaptive_eps import AdaptiveEpsCalculator


class HierarchicalClusterer:
    """PCA-based hierarchical clustering for breaking large clusters."""
    
    def __init__(self):
        self.eps_calculator = AdaptiveEpsCalculator()
        self.subdivision_stats = {}
    
    def hierarchical_clustering(
        self, 
        X: np.ndarray, 
        initial_labels: np.ndarray,
        base_eps: float,
        base_min_samples: int,
        max_cluster_size: int = 50,
        max_depth: int = 3,
        current_depth: int = 0,
        min_subdivision_size: int = 10  # Minimum size to attempt subdivision
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform hierarchical clustering with improved subdivision logic.
        
        Args:
            X: Feature matrix
            initial_labels: Initial cluster labels
            base_eps: Base eps value
            base_min_samples: Base min_samples value
            max_cluster_size: Maximum allowed cluster size
            max_depth: Maximum hierarchical depth
            current_depth: Current depth level
            min_subdivision_size: Minimum cluster size to attempt subdivision
            
        Returns:
            Tuple of (final_labels, subdivision_stats)
        """
        if current_depth >= max_depth:
            return initial_labels, {"max_depth_reached": True}
        
        print(f"🧠 Level {current_depth + 1} improved hierarchical clustering...")
        
        # Get unique clusters (excluding noise -1)
        unique_clusters = np.unique(initial_labels)
        unique_clusters = unique_clusters[unique_clusters != -1]
        
        if len(unique_clusters) == 0:
            return initial_labels, {"no_clusters_to_subdivide": True}
        
        # Track the next available cluster ID
        next_cluster_id = max(unique_clusters) + 1 if len(unique_clusters) > 0 else 0
        
        new_labels = initial_labels.copy()
        clusters_split = 0
        total_subdivisions = 0
        subdivision_details = []
        
        for cluster_id in unique_clusters:
            # Get points in this cluster
            cluster_mask = initial_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            # Only subdivide if cluster is large enough and too large
            if cluster_size > max_cluster_size and cluster_size >= min_subdivision_size:
                print(f"  📊 Subdividing cluster {cluster_id} (size: {cluster_size})")
                
                # Extract features for this cluster
                cluster_X = X[cluster_mask]
                
                # Calculate adaptive eps using improved algorithm
                adaptive_eps, eps_info = self.eps_calculator.calculate_adaptive_eps(
                    cluster_X, base_eps, current_depth, cluster_size, use_pca=True
                )
                
                print(f"    📏 Adaptive eps: {adaptive_eps:.4f} (base: {base_eps:.4f})")
                if eps_info["pca_used"]:
                    print(f"    🔄 PCA: {eps_info['original_shape'][1]} → {eps_info['pca_components']} dimensions")
                
                # Calculate adaptive min_samples
                adaptive_min_samples = self._calculate_adaptive_min_samples(
                    cluster_size, base_min_samples, current_depth
                )
                
                # Perform subdivision
                subdivision_result = self._perform_subdivision(
                    cluster_X, adaptive_eps, adaptive_min_samples, 
                    cluster_mask, cluster_id, new_labels, next_cluster_id
                )
                
                if subdivision_result["success"]:
                    clusters_split += 1
                    total_subdivisions += subdivision_result["new_clusters"]
                    next_cluster_id = subdivision_result["next_cluster_id"]
                    
                    subdivision_details.append({
                        "cluster_id": cluster_id,
                        "original_size": cluster_size,
                        "new_clusters": subdivision_result["new_clusters"],
                        "noise_points": subdivision_result["noise_points"],
                        "eps_used": adaptive_eps,
                        "min_samples_used": adaptive_min_samples,
                        "eps_info": eps_info
                    })
                    
                    print(f"    ✅ Split into {subdivision_result['new_clusters']} sub-clusters")
                else:
                    print(f"    ⚠️  No meaningful subdivision possible")
            
            elif cluster_size > max_cluster_size:
                print(f"  ⚠️  Cluster {cluster_id} (size: {cluster_size}) too small for subdivision (min: {min_subdivision_size})")
        
        # Store subdivision statistics
        self.subdivision_stats[current_depth + 1] = {
            "clusters_split": clusters_split,
            "total_subdivisions": total_subdivisions,
            "subdivision_details": subdivision_details
        }
        
        print(f"  📈 Split {clusters_split} large clusters into {total_subdivisions} sub-clusters at level {current_depth + 1}")
        
        # Recursive call for next level if we made significant changes
        if clusters_split > 0 and current_depth < max_depth - 1:
            return self.hierarchical_clustering(
                X, new_labels, base_eps, base_min_samples, 
                max_cluster_size, max_depth, current_depth + 1, min_subdivision_size
            )
        else:
            final_stats = {
                "total_levels": current_depth + 1,
                "subdivision_stats": self.subdivision_stats
            }
            return new_labels, final_stats
    
    def _calculate_adaptive_min_samples(
        self, 
        cluster_size: int, 
        base_min_samples: int, 
        current_depth: int
    ) -> int:
        """Calculate adaptive min_samples based on cluster characteristics."""
        
        # For large clusters, be more permissive to enable subdivision
        if cluster_size > 1000:
            adaptive_min_samples = max(2, base_min_samples - 2)
        elif cluster_size > 100:
            adaptive_min_samples = max(2, base_min_samples - 1)
        else:
            adaptive_min_samples = max(2, base_min_samples)
        
        # Reduce min_samples at deeper levels
        adaptive_min_samples = max(2, adaptive_min_samples - current_depth)
        
        return adaptive_min_samples
    
    def _perform_subdivision(
        self,
        cluster_X: np.ndarray,
        eps: float,
        min_samples: int,
        cluster_mask: np.ndarray,
        cluster_id: int,
        new_labels: np.ndarray,
        next_cluster_id: int
    ) -> Dict[str, Any]:
        """Perform the actual DBSCAN subdivision."""
        
        try:
            # Perform DBSCAN clustering
            sub_model = DBSCAN(eps=eps, min_samples=min_samples)
            sub_labels = sub_model.fit_predict(cluster_X)
            
            # Analyze results
            unique_sub_labels = np.unique(sub_labels)
            unique_sub_labels = unique_sub_labels[unique_sub_labels != -1]
            n_noise = np.sum(sub_labels == -1)
            
            # Check if subdivision was meaningful
            if len(unique_sub_labels) <= 1:
                return {
                    "success": False,
                    "reason": "no_subdivision",
                    "new_clusters": 0,
                    "noise_points": n_noise,
                    "next_cluster_id": next_cluster_id
                }
            
            # Check if subdivision created reasonably sized clusters
            sub_cluster_sizes = [np.sum(sub_labels == label) for label in unique_sub_labels]
            min_reasonable_size = 2
            reasonable_clusters = [size for size in sub_cluster_sizes if size >= min_reasonable_size]
            
            if len(reasonable_clusters) <= 1:
                return {
                    "success": False,
                    "reason": "clusters_too_small",
                    "new_clusters": len(unique_sub_labels),
                    "noise_points": n_noise,
                    "next_cluster_id": next_cluster_id
                }
            
            # Apply the subdivision
            new_clusters_created = 0
            for sub_label in unique_sub_labels:
                sub_mask = sub_labels == sub_label
                cluster_size = np.sum(sub_mask)
                
                if cluster_size >= min_reasonable_size:
                    original_indices = np.where(cluster_mask)[0][sub_mask]
                    new_labels[original_indices] = next_cluster_id
                    next_cluster_id += 1
                    new_clusters_created += 1
                else:
                    # Small clusters become noise - keep in original cluster
                    original_indices = np.where(cluster_mask)[0][sub_mask]
                    new_labels[original_indices] = cluster_id
            
            # Handle noise points from sub-clustering (keep in original cluster)
            if n_noise > 0:
                noise_mask = sub_labels == -1
                original_noise_indices = np.where(cluster_mask)[0][noise_mask]
                new_labels[original_noise_indices] = cluster_id
            
            return {
                "success": True,
                "new_clusters": new_clusters_created,
                "noise_points": n_noise,
                "next_cluster_id": next_cluster_id,
                "sub_cluster_sizes": sub_cluster_sizes
            }
            
        except Exception as e:
            return {
                "success": False,
                "reason": f"error: {str(e)}",
                "new_clusters": 0,
                "noise_points": 0,
                "next_cluster_id": next_cluster_id
            }
    
    def get_subdivision_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the hierarchical subdivision process."""
        return self.subdivision_stats.copy()
    
    def connectivity_aware_clustering(
        self,
        X: np.ndarray,
        initial_labels: np.ndarray,
        base_eps: float,
        min_samples: int,
        max_cluster_size: int,
        max_depth: int,
        similarity_threshold: float = 2.0,
        adaptive_threshold: bool = True,
        high_similarity_threshold: float = 1.0  # Always preserve these connections
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform hierarchical clustering while preserving high-similarity connections.
        
        This method ensures that records with distances below similarity_threshold
        are kept in the same cluster during subdivision. Uses adaptive thresholds
        that become more permissive at deeper levels.
        
        Args:
            X: Feature matrix
            initial_labels: Initial cluster labels  
            base_eps: Base eps value
            min_samples: Minimum samples for DBSCAN
            max_cluster_size: Maximum cluster size before subdivision
            max_depth: Maximum subdivision depth
            similarity_threshold: Maximum distance to preserve connections (default: 2.0)
            adaptive_threshold: Whether to use adaptive thresholds by depth
            high_similarity_threshold: Always preserve connections below this (default: 1.0)
        
        Returns:
            Tuple of (final_labels, hierarchy_stats)
        """
        print(f"🔗 Adaptive connectivity-aware clustering (max_depth={max_depth})")
        if adaptive_threshold:
            print(f"   🎯 High-similarity threshold: {high_similarity_threshold:.1f} (always preserved)")
            print(f"   📈 Adaptive threshold: {similarity_threshold:.1f} → more permissive at deeper levels")
        
        hierarchy_stats = {
            "levels_processed": 0,
            "clusters_subdivided": 0,
            "total_subdivisions": 0,
            "connectivity_preserved": 0,
            "similarity_threshold": similarity_threshold,
            "adaptive_threshold": adaptive_threshold,
            "high_similarity_threshold": high_similarity_threshold,
            "depth_thresholds": {}  # Track thresholds used at each depth
        }
        
        current_labels = initial_labels.copy()
        
        for depth in range(max_depth):
            print(f"🧠 Level {depth + 1} connectivity-aware clustering...")
            
            # Calculate adaptive similarity threshold for this depth
            if adaptive_threshold:
                current_threshold = self._calculate_adaptive_threshold(
                    depth, max_depth, similarity_threshold, high_similarity_threshold
                )
            else:
                current_threshold = similarity_threshold
            
            hierarchy_stats["depth_thresholds"][depth + 1] = current_threshold
            print(f"   🎯 Depth {depth + 1} threshold: {current_threshold:.3f}")
            
            clusters_to_subdivide = []
            unique_labels = np.unique(current_labels)
            
            # Find clusters that need subdivision
            for cluster_id in unique_labels:
                if cluster_id == -1:  # Skip noise
                    continue
                    
                cluster_mask = current_labels == cluster_id
                cluster_size = np.sum(cluster_mask)
                
                if cluster_size > max_cluster_size:
                    clusters_to_subdivide.append((cluster_id, cluster_mask, cluster_size))
            
            if not clusters_to_subdivide:
                print(f"  ✅ No more clusters to subdivide at level {depth + 1}")
                break
            
            subdivisions_made = 0
            
            for cluster_id, cluster_mask, cluster_size in clusters_to_subdivide:
                print(f"  📊 Subdividing cluster {cluster_id} (size: {cluster_size})")
                
                # Extract cluster data
                cluster_X = X[cluster_mask]
                cluster_indices = np.where(cluster_mask)[0]
                
                # Calculate adaptive eps for this cluster
                adaptive_eps, eps_info = self.eps_calculator.calculate_adaptive_eps(
                    cluster_X, base_eps, current_depth=0, cluster_size=cluster_size, use_pca=True
                )
                
                # Apply PCA transformation if it was used for eps calculation
                if eps_info.get("pca_used", False):
                    print(f"    🔄 PCA: {eps_info['original_shape'][1]} → {eps_info['pca_components']} dimensions")
                    # Re-apply PCA transformation for consistency
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=eps_info['pca_components'])
                    cluster_X_for_clustering = pca.fit_transform(cluster_X)
                else:
                    cluster_X_for_clustering = cluster_X
                
                print(f"    📏 Adaptive eps: {adaptive_eps:.4f} (base: {base_eps:.4f})")
                
                # Perform DBSCAN subdivision
                dbscan = DBSCAN(eps=adaptive_eps, min_samples=min_samples)
                sub_labels = dbscan.fit_predict(cluster_X_for_clustering)
                
                # Check connectivity preservation
                connectivity_violations = self._check_connectivity_violations(
                    cluster_X, sub_labels, similarity_threshold
                )
                
                if connectivity_violations > 0:
                    print(f"    ⚠️  {connectivity_violations} connectivity violations detected")
                    
                    # Try to preserve connections by adjusting eps
                    preserved_labels = self._preserve_high_similarity_connections(
                        cluster_X, sub_labels, similarity_threshold, adaptive_eps, min_samples
                    )
                    
                    if preserved_labels is not None:
                        sub_labels = preserved_labels
                        hierarchy_stats["connectivity_preserved"] += connectivity_violations
                        print(f"    ✅ Preserved {connectivity_violations} high-similarity connections")
                
                # Apply subdivision if meaningful
                n_sub_clusters = len(set(sub_labels[sub_labels != -1]))
                if n_sub_clusters > 1:
                    # Assign new cluster IDs
                    max_existing_label = np.max(current_labels) if len(current_labels) > 0 else -1
                    
                    for i, sub_label in enumerate(np.unique(sub_labels)):
                        if sub_label == -1:
                            # Keep noise as noise
                            sub_mask = sub_labels == sub_label
                            current_labels[cluster_indices[sub_mask]] = -1
                        else:
                            # Assign new cluster ID
                            new_cluster_id = max_existing_label + 1 + i
                            sub_mask = sub_labels == sub_label
                            current_labels[cluster_indices[sub_mask]] = new_cluster_id
                    
                    print(f"    ✅ Split into {n_sub_clusters} sub-clusters")
                    subdivisions_made += 1
                    hierarchy_stats["total_subdivisions"] += n_sub_clusters
                else:
                    print(f"    ⚠️  No meaningful subdivision possible")
            
            hierarchy_stats["clusters_subdivided"] += len(clusters_to_subdivide)
            hierarchy_stats["levels_processed"] += 1
            
            print(f"  📈 Split {subdivisions_made} large clusters into sub-clusters at level {depth + 1}")
            
            if subdivisions_made == 0:
                print(f"  ✅ No subdivisions made at level {depth + 1}, stopping")
                break
        
        return current_labels, hierarchy_stats
    
    def _check_connectivity_violations(
        self, 
        X: np.ndarray, 
        labels: np.ndarray, 
        similarity_threshold: float
    ) -> int:
        """Check how many high-similarity pairs are split across different clusters."""
        violations = 0
        n_points = len(X)
        
        # Sample pairs to avoid O(n²) complexity for large clusters
        max_pairs_to_check = min(1000, n_points * (n_points - 1) // 2)
        pairs_checked = 0
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if pairs_checked >= max_pairs_to_check:
                    break
                    
                # Calculate distance between points
                distance = np.linalg.norm(X[i] - X[j])
                
                # If points are very similar but in different clusters, it's a violation
                if distance <= similarity_threshold and labels[i] != labels[j] and labels[i] != -1 and labels[j] != -1:
                    violations += 1
                
                pairs_checked += 1
            
            if pairs_checked >= max_pairs_to_check:
                break
        
        return violations
    
    def _preserve_high_similarity_connections(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        similarity_threshold: float,
        current_eps: float,
        min_samples: int
    ) -> Optional[np.ndarray]:
        """Attempt to preserve high-similarity connections by adjusting clustering."""
        
        # Try increasing eps to capture more connections
        increased_eps = min(current_eps * 1.5, similarity_threshold)
        
        if increased_eps > current_eps:
            print(f"    🔧 Trying increased eps: {increased_eps:.4f}")
            
            dbscan = DBSCAN(eps=increased_eps, min_samples=min_samples)
            adjusted_labels = dbscan.fit_predict(X)
            
            # Check if this reduces connectivity violations
            new_violations = self._check_connectivity_violations(X, adjusted_labels, similarity_threshold)
            old_violations = self._check_connectivity_violations(X, labels, similarity_threshold)
            
            if new_violations < old_violations:
                return adjusted_labels
        
        return None
    
    def _calculate_adaptive_threshold(
        self,
        current_depth: int,
        max_depth: int,
        base_threshold: float,
        high_similarity_threshold: float
    ) -> float:
        """
        Calculate adaptive similarity threshold based on current depth.
        
        Strategy:
        - Depth 1-2: Use strict thresholds (close to high_similarity_threshold)
        - Depth 3+: Gradually increase threshold allowing more subdivision
        - Always preserve high-similarity connections
        
        Args:
            current_depth: Current subdivision depth (0-indexed)
            max_depth: Maximum allowed depth
            base_threshold: Base similarity threshold
            high_similarity_threshold: Always preserve threshold
        
        Returns:
            Adaptive threshold for current depth
        """
        depth_1_indexed = current_depth + 1
        
        # Always preserve high-similarity connections
        if depth_1_indexed <= 2:
            # Early depths: Be very strict, only allow subdivision if we can preserve tight connections
            threshold = high_similarity_threshold + (base_threshold - high_similarity_threshold) * 0.3
        elif depth_1_indexed <= 4:
            # Mid depths: Moderate threshold
            threshold = high_similarity_threshold + (base_threshold - high_similarity_threshold) * 0.6
        else:
            # Deep levels: Allow more subdivision but still preserve high-similarity
            threshold = base_threshold
        
        # Ensure we never go below high_similarity_threshold for preservation
        return max(threshold, high_similarity_threshold)
