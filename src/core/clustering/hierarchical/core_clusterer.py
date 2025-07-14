"""Core Hierarchical Clusterer

Main orchestrator for hierarchical clustering with modular components.
"""

from typing import Tuple, Dict, Any
import numpy as np
from .adaptive_threshold import AdaptiveThresholdCalculator
from .connectivity_manager import ConnectivityManager
from .subdivision_engine_v2 import SubdivisionEngineV2


class HierarchicalClusterer:
    """
    Core hierarchical clustering orchestrator using modular components.
    
    This class coordinates the adaptive threshold calculation, connectivity preservation,
    and subdivision engine to perform sophisticated hierarchical clustering.
    """
    
    def __init__(self):
        self.threshold_calculator = AdaptiveThresholdCalculator()
        self.connectivity_manager = ConnectivityManager()
        self.subdivision_engine = SubdivisionEngineV2()
        self.clustering_stats = {}
    
    def adaptive_hierarchical_clustering(
        self,
        X: np.ndarray,
        initial_labels: np.ndarray,
        base_eps: float,
        min_samples: int,
        max_cluster_size: int,
        max_depth: int,
        base_similarity_threshold: float = 2.0,
        high_similarity_threshold: float = 1.0,
        use_adaptive_thresholds: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform adaptive hierarchical clustering with depth-aware similarity preservation.
        
        Args:
            X: Feature matrix
            initial_labels: Initial cluster labels
            base_eps: Base eps value for DBSCAN
            min_samples: Minimum samples for DBSCAN
            max_cluster_size: Maximum cluster size before subdivision
            max_depth: Maximum subdivision depth (can be 4-6 with adaptive thresholds)
            base_similarity_threshold: Base threshold for similarity preservation
            high_similarity_threshold: Always preserve connections below this
            use_adaptive_thresholds: Whether to use depth-adaptive thresholds
        
        Returns:
            Tuple of (final_labels, comprehensive_stats)
        """
        print(f"🔗 Adaptive hierarchical clustering (max_depth={max_depth})")
        if use_adaptive_thresholds:
            print(f"   🎯 High-similarity threshold: {high_similarity_threshold:.1f} (always preserved)")
            print(f"   📈 Adaptive threshold: {base_similarity_threshold:.1f} → depth-aware progression")
            
            # Show threshold progression
            threshold_progression = self.threshold_calculator.get_threshold_progression(
                max_depth, base_similarity_threshold, high_similarity_threshold
            )
            print(f"   📊 Threshold progression: {', '.join([f'D{d}:{t:.2f}' for d, t in threshold_progression.items()])}")
        
        # Initialize comprehensive stats
        comprehensive_stats = {
            "max_depth": max_depth,
            "levels_processed": 0,
            "total_subdivisions": 0,
            "connectivity_preserved_total": 0,
            "use_adaptive_thresholds": use_adaptive_thresholds,
            "base_similarity_threshold": base_similarity_threshold,
            "high_similarity_threshold": high_similarity_threshold,
            "depth_stats": {},
            "threshold_progression": threshold_progression if use_adaptive_thresholds else {}
        }
        
        current_labels = initial_labels.copy()
        
        # Main hierarchical clustering loop
        for depth in range(max_depth):
            print(f"🧠 Level {depth + 1} adaptive hierarchical clustering...")
            
            # Calculate adaptive threshold for this depth
            if use_adaptive_thresholds:
                current_threshold = self.threshold_calculator.calculate_adaptive_threshold(
                    depth, max_depth, base_similarity_threshold, high_similarity_threshold
                )
            else:
                current_threshold = base_similarity_threshold
            
            print(f"   🎯 Depth {depth + 1} similarity threshold: {current_threshold:.3f}")
            
            # Find clusters that need subdivision
            clusters_to_subdivide = self._find_clusters_to_subdivide(
                current_labels, max_cluster_size
            )
            
            if not clusters_to_subdivide:
                print(f"  ✅ No more clusters to subdivide at level {depth + 1}")
                break
            
            # Process each cluster for subdivision
            depth_stats = self._process_clusters_for_subdivision(
                X, current_labels, clusters_to_subdivide, base_eps, min_samples,
                current_threshold, high_similarity_threshold, depth + 1
            )
            
            comprehensive_stats["depth_stats"][depth + 1] = depth_stats
            comprehensive_stats["levels_processed"] += 1
            comprehensive_stats["total_subdivisions"] += depth_stats["subdivisions_made"]
            comprehensive_stats["connectivity_preserved_total"] += depth_stats["connections_preserved"]
            
            # Update current_labels with results
            current_labels = depth_stats["final_labels"]
            
            print(f"  📈 Level {depth + 1}: {depth_stats['subdivisions_made']} subdivisions, "
                  f"{depth_stats['connections_preserved']} connections preserved")
            
            # Stop early if no meaningful progress
            if depth_stats["subdivisions_made"] == 0:
                print(f"  ✅ No subdivisions made at level {depth + 1}, stopping early")
                break
        
        # Collect final statistics
        comprehensive_stats.update({
            "subdivision_stats": self.subdivision_engine.get_subdivision_stats(),
            "connectivity_stats": self.connectivity_manager.get_connectivity_stats(),
            "threshold_stats": self.threshold_calculator.get_threshold_stats()
        })
        
        return current_labels, comprehensive_stats
    
    def _find_clusters_to_subdivide(self, labels: np.ndarray, max_cluster_size: int) -> list:
        """Find clusters that exceed the maximum size and need subdivision."""
        clusters_to_subdivide = []
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise
                continue
                
            cluster_mask = labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size > max_cluster_size:
                clusters_to_subdivide.append((cluster_id, cluster_mask, cluster_size))
        
        return clusters_to_subdivide
    
    def _process_clusters_for_subdivision(
        self,
        X: np.ndarray,
        current_labels: np.ndarray,
        clusters_to_subdivide: list,
        base_eps: float,
        min_samples: int,
        similarity_threshold: float,
        high_similarity_threshold: float,
        depth: int
    ) -> Dict[str, Any]:
        """Process all clusters for subdivision at a given depth."""
        
        working_labels = current_labels.copy()
        subdivisions_made = 0
        connections_preserved = 0
        subdivision_details = []
        
        for cluster_id, cluster_mask, cluster_size in clusters_to_subdivide:
            print(f"  📊 Subdividing cluster {cluster_id} (size: {cluster_size})")
            
            # Extract cluster data
            cluster_X = X[cluster_mask]
            
            # Perform subdivision
            updated_labels, subdivision_info = self.subdivision_engine.perform_subdivision(
                cluster_X, cluster_mask, cluster_id, working_labels,
                base_eps, min_samples, cluster_size
            )
            
            if subdivision_info["success"]:
                # Check and preserve connectivity
                violations = self.connectivity_manager.check_connectivity_violations(
                    cluster_X, updated_labels[cluster_mask], similarity_threshold
                )
                
                if violations > 0:
                    print(f"    ⚠️  {violations} connectivity violations detected")
                    
                    # Try standard preservation
                    preserved_labels = self.connectivity_manager.preserve_high_similarity_connections(
                        cluster_X, updated_labels[cluster_mask], similarity_threshold,
                        subdivision_info["eps_used"], min_samples
                    )
                    
                    if preserved_labels is not None:
                        # Apply preserved labels to the working set
                        cluster_indices = np.where(cluster_mask)[0]
                        working_labels[cluster_indices] = preserved_labels
                        connections_preserved += violations
                        print(f"    ✅ Preserved {violations} connections via eps adjustment")
                    else:
                        # Force preservation for high-similarity connections
                        force_preserved = self.connectivity_manager.force_preserve_connections(
                            cluster_X, updated_labels[cluster_mask], high_similarity_threshold
                        )
                        cluster_indices = np.where(cluster_mask)[0]
                        working_labels[cluster_indices] = force_preserved
                        
                        # Count forced preservations
                        high_sim_violations = self.connectivity_manager.check_connectivity_violations(
                            cluster_X, updated_labels[cluster_mask], high_similarity_threshold
                        )
                        connections_preserved += high_sim_violations
                        print(f"    🔧 Force-preserved {high_sim_violations} high-similarity connections")
                else:
                    # No violations, use subdivision results
                    working_labels = updated_labels
                
                subdivisions_made += 1
                subdivision_details.append(subdivision_info)
                
                print(f"    ✅ Split into {subdivision_info['new_clusters']} sub-clusters")
                if subdivision_info["pca_used"]:
                    print(f"    🔄 PCA: {subdivision_info['eps_info']['original_shape'][1]} → {subdivision_info['pca_components']} dimensions")
            else:
                print(f"    ⚠️  Subdivision failed: {subdivision_info.get('failure_reason', 'unknown')}")
        
        return {
            "subdivisions_made": subdivisions_made,
            "connections_preserved": connections_preserved,
            "subdivision_details": subdivision_details,
            "final_labels": working_labels,
            "clusters_processed": len(clusters_to_subdivide)
        }
    
    # Legacy method for backward compatibility
    def connectivity_aware_clustering(
        self,
        X: np.ndarray,
        initial_labels: np.ndarray,
        base_eps: float,
        min_samples: int,
        max_cluster_size: int,
        max_depth: int,
        similarity_threshold: float = 2.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Legacy method - redirects to adaptive_hierarchical_clustering."""
        return self.adaptive_hierarchical_clustering(
            X, initial_labels, base_eps, min_samples, max_cluster_size, max_depth,
            base_similarity_threshold=similarity_threshold,
            high_similarity_threshold=1.0,
            use_adaptive_thresholds=True
        )
