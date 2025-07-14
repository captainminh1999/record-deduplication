"""Connectivity Manager

Manages and preserves high-similarity connections during hierarchical clustering.
"""

from typing import Optional
import numpy as np
from sklearn.cluster import DBSCAN


class ConnectivityManager:
    """Manage connectivity preservation during hierarchical subdivision."""
    
    def __init__(self):
        self.violation_stats = {}
        self.preservation_stats = {}
    
    def check_connectivity_violations(
        self, 
        X: np.ndarray, 
        labels: np.ndarray, 
        similarity_threshold: float,
        max_pairs_to_check: int = 1000
    ) -> int:
        """
        Check how many high-similarity pairs are split across different clusters.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            similarity_threshold: Distance threshold for high-similarity
            max_pairs_to_check: Maximum pairs to check (for performance)
        
        Returns:
            Number of connectivity violations found
        """
        violations = 0
        n_points = len(X)
        
        # Sample pairs to avoid O(n²) complexity for large clusters
        max_pairs = min(max_pairs_to_check, n_points * (n_points - 1) // 2)
        pairs_checked = 0
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if pairs_checked >= max_pairs:
                    break
                    
                # Calculate distance between points
                distance = np.linalg.norm(X[i] - X[j])
                
                # If points are very similar but in different clusters, it's a violation
                if (distance <= similarity_threshold and 
                    labels[i] != labels[j] and 
                    labels[i] != -1 and 
                    labels[j] != -1):
                    violations += 1
                
                pairs_checked += 1
            
            if pairs_checked >= max_pairs:
                break
        
        return violations
    
    def preserve_high_similarity_connections(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        similarity_threshold: float,
        current_eps: float,
        min_samples: int,
        max_eps_increase: float = 1.5
    ) -> Optional[np.ndarray]:
        """
        Attempt to preserve high-similarity connections by adjusting clustering.
        
        Args:
            X: Feature matrix
            labels: Current cluster labels
            similarity_threshold: Distance threshold for preservation
            current_eps: Current eps value
            min_samples: Minimum samples for DBSCAN
            max_eps_increase: Maximum factor to increase eps
        
        Returns:
            Adjusted labels if improvement found, None otherwise
        """
        # Try increasing eps to capture more connections
        increased_eps = min(current_eps * max_eps_increase, similarity_threshold)
        
        if increased_eps > current_eps:
            # Re-cluster with increased eps
            dbscan = DBSCAN(eps=increased_eps, min_samples=min_samples)
            adjusted_labels = dbscan.fit_predict(X)
            
            # Check if this reduces connectivity violations
            new_violations = self.check_connectivity_violations(X, adjusted_labels, similarity_threshold)
            old_violations = self.check_connectivity_violations(X, labels, similarity_threshold)
            
            if new_violations < old_violations:
                self.preservation_stats[len(self.preservation_stats)] = {
                    "original_eps": current_eps,
                    "adjusted_eps": increased_eps,
                    "violations_reduced": old_violations - new_violations,
                    "old_violations": old_violations,
                    "new_violations": new_violations
                }
                return adjusted_labels
        
        return None
    
    def force_preserve_connections(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        high_similarity_threshold: float
    ) -> np.ndarray:
        """
        Force preservation of the highest-similarity connections by merging clusters.
        
        This is a more aggressive approach that directly merges clusters containing
        very similar points.
        
        Args:
            X: Feature matrix
            labels: Current cluster labels
            high_similarity_threshold: Threshold for forced preservation
        
        Returns:
            Labels with forced connection preservation
        """
        preserved_labels = labels.copy()
        n_points = len(X)
        
        # Find all high-similarity pairs that are in different clusters
        merges_needed = []
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                distance = np.linalg.norm(X[i] - X[j])
                
                if (distance <= high_similarity_threshold and
                    preserved_labels[i] != preserved_labels[j] and
                    preserved_labels[i] != -1 and preserved_labels[j] != -1):
                    
                    merges_needed.append((i, j, preserved_labels[i], preserved_labels[j], distance))
        
        # Perform cluster merges
        merges_performed = 0
        for i, j, cluster_i, cluster_j, distance in merges_needed:
            if preserved_labels[i] != preserved_labels[j]:  # Still need to merge
                # Merge smaller cluster into larger cluster
                cluster_i_size = np.sum(preserved_labels == cluster_i)
                cluster_j_size = np.sum(preserved_labels == cluster_j)
                
                if cluster_i_size >= cluster_j_size:
                    # Merge j's cluster into i's cluster
                    preserved_labels[preserved_labels == cluster_j] = cluster_i
                else:
                    # Merge i's cluster into j's cluster
                    preserved_labels[preserved_labels == cluster_i] = cluster_j
                
                merges_performed += 1
        
        if merges_performed > 0:
            self.preservation_stats[f"forced_{len(self.preservation_stats)}"] = {
                "merges_performed": merges_performed,
                "high_similarity_threshold": high_similarity_threshold,
                "total_high_similarity_pairs": len(merges_needed)
            }
        
        return preserved_labels
    
    def get_connectivity_stats(self) -> dict:
        """Get statistics about connectivity preservation efforts."""
        return {
            "violation_checks": len(self.violation_stats),
            "preservation_attempts": len(self.preservation_stats),
            "preservation_history": self.preservation_stats.copy()
        }
