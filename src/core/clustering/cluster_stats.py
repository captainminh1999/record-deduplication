"""Cluster Statistics Calculator

Calculates comprehensive clustering statistics and quality metrics.
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN


class ClusterStatsCalculator:
    """Calculates and tracks clustering statistics."""
    
    def __init__(self):
        self.last_stats = {}
    
    def calculate_cluster_stats(
        self, 
        X: np.ndarray, 
        labels: np.ndarray, 
        eps: float, 
        min_samples: int,
        include_silhouette: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive clustering statistics.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            eps: DBSCAN eps parameter
            min_samples: DBSCAN min_samples parameter
            include_silhouette: Whether to calculate silhouette score
            
        Returns:
            Dictionary of clustering statistics
        """
        # Basic cluster statistics
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        n_noise = np.sum(labels == -1)
        n_total = len(labels)
        
        # Cluster size statistics
        cluster_sizes = []
        if n_clusters > 0:
            for cluster_id in unique_labels[unique_labels != -1]:
                cluster_size = np.sum(labels == cluster_id)
                cluster_sizes.append(cluster_size)
        
        # Calculate silhouette score if possible and requested
        silhouette = None
        if include_silhouette and n_clusters > 1 and n_total - n_noise > 1:
            try:
                # Only calculate silhouette for non-noise points
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 1 and len(np.unique(labels[non_noise_mask])) > 1:
                    silhouette = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
                else:
                    silhouette = 0.0
            except Exception as e:
                print(f"    ⚠️  Silhouette calculation failed: {str(e)}")
                silhouette = 0.0
        elif n_clusters <= 1:
            silhouette = 0.0
        
        # Cluster quality metrics
        quality_metrics = self._calculate_quality_metrics(cluster_sizes, n_total, n_noise)
        
        # Distribution analysis
        size_distribution = self._analyze_size_distribution(cluster_sizes)
        
        stats = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "n_total": n_total,
            "noise_ratio": n_noise / n_total if n_total > 0 else 0.0,
            "silhouette_score": silhouette,
            "cluster_sizes": cluster_sizes,
            "quality_metrics": quality_metrics,
            "size_distribution": size_distribution,
            "parameters": {
                "eps": eps,
                "min_samples": min_samples
            }
        }
        
        self.last_stats = stats.copy()
        return stats
    
    def _calculate_quality_metrics(
        self, 
        cluster_sizes: List[int], 
        n_total: int, 
        n_noise: int
    ) -> Dict[str, Any]:
        """Calculate cluster quality metrics."""
        
        if not cluster_sizes:
            return {
                "avg_cluster_size": 0.0,
                "largest_cluster": 0,
                "smallest_cluster": 0,
                "size_variance": 0.0,
                "balance_score": 0.0,
                "coverage_ratio": 0.0
            }
        
        cluster_sizes_array = np.array(cluster_sizes)
        
        # Basic size metrics
        avg_size = np.mean(cluster_sizes_array)
        largest = np.max(cluster_sizes_array)
        smallest = np.min(cluster_sizes_array)
        size_variance = np.var(cluster_sizes_array)
        
        # Balance score (lower variance relative to mean is better)
        balance_score = 1.0 / (1.0 + size_variance / (avg_size + 1e-8))
        
        # Coverage ratio (fraction of data in clusters vs noise)
        clustered_points = np.sum(cluster_sizes_array)
        coverage_ratio = clustered_points / n_total if n_total > 0 else 0.0
        
        return {
            "avg_cluster_size": float(avg_size),
            "largest_cluster": int(largest),
            "smallest_cluster": int(smallest),
            "size_variance": float(size_variance),
            "balance_score": float(balance_score),
            "coverage_ratio": float(coverage_ratio)
        }
    
    def _analyze_size_distribution(self, cluster_sizes: List[int]) -> Dict[str, Any]:
        """Analyze the distribution of cluster sizes."""
        
        if not cluster_sizes:
            return {
                "size_ranges": {},
                "percentiles": {},
                "distribution_type": "empty"
            }
        
        cluster_sizes_array = np.array(cluster_sizes)
        
        # Define size ranges
        size_ranges = {
            "tiny (2-5)": np.sum((cluster_sizes_array >= 2) & (cluster_sizes_array <= 5)),
            "small (6-20)": np.sum((cluster_sizes_array >= 6) & (cluster_sizes_array <= 20)),
            "medium (21-100)": np.sum((cluster_sizes_array >= 21) & (cluster_sizes_array <= 100)),
            "large (101-1000)": np.sum((cluster_sizes_array >= 101) & (cluster_sizes_array <= 1000)),
            "huge (1000+)": np.sum(cluster_sizes_array > 1000)
        }
        
        # Calculate percentiles
        percentiles = {}
        if len(cluster_sizes_array) > 0:
            for p in [25, 50, 75, 90, 95]:
                percentiles[f"p{p}"] = float(np.percentile(cluster_sizes_array, p))
        
        # Determine distribution type
        if len(cluster_sizes) == 1:
            distribution_type = "single_cluster"
        elif size_ranges["huge (1000+)"] > 0:
            distribution_type = "skewed_large"
        elif size_ranges["tiny (2-5)"] > len(cluster_sizes) * 0.8:
            distribution_type = "mostly_tiny"
        else:
            distribution_type = "mixed"
        
        return {
            "size_ranges": size_ranges,
            "percentiles": percentiles,
            "distribution_type": distribution_type
        }
    
    def format_cluster_summary(self, stats: Dict[str, Any] = None) -> str:
        """Format clustering statistics for display."""
        
        if stats is None:
            stats = self.last_stats
        
        if not stats:
            return "No clustering statistics available."
        
        lines = []
        lines.append("📊 Clustering Summary:")
        lines.append(f"  • Total records: {stats['n_total']:,}")
        lines.append(f"  • Clusters formed: {stats['n_clusters']:,}")
        lines.append(f"  • Noise points: {stats['n_noise']:,} ({stats['noise_ratio']:.1%})")
        
        if stats['cluster_sizes']:
            quality = stats['quality_metrics']
            lines.append(f"  • Largest cluster: {quality['largest_cluster']:,} records")
            lines.append(f"  • Smallest cluster: {quality['smallest_cluster']:,} records")
            lines.append(f"  • Average cluster size: {quality['avg_cluster_size']:.1f} records")
            lines.append(f"  • Coverage ratio: {quality['coverage_ratio']:.1%}")
        
        if stats['silhouette_score'] is not None:
            lines.append(f"  • Silhouette score: {stats['silhouette_score']:.3f}")
        
        # Size distribution
        dist = stats['size_distribution']
        lines.append("📈 Size Distribution:")
        for range_name, count in dist['size_ranges'].items():
            if count > 0:
                lines.append(f"  • {range_name}: {count} clusters")
        
        return "\n".join(lines)
    
    def get_last_stats(self) -> Dict[str, Any]:
        """Get the last calculated statistics."""
        return self.last_stats.copy()
