"""Clustering Formatter - Terminal Output

Handles all terminal output formatting for the clustering step.
This module is responsible for displaying clustering progress, results,
and statistics in a user-friendly format.
"""

from typing import Dict, Any, List, Optional
import json


class ClusteringFormatter:
    """Formats clustering output for terminal display."""
    
    @staticmethod
    def format_progress(message: str) -> str:
        """Format a progress message."""
        return f"🔄 {message}"
    
    @staticmethod
    def format_auto_eps_initial(eps: float, min_samples: int, score: float) -> str:
        """Format initial auto-eps selection message."""
        return f"[Auto] Initial eps: {eps:.6f}, min_samples: {min_samples} (Silhouette Score: {score:.4f})"
    
    @staticmethod
    def format_refinement_iteration(
        iteration: int, 
        eps: float, 
        min_samples: int, 
        score: float, 
        improvement: float
    ) -> str:
        """Format refinement iteration message."""
        return (f"[Refinement {iteration}] eps: {eps:.6f}, min_samples: {min_samples} "
                f"(Score: {score:.4f}, Improvement: {improvement:.4f})")
    
    @staticmethod
    def format_convergence(iterations: int, threshold: float) -> str:
        """Format convergence message."""
        return f"Converged after {iterations} iterations (improvement < {threshold})"
    
    @staticmethod
    def format_final_parameters(eps: float, min_samples: int, score: float) -> str:
        """Format final parameter selection message."""
        return f"[Final] Best eps: {eps:.6f}, min_samples: {min_samples} (Silhouette Score: {score:.4f})"
    
    @staticmethod
    def format_file_output(output_path: str, count: int) -> str:
        """Format file output message."""
        return f"Wrote {count} clustered records to {output_path}"
    
    @staticmethod
    def format_agg_output(agg_path: str) -> str:
        """Format aggregated features output message."""
        return f"Wrote aggregated features (incl. cluster) to {agg_path}"
    
    @staticmethod
    def format_clustering_complete() -> str:
        """Format clustering completion header."""
        return "\n📊 Clustering Complete!"
    
    @staticmethod
    def format_separator() -> str:
        """Format section separator."""
        return "─" * 50
    
    @staticmethod
    def format_cluster_results(stats: Dict[str, Any]) -> List[str]:
        """Format cluster results section."""
        results = stats.get("results", {})
        lines = ["📈 Cluster Results:"]
        
        total_records = results.get("n_total_points", 0)
        n_clusters = results.get("n_clusters", 0)
        n_noise = results.get("n_noise_points", 0)
        noise_ratio = results.get("noise_ratio", 0) * 100
        
        lines.append(f"  • Total records:         {total_records:,}")
        lines.append(f"  • Clusters formed:       {n_clusters:,}")
        lines.append(f"  • Noise points:          {n_noise:,} ({noise_ratio:.1f}%)")
        
        cluster_stats = results.get("cluster_stats", {})
        if cluster_stats.get("max_size", 0) > 0:
            lines.append(f"  • Largest cluster:       {cluster_stats['max_size']:,} records")
            lines.append(f"  • Smallest cluster:      {cluster_stats['min_size']:,} records")
            lines.append(f"  • Average cluster size:  {cluster_stats['mean_size']:.1f} records")
        
        return lines
    
    @staticmethod
    def format_parameters_used(stats: Dict[str, Any]) -> List[str]:
        """Format parameters used section."""
        results = stats.get("results", {})
        params = stats.get("parameters", {})
        
        lines = ["⚙️ Parameters Used:"]
        lines.append(f"  • eps:                   {results.get('final_eps', 0):.6f}")
        lines.append(f"  • min_samples:           {results.get('final_min_samples', 0)}")
        lines.append(f"  • scaling:               {'Yes' if params.get('scale', False) else 'No'}")
        lines.append(f"  • auto_eps:              {'Yes' if params.get('auto_eps', False) else 'No'}")
        
        silhouette_score = results.get("silhouette_score", 0)
        if silhouette_score > 0:
            lines.append(f"  • silhouette_score:      {silhouette_score:.3f}")
        
        return lines
    
    @staticmethod
    def format_files_created(output_path: str, agg_path: str) -> List[str]:
        """Format files created section."""
        return [
            "💾 Files Created:",
            f"  • Cluster assignments:   {output_path}",
            f"  • Aggregated features:   {agg_path}"
        ]
    
    @staticmethod
    def format_success_message(n_clusters: int) -> List[str]:
        """Format success message."""
        if n_clusters > 1:
            return [
                f"✅ Success! Created {n_clusters:,} clusters",
                "   Next step: Analyze clusters with OpenAI integration",
                "   Command: python -m src.cli.openai_deduplication --exclude-clusters 0 --min-cluster-size 3"
            ]
        else:
            return [
                f"⚠️  Only {n_clusters} cluster formed - consider adjusting parameters",
                "   • Try smaller eps value for more clusters",
                "   • Try auto-eps: python -m src.cli.clustering --auto-eps --scale"
            ]
    
    @staticmethod
    def format_comprehensive_results(
        stats: Dict[str, Any], 
        output_path: str, 
        agg_path: str
    ) -> str:
        """Format comprehensive clustering results."""
        lines = []
        
        # Header
        lines.append(ClusteringFormatter.format_clustering_complete())
        lines.append(ClusteringFormatter.format_separator())
        
        # Cluster results
        lines.extend(ClusteringFormatter.format_cluster_results(stats))
        lines.append("")
        
        # Parameters used
        lines.extend(ClusteringFormatter.format_parameters_used(stats))
        lines.append("")
        
        # Files created
        lines.extend(ClusteringFormatter.format_files_created(output_path, agg_path))
        lines.append("")
        
        # Success message
        n_clusters = stats.get("results", {}).get("n_clusters", 0)
        lines.extend(ClusteringFormatter.format_success_message(n_clusters))
        
        return "\n".join(lines)
    
    @staticmethod
    def format_error(error_message: str) -> str:
        """Format error message."""
        return f"❌ Error: {error_message}"
    
    @staticmethod
    def format_warning(warning_message: str) -> str:
        """Format warning message."""
        return f"⚠️  Warning: {warning_message}"
    
    @staticmethod
    def format_auto_eps_details(stats: Dict[str, Any]) -> List[str]:
        """Format auto-eps iteration details."""
        lines = []
        iterations = stats.get("iterations", [])
        
        if iterations:
            lines.append("🔍 Auto-eps Refinement Details:")
            for iteration_data in iterations:
                iteration = iteration_data.get("iteration", 0)
                eps = iteration_data.get("eps", 0)
                min_samples = iteration_data.get("min_samples", 0)
                score = iteration_data.get("silhouette_score", 0)
                improvement = iteration_data.get("improvement", 0)
                
                lines.append(ClusteringFormatter.format_refinement_iteration(
                    iteration, eps, min_samples, score, improvement
                ))
        
        return lines
