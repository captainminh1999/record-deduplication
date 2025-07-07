"""
OpenAI-powered cluster review functionality.
"""

from __future__ import annotations

import json
import time
import concurrent.futures
from typing import Dict, Any, List

import pandas as pd

from .openai_client import OpenAIClient
from .openai_types import OpenAIConfig, ClusterReviewResult


class OpenAIClusterReviewer:
    """Handles OpenAI-powered cluster review."""
    
    def __init__(self, client: OpenAIClient):
        self.client = client
    
    def _create_cluster_prompt(self, cluster_df: pd.DataFrame, cluster_id: int) -> List[Dict[str, str]]:
        """Create prompt for cluster review."""
        cluster_size = len(cluster_df)
        
        # Create prompt
        lines = [f"Cluster {cluster_id} contains {cluster_size} records:"]
        for _, row in cluster_df.iterrows():
            lines.append(
                f"  - ID {row['record_id']}: {row.get('company_clean', '')}, {row.get('domain_clean', '')}"
            )
        
        lines.extend([
            "Analyze the records above. There may be more than one group of duplicates or companies that are subsidiaries/brands under a parent entity in this cluster.",
            "For each group of records that are duplicates or subsidiaries/brands of the same organization, return a JSON object with these fields:",
            '- "primary_organization": the canonical name for the group (required)',
            '- "canonical_domains": a list of all domains belonging to this group (required)',
            '- "record_ids": a list of record IDs belonging to this organization (required)',
            '- "confidence": a number from 0 to 1 indicating your confidence in this grouping (required)',
            "",
            "Only include groups with actual duplicates or subsidiaries/brands that should be grouped together.",
            "If all records are unique and unrelated, return an empty JSON array [].",
            "Return a JSON array of these objects. Only output valid JSON, no explanations."
        ])
        
        prompt_text = "\n".join(lines)
        return [{"role": "user", "content": prompt_text}]
    
    def _process_cluster(self, cluster_data: tuple, config: OpenAIConfig, api_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single cluster."""
        cluster_id, group = cluster_data
        group = group.reset_index(drop=False)
        if "record_id" not in group.columns:
            return None
        
        cluster_id_int = int(cluster_id)
        
        try:
            call_start = time.time()
            messages = self._create_cluster_prompt(group, cluster_id_int)
            content, usage = self.client.make_request(
                messages=messages,
                model=config.model,
                temperature=0,
                max_tokens=2000
            )
            call_duration = time.time() - call_start
            self.client.update_stats(api_stats, call_duration, type('Response', (), {'usage': type('Usage', (), usage)})())
            
            try:
                canonical_groups = json.loads(content.strip())
                filtered_groups = []
                for canonical_group in canonical_groups:
                    filtered_group = {
                        "primary_organization": canonical_group.get("primary_organization", ""),
                        "canonical_domains": canonical_group.get("canonical_domains", []),
                        "record_ids": canonical_group.get("record_ids", []),
                        "confidence": canonical_group.get("confidence", None),
                    }
                    filtered_groups.append(filtered_group)
            except Exception:
                filtered_groups = []
            
            return {
                "cluster_id": cluster_id_int,
                "record_ids": [str(r) for r in group["record_id"].tolist()],
                "canonical_groups": filtered_groups,
                "raw_response": content,
            }
        except Exception as e:
            call_duration = time.time() - time.time()  # Simplified for business logic
            self.client.update_stats(api_stats, call_duration, None, str(type(e).__name__))
            return {
                "cluster_id": cluster_id_int,
                "record_ids": [str(r) for r in group["record_id"].tolist()] if "record_id" in group.columns else [],
                "canonical_groups": [],
                "raw_response": f"ERROR: {str(e)}",
            }
    
    def review_clusters(self, clusters_df: pd.DataFrame, config: OpenAIConfig) -> ClusterReviewResult:
        """
        Review DBSCAN clusters with GPT for validation.
        
        Pure business logic - no file I/O or terminal output.
        """
        api_stats = self.client.init_stats()
        results: List[Dict[str, Any]] = []
        
        # Filter clusters according to config
        clusters = list(clusters_df.groupby("cluster"))
        filtered_clusters = []
        
        for cluster_id, group in clusters:
            cluster_id_int = int(cluster_id)
            cluster_size = len(group)
            
            # Apply filters
            if cluster_id_int in config.exclude_clusters:
                continue
            if config.exclude_noise and cluster_id_int == -1:
                continue
            if cluster_size < config.min_cluster_size:
                continue
            if config.max_cluster_size and cluster_size > config.max_cluster_size:
                continue
            
            # Sample large clusters if requested
            if config.sample_large_clusters and cluster_size > config.sample_large_clusters:
                group = group.sample(n=config.sample_large_clusters, random_state=42)
                cluster_size = len(group)
            
            filtered_clusters.append((cluster_id_int, group))
        
        # Process clusters with threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {executor.submit(self._process_cluster, c, config, api_stats): c for c in filtered_clusters}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
        
        # Convert to OpenAIStats format
        stats = self.client.convert_to_stats(api_stats)
        
        return ClusterReviewResult(cluster_reviews=results, stats=stats)
