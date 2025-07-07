"""
OpenAI-powered record deduplication functionality.
"""

from __future__ import annotations

import json
import time
import concurrent.futures
from typing import Dict, Any, List, Set, Optional, Callable, Tuple

import pandas as pd

from .openai_client import OpenAIClient
from .openai_types import OpenAIConfig, DeduplicationResult


class OpenAIDeduplicator:
    """Handles OpenAI-powered record deduplication."""
    
    def __init__(self, client: OpenAIClient):
        self.client = client
    
    def _create_batch_prompt(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Create prompt for batch deduplication analysis."""
        prompt_lines = [
            "Analyze the following pairs of company records with their similarity scores.",
            "For each pair, determine:",
            "1. Are they the SAME organization? (true/false)",
            "2. Your confidence level (0.0-1.0)", 
            "3. Which record should be the primary/canonical one?",
            "4. What should the merged company name be?",
            "",
            "Consider:",
            "- Companies with slight name variations (abbreviations, punctuation) are likely the same",
            "- Subsidiaries and parent companies are DIFFERENT organizations",
            "- Different divisions/brands of the same company are the SAME organization",
            "",
            "Records to analyze:"
        ]
        
        for i, pair in enumerate(batch_data, 1):
            prompt_lines.extend([
                f"Pair {i} (ID: {pair['pair_id']}):",
                f"  Similarity Score: {pair['similarity_score']:.3f}",
                f"  Record A (ID: {pair['record_1']['id']}): '{pair['record_1']['company']}'",
                f"    Domain: '{pair['record_1']['domain']}', Phone: '{pair['record_1']['phone']}'",
                f"  Record B (ID: {pair['record_2']['id']}): '{pair['record_2']['company']}'", 
                f"    Domain: '{pair['record_2']['domain']}', Phone: '{pair['record_2']['phone']}'",
                ""
            ])
        
        prompt_lines.extend([
            "Return a JSON array with this exact format:",
            '[{"pair_id": "id1-id2", "same_organization": true, "confidence": 0.95, "primary_record_id": "id1", "canonical_name": "Canonical Company Name"}]',
            "",
            "Requirements:",
            "- Return valid JSON only, no explanations",
            "- Include ALL pairs in your response", 
            "- Use exact pair_id values provided",
            "- confidence must be 0.0-1.0",
            "- same_organization must be true or false",
            "- primary_record_id must be one of the two record IDs",
            "- canonical_name should be the best/most complete company name"
        ])
        
        prompt_text = "\n".join(prompt_lines)
        return [{"role": "user", "content": prompt_text}]
    
    def _analyze_duplicate_batch(self, batch_data: List[Dict[str, Any]], config: OpenAIConfig) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Analyze a batch of candidate duplicate pairs using OpenAI."""
        if not batch_data:
            return [], {}
        
        try:
            messages = self._create_batch_prompt(batch_data)
            content, usage = self.client.make_request(
                messages=messages,
                model=config.model,
                temperature=0.1,
                max_tokens=2000
            )
            
            # Clean up potential markdown formatting
            answer = content.strip()
            if answer.startswith('```json'):
                answer = answer[7:]
            if answer.endswith('```'):
                answer = answer[:-3]
            answer = answer.strip()
            
            try:
                analysis_results = json.loads(answer)
                if not isinstance(analysis_results, list):
                    analysis_results = []
            except json.JSONDecodeError:
                analysis_results = []
            
            return analysis_results, usage
            
        except Exception:
            # Return error results for this batch
            error_results = [
                {
                    "pair_id": pair['pair_id'],
                    "same_organization": False,
                    "confidence": 0.0,
                    "primary_record_id": pair['record_1']['id'],
                    "canonical_name": pair['record_1']['company'],
                    "error": True
                }
                for pair in batch_data
            ]
            return error_results, {}
    
    def _create_unique_records(
        self,
        features_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        ai_results: List[Dict[str, Any]],
        config: OpenAIConfig
    ) -> pd.DataFrame:
        """Create unique records by merging duplicates based on AI decisions. Optimized for speed."""
        
        # Use Union-Find data structure for efficient group management
        record_to_group = {}  # record_id -> group_representative
        canonical_names = {}  # group_representative -> canonical_name
        
        def find_root(record_id: str) -> str:
            """Find the root representative of a group (with path compression)."""
            if record_id not in record_to_group:
                record_to_group[record_id] = record_id
                return record_id
            
            if record_to_group[record_id] != record_id:
                record_to_group[record_id] = find_root(record_to_group[record_id])  # Path compression
            return record_to_group[record_id]
        
        def union_groups(id1: str, id2: str, primary_id: str, canonical_name: str) -> None:
            """Union two groups with the specified primary as root."""
            root1, root2 = find_root(id1), find_root(id2)
            
            if root1 != root2:
                # Make primary_id the root of the merged group
                if primary_id == id1:
                    record_to_group[root2] = root1
                    canonical_names[root1] = canonical_name
                elif primary_id == id2:
                    record_to_group[root1] = root2
                    canonical_names[root2] = canonical_name
                else:
                    # Primary is neither root, make it the new root
                    record_to_group[root1] = primary_id
                    record_to_group[root2] = primary_id
                    record_to_group[primary_id] = primary_id
                    canonical_names[primary_id] = canonical_name
            else:
                # Already in same group, just update canonical name
                canonical_names[root1] = canonical_name
        
        # Process AI results to build duplicate groups efficiently
        for result in ai_results:
            if (result.get('same_organization', False) and 
                result.get('confidence', 0) >= config.confidence_threshold and
                not result.get('error', False)):
                
                pair_id = result['pair_id']
                if '-' in pair_id:
                    id1, id2 = pair_id.split('-', 1)
                    primary_id = result.get('primary_record_id', id1)
                    canonical_name = result.get('canonical_name', '')
                    union_groups(id1, id2, primary_id, canonical_name)
        
        # Build group mapping efficiently
        groups = {}  # group_representative -> list of member_ids
        for record_id in record_to_group:
            root = find_root(record_id)
            if root not in groups:
                groups[root] = []
            groups[root].append(record_id)
        
        # Start with a copy of the original dataframe
        unique_df = cleaned_df.copy()
        unique_df = unique_df.reset_index()  # Convert index to column
        
        # Initialize new columns efficiently
        unique_df['is_merged'] = False
        unique_df['merged_from'] = [[] for _ in range(len(unique_df))]
        unique_df['canonical_company'] = unique_df.get('company_clean', '')
        
        # Track which records to keep
        records_to_remove = set()
        
        # Process each group
        for group_root, group_members in groups.items():
            if len(group_members) > 1:  # Only process actual groups
                # Find the primary record in the dataframe
                primary_idx = unique_df[unique_df['record_id'] == group_root].index
                
                if len(primary_idx) > 0:
                    primary_idx = primary_idx[0]
                    
                    # Update primary record
                    unique_df.at[primary_idx, 'is_merged'] = True
                    unique_df.at[primary_idx, 'merged_from'] = group_members
                    if group_root in canonical_names:
                        unique_df.at[primary_idx, 'canonical_company'] = canonical_names[group_root]
                    
                    # Mark other group members for removal
                    for member_id in group_members:
                        if member_id != group_root:
                            records_to_remove.add(member_id)
        
        # Remove duplicates efficiently
        if records_to_remove:
            unique_df = unique_df[~unique_df['record_id'].isin(records_to_remove)]
        
        return unique_df
    
    def _auto_merge_high_similarity_pairs(
        self,
        features_df: pd.DataFrame,
        config: OpenAIConfig,
        auto_merge_threshold: float = 0.98
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Auto-merge pairs with very high similarity scores without AI analysis."""
        
        # Find pairs with extremely high similarity that are likely exact matches
        auto_merge_pairs = features_df[features_df['company_sim'] >= auto_merge_threshold].copy()
        remaining_pairs = features_df[features_df['company_sim'] < auto_merge_threshold].copy()
        
        # Create auto-merge results
        auto_results = []
        for _, row in auto_merge_pairs.iterrows():
            id1, id2 = row['record_id_1'], row['record_id_2']
            
            # Use the first record as primary by default
            auto_results.append({
                "pair_id": f"{id1}-{id2}",
                "same_organization": True,
                "confidence": 1.0,  # High confidence for auto-merge
                "primary_record_id": id1,
                "canonical_name": row.get('company_clean_1', ''),
                "auto_merged": True
            })
        
        return remaining_pairs, auto_results

    def deduplicate_records(
        self,
        features_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        config: OpenAIConfig,
        sample_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> DeduplicationResult:
        """
        Perform AI-powered deduplication using similarity features.
        
        Pure business logic - no file I/O or terminal output.
        """
        api_stats = self.client.init_stats()
        
        # Filter pairs by similarity threshold
        high_sim_pairs = features_df[features_df['company_sim'] >= config.similarity_threshold].copy()
        
        if len(high_sim_pairs) == 0:
            # No pairs to analyze, return original data
            unique_df = cleaned_df.copy().reset_index()
            unique_df['is_merged'] = False
            unique_df['merged_from'] = []
            unique_df['canonical_company'] = unique_df.get('company_clean', '')
            
            stats = self.client.convert_to_stats(api_stats)
            
            return DeduplicationResult(
                unique_records=unique_df,
                analysis={"results": [], "summary": "No pairs above similarity threshold"},
                stats=stats
            )
        
        # Auto-merge very high similarity pairs (DISABLED per user request)
        # high_sim_pairs, auto_merge_results = self._auto_merge_high_similarity_pairs(high_sim_pairs, config)
        auto_merge_results = []  # Skip auto-merge optimization
        
        # Sample if requested
        if sample_size and len(high_sim_pairs) > sample_size:
            high_sim_pairs = high_sim_pairs.sample(n=sample_size, random_state=42)
        
        # Prepare analysis data
        analysis_data = []
        for _, row in high_sim_pairs.iterrows():
            id1, id2 = row['record_id_1'], row['record_id_2']
            
            pair_info = {
                'pair_id': f"{id1}-{id2}",
                'similarity_score': row['company_sim'],
                'record_1': {
                    'id': id1,
                    'company': row.get('company_clean_1', ''),
                    'domain': row.get('domain_clean_1', ''),
                    'phone': row.get('phone_clean_1', '')
                },
                'record_2': {
                    'id': id2,
                    'company': row.get('company_clean_2', ''),
                    'domain': row.get('domain_clean_2', ''),
                    'phone': row.get('phone_clean_2', '')
                }
            }
            analysis_data.append(pair_info)
        
        # Combine auto-merge results with AI analysis results
        all_results = auto_merge_results
        
        if len(analysis_data) > 0:
            # Split into batches for parallel processing
            batches = [analysis_data[i:i + config.batch_size] for i in range(0, len(analysis_data), config.batch_size)]
            
            # Process batches with AI
            completed_batches = 0
            total_batches = len(batches)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(self._analyze_duplicate_batch, batch, config): batch 
                    for batch in batches
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_batch):
                    try:
                        batch_results, usage_stats = future.result()
                        all_results.extend(batch_results)
                        
                        # Update API stats with actual usage
                        if usage_stats and not usage_stats.get('error'):
                            self.client.update_stats(
                                api_stats, 
                                usage_stats.get('duration', 0.0),
                                usage_stats
                            )
                        else:
                            self.client.update_stats(api_stats, 0.0, None, "api_error")
                            
                    except Exception:
                        # Handle batch failure
                        failed_batch = future_to_batch[future]
                        for pair in failed_batch:
                            all_results.append({
                                "pair_id": pair['pair_id'],
                                "same_organization": False,
                                "confidence": 0.0,
                                "primary_record_id": pair['record_1']['id'],
                                "canonical_name": pair['record_1']['company'],
                                "error": True
                            })
                        self.client.update_stats(api_stats, 0.0, None, "batch_failure")
                    
                    # Update progress
                    completed_batches += 1
                    if progress_callback:
                        progress_callback(completed_batches, total_batches)
        
        # Create unique records
        unique_df = self._create_unique_records(features_df, cleaned_df, all_results, config)
        
        # Create summary statistics
        total_pairs = len(all_results)
        merged_pairs = sum(1 for r in all_results if r.get('same_organization', False))
        summary_stats = {
            "total_pairs_analyzed": total_pairs,
            "pairs_merged": merged_pairs,
            "merge_rate": merged_pairs / total_pairs if total_pairs > 0 else 0.0
        }
        
        # Convert to OpenAIStats format
        stats = self.client.convert_to_stats(api_stats)
        
        return DeduplicationResult(
            unique_records=unique_df,
            analysis={"results": all_results, "summary": summary_stats},
            stats=stats
        )
