"""
OpenAI-powered record deduplication functionality.
"""

from __future__ import annotations

import json
import time
import concurrent.futures
from typing import Dict, Any, List, Set, Optional

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
    
    def _analyze_duplicate_batch(self, batch_data: List[Dict[str, Any]], config: OpenAIConfig) -> List[Dict[str, Any]]:
        """Analyze a batch of candidate duplicate pairs using OpenAI."""
        if not batch_data:
            return []
        
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
            
            return analysis_results
            
        except Exception:
            # Return error results for this batch
            return [
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
    
    def _create_unique_records(
        self,
        features_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        ai_results: List[Dict[str, Any]],
        config: OpenAIConfig
    ) -> pd.DataFrame:
        """Create unique records by merging duplicates based on AI decisions."""
        
        # Create a mapping of duplicates
        duplicate_groups: Dict[str, Set[str]] = {}
        canonical_names: Dict[str, str] = {}
        
        # Process AI results to build duplicate groups
        for result in ai_results:
            if (result.get('same_organization', False) and 
                result.get('confidence', 0) >= config.confidence_threshold and
                not result.get('error', False)):
                
                pair_id = result['pair_id']
                # Handle pair_id parsing - assume format is "id1-id2"
                if '-' in pair_id:
                    id1, id2 = pair_id.split('-', 1)
                else:
                    continue
                
                primary_id = result.get('primary_record_id', id1)
                canonical_name = result.get('canonical_name', '')
                
                # Find existing group or create new one
                group_key = None
                for key, group in duplicate_groups.items():
                    if id1 in group or id2 in group:
                        group_key = key
                        break
                
                if group_key is None:
                    # Create new group with primary record as key
                    group_key = primary_id
                    duplicate_groups[group_key] = {id1, id2}
                else:
                    # Add to existing group
                    duplicate_groups[group_key].update({id1, id2})
                    # Update group key to primary record if needed
                    if primary_id in duplicate_groups[group_key] and group_key != primary_id:
                        duplicate_groups[primary_id] = duplicate_groups.pop(group_key)
                        group_key = primary_id
                
                canonical_names[group_key] = canonical_name
        
        # Create unique records dataframe
        unique_records = []
        processed_ids = set()
        
        # Process duplicate groups
        for primary_id, group_ids in duplicate_groups.items():
            if primary_id in processed_ids:
                continue
                
            # Get primary record
            if primary_id in cleaned_df.index:
                primary_record = cleaned_df.loc[primary_id].copy()
                primary_record['record_id'] = primary_id
                primary_record['is_merged'] = len(group_ids) > 1
                primary_record['merged_from'] = list(group_ids) if len(group_ids) > 1 else []
                primary_record['canonical_company'] = canonical_names.get(primary_id, primary_record.get('company_clean', ''))
                
                unique_records.append(primary_record)
                processed_ids.update(group_ids)
        
        # Add records that weren't identified as duplicates
        for record_id in cleaned_df.index:
            if record_id not in processed_ids:
                record = cleaned_df.loc[record_id].copy()
                record['record_id'] = record_id
                record['is_merged'] = False
                record['merged_from'] = []
                record['canonical_company'] = record.get('company_clean', '')
                unique_records.append(record)
        
        return pd.DataFrame(unique_records)
    
    def deduplicate_records(
        self,
        features_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        config: OpenAIConfig,
        sample_size: Optional[int] = None
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
                unique_records_df=unique_df,
                analysis_results=[],
                stats=stats
            )
        
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
        
        # Split into batches for parallel processing
        batches = [analysis_data[i:i + config.batch_size] for i in range(0, len(analysis_data), config.batch_size)]
        
        # Process batches with AI
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._analyze_duplicate_batch, batch, config): batch 
                for batch in batches
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    self.client.update_stats(api_stats, 0.0)  # Simplified for business logic
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
        
        # Create unique records
        unique_df = self._create_unique_records(features_df, cleaned_df, all_results, config)
        
        # Convert to OpenAIStats format
        stats = self.client.convert_to_stats(api_stats)
        
        return DeduplicationResult(
            unique_records_df=unique_df,
            analysis_results=all_results,
            stats=stats
        )
