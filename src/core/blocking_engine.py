"""
Business logic for blocking (candidate pair generation).

Pure business logic with no I/O or terminal output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd
import recordlinkage


@dataclass
class BlockingConfig:
    """Configuration for blocking operations."""
    use_phone_block: bool = True
    use_domain_block: bool = True
    use_company_block: bool = True
    use_fuzzy_block: bool = True
    fuzzy_window_size: int = 5


@dataclass
class BlockingStats:
    """Statistics from blocking operation."""
    input_records: int
    output_pairs: int
    total_possible_pairs: int
    reduction_ratio: float
    block_counts: Dict[str, int]
    available_columns: list[str]
    used_methods: list[str]


@dataclass
class BlockingResult:
    """Result of blocking operation."""
    pairs_df: pd.DataFrame
    stats: BlockingStats


class BlockingEngine:
    """Pure business logic for generating candidate pairs using blocking techniques."""
    
    def __init__(self):
        self.block_counts = {}
    
    def generate_candidate_pairs(self, df: pd.DataFrame) -> pd.MultiIndex:
        """
        Return a MultiIndex of candidate pairs using simple blocking rules.
        
        Parameters
        ----------
        df : pd.DataFrame
            Cleaned dataframe with normalized columns
            
        Returns
        -------
        pd.MultiIndex
            Index of candidate pairs
        """
        indexer = recordlinkage.Index()
        
        # Phone blocking (if available)
        if "phone_clean" in df.columns and not df["phone_clean"].isna().all():
            try:
                phone_indexer = recordlinkage.Index()
                phone_indexer.block("phone_clean")
                phone_pairs = phone_indexer.index(df)
                self.block_counts["phone_block"] = len(phone_pairs)
            except:
                phone_pairs = pd.MultiIndex.from_tuples([], names=['record_id', 'record_id'])
                self.block_counts["phone_block"] = 0
        else:
            phone_pairs = pd.MultiIndex.from_tuples([], names=['record_id', 'record_id'])
            self.block_counts["phone_block"] = 0
        
        # Domain blocking (if available)
        if "domain_clean" in df.columns and not df["domain_clean"].isna().all():
            try:
                domain_indexer = recordlinkage.Index()
                domain_indexer.block("domain_clean")
                domain_pairs = domain_indexer.index(df)
                self.block_counts["domain_block"] = len(domain_pairs)
            except:
                domain_pairs = pd.MultiIndex.from_tuples([], names=['record_id', 'record_id'])
                self.block_counts["domain_block"] = 0
        else:
            domain_pairs = pd.MultiIndex.from_tuples([], names=['record_id', 'record_id'])
            self.block_counts["domain_block"] = 0
        
        # Company blocking (always available)
        try:
            company_indexer = recordlinkage.Index()
            company_indexer.block("company_clean")
            company_pairs = company_indexer.index(df)
            self.block_counts["company_block"] = len(company_pairs)
        except:
            company_pairs = pd.MultiIndex.from_tuples([], names=['record_id', 'record_id'])
            self.block_counts["company_block"] = 0
        
        # Fuzzy company blocking (sorted neighbourhood)
        try:
            fuzzy_indexer = recordlinkage.Index()
            fuzzy_indexer.sortedneighbourhood("company_clean", window=5)
            fuzzy_pairs = fuzzy_indexer.index(df)
            self.block_counts["fuzzy_block"] = len(fuzzy_pairs)
        except:
            fuzzy_pairs = pd.MultiIndex.from_tuples([], names=['record_id', 'record_id'])
            self.block_counts["fuzzy_block"] = 0
        
        # Combine all pairs
        all_pairs = []
        for pairs in [phone_pairs, domain_pairs, company_pairs, fuzzy_pairs]:
            if len(pairs) > 0:
                all_pairs.append(pairs)
        
        if all_pairs:
            # Union all pairs and remove duplicates
            combined = all_pairs[0]
            for pairs in all_pairs[1:]:
                combined = combined.union(pairs)
            return combined
        else:
            return pd.MultiIndex.from_tuples([], names=['record_id', 'record_id'])
    
    def _get_block_size(self, indexer: recordlinkage.Index, df: pd.DataFrame) -> int:
        """Get the size of a block with error handling."""
        try:
            pairs = indexer.index(df)
            return len(pairs)
        except Exception:
            return 0
    
    def _analyze_available_columns(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Analyze which columns are available and which blocking methods were used."""
        available_cols = ["company"]  # Always have company
        used_methods = []
        
        if "phone_clean" in df.columns and not df["phone_clean"].isna().all():
            available_cols.append("phone")
            if self.block_counts.get("phone_block", 0) > 0:
                used_methods.append(f"phone ({self.block_counts['phone_block']:,})")
        
        if "domain_clean" in df.columns and not df["domain_clean"].isna().all():
            available_cols.append("domain")
            if self.block_counts.get("domain_block", 0) > 0:
                used_methods.append(f"domain ({self.block_counts['domain_block']:,})")
        
        if self.block_counts.get("company_block", 0) > 0:
            used_methods.append(f"company ({self.block_counts['company_block']:,})")
        
        if self.block_counts.get("fuzzy_block", 0) > 0:
            used_methods.append(f"fuzzy company ({self.block_counts['fuzzy_block']:,})")
        
        return available_cols, used_methods
    
    def process(self, df: pd.DataFrame, config: BlockingConfig) -> BlockingResult:
        """
        Process the dataframe to generate candidate pairs.
        
        This is pure business logic - no file I/O, no terminal output.
        """
        # Validate required columns
        required_columns = {"record_id", "company_clean"}
        missing = required_columns.difference(df.columns)
        if missing:
            cols = ", ".join(sorted(missing))
            raise KeyError(f"Missing required columns: {cols}")
        
        initial_rows = len(df)
        
        # Set record_id as index if it's not already
        if df.index.name != "record_id":
            df = df.set_index("record_id")
        
        # Generate candidate pairs
        self.block_counts = {}  # Reset counts
        candidates = self.generate_candidate_pairs(df)
        
        # Convert to DataFrame
        pair_df = candidates.to_frame(index=False)
        pair_df.columns = ["record_id_1", "record_id_2"]
        
        # Calculate statistics
        total_possible_pairs = initial_rows * (initial_rows - 1) / 2 if initial_rows > 1 else 0
        reduction_ratio = 1 - (len(pair_df) / total_possible_pairs) if total_possible_pairs > 0 else 0
        
        available_cols, used_methods = self._analyze_available_columns(df)
        
        stats = BlockingStats(
            input_records=initial_rows,
            output_pairs=len(pair_df),
            total_possible_pairs=int(total_possible_pairs),
            reduction_ratio=reduction_ratio,
            block_counts=self.block_counts.copy(),
            available_columns=available_cols,
            used_methods=used_methods
        )
        
        return BlockingResult(
            pairs_df=pair_df,
            stats=stats
        )
