"""
Business logic for similarity feature computation.

Pure business logic with no I/O or terminal output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import pandas as pd
import recordlinkage
from rapidfuzz import fuzz


@dataclass
class SimilarityConfig:
    """Configuration for similarity computation."""
    use_company_sim: bool = True
    use_domain_sim: bool = True
    use_phone_exact: bool = True
    use_address_sim: bool = True


@dataclass
class SimilarityStats:
    """Statistics from similarity computation."""
    input_records: int
    input_pairs: int
    output_features: int
    columns_used: List[str]
    missing_columns: List[str]
    similarity_metrics: Dict[str, Any]


@dataclass
class SimilarityResult:
    """Result of similarity computation."""
    features_df: pd.DataFrame
    stats: SimilarityStats


class SimilarityEngine:
    """Pure business logic for computing similarity features between record pairs."""
    
    def compute_features(self, df: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute similarity features for candidate pairs.
        
        Parameters
        ----------
        df : pd.DataFrame
            Cleaned dataframe with normalized columns (indexed by record_id)
        pairs_df : pd.DataFrame
            Candidate pairs with columns record_id_1, record_id_2
            
        Returns
        -------
        pd.DataFrame
            Features dataframe with similarity scores
        """
        # Convert pairs to MultiIndex for recordlinkage
        pairs_multiindex = pd.MultiIndex.from_arrays([
            pairs_df["record_id_1"],
            pairs_df["record_id_2"]
        ])
        
        # Initialize comparison object
        compare = recordlinkage.Compare()
        
        # Available columns for comparison
        available_cols = []
        
        # Company similarity (always required)
        if "company_clean" in df.columns:
            compare.string("company_clean", "company_clean", method="jarowinkler", label="company_sim")
            available_cols.append("company")
        
        # Domain exact match (if available)
        if "domain_clean" in df.columns and not df["domain_clean"].isna().all():
            compare.exact("domain_clean", "domain_clean", label="domain_sim")
            available_cols.append("domain")
        
        # Phone exact match (if available)
        if "phone_clean" in df.columns and not df["phone_clean"].isna().all():
            compare.exact("phone_clean", "phone_clean", label="phone_exact")
            available_cols.append("phone")
        
        # Address similarity using token set ratio (if available)
        address_sim_added = False
        if "address_clean" in df.columns and not df["address_clean"].isna().all():
            # We'll add this manually after the recordlinkage comparison
            available_cols.append("address")
            address_sim_added = True
        
        # Compute features
        if len(pairs_multiindex) == 0:
            # No pairs to compare
            features = pd.DataFrame()
        else:
            features = compare.compute(pairs_multiindex, df)
        
        # Add address similarity manually if needed
        if address_sim_added and len(features) > 0:
            address_scores = []
            for idx, (id1, id2) in enumerate(pairs_multiindex):
                try:
                    addr1 = str(df.loc[id1, "address_clean"]) if pd.notna(df.loc[id1, "address_clean"]) else ""
                    addr2 = str(df.loc[id2, "address_clean"]) if pd.notna(df.loc[id2, "address_clean"]) else ""
                    
                    if addr1 and addr2:
                        score = fuzz.token_set_ratio(addr1, addr2) / 100.0
                    else:
                        score = 0.0
                    
                    address_scores.append(score)
                except (KeyError, IndexError):
                    address_scores.append(0.0)
            
            features["address_sim"] = address_scores
        
        # Reset index to get record IDs as columns
        features = features.reset_index()
        
        return features, available_cols
    
    def _get_missing_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of missing optional columns."""
        optional_cols = ["phone_clean", "domain_clean", "address_clean"]
        missing_cols = []
        
        for col in optional_cols:
            if col not in df.columns or df[col].isna().all():
                missing_cols.append(col)
        
        return missing_cols
    
    def _compute_similarity_metrics(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistics about similarity scores."""
        metrics = {}
        
        similarity_cols = [col for col in features_df.columns if col.endswith('_sim')]
        exact_cols = [col for col in features_df.columns if col.endswith('_exact')]
        
        for col in similarity_cols + exact_cols:
            if col in features_df.columns:
                metrics[col] = {
                    "mean": float(features_df[col].mean()),
                    "std": float(features_df[col].std()),
                    "min": float(features_df[col].min()),
                    "max": float(features_df[col].max()),
                    "null_count": int(features_df[col].isnull().sum())
                }
        
        return metrics
    
    def process(self, cleaned_df: pd.DataFrame, pairs_df: pd.DataFrame, config: SimilarityConfig) -> SimilarityResult:
        """
        Process the dataframes to compute similarity features.
        
        This is pure business logic - no file I/O, no terminal output.
        """
        # Validate required columns
        required_columns = {"record_id", "company_clean"}
        missing = required_columns.difference(cleaned_df.columns)
        if missing:
            cols = ", ".join(sorted(missing))
            raise KeyError(f"Missing required columns: {cols}")
        
        # Set record_id as index if it's not already
        if cleaned_df.index.name != "record_id":
            cleaned_df = cleaned_df.set_index("record_id")
        
        initial_rows = len(cleaned_df)
        input_pairs = len(pairs_df)
        
        # Compute features
        features_df, available_cols = self.compute_features(cleaned_df, pairs_df)
        
        # Merge additional columns from cleaned data for context
        if len(features_df) > 0:
            # Add company and domain info for both records in each pair
            for suffix, id_col in [("_1", "record_id_1"), ("_2", "record_id_2")]:
                # Merge company info
                company_data = cleaned_df[["company_clean"]].rename(columns={"company_clean": f"company_clean{suffix}"})
                features_df = features_df.merge(company_data, left_on=id_col, right_index=True, how="left")
                
                # Merge domain info if available
                if "domain_clean" in cleaned_df.columns:
                    domain_data = cleaned_df[["domain_clean"]].rename(columns={"domain_clean": f"domain_clean{suffix}"})
                    features_df = features_df.merge(domain_data, left_on=id_col, right_index=True, how="left")
        
        # Calculate statistics
        missing_cols = self._get_missing_columns(cleaned_df)
        similarity_metrics = self._compute_similarity_metrics(features_df)
        
        stats = SimilarityStats(
            input_records=initial_rows,
            input_pairs=input_pairs,
            output_features=len(features_df),
            columns_used=available_cols,
            missing_columns=missing_cols,
            similarity_metrics=similarity_metrics
        )
        
        return SimilarityResult(
            features_df=features_df,
            stats=stats
        )
