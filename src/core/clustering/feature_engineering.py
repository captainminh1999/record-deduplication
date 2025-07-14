"""Feature Engineering Module

Enhanced feature engineering for similarity data to improve clustering performance.
"""

from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np


class FeatureEngineer:
    """Enhanced feature engineering for clustering."""
    
    def __init__(self):
        self.feature_weights = {}
        self.features_used = []
    
    def prepare_enhanced_features(
        self, 
        feats: pd.DataFrame, 
        cleaned: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Generate enhanced features for clustering with domain-focused weights.
        
        This creates 4-6 new features from base similarity scores to improve
        clustering performance, particularly in the eps range 0.01-0.15.
        
        Args:
            feats: Features DataFrame with similarity scores
            cleaned: Cleaned records DataFrame
            
        Returns:
            Tuple of (enhanced_features_df, feature_weights_dict)
        """
        print("🔧 Generating enhanced features for clustering...")
        
        # Aggregate features per record (both directions)
        print("  📊 Aggregating similarity features per record...")
        
        # Process record_id_1 direction
        melted1 = feats.groupby("record_id_1").agg({
            "company_sim": "mean",
            "domain_sim": "mean", 
            "phone_exact": "mean",
            "address_sim": "mean"
        }).reset_index()
        melted1.rename(columns={"record_id_1": "record_id"}, inplace=True)
        
        # Process record_id_2 direction
        melted2 = feats.groupby("record_id_2").agg({
            "company_sim": "mean",
            "domain_sim": "mean", 
            "phone_exact": "mean",
            "address_sim": "mean"
        }).reset_index()
        melted2.rename(columns={"record_id_2": "record_id"}, inplace=True)
        
        # Combine both directions by taking maximum similarity scores
        melted = pd.concat([melted1, melted2], ignore_index=True)
        melted = melted.groupby("record_id").max().reset_index()
        
        # Ensure record_id is string type for consistent merging
        melted["record_id"] = melted["record_id"].astype(str)
        
        # Merge with cleaned data to preserve all records
        cleaned_subset = cleaned.reset_index()[["record_id"]].copy()
        cleaned_subset["record_id"] = cleaned_subset["record_id"].astype(str)
        melted = cleaned_subset.merge(melted, on="record_id", how="left").fillna(0)
        
        # Base similarity columns
        sim_cols = ["company_sim", "domain_sim", "phone_exact", "address_sim"]
        sim_cols = [col for col in sim_cols if col in melted.columns]
        
        # Track engineered features
        engineered_features = []
        
        # 1. Interaction features (cross-products and relationships)
        print("  🔀 Creating interaction features...")
        if "company_sim" in melted.columns and "domain_sim" in melted.columns:
            melted["company_domain_product"] = melted["company_sim"] * melted["domain_sim"]
            melted["company_domain_sum"] = melted["company_sim"] + melted["domain_sim"]
            # Safe division
            melted["company_domain_ratio"] = np.where(
                melted["domain_sim"] > 1e-8,
                melted["company_sim"] / melted["domain_sim"],
                0
            )
            engineered_features.extend(["company_domain_product", "company_domain_sum", "company_domain_ratio"])
        
        # 2. Non-linear transformations
        print("  🌟 Applying non-linear transformations...")
        for col in ["company_sim", "domain_sim"]:
            if col in melted.columns:
                melted[f"{col}_squared"] = melted[col] ** 2
                melted[f"{col}_sqrt"] = np.sqrt(melted[col] + 1e-8)
                melted[f"{col}_log"] = np.log(melted[col] + 1e-8)
                engineered_features.extend([f"{col}_squared", f"{col}_sqrt", f"{col}_log"])
        
        # 3. Statistical features (variance, skewness indicators)
        print("  📈 Computing statistical features...")
        if len(sim_cols) > 1:
            melted["sim_variance"] = melted[sim_cols].var(axis=1)
            melted["sim_mean"] = melted[sim_cols].mean(axis=1)
            melted["sim_max"] = melted[sim_cols].max(axis=1)
            melted["sim_min"] = melted[sim_cols].min(axis=1)
            melted["sim_range"] = melted["sim_max"] - melted["sim_min"]
            engineered_features.extend(["sim_variance", "sim_mean", "sim_max", "sim_min", "sim_range"])
        
        # Update features list
        all_features = sim_cols + engineered_features
        self.features_used = all_features
        
        # Handle NaN values from log and sqrt operations
        for col in engineered_features:
            if col in melted.columns:
                melted[col] = melted[col].fillna(0)
        
        # Apply enhanced feature weights - prioritize domain similarity
        print("  ⚖️  Applying enhanced feature weights...")
        weights = self._calculate_feature_weights(melted, sim_cols, engineered_features)
        
        # Apply weights to features
        for feature, weight in weights.items():
            if feature in melted.columns:
                melted[feature] = melted[feature] * weight
        
        self.feature_weights = weights
        
        print(f"  ✅ Enhanced features created: {len(all_features)} total features")
        print(f"     📋 Base features: {len(sim_cols)}")
        print(f"     🚀 Engineered features: {len(engineered_features)}")
        
        return melted, weights
    
    def _calculate_feature_weights(
        self, 
        melted: pd.DataFrame, 
        sim_cols: List[str], 
        engineered_features: List[str]
    ) -> Dict[str, float]:
        """Calculate feature weights to prioritize domain clustering."""
        weights = {}
        
        # Base features - emphasize domain similarity for company clustering
        for col in sim_cols:
            if col == "domain_sim":
                weights[col] = 2.0  # Strong domain focus
            elif col == "company_sim":
                weights[col] = 1.5  # Moderate company focus  
            else:
                weights[col] = 1.0  # Standard weight
        
        # Interaction features - boost domain-company interactions
        for col in engineered_features:
            if "domain" in col and "company" in col:
                weights[col] = 1.8  # High weight for domain-company interactions
            elif "domain" in col:
                weights[col] = 1.6  # Medium-high for domain features
            elif "company" in col:
                weights[col] = 1.3  # Medium for company features
            elif col.startswith("sim_"):
                weights[col] = 1.2  # Medium for statistical features
            else:
                weights[col] = 1.0  # Default weight
        
        return weights
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the features and weights used."""
        return {
            "features_used": self.features_used,
            "feature_weights": self.feature_weights,
            "total_features": len(self.features_used)
        }
