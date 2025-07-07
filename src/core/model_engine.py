"""
Business logic for model training and scoring.

Pure business logic with no I/O or terminal output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os
import json

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression


@dataclass
class ModelConfig:
    """Configuration for model training."""
    confidence_threshold: float = 0.8
    use_heuristic_labels: bool = True
    random_state: int = 42


@dataclass
class ModelStats:
    """Statistics from model training."""
    input_pairs: int
    features_used: list[str]
    labels_source: str
    training_pairs: int
    high_confidence_pairs: int
    model_score: Optional[float]
    label_distribution: Dict[str, int]


@dataclass
class ModelResult:
    """Result of model training and scoring."""
    scored_df: pd.DataFrame
    high_confidence_df: pd.DataFrame
    trained_model: LogisticRegression
    stats: ModelStats


class ModelEngine:
    """Pure business logic for training models and scoring candidate pairs."""
    
    def __init__(self):
        self.model_stats = {}
    
    def _heuristic_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return pseudo labels based on simple similarity thresholds."""
        required = {"company_sim", "domain_sim"}
        missing = required.difference(df.columns)
        if missing:
            raise FileNotFoundError(
                "Labels file not found and required similarity columns are missing"
            )

        # Use more reasonable thresholds for real data
        pos_mask = (df.get("company_sim", 0) >= 0.8) & (df.get("domain_sim", 0) >= 0.8)
        if "phone_exact" in df.columns:
            pos_mask |= df["phone_exact"] == 1  # OR condition for phone matches
        if "address_sim" in df.columns:
            pos_mask |= df["address_sim"] >= 0.9  # OR condition for address matches

        # Negative examples: low similarity across all features
        neg_mask = (df.get("company_sim", 1) <= 0.2) & (df.get("domain_sim", 1) <= 0.2)
        if "phone_exact" in df.columns:
            neg_mask &= df["phone_exact"] == 0
        if "address_sim" in df.columns:
            neg_mask &= df["address_sim"] <= 0.2

        pos_df = df[pos_mask]
        neg_df = df[neg_mask]

        if pos_df.empty or neg_df.empty:
            # If no clear examples, use middle range as examples
            print("No clear positive/negative examples found. Using middle range for training...")
            
            # Sort by combined similarity
            df_sorted = df.copy()
            df_sorted['combined_sim'] = (df_sorted.get('company_sim', 0) + 
                                       df_sorted.get('domain_sim', 0) + 
                                       df_sorted.get('address_sim', 0)) / 3
            df_sorted = df_sorted.sort_values('combined_sim', ascending=False)
            
            # For minimal data, create examples by duplicating with different labels
            if len(df_sorted) == 1:
                # Use the single example as both positive and negative (will lead to 50% probability)
                pos_df = df_sorted.copy()
                neg_df = df_sorted.copy()
            else:
                n_examples = min(10, len(df_sorted) // 2)  # Use up to 10 examples of each
                if n_examples == 0:
                    n_examples = 1  # At least 1 example
                
                pos_df = df_sorted.head(n_examples)
                neg_df = df_sorted.tail(n_examples)

        # Balance classes
        n = min(len(pos_df), len(neg_df))
        
        # Extract record IDs with proper data types
        pos_ids = pos_df.head(n)[["record_id_1", "record_id_2"]].copy()
        neg_ids = neg_df.head(n)[["record_id_1", "record_id_2"]].copy()
        
        # Ensure consistent data types with original dataframe
        pos_ids["record_id_1"] = pos_ids["record_id_1"].astype(df["record_id_1"].dtype)
        pos_ids["record_id_2"] = pos_ids["record_id_2"].astype(df["record_id_2"].dtype)
        neg_ids["record_id_1"] = neg_ids["record_id_1"].astype(df["record_id_1"].dtype)
        neg_ids["record_id_2"] = neg_ids["record_id_2"].astype(df["record_id_2"].dtype)
        
        labeled = pd.concat(
            [
                pos_ids.assign(label=1),
                neg_ids.assign(label=0),
            ],
            ignore_index=True,
        )
        return labeled
    
    def _prepare_features(self, feat_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Prepare feature matrix for training."""
        # Identify feature columns (exclude ID, descriptive, and target columns)
        exclude_cols = [
            "record_id_1", "record_id_2", 
            "company_clean_1", "company_clean_2",
            "domain_clean_1", "domain_clean_2",
            "label",  # Exclude target variable
            "combined_sim"  # Exclude computed columns
        ]
        
        feature_cols = [col for col in feat_df.columns if col not in exclude_cols]
        
        # Ensure we have numeric features
        X = feat_df[feature_cols].fillna(0)  # Fill NaN with 0
        
        return X, feature_cols
    
    def train_model(self, feat_df: pd.DataFrame, labels_df: Optional[pd.DataFrame] = None) -> tuple[LogisticRegression, str, Dict[str, int]]:
        """
        Train a logistic regression model.
        
        Parameters
        ----------
        feat_df : pd.DataFrame
            Features dataframe
        labels_df : pd.DataFrame, optional
            Labels dataframe, if None will use heuristic labels
            
        Returns
        -------
        tuple
            (trained_model, labels_source, label_distribution)
        """
        # Get labels
        if labels_df is not None:
            labels_source = "manual"
            label_pairs = labels_df
        else:
            labels_source = "heuristic"
            label_pairs = self._heuristic_labels(feat_df)
        
        # Merge labels with features with proper type conversion
        # Ensure consistent data types before merging
        labels_df_clean = label_pairs[["record_id_1", "record_id_2", "label"]].copy()
        
        # Try to convert label IDs to match features IDs
        try:
            # If features uses int64, try to convert labels to int
            if feat_df["record_id_1"].dtype == 'int64':
                # Remove any prefix and convert to int if possible
                if labels_df_clean["record_id_1"].dtype == 'object':
                    # Try to extract numeric part (e.g., 'C33004' -> 33004)
                    labels_df_clean["record_id_1"] = labels_df_clean["record_id_1"].str.extract('(\d+)').astype('int64')
                    labels_df_clean["record_id_2"] = labels_df_clean["record_id_2"].str.extract('(\d+)').astype('int64')
                else:
                    labels_df_clean["record_id_1"] = labels_df_clean["record_id_1"].astype(feat_df["record_id_1"].dtype)
                    labels_df_clean["record_id_2"] = labels_df_clean["record_id_2"].astype(feat_df["record_id_2"].dtype)
            else:
                # If features uses objects, ensure labels are also objects
                labels_df_clean["record_id_1"] = labels_df_clean["record_id_1"].astype(str)
                labels_df_clean["record_id_2"] = labels_df_clean["record_id_2"].astype(str)
        except (ValueError, AttributeError) as e:
            # If conversion fails, fall back to heuristic labels
            print(f"Warning: Could not align label IDs with feature IDs ({e}). Using heuristic labels instead.")
            label_pairs = self._heuristic_labels(feat_df)
            labels_source = "heuristic (ID mismatch)"
            labels_df_clean = label_pairs[["record_id_1", "record_id_2", "label"]].copy()
        
        train_df = feat_df.merge(
            labels_df_clean,
            on=["record_id_1", "record_id_2"],
            how="inner"
        )
        
        if len(train_df) == 0:
            # No overlap with provided labels, fall back to heuristic
            print("No overlap between features and labels. Falling back to heuristic labels.")
            label_pairs = self._heuristic_labels(feat_df)
            labels_source = "heuristic (no overlap)"
            labels_df_clean = label_pairs[["record_id_1", "record_id_2", "label"]].copy()
            train_df = feat_df.merge(labels_df_clean, on=["record_id_1", "record_id_2"], how="inner")
            
            if len(train_df) == 0:
                raise ValueError("No training data after merging features and heuristic labels")
        
        
        # Prepare features
        X, feature_cols = self._prepare_features(train_df)
        y = train_df["label"]
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        
        # Calculate label distribution
        label_distribution = {
            "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()),
            "total": len(y)
        }
        
        return model, labels_source, label_distribution
    
    def score_pairs(self, feat_df: pd.DataFrame, model: LogisticRegression, config: ModelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Score all pairs using the trained model.
        
        Parameters
        ----------
        feat_df : pd.DataFrame
            Features dataframe
        model : LogisticRegression
            Trained model
        config : ModelConfig
            Configuration with threshold
            
        Returns
        -------
        tuple
            (scored_dataframe, high_confidence_dataframe)
        """
        # Prepare features for scoring
        X, feature_cols = self._prepare_features(feat_df)
        
        # Score all pairs
        probabilities = model.predict_proba(X)[:, 1]  # Probability of being duplicate
        
        # Add scores to dataframe
        scored_df = feat_df.copy()
        scored_df["prob"] = probabilities
        
        # Filter high confidence pairs
        high_confidence_df = scored_df[scored_df["prob"] >= config.confidence_threshold].copy()
        
        return scored_df, high_confidence_df
    
    def process(self, feat_df: pd.DataFrame, labels_df: Optional[pd.DataFrame], config: ModelConfig) -> ModelResult:
        """
        Process the features to train model and score pairs.
        
        This is pure business logic - no file I/O, no terminal output.
        """
        input_pairs = len(feat_df)
        
        # Train model
        model, labels_source, label_distribution = self.train_model(feat_df, labels_df)
        
        # Score all pairs
        scored_df, high_confidence_df = self.score_pairs(feat_df, model, config)
        
        # Calculate model score (if we have training data)
        try:
            if labels_df is not None:
                # Use provided labels for scoring with proper type conversion
                labels_df_clean = labels_df[["record_id_1", "record_id_2", "label"]].copy()
                
                # Try to convert label IDs to match features IDs
                try:
                    if feat_df["record_id_1"].dtype == 'int64':
                        if labels_df_clean["record_id_1"].dtype == 'object':
                            # Try to extract numeric part
                            labels_df_clean["record_id_1"] = labels_df_clean["record_id_1"].str.extract('(\d+)').astype('int64')
                            labels_df_clean["record_id_2"] = labels_df_clean["record_id_2"].str.extract('(\d+)').astype('int64')
                        else:
                            labels_df_clean["record_id_1"] = labels_df_clean["record_id_1"].astype(feat_df["record_id_1"].dtype)
                            labels_df_clean["record_id_2"] = labels_df_clean["record_id_2"].astype(feat_df["record_id_2"].dtype)
                    else:
                        labels_df_clean["record_id_1"] = labels_df_clean["record_id_1"].astype(str)
                        labels_df_clean["record_id_2"] = labels_df_clean["record_id_2"].astype(str)
                except (ValueError, AttributeError):
                    # If conversion fails, skip scoring
                    model_score = 0.0
                    print("Warning: Could not calculate model score due to ID mismatch.")
                
                train_df = feat_df.merge(
                    labels_df_clean,
                    on=["record_id_1", "record_id_2"],
                    how="inner"
                )
                X, _ = self._prepare_features(train_df)
                y = train_df["label"]
                model_score = float(model.score(X, y))
            else:
                model_score = None
        except Exception:
            model_score = None
        
        # Prepare feature list
        X, feature_cols = self._prepare_features(feat_df)
        
        stats = ModelStats(
            input_pairs=input_pairs,
            features_used=feature_cols,
            labels_source=labels_source,
            training_pairs=label_distribution["total"],
            high_confidence_pairs=len(high_confidence_df),
            model_score=model_score,
            label_distribution=label_distribution
        )
        
        return ModelResult(
            scored_df=scored_df,
            high_confidence_df=high_confidence_df,
            trained_model=model,
            stats=stats
        )
    
    def train_and_score(
        self,
        features_path: str,
        labels_path: str,
        model_path: str,
        duplicates_path: str,
        confidence_threshold: float = 0.9
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Train a model and score candidate pairs with file I/O.
        
        Args:
            features_path: Path to features CSV
            labels_path: Path to labels CSV (optional)
            model_path: Path to save trained model
            duplicates_path: Path to save high-confidence pairs
            confidence_threshold: Minimum probability for high-confidence pairs
            
        Returns:
            Tuple of (scored_dataframe, statistics_dict)
        """
        # Load features
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        feat_df = pd.read_csv(features_path)
        
        # Load labels if available
        labels_df = None
        if os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
            # Keep only required columns
            labels_df = labels_df[["record_id_1", "record_id_2", "label"]]
        
        # Configure training
        config = ModelConfig(confidence_threshold=confidence_threshold)
        
        # Process with business logic
        result = self.process(feat_df, labels_df, config)
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(result.trained_model, model_path)
        
        # Save high-confidence pairs
        os.makedirs(os.path.dirname(duplicates_path), exist_ok=True)
        result.high_confidence_df.to_csv(duplicates_path, index=False)
        
        # Calculate additional statistics
        probabilities = result.scored_df["prob"]
        
        # Create comprehensive stats dictionary
        self.model_stats = {
            "input_pairs": result.stats.input_pairs,
            "training_pairs": result.stats.training_pairs,
            "high_confidence_pairs": result.stats.high_confidence_pairs,
            "label_stats": {
                "source": result.stats.labels_source,
                "total_labels": result.stats.label_distribution["total"],
                "positive_labels": result.stats.label_distribution["positive"],
                "negative_labels": result.stats.label_distribution["negative"],
                "label_ratio": result.stats.label_distribution["positive"] / result.stats.label_distribution["total"] if result.stats.label_distribution["total"] > 0 else 0
            },
            "model_stats": {
                "features": result.stats.features_used,
                "coefficients": dict(zip(result.stats.features_used, result.trained_model.coef_[0].tolist())),
                "intercept": float(result.trained_model.intercept_[0])
            },
            "prediction_stats": {
                "mean_prob": float(probabilities.mean()),
                "prob_dist": {
                    "p90": float(probabilities.quantile(0.9)),
                    "p95": float(probabilities.quantile(0.95)),
                    "p99": float(probabilities.quantile(0.99))
                }
            }
        }
        
        return result.scored_df, self.model_stats
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get simplified summary statistics for logging."""
        if not self.model_stats:
            return {}
        
        return {
            "input_pairs": self.model_stats.get("input_pairs", 0),
            "training_pairs": self.model_stats.get("training_pairs", 0),
            "high_confidence_pairs": self.model_stats.get("high_confidence_pairs", 0),
            "labels_source": self.model_stats.get("label_stats", {}).get("source", "unknown"),
            "mean_probability": self.model_stats.get("prediction_stats", {}).get("mean_prob", 0.0)
        }
