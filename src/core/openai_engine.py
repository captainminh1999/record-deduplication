"""
Main OpenAI engine - orchestrates OpenAI-powered operations.

Pure business logic with no I/O or terminal output.
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, Any, List, Optional, Iterable

from .openai_client import OpenAIClient
from .openai_translator import OpenAITranslator
from .openai_cluster_reviewer import OpenAIClusterReviewer
from .openai_deduplicator import OpenAIDeduplicator
from .openai_types import OpenAIConfig, TranslationResult, ClusterReviewResult, DeduplicationResult


class OpenAIEngine:
    """Main OpenAI engine that orchestrates different OpenAI-powered operations."""
    
    def __init__(self):
        self.client = OpenAIClient()
        self.translator = OpenAITranslator(self.client)
        self.cluster_reviewer = OpenAIClusterReviewer(self.client)
        self.deduplicator = OpenAIDeduplicator(self.client)
    
    def translate_to_english(self, texts: Iterable[str], config: OpenAIConfig) -> TranslationResult:
        """Translate company names to English using OpenAI."""
        return self.translator.translate_to_english(texts, config)
    
    def review_clusters(self, clusters_df: pd.DataFrame, config: OpenAIConfig) -> ClusterReviewResult:
        """Review DBSCAN clusters with GPT for validation."""
        return self.cluster_reviewer.review_clusters(clusters_df, config)
    
    def deduplicate_records(
        self,
        features_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        config: OpenAIConfig,
        sample_size: Optional[int] = None
    ) -> DeduplicationResult:
        """Perform AI-powered deduplication using similarity features."""
        return self.deduplicator.deduplicate_records(features_df, cleaned_df, config, sample_size)
