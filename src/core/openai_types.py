"""
Data types and configurations for OpenAI operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

# Default model
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI operations."""
    model: str = DEFAULT_MODEL
    max_workers: int = 10
    batch_size: int = 10
    similarity_threshold: float = 0.6
    confidence_threshold: float = 0.7
    min_cluster_size: int = 2
    max_cluster_size: Optional[int] = None
    exclude_noise: bool = True
    sample_large_clusters: Optional[int] = None
    exclude_clusters: Tuple[int, ...] = ()
    sample_size: Optional[int] = None


@dataclass
class OpenAIStats:
    """Statistics from OpenAI operations."""
    total_calls: int
    total_tokens: Dict[str, int]
    total_cost: float
    durations: List[float]
    errors: Dict[str, int]
    successful_calls: int
    failed_calls: int


@dataclass
class TranslationResult:
    """Result of translation operation."""
    translations: List[str]
    stats: OpenAIStats


@dataclass
class ClusterReviewResult:
    """Result of cluster review operation."""
    cluster_reviews: List[Dict[str, Any]]
    stats: OpenAIStats


@dataclass
class DeduplicationResult:
    """Result of deduplication operation."""
    unique_records: Any  # pd.DataFrame
    analysis: Dict[str, Any]
    stats: OpenAIStats
