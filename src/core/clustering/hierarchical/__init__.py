"""Hierarchical Clustering Package

Modular components for hierarchical clustering with adaptive subdivision.
"""

from .core_clusterer import HierarchicalClusterer
from .connectivity_manager import ConnectivityManager
from .adaptive_threshold import AdaptiveThresholdCalculator
from .subdivision_engine_v2 import SubdivisionEngineV2

__all__ = [
    'HierarchicalClusterer',
    'ConnectivityManager', 
    'AdaptiveThresholdCalculator',
    'SubdivisionEngineV2'
]
