"""Adaptive Threshold Calculator

Calculates depth-aware similarity thresholds for hierarchical clustering.
"""

from typing import Dict, Any
import numpy as np


class AdaptiveThresholdCalculator:
    """Calculate adaptive similarity thresholds based on clustering depth."""
    
    def __init__(self):
        self.threshold_history = {}
    
    def calculate_adaptive_threshold(
        self,
        current_depth: int,
        max_depth: int,
        base_threshold: float,
        high_similarity_threshold: float
    ) -> float:
        """
        Calculate adaptive similarity threshold based on current depth.
        
        Strategy:
        - Depth 1-2: Use strict thresholds (close to high_similarity_threshold)
        - Depth 3+: Gradually increase threshold allowing more subdivision
        - Always preserve high-similarity connections
        
        Args:
            current_depth: Current subdivision depth (0-indexed)
            max_depth: Maximum allowed depth
            base_threshold: Base similarity threshold
            high_similarity_threshold: Always preserve threshold
        
        Returns:
            Adaptive threshold for current depth
        """
        depth_1_indexed = current_depth + 1
        
        # Progressive threshold strategy
        if depth_1_indexed <= 2:
            # Early depths: Be very strict, only allow subdivision if we can preserve tight connections
            # Use 30% of the range between high and base threshold
            progression_factor = 0.3
        elif depth_1_indexed <= 4:
            # Mid depths: Moderate threshold
            # Use 60% of the range
            progression_factor = 0.6
        else:
            # Deep levels: Allow more subdivision but still preserve high-similarity
            # Use full range
            progression_factor = 1.0
        
        # Calculate threshold
        threshold_range = base_threshold - high_similarity_threshold
        threshold = high_similarity_threshold + (threshold_range * progression_factor)
        
        # Ensure we never go below high_similarity_threshold for preservation
        adaptive_threshold = max(threshold, high_similarity_threshold)
        
        # Store for analysis
        self.threshold_history[depth_1_indexed] = {
            "threshold": adaptive_threshold,
            "progression_factor": progression_factor,
            "base_threshold": base_threshold,
            "high_similarity_threshold": high_similarity_threshold
        }
        
        return adaptive_threshold
    
    def get_threshold_progression(self, max_depth: int, base_threshold: float, high_similarity_threshold: float) -> Dict[int, float]:
        """Get the full threshold progression for visualization."""
        progression = {}
        for depth in range(max_depth):
            progression[depth + 1] = self.calculate_adaptive_threshold(
                depth, max_depth, base_threshold, high_similarity_threshold
            )
        return progression
    
    def get_threshold_stats(self) -> Dict[str, Any]:
        """Get statistics about threshold usage."""
        return {
            "thresholds_used": len(self.threshold_history),
            "threshold_history": self.threshold_history.copy(),
            "threshold_range": {
                "min": min(data["threshold"] for data in self.threshold_history.values()) if self.threshold_history else 0,
                "max": max(data["threshold"] for data in self.threshold_history.values()) if self.threshold_history else 0
            }
        }
