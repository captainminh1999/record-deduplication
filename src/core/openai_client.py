"""
OpenAI client wrapper with error handling and statistics tracking.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Dict, Any, List, Tuple

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Optional OpenAI dependency
try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore

from .openai_types import OpenAIStats


class OpenAIClient:
    """Wrapper for OpenAI client with error handling and statistics."""
    
    def __init__(self):
        self.client = None
        self._init_client()
    
    def _init_client(self) -> None:
        """Initialize OpenAI client."""
        if OpenAI is None:
            raise RuntimeError(
                "openai package is not installed. Install 'openai' to enable integration."
            )
        
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API key is not configured. Set 'OPENAI_API_KEY' or 'OPENAI_KEY' environment variable."
            )
        
        self.client = OpenAI(api_key=api_key)
    
    def init_stats(self) -> Dict[str, Any]:
        """Initialize API call statistics."""
        return {
            "calls": 0,
            "tokens": {
                "prompt": 0,
                "completion": 0,
                "total": 0
            },
            "durations": [],
            "errors": defaultdict(int),
            "costs": {
                "total": 0.0
            }
        }
    
    def update_stats(self, stats: Dict[str, Any], duration: float, response: Any = None, error: str = None) -> None:
        """Update API call statistics."""
        stats["calls"] += 1
        stats["durations"].append(duration)
        
        if error:
            stats["errors"][error] += 1
        elif hasattr(response, "usage"):
            stats["tokens"]["prompt"] += response.usage.prompt_tokens
            stats["tokens"]["completion"] += response.usage.completion_tokens
            stats["tokens"]["total"] += response.usage.total_tokens
            
            # Calculate costs based on gpt-4o-mini pricing
            input_price_per_1m = 0.15   # $0.15 per 1M input tokens
            output_price_per_1m = 0.60  # $0.60 per 1M output tokens
            
            input_cost = (response.usage.prompt_tokens / 1_000_000) * input_price_per_1m
            output_cost = (response.usage.completion_tokens / 1_000_000) * output_price_per_1m
            
            stats["costs"]["total"] += input_cost + output_cost
    
    def make_request(
        self, 
        messages: List[Dict[str, str]], 
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1000
    ) -> Tuple[str, Dict[str, Any]]:
        """Make a single OpenAI API request with error handling."""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            duration = time.time() - start_time
            
            # Extract response data
            content = response.choices[0].message.content or ""
            
            # Track usage
            usage_stats = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "duration": duration
            }
            
            return content, usage_stats
            
        except Exception as e:
            duration = time.time() - start_time
            error_stats = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "duration": duration,
                "error": str(e)
            }
            raise RuntimeError(f"OpenAI API error: {e}") from e
    
    def convert_to_stats(self, api_stats: Dict[str, Any]) -> OpenAIStats:
        """Convert internal stats dict to OpenAIStats dataclass."""
        return OpenAIStats(
            total_calls=api_stats["calls"],
            total_tokens=api_stats["tokens"],
            total_cost=api_stats["costs"]["total"],
            durations=api_stats["durations"],
            errors=dict(api_stats["errors"]),
            successful_calls=api_stats["calls"] - sum(api_stats["errors"].values()),
            failed_calls=sum(api_stats["errors"].values())
        )
