"""
OpenAI-powered text translation functionality.
"""

from __future__ import annotations

import time
from typing import Iterable, List

from .openai_client import OpenAIClient
from .openai_types import OpenAIConfig, TranslationResult


class OpenAITranslator:
    """Handles OpenAI-powered text translation."""
    
    def __init__(self, client: OpenAIClient):
        self.client = client
    
    def translate_to_english(self, texts: Iterable[str], config: OpenAIConfig) -> TranslationResult:
        """
        Translate company names to English using OpenAI.
        
        Pure business logic - no file I/O or terminal output.
        """
        texts_list = list(texts)
        results: List[str] = []
        api_stats = self.client.init_stats()
        translation_stats = {
            "total": 0,
            "non_latin": 0,
            "changed": 0,
            "unchanged": 0,
            "failed": 0
        }
        
        for text in texts_list:
            translation_stats["total"] += 1
            call_start = time.time()
            
            # Check if translation is needed
            needs_translation = any(ord(c) > 127 for c in text)
            if needs_translation:
                translation_stats["non_latin"] += 1
                try:
                    prompt = (
                        "Translate the following company name to English using Latin characters only: "
                        f"{text}"
                    )
                    content, usage = self.client.make_request(
                        messages=[{"role": "user", "content": prompt}],
                        model=config.model,
                        temperature=0.0,
                        max_tokens=60
                    )
                    call_duration = time.time() - call_start
                    self.client.update_stats(api_stats, call_duration, type('Response', (), {'usage': type('Usage', (), usage)})())
                    
                    translated = content.strip()
                    if translated.lower() != text.lower():
                        translation_stats["changed"] += 1
                    else:
                        translation_stats["unchanged"] += 1
                    results.append(translated)
                except Exception as e:
                    call_duration = time.time() - call_start
                    self.client.update_stats(api_stats, call_duration, None, str(type(e).__name__))
                    translation_stats["failed"] += 1
                    results.append(text)
            else:
                translation_stats["unchanged"] += 1
                results.append(text)
        
        # Convert to OpenAIStats format
        stats = self.client.convert_to_stats(api_stats)
        
        return TranslationResult(translations=results, stats=stats)
