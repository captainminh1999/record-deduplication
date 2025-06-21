"""Optional step of the 10-step deduplication pipeline: GPT assistance.

This module demonstrates how the OpenAI API might be used to suggest
merges or normalise data. It is intentionally minimal and optional.
"""

from __future__ import annotations

from typing import Iterable, List

# The OpenAI package is optional and may not be installed by default
try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openai = None


def translate_to_english(texts: Iterable[str], model: str = "gpt-4o-mini") -> List[str]:
    """Translate a sequence of company names to English using the OpenAI API.

    Parameters
    ----------
    texts:
        An iterable of raw text values (e.g. company names) to translate.
    model:
        Chat completion model to use.

    Returns
    -------
    List[str]
        The translated strings in the same order as ``texts``.
    """
    if openai is None:
        raise RuntimeError(
            "openai package is not installed. Install 'openai' to enable integration."
        )

    if not getattr(openai, "api_key", None):
        raise RuntimeError(
            "OpenAI API key is not configured. Set 'openai.api_key' or the 'OPENAI_API_KEY' environment variable."
        )

    results: List[str] = []
    for text in texts:
        prompt = (
            "Translate the following company name to English using Latin characters only: "
            f"{text}"
        )
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        translated = resp.choices[0].message["content"].strip()
        results.append(translated)

    return results

def main() -> None:
    """Example helper that could call the OpenAI API.

    TODO:
        * check that ``openai`` is available
        * formulate a prompt describing the duplicate records
        * send the request and parse the response
    """
    if openai is None:
        raise RuntimeError(
            "openai package is not installed. Install 'openai' to enable integration."
        )

    if not getattr(openai, "api_key", None):
        raise RuntimeError(
            "OpenAI API key is not configured. Set 'openai.api_key' or the 'OPENAI_API_KEY' environment variable."
        )

    # TODO: formulate prompt and send request
    _ = translate_to_english(["Example"])


if __name__ == "__main__":  # pragma: no cover - sanity run
    print("\u23e9 started openai integration")
    main()
