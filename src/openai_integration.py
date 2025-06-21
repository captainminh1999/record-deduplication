"""Optional step of the 10-step deduplication pipeline: GPT assistance.

This module demonstrates how the OpenAI API might be used to suggest
merges or normalise data. It is intentionally minimal and optional.
"""

from __future__ import annotations

# The OpenAI package is optional and may not be installed by default
try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openai = None


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
    pass


if __name__ == "__main__":  # pragma: no cover - sanity run
    print("\u23e9 started openai integration")
    main()
