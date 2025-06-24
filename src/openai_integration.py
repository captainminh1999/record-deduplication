"""Optional step of the 10-step deduplication pipeline: GPT assistance.

This module demonstrates how the OpenAI API might be used to suggest
merges or normalise data. It is intentionally minimal and optional.
"""

from __future__ import annotations

from typing import Iterable, List, Dict, Any, cast

import json
import os
import time

import click
import pandas as pd
from .utils import log_run, LOG_PATH

# The OpenAI package is optional and may not be installed by default
try:
    import openai  # type: ignore
    if not getattr(openai, "api_key", None):
        openai.api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
except Exception:  # pragma: no cover - optional dependency
    openai = None

# Default chat model used across this module
DEFAULT_MODEL = "gpt-4o-mini"


def translate_to_english(
    texts: Iterable[str], model: str = DEFAULT_MODEL
) -> List[str]:
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
            "OpenAI API key is not configured. Set 'openai.api_key' or the 'OPENAI_KEY' environment variable."
        )

    results: List[str] = []
    for text in texts:
        prompt = (
            "Translate the following company name to English using Latin characters only: "
            f"{text}"
        )
        resp = cast(Any, openai).ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        translated = resp.choices[0].message["content"].strip()
        results.append(translated)

    return results


def _check_openai() -> None:
    """Ensure the OpenAI dependency and API key are available."""
    if openai is None:
        raise RuntimeError(
            "openai package is not installed. Install 'openai' to enable integration."
        )
    if not getattr(openai, "api_key", None):
        raise RuntimeError(
            "OpenAI API key is not configured. Set 'openai.api_key' or the 'OPENAI_KEY' environment variable."
        )


def _parse_gpt_response(answer: str) -> Dict[str, Any]:
    """Parse the GPT answer into structured fields."""
    try:
        parsed = json.loads(answer)
        same_org = bool(parsed.get("same_org"))
        primary_name = str(parsed.get("primary_name", ""))
        canonical_record = parsed.get("canonical_record", {})
        if not isinstance(canonical_record, dict):
            canonical_record = {}
        excluded = parsed.get("excluded_ids", [])
        if not isinstance(excluded, list):
            excluded = []
    except Exception:
        same_org = False
        primary_name = ""
        canonical_record = {}
        excluded = []
    return {
        "same_org": same_org,
        "primary_name": primary_name,
        "canonical_record": canonical_record,
        "excluded_ids": excluded,
    }


def main(
    clusters_path: str = "data/outputs/clusters.csv",
    review_path: str = "data/outputs/gpt_review.json",
    openai_model: str = DEFAULT_MODEL,
    log_path: str = LOG_PATH,
) -> None:
    """Review DBSCAN clusters with GPT for validation."""

    start_time = time.time()

    if not os.path.exists(clusters_path):
        raise FileNotFoundError(f"Clusters file not found: {clusters_path}")

    clusters_df = pd.read_csv(clusters_path)
    cluster_count = clusters_df["cluster"].nunique()

    _check_openai()

    # eps, min_samples and prompt wording can be tuned later

    results: List[Dict[str, Any]] = []
    for cluster_id, group in clusters_df.groupby("cluster"):
        cluster_id_int = int(cast(Any, cluster_id))
        # Skip noise (-1) and singleton clusters; this threshold can be tuned later
        if cluster_id_int == -1 or len(group) <= 1:
            continue

        lines = [f"Cluster {cluster_id_int} contains {len(group)} records:"]
        for _, row in group.iterrows():
            lines.append(
                f"  - ID {int(row['record_id'])}: {row.get('company_clean', '')}, {row.get('domain_clean', '')}, {row.get('phone_clean', '')}, {row.get('address_clean', '')}"
            )
        lines.append("1) Take some times to think. Do these all refer to the same organization?")
        lines.append("2) If yes, what should be the primary organization name?")
        lines.append("3) Please produce a single canonical record (merge or pick fields).")
        lines.append("If any record does NOT belong, list its ID.")
        prompt_text = "\n".join(lines)

        resp = cast(Any, openai).ChatCompletion.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip()

        results.append(
            {
                "cluster_id": cluster_id_int,
                "record_ids": [int(r) for r in group["record_id"].tolist()],
                "gpt_response": answer,
            }
        )

    os.makedirs(os.path.dirname(review_path), exist_ok=True)
    with open(review_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"Read {cluster_count} clusters from {clusters_path}.")
    print(
        f"Queried OpenAI for {len(results)} clusters, saved JSON to {review_path}."
    )

    end_time = time.time()
    log_run("openai_integration", start_time, end_time, len(results), log_path=log_path)


@click.command()
@click.option(
    "--clusters-path", default="data/outputs/clusters.csv", show_default=True
)
@click.option(
    "--review-path", default="data/outputs/gpt_review.json", show_default=True
)
@click.option("--openai-model", default=DEFAULT_MODEL, show_default=True)
@click.option("--log-path", default=LOG_PATH, show_default=True)
def cli(
    clusters_path: str,
    review_path: str,
    openai_model: str,
    log_path: str,
) -> None:
    """CLI wrapper for :func:`main`."""

    main(clusters_path, review_path, openai_model, log_path)


if __name__ == "__main__":  # pragma: no cover - sanity run
    print("\u23e9 started openai integration")
    cli()
