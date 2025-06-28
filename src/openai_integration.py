"""
Optional: GPT Integration for Data Normalization and Cluster Review

Provides optional integration with OpenAI's GPT for translating company names and reviewing clusters. Not required for the core pipeline. See README for details.
"""

from __future__ import annotations

from typing import Iterable, List, Dict, Any, cast

import json
import os
import sys
import time
import concurrent.futures

import click
import pandas as pd
from tqdm import tqdm

from .utils import log_run, LOG_PATH

# The OpenAI package is optional and may not be installed by default
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

client: Any = None  # type: ignore

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
    _check_openai()

    results: List[str] = []
    for text in texts:
        prompt = (
            "Translate the following company name to English using Latin characters only: "
            f"{text}"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        translated = resp.choices[0].message["content"].strip()
        results.append(translated)

    return results


def _check_openai() -> None:
    """Ensure the OpenAI dependency and API key are available."""
    if OpenAI is None:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "openai package is not installed. Install 'openai' to enable integration."
        )

    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API key is not configured. Set 'OPENAI_API_KEY' or 'OPENAI_KEY' environment variable."
            )
        client = OpenAI(api_key=api_key)


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
    max_workers: int = 10,
) -> None:
    """Review DBSCAN clusters with GPT for validation."""

    start_time = time.time()

    if not os.path.exists(clusters_path):
        raise FileNotFoundError(f"Clusters file not found: {clusters_path}")

    clusters_df = pd.read_csv(clusters_path)
    # Ensure record_id is a column, not an index
    if "record_id" not in clusters_df.columns and clusters_df.index.name == "record_id":
        clusters_df = clusters_df.reset_index()
    cluster_count = clusters_df["cluster"].nunique()

    _check_openai()

    # eps, min_samples and prompt wording can be tuned later

    results: List[Dict[str, Any]] = []
    clusters = list(clusters_df.groupby("cluster"))

    def process_cluster(args):
        cluster_id, group = args
        # Always reset index to ensure 'record_id' is a column
        group = group.reset_index(drop=False)
        if "record_id" not in group.columns:
            raise KeyError(f"'record_id' column missing in group DataFrame. Columns: {group.columns.tolist()}, Index name: {group.index.name}, First rows: {group.head().to_dict()}")
        cluster_id_int = int(cast(Any, cluster_id))
        if cluster_id_int == -1 or len(group) <= 1:
            return None
        lines = [f"Cluster {cluster_id_int} contains {len(group)} records:"]
        for _, row in group.iterrows():
            lines.append(
                f"  - ID {row['record_id']}: {row.get('company_clean', '')}, {row.get('domain_clean', '')}"
            )
        lines.append(
            "Take some times to analyze the records above. There may be more than one group of duplicates or companies that are subsidiaries/brands under a parent entity in this cluster."
        )
        lines.append(
            "For each group of records that are duplicates or subsidiaries/brands of the same organization, return a JSON object with these fields (all required and not blank):"
        )
        lines.append('- "primary_organization": the canonical name for the group (required, not blank). If all companies are subsidiaries, use the parent company name as the canonical name, based on your knowledge.')
        lines.append('- "canonical_domains": a list of all domains belonging to this group (required, not blank)')
        lines.append('- "record_ids": a list of record IDs belonging to this organization (required, not blank)')
        lines.append('- "confidence": a number from 0 to 1 indicating your confidence in this grouping (required)')
        lines.append(
            "Only include a group if there are actual duplicates or subsidiaries/brands that should be grouped together. "
            "If all records are unique and unrelated, return an empty JSON array []."
        )
        lines.append(
            "Return a JSON array of these objects. Only output valid JSON, no explanations or formatting, and do not include any extra fields. If you cannot provide all required fields for a group, omit that group from the output."
        )
        prompt_text = "\n".join(lines)
        try:
            resp = client.chat.completions.create(
                model=openai_model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0,
                max_tokens=2000
            )
            answer = resp.choices[0].message.content.strip()
            try:
                canonical_groups = json.loads(answer)
                # Filter to only keep requested fields
                filtered_groups = []
                for canonical_group in canonical_groups:
                    filtered_group = {
                        "primary_organization": canonical_group.get("primary_organization", ""),
                        "canonical_domains": canonical_group.get("canonical_domains", []),
                        "record_ids": canonical_group.get("record_ids", []),
                        "confidence": canonical_group.get("confidence", None),
                    }
                    filtered_groups.append(filtered_group)
            except Exception as e:
                filtered_groups = []
            return {
                "cluster_id": cluster_id_int,
                "record_ids": [str(r) for r in group["record_id"].tolist()],
                "canonical_groups": filtered_groups,
                "raw_response": answer,
            }
        except Exception as e:
            return {
                "cluster_id": cluster_id_int,
                "record_ids": [str(r) for r in group["record_id"].tolist()] if "record_id" in group.columns else [],
                "canonical_groups": [],
                "raw_response": f"ERROR: {str(e)}",
            }

    with tqdm(total=len(clusters), desc="Processing clusters", file=sys.stdout) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_cluster, c): c for c in clusters}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                pbar.update(1)

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
@click.option("--max-workers", default=10, show_default=True, help="Number of parallel OpenAI requests to run.")
def cli(
    clusters_path: str,
    review_path: str,
    openai_model: str,
    log_path: str,
    max_workers: int,
) -> None:
    """CLI wrapper for :func:`main`."""

    global client
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Install 'openai' to enable integration.")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    main(clusters_path, review_path, openai_model, log_path, max_workers)


if __name__ == "__main__":  # pragma: no cover - sanity run
    print("\u23e9 started openai integration")
    cli()
