"""Optional step of the 10-step deduplication pipeline: GPT assistance.

This module demonstrates how the OpenAI API might be used to suggest
merges or normalise data. It is intentionally minimal and optional.
"""

from __future__ import annotations

from typing import Iterable, List, Dict, Any

import json
import os
import time

import click
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from .utils import log_run, LOG_PATH

# The OpenAI package is optional and may not be installed by default
try:
    import openai  # type: ignore
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


def _check_openai() -> None:
    """Ensure the OpenAI dependency and API key are available."""
    if openai is None:
        raise RuntimeError(
            "openai package is not installed. Install 'openai' to enable integration."
        )
    if not getattr(openai, "api_key", None):
        raise RuntimeError(
            "OpenAI API key is not configured. Set 'openai.api_key' or the 'OPENAI_API_KEY' environment variable."
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
    features_path: str = "data/outputs/features.csv",
    cleaned_path: str = "data/outputs/cleaned.csv",
    eps: float = 0.5,
    min_samples: int = 2,
    report_path: str = "data/outputs/gpt_cluster_report.json",
    openai_model: str = DEFAULT_MODEL,
    log_path: str = LOG_PATH,
) -> None:
    """Run a GPT-assisted cluster sanity check over deduplication results."""

    start_time = time.time()

    _check_openai()

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned data not found: {cleaned_path}")

    feats = pd.read_csv(features_path)
    cleaned = pd.read_csv(cleaned_path).set_index("record_id")

    similarity_cols = [
        c for c in feats.columns if c.endswith("_sim") or c == "phone_exact"
    ]
    if not similarity_cols:
        raise ValueError("No similarity columns found in features file")

    ids = pd.unique(feats[["record_id_1", "record_id_2"]].values.ravel())
    id_to_idx = {rid: i for i, rid in enumerate(ids)}
    dist = np.ones((len(ids), len(ids)))
    np.fill_diagonal(dist, 0)
    for _, row in feats.iterrows():
        i = id_to_idx[row["record_id_1"]]
        j = id_to_idx[row["record_id_2"]]
        vals = [float(row[c]) for c in similarity_cols if c in row]
        d = 1 - (sum(vals) / len(vals))
        dist[i, j] = d
        dist[j, i] = d

    # Eps and min_samples can be tuned later
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = clustering.fit_predict(dist)

    results: List[Dict[str, Any]] = []
    for label in sorted(set(labels)):
        if label < 0:
            continue
        members = [ids[i] for i, lbl in enumerate(labels) if lbl == label]
        if len(members) <= 1:
            continue

        details = []
        for rid in members:
            rec = cleaned.loc[rid]
            details.append(
                {
                    "id": int(rid),
                    "company": rec.get("company_clean", ""),
                    "domain": rec.get("domain_clean", ""),
                    "phone": rec.get("phone_clean", ""),
                    "address": rec.get("address_clean", ""),
                }
            )

        lines = [f"Cluster {label} contains {len(members)} records:"]
        for d in details:
            lines.append(
                f"  - ID {d['id']}: {d['company']}, {d['domain']}, {d['phone']}, {d['address']}"
            )
        lines.append("1) Do these all refer to the same organization?")
        lines.append(
            "2) If yes, what should be the **primary organization name**?"
        )
        lines.append(
            "3) Please provide a **single canonical record** (choose or merge the fields above)."
        )
        lines.append(
            "If any record does NOT belong in this cluster, please list its ID."
        )
        prompt_text = "\n".join(lines)

        resp = openai.ChatCompletion.create(
            model=openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a data quality assistant.",
                },
                {"role": "user", "content": prompt_text},
            ],
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip()
        parsed = _parse_gpt_response(answer)
        results.append(
            {
                "cluster": int(label),
                "records": [int(r) for r in members],
                "same_org": parsed["same_org"],
                "primary_name": parsed["primary_name"],
                "canonical_record": parsed["canonical_record"],
                "excluded_ids": parsed["excluded_ids"],
                "gpt_response": answer,
            }
        )

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(
        f"Processed {len(results)} clusters and saved report to {report_path}"
    )

    end_time = time.time()
    log_run(
        "openai_integration",
        start_time,
        end_time,
        len(results),
        log_path=log_path,
    )


@click.command()
@click.option(
    "--features-path", default="data/outputs/features.csv", show_default=True
)
@click.option(
    "--cleaned-path", default="data/outputs/cleaned.csv", show_default=True
)
@click.option("--eps", default=0.5, show_default=True)
@click.option("--min-samples", default=2, show_default=True)
@click.option(
    "--report-path",
    default="data/outputs/gpt_cluster_report.json",
    show_default=True,
)
@click.option("--openai-model", default=DEFAULT_MODEL, show_default=True)
@click.option("--log-path", default=LOG_PATH, show_default=True)
def cli(
    features_path: str,
    cleaned_path: str,
    eps: float,
    min_samples: int,
    report_path: str,
    openai_model: str,
    log_path: str,
) -> None:
    """CLI wrapper for :func:`main`."""

    main(
        features_path,
        cleaned_path,
        eps,
        min_samples,
        report_path,
        openai_model,
        log_path,
    )


if __name__ == "__main__":  # pragma: no cover - sanity run
    print("\u23e9 started openai integration")
    cli()
