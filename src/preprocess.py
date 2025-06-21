"""Step 1 of the 10-step deduplication pipeline: data cleaning.

This module prepares the raw spreadsheet for later stages by normalising
fields and writing the cleaned output to ``data/outputs/cleaned.csv``.
"""

from __future__ import annotations

import os
import re
import unicodedata
import time

import pandas as pd

from .openai_integration import translate_to_english
from .utils import log_run, clear_all_data


def _normalize_name(name: str) -> str:
    """Return a basic normalised representation of a company name."""
    text = str(name) if pd.notnull(name) else ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.strip().lower()


def _normalize_domain(domain: str) -> str:
    """Return a simplified representation of a website domain."""
    text = str(domain) if pd.notnull(domain) else ""
    text = text.strip().lower()
    text = re.sub(r"^https?://", "", text)
    text = re.sub(r"^www\.", "", text)
    text = text.rstrip("/")
    return text


def _normalize_phone(phone: str) -> str:
    """Return digits only from a phone number."""
    text = str(phone) if pd.notnull(phone) else ""
    digits = re.sub(r"\D", "", text)
    return digits


def _clean_column_name(name: str) -> str:
    """Return a simplified column identifier.

    Lowercase and strip spaces and underscores so that variations like
    ``Record ID`` or ``record_id`` map to the same key.
    """
    return re.sub(r"[ _]+", "", name).lower()



def main(
    input_path: str = "data/your_spreadsheet.csv",
    output_path: str = "data/outputs/cleaned.csv",
    audit_path: str = "data/outputs/removed_rows.csv",
    use_openai: bool = False,
    openai_model: str = "gpt-4o-mini",
    log_path: str = "data/outputs/run_history.log",
    clear: bool = False,
) -> int:
    """Clean the raw spreadsheet and save a CSV.

    TODO:
        * load the source spreadsheet with :func:`pandas.read_csv` or
          ``read_excel``
        * normalise company, domain and phone columns
        * remove exact duplicate rows
        * write the result to ``output_path``
    """
    if clear:
        clear_all_data(os.path.dirname(output_path))

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input spreadsheet not found: {input_path}")

    df = pd.read_csv(input_path)

    col_map = {_clean_column_name(c): c for c in df.columns}

    if "recordid" not in col_map and "sysid" not in col_map:
        raise KeyError("Missing required column: record_id or sys_id")
    df["record_id"] = df[col_map.get("recordid", col_map.get("sysid"))]

    company_key = col_map.get("company") or col_map.get("name")
    if not company_key:
        raise KeyError("Missing required column: company")
    df["company"] = df[company_key]

    start_time = time.time()

    if use_openai:
        df["company_clean"] = translate_to_english(df["company"].tolist(), model=openai_model)
    else:
        df["company_clean"] = df["company"].map(_normalize_name)

    domain_key = col_map.get("domain") or col_map.get("website")
    if domain_key:
        domain_col = df[domain_key]
    else:
        domain_col = pd.Series("", index=df.index)
    df["domain_clean"] = domain_col.map(_normalize_domain)

    phone_key = col_map.get("companyphone") or col_map.get("phone")
    if phone_key:
        phone_col = df[phone_key]
    else:
        phone_col = pd.Series("", index=df.index)
    df["phone_clean"] = phone_col.map(_normalize_phone)

    df["combined_id"] = (
        df["company_clean"] + ";" + df["domain_clean"] + ";" + df["phone_clean"]
    )

    duplicates = df[df.duplicated(subset="combined_id", keep="first")]
    if not duplicates.empty:
        duplicates.assign(reason="duplicate combined_id").to_csv(audit_path, index=False)

    df = df.drop_duplicates(subset="combined_id", keep="first")

    df.to_csv(output_path, index=False)

    end_time = time.time()
    log_run("preprocess", start_time, end_time, len(df), log_path=log_path)

    return len(df)


if __name__ == "__main__":  # pragma: no cover - sanity run
    import argparse

    parser = argparse.ArgumentParser(description="Clean raw spreadsheet")
    parser.add_argument("--input-path", default="data/your_spreadsheet.csv")
    parser.add_argument("--output-path", default="data/outputs/cleaned.csv")
    parser.add_argument("--audit-path", default="data/outputs/removed_rows.csv")
    parser.add_argument("--use-openai", action="store_true", help="Translate company names with OpenAI")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--log-path", default="data/outputs/run_history.log")
    parser.add_argument("--clear", action="store_true", help="Remove previous outputs")
    args = parser.parse_args()

    print("\u23e9 started preprocessing")
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        audit_path=args.audit_path,
        use_openai=args.use_openai,
        openai_model=args.openai_model,
        log_path=args.log_path,
        clear=args.clear,
    )
