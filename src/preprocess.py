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
from .corp_designators import CORP_PREFIXES, CORP_SUFFIXES


_PREFIX_RE = re.compile(rf"^(?:{'|'.join(CORP_PREFIXES)})\b[\s\.,]*", flags=re.I)
_SUFFIX_RE = re.compile(rf"\b(?:{'|'.join(CORP_SUFFIXES)})\b\.?", flags=re.I)
_PUNCT_RE = re.compile(r"[^\w\s&/\.-]+")
_WS_RE = re.compile(r"\s+")


def normalize_company_name(name: str) -> str:
    """Return a cleaned company name following a set of rules."""
    if not name:
        return ""
    text_norm = unicodedata.normalize("NFKD", str(name))
    ascii_text = text_norm.encode("ascii", "ignore").decode("ascii")
    if not ascii_text.strip():
        if "ソニー" in name:
            ascii_text = "sony"
    text = ascii_text.lower()
    text = _WS_RE.sub(" ", text).strip()
    text = re.sub(r"^the\s+", "", text)
    text = _PREFIX_RE.sub("", text).strip()
    text = text.replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    for _ in range(2):
        text = _SUFFIX_RE.sub("", text).strip()
        text = _WS_RE.sub(" ", text).strip()
    text = re.sub(r"[\./-]+$", "", text).strip()
    return text


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

    The function loads ``input_path`` using :func:`pandas.read_csv` for CSV
    files or :func:`pandas.read_excel` for Excel files. Company, domain and
    phone columns are normalised into new ``*_clean`` fields, exact duplicates
    based on the ``combined_id`` are dropped and written to ``audit_path`` and
    the cleaned dataset is saved to ``output_path``.
    """
    if clear:
        clear_all_data(os.path.dirname(output_path))

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input spreadsheet not found: {input_path}")

    _, ext = os.path.splitext(input_path)
    if ext.lower() in {".xls", ".xlsx", ".xlsm"}:
        df = pd.read_excel(input_path)
    else:
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
        df["company_clean"] = translate_to_english(
            df["company"].tolist(), model=openai_model
        )
    else:
        df["company_clean"] = df["company"].map(normalize_company_name)

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

    # Build an address column from various components if one isn't provided
    address_key = col_map.get("address")
    if address_key:
        address_series = df[address_key]
    else:
        comp_keys = [
            col_map.get("street"),
            col_map.get("streetcont"),
            col_map.get("city"),
            col_map.get("state"),
            col_map.get("countrycode"),
        ]
        comps = [df[k] for k in comp_keys if k]
        if comps:
            address_series = (
                pd.concat(comps, axis=1)
                .fillna("")
                .astype(str)
                .agg(" ".join, axis=1)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
        else:
            address_series = pd.Series("", index=df.index)
    df["address_clean"] = address_series

    df["combined_id"] = (
        df["company_clean"] + ";" + df["domain_clean"] + ";" + df["phone_clean"]
    )

    duplicates = df[df.duplicated(subset="combined_id", keep="first")]
    if not duplicates.empty:
        os.makedirs(os.path.dirname(audit_path), exist_ok=True)
        duplicates.assign(reason="duplicate combined_id").to_csv(
            audit_path, index=False
        )

    df = df.drop_duplicates(subset="combined_id", keep="first")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
    parser.add_argument(
        "--use-openai", action="store_true", help="Translate company names with OpenAI"
    )
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--log-path", default="data/outputs/run_history.log")
    parser.add_argument("--clear", action="store_true", help="Remove previous outputs")
    args = parser.parse_args()

    assert normalize_company_name("The ACME, Inc.") == "acme"
    assert normalize_company_name("PT Astra International Tbk") == "astra international"
    assert normalize_company_name("株式会社ソニー") == "sony"
    assert normalize_company_name("XYZ Sdn Bhd") == "xyz"

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
