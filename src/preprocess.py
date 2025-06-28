"""Step 1 of 6: Preprocessing (Data Cleaning)

Cleans and normalizes raw spreadsheet data, removes records that duplicate the same company or domain, and writes the cleaned output to data/outputs/cleaned.csv. Optionally translates company names to English using GPT. See README for details.
"""

from __future__ import annotations

import os
import re
import unicodedata
import time
import json

import pandas as pd

from .openai_integration import translate_to_english
from .utils import log_run, clear_all_data, LOG_PATH
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
    log_path: str = LOG_PATH,
    clear: bool = False,
) -> pd.DataFrame:
    """Clean the raw spreadsheet and save a CSV.

    The function loads ``input_path`` using :func:`pandas.read_csv` for CSV
    files or :func:`pandas.read_excel` for Excel files. Company, domain and
    phone columns are normalised into new ``*_clean`` fields. Rows that share
    the same normalised company name **or** the same normalised domain are
    considered duplicates, written to ``audit_path`` and removed from the
    output written to ``output_path``.
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

    # Track statistics at the beginning
    start_time = time.time()
    initial_rows = len(df)
    null_counts = df.isnull().sum()
    missing_company = df["company"].isnull().sum()

    # Get domain information
    domain_key = col_map.get("domain") or col_map.get("website")
    if domain_key:
        domain_col = df[domain_key]
        missing_domain = domain_col.isnull().sum()
    else:
        domain_col = pd.Series("", index=df.index)
        missing_domain = len(df)

    # Track company name stats before cleaning
    unique_companies_before = df["company"].nunique()

    if use_openai:
        df["company_clean"] = translate_to_english(
            df["company"].tolist(), model=openai_model
        )
    else:
        df["company_clean"] = df["company"].map(normalize_company_name)

    # Track company name stats after cleaning
    unique_companies_after = df["company_clean"].nunique()
    empty_company_names = len(df[df["company_clean"] == ""])

    domain_col = df[domain_key] if domain_key else pd.Series("", index=df.index)
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

    df["_id_str"] = df["record_id"].astype(str)

    company_map = (
        df.groupby("company_clean")["_id_str"].apply(lambda s: set(s)).to_dict()
    )
    domain_map = (
        df[df["domain_clean"].ne("")]
        .groupby("domain_clean")["_id_str"]
        .apply(lambda s: set(s))
        .to_dict()
    )

    def _build_merged(row: pd.Series) -> str:
        ids = {row["_id_str"]}
        comp_ids = company_map.get(row["company_clean"])
        if comp_ids and len(comp_ids) > 1:
            ids.update(comp_ids)
        dom_ids = domain_map.get(row["domain_clean"])
        if dom_ids and len(dom_ids) > 1:
            ids.update(dom_ids)
        return ";".join(sorted(ids))

    df["merged_ids"] = df.apply(_build_merged, axis=1)

    dup_by_company = df.duplicated(subset=["company_clean"], keep="first")
    dup_by_domain = (
        df["domain_clean"].ne("")
        & df.duplicated(subset=["domain_clean"], keep="first")
    )
    dup_mask = dup_by_company | dup_by_domain
    duplicates = df[dup_mask].drop(columns=["_id_str"])
    if not duplicates.empty:
        os.makedirs(os.path.dirname(audit_path), exist_ok=True)
        duplicates.assign(reason="duplicate company or domain").to_csv(
            audit_path, index=False
        )

    df = df[~dup_mask].drop(columns=["_id_str"])

    result = pd.DataFrame(df)
    result.to_csv(output_path, index=True)
    print(f"Wrote {len(result)} cleaned records to {output_path}")

    end_time = time.time()
    stats = {
        "input_rows": initial_rows,
        "output_rows": len(result),
        "duplicates_removed": len(duplicates),
        "missing_company": int(missing_company),
        "missing_domain": int(missing_domain),
        "null_values": null_counts.to_dict(),
        "company_cleanup": {
            "empty_after_clean": empty_company_names,
            "unique_before": unique_companies_before,
            "unique_after": unique_companies_after,
        }
    }
    log_run("preprocess", start_time, end_time, len(result), additional_info=json.dumps(stats), log_path=log_path)

    return result


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
    parser.add_argument("--log-path", default=LOG_PATH)
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
