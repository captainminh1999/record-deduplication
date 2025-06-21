"""Step 1 of the 10-step deduplication pipeline: data cleaning.

This module prepares the raw spreadsheet for later stages by normalising
fields and writing the cleaned output to ``data/cleaned.csv``.
"""

from __future__ import annotations

import os
import re
import unicodedata

import pandas as pd

from .openai_integration import translate_to_english


def _normalize_name(name: str) -> str:
    """Return a basic normalised representation of ``name``."""
    text = str(name) if pd.notnull(name) else ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.strip().lower()


def _normalize_phone(phone: str) -> str:
    text = str(phone) if pd.notnull(phone) else ""
    digits = re.sub(r"\D", "", text)
    return digits


def main(
    input_path: str = "data/your_spreadsheet.csv",
    output_path: str = "data/cleaned.csv",
    audit_path: str = "data/removed_rows.csv",
    use_openai: bool = False,
    openai_model: str = "gpt-4o-mini",
) -> None:
    """Clean the raw spreadsheet and save a CSV.

    TODO:
        * load the source spreadsheet with :func:`pandas.read_csv` or
          ``read_excel``
        * normalise name and phone columns
        * remove exact duplicate rows
        * write the result to ``output_path``
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input spreadsheet not found: {input_path}")

    df = pd.read_csv(input_path)

    if use_openai:
        df["name_clean"] = translate_to_english(df["name"].tolist(), model=openai_model)
    else:
        df["name_clean"] = df["name"].map(_normalize_name)

    df["phone_clean"] = df.get("phone", "").map(_normalize_phone)

    df["combined_id"] = df["name_clean"] + ";" + df["phone_clean"]

    duplicates = df[df.duplicated(subset="combined_id", keep="first")]
    if not duplicates.empty:
        duplicates.assign(reason="duplicate combined_id").to_csv(audit_path, index=False)

    df = df.drop_duplicates(subset="combined_id", keep="first")

    df.to_csv(output_path, index=False)


if __name__ == "__main__":  # pragma: no cover - sanity run
    import argparse

    parser = argparse.ArgumentParser(description="Clean raw spreadsheet")
    parser.add_argument("--input-path", default="data/your_spreadsheet.csv")
    parser.add_argument("--output-path", default="data/cleaned.csv")
    parser.add_argument("--audit-path", default="data/removed_rows.csv")
    parser.add_argument("--use-openai", action="store_true", help="Translate names with OpenAI")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    args = parser.parse_args()

    print("\u23e9 started preprocessing")
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        audit_path=args.audit_path,
        use_openai=args.use_openai,
        openai_model=args.openai_model,
    )
