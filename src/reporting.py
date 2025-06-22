"""Step 5 of the 10-step deduplication pipeline: reporting.

This module prepares an Excel workbook that lists high-confidence duplicates
side by side for human validation.
"""

from __future__ import annotations

import click
import os
import pandas as pd
from openpyxl.styles import PatternFill
from pandas import ExcelWriter


def main(
    dupes_path: str = "data/outputs/high_confidence.csv",
    cleaned_path: str = "data/outputs/cleaned.csv",
    report_path: str = "data/outputs/manual_review.xlsx",
) -> None:
    """Create a merge suggestion workbook."""

    dupes = pd.read_csv(dupes_path)
    cleaned = pd.read_csv(cleaned_path).set_index("record_id")

    # Combine duplicate pairs with full record details from both sides.
    left = cleaned.loc[dupes["record_id_1"]].add_suffix("_1").reset_index()
    right = cleaned.loc[dupes["record_id_2"]].add_suffix("_2").reset_index()
    merged = pd.concat([dupes.reset_index(drop=True), left, right], axis=1)

    # Probability thresholds may be tuned later.
    high_conf = merged[merged["prob"] >= 0.9]
    manual_review = merged[(merged["prob"] >= 0.6) & (merged["prob"] < 0.9)]

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with ExcelWriter(report_path, engine="openpyxl") as writer:
        high_conf.to_excel(writer, sheet_name="high_confidence", index=False)
        manual_review.to_excel(writer, sheet_name="manual_review", index=False)

        # Optional formatting highlighting borderline pairs for review.
        workbook = writer.book
        review_ws = writer.sheets["manual_review"]
        fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
        score_col = manual_review.columns.get_loc("prob") + 1
        for row in review_ws.iter_rows(
            min_row=2, max_row=review_ws.max_row, min_col=score_col, max_col=score_col
        ):
            for cell in row:
                cell.fill = fill

    print(
        f"Saved {len(high_conf)} high-confidence pairs and {len(manual_review)} "
        f"pairs for manual review to {report_path}"
    )


@click.command()
@click.option("--duplicates-path", default="data/outputs/high_confidence.csv", show_default=True)
@click.option("--cleaned-path", default="data/outputs/cleaned.csv", show_default=True)
@click.option("--report-path", default="data/outputs/manual_review.xlsx", show_default=True)
def cli(duplicates_path: str, cleaned_path: str, report_path: str) -> None:
    """CLI wrapper for :func:`main`."""

    main(duplicates_path, cleaned_path, report_path)


if __name__ == "__main__":  # pragma: no cover - sanity run
    print("\u23e9 started reporting")
    cli()
