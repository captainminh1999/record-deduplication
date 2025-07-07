"""Step 5 of 6: Reporting (Review and Export)

Generates an Excel workbook listing high-confidence duplicate pairs side by side for human review. See README for details.

This module now serves as a bridge to the new modular architecture.
For new development, use the modular components:
- src.core.reporting_engine (business logic)
- src.formatters.reporting_formatter (terminal output)
- src.cli.reporting (CLI orchestration)
"""

from __future__ import annotations

import click
import time
import json

from .core.reporting_engine import ReportingEngine
from .formatters.reporting_formatter import ReportingFormatter
from .utils import log_run, LOG_PATH


def main(
    dupes_path: str = "data/outputs/high_confidence.csv",
    cleaned_path: str = "data/outputs/cleaned.csv",
    report_path: str = "data/outputs/manual_review.xlsx",
    log_path: str = LOG_PATH,
    gpt_review_path: str = "data/outputs/gpt_review.json",
) -> None:
    """Create a merge suggestion workbook, now with GPT review results."""

    start_time = time.time()
    formatter = ReportingFormatter()
    
    try:
        # Initialize reporting engine
        engine = ReportingEngine()
        
        # Display progress
        print(formatter.format_progress("Starting Excel report generation..."))
        
        # Generate report
        stats = engine.generate_report(
            dupes_path=dupes_path,
            cleaned_path=cleaned_path,
            report_path=report_path,
            gpt_review_path=gpt_review_path
        )
        
        # Display comprehensive results
        results_output = formatter.format_comprehensive_results(stats, report_path)
        print(results_output)
        
        # Log the run
        summary_stats = engine.get_summary_stats()
        end_time = time.time()
        log_run(
            "reporting",
            start_time,
            end_time,
            summary_stats.get("total_pairs", 0),
            additional_info=json.dumps(summary_stats),
            log_path=log_path,
        )
        
    except Exception as e:
        print(formatter.format_error(f"Report generation failed: {e}"))
        raise


@click.command()
@click.option("--duplicates-path", default="data/outputs/high_confidence.csv", show_default=True)
@click.option("--cleaned-path", default="data/outputs/cleaned.csv", show_default=True)
@click.option("--report-path", default="data/outputs/manual_review.xlsx", show_default=True)
@click.option("--log-path", default=LOG_PATH, show_default=True)
@click.option("--gpt-review-path", default="data/outputs/gpt_review.json", show_default=True, help="Path to GPT review JSON file.")
def cli(duplicates_path: str, cleaned_path: str, report_path: str, log_path: str, gpt_review_path: str) -> None:
    """CLI wrapper for :func:`main`."""

    main(duplicates_path, cleaned_path, report_path, log_path, gpt_review_path)


if __name__ == "__main__":  # pragma: no cover - sanity run
    print("\u23e9 started reporting")
    cli()
