"""Reporting CLI Module

Command-line interface for the reporting step of the record deduplication pipeline.
This module orchestrates the reporting engine and formatter to provide a clean CLI experience.
"""

import time
import json
import click

from ..core.reporting_engine import ReportingEngine
from ..formatters.reporting_formatter import ReportingFormatter
from ..utils import log_run, LOG_PATH


@click.command()
@click.option("--duplicates-path", default="data/outputs/high_confidence.csv", show_default=True,
              help="Path to high-confidence duplicates CSV file")
@click.option("--cleaned-path", default="data/outputs/cleaned.csv", show_default=True,
              help="Path to cleaned records CSV file")
@click.option("--report-path", default="data/outputs/manual_review.xlsx", show_default=True,
              help="Path to save Excel report")
@click.option("--gpt-review-path", default="data/outputs/gpt_review.json", show_default=True,
              help="Path to GPT review JSON file (optional)")
@click.option("--log-path", default=LOG_PATH, show_default=True,
              help="Path to log file")
def reporting(
    duplicates_path: str,
    cleaned_path: str,
    report_path: str,
    gpt_review_path: str,
    log_path: str,
) -> None:
    """
    Generate comprehensive Excel report from duplicate pairs and GPT analysis.
    
    This step creates a detailed Excel workbook with multiple sheets containing
    high-confidence duplicates, manual review cases, and GPT analysis results
    for human review and validation.
    
    Examples:
    
        # Basic report generation
        python -m src.cli.reporting
        
        # Custom paths
        python -m src.cli.reporting --duplicates-path data/dupes.csv --report-path output/report.xlsx
        
        # Without GPT review
        python -m src.cli.reporting --gpt-review-path ""
    """
    start_time = time.time()
    formatter = ReportingFormatter()
    
    try:
        # Initialize reporting engine
        engine = ReportingEngine()
        
        # Display progress
        print(formatter.format_progress("Starting Excel report generation..."))
        
        # Generate report
        gpt_path = gpt_review_path if gpt_review_path else None
        stats = engine.generate_report(
            dupes_path=duplicates_path,
            cleaned_path=cleaned_path,
            report_path=report_path,
            gpt_review_path=gpt_path
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
        
    except FileNotFoundError as e:
        error_msg = f"Input file not found: {e}"
        print(formatter.format_error(error_msg))
        raise click.ClickException(error_msg)
    
    except PermissionError as e:
        error_msg = f"Permission denied writing report: {e}"
        print(formatter.format_error(error_msg))
        raise click.ClickException(error_msg)
    
    except Exception as e:
        error_msg = f"Report generation failed: {e}"
        print(formatter.format_error(error_msg))
        raise click.ClickException(error_msg)


if __name__ == "__main__":
    reporting()
