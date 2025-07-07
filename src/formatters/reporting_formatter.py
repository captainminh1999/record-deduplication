"""Reporting Formatter - Terminal Output

Handles all terminal output formatting for the reporting step.
This module is responsible for displaying report generation progress, results,
and statistics in a user-friendly format.
"""

from typing import Dict, Any, List, Optional


class ReportingFormatter:
    """Formats reporting output for terminal display."""
    
    @staticmethod
    def format_progress(message: str) -> str:
        """Format a progress message."""
        return f"📊 {message}"
    
    @staticmethod
    def format_report_complete() -> str:
        """Format report completion header."""
        return "\n📊 Excel Report Generation Complete!"
    
    @staticmethod
    def format_separator() -> str:
        """Format section separator."""
        return "─" * 50
    
    @staticmethod
    def format_data_overview(stats: Dict[str, Any]) -> List[str]:
        """Format data overview section."""
        input_stats = stats.get("input_stats", {})
        lines = ["📊 Data Overview:"]
        lines.append(f"  • Total pairs analyzed:  {input_stats.get('total_pairs', 0):,}")
        lines.append(f"  • Unique records:        {input_stats.get('unique_records', 0):,}")
        lines.append(f"  • Mean probability:      {input_stats.get('mean_probability', 0.0):.3f}")
        return lines
    
    @staticmethod
    def format_probability_distribution(stats: Dict[str, Any]) -> List[str]:
        """Format probability distribution section."""
        quants = stats.get("input_stats", {}).get("probability_quantiles", {})
        lines = ["📈 Probability Distribution:"]
        lines.append(f"  • 25th percentile:       {quants.get('25%', 0.0):.3f}")
        lines.append(f"  • 50th percentile:       {quants.get('50%', 0.0):.3f}")
        lines.append(f"  • 75th percentile:       {quants.get('75%', 0.0):.3f}")
        lines.append(f"  • 90th percentile:       {quants.get('90%', 0.0):.3f}")
        return lines
    
    @staticmethod
    def format_excel_sheets(stats: Dict[str, Any]) -> List[str]:
        """Format Excel sheets created section."""
        output_sheets = stats.get("output_sheets", {})
        lines = ["📋 Excel Sheets Created:"]
        for sheet_name, count in output_sheets.items():
            lines.append(f"  • {sheet_name:<15} {count:,} pairs")
        return lines
    
    @staticmethod
    def format_gpt_review_results(stats: Dict[str, Any]) -> List[str]:
        """Format GPT review results section."""
        gpt_stats = stats.get("gpt_review", {})
        
        if not gpt_stats.get("available", False):
            return ["🤖 GPT Review: Not available (no GPT analysis found)"]
        
        lines = ["🤖 GPT Review Results:"]
        try:
            lines.append(f"  • Total groups:          {gpt_stats.get('total_groups', 0):,}")
            lines.append(f"  • Total records:         {gpt_stats.get('total_records_processed', 0):,}")
            lines.append(f"  • Records per group:     {gpt_stats.get('records_per_group', 0.0):.1f}")
            if 'processed_clusters' in gpt_stats:
                lines.append(f"  • Processed clusters:    {gpt_stats['processed_clusters']:,}")
            if 'mean_confidence' in gpt_stats:
                lines.append(f"  • Mean confidence:       {gpt_stats['mean_confidence']:.3f}")
        except Exception as e:
            lines.append(f"  • Error displaying GPT stats: {e}")
        
        return lines
    
    @staticmethod
    def format_files_created(report_path: str, total_pairs: int) -> List[str]:
        """Format files created section."""
        return [
            "💾 Files Created:",
            f"  • Excel workbook:        {report_path}",
            f"  • Total pairs for review: {total_pairs:,}"
        ]
    
    @staticmethod
    def format_success_message(stats: Dict[str, Any]) -> List[str]:
        """Format success message with next steps."""
        output_sheets = stats.get("output_sheets", {})
        high_confidence = output_sheets.get("high_confidence", 0)
        manual_review = output_sheets.get("manual_review", 0)
        total_pairs = high_confidence + manual_review
        
        lines = []
        
        if high_confidence > 0:
            lines.append(f"✅ Success! Found {high_confidence:,} high-confidence duplicate pairs")
            lines.append("   Review the 'high_confidence' sheet first")
        
        if manual_review > 0:
            lines.append(f"   Also review {manual_review:,} borderline pairs in 'manual_review' sheet")
        
        if total_pairs == 0:
            lines.extend([
                "⚠️  No duplicate pairs found for review",
                "   Consider lowering confidence thresholds or improving training data"
            ])
        
        return lines
    
    @staticmethod
    def format_review_instructions(report_path: str, gpt_available: bool) -> List[str]:
        """Format review instructions."""
        lines = [
            "📖 How to Review:",
            f"   1. Open {report_path} in Excel",
            "   2. Check 'high_confidence' sheet for likely duplicates",
            "   3. Review 'manual_review' sheet for borderline cases"
        ]
        
        if gpt_available:
            lines.append("   4. Reference 'gpt_review' sheet for AI analysis")
        
        return lines
    
    @staticmethod
    def format_comprehensive_results(stats: Dict[str, Any], report_path: str) -> str:
        """Format comprehensive reporting results."""
        lines = []
        
        # Header
        lines.append(ReportingFormatter.format_report_complete())
        lines.append(ReportingFormatter.format_separator())
        
        # Data overview
        lines.extend(ReportingFormatter.format_data_overview(stats))
        lines.append("")
        
        # Probability distribution
        lines.extend(ReportingFormatter.format_probability_distribution(stats))
        lines.append("")
        
        # Excel sheets
        lines.extend(ReportingFormatter.format_excel_sheets(stats))
        lines.append("")
        
        # GPT review results
        lines.extend(ReportingFormatter.format_gpt_review_results(stats))
        lines.append("")
        
        # Files created
        output_sheets = stats.get("output_sheets", {})
        total_pairs = output_sheets.get("high_confidence", 0) + output_sheets.get("manual_review", 0)
        lines.extend(ReportingFormatter.format_files_created(report_path, total_pairs))
        lines.append("")
        
        # Success message
        lines.extend(ReportingFormatter.format_success_message(stats))
        lines.append("")
        
        # Review instructions
        gpt_available = stats.get("gpt_review", {}).get("available", False)
        lines.extend(ReportingFormatter.format_review_instructions(report_path, gpt_available))
        
        return "\n".join(lines)
    
    @staticmethod
    def format_error(error_message: str) -> str:
        """Format error message."""
        return f"❌ Error: {error_message}"
    
    @staticmethod
    def format_warning(warning_message: str) -> str:
        """Format warning message."""
        return f"⚠️  Warning: {warning_message}"
    
    @staticmethod
    def format_file_not_found(file_path: str) -> str:
        """Format file not found message."""
        return f"⚠️  File not found: {file_path}"
    
    @staticmethod
    def format_gpt_processing_error(error: str) -> str:
        """Format GPT processing error message."""
        return f"⚠️  GPT review processing error: {error}"
