"""OpenAI Formatter - Terminal Output

Handles all terminal output formatting for OpenAI-powered operations.
This module is responsible for displaying AI processing progress, API statistics,
and results in a user-friendly format.
"""

from typing import Dict, Any, List, Optional


class OpenAIFormatter:
    """Formats OpenAI operation output for terminal display."""
    
    @staticmethod
    def format_progress(message: str) -> str:
        """Format a progress message."""
        return f"🤖 {message}"
    
    @staticmethod
    def format_separator() -> str:
        """Format section separator."""
        return "─" * 60
    
    @staticmethod
    def format_error(message: str) -> str:
        """Format an error message."""
        return f"❌ Error: {message}"
    
    @staticmethod
    def format_success(message: str) -> str:
        """Format a success message."""
        return f"✅ {message}"
    
    @staticmethod
    def format_warning(message: str) -> str:
        """Format a warning message."""
        return f"⚠️  Warning: {message}"
    
    @staticmethod
    def format_cluster_review_start(total_clusters: int) -> str:
        """Format cluster review start message."""
        return f"🔍 Analyzing {total_clusters:,} clusters with AI..."
    
    @staticmethod
    def format_deduplication_start(total_pairs: int) -> str:
        """Format deduplication start message."""
        return f"🔗 Analyzing {total_pairs:,} candidate pairs for merging..."
    
    @staticmethod
    def format_api_progress(current: int, total: int, operation: str) -> str:
        """Format API call progress."""
        percentage = (current / total) * 100 if total > 0 else 0
        return f"📡 {operation}: {current:,}/{total:,} ({percentage:.1f}%)"
    
    @staticmethod
    def format_cluster_review_results(review_data: Dict[str, Any]) -> List[str]:
        """Format cluster review results."""
        lines = []
        lines.append("🎯 Cluster Review Complete!")
        lines.append(OpenAIFormatter.format_separator())
        
        lines.append("📊 Analysis Summary:")
        lines.append(f"  • Clusters analyzed:     {review_data.get('clusters_analyzed', 0):,}")
        lines.append(f"  • Valid clusters:        {review_data.get('valid_clusters', 0):,}")
        lines.append(f"  • Invalid clusters:      {review_data.get('invalid_clusters', 0):,}")
        lines.append(f"  • Average confidence:    {review_data.get('average_confidence', 0):.2f}")
        lines.append(f"  • Model used:           {review_data.get('model_used', 'N/A')}")
        
        # API Statistics
        api_stats = review_data.get('api_stats', {})
        lines.append(f"\n🔌 API Usage:")
        lines.append(f"  • Total API calls:       {api_stats.get('calls', 0):,}")
        lines.append(f"  • Successful batches:    {api_stats.get('successful_batches', 0):,}")
        lines.append(f"  • Failed batches:        {api_stats.get('failed_batches', 0):,}")
        
        tokens = api_stats.get('tokens', {})
        lines.append(f"  • Tokens used:           {tokens.get('total', 0):,}")
        lines.append(f"    - Prompt tokens:       {tokens.get('prompt', 0):,}")
        lines.append(f"    - Completion tokens:   {tokens.get('completion', 0):,}")
        
        durations = api_stats.get('durations', [])
        if durations:
            avg_duration = sum(durations) / len(durations)
            lines.append(f"  • Avg response time:     {avg_duration:.2f}s")
        
        lines.append(f"  • Total processing time: {review_data.get('processing_time', 0):.1f}s")
        
        # Error summary
        errors = api_stats.get('errors', {})
        if errors:
            lines.append(f"\n⚠️  Errors encountered:")
            for error_type, count in errors.items():
                lines.append(f"  • {error_type}: {count}")
        
        return lines
    
    @staticmethod
    def format_deduplication_results(analysis_data: Dict[str, Any]) -> List[str]:
        """Format deduplication results."""
        lines = []
        lines.append("🎯 AI Deduplication Complete!")
        lines.append(OpenAIFormatter.format_separator())
        
        lines.append("📊 Deduplication Summary:")
        lines.append(f"  • Original records:      {analysis_data.get('original_records', 0):,}")
        lines.append(f"  • Unique records:        {analysis_data.get('unique_records', 0):,}")
        lines.append(f"  • Records merged:        {analysis_data.get('original_records', 0) - analysis_data.get('unique_records', 0):,}")
        lines.append(f"  • Reduction rate:        {analysis_data.get('reduction_rate', 0):.1%}")
        lines.append(f"  • Model used:           {analysis_data.get('model_used', 'N/A')}")
        
        lines.append(f"\n🔗 Merge Analysis:")
        lines.append(f"  • Pairs analyzed:        {analysis_data.get('total_pairs_analyzed', 0):,}")
        lines.append(f"  • Confident merges:      {analysis_data.get('confident_merges', 0):,}")
        lines.append(f"  • Merge groups created:  {analysis_data.get('merge_groups', 0):,}")
        
        # API Statistics
        api_stats = analysis_data.get('api_stats', {})
        lines.append(f"\n🔌 API Usage:")
        lines.append(f"  • Total API calls:       {api_stats.get('calls', 0):,}")
        lines.append(f"  • Successful batches:    {api_stats.get('successful_batches', 0):,}")
        lines.append(f"  • Failed batches:        {api_stats.get('failed_batches', 0):,}")
        
        tokens = api_stats.get('tokens', {})
        lines.append(f"  • Tokens used:           {tokens.get('total', 0):,}")
        lines.append(f"    - Prompt tokens:       {tokens.get('prompt', 0):,}")
        lines.append(f"    - Completion tokens:   {tokens.get('completion', 0):,}")
        
        durations = api_stats.get('durations', [])
        if durations:
            avg_duration = sum(durations) / len(durations)
            lines.append(f"  • Avg response time:     {avg_duration:.2f}s")
        
        lines.append(f"  • Total processing time: {analysis_data.get('processing_time', 0):.1f}s")
        
        # Error summary
        errors = api_stats.get('errors', {})
        if errors:
            lines.append(f"\n⚠️  Errors encountered:")
            for error_type, count in errors.items():
                lines.append(f"  • {error_type}: {count}")
        
        return lines
    
    @staticmethod
    def format_file_info(review_path: Optional[str] = None, output_path: Optional[str] = None, analysis_path: Optional[str] = None) -> List[str]:
        """Format file creation information."""
        lines = []
        lines.append("💾 Files Created:")
        
        if review_path:
            lines.append(f"  • Cluster review: {review_path}")
        if output_path:
            lines.append(f"  • Unique records: {output_path}")
        if analysis_path:
            lines.append(f"  • Analysis data: {analysis_path}")
        
        return lines
    
    @staticmethod
    def format_next_steps(operation_type: str, has_results: bool = True) -> List[str]:
        """Format next steps guidance."""
        lines = []
        
        if operation_type == "cluster_review":
            if has_results:
                lines.append("✅ Next steps:")
                lines.append("   • Review the cluster analysis in the JSON output")
                lines.append("   • Use invalid clusters to refine clustering parameters")
                lines.append("   • Proceed with reporting or manual review")
            else:
                lines.append("💡 Suggestions:")
                lines.append("   • Check clustering results for valid clusters")
                lines.append("   • Adjust cluster size filters")
                lines.append("   • Review API configuration and credentials")
        
        elif operation_type == "deduplication":
            if has_results:
                lines.append("✅ Next steps:")
                lines.append("   • Review the unique records output")
                lines.append("   • Validate merge decisions in analysis file")
                lines.append("   • Use deduplicated records for final processing")
            else:
                lines.append("💡 Suggestions:")
                lines.append("   • Lower similarity or confidence thresholds")
                lines.append("   • Check feature quality and completeness")
                lines.append("   • Review API configuration and credentials")
        
        return lines
    
    @staticmethod
    def format_cost_estimate(tokens_used: int, model: str) -> List[str]:
        """Format estimated API cost information."""
        lines = []
        
        # Rough cost estimates (as of 2024, subject to change)
        cost_per_1k_tokens = {
            "gpt-4o-mini-2024-07-18": 0.00015,  # Input tokens
            "gpt-4o-mini": 0.00015,
            "gpt-3.5-turbo": 0.0015,
            "gpt-4": 0.03
        }
        
        base_model = model.split("-")[0] + "-" + model.split("-")[1] if "-" in model else model
        rate = cost_per_1k_tokens.get(model, cost_per_1k_tokens.get(base_model, 0.002))
        
        estimated_cost = (tokens_used / 1000) * rate
        
        lines.append("💰 Estimated API Cost:")
        lines.append(f"  • Tokens used: {tokens_used:,}")
        lines.append(f"  • Model: {model}")
        lines.append(f"  • Estimated cost: ${estimated_cost:.4f}")
        lines.append("  • (Estimate only - check OpenAI billing for actual costs)")
        
        return lines
    
    @staticmethod
    def format_comprehensive_cluster_results(review_data: Dict[str, Any], review_path: str) -> str:
        """Format comprehensive cluster review results."""
        lines = []
        
        # Header
        lines.extend(OpenAIFormatter.format_cluster_review_results(review_data))
        
        # Files
        lines.append("")
        lines.extend(OpenAIFormatter.format_file_info(review_path=review_path))
        
        # Cost estimate
        tokens = review_data.get('api_stats', {}).get('tokens', {}).get('total', 0)
        if tokens > 0:
            lines.append("")
            lines.extend(OpenAIFormatter.format_cost_estimate(tokens, review_data.get('model_used', 'gpt-4o-mini')))
        
        # Next steps
        lines.append("")
        has_results = review_data.get('clusters_analyzed', 0) > 0
        lines.extend(OpenAIFormatter.format_next_steps("cluster_review", has_results))
        
        return "\n".join(lines)
    
    @staticmethod
    def format_comprehensive_deduplication_results(analysis_data: Dict[str, Any], output_path: str, analysis_path: str) -> str:
        """Format comprehensive deduplication results."""
        lines = []
        
        # Header
        lines.extend(OpenAIFormatter.format_deduplication_results(analysis_data))
        
        # Files
        lines.append("")
        lines.extend(OpenAIFormatter.format_file_info(output_path=output_path, analysis_path=analysis_path))
        
        # Cost estimate
        tokens = analysis_data.get('api_stats', {}).get('tokens', {}).get('total', 0)
        if tokens > 0:
            lines.append("")
            lines.extend(OpenAIFormatter.format_cost_estimate(tokens, analysis_data.get('model_used', 'gpt-4o-mini')))
        
        # Next steps
        lines.append("")
        has_results = analysis_data.get('unique_records', 0) > 0
        lines.extend(OpenAIFormatter.format_next_steps("deduplication", has_results))
        
        return "\n".join(lines)
