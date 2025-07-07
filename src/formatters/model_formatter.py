"""Model Formatter - Terminal Output

Handles all terminal output formatting for the model training step.
This module is responsible for displaying training progress, model performance,
and results in a user-friendly format.
"""

from typing import Dict, Any, List, Optional


class ModelFormatter:
    """Formats model training output for terminal display."""
    
    @staticmethod
    def format_progress(message: str) -> str:
        """Format a progress message."""
        return f"🚀 {message}"
    
    @staticmethod
    def format_training_complete() -> str:
        """Format training completion header."""
        return "\n🎯 Model Training Complete!"
    
    @staticmethod
    def format_separator() -> str:
        """Format section separator."""
        return "─" * 50
    
    @staticmethod
    def format_data_overview(stats: Dict[str, Any]) -> List[str]:
        """Format data overview section."""
        lines = ["📊 Data Overview:"]
        lines.append(f"  • Input pairs:           {stats.get('input_pairs', 0):,}")
        lines.append(f"  • Training pairs:        {stats.get('training_pairs', 0):,}")
        
        features = stats.get('model_stats', {}).get('features', [])
        if features:
            lines.append(f"  • Features used:         {len(features)} ({', '.join(features)})")
        
        return lines
    
    @staticmethod
    def format_training_labels(stats: Dict[str, Any]) -> List[str]:
        """Format training labels section."""
        label_stats = stats.get('label_stats', {})
        label_source = label_stats.get('source', 'unknown')
        
        lines = [f"🏷️  Training Labels ({label_source}):"]
        lines.append(f"  • Total labels:          {label_stats.get('total_labels', 0):,}")
        
        positive = label_stats.get('positive_labels', 0)
        ratio = label_stats.get('label_ratio', 0) * 100
        lines.append(f"  • Positive (duplicates): {positive:,} ({ratio:.1f}%)")
        lines.append(f"  • Negative (unique):     {label_stats.get('negative_labels', 0):,}")
        
        return lines
    
    @staticmethod
    def format_model_performance(stats: Dict[str, Any]) -> List[str]:
        """Format model performance section."""
        pred_stats = stats.get('prediction_stats', {})
        prob_dist = pred_stats.get('prob_dist', {})
        
        lines = ["🤖 Model Performance:"]
        lines.append(f"  • Mean probability:      {pred_stats.get('mean_prob', 0):.3f}")
        lines.append(f"  • 90th percentile:       {prob_dist.get('p90', 0):.3f}")
        lines.append(f"  • 95th percentile:       {prob_dist.get('p95', 0):.3f}")
        lines.append(f"  • 99th percentile:       {prob_dist.get('p99', 0):.3f}")
        
        return lines
    
    @staticmethod
    def format_results(stats: Dict[str, Any]) -> List[str]:
        """Format results section."""
        high_conf_pairs = stats.get('high_confidence_pairs', 0)
        
        lines = ["📈 Results:"]
        lines.append(f"  • High-confidence pairs: {high_conf_pairs:,} (≥90% probability)")
        
        if high_conf_pairs > 0:
            lines.append(f"  • Success! Found {high_conf_pairs:,} likely duplicate pairs")
        else:
            lines.append("  • No high-confidence duplicates found (try lowering threshold)")
        
        return lines
    
    @staticmethod
    def format_files_created(model_path: str, duplicates_path: str) -> List[str]:
        """Format files created section."""
        return [
            "💾 Files Created:",
            f"  • Model: {model_path}",
            f"  • High-confidence pairs: {duplicates_path}"
        ]
    
    @staticmethod
    def format_next_steps(high_conf_pairs: int) -> List[str]:
        """Format next steps based on results."""
        if high_conf_pairs > 0:
            return [
                "✅ Next step: Run reporting to create Excel review file",
                "   Command: python -m src.reporting"
            ]
        else:
            return [
                "💡 Suggestions:",
                "   • Lower confidence threshold in code (currently 0.9)",
                "   • Add more training examples to labels.csv",
                "   • Review features.csv for data quality issues"
            ]
    
    @staticmethod
    def format_heuristic_warning() -> str:
        """Format warning about using heuristic labels."""
        return ("Labels file not found. Using heuristic positive/negative pairs for training.")
    
    @staticmethod
    def format_comprehensive_results(
        stats: Dict[str, Any], 
        model_path: str, 
        duplicates_path: str
    ) -> str:
        """Format comprehensive model training results."""
        lines = []
        
        # Header
        lines.append(ModelFormatter.format_training_complete())
        lines.append(ModelFormatter.format_separator())
        
        # Data overview
        lines.extend(ModelFormatter.format_data_overview(stats))
        lines.append("")
        
        # Training labels
        lines.extend(ModelFormatter.format_training_labels(stats))
        lines.append("")
        
        # Model performance
        lines.extend(ModelFormatter.format_model_performance(stats))
        lines.append("")
        
        # Results
        lines.extend(ModelFormatter.format_results(stats))
        lines.append("")
        
        # Files created
        lines.extend(ModelFormatter.format_files_created(model_path, duplicates_path))
        lines.append("")
        
        # Next steps
        high_conf_pairs = stats.get('high_confidence_pairs', 0)
        lines.extend(ModelFormatter.format_next_steps(high_conf_pairs))
        
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
    def format_model_coefficients(stats: Dict[str, Any]) -> List[str]:
        """Format model coefficients for debugging."""
        model_stats = stats.get('model_stats', {})
        coefficients = model_stats.get('coefficients', {})
        
        if not coefficients:
            return []
        
        lines = ["🔍 Model Coefficients:"]
        for feature, coef in coefficients.items():
            direction = "📈" if coef > 0 else "📉"
            lines.append(f"  • {feature:<15} {direction} {coef:+.3f}")
        
        intercept = model_stats.get('intercept', 0)
        lines.append(f"  • intercept         🎯 {intercept:+.3f}")
        
        return lines
