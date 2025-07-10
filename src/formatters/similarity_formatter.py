"""
Terminal output formatting for similarity results.
"""

from __future__ import annotations

from ..core.similarity_engine import SimilarityResult


class SimilarityTerminalFormatter:
    """Formats similarity results for terminal output."""
    
    @staticmethod
    def format_start_message(cleaned_path: str, pairs_path: str) -> None:
        """Format the starting message."""
        print("📏 Starting similarity feature computation...")
        print(f"📂 Cleaned data: {cleaned_path}")
        print(f"📂 Candidate pairs: {pairs_path}")
    
    @staticmethod
    def format_results(result: SimilarityResult, cleaned_path: str, pairs_path: str, features_path: str) -> None:
        """
        Format and print comprehensive similarity results.
        
        Parameters
        ----------
        result : SimilarityResult
            The similarity result to format
        cleaned_path : str
            Path to cleaned data file
        pairs_path : str
            Path to pairs file
        features_path : str
            Path to features output file
        """
        stats = result.stats
        
        print(f"\n📏 Similarity Feature Computation Complete!")
        print(f"─" * 50)
        print(f"📊 Data Overview:")
        print(f"  • Input records:         {stats.input_records:,}")
        print(f"  • Input pairs:           {stats.input_pairs:,}")
        print(f"  • Output features:       {stats.output_features:,}")
        
        # Show which columns were used
        print(f"\n🔧 Feature Computation:")
        print(f"  • Available columns:     {', '.join(stats.columns_used)}")
        
        # Show feature statistics
        if stats.similarity_metrics:
            print(f"  • Features computed:")
            for feature, metrics in stats.similarity_metrics.items():
                print(f"    - {feature:<15} (mean: {metrics['mean']:.3f}, max: {metrics['max']:.3f})")
        
        # Show missing columns warning
        if stats.missing_columns:
            print(f"\n⚠️  Missing Optional Data:")
            for col in stats.missing_columns:
                clean_name = col.replace("_clean", "")
                print(f"  • {clean_name:<15} (will reduce accuracy)")
        
        print(f"\n💾 Files Created:")
        print(f"  • Feature matrix:        {features_path}")
        
        print(f"\n✅ Next step: Train model or run clustering")
        print(f"   Model:      python -m src.cli.model")
        print(f"   Clustering: python -m src.cli.clustering")
    
    @staticmethod
    def format_error(error: Exception) -> None:
        """Format error messages."""
        print(f"\n❌ Similarity computation failed:")
        print(f"  • Error: {str(error)}")
        print(f"  • Check your cleaned data and pairs files")
