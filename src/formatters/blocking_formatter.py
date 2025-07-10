"""
Terminal output formatting for blocking results.
"""

from __future__ import annotations

from ..core.blocking_engine import BlockingResult


class BlockingTerminalFormatter:
    """Formats blocking results for terminal output."""
    
    @staticmethod
    def format_start_message(input_path: str) -> None:
        """Format the starting message."""
        print("🔗 Starting candidate pair generation...")
        print(f"📂 Input file: {input_path}")
    
    @staticmethod
    def format_results(result: BlockingResult, input_path: str, output_path: str) -> None:
        """
        Format and print comprehensive blocking results.
        
        Parameters
        ----------
        result : BlockingResult
            The blocking result to format
        input_path : str
            Path to input file
        output_path : str
            Path to output file
        """
        stats = result.stats
        
        print(f"\n🔗 Candidate Pair Generation Complete!")
        print(f"─" * 50)
        print(f"📊 Data Overview:")
        print(f"  • Input records:         {stats.input_records:,}")
        print(f"  • Generated pairs:       {stats.output_pairs:,}")
        print(f"  • Total possible pairs:  {stats.total_possible_pairs:,}")
        print(f"  • Reduction ratio:       {stats.reduction_ratio:.1%}")
        
        print(f"\n🔧 Blocking Strategy:")
        print(f"  • Available fields:      {', '.join(stats.available_columns)}")
        print(f"  • Methods used:          {', '.join(stats.used_methods) if stats.used_methods else 'fuzzy company only'}")
        
        # Warnings and recommendations
        if stats.output_pairs == 0:
            print(f"\n⚠️  Warning: No candidate pairs generated!")
            print(f"   • This might indicate no records share blocking keys")
            print(f"   • Consider more lenient blocking or data quality review")
        elif stats.output_pairs > 100000:
            print(f"\n⚠️  Warning: Large number of pairs generated ({stats.output_pairs:,})")
            print(f"   • This might slow down similarity computation")
            print(f"   • Consider more aggressive blocking")
        else:
            print(f"\n✅ Good pair count for similarity analysis")
        
        print(f"\n💾 Files Created:")
        print(f"  • Candidate pairs:       {output_path}")
        
        print(f"\n✅ Next step: Compute similarity features")
        print(f"   Command: python -m src.cli.similarity data/outputs/cleaned.csv data/outputs/pairs.csv")
    
    @staticmethod
    def format_error(error: Exception) -> None:
        """Format error messages."""
        print(f"\n❌ Blocking failed:")
        print(f"  • Error: {str(error)}")
        print(f"  • Check your cleaned data file and column names")
