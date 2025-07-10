"""
Terminal output formatting for preprocessing results.
"""

from __future__ import annotations

from ..core.preprocess_engine import PreprocessResult


class PreprocessTerminalFormatter:
    """Formats preprocessing results for terminal output."""
    
    @staticmethod
    def format_results(result: PreprocessResult, input_path: str, output_path: str, audit_path: str) -> None:
        """
        Format and print comprehensive preprocessing results.
        
        Parameters
        ----------
        result : PreprocessResult
            The preprocessing result to format
        input_path : str
            Path to input file
        output_path : str
            Path to output file  
        audit_path : str
            Path to audit file
        """
        stats = result.stats
        
        print(f"\n🧹 Data Preprocessing Complete!")
        print(f"─" * 50)
        print(f"📊 Data Overview:")
        print(f"  • Input records:         {stats.initial_rows:,}")
        print(f"  • Output records:        {stats.final_rows:,}")
        
        # Build available columns list for display
        available_cols = []
        if stats.domain_key:
            available_cols.append("domain")
        # Add phone and address if they were detected (we'd need to track this in stats)
        
        print(f"  • Available columns:     {len(available_cols) + 2} (company, record_id" + 
              (f", {', '.join(available_cols)}" if available_cols else "") + ")")
        print(f"  • Company normalization: {'GPT-4o-mini' if 'use_openai' in str(result) else 'Rule-based'}")
        print(f"  • Unique companies:      {stats.unique_companies_before:,} → {stats.unique_companies_after:,}")
        
        if stats.empty_company_names > 0:
            print(f"  • Empty after cleaning:  {stats.empty_company_names:,} companies")
        
        print(f"\n📈 Quality Metrics:")
        print(f"  • Missing companies:     {stats.missing_company:,} ({stats.missing_company/stats.initial_rows:.1%})")
        
        if stats.domain_key:
            print(f"  • Missing domains:       {stats.missing_domain:,} ({stats.missing_domain/stats.initial_rows:.1%})")
        else:
            print(f"  • Missing domains:       {stats.initial_rows:,} (no domain column)")
        
        if stats.duplicates_removed > 0:
            print(f"\n🔍 Duplicates Found:")
            print(f"  • Company duplicates:    {stats.dup_by_company:,}")
            if stats.domain_key:
                print(f"  • Domain duplicates:     {stats.dup_by_domain:,}")
            print(f"  • Total removed:         {stats.duplicates_removed:,}")
            print(f"  • Saved to:              {audit_path}")
        else:
            print(f"\n✅ No duplicates found!")
        
        print(f"\n💾 Files Created:")
        print(f"  • Cleaned data:          {output_path}")
        if stats.duplicates_removed > 0:
            print(f"  • Removed duplicates:    {audit_path}")
        
        print(f"\n✅ Next step: Generate candidate pairs")
        print(f"   Command: python -m src.cli.blocking data/outputs/cleaned.csv")
    
    @staticmethod
    def format_start_message(input_path: str, use_openai: bool = False) -> None:
        """Format the starting message."""
        print("🧹 Starting data preprocessing...")
        print(f"📂 Input file: {input_path}")
        if use_openai:
            print("🤖 OpenAI translation: Enabled")
    
    @staticmethod
    def format_error(error: Exception) -> None:
        """Format error messages."""
        print(f"\n❌ Preprocessing failed:")
        print(f"  • Error: {str(error)}")
        print(f"  • Check your input file format and column names")
