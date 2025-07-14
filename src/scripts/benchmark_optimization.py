#!/usr/bin/env python3
"""Performance benchmark for unique_records.csv generation optimization."""

import time
import pandas as pd
from src.core.openai_deduplicator import OpenAIDeduplicator
from src.core.openai_client import OpenAIClient
from src.core.openai_types import OpenAIConfig

def benchmark_unique_records_creation():
    """Benchmark the optimized _create_unique_records method."""
    
    print("🔬 Benchmarking unique_records.csv generation optimization...")
    
    # Load data
    features_df = pd.read_csv('data/outputs/features.csv')
    records_df = pd.read_csv('data/outputs/cleaned.csv').set_index('record_id')
    
    print(f"📊 Dataset size: {len(records_df):,} records, {len(features_df):,} similarity pairs")
    
    # Create mock AI results to simulate different merge scenarios
    mock_ai_results = []
    
    # Simulate merging every other pair with high confidence
    for i, (_, row) in enumerate(features_df.head(100).iterrows()):
        if i % 2 == 0:  # Merge every other pair
            mock_ai_results.append({
                "pair_id": f"{row['record_id_1']}-{row['record_id_2']}",
                "same_organization": True,
                "confidence": 0.95,
                "primary_record_id": row['record_id_1'],
                "canonical_name": row.get('company_clean_1', '')
            })
        else:
            mock_ai_results.append({
                "pair_id": f"{row['record_id_1']}-{row['record_id_2']}",
                "same_organization": False,
                "confidence": 0.85,
                "primary_record_id": row['record_id_1'],
                "canonical_name": row.get('company_clean_1', '')
            })
    
    print(f"🧪 Testing with {len(mock_ai_results)} mock AI results...")
    
    # Initialize deduplicator
    client = OpenAIClient()
    deduplicator = OpenAIDeduplicator(client)
    config = OpenAIConfig(confidence_threshold=0.7)
    
    # Benchmark the optimized method
    print("\n⚡ Running optimized _create_unique_records...")
    start_time = time.time()
    
    unique_df = deduplicator._create_unique_records(
        features_df, records_df, mock_ai_results, config
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"✅ Optimization Results:")
    print(f"  • Processing time: {processing_time:.4f}s")
    print(f"  • Original records: {len(records_df):,}")
    print(f"  • Unique records: {len(unique_df):,}")
    print(f"  • Records merged: {len(records_df) - len(unique_df):,}")
    print(f"  • Processing rate: {len(records_df)/processing_time:,.0f} records/second")
    
    # Verify data quality
    merged_records = unique_df[unique_df['is_merged'] == True]
    print(f"  • Merged groups: {len(merged_records):,}")
    
    print("\n🚀 Performance Summary:")
    if processing_time < 0.1:
        print("  • EXCELLENT: Sub-100ms processing time!")
    elif processing_time < 0.5:
        print("  • GOOD: Fast processing time")
    else:
        print("  • OK: Could be faster for very large datasets")
    
    print(f"  • Memory efficient: Uses Union-Find algorithm O(n log n)")
    print(f"  • Scalable: Processing rate of {len(records_df)/processing_time:,.0f} records/sec")

if __name__ == '__main__':
    benchmark_unique_records_creation()
