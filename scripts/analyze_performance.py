#!/usr/bin/env python3
"""Performance analysis for unique_records.csv generation."""

import pandas as pd
import time

def analyze_data():
    print('Loading data...')
    start = time.time()
    features_df = pd.read_csv('data/outputs/features.csv')
    records_df = pd.read_csv('data/outputs/cleaned.csv')
    print(f'Data loading took: {time.time() - start:.2f}s')

    print(f'\nData shapes:')
    print(f'  Features: {features_df.shape}')
    print(f'  Records: {records_df.shape}')

    # Check candidate pairs above threshold
    candidate_pairs = features_df[features_df['company_sim'] >= 0.6]
    print(f'\nCandidate pairs above 0.6 threshold: {len(candidate_pairs):,}')

    # Check distribution of similarity scores
    print('\nSimilarity score distribution:')
    print(f'  Min: {features_df["company_sim"].min():.3f}')
    print(f'  Max: {features_df["company_sim"].max():.3f}')
    print(f'  Mean: {features_df["company_sim"].mean():.3f}')
    print(f'  Median: {features_df["company_sim"].median():.3f}')
    print(f'  > 0.8: {len(features_df[features_df["company_sim"] > 0.8]):,}')
    print(f'  > 0.9: {len(features_df[features_df["company_sim"] > 0.9]):,}')
    print(f'  > 0.95: {len(features_df[features_df["company_sim"] > 0.95]):,}')

    # Check very high similarity pairs that could be auto-merged
    very_high_sim = features_df[features_df['company_sim'] > 0.95]
    print(f'\nVery high similarity pairs (>0.95): {len(very_high_sim):,}')
    
    if len(very_high_sim) > 0:
        print('Sample high-similarity pairs:')
        for _, row in very_high_sim.head().iterrows():
            print(f'  {row["company_sim"]:.3f}: {row.get("company_clean_1", "?")} vs {row.get("company_clean_2", "?")}')

if __name__ == '__main__':
    analyze_data()
