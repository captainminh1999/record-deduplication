#!/usr/bin/env python3
"""Analyze the remaining scattered domains to understand why they're not clustering together."""

import pandas as pd
import numpy as np

def analyze_scattered_domains():
    # Read the clusters
    clusters_df = pd.read_csv('data/outputs/clusters.csv')
    
    # Read the features to understand similarity scores
    try:
        features_df = pd.read_csv('data/outputs/agg_features.csv')
        print("Features file loaded successfully")
    except:
        features_df = None
        print("No features file found")
    
    # Check the scattered domains
    scattered_domains = ['axians.com', 'bunnings.com.au', 'cymax.com.au', 'kwm.com', 'perpetual.com.au', 'perthmint.com.au']
    
    for domain in scattered_domains:
        domain_records = clusters_df[clusters_df['Domain'] == domain]
        if len(domain_records) > 0:
            print(f'\n=== {domain} ({len(domain_records)} records) ===')
            print(f'Clusters: {list(domain_records["cluster_id"].unique())}')
            print('Sample records:')
            for _, row in domain_records.iterrows():
                print(f'  Cluster {row["cluster_id"]}: "{row["Name"]}" | Domain: {row["Domain"]}')
                
            # Check if any are in noise cluster (-1)
            noise_records = domain_records[domain_records['cluster_id'] == -1]
            if len(noise_records) > 0:
                print(f'  ⚠️  {len(noise_records)} records in NOISE cluster (-1)')
                
            # Check cluster sizes for this domain
            cluster_counts = domain_records['cluster_id'].value_counts()
            print(f'  Cluster distribution: {dict(cluster_counts)}')

if __name__ == "__main__":
    analyze_scattered_domains()
