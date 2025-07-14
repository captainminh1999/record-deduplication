#!/usr/bin/env python3
"""Test domain clustering with minimal cluster size to force more Force strategy usage"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.clustering.hierarchical.adaptive_clusterer_v3 import AdaptiveHierarchicalClusterer
from src.core.clustering.hierarchical.subdivision_engine_v3 import SubdivisionEngineV3
import pandas as pd
import numpy as np

print("Testing domain-aware clustering on small dataset...")

# Load a small subset of data
agg_features = pd.read_csv('data/outputs/agg_features.csv')
clusters = pd.read_csv('data/outputs/clusters.csv')

# Get first 200 records for testing
feature_cols = ['company_sim', 'domain_sim', 'company_domain_product', 'company_domain_sum', 'domain_priority', 'domain_dominance']
subset_size = 200
test_features = agg_features.iloc[:subset_size][feature_cols].copy()
test_records = clusters.iloc[:subset_size].copy()

print(f"Test dataset: {len(test_features)} records")
print(f"Perfect domain matches in test: {np.sum(test_features['domain_sim'] == 5000.0)}")

# Create clusterer with small max cluster size to force many subdivisions
clusterer = AdaptiveHierarchicalClusterer(timeout_seconds=30)
subdivision_engine = SubdivisionEngineV3()
clusterer.subdivision_engine = subdivision_engine

# Prepare data
X = test_features.values
initial_labels = np.zeros(len(X))

print("\nRunning clustering with max_cluster_size=5 to force many Force strategy applications...")

try:
    # Run clustering
    final_labels, stats = clusterer.cluster_with_adaptive_depth(
        X=X,
        initial_labels=initial_labels,
        base_eps=0.5,
        min_samples=2,
        max_cluster_size=5,  # Very small to force Force strategy
        max_absolute_depth=10,
        performance_mode=True
    )
    
    print(f"\nClustering completed!")
    print(f"Final clusters: {len(np.unique(final_labels))}")
    
    # Add cluster results to test data
    test_results = test_records.copy()
    test_results['new_cluster_id'] = final_labels
    
    # Analyze domain clustering in results
    test_features_with_clusters = test_features.copy()
    test_features_with_clusters['new_cluster_id'] = final_labels
    
    # Focus on perfect domain matches
    perfect_matches = test_features_with_clusters[test_features_with_clusters['domain_sim'] == 5000.0]
    if len(perfect_matches) > 0:
        # Merge with domain info
        perfect_with_domains = perfect_matches.merge(test_records[['Domain']], left_index=True, right_index=True)
        
        print(f"\nDomain clustering analysis for {len(perfect_matches)} perfect matches:")
        domain_cluster_quality = perfect_with_domains.groupby('Domain')['new_cluster_id'].nunique()
        scattered_domains = domain_cluster_quality[domain_cluster_quality > 1]
        
        print(f"Domains split across multiple clusters: {len(scattered_domains)}")
        if len(scattered_domains) > 0:
            print("Scattered domains:")
            for domain, num_clusters in scattered_domains.items():
                domain_records = perfect_with_domains[perfect_with_domains['Domain'] == domain]
                cluster_dist = domain_records['new_cluster_id'].value_counts()
                print(f"  {domain}: {num_clusters} clusters - {dict(cluster_dist)}")
    else:
        print("No perfect domain matches in test subset")
        
except Exception as e:
    print(f"Error during clustering: {e}")
    import traceback
    traceback.print_exc()
