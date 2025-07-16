#!/usr/bin/env python3
"""
Run hierarchical clustering directly
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from core.clustering.hierarchical.adaptive_clusterer_v3 import AdaptiveHierarchicalClusterer

def run_hierarchical_clustering():
    print("🚀 RUNNING FRESH HIERARCHICAL CLUSTERING")
    print("========================================")
    
    # Load features
    print("📂 Loading features...")
    features_df = pd.read_csv("data/outputs/features.csv")
    print(f"Loaded {len(features_df)} feature pairs")
    
    # Initialize clusterer
    print("🔧 Initializing hierarchical clusterer...")
    clusterer = AdaptiveHierarchicalClusterer(
        timeout_seconds=300,
        max_cluster_size=10,
        max_depth=20
    )
    
    # Run clustering
    print("🚀 Running hierarchical clustering...")
    try:
        result = clusterer.hierarchical_cluster(
            features_df=features_df,
            eps=0.5,
            max_iterations=50
        )
        
        print(f"📊 Clustering completed!")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Clusters created: {result.get('n_clusters', 0)}")
        print(f"   Records processed: {result.get('n_records', 0)}")
        
        # Save results
        if 'clustered_data' in result:
            result['clustered_data'].to_csv("data/outputs/clusters.csv", index=False)
            print("💾 Saved clusters to: data/outputs/clusters.csv")
            
        if 'aggregated_features' in result:
            result['aggregated_features'].to_csv("data/outputs/agg_features.csv", index=False)
            print("💾 Saved aggregated features to: data/outputs/agg_features.csv")
            
    except Exception as e:
        print(f"❌ Error during clustering: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_hierarchical_clustering()
