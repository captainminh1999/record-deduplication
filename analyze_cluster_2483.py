#!/usr/bin/env python3
"""
Quick analysis of cluster 2483 to check for boosted domain values
"""
import pandas as pd
import numpy as np

def analyze_cluster_2483():
    print("CLUSTER 2483 ANALYSIS")
    print("====================")
    
    # Load data
    try:
        df_agg = pd.read_csv("data/outputs/agg_features.csv")
        print(f"Loaded agg_features.csv: {len(df_agg)} records")
        
        # Find records in cluster 2483
        cluster_2483 = df_agg[df_agg['cluster_id'] == 2483]
        print(f"Cluster 2483 records: {len(cluster_2483)}")
        
        if len(cluster_2483) > 0:
            # Check for domain_sim column
            if 'domain_sim' in cluster_2483.columns:
                domain_values = cluster_2483['domain_sim']
                print(f"\nDomain_sim values in cluster 2483:")
                print(domain_values.value_counts().head(10))
                
                # Check for boosted values
                boosted_count = len(domain_values[domain_values >= 5000.0])
                print(f"Records with boosted values (>=5000.0): {boosted_count}")
                
                # Check unique domains
                unique_domains = cluster_2483['domain_clean_1'].nunique()
                print(f"Unique domains: {unique_domains}")
                
                if unique_domains > 1:
                    print("\nTop 10 domains:")
                    print(cluster_2483['domain_clean_1'].value_counts().head(10))
            else:
                print("No domain_sim column found")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_cluster_2483()
