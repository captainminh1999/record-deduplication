#!/usr/bin/env python3

import pandas as pd

def analyze_cymax_domain():
    """Analyze the remaining scattered cymax.com.au domain"""
    
    rescued_df = pd.read_csv('data/outputs/clusters_rescued.csv')
    features_df = pd.read_csv('data/outputs/features.csv')
    
    # Look at cymax.com.au records
    cymax_records = rescued_df[rescued_df['Domain'] == 'cymax.com.au']
    print('=== cymax.com.au Records ===')
    for _, record in cymax_records.iterrows():
        print(f'Cluster {record["cluster_id"]}: "{record["Name"]}" | Domain: {record["Domain"]}')
    
    print()
    
    # Check features between cymax records
    cymax_ids = cymax_records['record_id'].tolist()
    print('=== Features between cymax records ===')
    for i, id1 in enumerate(cymax_ids):
        for j, id2 in enumerate(cymax_ids[i+1:], i+1):
            # Find features between these two records
            pair_features = features_df[
                ((features_df['record_id_1'] == id1) & (features_df['record_id_2'] == id2)) |
                ((features_df['record_id_1'] == id2) & (features_df['record_id_2'] == id1))
            ]
            
            if len(pair_features) > 0:
                for _, feature in pair_features.iterrows():
                    print(f'{feature["company_clean_1"]} <-> {feature["company_clean_2"]}')
                    print(f'  Company: {feature["company_sim"]:.3f} | Domain: {feature["domain_sim"]:.3f}')
                    print()
    
    return cymax_records

if __name__ == "__main__":
    analyze_cymax_domain()
