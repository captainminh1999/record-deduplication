#!/usr/bin/env python3

import pandas as pd
import numpy as np

def check_domain_scatter_details():
    # Load the clustering results
    df = pd.read_csv('data/outputs/clusters.csv')
    
    # Check specific examples you mentioned
    examples = [
        ["2aa974631b49a910b37020e6b04bcbb6", "b669b6641bde7510b37020e6b04bcb7c"],
        ["b8e8c1681bedb410a18b311d0d4bcb39", "ea5afa29831ed2d058f18798beaad3f3", "93de2bb7c37922901a9c742a05013147"],
        ["82cafcc41b7380108321311d0d4bcbc7", "c81052201bcf6850d5d38738dc4bcb46", "3c4d22adc3a392d01a9c742a050131c7"],
        ["641b492783758a5058f18798beaad3c1", "9c489edb33b34210a0c67c845d5c7b0e"]
    ]
    
    print("=== CHECKING SPECIFIC DOMAIN EXAMPLES ===")
    for i, id_list in enumerate(examples, 1):
        print(f"\nExample {i}: IDs {id_list}")
        
        # Find these records in the dataset
        example_records = df[df['ID'].isin(id_list)]
        
        if len(example_records) > 0:
            print(f"Found {len(example_records)} records:")
            for _, row in example_records.iterrows():
                print(f"  ID: {row['ID'][:16]}... | Domain: {row['Domain']} | Cluster: {row['cluster_id']}")
            
            # Check if they're in the same cluster
            clusters = example_records['cluster_id'].unique()
            if len(clusters) == 1:
                print(f"  ✅ ALL in same cluster: {clusters[0]}")
            else:
                print(f"  ❌ SCATTERED across {len(clusters)} clusters: {clusters}")
                
                # Check if they have the same domain
                domains = example_records['Domain'].unique()
                if len(domains) == 1:
                    print(f"  🔥 PROBLEM: Same domain '{domains[0]}' but different clusters!")
                else:
                    print(f"  ℹ️ Different domains: {domains}")
        else:
            print(f"  ⚠️ No records found with these IDs")

if __name__ == "__main__":
    check_domain_scatter_details()
