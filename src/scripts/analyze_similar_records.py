import pandas as pd

# Load the clusters data
clusters_df = pd.read_csv('data/outputs/clusters.csv')
agg_features_df = pd.read_csv('data/outputs/agg_features.csv')

# Record IDs to analyze
record_groups = [
    ["06a77fa687c76550852d0e170cbb35ad", "e500daec1b8f6850d5d38738dc4bcb85", "4e001eec1b8f6850d5d38738dc4bcbca"],
    ["82cafcc41b7380108321311d0d4bcbc7", "c81052201bcf6850d5d38738dc4bcb46", "3c4d22adc3a392d01a9c742a050131c7"],
    ["0f3b56b01b3665101c3ddc6fcc4bcbe4", "9f2ad3e9c39a9a901a9c742a0501313d", "55a044501b6f5d90b37020e6b04bcbe5", "9998f2061b3139d0d1f74119b04bcb7d", "46dd74981bf1f990d1f74119b04bcb1e"]
]

print("=== CLUSTER ANALYSIS FOR SIMILAR RECORDS ===\n")

for i, group in enumerate(record_groups, 1):
    print(f"Group {i}: Records that should be clustered together")
    print("-" * 60)
    
    for record_id in group:
        # Find cluster assignment
        cluster_info = clusters_df[clusters_df['record_id'] == record_id]
        if len(cluster_info) > 0:
            cluster_id = cluster_info['cluster'].iloc[0]
            name = cluster_info['Name'].iloc[0]
            domain = cluster_info['Domain'].iloc[0] if pd.notna(cluster_info['Domain'].iloc[0]) else "N/A"
            company_clean = cluster_info['company_clean'].iloc[0]
            domain_clean = cluster_info['domain_clean'].iloc[0] if pd.notna(cluster_info['domain_clean'].iloc[0]) else "N/A"
            
            print(f"  Record: {record_id}")
            print(f"    Cluster: {cluster_id}")
            print(f"    Name: {name}")
            print(f"    Domain: {domain}")
            print(f"    Company Clean: {company_clean}")
            print(f"    Domain Clean: {domain_clean}")
            
            # Get features for this record
            features = agg_features_df[agg_features_df['record_id'] == record_id]
            if len(features) > 0:
                print(f"    Features: company_sim={features['company_sim'].iloc[0]:.3f}, domain_sim={features['domain_sim'].iloc[0]:.3f}, phone_exact={features['phone_exact'].iloc[0]:.3f}, address_sim={features['address_sim'].iloc[0]:.3f}")
            print()
        else:
            print(f"  Record {record_id}: NOT FOUND")
    
    # Check if any records in this group are in the same cluster
    group_clusters = []
    for record_id in group:
        cluster_info = clusters_df[clusters_df['record_id'] == record_id]
        if len(cluster_info) > 0:
            group_clusters.append(cluster_info['cluster'].iloc[0])
    
    unique_clusters = set(group_clusters)
    if len(unique_clusters) == 1:
        print(f"  ✅ All records are in the same cluster: {list(unique_clusters)[0]}")
    else:
        print(f"  ❌ Records are split across {len(unique_clusters)} clusters: {sorted(unique_clusters)}")
    
    print("\n" + "="*80 + "\n")
