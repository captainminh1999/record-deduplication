import pandas as pd

# Load agg_features and analyze domain_sim values
agg_features = pd.read_csv('data/outputs/agg_features.csv')
print("DOMAIN_SIM ANALYSIS")
print("=" * 20)

# Look at distribution of domain_sim values
print("Domain_sim value distribution:")
print(f"Zero values: {(agg_features['domain_sim'] == 0).sum()}")
print(f"Non-zero values: {(agg_features['domain_sim'] > 0).sum()}")
print(f"High values (>100): {(agg_features['domain_sim'] > 100).sum()}")
print(f"Very high values (>1000): {(agg_features['domain_sim'] > 1000).sum()}")
print()

# Look at records with high domain_sim values
high_domain = agg_features[agg_features['domain_sim'] > 100].copy()
print(f"Records with domain_sim > 100: {len(high_domain)}")
if len(high_domain) > 0:
    print("Distribution of high domain_sim values:")
    print(high_domain['domain_sim'].describe())
    print()
    
    print("Cluster distribution of high domain_sim records:")
    cluster_dist = high_domain['cluster_id'].value_counts().head(10)
    print(cluster_dist)
    print()
    
    # Check if cluster 4207 has high domain_sim values
    cluster_4207_high = high_domain[high_domain['cluster_id'] == 4207]
    print(f"Records in cluster 4207 with high domain_sim: {len(cluster_4207_high)}")
    if len(cluster_4207_high) > 0:
        print("Sample domain_sim values in cluster 4207:")
        print(cluster_4207_high['domain_sim'].value_counts().head(10))

# Also check the overall cluster 4207 records
cluster_4207_all = agg_features[agg_features['cluster_id'] == 4207]
print(f"\nCluster 4207 total records: {len(cluster_4207_all)}")
print("Domain_sim statistics for cluster 4207:")
print(cluster_4207_all['domain_sim'].describe())
