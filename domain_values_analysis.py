import pandas as pd
import numpy as np

# Load features to see actual domain values for cluster 4207 records
features = pd.read_csv('data/outputs/features.csv')
agg_features = pd.read_csv('data/outputs/agg_features.csv')
cluster_4207_agg = agg_features[agg_features['cluster_id'] == 4207]

print("DOMAIN VALUES ANALYSIS FOR CLUSTER 4207")
print("=" * 45)
print(f"Cluster 4207 records in agg_features: {len(cluster_4207_agg)}")
print(f"Domain_sim values in cluster 4207:")
print(cluster_4207_agg['domain_sim'].value_counts())
print()

# Check if we can find the original domain pairs that led to these records
print("ORIGINAL DOMAIN SIMILARITY ANALYSIS")
print("=" * 35)

# Get record IDs from cluster 4207
cluster_records = set(cluster_4207_agg.index.astype(str))

# Find features pairs involving these records
cluster_pairs = features[
    (features['record_id_1'].astype(str).isin(cluster_records)) |
    (features['record_id_2'].astype(str).isin(cluster_records))
]

print(f"Feature pairs involving cluster 4207 records: {len(cluster_pairs)}")
print("Domain_sim values in original features:")
print(cluster_pairs['domain_sim'].value_counts().head(10))
print()

# Check if all domain pairs are exact matches (1.0)
exact_domain_matches = cluster_pairs[cluster_pairs['domain_sim'] == 1.0]
print(f"Exact domain matches (1.0) in features: {len(exact_domain_matches)}")

# Sample some exact domain matches to see if they're actually different domains
if len(exact_domain_matches) > 0:
    print("\nSample exact domain matches:")
    sample_pairs = exact_domain_matches[['domain_clean_1', 'domain_clean_2']].head(10)
    for i, row in sample_pairs.iterrows():
        print(f"  {row['domain_clean_1']} ←→ {row['domain_clean_2']}")
        
    # Check if these are actually the SAME domains
    same_domains = exact_domain_matches[exact_domain_matches['domain_clean_1'] == exact_domain_matches['domain_clean_2']]
    diff_domains = exact_domain_matches[exact_domain_matches['domain_clean_1'] != exact_domain_matches['domain_clean_2']]
    
    print(f"\nExact matches with SAME domains: {len(same_domains)}")
    print(f"Exact matches with DIFFERENT domains: {len(diff_domains)}")
    
    if len(diff_domains) > 0:
        print("⚠️ WARNING: Found exact domain matches between DIFFERENT domains!")
        print("Sample different domain pairs with 1.0 similarity:")
        for i, row in diff_domains[['domain_clean_1', 'domain_clean_2']].head(5).iterrows():
            print(f"  {row['domain_clean_1']} ←→ {row['domain_clean_2']}")
    else:
        print("✅ All exact domain matches are between records with the SAME domain")
