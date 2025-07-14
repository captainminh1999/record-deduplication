#!/usr/bin/env python3
"""Check if domain-aware clustering was used"""

import pandas as pd
import numpy as np

# Load features and clusters
agg_features = pd.read_csv('data/outputs/agg_features.csv')
clusters = pd.read_csv('data/outputs/clusters.csv')

print(f"Total records: {len(agg_features)}")
print(f"Total clusters: {agg_features['cluster_id'].nunique()}")
print()

# Find perfect domain matches
perfect_domain = agg_features[agg_features['domain_sim'] == 5000.0]
print(f"Records with perfect domain matches: {len(perfect_domain)}")

# Merge with actual domain data
perfect_with_domains = perfect_domain.merge(clusters[['Domain']], left_index=True, right_index=True)

# Group by actual domain to see clustering quality
domain_clustering_quality = perfect_with_domains.groupby('Domain').agg({
    'cluster_id': ['nunique', 'count']
}).round(2)

domain_clustering_quality.columns = ['num_clusters', 'total_records']
domain_clustering_quality['clustering_quality'] = domain_clustering_quality['num_clusters'] / domain_clustering_quality['total_records']

print("\nDomain clustering quality (lower is better):")
print("Perfect = 1 cluster per domain, Worst = many clusters per domain")
print()

# Show worst clustered domains (most scattered)
worst_clustered = domain_clustering_quality.sort_values('num_clusters', ascending=False).head(10)
print("Top 10 most scattered domains:")
print(worst_clustered)
print()

# Show some good examples (1 cluster per domain)
good_clustered = domain_clustering_quality[domain_clustering_quality['num_clusters'] == 1]
print(f"Domains with perfect clustering (1 cluster): {len(good_clustered)}")
print(f"Domains with scattered clustering (>1 cluster): {len(domain_clustering_quality) - len(good_clustered)}")
print()

# Overall quality metrics
avg_clusters_per_domain = domain_clustering_quality['num_clusters'].mean()
median_clusters_per_domain = domain_clustering_quality['num_clusters'].median()
print(f"Average clusters per domain: {avg_clusters_per_domain:.2f}")
print(f"Median clusters per domain: {median_clusters_per_domain:.2f}")

# Check if our domain-aware clustering improved anything
print(f"\nClustering quality analysis:")
print(f"- Perfect domains (1 cluster): {len(good_clustered)} ({len(good_clustered)/len(domain_clustering_quality)*100:.1f}%)")
print(f"- Scattered domains (>1 cluster): {len(domain_clustering_quality) - len(good_clustered)} ({(len(domain_clustering_quality) - len(good_clustered))/len(domain_clustering_quality)*100:.1f}%)")
