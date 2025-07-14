import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('data/outputs/clusters_final.csv')

# Analyze cluster distribution
cluster_sizes = df['cluster_id'].value_counts().sort_values(ascending=False)
print("Top 20 largest clusters:")
print(cluster_sizes.head(20))
print()

# Look at cluster 4207 specifically
cluster_4207 = df[df['cluster_id'] == 4207]
print(f"Cluster 4207 analysis:")
print(f"  Size: {len(cluster_4207)}")
print(f"  Unique domains: {cluster_4207['Domain'].nunique()}")
print(f"  Domain distribution (top 10):")
domain_dist = cluster_4207['Domain'].value_counts()
print(domain_dist.head(10))
print()

# Check if there are other large clusters
large_clusters = cluster_sizes[cluster_sizes > 100]
print(f"Clusters with > 100 records: {len(large_clusters)}")
print("Large clusters:")
for cluster_id, size in large_clusters.head(10).items():
    cluster_data = df[df['cluster_id'] == cluster_id]
    unique_domains = cluster_data['Domain'].nunique()
    print(f"  Cluster {cluster_id}: {size} records, {unique_domains} domains")
