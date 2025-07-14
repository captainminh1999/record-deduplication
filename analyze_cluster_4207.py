import pandas as pd

df = pd.read_csv('data/outputs/clusters_final.csv')
cluster_4207 = df[df['cluster_id'] == 4207]

print(f'Cluster 4207 analysis:')
print(f'Total records: {len(cluster_4207)}')
print(f'Unique domains: {cluster_4207["Domain"].nunique()}')
print(f'Domain value counts (top 10):')
print(cluster_4207['Domain'].value_counts().head(10))
print()
print('Sample records from different domains:')
domains = cluster_4207['Domain'].unique()[:5]
for domain in domains:
    sample = cluster_4207[cluster_4207['Domain'] == domain][['Name', 'Domain']].iloc[0]
    print(f'  {domain}: {sample["Name"]}')
