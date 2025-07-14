#!/usr/bin/env python3
"""Check domain clustering results"""

import pandas as pd

# Load the results
agg_features = pd.read_csv('data/outputs/agg_features.csv')
clusters = pd.read_csv('data/outputs/clusters.csv')

# Find records with perfect domain matches
perfect_domain = agg_features[agg_features['domain_sim'] == 5000.0]
print(f'Records with perfect domain matches: {len(perfect_domain)}')

# Merge with cluster data to see actual domains
perfect_with_clusters = perfect_domain.merge(clusters[['Name', 'Domain']], left_index=True, right_index=True)

# Check if same domains are scattered across different clusters  
print('\nChecking if same domain appears in multiple clusters...')
domain_cluster_mapping = perfect_with_clusters.groupby('Domain')['cluster_id'].nunique().sort_values(ascending=False)
scattered_domains = domain_cluster_mapping[domain_cluster_mapping > 1]
print(f'Domains appearing in multiple clusters: {len(scattered_domains)}')

if len(scattered_domains) > 0:
    print('\nTop 10 scattered domains:')
    print(scattered_domains.head(10))
    
    # Check a specific scattered domain
    scattered_domain = scattered_domains.index[0]
    scattered_records = perfect_with_clusters[perfect_with_clusters['Domain'] == scattered_domain]
    print(f'\nDomain "{scattered_domain}" appears in {domain_cluster_mapping[scattered_domain]} clusters:')
    cluster_dist = scattered_records['cluster_id'].value_counts()
    print(cluster_dist)
    
    print(f'\nSample records for domain "{scattered_domain}":')
    print(scattered_records[['Name', 'Domain', 'cluster_id']].head())
    
    # Check if this is the 1call.co.nz domain from our earlier analysis
    if '1call.co.nz' in domain_cluster_mapping.index:
        call_records = perfect_with_clusters[perfect_with_clusters['Domain'] == '1call.co.nz']
        print(f'\n"1call.co.nz" domain appears in {domain_cluster_mapping["1call.co.nz"]} clusters:')
        print(call_records['cluster_id'].value_counts())
else:
    print('✅ All perfect domain matches are properly grouped within individual clusters!')
