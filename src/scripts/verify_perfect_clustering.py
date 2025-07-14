#!/usr/bin/env python3

import pandas as pd
from collections import defaultdict

def analyze_domain_clustering(clusters_file):
    """Analyze domain clustering quality"""
    
    df = pd.read_csv(clusters_file)
    
    # Group by domain and check clustering
    domain_clusters = {}
    scattered_domains = []
    perfect_domains = 0
    total_domains = 0
    
    domain_groups = df.groupby('Domain')
    
    for domain, group in domain_groups:
        if pd.isna(domain) or domain == '':
            continue
            
        total_domains += 1
        clusters = group['cluster_id'].unique()
        
        # Remove noise cluster if present
        clusters = [c for c in clusters if c != -1]
        
        if len(clusters) == 1:
            perfect_domains += 1
        elif len(clusters) > 1:
            scattered_domains.append(domain)
            
        domain_clusters[domain] = clusters
    
    print(f'📊 DOMAIN CLUSTERING ANALYSIS')
    print(f'════════════════════════════════════════════════')
    print(f'File: {clusters_file}')
    print(f'Total domains analyzed: {total_domains}')
    print(f'Perfect domain clustering: {perfect_domains}')
    print(f'Scattered domains: {len(scattered_domains)}')
    print(f'Success rate: {(perfect_domains / total_domains * 100):.1f}%')
    print()
    
    if scattered_domains:
        print(f'❌ Remaining scattered domains: {len(scattered_domains)}')
        for domain in scattered_domains[:10]:  # Show first 10
            clusters = domain_clusters[domain]
            count = len(df[df['Domain'] == domain])
            print(f'  {domain}: {count} records in clusters {clusters}')
    else:
        print('🎉 PERFECT! No scattered domains remaining!')
        print('🎯 100.0% domain clustering achieved!')
    
    # Check for noise records
    noise_count = len(df[df['cluster_id'] == -1])
    print(f'\n📋 Noise records: {noise_count}')
    
    return {
        'total_domains': total_domains,
        'perfect_domains': perfect_domains, 
        'scattered_domains': len(scattered_domains),
        'success_rate': perfect_domains / total_domains * 100,
        'noise_records': noise_count
    }

if __name__ == "__main__":
    print("=== NEW HIERARCHICAL CLUSTERING WITH AGGRESSIVE DOMAIN GROUPING ===")
    analyze_domain_clustering('data/outputs/clusters.csv')
    
    print("\n\n=== PREVIOUS RESULTS (FOR COMPARISON) ===")
    analyze_domain_clustering('data/outputs/clusters_rescued.csv')
