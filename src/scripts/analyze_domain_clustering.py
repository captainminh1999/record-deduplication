#!/usr/bin/env python3
"""
Analyze how well records with the same domain are clustered together.
"""

import pandas as pd
from collections import defaultdict, Counter

def analyze_domain_clustering():
    """Analyze clustering quality for records with identical domains."""
    
    # Load the clustered data
    df = pd.read_csv('data/outputs/clusters.csv')
    
    print(f"📊 Analyzing clustering quality for {len(df)} records...")
    
    # Group by domain and analyze clustering
    domain_clusters = defaultdict(list)
    
    for _, row in df.iterrows():
        domain = row['Domain']
        cluster_id = row['cluster_id']
        if pd.notna(domain) and domain.strip():
            domain_clusters[domain].append(cluster_id)
    
    # Analyze domains with multiple records
    multi_record_domains = {domain: clusters for domain, clusters 
                           in domain_clusters.items() if len(clusters) > 1}
    
    print(f"\n🔍 Found {len(multi_record_domains)} domains with multiple records")
    
    # Calculate clustering statistics
    perfectly_clustered = 0
    split_domains = 0
    total_domains_analyzed = 0
    cluster_spread_stats = []
    
    print("\n📈 Domain Clustering Analysis:")
    print("=" * 60)
    
    for domain, clusters in multi_record_domains.items():
        unique_clusters = set(clusters)
        total_domains_analyzed += 1
        
        if len(unique_clusters) == 1:
            perfectly_clustered += 1
            cluster_spread = 0
        else:
            split_domains += 1
            # Calculate spread as difference between max and min cluster IDs
            cluster_spread = max(clusters) - min(clusters)
            cluster_spread_stats.append(cluster_spread)
            
            # Show first few examples of split domains
            if split_domains <= 5:
                print(f"   🔄 {domain}: {len(clusters)} records in clusters {sorted(unique_clusters)}")
    
    print("\n📊 Summary Statistics:")
    print(f"   • Total domains analyzed: {total_domains_analyzed}")
    print(f"   • Perfectly clustered domains: {perfectly_clustered} ({perfectly_clustered/total_domains_analyzed*100:.1f}%)")
    print(f"   • Split domains: {split_domains} ({split_domains/total_domains_analyzed*100:.1f}%)")
    
    if cluster_spread_stats:
        avg_spread = sum(cluster_spread_stats) / len(cluster_spread_stats)
        max_spread = max(cluster_spread_stats)
        print(f"   • Average cluster spread for split domains: {avg_spread:.1f}")
        print(f"   • Maximum cluster spread: {max_spread}")
    
    # Show some examples of well-clustered domains
    print(f"\n✅ Examples of perfectly clustered domains:")
    perfect_examples = [(domain, clusters) for domain, clusters in multi_record_domains.items() 
                       if len(set(clusters)) == 1][:5]
    
    for domain, clusters in perfect_examples:
        cluster_id = clusters[0]
        print(f"   • {domain}: {len(clusters)} records all in cluster {cluster_id}")

if __name__ == "__main__":
    analyze_domain_clustering()
