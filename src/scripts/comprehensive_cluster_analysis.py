#!/usr/bin/env python3
"""
Cluster Analysis Suite
Combines functionality for analyzing specific clusters and cluster patterns
"""
import pandas as pd
import numpy as np
import argparse


def analyze_large_clusters(top_n=20):
    """Analyze the largest clusters in the dataset"""
    print(f"TOP {top_n} LARGEST CLUSTERS ANALYSIS")
    print("=" * 40)
    
    try:
        df = pd.read_csv('data/outputs/clusters_final.csv')
        
        # Get cluster sizes
        cluster_sizes = df['cluster_id'].value_counts().head(top_n)
        
        print(f"Largest clusters:")
        for cluster_id, size in cluster_sizes.items():
            cluster_data = df[df['cluster_id'] == cluster_id]
            unique_domains = cluster_data['Domain'].nunique()
            unique_companies = cluster_data['Name'].nunique()
            print(f"Cluster {cluster_id}: {size} records, {unique_domains} domains, {unique_companies} companies")
        
        # Identify clusters that might need subdivision
        print(f"\nClusters potentially needing subdivision (>10 records):")
        large_clusters = cluster_sizes[cluster_sizes > 10]
        for cluster_id, size in large_clusters.items():
            cluster_data = df[df['cluster_id'] == cluster_id]
            unique_domains = cluster_data['Domain'].nunique()
            if unique_domains > 1:
                print(f"⚠️  Cluster {cluster_id}: {size} records, {unique_domains} domains (MIXED)")
            else:
                print(f"✅ Cluster {cluster_id}: {size} records, {unique_domains} domain (PURE)")
                
    except FileNotFoundError:
        print("clusters_final.csv not found. Run clustering first.")


def analyze_specific_cluster_detailed(cluster_id):
    """Detailed analysis of a specific cluster including domain boosting analysis"""
    print(f"DETAILED ANALYSIS: CLUSTER {cluster_id}")
    print("=" * 40)
    
    try:
        # Load main cluster data
        df = pd.read_csv('data/outputs/clusters_final.csv')
        cluster_data = df[df['cluster_id'] == cluster_id]
        
        if len(cluster_data) == 0:
            print(f"No records found for cluster {cluster_id}")
            return
            
        print(f"Records in cluster: {len(cluster_data)}")
        print(f"Unique domains: {cluster_data['Domain'].nunique()}")
        print(f"Unique companies: {cluster_data['Name'].nunique()}")
        
        # Domain distribution
        print(f"\nDomain distribution:")
        domain_counts = cluster_data['Domain'].value_counts()
        for domain, count in domain_counts.items():
            print(f"  {domain}: {count} records")
        
        # Sample records by domain
        print(f"\nSample records by domain:")
        for domain in domain_counts.index[:5]:  # Top 5 domains
            domain_records = cluster_data[cluster_data['Domain'] == domain]
            print(f"\n{domain} records:")
            sample = domain_records[['record_id', 'Name']].head(3)
            for _, record in sample.iterrows():
                print(f"  ID {record['record_id']}: {record['Name']}")
        
        # Check for aggregated features and domain_sim values
        try:
            agg_features = pd.read_csv('data/outputs/agg_features.csv')
            cluster_agg = agg_features[agg_features['cluster_id'] == cluster_id]
            
            if len(cluster_agg) > 0:
                print(f"\nAggregated features analysis:")
                print(f"Records in agg_features: {len(cluster_agg)}")
                print(f"Domain_sim values:")
                domain_sim_counts = cluster_agg['domain_sim'].value_counts()
                for value, count in domain_sim_counts.items():
                    print(f"  {value}: {count} records")
                
                # Check for artificially boosted values
                high_values = cluster_agg[cluster_agg['domain_sim'] > 100]
                if len(high_values) > 0:
                    print(f"\n⚠️  Found {len(high_values)} records with domain_sim > 100 (potentially boosted)")
                    
        except FileNotFoundError:
            print("\nagg_features.csv not found - cannot analyze domain_sim values")
        
        # Check original features for domain similarity patterns
        try:
            features = pd.read_csv('data/outputs/features.csv')
            cluster_record_ids = set(cluster_data['record_id'].astype(str))
            
            # Find feature pairs involving these records
            cluster_pairs = features[
                (features['record_id_1'].astype(str).isin(cluster_record_ids)) |
                (features['record_id_2'].astype(str).isin(cluster_record_ids))
            ]
            
            if len(cluster_pairs) > 0:
                print(f"\nOriginal feature pairs involving cluster records: {len(cluster_pairs)}")
                print(f"Domain_sim distribution in original features:")
                original_domain_sim = cluster_pairs['domain_sim'].value_counts()
                for value, count in original_domain_sim.head(10).items():
                    print(f"  {value}: {count} pairs")
                    
        except FileNotFoundError:
            print("\nfeatures.csv not found - cannot analyze original features")
            
    except FileNotFoundError:
        print("clusters_final.csv not found. Run clustering first.")


def subdivision_candidates():
    """Identify clusters that are candidates for subdivision"""
    print("SUBDIVISION CANDIDATES ANALYSIS")
    print("=" * 35)
    
    try:
        df = pd.read_csv('data/outputs/clusters_final.csv')
        
        # Find clusters with multiple domains
        candidates = []
        for cluster_id in df['cluster_id'].unique():
            cluster_data = df[df['cluster_id'] == cluster_id]
            unique_domains = cluster_data['Domain'].nunique()
            cluster_size = len(cluster_data)
            
            if unique_domains > 1 and cluster_size > 5:  # Mixed clusters with reasonable size
                candidates.append({
                    'cluster_id': cluster_id,
                    'size': cluster_size,
                    'domains': unique_domains,
                    'domain_list': list(cluster_data['Domain'].unique())
                })
        
        candidates.sort(key=lambda x: x['size'], reverse=True)
        
        print(f"Found {len(candidates)} subdivision candidates:")
        for candidate in candidates[:10]:  # Top 10
            domains_str = ', '.join(candidate['domain_list'][:3])
            if len(candidate['domain_list']) > 3:
                domains_str += f" (and {len(candidate['domain_list'])-3} more)"
            print(f"Cluster {candidate['cluster_id']}: {candidate['size']} records, "
                  f"{candidate['domains']} domains ({domains_str})")
        
        return candidates
        
    except FileNotFoundError:
        print("clusters_final.csv not found. Run clustering first.")
        return []


def cluster_statistics():
    """Generate comprehensive cluster statistics"""
    print("COMPREHENSIVE CLUSTER STATISTICS")
    print("=" * 35)
    
    try:
        df = pd.read_csv('data/outputs/clusters_final.csv')
        
        total_records = len(df)
        total_clusters = df['cluster_id'].nunique()
        total_domains = df['Domain'].nunique()
        
        print(f"Dataset Overview:")
        print(f"  Total records: {total_records:,}")
        print(f"  Total clusters: {total_clusters:,}")
        print(f"  Total unique domains: {total_domains:,}")
        print(f"  Average records per cluster: {total_records/total_clusters:.2f}")
        print(f"  Average records per domain: {total_records/total_domains:.2f}")
        
        # Cluster size distribution
        cluster_sizes = df['cluster_id'].value_counts()
        print(f"\nCluster size distribution:")
        print(f"  Singleton clusters (1 record): {(cluster_sizes == 1).sum()}")
        print(f"  Small clusters (2-5 records): {((cluster_sizes >= 2) & (cluster_sizes <= 5)).sum()}")
        print(f"  Medium clusters (6-10 records): {((cluster_sizes >= 6) & (cluster_sizes <= 10)).sum()}")
        print(f"  Large clusters (11+ records): {(cluster_sizes >= 11).sum()}")
        print(f"  Largest cluster size: {cluster_sizes.max()}")
        
        # Domain purity analysis
        perfect_clusters = 0
        mixed_clusters = 0
        
        for cluster_id in df['cluster_id'].unique():
            cluster_data = df[df['cluster_id'] == cluster_id]
            unique_domains = cluster_data['Domain'].nunique()
            
            if unique_domains == 1:
                perfect_clusters += 1
            else:
                mixed_clusters += 1
        
        print(f"\nDomain purity:")
        print(f"  Perfect clusters (1 domain): {perfect_clusters:,} ({perfect_clusters/total_clusters*100:.2f}%)")
        print(f"  Mixed clusters (multiple domains): {mixed_clusters:,} ({mixed_clusters/total_clusters*100:.2f}%)")
        
    except FileNotFoundError:
        print("clusters_final.csv not found. Run clustering first.")


def main():
    parser = argparse.ArgumentParser(description='Cluster Analysis Suite')
    parser.add_argument('--large', action='store_true',
                       help='Analyze largest clusters')
    parser.add_argument('--cluster', type=int,
                       help='Detailed analysis of specific cluster')
    parser.add_argument('--subdivision', action='store_true',
                       help='Find subdivision candidates')
    parser.add_argument('--stats', action='store_true',
                       help='Generate comprehensive statistics')
    parser.add_argument('--all', action='store_true',
                       help='Run all analyses')
    
    args = parser.parse_args()
    
    if not any([args.large, args.cluster, args.subdivision, args.stats, args.all]):
        args.stats = True  # Default to stats if no specific analysis chosen
    
    if args.all or args.stats:
        cluster_statistics()
        print()
    
    if args.all or args.large:
        analyze_large_clusters()
        print()
    
    if args.all or args.subdivision:
        subdivision_candidates()
        print()
    
    if args.cluster:
        analyze_specific_cluster_detailed(args.cluster)


if __name__ == "__main__":
    main()
