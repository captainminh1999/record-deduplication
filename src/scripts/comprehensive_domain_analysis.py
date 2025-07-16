#!/usr/bin/env python3
"""
Comprehensive Domain Analysis Suite
Combines functionality from multiple domain analysis scripts
"""
import pandas as pd
import numpy as np
import argparse


def analyze_domain_distribution():
    """Analyze domain_sim value distribution across the dataset"""
    print("DOMAIN_SIM VALUE DISTRIBUTION ANALYSIS")
    print("=" * 40)
    
    try:
        agg_features = pd.read_csv('data/outputs/agg_features.csv')
        print(f"Loaded agg_features.csv: {len(agg_features)} records")
        
        # Distribution analysis
        print("\nDomain_sim value distribution:")
        print(f"Zero values: {(agg_features['domain_sim'] == 0).sum()}")
        print(f"Non-zero values: {(agg_features['domain_sim'] > 0).sum()}")
        print(f"High values (>100): {(agg_features['domain_sim'] > 100).sum()}")
        print(f"Very high values (>1000): {(agg_features['domain_sim'] > 1000).sum()}")
        
        # Show unique values
        unique_values = sorted(agg_features['domain_sim'].unique())
        print(f"\nUnique domain_sim values: {unique_values}")
        
        # Value counts
        print(f"\nDomain_sim value distribution:")
        print(agg_features['domain_sim'].value_counts().head(10))
        
    except FileNotFoundError:
        print("agg_features.csv not found. Run clustering first.")


def analyze_specific_cluster(cluster_id):
    """Analyze domain composition of a specific cluster"""
    print(f"CLUSTER {cluster_id} DOMAIN ANALYSIS")
    print("=" * 40)
    
    try:
        df = pd.read_csv('data/outputs/clusters_final.csv')
        cluster_data = df[df['cluster_id'] == cluster_id]
        
        if len(cluster_data) == 0:
            print(f"No records found for cluster {cluster_id}")
            return
            
        print(f"Records in cluster {cluster_id}: {len(cluster_data)}")
        print(f"Unique domains: {cluster_data['Domain'].nunique()}")
        print(f"Unique companies: {cluster_data['Name'].nunique()}")
        
        # Domain distribution
        print(f"\nDomain distribution in cluster {cluster_id}:")
        domain_counts = cluster_data['Domain'].value_counts()
        print(domain_counts.head(10))
        
        # Sample records
        print(f"\nSample records from cluster {cluster_id}:")
        sample = cluster_data[['record_id', 'Name', 'Domain']].head(10)
        print(sample.to_string())
        
        # Check for domain_sim values if available
        try:
            agg_features = pd.read_csv('data/outputs/agg_features.csv')
            cluster_agg = agg_features[agg_features['cluster_id'] == cluster_id]
            if len(cluster_agg) > 0:
                print(f"\nDomain_sim values in cluster {cluster_id}:")
                print(cluster_agg['domain_sim'].value_counts())
        except FileNotFoundError:
            pass
            
    except FileNotFoundError:
        print("clusters_final.csv not found. Run clustering first.")


def analyze_domain_quality():
    """Analyze overall domain clustering quality"""
    print("DOMAIN CLUSTERING QUALITY ANALYSIS")
    print("=" * 35)
    
    try:
        df = pd.read_csv('data/outputs/clusters_final.csv')
        
        # Overall statistics
        total_records = len(df)
        total_clusters = df['cluster_id'].nunique()
        total_domains = df['Domain'].nunique()
        
        print(f"Total records: {total_records}")
        print(f"Total clusters: {total_clusters}")
        print(f"Total unique domains: {total_domains}")
        
        # Cluster purity analysis
        perfect_clusters = 0
        mixed_clusters = 0
        cluster_domain_stats = []
        
        for cluster_id in df['cluster_id'].unique():
            cluster_data = df[df['cluster_id'] == cluster_id]
            unique_domains = cluster_data['Domain'].nunique()
            cluster_size = len(cluster_data)
            
            cluster_domain_stats.append({
                'cluster_id': cluster_id,
                'size': cluster_size,
                'unique_domains': unique_domains,
                'is_perfect': unique_domains == 1
            })
            
            if unique_domains == 1:
                perfect_clusters += 1
            else:
                mixed_clusters += 1
        
        print(f"\nCluster purity:")
        print(f"Perfect clusters (1 domain): {perfect_clusters}")
        print(f"Mixed clusters (multiple domains): {mixed_clusters}")
        print(f"Perfect clustering rate: {perfect_clusters/total_clusters*100:.2f}%")
        
        # Show problematic clusters
        if mixed_clusters > 0:
            print(f"\nLargest mixed clusters:")
            mixed_df = pd.DataFrame(cluster_domain_stats)
            mixed_df = mixed_df[~mixed_df['is_perfect']].sort_values('size', ascending=False)
            print(mixed_df.head(10).to_string())
        
    except FileNotFoundError:
        print("clusters_final.csv not found. Run clustering first.")


def analyze_domain_vs_clean():
    """Compare Domain vs domain_clean fields"""
    print("DOMAIN vs DOMAIN_CLEAN COMPARISON")
    print("=" * 35)
    
    try:
        df = pd.read_csv('data/outputs/clusters_final.csv')
        
        if 'domain_clean' not in df.columns:
            print("domain_clean column not found in dataset")
            return
            
        # Check differences
        domain_matches = (df['Domain'] == df['domain_clean']).sum()
        total_records = len(df)
        
        print(f"Records where Domain == domain_clean: {domain_matches}")
        print(f"Records where Domain != domain_clean: {total_records - domain_matches}")
        print(f"Match rate: {domain_matches/total_records*100:.2f}%")
        
        # Show differences
        different_domains = df[df['Domain'] != df['domain_clean']]
        if len(different_domains) > 0:
            print(f"\nSample differences:")
            sample_diff = different_domains[['Domain', 'domain_clean']].head(10)
            print(sample_diff.to_string())
        
    except FileNotFoundError:
        print("clusters_final.csv not found. Run clustering first.")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Domain Analysis')
    parser.add_argument('--distribution', action='store_true', 
                       help='Analyze domain_sim value distribution')
    parser.add_argument('--quality', action='store_true',
                       help='Analyze domain clustering quality')
    parser.add_argument('--cluster', type=int,
                       help='Analyze specific cluster by ID')
    parser.add_argument('--compare-clean', action='store_true',
                       help='Compare Domain vs domain_clean fields')
    parser.add_argument('--all', action='store_true',
                       help='Run all analyses')
    
    args = parser.parse_args()
    
    if not any([args.distribution, args.quality, args.cluster, args.compare_clean, args.all]):
        args.all = True  # Default to all if no specific analysis chosen
    
    if args.all or args.distribution:
        analyze_domain_distribution()
        print()
    
    if args.all or args.quality:
        analyze_domain_quality()
        print()
    
    if args.all or args.compare_clean:
        analyze_domain_vs_clean()
        print()
    
    if args.cluster:
        analyze_specific_cluster(args.cluster)
        print()


if __name__ == "__main__":
    main()
