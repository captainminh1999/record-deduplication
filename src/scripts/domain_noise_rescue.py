#!/usr/bin/env python3
"""Domain Noise Rescue Function

This module rescues noise records (-1) that should be in domain clusters.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def rescue_domain_noise_records(clustered_df: pd.DataFrame, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Rescue noise records (-1) that have 85%+ domain similarity with existing clusters.
    
    This fixes the fundamental issue where DBSCAN assigns records to noise (-1)
    but they should be grouped with their domain matches.
    
    Args:
        clustered_df: DataFrame with cluster assignments
        features_df: DataFrame with similarity features
        
    Returns:
        (updated_clustered_df, rescue_stats)
    """
    rescue_stats = {
        "noise_records_before": 0,
        "noise_records_after": 0, 
        "records_rescued": 0,
        "domains_fixed": [],
        "rescue_details": []
    }
    
    # Find noise records
    noise_mask = clustered_df['cluster_id'] == -1
    noise_records = clustered_df[noise_mask].copy()
    rescue_stats["noise_records_before"] = len(noise_records)
    
    if len(noise_records) == 0:
        print("    [DOMAIN-RESCUE] 🎉 No noise records found - perfect clustering!")
        return clustered_df, rescue_stats
    
    print(f"    [DOMAIN-RESCUE] 🔍 Found {len(noise_records)} noise records to analyze")
    
    # Get domain similarity column (usually index 0 or 1)
    domain_col = None
    for col in features_df.columns:
        if 'domain' in col.lower() and 'sim' in col.lower():
            domain_col = col
            break
    
    if domain_col is None:
        print("    [DOMAIN-RESCUE] ❌ No domain similarity column found")
        return clustered_df, rescue_stats
    
    print(f"    [DOMAIN-RESCUE] 📊 Using domain column: {domain_col}")
    print(f"    [DOMAIN-RESCUE] 🔧 Features columns: {list(features_df.columns)}")
    
    # For each noise record, check if it should be rescued
    updated_clustered_df = clustered_df.copy()
    
    for noise_idx, noise_record in noise_records.iterrows():
        print(f"    [DOMAIN-RESCUE] 🔍 Processing record: {noise_record['record_id']} | Domain: {noise_record['Domain']}")
        
        noise_domain = noise_record['Domain']
        if pd.isna(noise_domain):
            print(f"    [DOMAIN-RESCUE] ⚠️  Skipping record with missing domain")
            continue
            
        # Find existing clusters with the same domain
        same_domain_mask = (clustered_df['Domain'] == noise_domain) & (clustered_df['cluster_id'] != -1)
        same_domain_records = clustered_df[same_domain_mask]
        
        print(f"    [DOMAIN-RESCUE] 📊 Found {len(same_domain_records)} existing records with domain '{noise_domain}'")
        
        if len(same_domain_records) > 0:
            # Check domain similarity from features
            noise_features = features_df[
                ((features_df['record_id_1'] == noise_record['record_id']) | 
                 (features_df['record_id_2'] == noise_record['record_id']))
            ]
            
            print(f"    [DOMAIN-RESCUE] 🔗 Found {len(noise_features)} feature pairs for this record")
            
            # Find highest domain similarity with same domain records
            max_domain_sim = 0
            target_cluster = None
            
            for _, same_domain_record in same_domain_records.iterrows():
                # Find pairs involving both records
                pair_features = noise_features[
                    ((noise_features['record_id_1'] == same_domain_record['record_id']) |
                     (noise_features['record_id_2'] == same_domain_record['record_id']))
                ]
                
                if len(pair_features) > 0:
                    domain_sim = pair_features[domain_col].max()
                    if domain_sim > max_domain_sim:
                        max_domain_sim = domain_sim
                        target_cluster = same_domain_record['cluster_id']
            
            # Rescue if domain similarity >= 85%
            if max_domain_sim >= 0.85 and target_cluster is not None:
                print(f"    [DOMAIN-RESCUE] 🚀 RESCUING: '{noise_record['Name']}' | Domain: {noise_domain} | Sim: {max_domain_sim:.3f} → Cluster {target_cluster}")
                
                updated_clustered_df.at[noise_idx, 'cluster_id'] = target_cluster
                
                rescue_stats["records_rescued"] += 1
                if noise_domain not in rescue_stats["domains_fixed"]:
                    rescue_stats["domains_fixed"].append(noise_domain)
                    
                rescue_stats["rescue_details"].append({
                    "record_name": noise_record['Name'],
                    "domain": noise_domain,
                    "domain_similarity": max_domain_sim,
                    "target_cluster": target_cluster
                })
    
    # Update final stats
    final_noise_mask = updated_clustered_df['cluster_id'] == -1
    rescue_stats["noise_records_after"] = len(updated_clustered_df[final_noise_mask])
    
    print(f"    [DOMAIN-RESCUE] ✅ RESCUED {rescue_stats['records_rescued']} records from {len(rescue_stats['domains_fixed'])} domains")
    print(f"    [DOMAIN-RESCUE] 📊 Noise records: {rescue_stats['noise_records_before']} → {rescue_stats['noise_records_after']}")
    
    return updated_clustered_df, rescue_stats


if __name__ == "__main__":
    # Test the rescue function
    clustered_df = pd.read_csv('data/outputs/clusters.csv')
    features_df = pd.read_csv('data/outputs/features.csv')
    
    print("Testing domain noise rescue...")
    updated_df, stats = rescue_domain_noise_records(clustered_df, features_df)
    
    # Save updated results
    updated_df.to_csv('data/outputs/clusters_rescued.csv', index=False)
    print(f"\nSaved rescued clusters to: data/outputs/clusters_rescued.csv")
    print(f"Rescue stats: {stats}")
