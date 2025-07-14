#!/usr/bin/env python3
"""
Hierarchical subdivision for a specific cluster to break it down by domain
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from core.clustering.hierarchical.subdivision_engine_v3 import SubdivisionEngineV3
from io.file_handler import FileHandler

def subdivide_cluster_2483():
    print("🔧 HIERARCHICAL SUBDIVISION OF CLUSTER 2483")
    print("============================================")
    
    # Load current features and clusters
    file_handler = FileHandler()
    
    print("📂 Loading current clustering results...")
    try:
        agg_features = pd.read_csv("data/outputs/agg_features.csv")
        clusters_df = pd.read_csv("data/outputs/clusters_final.csv")
        
        print(f"Loaded {len(agg_features)} aggregated features")
        print(f"Loaded {len(clusters_df)} clustered records")
        
        # Filter for cluster 2483
        cluster_2483_features = agg_features[agg_features['cluster_id'] == 2483]
        cluster_2483_records = clusters_df[clusters_df['cluster_id'] == 2483]
        
        print(f"Cluster 2483: {len(cluster_2483_features)} feature records, {len(cluster_2483_records)} data records")
        print(f"Unique domains: {cluster_2483_records['domain_clean'].nunique()}")
        
        if len(cluster_2483_features) == 0:
            print("❌ No features found for cluster 2483")
            return
            
        # Convert to numpy array for subdivision
        feature_columns = ['company_sim', 'domain_sim', 'company_domain_product', 
                          'company_domain_sum', 'domain_priority', 'domain_dominance']
        cluster_X = cluster_2483_features[feature_columns].values
        
        print(f"Feature matrix shape: {cluster_X.shape}")
        print(f"Domain_sim values: {np.unique(cluster_2483_features['domain_sim'].values)}")
        
        # Initialize subdivision engine
        subdivision_engine = SubdivisionEngineV3()
        
        # Create original cluster mask (all True for this cluster)
        cluster_mask = np.ones(len(cluster_X), dtype=bool)
        
        # Create dummy current labels (all have same cluster ID)
        current_labels = np.full(len(cluster_X), 2483)
        
        print("🚀 Starting hierarchical subdivision...")
        
        # Apply subdivision with aggressive parameters
        success, new_labels, subdivision_info = subdivision_engine.subdivide_cluster(
            cluster_X=cluster_X,
            cluster_mask=cluster_mask,
            current_labels=current_labels,
            cluster_id=2483,
            cluster_size=len(cluster_X),
            depth=0,
            base_eps=0.3,
            max_subdivisions=50  # Allow many subdivisions
        )
        
        print(f"📊 Subdivision result: {success}")
        print(f"📊 Subdivision info: {subdivision_info}")
        
        if success and new_labels is not None:
            unique_new_labels = np.unique(new_labels)
            print(f"✅ Successfully subdivided into {len(unique_new_labels)} clusters")
            
            # Update the cluster assignments
            cluster_2483_features_copy = cluster_2483_features.copy()
            cluster_2483_features_copy['cluster_id'] = new_labels
            
            # Update the main features dataframe
            agg_features_updated = agg_features.copy()
            agg_features_updated.loc[agg_features_updated['cluster_id'] == 2483, 'cluster_id'] = new_labels
            
            # Save updated features
            agg_features_updated.to_csv("data/outputs/agg_features_subdivided.csv", index=False)
            print("💾 Saved updated features to: data/outputs/agg_features_subdivided.csv")
            
            # Update clusters dataframe
            # Create mapping from original indices to new cluster IDs
            cluster_2483_indices = clusters_df[clusters_df['cluster_id'] == 2483].index
            
            if len(cluster_2483_indices) == len(new_labels):
                clusters_updated = clusters_df.copy()
                clusters_updated.loc[cluster_2483_indices, 'cluster_id'] = new_labels
                
                clusters_updated.to_csv("data/outputs/clusters_subdivided.csv", index=False)
                print("💾 Saved updated clusters to: data/outputs/clusters_subdivided.csv")
                
                # Analyze the results
                print("\n📊 SUBDIVISION ANALYSIS:")
                for label in unique_new_labels:
                    label_mask = new_labels == label
                    label_records = clusters_updated[clusters_updated['cluster_id'] == label]
                    if len(label_records) > 0:
                        unique_domains = label_records['domain_clean'].nunique()
                        print(f"  Cluster {label}: {len(label_records)} records, {unique_domains} domains")
                        if unique_domains <= 5:
                            print(f"    Domains: {list(label_records['domain_clean'].unique())}")
            else:
                print(f"❌ Mismatch in record counts: {len(cluster_2483_indices)} vs {len(new_labels)}")
        else:
            print("❌ Subdivision failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    subdivide_cluster_2483()
