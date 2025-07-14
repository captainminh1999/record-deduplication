import pandas as pd
import numpy as np

# Load the aggregated features
agg_features_df = pd.read_csv('data/outputs/agg_features.csv')

print("=== CLUSTERING PARAMETER ANALYSIS ===\n")

# Record groups that should be clustered together
record_groups = [
    ["06a77fa687c76550852d0e170cbb35ad", "e500daec1b8f6850d5d38738dc4bcb85", "4e001eec1b8f6850d5d38738dc4bcbca"],
    ["82cafcc41b7380108321311d0d4bcbc7", "c81052201bcf6850d5d38738dc4bcb46", "3c4d22adc3a392d01a9c742a050131c7"],
    ["0f3b56b01b3665101c3ddc6fcc4bcbe4", "9f2ad3e9c39a9a901a9c742a0501313d", "55a044501b6f5d90b37020e6b04bcbe5"]
]

# Get feature columns (exclude record_id and cluster)
feature_cols = [col for col in agg_features_df.columns if col not in ['record_id', 'cluster']]
print(f"Feature columns used for clustering: {feature_cols[:5]}... (total: {len(feature_cols)})")
print()

for i, group in enumerate(record_groups, 1):
    print(f"Group {i}: Feature distance analysis")
    print("-" * 50)
    
    group_features = []
    group_records = []
    
    for record_id in group:
        record_data = agg_features_df[agg_features_df['record_id'] == record_id]
        if len(record_data) > 0:
            features = record_data[feature_cols].values[0]
            group_features.append(features)
            group_records.append(record_id)
            print(f"  {record_id[:12]}... cluster: {record_data['cluster'].iloc[0]}")
    
    if len(group_features) >= 2:
        # Calculate pairwise distances
        group_features = np.array(group_features)
        print(f"\n  Feature matrix shape: {group_features.shape}")
        print(f"  Sample features: {group_features[0][:5]}...")
        
        # Calculate Euclidean distances between all pairs
        distances = []
        for j in range(len(group_features)):
            for k in range(j+1, len(group_features)):
                dist = np.linalg.norm(group_features[j] - group_features[k])
                distances.append(dist)
                print(f"  Distance {group_records[j][:8]}...↔{group_records[k][:8]}...: {dist:.4f}")
        
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)
        min_distance = np.min(distances)
        
        print(f"\n  📊 Distance Statistics:")
        print(f"    Min distance: {min_distance:.4f}")
        print(f"    Max distance: {max_distance:.4f}")
        print(f"    Avg distance: {avg_distance:.4f}")
        
        print(f"\n  💡 Clustering Insight:")
        print(f"    Current eps was likely > {max_distance:.4f}")
        print(f"    For these to cluster together, eps should be ≥ {max_distance:.4f}")
        
    print("\n" + "="*70 + "\n")

# Analyze overall feature distribution to understand eps scaling
print("=== EPS PARAMETER RECOMMENDATIONS ===")
print()

# Sample a subset of features to calculate typical distances
sample_features = agg_features_df[feature_cols].sample(min(1000, len(agg_features_df)), random_state=42).values

# Calculate some representative distances
sample_distances = []
for i in range(0, min(100, len(sample_features)), 10):
    for j in range(i+1, min(i+10, len(sample_features))):
        dist = np.linalg.norm(sample_features[i] - sample_features[j])
        sample_distances.append(dist)

sample_distances = np.array(sample_distances)
print(f"Sample distance statistics from {len(sample_distances)} random pairs:")
print(f"  10th percentile: {np.percentile(sample_distances, 10):.4f}")
print(f"  25th percentile: {np.percentile(sample_distances, 25):.4f}")
print(f"  50th percentile: {np.percentile(sample_distances, 50):.4f}")
print(f"  75th percentile: {np.percentile(sample_distances, 75):.4f}")
print(f"  90th percentile: {np.percentile(sample_distances, 90):.4f}")

print(f"\n💡 Recommendation:")
print(f"   For similar records to cluster together, try eps around {np.percentile(sample_distances, 25):.3f}")
print(f"   Current hierarchical clustering may be too aggressive with subdivision")
