#!/usr/bin/env python3
"""Simple domain clustering test"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the aggregated features to test our logic
agg_features = pd.read_csv('data/outputs/agg_features.csv')

# Get just the similarity features for clustering
feature_cols = ['company_sim', 'domain_sim', 'company_domain_product', 'company_domain_sum', 'domain_priority', 'domain_dominance']
X = agg_features[feature_cols].values

print(f"Total records: {len(X)}")
print(f"Feature matrix shape: {X.shape}")
print()

# Check the domain_sim column before standardization
domain_col_idx = 1  # domain_sim is the 2nd column
domain_values = X[:, domain_col_idx]
print(f"Domain similarity column (before standardization):")
print(f"  Min: {np.min(domain_values):.1f}")
print(f"  Max: {np.max(domain_values):.1f}")
print(f"  Unique values: {sorted(np.unique(domain_values))}")
print()

# Find records with perfect domain matches (5000.0)
perfect_domain_mask = domain_values == 5000.0
print(f"Records with perfect domain matches: {np.sum(perfect_domain_mask)}")
print()

# Now standardize the features (like the clustering does)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"After standardization:")
domain_values_scaled = X_scaled[:, domain_col_idx]
print(f"  Domain column min: {np.min(domain_values_scaled):.2f}")
print(f"  Domain column max: {np.max(domain_values_scaled):.2f}")

# Check if perfect domain matches still have the same scaled value
perfect_scaled_values = domain_values_scaled[perfect_domain_mask]
if len(perfect_scaled_values) > 0:
    print(f"  Perfect domain matches after scaling: {np.unique(perfect_scaled_values)}")
    print(f"  All perfect matches have same scaled value: {len(np.unique(perfect_scaled_values)) == 1}")
else:
    print("  No perfect domain matches found")
print()

# Test our domain detection logic on a small subset
test_size = 100
subset_indices = np.random.choice(len(X), size=test_size, replace=False)
X_test = X_scaled[subset_indices]

print(f"Testing domain detection on subset of {test_size} records:")
print(f"Subset feature matrix shape: {X_test.shape}")

# Replicate the domain detection logic
domain_col_detected = None
for i in range(X_test.shape[1]):
    col_values = X_test[:, i]
    max_val = np.max(col_values)
    min_val = np.min(col_values)
    print(f"  Column {i}: min={min_val:.2f}, max={max_val:.2f}")
    
    if max_val > min_val:  # Look for any variation
        unique_vals, counts = np.unique(col_values, return_counts=True)
        max_val_idx = np.argmax(unique_vals)
        max_val_count = counts[max_val_idx]
        highest_val = unique_vals[max_val_idx]
        
        print(f"    Highest value {highest_val:.2f} appears {max_val_count} times")
        
        if max_val_count >= 2 and highest_val > 0:
            domain_col_detected = i
            print(f"    -> DETECTED potential domain column at index {i}!")
            break

if domain_col_detected is not None:
    print(f"\nDomain column detected: {domain_col_detected}")
else:
    print(f"\nNo domain column detected in subset")

# Check if the subset contains any perfect domain matches
perfect_in_subset = perfect_domain_mask[subset_indices]
print(f"Perfect domain matches in subset: {np.sum(perfect_in_subset)}")
