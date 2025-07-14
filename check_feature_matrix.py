import pandas as pd

# Check the structure of agg_features to understand the feature matrix
print("AGG_FEATURES ANALYSIS")
print("=" * 25)

try:
    agg_features = pd.read_csv('data/outputs/agg_features.csv')
    print(f"Shape: {agg_features.shape}")
    print(f"Columns: {list(agg_features.columns)}")
    print()
    
    # Check first few rows
    print("First 5 rows:")
    print(agg_features.head())
    print()
    
    # Check data types and ranges
    print("Column statistics:")
    print(agg_features.describe())
    
except FileNotFoundError:
    print("agg_features.csv not found, checking features.csv instead...")
    features = pd.read_csv('data/outputs/features.csv')
    print(f"Features shape: {features.shape}")
    print(f"Features columns: {list(features.columns)}")
    print()
    print("First 5 rows of features:")
    print(features.head())
