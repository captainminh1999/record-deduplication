import pandas as pd
import numpy as np

# Load sample data to understand the feature structure
df = pd.read_csv('data/outputs/clusters_final.csv')
cluster_4207 = df[df['cluster_id'] == 4207]

print("CLUSTER 4207 SAMPLE DATA ANALYSIS")
print("=" * 40)

# Show sample records to understand the data structure
print("Sample records from cluster 4207:")
sample = cluster_4207[['record_id', 'Name', 'Domain', 'company_clean', 'domain_clean']].head(10)
print(sample.to_string())
print()

# Check for domain_clean vs Domain differences
print("Domain vs domain_clean comparison:")
print("-" * 35)
for i in range(min(10, len(cluster_4207))):
    row = cluster_4207.iloc[i]
    print(f"Domain: {row['Domain']:<30} | domain_clean: {row['domain_clean']}")

print()

# Check if domain_clean has different values than Domain
domain_clean_unique = cluster_4207['domain_clean'].nunique()
domain_unique = cluster_4207['Domain'].nunique()
print(f"Unique Domain values: {domain_unique}")
print(f"Unique domain_clean values: {domain_clean_unique}")

if domain_clean_unique != domain_unique:
    print("\n⚠️ DOMAIN VALUES DIFFER BETWEEN Domain and domain_clean!")
    
    # Show some examples of differences
    print("\nExamples of Domain vs domain_clean differences:")
    for i in range(min(20, len(cluster_4207))):
        row = cluster_4207.iloc[i]
        if str(row['Domain']) != str(row['domain_clean']):
            print(f"  {row['Domain']} → {row['domain_clean']}")
