import pandas as pd
import numpy as np

# Load the current clustering results
df = pd.read_csv('data/outputs/clusters_final.csv')
cluster_4207 = df[df['cluster_id'] == 4207]

print("CLUSTER 4207 DOMAIN ANALYSIS")
print("=" * 50)
print(f"Total records: {len(cluster_4207)}")
print(f"Unique domains: {cluster_4207['Domain'].nunique()}")
print()

# Check domain similarity within cluster 4207
# This will help us understand if all domains are indeed 85%+ similar
print("DOMAIN SIMILARITY ANALYSIS:")
print("-" * 30)

# Get domain values
domains = cluster_4207['Domain'].values
unique_domains = cluster_4207['Domain'].unique()

print(f"Sample of domains in cluster 4207:")
for i, domain in enumerate(unique_domains[:20]):
    count = len(cluster_4207[cluster_4207['Domain'] == domain])
    print(f"  {i+1:2}. {domain:<30} ({count} records)")

print(f"\n... and {len(unique_domains)-20} more domains" if len(unique_domains) > 20 else "")

# Check if there are any obvious patterns
print(f"\nDOMAIN PATTERNS:")
print("-" * 15)

# Count domains by TLD
tlds = {}
for domain in unique_domains:
    if pd.notna(domain) and '.' in str(domain):
        tld = str(domain).split('.')[-1]
        tlds[tld] = tlds.get(tld, 0) + 1

print("Top TLDs:")
for tld, count in sorted(tlds.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  .{tld}: {count} domains")

# Check for similar domain patterns
print(f"\nDOMAIN SIMILARITY PATTERNS:")
print("-" * 25)

# Look for domains that might be similar
domain_prefixes = {}
for domain in unique_domains:
    if pd.notna(domain) and '.' in str(domain):
        parts = str(domain).split('.')
        if len(parts) >= 2:
            prefix = parts[0]
            if len(prefix) >= 3:  # Only consider meaningful prefixes
                domain_prefixes[prefix] = domain_prefixes.get(prefix, 0) + 1

similar_prefixes = {k: v for k, v in domain_prefixes.items() if v > 1}
print(f"Domain prefixes that appear multiple times: {len(similar_prefixes)}")
for prefix, count in sorted(similar_prefixes.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {prefix}*: {count} domains")
