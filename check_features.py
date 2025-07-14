import pandas as pd

df = pd.read_csv('data/outputs/agg_features.csv')
print('Non-zero features:')
for col in ['company_sim', 'domain_sim', 'phone_exact', 'address_sim']:
    non_zero = (df[col] > 0).sum()
    max_val = df[col].max()
    print(f'{col}: {non_zero} records have values > 0, max: {max_val:.3f}')

print('\nSample of non-zero records:')
non_zero_mask = (df['company_sim'] > 0) | (df['domain_sim'] > 0) | (df['phone_exact'] > 0) | (df['address_sim'] > 0)
non_zero_records = df[non_zero_mask]
print(f'Total records with non-zero features: {len(non_zero_records)}')
if len(non_zero_records) > 0:
    print(non_zero_records[['record_id', 'company_sim', 'domain_sim', 'phone_exact', 'address_sim']].head())
