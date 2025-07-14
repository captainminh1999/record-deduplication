import pandas as pd

df = pd.read_csv('data/outputs/features.csv')
print('Features shape:', df.shape)
print('Sample data:')
print(df.head())
print('\nNon-zero values:')
for col in ['company_sim', 'domain_sim', 'phone_exact', 'address_sim']:
    non_zero = (df[col] > 0).sum()
    print(f'{col}: {non_zero} pairs have values > 0')

# Check data types
print(f'\nData types:')
print(df.dtypes)
