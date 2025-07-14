import pandas as pd

df = pd.read_csv('data/outputs/clusters_final.csv')
print('Column names:')
print(df.columns.tolist())
print(f'Total rows: {len(df)}')
print('\nFirst few rows:')
print(df.head())
