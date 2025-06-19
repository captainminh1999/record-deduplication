import pandas as pd

df = pd.read_csv("data/your_spreadsheet.csv")  # or pd.read_excel(...)
# normalize names
df["name_clean"] = df["Name"].str.lower().str.replace(r"[^\w\s]","", regex=True)
# normalize phones
df["phone_clean"] = df["Phone"].str.replace(r"\D","", regex=True)
# drop exact dupes
df = df.drop_duplicates()
df.to_csv("data/cleaned.csv", index=False)
