import pandas as pd

df = pd.read_csv("data/processed/clean_data.csv")

print("🔍 Missing values:\n", df.isnull().sum())
print("\n📊 Data types:\n", df.dtypes)
print("\n📌 Sample data:\n", df.head())