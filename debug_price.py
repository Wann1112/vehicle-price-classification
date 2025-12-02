import pandas as pd

df = pd.read_csv("Australian Vehicle Prices.csv")

print("=== 20 Data Price Mentah ===")
print(df["Price"].head(20))

print("\n=== Tipe Data ===")
print(df["Price"].apply(type).value_counts())

print("\n=== Data Price Unik (contoh 30 nilai unik) ===")
print(df["Price"].unique()[:30])
