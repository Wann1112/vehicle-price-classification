import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

# =============================
# 1. Load dataset
# =============================
df = pd.read_csv("Australian Vehicle Prices.csv")

# =============================
# 2. Membersihkan kolom Price
# =============================
# Hilangkan simbol $, koma, spasi
df["Price"] = (
    df["Price"]
    .astype(str)          
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

# Konversi ke numeric
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Drop harga yang tidak valid
df = df.dropna(subset=["Price"])

print("DEBUG: 10 harga pertama setelah dibersihkan:")
print(df["Price"].head(10))

# =============================
# 3. Buat kategori harga
# =============================
def categorize_price(price):
    price = float(price)
    if price < 15000:
        return "Low"
    elif price < 30000:
        return "Medium"
    else:
        return "High"

df["Price_Category"] = df["Price"].apply(categorize_price)

# =============================
# 4. Fitur & Target
# =============================
X = df[["Make", "Model", "Year", "Kilometres"]]
y = df["Price_Category"]

# =============================
# 5. Preprocessing
# =============================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Make", "Model"]),
        ("num", "passthrough", ["Year", "Kilometres"])
    ]
)

# =============================
# 6. Model
# =============================
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss"
        ))
    ]
)

# =============================
# 7. Train Model
# =============================
print("Training model...")
model.fit(X, y)

# =============================
# 8. Save Model
# =============================
joblib.dump(model, "xgb_model.pkl")
print("BERHASIL: Model disimpan sebagai xgb_model.pkl")
