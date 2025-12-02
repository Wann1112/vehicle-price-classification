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
# 2. Pastikan kolom Price numeric
# =============================
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df = df.dropna(subset=["Price"])  # hapus baris yang Price tidak bisa dikonversi

# =============================
# 3. Buat kolom kategori harga
# =============================
def categorize_price(price):
    if price < 15000:
        return "Low"
    elif price < 30000:
        return "Medium"
    else:
        return "High"

df["Price_Category"] = df["Price"].apply(categorize_price)

# =============================
# 4. Fitur & target
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
# 8. Simpan model
# =============================
joblib.dump(model, "xgb_model.pkl")
print("Model berhasil disimpan sebagai xgb_model.pkl")
