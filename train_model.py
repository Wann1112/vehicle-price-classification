import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("Australian Vehicle Prices.csv")

# Mengecek apakah kolom Price ada
if "Price" not in df.columns:
    raise ValueError("Kolom 'Price' tidak ditemukan dalam dataset!")

# =========================
# 2. Membuat Label Harga
# =========================
def categorize_price(price):
    if price < 15000:
        return "Murah"
    elif 15000 <= price < 30000:
        return "Sedang"
    elif 30000 <= price < 60000:
        return "Mahal"
    else:
        return "Sangat Mahal"

df["Price_Category"] = df["Price"].apply(categorize_price)

# =========================
# 3. Memisahkan Fitur & Label
# =========================
X = df.drop(["Price", "Price_Category"], axis=1)
y = df["Price_Category"]

# Mnegidentifikasi kolom numerik & kategorikal
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# =========================
# 4. Preprocessing
# =========================
scaler = StandardScaler()
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# Fit & Transform
X_num = scaler.fit_transform(X[num_cols])
X_cat = encoder.fit_transform(X[cat_cols])

# Gabungkan fitur numerik + kategorikal
X_processed = np.hstack([X_num, X_cat])

# =========================
# 5. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 6. Train Model XGBoost
# =========================
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
)

model.fit(X_train, y_train)

# =========================
# 7. Simpan Model & Preprocessor
# =========================
joblib.dump(model, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")

print("Model berhasil dilatih dan disimpan!")
