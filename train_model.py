import pandas as pd
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
# 2. Konversi Price ke numeric
# =============================
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Drop data invalid 
df = df.dropna(subset=["Price"])

df["Price"] = df["Price"].astype(int)

print("DEBUG: 10 harga pertama setelah convert → int")
print(df["Price"].head(10))

# =============================
# 3. Buat kategori harga
# =============================
def categorize_price(price):
    price = int(price)
    if price < 15000:
        return "Low"
    elif price < 30000:
        return "Medium"
    elif price < 60000:
        return "High"
    else:
        return "Very High"

df["Price_Category"] = df["Price"].apply(categorize_price)

# =============================
# 4. Pilih fitur
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
            num_class=4,
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
