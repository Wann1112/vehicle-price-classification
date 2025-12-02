import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Australian Vehicle Prices.csv")

# Convert Price to numeric
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df = df.dropna(subset=["Price"])

# Categorize price
def categorize_price(price):
    if price < 15000:
        return "Murah"
    elif price < 30000:
        return "Sedang"
    elif price < 60000:
        return "Mahal"
    else:
        return "Sangat Mahal"

df["Price_Category"] = df["Price"].apply(categorize_price)

# Encode target
label_encoder = LabelEncoder()
df["Price_Category"] = label_encoder.fit_transform(df["Price_Category"])

# Features
X = df.drop(columns=["Price", "Price_Category"])
y = df["Price_Category"]

# Convert categorical columns to numeric
X = pd.get_dummies(X)

# Save feature names for Streamlit
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

# Train model
model = XGBClassifier()
model.fit(X, y)

# Save model + encoder
joblib.dump(model, "xgb_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model training selesai! File .pkl berhasil dibuat.")
