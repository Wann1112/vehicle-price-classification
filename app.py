import streamlit as st
import pandas as pd
import joblib

st.title("Vehicle Price Category Classification")
st.write("Klasifikasi harga mobil: **Low – Medium – High – Very High**")

# Load model
try:
    model = joblib.load("xgb_model.pkl")
except:
    st.error("Model belum tersedia. Jalankan train_model.py terlebih dahulu.")
    st.stop()

# Input user
brand = st.text_input("Brand")
model_name = st.text_input("Model")
year = st.number_input("Year", min_value=1900, max_value=2030, value=2015)
km = st.number_input("Kilometres", min_value=0, value=50000)

if st.button("Prediksi"):
    input_df = pd.DataFrame({
        "Brand": [brand],
        "Model": [model_name],
        "Year": [year],
        "Kilometres": [km]
    })

    pred = model.predict(input_df)[0]

    st.success(f"Kategori Harga: **{pred}** 🚀")
