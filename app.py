import streamlit as st
import numpy as np
import joblib

# Langsung load model dari file lokal
model = joblib.load('cancer_model.pkl')

st.set_page_config(page_title="Prediksi Kanker", layout="centered")
st.title("🔬 Prediksi Kanker Payudara")

# Form input
with st.form("form_kanker"):
    st.subheader("📋 Masukkan Data Pasien")

    col1, col2 = st.columns(2)
    with col1:
        radius_mean = st.number_input("Radius Mean", 0.0, 30.0)
        texture_mean = st.number_input("Texture Mean", 0.0, 40.0)
        perimeter_mean = st.number_input("Perimeter Mean", 0.0, 200.0)
        area_mean = st.number_input("Area Mean", 0.0, 3000.0)
        smoothness_mean = st.number_input("Smoothness Mean", 0.0, 1.0)

    with col2:
        compactness_mean = st.number_input("Compactness Mean", 0.0, 1.0)
        concavity_mean = st.number_input("Concavity Mean", 0.0, 1.0)
        concave_points_mean = st.number_input("Concave Points Mean", 0.0, 1.0)
        symmetry_mean = st.number_input("Symmetry Mean", 0.0, 1.0)
        fractal_dimension_mean = st.number_input("Fractal Dimension Mean", 0.0, 1.0)

    submit = st.form_submit_button("🔍 Prediksi")

# Prediksi
if submit:
    input_data = np.array([[
        radius_mean, texture_mean, perimeter_mean, area_mean,
        smoothness_mean, compactness_mean, concavity_mean,
        concave_points_mean, symmetry_mean, fractal_dimension_mean
    ]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Hasil: Positif Kanker")
    else:
        st.success("✅ Hasil: Tidak Kanker")
