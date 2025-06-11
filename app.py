import streamlit as st
import numpy as np
import pickle

# ------------------------------
# Load Model
# ------------------------------
st.set_page_config(page_title="Prediksi Kanker", layout="wide")
st.title("🔬 Aplikasi Prediksi Kanker (Manual Input)")

# Upload model file
model_file = st.file_uploader("📦 Upload Model (.pkl)", type=["pkl"])
if model_file is not None:
    try:
        model = pickle.load(model_file)
        st.success("✅ Model berhasil dimuat!")
    except Exception as e:
        st.error("❌ Gagal load model.")
        st.exception(e)
        st.stop()
else:
    st.info("⬆️ Silakan upload model `.pkl` terlebih dahulu.")
    st.stop()

# ------------------------------
# Form Input Fitur (Contoh 10 fitur)
# ------------------------------
st.subheader("📋 Input Data Pasien")

with st.form("form_kanker"):
    col1, col2 = st.columns(2)

    with col1:
        radius_mean = st.number_input("Radius Mean", min_value=0.0, max_value=30.0)
        texture_mean = st.number_input("Texture Mean", min_value=0.0, max_value=40.0)
        perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, max_value=200.0)
        area_mean = st.number_input("Area Mean", min_value=0.0, max_value=3000.0)
        smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, max_value=1.0)

    with col2:
        compactness_mean = st.number_input("Compactness Mean", min_value=0.0, max_value=1.0)
        concavity_mean = st.number_input("Concavity Mean", min_value=0.0, max_value=1.0)
        concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, max_value=1.0)
        symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, max_value=1.0)
        fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, max_value=1.0)

    submit = st.form_submit_button("🔍 Prediksi")

# ------------------------------
# Prediksi
# ------------------------------
if submit:
    features = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean,
                          smoothness_mean, compactness_mean, concavity_mean,
                          concave_points_mean, symmetry_mean, fractal_dimension_mean]])

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error("⚠️ Hasil: **Positif Kanker**")
    else:
        st.success("✅ Hasil: **Negatif Kanker**")
