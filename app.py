import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("cancer_model.joblib")

# Set judul halaman
st.set_page_config(page_title="Prediksi Kanker Payudara", layout="wide")
st.title("🔬 Prediksi Kanker Payudara")
st.markdown("Masukkan nilai dari 30 fitur diagnosis kanker payudara berdasarkan hasil medis (seperti MRI/biopsi).")

# Fitur-fitur dari dataset Breast Cancer sklearn
feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

# Bagi jadi 3 kolom biar enak dilihat
cols = st.columns(3)
input_data = []

# Generate form input dari semua fitur
with st.form("form_kanker"):
    for i, name in enumerate(feature_names):
        val = cols[i % 3].number_input(label=name.title(), value=0.0)
        input_data.append(val)

    submit = st.form_submit_button("🔍 Prediksi")

# Prediksi setelah submit
if submit:
    input_array = np.array([input_data])
    prediction = model.predict(input_array)[0]

    st.subheader("📢 Hasil Prediksi")
    if prediction == 1:
        st.error("⚠️ Positif Kanker")
    else:
        st.success("✅ Tidak Terindikasi Kanker")
