import streamlit as st
import pandas as pd
import pickle

# ------------------------------
# Load Model
# ------------------------------
try:
    with open("cancer_model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ File 'cancer_model.pkl' tidak ditemukan. Pastikan file ini ada di direktori yang sama dengan app.py.")
    st.stop()
except Exception as e:
    st.error("❌ Gagal load model. File corrupt atau tidak kompatibel.")
    st.exception(e)
    st.stop()


# ------------------------------
# UI Streamlit
# ------------------------------
st.set_page_config(page_title="Prediksi Kanker", layout="wide")
st.title("🔬 Aplikasi Prediksi Kanker")
st.write("Upload data CSV dan dapatkan hasil prediksi apakah data menunjukkan kanker atau tidak.")

# ------------------------------
# Upload Data CSV
# ------------------------------
uploaded_file = st.file_uploader("📂 Upload file CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("🧾 Data yang Di-upload")
    st.dataframe(data.head())

    # Cek apakah model bisa memproses data
    try:
        st.subheader("🧠 Hasil Prediksi")
        predictions = model.predict(data)
        data["Prediksi"] = predictions
        st.dataframe(data)

        # Tambahkan ringkasan
        st.success("✅ Prediksi berhasil dilakukan!")
        st.write("**Keterangan:**")
        st.write("- `0` = Tidak Kanker")
        st.write("- `1` = Kanker")

        # Visualisasi ringkas
        st.subheader("📊 Statistik Prediksi")
        st.bar_chart(data["Prediksi"].value_counts())

    except Exception as e:
        st.error("⚠️ Gagal melakukan prediksi. Cek apakah format data cocok dengan model.")
        st.exception(e)
else:
    st.info("⬆️ Silakan upload file CSV untuk mulai.")

