import time
import streamlit as st
from src.predict import load_artifacts, predict_sentiment

# Konfigurasi Halaman & UI Adaptif
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Custom CSS 
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #eab308;
        color: black;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 700;
        border: none;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #ca8a04;
        color: white;
    }
    .metric-container {
        display: flex;
        gap: 15px;
        margin-top: 15px;
    }
    .metric-box {
        flex: 1;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 5px;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #eab308;
    }
    </style>
""", unsafe_allow_html=True)

# Caching Artifacts
@st.cache_resource
def init_model():
    model, vocab, device = load_artifacts()
    return model, vocab, device

try:
    model, vocab, device = init_model()
    is_ready = True
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file di folder 'models/' sudah lengkap. Error detail: {e}")
    is_ready = False

st.title("IMDb Movie Review Analyzer")
st.write("Klasifikasi sentimen ulasan film menggunakan Deep Learning (LSTM 2-Layer + Masked Mean Pooling).")

# Input Text Area
user_review = st.text_area(
    "Masukkan Ulasan Film (Bahasa Inggris):",
    height=150,
    placeholder="Contoh: The cinematography was absolutely fantastic! The acting was great and the plot kept me on the edge of my seat."
)

# Tombol Eksekusi
if st.button("Analisis Sentimen", use_container_width=True) and is_ready:
    if user_review.strip() == "":
        st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
    else:
        with st.spinner("Menganalisis teks..."):
            # Benchmark Latensi
            start_time = time.time()
            
            # Panggil fungsi dari src/predict.py
            sentiment, confidence = predict_sentiment(user_review, model, vocab, device, max_len=400)
            
            latency_ms = (time.time() - start_time) * 1000

            # Tampilkan Hasil Prediksi
            st.markdown("---")
            st.subheader("Hasil Analisis")
            
            if sentiment == "Positive":
                st.success(f"### Sentimen: **{sentiment.upper()}** 🌟")
            else:
                st.error(f"### Sentimen: **{sentiment.upper()}** 👎")
                
            st.markdown(f"**Confidence Score:** `{confidence:.2f}%`")
            st.progress(confidence / 100.0)

            # Metrik Benchmark Info
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-box">
                    <div class="metric-title">Inference Latency</div>
                    <div class="metric-value">{latency_ms:.2f} ms</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Model Architecture</div>
                    <div class="metric-value">LSTM (Config B)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)