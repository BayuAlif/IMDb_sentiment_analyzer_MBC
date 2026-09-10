import time
import io
import pandas as pd
import plotly.express as px
import streamlit as st
from src.predict import load_artifacts, predict_sentiment

# Konfigurasi Halaman & Tema
st.set_page_config(
    page_title="IMDb Sentiment Analyzer - v2.0",
    page_icon="🎬",
    layout="wide"
)

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

# Inisialisasi Model & Vocab
@st.cache_resource
def init_model():
    model, vocab, device = load_artifacts()
    return model, vocab, device

try:
    model, vocab, device = init_model()
    is_ready = True
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    is_ready = False

st.title("🎬 IMDb Movie Review Sentiment Analyzer (v2.0)")
st.caption("Aplikasi Analisis Sentimen Ulasan Film berbasis PyTorch LSTM (Config B: 2-Layer Recurrent + Masked Mean Pooling)")

# 3. Sistem Navigasi Tabs
tab1, tab2 = st.tabs([" Single Review", " Batch Processing (CSV)"])

# TAB 1: Single Review
with tab1:
    col_input, col_result = st.columns([1.2, 1])
    
    with col_input:
        user_review = st.text_area(
            "Masukkan Ulasan Film (Bahasa Inggris):",
            height=180,
            placeholder="Contoh: The movie was brilliant, stunning visuals and top tier acting!"
        )
        analyze_btn = st.button("Analisis Sentimen", key="single_btn", use_container_width=True)
        
    with col_result:
        if analyze_btn and is_ready:
            if user_review.strip() == "":
                st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
            else:
                with st.spinner("Menganalisis teks..."):
                    start_time = time.time()
                    sentiment, confidence = predict_sentiment(user_review, model, vocab, device, max_len=400)
                    latency_ms = (time.time() - start_time) * 1000

                    st.subheader("Hasil Analisis")
                    if sentiment == "Positive":
                        st.success(f"### Sentimen: **{sentiment.upper()}** 🌟")
                    else:
                        st.error(f"### Sentimen: **{sentiment.upper()}** 👎")

                    st.markdown(f"**Confidence:** `{confidence:.2f}%`")
                    st.progress(confidence / 100.0)

                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-box">
                            <div class="metric-title">Latency</div>
                            <div class="metric-value">{latency_ms:.2f} ms</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-title">Device</div>
                            <div class="metric-value">{str(device).upper()}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Batch Processing CSV
with tab2:
    st.subheader("Unggah File CSV")
    st.write("Unggah file `.csv` yang berisi kumpulan teks ulasan film.")
    
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    
    if uploaded_file is not None and is_ready:
        df = pd.read_csv(uploaded_file)
        st.write(f"Pratinjau Data ({len(df)} baris):")
        st.dataframe(df.head(5), use_container_width=True)
        
        # Deteksi otomatis atau pilih manual kolom teks
        text_col = st.selectbox("Pilih kolom ulasan film:", options=df.columns)
        
        if st.button("Jalankan Prediksi Batch", use_container_width=True):
            with st.spinner(f"Menganalisis {len(df)} baris ulasan..."):
                start_batch = time.time()
                
                sentiments = []
                confidences = []
                
                prog_bar = st.progress(0)
                total_rows = len(df)
                
                for idx, row in df.iterrows():
                    text_data = str(row[text_col])
                    sent, conf = predict_sentiment(text_data, model, vocab, device, max_len=400)
                    sentiments.append(sent)
                    confidences.append(conf)
                    prog_bar.progress((idx + 1) / total_rows)
                    
                total_batch_time = time.time() - start_batch
                
                df["Predicted_Sentiment"] = sentiments
                df["Confidence_Score(%)"] = [round(c, 2) for c in confidences]
                
                st.success(f"Selesai menganalisis {total_rows} ulasan dalam {total_batch_time:.2f} detik! ({total_batch_time/total_rows*1000:.2f} ms/ulasan)")
                
                # Visualisasi & Tabel Hasil
                col_chart, col_data = st.columns([1, 1.3])
                
                with col_chart:
                    st.subheader("Distribusi Sentimen")
                    counts = df["Predicted_Sentiment"].value_counts().reset_index()
                    counts.columns = ["Sentiment", "Total"]
                    
                    fig = px.pie(
                        counts, 
                        names="Sentiment", 
                        values="Total", 
                        color="Sentiment",
                        color_discrete_map={"Positive": "#22c55e", "Negative": "#ef4444"},
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col_data:
                    st.subheader("Tabel Hasil Prediksi")
                    st.dataframe(df[[text_col, "Predicted_Sentiment", "Confidence_Score(%)"]].head(10), use_container_width=True)
                    
                    # Unduh Hasil
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="Unduh Hasil Lengkap (.csv)",
                        data=csv_buffer.getvalue(),
                        file_name="imdb_sentiment_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )