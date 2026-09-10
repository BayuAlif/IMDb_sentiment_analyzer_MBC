import time
import io
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st
from src.predict import load_artifacts, predict_sentiment

st.set_page_config(
    page_title="IMDb Sentiment Analyzer - v3.0",
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
        font-size: 1.3rem;
        font-weight: 700;
        color: #eab308;
    }
    </style>
""", unsafe_allow_html=True)

# Inisialisasi Session History Log
if "history_log" not in st.session_state:
    st.session_state.history_log = []

# Sidebar: Pemilihan Model & Konfigurasi
st.sidebar.header("⚙️ Konfigurasi Model")
selected_model_type = st.sidebar.selectbox(
    "Pilih Arsitektur Recurrent:",
    options=["LSTM", "GRU"],
    help="Bandingkan performa konvergensi LSTM vs efisiensi inferensi GRU."
)

@st.cache_resource
def get_cached_model(m_type):
    return load_artifacts(model_type=m_type)

try:
    model, vocab, device = get_cached_model(selected_model_type)
    is_ready = True
except Exception as e:
    st.sidebar.error(f"Gagal memuat model {selected_model_type}: {e}")
    is_ready = False

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Detail Arsitektur Aktif:**
- **Model Type:** {selected_model_type} (Config B)
- **Sequence Length:** 400
- **Pooling:** Masked Mean Pooling
- **Device:** `{str(device).upper()}`
""")

if st.sidebar.button("Hapus Riwayat Sesi"):
    st.session_state.history_log = []
    st.rerun()

# Header Utama
st.title("IMDb Sentiment Analyzer v3.0")
st.caption(f"Multi-Model Text Classification & Live Benchmarking Dashboard | Model Aktif: **{selected_model_type}**")

tab1, tab2, tab3 = st.tabs(["Single Review", "Batch Processing (CSV)", "Session History Log"])

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
                        st.success(f"### Sentimen: **{sentiment.upper()}** ")
                    else:
                        st.error(f"### Sentimen: **{sentiment.upper()}** ")

                    st.markdown(f"**Confidence Score:** `{confidence:.2f}%`")
                    st.progress(confidence / 100.0)

                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-box">
                            <div class="metric-title">Inference Latency</div>
                            <div class="metric-value">{latency_ms:.2f} ms</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-title">Active Model</div>
                            <div class="metric-value">{selected_model_type}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Simpan ke Riwayat Log Sesi
                    st.session_state.history_log.append({
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                        "Model": selected_model_type,
                        "Input Text": user_review[:80] + ("..." if len(user_review) > 80 else ""),
                        "Sentiment": sentiment,
                        "Confidence (%)": round(confidence, 2),
                        "Latency (ms)": round(latency_ms, 2)
                    })

# TAB 2: Batch CSV
with tab2:
    st.subheader("Unggah File CSV")
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"], key="batch_uploader")
    
    if uploaded_file is not None and is_ready:
        df = pd.read_csv(uploaded_file)
        st.write(f"Pratinjau Data ({len(df)} baris):")
        st.dataframe(df.head(5), use_container_width=True)
        
        text_col = st.selectbox("Pilih kolom ulasan film:", options=df.columns)
        
        if st.button("Jalankan Prediksi Batch", use_container_width=True):
            with st.spinner(f"Menganalisis {len(df)} baris teks dengan {selected_model_type}..."):
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
                
                df["Model_Used"] = selected_model_type
                df["Predicted_Sentiment"] = sentiments
                df["Confidence_Score(%)"] = [round(c, 2) for c in confidences]
                
                st.success(f"Selesai! {total_rows} ulasan diproses dalam {total_batch_time:.2f} detik ({total_batch_time/total_rows*1000:.2f} ms/ulasan).")
                
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
                    st.dataframe(df[[text_col, "Model_Used", "Predicted_Sentiment", "Confidence_Score(%)"]].head(10), use_container_width=True)
                    
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Unduh Hasil Batch (.csv)",
                        data=csv_buffer.getvalue(),
                        file_name=f"imdb_sentiment_{selected_model_type.lower()}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# TAB 3: Session History Log
with tab3:
    st.subheader("Riwayat Prediksi Sesi Ini")
    if len(st.session_state.history_log) == 0:
        st.info("Belum ada riwayat inferensi pada sesi ini. Lakukan prediksi di Tab 1 untuk mencatat log.")
    else:
        history_df = pd.DataFrame(st.session_state.history_log)
        st.dataframe(history_df, use_container_width=True)
        
        # Download Riwayat Log
        hist_buffer = io.StringIO()
        history_df.to_csv(hist_buffer, index=False)
        st.download_button(
            label="Unduh Riwayat Sesi (.csv)",
            data=hist_buffer.getvalue(),
            file_name="session_inference_log.csv",
            mime="text/csv"
        )