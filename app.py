import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib

# =====================================
# CONFIG
# =====================================

st.set_page_config(
    page_title="GrowKids - Prediksi Stunting",
    page_icon="🌱",
    layout="wide"
)

# =====================================
# CSS
# =====================================

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #6B3FA0;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .risk-rendah { background: #d4edda; border-left: 5px solid #28a745; }
    .risk-sedang { background: #fff3cd; border-left: 5px solid #ffc107; }
    .risk-tinggi { background: #f8d7da; border-left: 5px solid #dc3545; }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #6B3FA0;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
    }
    .footer-note {
        font-size: 0.8rem;
        color: #aaa;
        text-align: center;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD MODEL
# Fitur: berat_lahir, tinggi_lahir, asi_eksklusif
# =====================================

MODEL_PATH  = os.path.join("model", "knn_stunting.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

model  = None
scaler = None

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ File tidak ditemukan: `{MODEL_PATH}`")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error(f"❌ File tidak ditemukan: `{SCALER_PATH}`")
    st.stop()

try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    st.error(f"❌ Gagal membaca model: {e}")
    st.stop()

# =====================================
# HELPER FUNCTIONS
# =====================================

def kategori_risiko(prob):
    persen = round(prob * 100, 1)
    if persen <= 30:
        return "Risiko Rendah", "risk-rendah", "✅", persen
    elif persen <= 70:
        return "Risiko Sedang", "risk-sedang", "⚠️", persen
    else:
        return "Risiko Tinggi", "risk-tinggi", "🔴", persen


def rekomendasi_nutrisi(age_months, kategori):
    if age_months <= 6:
        base = "**0–6 bulan:** ASI eksklusif adalah satu-satunya nutrisi yang dibutuhkan bayi."
    elif age_months <= 24:
        base = "**6–24 bulan:** ASI tetap diberikan, dampingi dengan MPASI."
    else:
        base = "**> 24 bulan:** Berikan makanan keluarga yang sehat dan bergizi."

    if kategori == "Risiko Rendah":
        detail = """
- ✅ Status gizi baik — pertahankan pola makan saat ini
- Konsumsi protein hewani (telur, ikan, daging) setiap hari
- Pantau berat dan tinggi badan setiap bulan di Posyandu
        """
    elif kategori == "Risiko Sedang":
        detail = """
- ⚠️ Tingkatkan asupan **protein hewani** (ikan, telur, ayam, daging)
- Tambahkan sumber **zat besi**: hati ayam, bayam, kacang-kacangan
- Kunjungi Posyandu minimal 1x sebulan
        """
    else:
        detail = """
- 🔴 Segera konsultasikan ke **dokter anak atau ahli gizi**
- Tingkatkan kalori harian: tambah porsi dan frekuensi makan
- Prioritaskan: **protein hewani, vitamin A, zinc, kalsium**
- Minta rujukan ke **Puskesmas atau RSUD** terdekat
        """
    return base, detail


# =====================================
# SESSION STATE
# =====================================

if "history" not in st.session_state:
    st.session_state.history = []

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:
    st.markdown("### 🌱 GrowKids")
    st.markdown("Aplikasi prediksi risiko stunting berbasis AI")
    st.markdown("---")
    menu = st.radio(
        "Navigasi",
        ["🔍 Prediksi", "📈 Riwayat", "ℹ️ Tentang"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("⚠️ Hasil prediksi bukan diagnosis medis.")

# =====================================
# PAGE: PREDIKSI
# =====================================

if menu == "🔍 Prediksi":

    st.markdown('<p class="main-title">🌱 GrowKids</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">AI-Based Stunting Risk Prediction</p>', unsafe_allow_html=True)

    st.markdown("""
    Masukkan data anak di bawah ini untuk mendapatkan prediksi risiko stunting.
    Prediksi ini **bukan diagnosis medis** — gunakan sebagai panduan awal.
    """)

    st.markdown("---")

    # ── INPUT — hanya 3 fitur sesuai training ──────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🍼 Data Saat Lahir**")
        birth_weight = st.number_input(
            "Berat Lahir (kg)",
            min_value=0.5,
            max_value=6.0,
            value=3.0,
            step=0.1,
            help="Berat badan anak saat dilahirkan"
        )
        birth_height = st.number_input(
            "Tinggi Lahir (cm)",
            min_value=30.0,
            max_value=60.0,
            value=49.0,
            step=0.1,
            help="Panjang badan anak saat dilahirkan"
        )

    with col2:
        st.markdown("**🤱 Data ASI**")
        asi = st.selectbox(
            "ASI Eksklusif",
            ["Ya", "Tidak"],
            help="Apakah anak mendapat ASI eksklusif selama 6 bulan pertama?"
        )
        # Info tambahan (tidak masuk model, hanya untuk rekomendasi nutrisi)
        age = st.number_input(
            "Usia Anak Sekarang (bulan)",
            min_value=0,
            max_value=60,
            value=17,
            help="Digunakan untuk rekomendasi nutrisi yang sesuai usia"
        )

    st.markdown("")
    predict_btn = st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True)

    if predict_btn:

        # Encode ASI: Ya=1, Tidak=0
        asi_enc = 1 if asi == "Ya" else 0

        # 3 fitur sesuai training: berat_lahir, tinggi_lahir, asi_eksklusif
        data = np.array([[birth_weight, birth_height, asi_enc]])

        try:
            data_scaled = scaler.transform(data)
            probability = model.predict_proba(data_scaled)[0][1]
        except Exception as e:
            st.error(f"❌ Error saat prediksi: {e}")
            st.stop()

        kategori, css_class, emoji, persen = kategori_risiko(probability)

        # Simpan riwayat
        st.session_state.history.append({
            "Berat Lahir (kg)": birth_weight,
            "Tinggi Lahir (cm)": birth_height,
            "ASI Eksklusif": asi,
            "Risiko (%)": persen,
            "Kategori": kategori
        })

        # ── HASIL ─────────────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-title">📊 Hasil Prediksi</p>', unsafe_allow_html=True)

        colA, colB = st.columns(2)
        colA.metric("Risiko Stunting", f"{persen}%")
        colB.metric("Kategori", f"{emoji} {kategori}")

        st.progress(float(probability))

        st.markdown(f"""
        <div class="result-box {css_class}">
        {emoji} <strong>{kategori}</strong><br>
        Berdasarkan data berat lahir <strong>{birth_weight} kg</strong>,
        tinggi lahir <strong>{birth_height} cm</strong>, dan
        ASI eksklusif <strong>{asi}</strong> —
        risiko stunting anak berada di angka <strong>{persen}%</strong>.
        </div>
        """, unsafe_allow_html=True)

        # ── NUTRISI ───────────────────────────────────────
        st.markdown('<p class="section-title">🥗 Rekomendasi Nutrisi</p>', unsafe_allow_html=True)
        base, detail = rekomendasi_nutrisi(age, kategori)
        st.info(base)
        st.markdown(detail)

        # ── EDUKASI ───────────────────────────────────────
        st.markdown('<p class="section-title">📚 Edukasi Stunting</p>', unsafe_allow_html=True)

        with st.expander("Apa itu Stunting?"):
            st.markdown("""
            Stunting adalah kondisi gagal tumbuh pada anak akibat kekurangan gizi kronis.
            Anak dikategorikan stunting jika **Z-Score tinggi badan per usia (TB/U) < -2 SD** standar WHO.

            **Dampak jangka panjang:**
            - Gangguan perkembangan otak dan kecerdasan
            - Rentan terhadap penyakit infeksi
            - Produktivitas rendah di masa dewasa
            """)

        with st.expander("Cara mencegah Stunting"):
            st.markdown("""
            - **1000 HPK** — 1000 Hari Pertama Kehidupan adalah fase paling kritis
            - **ASI eksklusif** selama 6 bulan pertama
            - **MPASI** yang tepat mulai usia 6 bulan
            - Pantau pertumbuhan rutin di **Posyandu** setiap bulan
            - Jaga **sanitasi** dan kebersihan lingkungan
            """)

        st.markdown('<p class="footer-note">⚠️ Prediksi ini bukan diagnosis medis. Konsultasikan ke tenaga kesehatan profesional.</p>', unsafe_allow_html=True)


# =====================================
# PAGE: RIWAYAT
# =====================================

elif menu == "📈 Riwayat":

    st.markdown('<p class="main-title">📈 Riwayat Prediksi</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Pantau tren risiko stunting anak dari waktu ke waktu</p>', unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.info("Belum ada riwayat. Lakukan prediksi terlebih dahulu di menu **🔍 Prediksi**.")
    else:
        df = pd.DataFrame(st.session_state.history)
        df.index = [f"Cek #{i+1}" for i in range(len(df))]

        st.dataframe(df, use_container_width=True)

        st.markdown("**Tren Risiko Stunting (%)**")
        st.line_chart(df["Risiko (%)"])

        if st.button("🗑️ Hapus Semua Riwayat"):
            st.session_state.history = []
            st.rerun()


# =====================================
# PAGE: TENTANG
# =====================================

elif menu == "ℹ️ Tentang":

    st.markdown('<p class="main-title">ℹ️ Tentang GrowKids</p>', unsafe_allow_html=True)

    st.markdown("""
    **GrowKids** adalah aplikasi prediksi risiko stunting berbasis kecerdasan buatan (AI)
    yang dikembangkan oleh Tim **Gritty Bytest** dari UIN Syarif Hidayatullah Jakarta
    dalam kompetisi **AI Hackathon Technofest 2025**.

    ---

    ### 🤖 Teknologi yang Digunakan

    | Komponen | Detail |
    |---|---|
    | Algoritma | K-Nearest Neighbors (KNN), K=5 |
    | Metrik Jarak | Euclidean Distance |
    | Fitur Model | berat_lahir, tinggi_lahir, asi_eksklusif |
    | Dataset | Kaggle — harnelia/faktor-stunting (10.000 baris) |
    | Framework | Streamlit |
    | Library ML | scikit-learn, numpy, pandas, joblib |

    ---

    ### 📊 Performa Model

    | Metrik | Nilai |
    |---|---|
    | Accuracy | 59.74% |
    | Recall (Stunting) | **86%** |
    | Precision (Stunting) | 62% |
    | F1-Score (Stunting) | 0.72 |

    ---

    ### 👥 Tim Pengembang

    - **Ulfatul Adawiyah**
    - Nadira Afsarina Biya
    - Siti Magfiroh

    ---

    ### ⚠️ Disclaimer

    Hasil prediksi GrowKids **bukan diagnosis medis**.
    Selalu konsultasikan hasil ini ke tenaga kesehatan profesional.
    """)
