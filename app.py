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
# Fitur: jenis_kelamin, umur_bulan, berat_lahir, tinggi_lahir,
#        berat_badan, tinggi_badan, asi_eksklusif, z_score
# =====================================

MODEL_PATH  = os.path.join("model", "knn_stunting.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

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
# WHO Z-SCORE REFERENCE TABLE
# =====================================

WHO_BOYS = {
    0:  (49.9, 1.89),  1:  (54.7, 2.09),  2:  (58.4, 2.22),
    3:  (61.4, 2.31),  4:  (63.9, 2.37),  5:  (65.9, 2.42),
    6:  (67.6, 2.46),  7:  (69.2, 2.49),  8:  (70.6, 2.52),
    9:  (72.0, 2.54),  10: (73.3, 2.57),  11: (74.5, 2.59),
    12: (75.7, 2.62),  13: (76.9, 2.64),  14: (78.0, 2.66),
    15: (79.1, 2.68),  16: (80.2, 2.70),  17: (81.2, 2.72),
    18: (82.3, 2.74),  19: (83.2, 2.76),  20: (84.2, 2.78),
    21: (85.1, 2.80),  22: (86.0, 2.82),  23: (86.9, 2.84),
    24: (87.8, 2.86),  25: (88.6, 2.88),  26: (89.5, 2.90),
    27: (90.3, 2.92),  28: (91.1, 2.94),  29: (91.9, 2.96),
    30: (92.7, 2.98),  36: (96.1, 3.10),  42: (99.9, 3.18),
    48: (103.3, 3.25), 54: (106.7, 3.33), 60: (110.0, 3.40),
}

WHO_GIRLS = {
    0:  (49.1, 1.86),  1:  (53.7, 2.01),  2:  (57.1, 2.15),
    3:  (59.8, 2.22),  4:  (62.1, 2.28),  5:  (64.0, 2.32),
    6:  (65.7, 2.37),  7:  (67.3, 2.41),  8:  (68.7, 2.44),
    9:  (70.1, 2.47),  10: (71.5, 2.50),  11: (72.8, 2.52),
    12: (74.0, 2.55),  13: (75.2, 2.57),  14: (76.4, 2.60),
    15: (77.5, 2.62),  16: (78.6, 2.64),  17: (79.7, 2.66),
    18: (80.7, 2.68),  19: (81.7, 2.70),  20: (82.7, 2.72),
    21: (83.7, 2.74),  22: (84.6, 2.76),  23: (85.5, 2.78),
    24: (86.4, 2.80),  25: (87.2, 2.82),  26: (88.0, 2.84),
    27: (88.8, 2.86),  28: (89.6, 2.88),  29: (90.4, 2.90),
    30: (91.2, 2.92),  36: (95.1, 3.08),  42: (98.7, 3.15),
    48: (102.7, 3.22), 54: (105.9, 3.30), 60: (109.4, 3.38),
}

def get_zscore(gender_str, age_months, height_cm):
    ref = WHO_BOYS if gender_str == "Laki-laki" else WHO_GIRLS
    available = sorted(ref.keys())
    closest   = min(available, key=lambda x: abs(x - age_months))
    median, sd = ref[closest]
    if sd == 0:
        return 0.0
    return round((height_cm - median) / sd, 2)

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

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📋 Data Anak Saat Ini**")
        gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        age    = st.number_input("Usia (bulan)", min_value=0, max_value=60, value=17)
        weight = st.number_input("Berat Badan (kg)", min_value=1.0, max_value=30.0, value=10.0, step=0.1)
        height = st.number_input("Tinggi Badan (cm)", min_value=40.0, max_value=130.0, value=72.2, step=0.1)

    with col2:
        st.markdown("**🍼 Data Saat Lahir**")
        birth_weight = st.number_input("Berat Lahir (kg)", min_value=0.5, max_value=6.0, value=3.0, step=0.1)
        birth_height = st.number_input("Tinggi Lahir (cm)", min_value=30.0, max_value=60.0, value=49.0, step=0.1)
        asi          = st.selectbox("ASI Eksklusif", ["Ya", "Tidak"])

    st.markdown("")
    predict_btn = st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True)

    if predict_btn:

        # Encode
        gender_enc = 1 if gender == "Laki-laki" else 0
        asi_enc    = 1 if asi == "Ya" else 0

        # Hitung z_score otomatis
        z_score = get_zscore(gender, age, height)

        # 8 fitur sesuai urutan training:
        # jenis_kelamin, umur_bulan, berat_lahir, tinggi_lahir,
        # berat_badan, tinggi_badan, asi_eksklusif, z_score
        data = np.array([[
            gender_enc,
            age,
            birth_weight,
            birth_height,
            weight,
            height,
            asi_enc,
            z_score
        ]])

        try:
            data_scaled = scaler.transform(data)
            probability = model.predict_proba(data_scaled)[0][1]
        except Exception as e:
            st.error(f"❌ Error saat prediksi: {e}")
            st.stop()

        kategori, css_class, emoji, persen = kategori_risiko(probability)

        # Simpan riwayat
        st.session_state.history.append({
            "Usia (bln)": age,
            "BB (kg)": weight,
            "TB (cm)": height,
            "Z-Score": z_score,
            "Risiko (%)": persen,
            "Kategori": kategori
        })

        # ── HASIL ─────────────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-title">📊 Hasil Prediksi</p>', unsafe_allow_html=True)

        colA, colB, colC = st.columns(3)
        colA.metric("Risiko Stunting", f"{persen}%")
        colB.metric("Kategori", f"{emoji} {kategori}")
        colC.metric("Z-Score TB/U", z_score)

        st.progress(float(probability))

        keterangan_z = (
            "berada di bawah **-2 SD standar WHO** — perlu perhatian lebih."
            if z_score < -2
            else "masih dalam rentang normal standar WHO."
        )

        st.markdown(f"""
        <div class="result-box {css_class}">
        {emoji} <strong>{kategori}</strong><br>
        Z-Score <strong>{z_score}</strong> — tinggi badan anak {keterangan_z}
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
    | Fitur Model | jenis_kelamin, umur_bulan, berat_lahir, tinggi_lahir, berat_badan, tinggi_badan, asi_eksklusif, z_score |
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
