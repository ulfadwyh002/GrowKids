import streamlit as st
import numpy as np
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
# CSS CUSTOM
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
    }
    .risk-rendah  { background: #d4edda; border-left: 5px solid #28a745; }
    .risk-sedang  { background: #fff3cd; border-left: 5px solid #ffc107; }
    .risk-tinggi  { background: #f8d7da; border-left: 5px solid #dc3545; }
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
# WHO Z-SCORE REFERENCE TABLE (sample)
# Sumber: WHO Child Growth Standards
# Ini adalah nilai median & SD untuk
# height-for-age (cm), subset 0-24 bulan
# Lengkapi sesuai tabel WHO resmi
# =====================================

# Format: {umur_bulan: {"male": (median, sd), "female": (median, sd)}}
WHO_REF = {
    0:  {"male": (49.9, 1.89), "female": (49.1, 1.86)},
    1:  {"male": (54.7, 2.09), "female": (53.7, 2.01)},
    2:  {"male": (58.4, 2.22), "female": (57.1, 2.15)},
    3:  {"male": (61.4, 2.31), "female": (59.8, 2.22)},
    4:  {"male": (63.9, 2.37), "female": (62.1, 2.28)},
    5:  {"male": (65.9, 2.42), "female": (64.0, 2.32)},
    6:  {"male": (67.6, 2.46), "female": (65.7, 2.37)},
    7:  {"male": (69.2, 2.49), "female": (67.3, 2.41)},
    8:  {"male": (70.6, 2.52), "female": (68.7, 2.44)},
    9:  {"male": (72.0, 2.54), "female": (70.1, 2.47)},
    10: {"male": (73.3, 2.57), "female": (71.5, 2.50)},
    11: {"male": (74.5, 2.59), "female": (72.8, 2.52)},
    12: {"male": (75.7, 2.62), "female": (74.0, 2.55)},
    13: {"male": (76.9, 2.64), "female": (75.2, 2.57)},
    14: {"male": (78.0, 2.66), "female": (76.4, 2.60)},
    15: {"male": (79.1, 2.68), "female": (77.5, 2.62)},
    16: {"male": (80.2, 2.70), "female": (78.6, 2.64)},
    17: {"male": (81.2, 2.72), "female": (79.7, 2.66)},
    18: {"male": (82.3, 2.74), "female": (80.7, 2.68)},
    19: {"male": (83.2, 2.76), "female": (81.7, 2.70)},
    20: {"male": (84.2, 2.78), "female": (82.7, 2.72)},
    21: {"male": (85.1, 2.80), "female": (83.7, 2.74)},
    22: {"male": (86.0, 2.82), "female": (84.6, 2.76)},
    23: {"male": (86.9, 2.84), "female": (85.5, 2.78)},
    24: {"male": (87.8, 2.86), "female": (86.4, 2.80)},
    36: {"male": (96.1, 3.10), "female": (95.1, 3.08)},
    48: {"male": (103.3, 3.25), "female": (102.7, 3.22)},
    60: {"male": (110.0, 3.40), "female": (109.4, 3.38)},
}

def get_zscore(gender_str, age_months, height_cm):
    """
    Hitung Z-Score TB/U berdasarkan standar WHO.
    Gunakan usia terdekat jika tidak ada di tabel.
    """
    gender_key = "male" if gender_str == "Laki-laki" else "female"

    # Cari usia terdekat di tabel
    available_ages = sorted(WHO_REF.keys())
    closest_age = min(available_ages, key=lambda x: abs(x - age_months))

    ref = WHO_REF[closest_age][gender_key]
    median, sd = ref[0], ref[1]

    if sd == 0:
        return 0.0

    z = (height_cm - median) / sd
    return round(z, 2)


def kategori_risiko(prob):
    persen = round(prob * 100, 1)

    if persen <= 30:
        kategori = "Risiko Rendah"
        css_class = "risk-rendah"
        emoji = "✅"
    elif persen <= 70:
        kategori = "Risiko Sedang"
        css_class = "risk-sedang"
        emoji = "⚠️"
    else:
        kategori = "Risiko Tinggi"
        css_class = "risk-tinggi"
        emoji = "🔴"

    return kategori, css_class, emoji, persen


def rekomendasi_nutrisi(age_months, kategori):
    """
    Rekomendasi berdasarkan kombinasi usia dan tingkat risiko.
    """
    if age_months <= 6:
        base = "**0–6 bulan:** ASI eksklusif adalah satu-satunya nutrisi yang dibutuhkan bayi."
    elif age_months <= 24:
        base = "**6–24 bulan:** ASI tetap diberikan, dampingi dengan MPASI (makanan pendamping ASI)."
    else:
        base = "**> 24 bulan:** Berikan makanan keluarga yang sehat, bergizi, dan bervariasi."

    if kategori == "Risiko Rendah":
        detail = """
- ✅ Status gizi baik — pertahankan pola makan saat ini
- Konsumsi protein hewani (telur, ikan, daging) setiap hari
- Pastikan asupan sayur dan buah cukup
- Pantau berat dan tinggi badan setiap bulan
        """
    elif kategori == "Risiko Sedang":
        detail = """
- ⚠️ Tingkatkan asupan **protein hewani** (ikan, telur, ayam, daging)
- Tambahkan sumber **zat besi**: hati ayam, bayam, kacang-kacangan
- Berikan **vitamin A** dan **zinc** sesuai anjuran dokter
- Kunjungi Posyandu minimal 1x sebulan untuk pemantauan
        """
    else:
        detail = """
- 🔴 Segera konsultasikan ke **dokter anak atau ahli gizi**
- Tingkatkan kalori harian: tambah porsi dan frekuensi makan
- Prioritaskan: **protein hewani, vitamin A, zinc, kalsium**
- Pemeriksaan lanjutan untuk identifikasi penyebab
- Minta rujukan ke **Puskesmas atau RSUD** terdekat
        """

    return base, detail


# =====================================
# HISTORY (session state)
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
    st.markdown("**Navigasi**")
    menu = st.radio(
        "",
        ["🔍 Prediksi", "📈 Riwayat", "ℹ️ Tentang"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("⚠️ Hasil prediksi bukan diagnosis medis. Konsultasikan ke tenaga kesehatan profesional.")

# =====================================
# PAGE: PREDIKSI
# =====================================

if menu == "🔍 Prediksi":

    st.markdown('<p class="main-title">🌱 GrowKids</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">AI-Based Stunting Risk Prediction — Deteksi Dini Risiko Stunting</p>', unsafe_allow_html=True)

    st.markdown("""
    Masukkan data anak di bawah ini. Hasil prediksi akan ditampilkan setelah menekan tombol **Prediksi**.
    Prediksi ini **bukan diagnosis medis** — gunakan sebagai panduan awal sebelum konsultasi ke tenaga kesehatan.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Data Saat Ini**")
        gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        age = st.number_input("Usia (bulan)", min_value=0, max_value=60, value=17)
        weight = st.number_input("Berat Badan (kg)", min_value=1.0, max_value=30.0, value=10.0, step=0.1)
        height = st.number_input("Tinggi Badan (cm)", min_value=40.0, max_value=130.0, value=72.2, step=0.1)

    with col2:
        st.markdown("**Data Saat Lahir**")
        birth_weight = st.number_input("Berat Lahir (kg)", min_value=0.5, max_value=6.0, value=3.0, step=0.1)
        birth_height = st.number_input("Tinggi Lahir (cm)", min_value=30.0, max_value=60.0, value=49.0, step=0.1)
        asi = st.selectbox("ASI Eksklusif", ["Ya", "Tidak"])

    st.markdown("")
    predict_btn = st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True)

    if predict_btn:

        # Encode input
        gender_enc = 1 if gender == "Laki-laki" else 0
        asi_enc    = 1 if asi == "Ya" else 0

        # Hitung z-score
        z_score = get_zscore(gender, age, height)

        # Susun fitur (8 fitur sesuai training)
        # Urutan: gender, age, birth_weight, birth_height, weight, height, asi, z_score
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
            st.error(f"Error saat prediksi: {e}")
            st.stop()

        kategori, css_class, emoji, persen = kategori_risiko(probability)

        # Simpan ke riwayat
        st.session_state.history.append({
            "Usia (bln)": age,
            "BB (kg)": weight,
            "TB (cm)": height,
            "Z-Score": z_score,
            "Risiko (%)": persen,
            "Kategori": kategori
        })

        # ── TAMPILKAN HASIL ───────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-title">📊 Hasil Prediksi</p>', unsafe_allow_html=True)

        colA, colB, colC = st.columns(3)
        colA.metric("Risiko Stunting", f"{persen}%")
        colB.metric("Kategori", kategori)
        colC.metric("Z-Score TB/U", z_score)

        st.progress(float(probability))

        st.markdown(f"""
        <div class="result-box {css_class}">
        {emoji} <strong>{kategori}</strong><br>
        Z-Score <strong>{z_score}</strong> menunjukkan bahwa tinggi badan anak
        {"<strong>berada di bawah -2 SD standar WHO</strong> — perlu perhatian lebih." if z_score < -2 else "masih dalam rentang standar WHO."}
        </div>
        """, unsafe_allow_html=True)

        # ── REKOMENDASI NUTRISI ───────────────────────────────
        st.markdown('<p class="section-title">🥗 Rekomendasi Nutrisi</p>', unsafe_allow_html=True)
        base, detail = rekomendasi_nutrisi(age, kategori)
        st.info(base)
        st.markdown(detail)

        # ── EDUKASI STATIS ───────────────────────────────────
        st.markdown('<p class="section-title">📚 Edukasi Stunting</p>', unsafe_allow_html=True)

        with st.expander("Apa itu Stunting?"):
            st.markdown("""
            Stunting adalah kondisi gagal tumbuh pada anak akibat kekurangan gizi kronis.
            Anak dikatakan stunting jika **Z-Score tinggi badan per usia (TB/U) < -2 SD** dari standar WHO.

            Dampak jangka panjang:
            - Gangguan perkembangan otak
            - Rentan terhadap penyakit
            - Produktivitas rendah di masa dewasa
            """)

        with st.expander("Bagaimana cara mencegah Stunting?"):
            st.markdown("""
            - **1000 HPK**: 1000 Hari Pertama Kehidupan adalah fase paling kritis
            - **ASI eksklusif** selama 6 bulan pertama
            - **MPASI** yang tepat mulai usia 6 bulan
            - **Pantau** pertumbuhan rutin di Posyandu
            - **Sanitasi** dan kebersihan lingkungan yang baik
            """)

        st.markdown('<p class="footer-note">⚠️ Prediksi ini bukan diagnosis medis. Selalu konsultasikan hasil ini ke tenaga kesehatan profesional.</p>', unsafe_allow_html=True)


# =====================================
# PAGE: RIWAYAT
# =====================================

elif menu == "📈 Riwayat":

    st.markdown('<p class="main-title">📈 Riwayat Prediksi</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Pantau tren risiko stunting anak dari waktu ke waktu</p>', unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.info("Belum ada riwayat prediksi. Lakukan prediksi terlebih dahulu.")
    else:
        import pandas as pd
        df = pd.DataFrame(st.session_state.history)
        df.index = [f"Cek #{i+1}" for i in range(len(df))]

        st.dataframe(df, use_container_width=True)

        st.markdown("**Tren Risiko Stunting (%)**")
        st.line_chart(df["Risiko (%)"])

        if st.button("🗑️ Hapus Riwayat"):
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
    dalam rangka kompetisi **AI Hackathon Technofest 2025**.

    ---

    ### 🤖 Teknologi yang Digunakan

    | Komponen | Detail |
    |---|---|
    | Algoritma | K-Nearest Neighbors (KNN), K=5 |
    | Metrik Jarak | Euclidean Distance |
    | Z-Score | Standar WHO (Height-for-Age) |
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

    > Recall 86% diprioritaskan karena dalam konteks skrining medis, **melewatkan kasus berisiko (false negative)
    > lebih berbahaya** daripada false positive.

    ---

    ### 👥 Tim Pengembang

    - **Ulfatul Adawiyah**
    - Nadira Afsarina Biya
    - Siti Magfiroh

    ---

    ### ⚠️ Disclaimer

    Hasil prediksi GrowKids **bukan diagnosis medis**.
    Aplikasi ini hanya sebagai alat bantu deteksi dini.
    Selalu konsultasikan hasil ini ke tenaga kesehatan profesional.
    """)
