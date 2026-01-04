# SPK_menu.py
# Hybrid AHP–TOPSIS menggunakan Streamlit
# Audrey Suitela

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Makanan Sehat (AHP–TOPSIS) - By Audrey Suitela",
    page_icon="SPK_Karyawan.png",
    layout="wide"
)

st.markdown("""
<style>
/* ===== DEFAULT BUTTON ===== */
div.stButton > button {
    background-color: #2563eb;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    padding: 0.6em 1.2em;
    border: none;
}

/* ===== BUTTON YA ===== */
button[key="btn_confirm_yes"],
button[key="btn_rank_yes"] {
    background-color: #16a34a !important;
}
button[key="btn_confirm_yes"]:hover,
button[key="btn_rank_yes"]:hover {
    background-color: #15803d !important;
}

/* ===== BUTTON TIDAK ===== */
button[key="btn_confirm_no"],
button[key="btn_rank_no"] {
    background-color: #dc2626 !important;
}
button[key="btn_confirm_no"]:hover,
button[key="btn_rank_no"]:hover {
    background-color: #b91c1c !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Helper Functions
# -------------------------

def ahp_weights(pairwise_matrix):
    col_sum = pairwise_matrix.sum(axis=0)
    norm_matrix = pairwise_matrix / col_sum
    weights = norm_matrix.mean(axis=1)
    return weights, norm_matrix

def ahp_consistency(pairwise_matrix, weights):
    n = pairwise_matrix.shape[0]
    lambda_max = np.mean((pairwise_matrix @ weights) / weights)
    ci = (lambda_max - n) / (n - 1)
    ri_dict = {1:0.0,2:0.0,3:0.58,4:0.9,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49}
    ri = ri_dict.get(n,1.49)
    cr = ci / ri if ri != 0 else 0
    return lambda_max, ci, cr

def topsis(df, weights, impacts):
    X = df.values.astype(float)
    norm = np.sqrt((X**2).sum(axis=0))
    R = X / norm
    V = R * weights

    ideal_pos = np.where(impacts==1, V.max(axis=0), V.min(axis=0))
    ideal_neg = np.where(impacts==1, V.min(axis=0), V.max(axis=0))

    D_pos = np.sqrt(((V - ideal_pos)**2).sum(axis=1))
    D_neg = np.sqrt(((V - ideal_neg)**2).sum(axis=1))

    C = D_neg / (D_pos + D_neg)
    return R, V, ideal_pos, ideal_neg, D_pos, D_neg, C

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data_menu.csv")
    return df

@st.cache_data
def load_pairwise():
    pw = pd.read_excel("Skala_Perbandingan_Berpasangan.xlsm", index_col=0)
    return pw

# -------------------------
# Session State Init
# -------------------------
for key in ["step","confirm_start","confirm_rank"]:
    if key not in st.session_state:
        st.session_state[key] = 1

# -------------------------
# Step 1 & 2: Dataset Display
# -------------------------
st.title("Sistem Penunjang Keputusan Menu Makanan Sehat")

df = load_data()

st.subheader("1. Dataset Menu Makanan")
st.dataframe(df, width="stretch", height=400)

st.subheader("2. Alternatif & Kriteria")
alt_col = "Menu"
criteria_cols = [c for c in df.columns if c != alt_col]

st.dataframe(pd.DataFrame({
    "Alternatif": [alt_col] + [""]*(len(criteria_cols)-1),
    "Kriteria": criteria_cols
}), width="stretch")

if st.button("Lanjutkan", key="btn_step1"):
    st.session_state.step = 3

# -------------------------
# Step 3: Confirmation
# -------------------------
if st.session_state.step == 3:
    st.warning("Lanjutkan untuk proses penentuan performa menu makanan sehat?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Ya", key="btn_confirm_yes"):
            st.session_state.step = 4
    with col2:
        if st.button("Tidak", key="btn_confirm_no"):
            st.session_state.step = 3

# -------------------------
# Step 4 & 5: AHP
# -------------------------
if st.session_state.step >= 4:
    st.header("Proses AHP – Penentuan Bobot Kriteria")

    pw = load_pairwise()
    st.subheader("Matriks Perbandingan Berpasangan")
    st.dataframe(pw, width="stretch")

    weights, norm_pw = ahp_weights(pw.values)
    st.subheader("Matriks Normalisasi AHP")
    norm_df = pd.DataFrame(
        norm_pw,
        index=pw.index,
        columns=pw.columns
    )
    st.dataframe(norm_df, width="stretch")

    lambda_max, ci, cr = ahp_consistency(pw.values, weights)

    if "ahp_result" not in st.session_state:
        st.session_state.ahp_result = pd.DataFrame({
            "Kriteria": pw.index,
            "Bobot": weights
        })

    st.subheader("Bobot Kriteria (AHP)")
    st.dataframe(st.session_state.ahp_result, width="stretch")
    st.info(f"Consistency Ratio (CR): {cr:.4f}")
    
    A = pw.values
    W = weights

    Ax = A @ W
    
    st.subheader("Perhitungan Ax (A × W)")
    ax_df = pd.DataFrame({
        "Kriteria": pw.index,
        "Ax": Ax
    })
    st.dataframe(ax_df, width="stretch")
    
    st.subheader("Uji Konsistensi AHP")
    consistency_df = pd.DataFrame({
        "Parameter": ["λmax", "CI", "CR"],
        "Nilai": [lambda_max, ci, cr]
    })
    st.dataframe(consistency_df, width="stretch")
    
    if st.session_state.step == 4:
        if st.button("Lanjutkan ke TOPSIS", key="btn_to_topsis"):
            st.session_state.step = 6

# -------------------------
# Step 6: Choose Cost/Benefit
# -------------------------
if st.session_state.step == 6:
    st.header("Tentukan Jenis Kriteria Menu Makanan Sehat")
    impacts = []
    for c in criteria_cols:
        choice = st.radio(f"{c}", ("Benefit","Cost"), horizontal=True)
        impacts.append(1 if choice=="Benefit" else -1)

    if st.button("Lanjutkan", key="btn_lanjutkan_cost_benefit"):
        st.session_state.impacts = np.array(impacts)
        st.session_state.step = 7

# -------------------------
# Step 7 & 8: Confirmation Ranking
# -------------------------
if st.session_state.step == 7:
    st.warning("Lanjutkan ke perankingan?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Ya", key="btn_rank_yes"):
            st.session_state.step = 8
    with col2:
        if st.button("Tidak", key="btn_rank_no"):
            st.session_state.step = 3

# -------------------------
# Step 9 & 10: TOPSIS
# -------------------------
if st.session_state.step == 8:
    st.header("Proses TOPSIS")

    data_topsis = df[criteria_cols]

    R, V, ideal_pos, ideal_neg, Dp, Dn, C = topsis(
        data_topsis,
        st.session_state.ahp_result["Bobot"].values,
        st.session_state.impacts
    )

    st.subheader("Matriks Normalisasi")
    st.dataframe(pd.DataFrame(R, columns=criteria_cols), width="stretch")

    st.subheader("Matriks Normalisasi Terbobot")
    st.dataframe(pd.DataFrame(V, columns=criteria_cols), width="stretch")
    
    st.subheader("Solusi Ideal Positif (A⁺)")
    ideal_pos_df = pd.DataFrame(
        [ideal_pos],
        columns=criteria_cols,
        index=["A⁺"]
    )
    st.dataframe(ideal_pos_df, width="stretch")

    st.subheader("Solusi Ideal Negatif (A⁻)")
    ideal_neg_df = pd.DataFrame(
        [ideal_neg],
        columns=criteria_cols,
        index=["A⁻"]
    )
    st.dataframe(ideal_neg_df, width="stretch")
    
    st.subheader("Jarak ke Solusi Ideal")
    distance_df = pd.DataFrame({
        "Menu": df[alt_col],
        "D_plus (D⁺)": Dp,
        "D_minus (D⁻)": Dn
    })
    st.dataframe(distance_df, width="stretch")

    result = pd.DataFrame({
        "Menu": df[alt_col],
        "Nilai_Preferensi": C
    }).sort_values(by="Nilai_Preferensi", ascending=False)

    st.subheader("Hasil Perankingan")
    st.dataframe(result, width="stretch")

    st.subheader("Top 10 Makanan Sehat")
    top10 = result.head(10)
    st.bar_chart(top10.set_index("Menu"))