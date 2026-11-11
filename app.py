"""
USO Price Prediction Dashboard — Pink Aesthetic Theme (Refined & Debugged)
Praktikum Machine Learning Modul 11 — Regresi (Kelompok K6)

Main Entrypoint Application
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import (
    get_feature_names,
    load_comparison_data,
    load_metrics,
    load_model,
    load_predictions,
    load_preprocessing_metadata,
    load_sample_dataset,
)
from utils.styles import COLORS, CUSTOM_CSS
from utils.validator import preprocess_uploaded_data, validate_uploaded_csv
from utils.visualizations import (
    create_actual_vs_predicted_chart,
    create_comparison_charts,
    create_error_barchart,
    create_new_predictions_timeline,
    create_prediction_distribution_chart,
)


# === STREAMLIT CONFIGURATION ===
st.set_page_config(
    page_title="USO Price Prediction — K6 ML Prak",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injeksi CSS Pink Aesthetic Refined
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# === SESSION STATE INITIALIZATION ===
if "active_df" not in st.session_state:
    st.session_state["active_df"] = None
if "active_source_name" not in st.session_state:
    st.session_state["active_source_name"] = ""
if "last_source_id" not in st.session_state:
    st.session_state["last_source_id"] = None
if "prediction_results" not in st.session_state:
    st.session_state["prediction_results"] = None


# === SIDEBAR NAVIGATION ===
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 10px 0 16px 0;">
            <div style="display: inline-block; background-color: #FCE7F0; width: 56px; height: 56px; border-radius: 50%; line-height: 56px; font-size: 26px; border: 1.5px solid #E879A8; box-shadow: 0 4px 12px rgba(201, 79, 131, 0.2);">
                🌸
            </div>
            <h2 style="margin: 10px 0 2px 0; color: #352B32; font-weight: 800; font-size: 1.25rem; letter-spacing: -0.02em;">USO Predictor</h2>
            <span style="background-color: #FCE7F0; color: #C94F83; padding: 3px 10px; border-radius: 12px; font-size: 0.76rem; font-weight: 700; border: 1px solid #EBD5DF;">Modul 11 • Kelompok K6</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p style="font-size: 0.82rem; font-weight: 700; color: #6F5965; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;">PILIH MENU</p>', unsafe_allow_html=True)

    page = st.radio(
        label="Pilih Halaman",
        options=[
            "📊 Dashboard",
            "🔮 Price Prediction",
            "⚖️ Model Comparison",
            "📖 About Model",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="pink-divider"></div>', unsafe_allow_html=True)

    # Info Status Model Utama di Sidebar (High Contrast)
    st.markdown(
        f"""
        <div style="background-color: #FFFFFF; border: 1.5px solid #EBD5DF; border-radius: 14px; padding: 16px; box-shadow: 0 4px 14px rgba(201, 79, 131, 0.05);">
            <div style="font-weight: 800; color: #C94F83; font-size: 0.9rem; margin-bottom: 6px;">
                ⭐ Model Utama Aktif
            </div>
            <div style="color: #352B32; font-weight: 700; font-size: 0.88rem; margin-bottom: 6px;">
                Gradient Boosting Regressor
            </div>
            <div style="color: #6F5965; font-size: 0.8rem; line-height: 1.5;">
                • Fitur Input: <b>74 Prediktor</b><br>
                • Loss: <code>absolute_error</code><br>
                • Status: <span style="color: #3D8563; font-weight: 700;">● Terlatih (.joblib)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="text-align: center; margin-top: 25px; font-size: 0.76rem; color: #6F5965; font-weight: 500;">
            <b>Praktikum Machine Learning 2026</b><br>
            D4 Teknik Informatika — UNAIR
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if page == "📊 Dashboard":
    # A. Welcome Section
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">USO Price <span>Prediction</span></div>
            <div class="hero-subtitle">Machine Learning-Based Financial Market Prediction</div>
            <p style="margin: 0; color: #352B32; font-size: 0.96rem; line-height: 1.6; max-width: 900px; font-weight: 500;">
                Dashboard analisis regresi cerdas untuk memprediksi harga penutupan instrumen ETF United States Oil Fund (<b>USO_Close</b>) 
                berdasarkan 74 indikator pasar keuangan global (harga emas, S&P 500, Dow Jones, minyak mentah Brent, perak, dan indeks dolar AS).
            </p>
            <div style="margin-top: 16px;">
                <span class="badge-winner">Model Utama: Gradient Boosting Regressor</span>
                <span class="badge-info" style="margin-left: 8px;">Kelompok K6 (Genap)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # B. Metric Cards (Hasil Evaluasi Test Set)
    st.markdown("### 📈 Evaluasi Model Utama pada Test Set")
    st.markdown('<p style="color: #6F5965; font-size: 0.88rem; margin-top: -6px; margin-bottom: 14px;">Nilai metrik berikut diperoleh secara objektif dari pengujian terhadap 344 sampel pengujian kronologis (11 Agustus 2017 – 31 Desember 2018):</p>', unsafe_allow_html=True)

    gb_metrics = load_metrics("gradient_boosting")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">MAE (Absolut)</div>
                <div class="metric-card-value">${gb_metrics['MAE']:.2f}</div>
                <div class="metric-card-subtitle">Mean Absolute Error</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">MAPE (%)</div>
                <div class="metric-card-value">{gb_metrics['MAPE']:.2f}%</div>
                <div class="metric-card-subtitle">Mean Absolute Pct Error</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">MSE (Kuadrat)</div>
                <div class="metric-card-value">{gb_metrics['MSE']:.2f}</div>
                <div class="metric-card-subtitle">Mean Squared Error ($²)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">RMSE (Root MSE)</div>
                <div class="metric-card-value">${gb_metrics['RMSE']:.2f}</div>
                <div class="metric-card-subtitle">Akar Kuadrat MSE ($)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">R² Score</div>
                <div class="metric-card-value">{gb_metrics['R2_Score']:.2f}</div>
                <div class="metric-card-subtitle">Koefisien Determinasi</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # C. Prediction Visualization
    st.markdown("### 📉 Visualisasi Nilai Aktual vs Prediksi Model")
    pred_df = load_predictions("gradient_boosting")

    fig_actual_pred = create_actual_vs_predicted_chart(pred_df)
    st.plotly_chart(fig_actual_pred, use_container_width=True)

    fig_err = create_error_barchart(pred_df)
    st.plotly_chart(fig_err, use_container_width=True)

    # D & E. Model Overview & Performance Comparison
    col_ov, col_cmp = st.columns([1, 1.25])

    with col_ov:
        st.markdown("### 📋 Ringkasan Model & Data")
        st.markdown(
            """
            <div class="pink-card">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
                    <tr style="border-bottom: 1px solid #EBD5DF;">
                        <td style="padding: 10px 0; color: #6F5965; font-weight: 600;">Algoritma Utama:</td>
                        <td style="padding: 10px 0; font-weight: 700; text-align: right; color: #C94F83;">Gradient Boosting Regressor</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #EBD5DF;">
                        <td style="padding: 10px 0; color: #6F5965; font-weight: 600;">Target Prediksi:</td>
                        <td style="padding: 10px 0; font-weight: 700; text-align: right; color: #352B32;">USO_Close (Harga ETF Minyak)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #EBD5DF;">
                        <td style="padding: 10px 0; color: #6F5965; font-weight: 600;">Jumlah Fitur Prediktor:</td>
                        <td style="padding: 10px 0; font-weight: 700; text-align: right; color: #352B32;">74 Fitur Pasar Keuangan</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #EBD5DF;">
                        <td style="padding: 10px 0; color: #6F5965; font-weight: 600;">Data Pelatihan (Train):</td>
                        <td style="padding: 10px 0; font-weight: 700; text-align: right; color: #352B32;">1.374 baris (80% kronologis)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #EBD5DF;">
                        <td style="padding: 10px 0; color: #6F5965; font-weight: 600;">Data Pengujian (Test):</td>
                        <td style="padding: 10px 0; font-weight: 700; text-align: right; color: #352B32;">344 baris (20% kronologis)</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px 0; color: #6F5965; font-weight: 600;">Hyperparameter Kunci:</td>
                        <td style="padding: 10px 0; font-weight: 700; text-align: right; color: #C94F83;">loss='absolute_error', lr=0.1, n=100, depth=2</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_cmp:
        st.markdown("### ⚖️ Perbandingan Performa Model")
        comp_df, _ = load_comparison_data()

        display_comp = comp_df[["Metric", "AdaBoost_Regressor", "Gradient_Boosting_Regressor", "Superior_Model"]].copy()
        display_comp.columns = ["Metrik", "AdaBoost", "Gradient Boosting", "Model Unggul"]

        display_comp["AdaBoost"] = display_comp["AdaBoost"].apply(lambda x: f"{x:.4f}")
        display_comp["Gradient Boosting"] = display_comp["Gradient Boosting"].apply(lambda x: f"{x:.4f}")

        st.dataframe(display_comp, use_container_width=True, hide_index=True)

        st.info(
            "💡 **Catatan Analisis:** Gradient Boosting unggul pada 4 dari 5 metrik (MAE, MSE, RMSE, R²). "
            "Namun, kedua model mencatat skor R² negatif akibat pergeseran tingkat harga pasar minyak (*regime shift*) "
            "antara masa training (rata-rata $24.49) dan masa testing (rata-rata $12.60)."
        )


# ==============================================================================
# PAGE 2: PRICE PREDICTION
# ==============================================================================
elif page == "🔮 Price Prediction":
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">Make a <span>Prediction</span></div>
            <div class="hero-subtitle">Upload your market indicators dataset to predict USO closing prices.</div>
            <p style="margin: 0; color: #352B32; font-size: 0.96rem; line-height: 1.6; font-weight: 500;">
                Gunakan modul ini untuk melakukan inferensi langsung menggunakan model <b>Gradient Boosting Regressor</b> 
                yang telah terkompilasi. Anda dapat mengunggah CSV kustom berisi 74 fitur prediktor atau mencoba template contoh.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    required_features = get_feature_names()

    # Pilihan Sumber Data
    st.markdown("### 📁 1. Sumber Data Prediksi")
    tab_upload, tab_sample = st.tabs(["📤 Unggah File CSV Kustom", "🎯 Gunakan Contoh Data Testing"])

    with tab_upload:
        st.markdown('<p style="color: #6F5965; font-size: 0.88rem;">Unggah file CSV yang memuat tepat 74 fitur prediktor (kolom tambahan seperti <code>Date</code> diizinkan sebagai metadata pasif):</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            label="Pilih berkas CSV",
            type=["csv"],
            key="file_uploader_input",
            help="Pastikan CSV berisi ke-74 fitur prediktor yang sesuai dengan metadata training.",
        )

        if uploaded_file is not None:
            source_id = f"upload_{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("last_source_id") != source_id:
                try:
                    uploaded_file.seek(0)
                    raw_df = pd.read_csv(uploaded_file)
                    st.session_state["active_df"] = raw_df
                    st.session_state["active_source_name"] = uploaded_file.name
                    st.session_state["last_source_id"] = source_id
                    st.session_state["prediction_results"] = None  # Reset prediksi lama
                except Exception as e:
                    st.error(f"Gagal membaca file CSV: {e}")

    with tab_sample:
        st.markdown('<p style="color: #6F5965; font-size: 0.88rem;">Tidak memiliki data? Gunakan data testing historis resmi (fitur target <code>USO_Close</code> sudah dikeluarkan dari input):</p>', unsafe_allow_html=True)
        sample_choice = st.radio(
            "Pilih ukuran sampel:",
            ["Sampel 20 Baris (Cepat)", "Sampel 344 Baris Lengkap"],
            horizontal=True,
        )

        sample_df_preview = load_sample_dataset(short=(sample_choice == "Sampel 20 Baris (Cepat)"))

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("📥 Muat Data Contoh ke Input"):
                source_id = f"sample_{sample_choice}"
                st.session_state["active_df"] = sample_df_preview.copy()
                st.session_state["active_source_name"] = f"Contoh Data Testing ({len(sample_df_preview)} baris)"
                st.session_state["last_source_id"] = source_id
                st.session_state["prediction_results"] = None  # Reset prediksi lama
                st.rerun()

        with col_btn2:
            # Tombol unduh template CSV
            csv_buffer = io.StringIO()
            sample_df_preview.to_csv(csv_buffer, index=False)
            st.download_button(
                label="💾 Unduh Template CSV Contoh",
                data=csv_buffer.getvalue(),
                file_name="uso_prediction_sample_template.csv",
                mime="text/csv",
            )

    # Ambil data aktif dari session state
    df_to_predict = st.session_state.get("active_df")
    data_source_name = st.session_state.get("active_source_name", "")

    # B & C. CSV Validation and Preview
    if df_to_predict is not None and not df_to_predict.empty:
        st.markdown('<div class="pink-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🔍 2. Validasi Struktur & Preview Dataset")

        val_result = validate_uploaded_csv(df_to_predict, required_features)

        # Status Badge
        if val_result["is_valid"]:
            st.success(
                f"✅ **Validasi Berhasil!** Sumber data `{data_source_name}` memiliki **{val_result['row_count']} baris** data "
                f"dan memenuhi seluruh 74 fitur prediktor yang dibutuhkan model."
            )
            for w in val_result["warnings"]:
                st.warning(f"⚠️ {w}")
        else:
            st.error("❌ **Validasi Gagal!** Ditemukan masalah pada struktur data:")
            for err in val_result["errors"]:
                st.markdown(f"- 🔴 **{err}**")
            for w in val_result["warnings"]:
                st.markdown(f"- ⚠️ {w}")

        # Tampilan Preview Data
        with st.expander(f"👁️ Pratinjau Dataset ({min(10, len(df_to_predict))} dari {len(df_to_predict)} baris)", expanded=True):
            st.dataframe(df_to_predict.head(10), use_container_width=True)

        # D. Prediction Button
        if val_result["is_valid"]:
            st.markdown('<div class="pink-divider"></div>', unsafe_allow_html=True)
            st.markdown("### ⚡ 3. Jalankan Prediksi")

            if st.button("🔮 Predict USO Price", type="primary", key="btn_run_pred"):
                try:
                    with st.spinner("Memproses prediksi menggunakan Gradient Boosting Regressor..."):
                        model = load_model("gradient_boosting")
                        X_input = preprocess_uploaded_data(val_result["features_df"], required_features)

                        # Lakukan prediksi murni
                        predictions = model.predict(X_input)

                        # Susun hasil
                        results_df = pd.DataFrame()
                        for meta_col in val_result["metadata_cols"]:
                            results_df[meta_col] = df_to_predict[meta_col].values

                        results_df["Predicted_USO_Close"] = np.round(predictions, 2)

                        # Jika kolom target asli tersedia pada input (untuk evaluasi/pembanding)
                        if "USO_Close" in df_to_predict.columns:
                            results_df["Actual_USO_Close"] = np.round(df_to_predict["USO_Close"].values, 2)
                            results_df["Difference ($)"] = np.round(results_df["Actual_USO_Close"] - results_df["Predicted_USO_Close"], 2)
                            results_df["Error (%)"] = np.round(np.abs(results_df["Difference ($)"] / results_df["Actual_USO_Close"]) * 100.0, 2)

                        st.session_state["prediction_results"] = results_df
                        st.rerun()
                except Exception as e:
                    st.error(f"Gagal memproses prediksi: {e}")
                    st.exception(e)

    # E. Prediction Results Display
    if st.session_state.get("prediction_results") is not None:
        res_df = st.session_state["prediction_results"]

        st.markdown('<div class="pink-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 4. Hasil Prediksi Model")
        st.success(f"✨ Prediksi berhasil diselesaikan untuk total **{len(res_df)} baris data** menggunakan model Gradient Boosting Regressor!")

        # Metrik Ringkasan Prediksi
        p_mean = res_df["Predicted_USO_Close"].mean()
        p_min = res_df["Predicted_USO_Close"].min()
        p_max = res_df["Predicted_USO_Close"].max()

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Total Prediksi</div>
                    <div class="metric-card-value">{len(res_df)}</div>
                    <div class="metric-card-subtitle">Baris Sampel</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with rc2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Rata-rata Prediksi</div>
                    <div class="metric-card-value">${p_mean:.2f}</div>
                    <div class="metric-card-subtitle">Mean USO Close</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with rc3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Harga Terendah</div>
                    <div class="metric-card-value">${p_min:.2f}</div>
                    <div class="metric-card-subtitle">Min Predicted</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with rc4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Harga Tertinggi</div>
                    <div class="metric-card-value">${p_max:.2f}</div>
                    <div class="metric-card-subtitle">Max Predicted</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabel Hasil Prediksi
        st.dataframe(res_df, use_container_width=True)

        # Visualisasi Hasil Prediksi
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig_timeline = create_new_predictions_timeline(res_df)
            st.plotly_chart(fig_timeline, use_container_width=True)
        with col_chart2:
            fig_dist = create_prediction_distribution_chart(res_df["Predicted_USO_Close"])
            st.plotly_chart(fig_dist, use_container_width=True)

        # Tombol Download Hasil
        res_csv_buffer = io.StringIO()
        res_df.to_csv(res_csv_buffer, index=False)
        st.download_button(
            label="💾 Unduh Hasil Prediksi (CSV)",
            data=res_csv_buffer.getvalue(),
            file_name="hasil_prediksi_uso_close.csv",
            mime="text/csv",
        )

        st.caption(
            "⚠️ **Disclaimer Finansial:** Hasil prediksi merupakan estimasi statistik supervised learning "
            "berdasarkan indikator pasar pada hari yang sama dan bukan merupakan saran atau jaminan investasi trading."
        )


# ==============================================================================
# PAGE 3: MODEL COMPARISON
# ==============================================================================
elif page == "⚖️ Model Comparison":
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">Model Performance <span>Comparison</span></div>
            <div class="hero-subtitle">Comprehensive benchmarking between AdaBoost and Gradient Boosting Regressors.</div>
            <p style="margin: 0; color: #352B32; font-size: 0.96rem; line-height: 1.6; font-weight: 500;">
                Evaluasi komparatif kedua arsitektur ensemble boosting pada 344 data pengujian kronologis independen yang sama.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    comp_df, comp_summary = load_comparison_data()

    # Visualisasi Bar Chart Perbandingan
    fig_comp = create_comparison_charts(comp_df)
    st.plotly_chart(fig_comp, use_container_width=True)

    # Tabel Perbandingan Terstruktur
    st.markdown("### 📊 Tabel Matriks Perbandingan Metrik")

    display_table = comp_df[["Metric", "Description", "AdaBoost_Regressor", "Gradient_Boosting_Regressor", "Difference_GB_vs_Ada", "Superior_Model", "Goal"]].copy()
    display_table.columns = ["Metrik", "Deskripsi", "AdaBoost", "Gradient Boosting", "Selisih (GB - Ada)", "Model Unggul", "Kriteria"]

    display_table["AdaBoost"] = display_table["AdaBoost"].apply(lambda x: f"{x:.4f}")
    display_table["Gradient Boosting"] = display_table["Gradient Boosting"].apply(lambda x: f"{x:.4f}")
    display_table["Selisih (GB - Ada)"] = display_table["Selisih (GB - Ada)"].apply(lambda x: f"{x:+.4f}")

    st.dataframe(display_table, use_container_width=True, hide_index=True)

    # Interpretasi Hasil Komparasi
    st.markdown("### 💡 Interpretasi & Pembahasan Analitik")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            <div class="pink-card">
                <h4 style="color: #C94F83; margin-top: 0;">1. Mengapa Gradient Boosting Unggul?</h4>
                <p style="font-size: 0.88rem; color: #352B32; line-height: 1.6;">
                    • <b>Reduksi Penalti Kuadrat (MSE & RMSE):</b> MSE pada Gradient Boosting turun sebesar <b>35.5%</b> 
                    (dari 64.29 ke 41.49) dan RMSE turun sebesar <b>19.7%</b> (dari $8.02 ke $6.44).<br>
                    • <b>Keunggulan Fungsi Loss:</b> Penggunaan loss <code>absolute_error</code> (LAD) secara matematis 
                    mengoptimalkan median bersyarat residu, memberikan ketahanan superior terhadap fluktuasi tajam dibanding 
                    pembobotan AdaBoost.R2.<br>
                    • <b>Shallow Tree Regularization:</b> Pembatasan <code>max_depth=2</code> mencegah pohon menangkap noise 
                    berlebih dari 74 fitur prediktor.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="pink-card">
                <h4 style="color: #C94F83; margin-top: 0;">2. Penjelasan Skor R² Negatif</h4>
                <p style="font-size: 0.88rem; color: #352B32; line-height: 1.6;">
                    • Skor R² bernilai negatif pada kedua model (-13.41 pada GB dan -21.33 pada AdaBoost) 
                    bukan disebabkan kesalahan kode, melainkan <b>pergeseran rezim harga historis (regime shift)</b>.<br>
                    • Pada data training (2011–2017), harga USO bergerak tinggi ($8 s.d. $42, rata-rata $24.49). 
                    Sedangkan pada masa testing (2017–2018), harga jatuh dan tertekan di level rendah ($9 s.d. $16, rata-rata $12.60) 
                    dengan variansi internal hanya <b>2.88</b>.<br>
                    • Model ensemble pohon tidak dapat melakukan ekstrapolasi linear melampaui distribusi daun historisnya.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 4: ABOUT MODEL
# ==============================================================================
elif page == "📖 About Model":
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">About This <span>Project</span></div>
            <div class="hero-subtitle">Comprehensive documentation of dataset, methodology, models, and limitations.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🏢 1. Gambaran Umum Proyek")
    st.markdown(
        """
        <p style="color: #352B32; font-size: 0.94rem; line-height: 1.6;">
            Aplikasi ini dikembangkan sebagai bagian dari <b>Praktikum Pembelajaran Mesin Modul 11 — Regresi</b> 
            oleh <b>Kelompok K6 (Genap)</b> Program Studi D4 Teknik Informatika, Fakultas Vokasi, Universitas Airlangga. 
            Fokus utama praktikum ini adalah pemodelan prediktif harga kontinu menggunakan dua algoritma ensemble learning tingkat lanjut: 
            <b>AdaBoost Regressor</b> dan <b>Gradient Boosting Regressor</b>.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 2. Informasi Dataset")
    st.markdown(
        """
        <div class="pink-card">
            <ul style="margin: 0; padding-left: 20px; font-size: 0.9rem; color: #352B32; line-height: 1.65;">
                <li><b>Nama Dataset:</b> <code>FINAL_USO.csv</code> (Gold Price Prediction / Financial Indices).</li>
                <li><b>Target Variabel (y):</b> <code>USO_Close</code> (harga penutupan harian ETF United States Oil Fund).</li>
                <li><b>Variabel Prediktor (X):</b> 74 fitur pasar keuangan global mencakup indeks S&P 500, Dow Jones, harga minyak Brent, perak, platinum, paladium, dan indeks dolar AS.</li>
                <li><b>Pencegahan Data Leakage:</b> Seluruh kolom derivatif <code>USO_*</code> (Open, High, Low, Adj Close, Volume) dikeluarkan dari variabel prediktor oleh Person 1 untuk mencegah pembocoran informasi pada hari yang sama.</li>
                <li><b>Pembagian Data (Train/Test Split):</b> 80% Training (1.374 baris, 15 Des 2011 – 10 Agu 2017) dan 20% Testing (344 baris, 11 Agu 2017 – 31 Des 2018) secara kronologis murni tanpa pengacakan (shuffle).</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🔄 3. Alur Metodologi Pemodelan")
    st.markdown(
        """
        <div style="background-color: #FFFFFF; border: 1.5px solid #EBD5DF; border-radius: 16px; padding: 22px; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 16px rgba(201, 79, 131, 0.05);">
            <span style="background: #FCE7F0; color: #C94F83; padding: 8px 14px; border-radius: 8px; font-weight: 700; font-size: 0.88rem; border: 1px solid #E879A8;">1. Dataset Mentah</span> ➔ 
            <span style="background: #FCE7F0; color: #C94F83; padding: 8px 14px; border-radius: 8px; font-weight: 700; font-size: 0.88rem; border: 1px solid #E879A8;">2. Preprocessing & Imputasi</span> ➔ 
            <span style="background: #FCE7F0; color: #C94F83; padding: 8px 14px; border-radius: 8px; font-weight: 700; font-size: 0.88rem; border: 1px solid #E879A8;">3. Split Kronologis (80:20)</span> ➔ 
            <span style="background: #FCE7F0; color: #C94F83; padding: 8px 14px; border-radius: 8px; font-weight: 700; font-size: 0.88rem; border: 1px solid #E879A8;">4. Pelatihan Model</span> ➔ 
            <span style="background: #FCE7F0; color: #C94F83; padding: 8px 14px; border-radius: 8px; font-weight: 700; font-size: 0.88rem; border: 1px solid #E879A8;">5. Evaluasi 5 Metrik</span> ➔ 
            <span style="background: #FCE7F0; color: #C94F83; padding: 8px 14px; border-radius: 8px; font-weight: 700; font-size: 0.88rem; border: 1px solid #E879A8;">6. Integrasi UI/UX</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ 4. Informasi Algoritma Model")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(
            """
            <div class="pink-card">
                <h4 style="color: #C94F83; margin-top: 0;">AdaBoost Regressor</h4>
                <p style="font-size: 0.88rem; color: #352B32; line-height: 1.55;">
                    Algoritma boosting adaptif yang bekerja dengan memberikan bobot lebih besar pada sampel data yang memiliki 
                    margin kesalahan prediksi tinggi dari pembelajar sebelumnya. Pohon-pohon selanjutnya difokuskan untuk 
                    memperbaiki residual sampel yang sulit diprediksi tersebut.
                </p>
                <div style="font-size: 0.8rem; color: #6F5965; font-weight: 600;">
                    • Base Estimator: DecisionTreeRegressor (depth=3)<br>
                    • n_estimators: 50 | learning_rate: 1.0 | loss: linear
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            """
            <div class="pink-card">
                <h4 style="color: #C94F83; margin-top: 0;">Gradient Boosting Regressor</h4>
                <p style="font-size: 0.88rem; color: #352B32; line-height: 1.55;">
                    Algoritma boosting sekuensial yang melatih setiap pohon baru untuk memprediksi gradien negatif dari fungsi kerugian 
                    (loss function) menggunakan teknik gradient descent. Mampu mengoptimalkan berbagai fungsi objektif non-linier secara bertahap.
                </p>
                <div style="font-size: 0.8rem; color: #6F5965; font-weight: 600;">
                    • loss: absolute_error (Least Absolute Deviations)<br>
                    • n_estimators: 100 | learning_rate: 0.1 | max_depth: 2
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### ⚠️ 5. Keterbatasan Model & Batasan Prediksi")
    st.markdown(
        """
        <div class="pink-card" style="border-left: 5px solid #C94F83;">
            <ol style="margin: 0; padding-left: 20px; font-size: 0.88rem; color: #352B32; line-height: 1.65;">
                <li><b>Keterbatasan Ekstrapolasi Pohon Regresi:</b> Model ensemble berbasis decision tree tidak dapat mengekstrapolasi tren linear ke bawah melampaui batas nilai daun terendah pada data latihnya.</li>
                <li><b>Tidak Adanya Fitur Autoregresif / Lag:</b> Model saat ini hanya mengandalkan fitur pasar pada hari yang sama (same-day cross market) tanpa menyertakan harga penutupan USO hari-hari sebelumnya (t-1, t-2).</li>
                <li><b>Pergeseran Rezim Pasar:</b> Karena adanya perbedaan level harga drastis antara periode 2011–2017 dan 2017–2018, skor R² pengujian bernilai negatif. Model belum layak dijadikan instrumen trading finansial otomatis riil.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
