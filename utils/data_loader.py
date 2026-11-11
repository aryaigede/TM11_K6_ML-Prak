"""
Data and Model Loader Utility
Centralized access to trained models, metadata, and evaluation results with Streamlit caching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]

P1_DIR = ROOT_DIR / "output" / "person1"
P2_DIR = ROOT_DIR / "output" / "person2"
P3_DIR = ROOT_DIR / "output" / "person3"
COMP_DIR = ROOT_DIR / "output" / "model_comparison"


@st.cache_resource(show_spinner=False)
def load_model(model_name: str = "gradient_boosting") -> Any:
    """Memuat model scikit-learn serialisasi joblib."""
    if model_name == "gradient_boosting":
        model_path = P3_DIR / "gradient_boosting_model.joblib"
    elif model_name == "adaboost":
        model_path = P2_DIR / "adaboost_model.joblib"
    else:
        raise ValueError(f"Model {model_name!r} tidak dikenal.")

    if not model_path.exists():
        raise FileNotFoundError(f"File model tidak ditemukan di {model_path}")

    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_preprocessing_metadata() -> dict[str, Any]:
    """Memuat metadata preprocessing Person 1 (termasuk daftar 74 fitur)."""
    meta_path = P1_DIR / "preprocessing_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file tidak ditemukan di {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def get_feature_names() -> list[str]:
    """Mendapatkan daftar 74 nama fitur yang digunakan saat pelatihan model."""
    meta = load_preprocessing_metadata()
    return meta["features"]


@st.cache_data(show_spinner=False)
def load_metrics(model_name: str = "gradient_boosting") -> dict[str, float]:
    """Memuat metrik evaluasi model (MAE, MAPE, MSE, RMSE, R2)."""
    if model_name == "gradient_boosting":
        csv_path = P3_DIR / "gradient_boosting_metrics.csv"
    elif model_name == "adaboost":
        csv_path = P2_DIR / "adaboost_metrics.csv"
    else:
        raise ValueError(f"Model {model_name!r} tidak dikenal.")

    df = pd.read_csv(csv_path)
    return df.set_index("Metric")["Value"].to_dict()


@st.cache_data(show_spinner=False)
def load_predictions(model_name: str = "gradient_boosting") -> pd.DataFrame:
    """Memuat tabel hasil prediksi testing (Date, Actual, Predicted, Residual, dll)."""
    if model_name == "gradient_boosting":
        csv_path = P3_DIR / "gradient_boosting_predictions.csv"
    elif model_name == "adaboost":
        csv_path = P2_DIR / "adaboost_predictions.csv"
    else:
        raise ValueError(f"Model {model_name!r} tidak dikenal.")

    return pd.read_csv(csv_path)


@st.cache_data(show_spinner=False)
def load_comparison_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Memuat tabel perbandingan model dan ringkasan JSON."""
    csv_path = COMP_DIR / "model_comparison.csv"
    json_path = COMP_DIR / "comparison_summary.json"

    df = pd.read_csv(csv_path)
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    return df, summary


@st.cache_data(show_spinner=False)
def load_sample_dataset(short: bool = True) -> pd.DataFrame:
    """Memuat contoh data CSV pengujian untuk demonstrasi input prediksi."""
    sample_file = ROOT_DIR / "sample_data" / ("sample_20_rows.csv" if short else "sample_test_features.csv")
    if not sample_file.exists():
        # Fallback jika belum terbuat
        test_df = pd.read_csv(P1_DIR / "test_preprocessed.csv")
        feats = get_feature_names()
        cols = ["Date"] + feats if "Date" in test_df.columns else feats
        return test_df[cols].head(20 if short else len(test_df))
    return pd.read_csv(sample_file)

