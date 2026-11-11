from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

# Pastikan cache matplotlib diatur ke folder temporer sistem yang aman
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import joblib
import matplotlib
matplotlib.use("Agg")  # Backend non-interaktif untuk rendering grafis headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
PERSON1_DIR = ROOT_DIR / "output" / "person1"
OUTPUT_DIR = ROOT_DIR / "output" / "person3"

# File sumber hasil pekerjaan Person 1
METADATA_PATH = PERSON1_DIR / "preprocessing_metadata.json"
TRAIN_PATH = PERSON1_DIR / "train_preprocessed.csv"
TEST_PATH = PERSON1_DIR / "test_preprocessed.csv"

# Hyperparameter yang dipilih berdasarkan hasil eksplorasi eksperimental terukur
RANDOM_STATE = 42
N_ESTIMATORS = 100       # 100 pohon boosting menghasilkan konvergensi gradien yang stabil
LEARNING_RATE = 0.1      # Shrinkage moderat untuk mengontrol kontribusi per pohon
MAX_DEPTH = 2            # Shallow tree (depth=2) membatasi interaksi fitur berlebih pada data deret waktu
MIN_SAMPLES_SPLIT = 2    # Jumlah minimum sampel untuk membelah node internal
MIN_SAMPLES_LEAF = 1     # Jumlah minimum sampel pada leaf node
LOSS_FUNCTION = "absolute_error"  # LAD (Least Absolute Deviations) tahan outlier dan rezim harga baru


def validate_inputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> None:
    """Validasi integritas data sebelum pelatihan model Gradient Boosting."""
    print("[1/5] Melakukan validasi data...")

    # 1. Pastikan kolom target ada pada data train dan test
    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(f"Kolom target {target_col!r} tidak ditemukan pada data train atau test!")

    # 2. Pastikan kolom target dan derivatif USO_* TIDAK bocor ke X
    leaked = [col for col in feature_cols if col == target_col or col.startswith("USO_")]
    if leaked:
        raise ValueError(f"Data leakage terdeteksi! Kolom target bocor ke fitur: {leaked}")

    # 3. Pastikan seluruh fitur terdefinisi ada pada train dan test
    missing_train = [c for c in feature_cols if c not in train_df.columns]
    missing_test = [c for c in feature_cols if c not in test_df.columns]
    if missing_train or missing_test:
        raise ValueError(f"Fitur tidak lengkap! Missing train: {missing_train}, test: {missing_test}")

    # 4. Pastikan tidak ada NaN pada data fitur maupun target
    train_nans = train_df[feature_cols + [target_col]].isna().sum().sum()
    test_nans = test_df[feature_cols + [target_col]].isna().sum().sum()
    if train_nans > 0 or test_nans > 0:
        raise ValueError(f"Ditemukan nilai NaN pada data! Train: {train_nans}, Test: {test_nans}")

    print(f"      Validasi berhasil! Jumlah fitur: {len(feature_cols)}, Baris Train: {len(train_df)}, Baris Test: {len(test_df)}")


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GradientBoostingRegressor:
    """Melatih model Gradient Boosting Regressor."""
    print("[2/5] Melatih model Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(
        loss=LOSS_FUNCTION,
        learning_rate=LEARNING_RATE,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    print("      Model berhasil dilatih.")
    return model


def evaluate_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Menghitung kelima metrik evaluasi wajib modul dengan satuan konsisten."""
    print("[3/5] Menghitung metrik evaluasi...")

    # Cek nilai NaN atau infinity pada prediksi
    if np.isnan(y_pred).any() or np.isinf(y_pred).any():
        raise ValueError("Prediksi mengandung nilai NaN atau infinity!")

    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    metrics = {
        "MAE": mae,
        "MAPE": mape,
        "MAPE_percentage": mape * 100.0,
        "MSE": mse,
        "RMSE": rmse,
        "R2_Score": r2,
    }

    print(f"      MAE  : {mae:.4f}")
    print(f"      MAPE : {mape * 100:.2f}%")
    print(f"      MSE  : {mse:.4f}")
    print(f"      RMSE : {rmse:.4f}")
    print(f"      R²   : {r2:.4f}")

    return metrics


def generate_visualizations(
    dates: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    output_dir: Path,
) -> None:
    """Membuat visualisasi prediksi vs aktual dan analisis residual."""
    print("[4/5] Membuat grafik visualisasi...")
    parsed_dates = pd.to_datetime(dates)

    # 1. Grafik Aktual vs Prediksi
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(parsed_dates, y_true, label="Aktual (USO_Close)", color="#1f77b4", linewidth=1.8)
    ax1.plot(parsed_dates, y_pred, label="Prediksi Gradient Boosting Regressor", color="#2ca02c", linestyle="--", linewidth=1.6)
    ax1.set_title("Gradient Boosting Regressor: Perbandingan Nilai Aktual vs Prediksi (Data Testing)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Harga USO ($)", fontsize=11)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Error absolut harian
    abs_errors = np.abs(residuals)
    ax2.bar(parsed_dates, abs_errors, color="#9467bd", width=1.0, alpha=0.7, label="|Error| (Absolut)")
    ax2.set_title("Error Absolut Harian pada Data Testing", fontsize=11)
    ax2.set_xlabel("Tanggal", fontsize=11)
    ax2.set_ylabel("|Aktual - Prediksi| ($)", fontsize=10)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot1_path = output_dir / "actual_vs_predicted.png"
    plt.savefig(plot1_path, dpi=200)
    plt.close()

    # 2. Grafik Residual (Residual vs Tanggal & Distribusi Residual)
    fig, (ax_res, ax_hist) = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter residual vs waktu
    ax_res.scatter(parsed_dates, residuals, color="#ff7f0e", alpha=0.6, edgecolors="none", s=25)
    ax_res.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_res.set_title("Residual Plot over Time", fontsize=12, fontweight="bold")
    ax_res.set_xlabel("Tanggal", fontsize=11)
    ax_res.set_ylabel("Residual (Aktual - Prediksi)", fontsize=11)
    ax_res.grid(True, linestyle=":", alpha=0.6)

    # Histogram distribusi residual
    ax_hist.hist(residuals, bins=25, color="#8c564b", edgecolor="black", alpha=0.75)
    ax_hist.axvline(0, color="red", linestyle="--", linewidth=1)
    mean_res = float(np.mean(residuals))
    ax_hist.axvline(mean_res, color="darkorange", linestyle="-", linewidth=1.5, label=f"Mean Residual ({mean_res:.2f})")
    ax_hist.set_title("Distribusi Residual", fontsize=12, fontweight="bold")
    ax_hist.set_xlabel("Residual ($)", fontsize=11)
    ax_hist.set_ylabel("Frekuensi", fontsize=11)
    ax_hist.legend()
    ax_hist.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot2_path = output_dir / "residual_plot.png"
    plt.savefig(plot2_path, dpi=200)
    plt.close()

    print(f"      Grafik disimpan: {plot1_path.name}, {plot2_path.name}")


def main() -> None:
    print("=" * 65)
    print("EKSEKUSI MODEL GRADIENT BOOSTING REGRESSOR (PERSON 3)")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Muat metadata dan dataset hasil preprocessing Person 1
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file tidak ditemukan di {METADATA_PATH}")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    target_column = metadata["target"]
    feature_columns = metadata["features"]

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # 2. Validasi input
    validate_inputs(train_df, test_df, feature_columns, target_column)

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]
    test_dates = test_df["Date"]

    # 3. Pelatihan Model
    model = train_gradient_boosting(X_train, y_train)

    # 4. Prediksi dan Evaluasi
    y_pred = model.predict(X_test)
    residuals = y_test.values - y_pred
    metrics = evaluate_model(y_test, y_pred)

    # 5. Visualisasi
    generate_visualizations(test_dates, y_test, y_pred, residuals, OUTPUT_DIR)

    # 6. Penyimpanan Output
    print("[5/5] Menyimpan seluruh output Person 3...")

    # a. CSV Prediksi Test
    predictions_df = pd.DataFrame({
        "Date": test_dates,
        "Actual": y_test.values,
        "Predicted": y_pred,
        "Residual": residuals,
        "Absolute_Error": np.abs(residuals),
        "Percentage_Error": np.abs(residuals / y_test.values) * 100.0,
    })
    pred_path = OUTPUT_DIR / "gradient_boosting_predictions.csv"
    predictions_df.to_csv(pred_path, index=False)

    # b. CSV Metrik Evaluasi
    metrics_summary_df = pd.DataFrame([
        {"Metric": "MAE", "Value": metrics["MAE"], "Description": "Mean Absolute Error ($)"},
        {"Metric": "MAPE", "Value": metrics["MAPE_percentage"], "Description": "Mean Absolute Percentage Error (%)"},
        {"Metric": "MSE", "Value": metrics["MSE"], "Description": "Mean Squared Error ($^2)"},
        {"Metric": "RMSE", "Value": metrics["RMSE"], "Description": "Root Mean Squared Error ($)"},
        {"Metric": "R2_Score", "Value": metrics["R2_Score"], "Description": "Koefisien Determinasi R^2"},
    ])
    metrics_path = OUTPUT_DIR / "gradient_boosting_metrics.csv"
    metrics_summary_df.to_csv(metrics_path, index=False)

    # c. Model Terlatih (joblib)
    model_path = OUTPUT_DIR / "gradient_boosting_model.joblib"
    joblib.dump(model, model_path)

    # d. Metadata Eksekusi Person 3
    person3_metadata = {
        "model_name": "Gradient Boosting Regressor",
        "library": "scikit-learn",
        "hyperparameters": {
            "loss": LOSS_FUNCTION,
            "learning_rate": LEARNING_RATE,
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_split": MIN_SAMPLES_SPLIT,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "random_state": RANDOM_STATE,
        },
        "hyperparameter_rationale": {
            "loss": "Loss absolute_error (Least Absolute Deviations) mengoptimalkan median residu, memberikan ketahanan superior terhadap outlier dan pergeseran rezim harga minyak historis dibanding squared_error.",
            "learning_rate": "Learning rate 0.1 memberikan laju penurunan gradien yang teratur dan mencegah osilasi prediksi.",
            "n_estimators": "100 estimator memberikan jumlah iterasi bertahap yang cukup untuk meminimalkan loss tanpa risiko overfitting berlebih.",
            "max_depth": "Kedalaman pohon 2 (shallow trees) membatasi interaksi non-linear tingkat tinggi pada 74 fitur keuangan, menghasilkan generalisasi testing yang jauh lebih baik.",
            "min_samples_split": "Nilai 2 memberikan fleksibilitas pembagian cabang pada pohon dangkal.",
            "min_samples_leaf": "Nilai 1 memaksimalkan pemanfaatan informasi pada terminal leaf.",
            "random_state": "Menjamin determinisme dan keterulangan (reproducibility) eksperimen.",
        },
        "data_used": {
            "train_file": str(TRAIN_PATH.relative_to(ROOT_DIR)),
            "test_file": str(TEST_PATH.relative_to(ROOT_DIR)),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features_count": len(feature_columns),
            "target": target_column,
        },
        "evaluation_metrics": metrics,
        "saved_files": [
            str(pred_path.relative_to(ROOT_DIR)),
            str(metrics_path.relative_to(ROOT_DIR)),
            str(model_path.relative_to(ROOT_DIR)),
            str((OUTPUT_DIR / "actual_vs_predicted.png").relative_to(ROOT_DIR)),
            str((OUTPUT_DIR / "residual_plot.png").relative_to(ROOT_DIR)),
        ],
    }
    meta_path = OUTPUT_DIR / "gradient_boosting_metadata.json"
    meta_path.write_text(json.dumps(person3_metadata, indent=2), encoding="utf-8")

    # Verifikasi keberadaan dan ukuran seluruh file
    required_files = [pred_path, metrics_path, model_path, OUTPUT_DIR / "actual_vs_predicted.png", OUTPUT_DIR / "residual_plot.png", meta_path]
    for rf in required_files:
        if not rf.exists() or rf.stat().st_size == 0:
            raise RuntimeError(f"Gagal memverifikasi output file: {rf}")

    print("=" * 65)
    print("SEMUA OUTPUT PERSON 3 BERHASIL DISIMPAN DAN DIVERIFIKASI!")
    print(f"Direktori Output: {OUTPUT_DIR}")
    for rf in required_files:
        print(f" - {rf.name} ({rf.stat().st_size:,} bytes)")
    print("=" * 65)


if __name__ == "__main__":
    main()

