"""
CSV Validation and Preprocessing Module for User Uploads
Ensures high integrity, explicit missing column reporting, and strict type safety.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def validate_uploaded_csv(
    df: pd.DataFrame,
    required_features: list[str],
) -> dict[str, Any]:
    """
    Validasi komprehensif terhadap file CSV yang diunggah pengguna:
    1. Dataframe tidak boleh kosong.
    2. Seluruh 74 fitur prediktor harus tersedia.
    3. Kolom fitur yang hilang dilaporkan secara eksplisit (tidak diisi 0 diam-diam).
    4. Seluruh fitur harus dapat dikonversi ke numerik tanpa error.
    5. Tidak boleh ada nilai NaN atau tak terhingga (infinity).
    6. Kolom tambahan (seperti 'Date') diizinkan sebagai metadata pasif.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Pengecekan data kosong
    if df is None or df.empty:
        return {
            "is_valid": False,
            "errors": ["File CSV kosong atau tidak memiliki baris data."],
            "warnings": [],
            "features_df": None,
            "metadata_cols": [],
            "row_count": 0,
        }

    # 2. Pengecekan kelengkapan kolom fitur
    uploaded_cols = set(df.columns)
    required_set = set(required_features)

    missing_cols = [col for col in required_features if col not in uploaded_cols]
    if missing_cols:
        errors.append(
            f"Terdapat {len(missing_cols)} kolom fitur prediktor yang hilang dari CSV! "
            f"Kolom yang belum ada: {', '.join(missing_cols[:8])}"
            + (f" ...dan {len(missing_cols) - 8} kolom lainnya." if len(missing_cols) > 8 else ".")
        )

    # 3. Deteksi kolom target jika pengguna sengaja/tidak sengaja menyertakannya
    if "USO_Close" in df.columns:
        warnings.append(
            "Kolom 'USO_Close' ditemukan dalam file. Kolom ini dikeluarkan dari variabel input prediktor "
            "dan hanya akan disimpan sebagai pembanding aktual jika relevan."
        )

    # 4. Deteksi kolom metadata tambahan (seperti 'Date', 'ID')
    metadata_cols = [col for col in df.columns if col not in required_set and col != "USO_Close"]
    if metadata_cols:
        warnings.append(
            f"Ditemukan kolom metadata tambahan: {', '.join(metadata_cols)}. "
            "Kolom ini tetap dipertahankan pada tabel hasil namun diisolasi dari input model."
        )

    # Jika ada error kolom hilang, langsung hentikan validasi sebelum konversi tipe
    if errors:
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "features_df": None,
            "metadata_cols": metadata_cols,
            "row_count": len(df),
        }

    # 5. Ekstraksi fitur dalam urutan tepat 74 fitur pelatihan
    features_df = df[required_features].copy()

    # 6. Validasi tipe data numerik
    non_numeric_cols: list[str] = []
    for col in required_features:
        if not pd.api.types.is_numeric_dtype(features_df[col]):
            # Coba konversi paksa untuk melihat apakah ada string non-angka
            converted = pd.to_numeric(features_df[col], errors="coerce")
            if converted.isna().any():
                non_numeric_cols.append(col)
            else:
                features_df[col] = converted

    if non_numeric_cols:
        errors.append(
            f"Ditemukan nilai non-numerik pada kolom: {', '.join(non_numeric_cols[:5])}"
            + (f" (+{len(non_numeric_cols) - 5} lainnya)" if len(non_numeric_cols) > 5 else "")
            + ". Seluruh 74 fitur harus berupa angka numerik."
        )

    # 7. Validasi NaN dan Infinity
    nan_counts = features_df.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0].index.tolist()
    if cols_with_nan:
        errors.append(
            f"Ditemukan nilai NaN / kosong pada kolom: {', '.join(cols_with_nan[:5])}"
            + (f" (+{len(cols_with_nan) - 5} lainnya)" if len(cols_with_nan) > 5 else "")
            + ". Semua fitur harus terisi lengkap."
        )

    # Cek Infinity hanya jika tidak ada error tipe data sebelumnya
    if not non_numeric_cols:
        try:
            numeric_vals = features_df.to_numpy(dtype=float)
            if np.isinf(numeric_vals).any():
                errors.append("Ditemukan nilai tak terhingga (infinity) dalam data fitur.")
        except Exception:
            pass

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "features_df": features_df if is_valid else None,
        "metadata_cols": metadata_cols,
        "row_count": len(df),
    }


def preprocess_uploaded_data(
    df: pd.DataFrame,
    required_features: list[str],
) -> pd.DataFrame:
    """Mengembalikan DataFrame fitur murni dalam urutan kolom yang tepat sesuai model."""
    return df[required_features].copy().astype(float)
