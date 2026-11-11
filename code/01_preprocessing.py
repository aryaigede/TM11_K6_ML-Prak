from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "dataset" / "FINAL_USO.csv"
OUTPUT_DIR = ROOT_DIR / "output" / "person1"
TARGET_COLUMN = "USO_Close"
TEST_SIZE = 0.20


def build_feature_columns(columns: list[str]) -> list[str]:
    """Return predictors that are available before the target is observed."""
    excluded = {"Date", TARGET_COLUMN}
    return [
        column for column in columns
        if column not in excluded and not column.startswith("USO_")
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    initial_rows, initial_columns = raw.shape

    raw["Date"] = pd.to_datetime(raw["Date"], errors="raise")
    raw = raw.sort_values("Date").drop_duplicates().reset_index(drop=True)

    numeric_columns = raw.select_dtypes(include="number").columns.tolist()
    invalid_numeric = {
        column: int((~raw[column].map(pd.notna)).sum())
        for column in numeric_columns
    }
    invalid_numeric = {
        column: count for column, count in invalid_numeric.items() if count
    }
    if raw[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column {TARGET_COLUMN!r} contains missing values.")

    feature_columns = build_feature_columns(raw.columns.tolist())
    if not feature_columns:
        raise ValueError("No predictor columns remain after leakage prevention.")

    split_index = int(len(raw) * (1 - TEST_SIZE))
    train = raw.iloc[:split_index].copy()
    test = raw.iloc[split_index:].copy()

    imputer = SimpleImputer(strategy="median")
    train_features = pd.DataFrame(
        imputer.fit_transform(train[feature_columns]),
        columns=feature_columns,
        index=train.index,
    )
    test_features = pd.DataFrame(
        imputer.transform(test[feature_columns]),
        columns=feature_columns,
        index=test.index,
    )

    train_output = train[["Date", TARGET_COLUMN]].copy()
    test_output = test[["Date", TARGET_COLUMN]].copy()
    train_output = pd.concat([train[["Date"]], train_features, train[[TARGET_COLUMN]]], axis=1)
    test_output = pd.concat([test[["Date"]], test_features, test[[TARGET_COLUMN]]], axis=1)

    train_output.to_csv(OUTPUT_DIR / "train_preprocessed.csv", index=False, date_format="%Y-%m-%d")
    test_output.to_csv(OUTPUT_DIR / "test_preprocessed.csv", index=False, date_format="%Y-%m-%d")
    raw.to_csv(OUTPUT_DIR / "dataset_cleaned.csv", index=False, date_format="%Y-%m-%d")

    quality = pd.DataFrame(
        {
            "column": raw.columns,
            "dtype": raw.dtypes.astype(str).values,
            "missing_count": raw.isna().sum().values,
            "unique_count": raw.nunique(dropna=True).values,
        }
    )
    quality.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)

    metadata = {
        "dataset_file": "dataset/FINAL_USO.csv",
        "target": TARGET_COLUMN,
        "target_description": "USO ETF closing price",
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "excluded_from_features": ["Date", "USO_Open", "USO_High", "USO_Low", "USO_Close", "USO_Adj Close", "USO_Volume"],
        "rows_before_cleaning": initial_rows,
        "columns_before_cleaning": initial_columns,
        "rows_after_cleaning": len(raw),
        "duplicate_rows_removed": initial_rows - len(raw),
        "missing_values_after_cleaning": int(raw.isna().sum().sum()),
        "invalid_numeric_values": invalid_numeric,
        "split_method": "chronological",
        "train_rows": len(train),
        "test_rows": len(test),
        "train_date_range": [train["Date"].min().strftime("%Y-%m-%d"), train["Date"].max().strftime("%Y-%m-%d")],
        "test_date_range": [test["Date"].min().strftime("%Y-%m-%d"), test["Date"].max().strftime("%Y-%m-%d")],
        "imputation": "median fitted on training features only",
    }
    (OUTPUT_DIR / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    plt.figure(figsize=(11, 4))
    plt.plot(raw["Date"], raw[TARGET_COLUMN], linewidth=1)
    plt.axvline(test["Date"].iloc[0], color="crimson", linestyle="--", label="Batas train/test")
    plt.title("USO Close dan Batas Pembagian Data Kronologis")
    plt.xlabel("Tanggal")
    plt.ylabel("USO Close")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_train_test_split.png", dpi=150)
    plt.close()

    print(f"Dataset awal: {initial_rows} baris, {initial_columns} kolom")
    print(f"Dataset bersih: {len(raw)} baris")
    print(f"Fitur: {len(feature_columns)} | Target: {TARGET_COLUMN}")
    print(f"Train: {len(train)} baris | Test: {len(test)} baris")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
