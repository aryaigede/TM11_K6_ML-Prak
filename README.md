# Praktikum Machine Learning Modul 11 — Regresi (Kelompok K6)

Repository ini berisi implementasi lengkap tugas **Praktikum Machine Learning Modul 11: Ensemble Regression** oleh **Kelompok K6**. Proyek ini berfokus pada prediksi harga penutupan United States Oil Fund (**`USO_Close`**) menggunakan algoritma ensemble **AdaBoost Regressor** dan **Gradient Boosting Regressor**, serta dilengkapi dengan dashboard interaktif berbasis **Streamlit** bertema *Pink Aesthetic*.

---

## 👥 Pembagian Tugas Kelompok K6

1. **Person 1 — Data Preprocessing & Pipeline**
   - Audit data, penanganan missing values, pembersihan data duplikat.
   - Pembagian dataset kronologis (80% training = 1.374 baris, 20% testing = 344 baris) untuk mencegah look-ahead bias deret waktu.
   - Eliminasi fitur target leakage (`USO_High`, `USO_Low`, `USO_Open`, `USO_Volume`).
   - Menyediakan 74 prediktor terstandarisasi untuk pemodelan bersama.

2. **Person 2 — AdaBoost Regressor**
   - Implementasi model `AdaBoostRegressor` (`sklearn.ensemble`).
   - Konfigurasi hyperparameter (`n_estimators=100`, `learning_rate=0.05`, `loss='exponential'`).
   - Evaluasi metrik regresi: MAE, MAPE, MSE, RMSE, R² pada data testing.

3. **Person 3 — Gradient Boosting Regressor**
   - Implementasi model `GradientBoostingRegressor` (`sklearn.ensemble`).
   - Konfigurasi hyperparameter (`loss='absolute_error'`, `n_estimators=100`, `learning_rate=0.1`, `max_depth=2`, `min_samples_split=4`, `min_samples_leaf=3`).
   - Evaluasi performa dan perbandingan terhadap baseline AdaBoost.

4. **Model Comparison & Streamlit Dashboard (Pink Aesthetic Theme)**
   - Audit keselarasan data test (344 baris identik).
   - Analisis residual, metrik perbandingan, dan summary JSON/CSV.
   - Dashboard analitik web interaktif Streamlit dengan tema pink elegan untuk visualisasi deret waktu, metrik, perbandingan model, dan prediksi interaktif (single input & batch CSV upload).

---

## 📊 Hasil Evaluasi Model

| Metrik Evaluasi | AdaBoost Regressor | Gradient Boosting Regressor | Model Terbaik |
| :--- | :---: | :---: | :---: |
| **MAE** | 6.1937 | **6.0659** | **Gradient Boosting** (lebih rendah) |
| **MAPE (%)** | **45.35%** | 47.00% | **AdaBoost** (lebih rendah) |
| **MSE** | 64.2927 | **41.4912** | **Gradient Boosting** (lebih rendah -35.5%) |
| **RMSE** | 8.0183 | **6.4414** | **Gradient Boosting** (lebih rendah -19.7%) |
| **R² Score** | -21.3287 | **-13.4098** | **Gradient Boosting** (penalti kuadratik lebih baik) |

> **Catatan**: Gradient Boosting Regressor ditetapkan sebagai model utama (default) pada dashboard aplikasi karena keunggulannya dalam meminimalkan outlier error secara signifikan (MSE dan RMSE jauh lebih rendah).

---

## 📁 Struktur Direktori Project

```text
TM11_K6_ML-Prak/
├── .streamlit/
│   └── config.toml                  # Konfigurasi tema Pink Aesthetic Streamlit
├── app.py                           # Aplikasi utama Streamlit Dashboard
├── requirements.txt                 # Dependensi Python
├── README.md                        # Dokumentasi lengkap proyek
├── dataset/
│   └── FINAL_USO.csv                # Dataset asli (1.718 baris, 81 kolom)
├── code/
│   ├── 01_preprocessing.py          # Script preprocessing Person 1
│   ├── 02_adaboost_person2.py       # Script AdaBoost Person 2
│   ├── 03_gradient_boosting_person3.py # Script Gradient Boosting Person 3
│   ├── 04_model_comparison.py       # Script validasi & perbandingan model
│   └── app_ui.py                    # Alternatif UI desktop (Tkinter)
├── utils/
│   ├── styles.py                    # CSS Injection styling tema pink & kartu metrik
│   ├── data_loader.py               # Utility pembacaan dataset & model
│   ├── validator.py                 # Validasi schema, tipe data, dan imputasi
│   └── visualizations.py            # Grafik interaktif Plotly
├── sample_data/
│   ├── sample_20_rows.csv           # Contoh data test untuk demonstrasi upload
│   └── sample_test_features.csv     # Contoh fitur tanpa target
├── docs/
│   └── reference/
│       ├── 11_STRUKTUR LAPORAN.pdf  # Panduan struktur laporan praktikum
│       └── 11_DRAFT LAPORAN PERSON 1.pdf # Draft laporan referensi Person 1
└── output/
    ├── person1/                     # Output dataset terbersihkan & metadata split
    ├── person2/                     # Model joblib, prediksi, dan metrik AdaBoost
    ├── person3/                     # Model joblib, prediksi, dan metrik Gradient Boosting
    └── model_comparison/            # Tabel perbandingan, visualisasi, & summary
```

---

## 🚀 Panduan Menjalankan Project

### 1. Menyiapkan Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Menjalankan Pipeline Pemodelan

Jalankan urutan script pemodelan jika ingin mereplikasi hasil eksperimen:

```powershell
# 1. Preprocessing (Person 1)
python code/01_preprocessing.py

# 2. Pelatihan & Evaluasi AdaBoost (Person 2)
python code/02_adaboost_person2.py

# 3. Pelatihan & Evaluasi Gradient Boosting (Person 3)
python code/03_gradient_boosting_person3.py

# 4. Validasi & Perbandingan Model
python code/04_model_comparison.py
```

### 3. Menjalankan Dashboard Streamlit

Untuk membuka dashboard web interaktif:

```powershell
streamlit run app.py
```

Aplikasi akan otomatis terbuka pada browser default di alamat: `http://localhost:8501`.

---

## 🌸 Fitur Utama Streamlit Dashboard

- **Header & Metric Cards:** Visualisasi skor metrik (MAE, MAPE, RMSE, R²) dengan styling pink modern.
- **Visualisasi Deret Waktu:** Plot perbandingan interaktif antara harga aktual dan harga prediksi kedua model.
- **Analisis Residual:** Distribusi galat/selisih prediksi untuk analisis performa model.
- **Prediksi Interaktif (Manual Input):** Slider & input numerik untuk memprediksi harga `USO_Close` secara real-time.
- **Batch Prediction (Upload CSV):** Validasi otomatis 74 kolom prediktor, imputasi otomatis missing values, dan tombol unduh hasil prediksi berupa CSV.
