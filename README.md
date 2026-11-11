# TM11 K6 ML Praktikum - Bagian person 1

Bagian ini menyiapkan dataset bersama untuk **AdaBoost Regressor** dan
**Gradient Boosting Regressor**.

## Dataset dan target

- Dataset: `dataset/FINAL_USO.csv`
- Target: `USO_Close` (harga penutupan ETF USO)
- Prediktor: seluruh kolom numerik selain `Date` dan seluruh kolom berawalan
  `USO_`. Kolom `USO_*` dikeluarkan karena merupakan informasi target yang sama
  atau informasi harga USO pada hari yang sama sehingga dapat menyebabkan data
  leakage.
- Split: kronologis 80% training dan 20% testing. Pembagian ini sama untuk
  kedua model dan tidak menggunakan pengacakan karena data berbentuk deret
  waktu.

Sumber online yang cocok dengan nama file dan struktur 81 kolom adalah Kaggle
["Gold price prediction"](https://www.kaggle.com/datasets/suryanachoudhary/gold-price-prediction).
Berkas lokal `FINAL_USO.csv` tetap menjadi salinan yang digunakan dalam
praktikum. Tanggal akses perlu diisi saat laporan final disusun.

## Menjalankan

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python code\01_preprocessing_person1.py
```

## Output

Folder `output/person1/` berisi:

- `dataset_cleaned.csv`: dataset terurut berdasarkan tanggal dan tanpa duplikasi.
- `train_preprocessed.csv` dan `test_preprocessed.csv`: input bersama untuk
  dua model, berisi `Date`, fitur, dan `USO_Close`.
- `data_quality_report.csv`: tipe data, missing value, dan jumlah nilai unik.
- `preprocessing_metadata.json`: definisi target, fitur, rentang tanggal, dan
  rincian split.
- `target_train_test_split.png`: visualisasi target dan batas split.

Imputasi median tetap disediakan sebagai perlindungan apabila data sumber
berubah. Nilai median dihitung hanya dari fitur training lalu diterapkan ke
testing, sehingga tidak terjadi data leakage.
