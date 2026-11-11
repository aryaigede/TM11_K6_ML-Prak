"""
Aplikasi UI/UX Prediksi Harga Minyak (ETF USO)
Praktikum Machine Learning Modul 11 — Regresi (Kelompok K6)

Model Terintegrasi: Gradient Boosting Regressor (Model Terbaik)
Framework: Tkinter (Desktop GUI bawaan Python - Native & Standalone)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Pastikan cache matplotlib diatur ke folder temporer yang aman
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import joblib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "output" / "person3" / "gradient_boosting_model.joblib"
METADATA_PATH = ROOT_DIR / "output" / "person1" / "preprocessing_metadata.json"
TEST_DATA_PATH = ROOT_DIR / "output" / "person1" / "test_preprocessed.csv"
COMPARISON_PATH = ROOT_DIR / "output" / "model_comparison" / "model_comparison.csv"


class USOPredictorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Aplikasi Prediksi Harga Minyak USO — Machine Learning K6")
        self.root.geometry("1180x820")
        self.root.minsize(1000, 700)

        # Muat Model dan Data
        self.load_model_and_data()

        # Bangun Komponen Antarmuka (UI/UX)
        self.setup_styles()
        self.create_header()
        self.create_main_content()
        self.create_status_bar()

        # Inisialisasi dengan data pertama
        if hasattr(self, "date_combobox") and self.date_list:
            self.date_combobox.current(0)
            self.on_date_selected(None)

    def load_model_and_data(self) -> None:
        """Memuat model Gradient Boosting terlatih dan data testing untuk pengetesan interaktif."""
        try:
            self.model = joblib.load(MODEL_PATH)
            self.meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            self.features = self.meta["features"]
            self.test_df = pd.read_csv(TEST_DATA_PATH)
            self.date_list = self.test_df["Date"].tolist()

            # Muat metrik komparasi jika ada
            if COMPARISON_PATH.exists():
                self.comp_df = pd.read_csv(COMPARISON_PATH)
            else:
                self.comp_df = None
        except Exception as e:
            messagebox.showerror("Error Pemuatan Data", f"Gagal memuat model atau dataset:\n{e}")
            sys.exit(1)

    def setup_styles(self) -> None:
        """Konfigurasi gaya visual modern dan kontras nyaman."""
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Palet Warna
        self.PRIMARY_COLOR = "#1b3a4b"     # Navy gelap
        self.ACCENT_COLOR = "#006466"      # Teal tua
        self.BG_COLOR = "#f4f6f8"          # Abu-abu muda
        self.CARD_BG = "#ffffff"           # Putih kartu
        self.SUCCESS_COLOR = "#2d6a4f"     # Hijau status
        self.ERROR_COLOR = "#d90429"       # Merah status

        self.root.configure(bg=self.BG_COLOR)

        self.style.configure("Header.TFrame", background=self.PRIMARY_COLOR)
        self.style.configure("Card.TFrame", background=self.CARD_BG, relief="ridge", borderwidth=1)
        self.style.configure("TLabel", background=self.CARD_BG, font=("Helvetica", 10))
        self.style.configure("Header.TLabel", background=self.PRIMARY_COLOR, foreground="#ffffff", font=("Helvetica", 16, "bold"))
        self.style.configure("SubHeader.TLabel", background=self.PRIMARY_COLOR, foreground="#e0e1dd", font=("Helvetica", 10))
        self.style.configure("CardTitle.TLabel", background=self.CARD_BG, foreground=self.PRIMARY_COLOR, font=("Helvetica", 11, "bold"))
        self.style.configure("MetricVal.TLabel", background=self.CARD_BG, foreground=self.ACCENT_COLOR, font=("Helvetica", 14, "bold"))
        self.style.configure("PredictBtn.TButton", font=("Helvetica", 11, "bold"), padding=6)

    def create_header(self) -> None:
        """Membuat banner judul dan informasi kelompok praktikum."""
        header_frame = ttk.Frame(self.root, style="Header.TFrame", padding=(20, 15))
        header_frame.pack(fill="x", side="top")

        title_lbl = ttk.Label(
            header_frame,
            text="SISTEM PREDIKSI HARGA MINYAK ETF USO (MODUL 11 — REGRESI)",
            style="Header.TLabel"
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = ttk.Label(
            header_frame,
            text="Kelompok K6 (Genap) — Model Terpilih: Gradient Boosting Regressor (Unggul vs AdaBoost: MAE $6.07 | RMSE $6.44 | MSE 41.49)",
            style="SubHeader.TLabel"
        )
        subtitle_lbl.pack(anchor="w", pady=(3, 0))

    def create_main_content(self) -> None:
        """Membuat tata letak dua kolom: panel input di kiri dan panel visual/hasil di kanan."""
        main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        main_paned.pack(fill="both", expand=True, padx=12, pady=10)

        # === KOLOM KIRI: INPUT & KONTROL (UX INTUITIF) ===
        left_frame = ttk.Frame(main_paned, style="Card.TFrame", padding=15)
        main_paned.add(left_frame, weight=1)

        # 1. Pilihan Sampel Pengujian Deret Waktu
        sample_sec_lbl = ttk.Label(left_frame, text="1. PILIH TANGGAL DATA TESTING", style="CardTitle.TLabel")
        sample_sec_lbl.pack(anchor="w", pady=(0, 5))

        sample_desc = ttk.Label(
            left_frame,
            text="Pilih tanggal bursa dari 344 sampel pengujian kronologis (11 Agu 2017 – 31 Des 2018) untuk mengisi ke-74 variabel prediktor secara otomatis:",
            wraplength=380,
            foreground="#495057",
            font=("Helvetica", 9)
        )
        sample_desc.pack(anchor="w", pady=(0, 6))

        date_sel_box = ttk.Frame(left_frame, style="Card.TFrame")
        date_sel_box.pack(fill="x", pady=(0, 15))

        ttk.Label(date_sel_box, text="Tanggal Bursa:", font=("Helvetica", 9, "bold")).pack(side="left", padx=5)
        self.date_var = tk.StringVar()
        self.date_combobox = ttk.Combobox(date_sel_box, textvariable=self.date_var, values=self.date_list, state="readonly", width=16)
        self.date_combobox.pack(side="left", padx=5)
        self.date_combobox.bind("<<ComboboxSelected>>", self.on_date_selected)

        # 2. Indikator Pasar Utama (Key Features Editor)
        features_sec_lbl = ttk.Label(left_frame, text="2. INDIKATOR PASAR UTAMA (KUNCI)", style="CardTitle.TLabel")
        features_sec_lbl.pack(anchor="w", pady=(10, 5))

        feat_desc = ttk.Label(
            left_frame,
            text="Berikut 6 dari 74 fitur prediktor terpenting yang mempengaruhi prediksi. Anda dapat melihat nilainya atau menyesuaikannya:",
            wraplength=380,
            foreground="#495057",
            font=("Helvetica", 9)
        )
        feat_desc.pack(anchor="w", pady=(0, 8))

        # Kontainer Entri Nilai Kunci
        self.key_feature_entries: dict[str, tk.StringVar] = {}
        key_features = [
            ("Close", "Harga Emas / GLD Close ($)"),
            ("SP_close", "Indeks S&P 500 Close"),
            ("DJ_close", "Indeks Dow Jones Close"),
            ("OF_Price", "Harga Minyak Brent ($)"),
            ("SF_Price", "Harga Perak / Silver ($)"),
            ("USDI_Price", "US Dollar Index (USDX)"),
        ]

        key_grid = ttk.Frame(left_frame, style="Card.TFrame")
        key_grid.pack(fill="x", pady=(0, 15))

        for idx, (f_col, f_label) in enumerate(key_features):
            row_f = ttk.Frame(key_grid, style="Card.TFrame")
            row_f.pack(fill="x", pady=2)

            lbl = ttk.Label(row_f, text=f_label, width=28, anchor="w", font=("Helvetica", 9))
            lbl.pack(side="left")

            s_var = tk.StringVar(value="0.0")
            self.key_feature_entries[f_col] = s_var
            ent = ttk.Entry(row_f, textvariable=s_var, width=12, justify="right")
            ent.pack(side="right")

        # Tombol Aksi Prediksi
        predict_btn = ttk.Button(
            left_frame,
            text="JALANKAN PREDIKSI HARGA USO",
            style="PredictBtn.TButton",
            command=self.run_prediction
        )
        predict_btn.pack(fill="x", pady=(15, 10))

        # Kartu Hasil Prediksi Singkat
        self.result_card = ttk.Frame(left_frame, style="Card.TFrame", padding=10)
        self.result_card.pack(fill="x", pady=(5, 0))

        ttk.Label(self.result_card, text="HASIL INFERENSI MODEL:", style="CardTitle.TLabel").pack(anchor="w")

        self.res_pred_lbl = ttk.Label(self.result_card, text="Prediksi USO: -", font=("Helvetica", 14, "bold"), foreground=self.SUCCESS_COLOR)
        self.res_pred_lbl.pack(anchor="w", pady=(4, 1))

        self.res_act_lbl = ttk.Label(self.result_card, text="Harga Aktual: -", font=("Helvetica", 10))
        self.res_act_lbl.pack(anchor="w")

        self.res_err_lbl = ttk.Label(self.result_card, text="Error Absolut: - | Error %: -", font=("Helvetica", 9, "italic"), foreground="#495057")
        self.res_err_lbl.pack(anchor="w", pady=(2, 0))

        # === KOLOM KANAN: VISUALISASI DERET WAKTU & ANALISIS (UX INFORMATIF) ===
        right_frame = ttk.Frame(main_paned, style="Card.TFrame", padding=15)
        main_paned.add(right_frame, weight=2)

        chart_title = ttk.Label(right_frame, text="VISUALISASI PERFORMA PREDIKSI & DERET WAKTU PENGUJIAN", style="CardTitle.TLabel")
        chart_title.pack(anchor="w", pady=(0, 6))

        # Area Kanvas Matplotlib
        self.fig, (self.ax_ts, self.ax_err) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
        self.fig.patch.set_facecolor("#ffffff")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.draw_initial_chart()

    def create_status_bar(self) -> None:
        """Status bar bawah dengan informasi status model."""
        status_frame = ttk.Frame(self.root, relief="sunken", padding=(10, 4))
        status_frame.pack(fill="x", side="bottom")

        self.status_lbl = ttk.Label(
            status_frame,
            text="Status: Model Gradient Boosting Regressor siap digunakan. Pilih tanggal pengujian dan klik 'JALANKAN PREDIKSI'.",
            font=("Helvetica", 9),
            foreground="#2b2d42"
        )
        self.status_lbl.pack(side="left")

    def on_date_selected(self, event=None) -> None:
        """Handler saat tanggal pengujian dipilih: memuat 74 fitur dari data test asli."""
        sel_date = self.date_var.get()
        if not sel_date:
            return

        matching_rows = self.test_df[self.test_df["Date"] == sel_date]
        if matching_rows.empty:
            return

        row = matching_rows.iloc[0]
        self.current_full_row = row.copy()

        # Update entri fitur kunci
        for f_col, var in self.key_feature_entries.items():
            if f_col in row:
                var.set(f"{float(row[f_col]):.2f}")

        # Jalankan prediksi otomatis untuk kenyamanan UX
        self.run_prediction()

    def run_prediction(self) -> None:
        """Mengambil data fitur lengkap, memasukkan modifikasi input pengguna, dan memprediksi target."""
        try:
            sel_date = self.date_var.get()
            if not hasattr(self, "current_full_row") or self.current_full_row is None:
                if sel_date and sel_date in self.date_list:
                    self.current_full_row = self.test_df[self.test_df["Date"] == sel_date].iloc[0].copy()
                else:
                    self.current_full_row = self.test_df.iloc[0].copy()

            # Buat vektor fitur berukuran 74
            feature_vector = self.current_full_row[self.features].copy()

            # Terapkan modifikasi dari entri input
            for f_col, var in self.key_feature_entries.items():
                val_str = var.get().strip()
                feature_vector[f_col] = float(val_str)

            # Lakukan inferensi model
            input_df = pd.DataFrame([feature_vector], columns=self.features)
            predicted_val = float(self.model.predict(input_df)[0])
            actual_val = float(self.current_full_row["USO_Close"])

            abs_err = abs(actual_val - predicted_val)
            pct_err = (abs_err / actual_val) * 100.0

            # Perbarui label hasil
            self.res_pred_lbl.config(text=f"Prediksi USO : ${predicted_val:.2f}")
            self.res_act_lbl.config(text=f"Harga Aktual  : ${actual_val:.2f} (Tanggal: {sel_date})")
            self.res_err_lbl.config(text=f"Error Absolut: ${abs_err:.2f}  |  Error Relatif: {pct_err:.2f}%")

            # Perbarui status bar
            self.status_lbl.config(
                text=f"Sukses: Prediksi ${predicted_val:.2f} vs Aktual ${actual_val:.2f} (Error: {pct_err:.2f}%) untuk tanggal {sel_date}"
            )

            # Perbarui grafik dengan titik prediksi saat ini
            self.update_chart(sel_date, actual_val, predicted_val)

        except ValueError as ve:
            messagebox.showwarning("Input Tidak Valid", f"Pastikan seluruh nilai input berupa angka desimal valid:\n{ve}")
        except Exception as e:
            messagebox.showerror("Kesalahan Prediksi", f"Terjadi kesalahan saat memproses prediksi:\n{e}")

    def draw_initial_chart(self) -> None:
        """Menggambar kurva deret waktu pengujian awal."""
        dates = pd.to_datetime(self.test_df["Date"])
        actual_prices = self.test_df["USO_Close"].values

        # Jalankan prediksi batch untuk seluruh data uji guna menggambar garis model
        all_X = self.test_df[self.features]
        all_pred = self.model.predict(all_X)
        self.all_pred = all_pred
        self.dates_parsed = dates

        self.ax_ts.clear()
        self.ax_err.clear()

        # Plot Kurva Harga
        self.ax_ts.plot(dates, actual_prices, label="Aktual (USO_Close)", color="#1f77b4", linewidth=1.8)
        self.ax_ts.plot(dates, all_pred, label="Prediksi Gradient Boosting", color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.9)
        self.ax_ts.set_ylabel("Harga ($)", fontsize=9)
        self.ax_ts.set_title("Deret Waktu Data Testing (11 Agu 2017 – 31 Des 2018)", fontsize=10, fontweight="bold")
        self.ax_ts.legend(loc="upper left", fontsize=8, frameon=True)
        self.ax_ts.grid(True, linestyle=":", alpha=0.6)

        # Plot Error Harian
        diffs = np.abs(actual_prices - all_pred)
        self.ax_err.bar(dates, diffs, color="#9467bd", width=1.0, alpha=0.7, label="|Error| ($)")
        self.ax_err.set_ylabel("|Error| ($)", fontsize=9)
        self.ax_err.set_xlabel("Tanggal Pengujian", fontsize=9)
        self.ax_err.grid(True, linestyle=":", alpha=0.6)
        self.ax_err.legend(loc="upper left", fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

    def update_chart(self, current_date_str: str, actual_val: float, pred_val: float) -> None:
        """Menyorot sampel yang sedang diprediksi pada grafik."""
        self.draw_initial_chart()

        # Sorot titik yang sedang dipilih dengan marker merah menyala
        dt = pd.to_datetime(current_date_str)
        self.ax_ts.scatter([dt], [actual_val], color="crimson", s=70, zorder=5, label="Titik Terpilih (Aktual)")
        self.ax_ts.scatter([dt], [pred_val], color="darkorange", marker="X", s=70, zorder=5, label="Titik Terpilih (Prediksi)")
        self.ax_ts.legend(loc="upper left", fontsize=8, frameon=True)

        self.canvas.draw()


def main() -> None:
    # Mode pengujian otomatis tanpa blocking UI
    if "--test" in sys.argv:
        print("Menjalankan validasi inisialisasi UI headless...")
        root = tk.Tk()
        root.withdraw()
        app = USOPredictorApp(root)
        app.run_prediction()
        print("Validasi antarmuka Tkinter berhasil 100%!")
        root.destroy()
        return

    root = tk.Tk()
    app = USOPredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

