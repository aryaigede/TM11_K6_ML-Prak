"""
Interactive Visualizations Module using Plotly
Styled with the Refined Pink Aesthetic Design Palette
Completely avoids duplicate layout arguments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.styles import COLORS


# Base Layout Dictionary (tidak langsung di-unpack dengan keyword berulang)
BASE_LAYOUT = dict(
    font=dict(family="Plus Jakarta Sans, sans-serif", color=COLORS["text_primary"]),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FFFFFF",
    margin=dict(l=40, r=30, t=50, b=40),
    hoverlabel=dict(
        bgcolor="#FFFFFF",
        font_size=12,
        font_family="Plus Jakarta Sans, sans-serif",
        bordercolor=COLORS["border"],
    ),
    autosize=True,
)


def apply_layout_safely(fig: go.Figure, **custom_kwargs) -> None:
    """Menggabungkan layout default dengan custom overrides tanpa duplikasi keyword."""
    merged = dict(BASE_LAYOUT)
    merged.update(custom_kwargs)
    fig.update_layout(**merged)


def create_actual_vs_predicted_chart(pred_df: pd.DataFrame) -> go.Figure:
    """Grafik interaktif deret waktu: Aktual vs Prediksi Gradient Boosting."""
    df = pred_df.copy()
    if "Date" in df.columns:
        df["Date_dt"] = pd.to_datetime(df["Date"])
        x_col = "Date_dt"
        x_title = "Tanggal Bursa"
    else:
        df["Index"] = np.arange(len(df))
        x_col = "Index"
        x_title = "Sampel Index"

    fig = go.Figure()

    # Garis Harga Aktual
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df["Actual"],
            mode="lines",
            name="Harga Aktual (USO_Close)",
            line=dict(color=COLORS["text_primary"], width=2.4),
            hovertemplate="<b>Aktual:</b> $%{y:.2f}<br>Tanggal: %{x|%d %b %Y}<extra></extra>",
        )
    )

    # Garis Prediksi Gradient Boosting
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df["Predicted"],
            mode="lines",
            name="Prediksi Gradient Boosting",
            line=dict(color=COLORS["primary_dark"], width=2.2, dash="dash"),
            hovertemplate="<b>Prediksi:</b> $%{y:.2f}<extra></extra>",
        )
    )

    apply_layout_safely(
        fig,
        title=dict(
            text="<b>Pergerakan Nilai Aktual vs Prediksi Model (Data Testing 2017–2018)</b>",
            font=dict(size=15, color=COLORS["text_primary"]),
            x=0.02,
        ),
        xaxis=dict(
            title=x_title,
            showgrid=True,
            gridcolor=COLORS["grid"],
            zeroline=False,
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        yaxis=dict(
            title="Harga USO Close ($)",
            showgrid=True,
            gridcolor=COLORS["grid"],
            zeroline=False,
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=COLORS["border"],
            borderwidth=1,
            font=dict(color=COLORS["text_primary"]),
        ),
        height=420,
    )

    return fig


def create_error_barchart(pred_df: pd.DataFrame) -> go.Figure:
    """Grafik batang error absolut harian (|Aktual - Prediksi|)."""
    df = pred_df.copy()
    x_col = pd.to_datetime(df["Date"]) if "Date" in df.columns else np.arange(len(df))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_col,
            y=df["Absolute_Error"],
            marker_color=COLORS["primary"],
            opacity=0.85,
            name="|Error| ($)",
            hovertemplate="<b>Error Absolut:</b> $%{y:.2f}<extra></extra>",
        )
    )

    apply_layout_safely(
        fig,
        title=dict(
            text="<b>Error Absolut Harian (|Aktual − Prediksi|)</b>",
            font=dict(size=14, color=COLORS["text_primary"]),
            x=0.02,
        ),
        xaxis=dict(
            title="Tanggal",
            showgrid=False,
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        yaxis=dict(
            title="Error ($)",
            gridcolor=COLORS["grid"],
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        height=240,
        margin=dict(l=40, r=30, t=40, b=30),  # Override margin khusus tanpa error
    )
    return fig


def create_comparison_charts(comp_df: pd.DataFrame) -> go.Figure:
    """Grafik perbandingan terpadu AdaBoost vs Gradient Boosting untuk MAE, RMSE, dan MAPE."""
    metrics_subset = comp_df[comp_df["Metric"].isin(["MAE", "RMSE", "MAPE"])].copy()

    fig = go.Figure()

    # Bar AdaBoost
    fig.add_trace(
        go.Bar(
            name="AdaBoost Regressor",
            x=metrics_subset["Metric"],
            y=metrics_subset["AdaBoost_Regressor"],
            marker_color=COLORS["adaboost"],
            marker_line=dict(color=COLORS["primary_dark"], width=1.2),
            text=[f"{v:.2f}" for v in metrics_subset["AdaBoost_Regressor"]],
            textposition="auto",
            textfont=dict(color=COLORS["text_primary"], weight=700),
        )
    )

    # Bar Gradient Boosting
    fig.add_trace(
        go.Bar(
            name="Gradient Boosting Regressor",
            x=metrics_subset["Metric"],
            y=metrics_subset["Gradient_Boosting_Regressor"],
            marker_color=COLORS["primary_dark"],
            text=[f"{v:.2f}" for v in metrics_subset["Gradient_Boosting_Regressor"]],
            textposition="auto",
            textfont=dict(color="#FFFFFF", weight=700),
        )
    )

    apply_layout_safely(
        fig,
        barmode="group",
        title=dict(
            text="<b>Perbandingan Metrik Evaluasi Utama: MAE, RMSE ($), dan MAPE (%)</b>",
            font=dict(size=14, color=COLORS["text_primary"]),
            x=0.02,
        ),
        xaxis=dict(
            title="Metrik Evaluasi",
            showgrid=False,
            tickfont=dict(color=COLORS["text_primary"], size=11),
        ),
        yaxis=dict(
            title="Nilai Metrik",
            gridcolor=COLORS["grid"],
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.01,
            font=dict(color=COLORS["text_primary"]),
        ),
        height=380,
    )

    return fig


def create_prediction_distribution_chart(predictions: np.ndarray | pd.Series) -> go.Figure:
    """Distribusi harga hasil prediksi baru."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=predictions,
            nbinsx=20,
            marker_color=COLORS["primary"],
            marker_line=dict(color=COLORS["primary_dark"], width=1),
            opacity=0.85,
            hovertemplate="Rentang: $%{x}<br>Jumlah: %{y}<extra></extra>",
        )
    )

    mean_val = float(np.mean(predictions))
    fig.add_vline(
        x=mean_val,
        line_width=2,
        line_dash="dash",
        line_color=COLORS["text_primary"],
        annotation_text=f"Rata-rata (${mean_val:.2f})",
        annotation_position="top right",
        annotation_font=dict(color=COLORS["text_primary"], size=11),
    )

    apply_layout_safely(
        fig,
        title=dict(
            text="<b>Distribusi Sebaran Harga Prediksi USO ($)</b>",
            font=dict(size=14, color=COLORS["text_primary"]),
            x=0.02,
        ),
        xaxis=dict(
            title="Prediksi Harga USO ($)",
            gridcolor=COLORS["grid"],
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        yaxis=dict(
            title="Frekuensi",
            gridcolor=COLORS["grid"],
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        height=320,
        margin=dict(l=40, r=30, t=40, b=40),
    )

    return fig


def create_new_predictions_timeline(df_pred: pd.DataFrame) -> go.Figure:
    """Timeline kurva hasil prediksi baru."""
    fig = go.Figure()
    x_axis = df_pred["Date"] if "Date" in df_pred.columns else np.arange(1, len(df_pred) + 1)
    x_title = "Tanggal" if "Date" in df_pred.columns else "Nomor Baris Sampel"

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=df_pred["Predicted_USO_Close"],
            mode="lines+markers",
            marker=dict(size=6, color=COLORS["primary_dark"]),
            line=dict(color=COLORS["primary"], width=2.5),
            name="Prediksi USO Close",
            hovertemplate="<b>Prediksi:</b> $%{y:.2f}<extra></extra>",
        )
    )

    apply_layout_safely(
        fig,
        title=dict(
            text="<b>Tren Nilai Prediksi USO Close Sepanjang Sampel</b>",
            font=dict(size=14, color=COLORS["text_primary"]),
            x=0.02,
        ),
        xaxis=dict(
            title=x_title,
            gridcolor=COLORS["grid"],
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        yaxis=dict(
            title="Prediksi Harga USO ($)",
            gridcolor=COLORS["grid"],
            tickfont=dict(color=COLORS["text_secondary"]),
        ),
        height=340,
        margin=dict(l=40, r=30, t=40, b=40),
    )

    return fig
