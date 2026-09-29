"""Interactive forecasting dashboard — TimesFM-style CO₂ forecast."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

from timesfm.timesfm import TimesFM, TimesFMConfig

ROOT = Path(__file__).resolve().parent

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Time-Series Transformer Forecasting", layout="wide")
st.title("Time-Series Transformer Forecasting")
st.caption("A compact TimesFM-style decoder evaluated on a real-world CO₂ time series.")

# ── Load model ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    config = TimesFMConfig()
    model = TimesFM(config).to(config.device)
    weights = ROOT / "weights" / "best" / "timesfm.pth"
    if not weights.exists():
        return None, config
    model.load_state_dict(
        torch.load(weights, map_location=config.device, weights_only=True)
    )
    model.eval()
    return model, config

# ── Load data ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    full = pd.read_csv(ROOT / "data" / "co2_weekly.csv", parse_dates=["Date"])
    full["CO2"] = full["CO2"].interpolate().ffill().bfill()
    test = pd.read_csv(ROOT / "data" / "splits" / "test.csv", parse_dates=["Date"])
    test["CO2"] = test["CO2"].interpolate().ffill().bfill()
    return full, test

# ── Forecast helper ─────────────────────────────────────────────────────────────
def run_forecast(model, history: np.ndarray, horizon: int, patch_len: int) -> np.ndarray:
    x = history.astype(np.float32).copy()
    out = []
    for _ in range(math.ceil(horizon / patch_len)):
        y = model.infer(x)
        out.extend(y.tolist())
        x = np.concatenate([x, y])[-len(history):]
    return np.asarray(out[:horizon])

# ── Metrics helper ──────────────────────────────────────────────────────────────
def compute_metrics(actual: np.ndarray, predicted: np.ndarray):
    mae  = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mape = float(np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1e-8))) * 100)
    return mae, rmse, mape

# ── Boot ────────────────────────────────────────────────────────────────────────
model, config = load_model()
if model is None:
    st.error("Trained weights not found at weights/best/timesfm.pth — run the training script first.")
    st.stop()

full_df, test_df = load_data()

# ── Sidebar controls ─────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
context = st.sidebar.slider(
    "Context length",
    min_value=config.patch_len, max_value=512, value=160, step=config.patch_len,
)
horizon = st.sidebar.slider(
    "Forecast horizon",
    min_value=config.patch_len, max_value=256, value=64, step=config.patch_len,
)

st.markdown(f"**Context Length:** {context} &nbsp;&nbsp;|&nbsp;&nbsp; **Forecast Horizon:** {horizon}")
st.divider()

# ── Compute ──────────────────────────────────────────────────────────────────────
test_series  = test_df["CO2"].to_numpy(dtype=np.float32)
full_series  = full_df["CO2"].to_numpy(dtype=np.float32)
cutoff       = len(full_series) - len(test_series)
context_vals = full_series[max(0, cutoff - context): cutoff]

pred  = run_forecast(model, context_vals, horizon, config.patch_len)
naive = np.repeat(context_vals[-1], horizon)
actual = test_series[:horizon]

mae,  rmse,  mape   = compute_metrics(actual, pred)
n_mae, n_rmse, n_mape = compute_metrics(actual, naive)

# ── Metrics row ──────────────────────────────────────────────────────────────────
st.subheader("Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("MAE",  f"{mae:.2f}",   delta=f"{n_mae  - mae :.2f} vs baseline")
col2.metric("RMSE", f"{rmse:.2f}",  delta=f"{n_rmse - rmse:.2f} vs baseline")
col3.metric("MAPE", f"{mape:.2f}%", delta=f"{n_mape - mape:.2f}% vs baseline")
st.divider()

# ── Chart ────────────────────────────────────────────────────────────────────────
st.subheader("Forecast vs Reality")

history_dates  = full_df["Date"].iloc[max(0, cutoff - context): cutoff]
test_dates     = test_df["Date"].iloc[:horizon]
forecast_start = history_dates.iloc[-1]

fig, ax = plt.subplots(figsize=(13, 4))
fig.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")

ax.plot(history_dates, context_vals, color="#4fc3f7",
        linewidth=1.6, label="Historical CO₂")
ax.plot(test_dates, actual, color="#81c784",
        linewidth=1.6, label="Actual (test)")
ax.plot(test_dates, pred, color="#ff8a65",
        linewidth=1.8, linestyle="--", label="Transformer forecast")

ax.axvline(x=forecast_start, color="#ffffff", linewidth=1.0,
           linestyle=":", alpha=0.6)
ymin, ymax = ax.get_ylim()
ax.text(forecast_start, ymax, "  forecast start",
        color="#cccccc", fontsize=8, va="top", alpha=0.8)

ax.tick_params(colors="#aaaaaa")
for sp in ax.spines.values():
    sp.set_edgecolor("#333333")
ax.set_ylabel("CO₂ (ppm)", color="#aaaaaa")
ax.legend(facecolor="#1e1e1e", edgecolor="#444444",
          labelcolor="#dddddd", fontsize=9, loc="upper left")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ── Footer ────────────────────────────────────────────────────────────────────────
st.divider()
fc1, fc2, fc3 = st.columns(3)
fc1.markdown("**Model:** TimesFM-style Transformer")
fc2.markdown("**Baseline:** Persistence (last value)")
fc3.markdown(f"**Forecast horizon:** {horizon} steps")
