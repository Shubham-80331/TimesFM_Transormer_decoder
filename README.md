# Time-Series Transformer Forecasting

A compact, research-oriented extension of a simplified TimesFM implementation for **real-world time-series forecasting**.

**Author:** Shubham · [shubhams80331@gmail.com](mailto:shubhams80331@gmail.com) · [github.com/Shubham-80331](https://github.com/Shubham-80331)

---

## What this project does

The base repository is a small educational implementation inspired by TimesFM. This version turns it into a more complete forecasting experiment by adding:

- A real-world weekly **atmospheric CO₂** forecasting dataset.
- Chronological **65/25/10 train-validation-test splitting** to avoid future-data leakage.
- Validation-based **early stopping** and best-checkpoint selection.
- Rolling **multi-step forecasting** rather than evaluating only a single forecast patch.
- **MAE, RMSE and MAPE** evaluation metrics with a persistence baseline for comparison.
- A **Streamlit** dashboard for interactive forecasting (see below).
- Configurable context length and forecast horizon.

## Model

The model follows the compact TimesFM-style design:

1. Convert a 1D time series into fixed-length patches.
2. Project patches into an embedding space.
3. Add learnable positional embeddings.
4. Process the patch sequence with a **causal Transformer decoder**.
5. Project decoder outputs back into time-series patches.
6. Apply reversible instance-style normalization during inference.

The implementation is intentionally small (2 decoder layers, 2 attention heads) so the architecture can be studied and modified on a student workstation.

## Dataset

`data/co2_weekly.csv` contains the weekly atmospheric CO₂ series bundled with the Python `statsmodels` dataset collection. The series is interpolated for missing observations before use.

The synthetic sine-wave datasets from the base repository remain in `datas/` for controlled experiments.

## Reproducible workflow

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare chronological splits

```bash
python prepare_data.py
```

This creates:

- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

### 3. Train

```bash
python train.py --epochs 40
```

The best checkpoint is saved to `weights/best/timesfm.pth`.

### 4. Evaluate

```bash
python evaluate.py --weights weights/best/timesfm.pth
```

Prints MAE, RMSE and MAPE for both the Transformer model and the persistence baseline.

### 5. Run the dashboard

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

## Streamlit Dashboard

The interactive dashboard provides a self-contained view of model performance:

- **Sidebar sliders** to adjust context length and forecast horizon in real time.
- **Metrics panel** — MAE, RMSE and MAPE displayed side by side, each with a delta showing how much the Transformer improves over the persistence baseline.
- **Three-line chart** rendered with Matplotlib:
  - 🔵 *Historical CO₂* — solid line of the context window fed to the model.
  - 🟢 *Actual (test)* — real ground-truth values from the held-out test set.
  - 🟠 *Transformer forecast* — dashed line of the model's recursive prediction.
  - A dotted vertical marker labelled **"forecast start"** separates history from prediction.
- **Footer row** — model name, baseline type, and active forecast horizon at a glance.

## Research basis

The starting implementation is inspired by:

- **TimesFM:** *A decoder-only foundation model for time-series forecasting*.
- **RevIN:** *Reversible Instance Normalization for Accurate Time-Series Forecasting Against Distribution Shift*.

This repository is **not the official TimesFM implementation** and does not claim to reproduce Google's full-scale pretrained foundation model. The goal is to understand the architecture and evaluate a compact Transformer-based forecasting system under a reproducible experimental setup.

## Extensions by Shubham

Built on top of the original educational skeleton, this project adds:

- Real-world CO₂ dataset replacing the toy sine-wave data.
- Chronological train/val/test split to prevent future-data leakage.
- Early stopping with best-checkpoint saving.
- Rolling multi-step recursive forecasting and quantitative evaluation (MAE / RMSE / MAPE).
- Persistence baseline for comparison.
- Full Streamlit dashboard with live metrics, three-line comparative chart, and forecast-start marker.

