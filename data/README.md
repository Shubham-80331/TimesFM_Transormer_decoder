# Dataset

`co2_weekly.csv` is the weekly atmospheric CO2 series bundled with the Python `statsmodels` dataset collection. Missing observations are linearly interpolated and remaining edge gaps are forward/back-filled in the preparation script.

The dataset is used as a real-world forecasting benchmark for the extended project. It is kept separate from the synthetic sine-wave examples in `datas/` that came with the original simplified TimesFM implementation.
