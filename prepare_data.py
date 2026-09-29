"""Prepare the bundled CO2 series into chronological train/validation/test CSV files."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
src = ROOT / "data" / "co2_weekly.csv"
out = ROOT / "data" / "splits"
out.mkdir(exist_ok=True)

df = pd.read_csv(src, parse_dates=["Date"])
df["CO2"] = df["CO2"].interpolate().ffill().bfill()

n = len(df)
train_end = int(n * 0.65)
val_end = int(n * 0.90)

# Chronological split: no future observations are used to construct training data.
df.iloc[:train_end].to_csv(out / "train.csv", index=False)
df.iloc[train_end:val_end].to_csv(out / "val.csv", index=False)
df.iloc[val_end:].to_csv(out / "test.csv", index=False)

print(f"Total: {n} | train: {train_end} | val: {val_end-train_end} | test: {n-val_end}")
