"""Evaluate the trained model with rolling multi-step forecasts and simple baselines."""
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from timesfm.timesfm import TimesFM, TimesFMConfig


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    denom = np.maximum(np.abs(actual), 1e-8)
    mape = np.mean(np.abs((actual - predicted) / denom)) * 100
    return mae, rmse, mape


def forecast(model, history, horizon, patch_len):
    x = np.asarray(history, dtype=np.float32).copy()
    out = []
    for _ in range(math.ceil(horizon / patch_len)):
        y = model.infer(x)
        out.extend(y.tolist())
        x = np.concatenate([x, y])[-len(history):]
    return np.asarray(out[:horizon])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/splits/test.csv')
    ap.add_argument('--weights', default='weights/24/timesfm.pth')
    ap.add_argument('--context', type=int, default=128)
    ap.add_argument('--horizon', type=int, default=64)
    ap.add_argument('--step', type=int, default=64)
    ap.add_argument('--plot', default='experiments/test_forecast.png')
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    series = df['CO2'].to_numpy(dtype=np.float32)

    config = TimesFMConfig()
    model = TimesFM(config).to(config.device)
    model.load_state_dict(torch.load(args.weights, map_location=config.device, weights_only=True))
    model.eval()

    all_actual, all_pred, all_naive = [], [], []
    for end in range(args.context, len(series) - args.horizon + 1, args.step):
        history = series[end-args.context:end]
        actual = series[end:end+args.horizon]
        pred = forecast(model, history, args.horizon, config.patch_len)
        naive = np.repeat(history[-1], args.horizon)
        all_actual.extend(actual)
        all_pred.extend(pred)
        all_naive.extend(naive)

    mae, rmse, mape = metrics(all_actual, all_pred)
    n_mae, n_rmse, n_mape = metrics(all_actual, all_naive)
    print('\nTimesFM-style model')
    print(f'MAE:  {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%')
    print('\nPersistence baseline')
    print(f'MAE:  {n_mae:.4f}\nRMSE: {n_rmse:.4f}\nMAPE: {n_mape:.2f}%')

    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 4))
    plt.plot(all_actual, label='Ground truth')
    plt.plot(all_pred, label='Transformer forecast')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot, dpi=160)
    plt.close()


if __name__ == '__main__':
    main()
