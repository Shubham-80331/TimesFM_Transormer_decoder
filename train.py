"""Train the compact TimesFM-style model with a chronological validation split."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.dataset import SeqDataset
from timesfm.timesfm import TimesFM, TimesFMConfig


def make_windows(values, patch_len, seq_len):
    width = patch_len * (seq_len + 1)
    return np.asarray([values[i:i+width] for i in range(len(values) - width + 1)], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/splits/train.csv')
    ap.add_argument('--val_data', default='data/splits/val.csv')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2.5e-4)
    ap.add_argument('--patience', type=int, default=5)
    args = ap.parse_args()

    config = TimesFMConfig()
    patch_len, seq_len = config.patch_len, config.seq_len

    train_values = pd.read_csv(args.data)['CO2'].to_numpy(dtype=np.float32)
    val_values = pd.read_csv(args.val_data)['CO2'].to_numpy(dtype=np.float32)
    train_windows = make_windows(train_values, patch_len, seq_len)
    val_windows = make_windows(val_values, patch_len, seq_len)

    train_loader = DataLoader(SeqDataset(train_windows, patch_len), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(SeqDataset(val_windows, patch_len), batch_size=args.batch, shuffle=False)

    model = TimesFM(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float('inf')
    stale = 0
    out_dir = Path('weights/best')
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, target in train_loader:
            x, target = x.to(config.device), target.to(config.device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, target in val_loader:
                x, target = x.to(config.device), target.to(config.device)
                val_losses.append(F.mse_loss(model(x), target).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f'Epoch {epoch:03d} | train MSE {train_loss:.6f} | val MSE {val_loss:.6f}')

        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            torch.save(model.state_dict(), out_dir / 'timesfm.pth')
        else:
            stale += 1
            if stale >= args.patience:
                print('Early stopping.')
                break

    print(f'Best validation MSE: {best_val:.6f}')


if __name__ == '__main__':
    main()
