"""
Train ConvLSTM on real IMD data (downsampled for CPU feasibility).
Produces: checkpoints/climate_twin_convlstm_final.pth
         checkpoints/climate_twin_convlstm_temp.pth
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.models.pytorch_convlstm import SpatioTemporalConvLSTM

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SEQ_LEN   = 10
EPOCHS    = 25
LR        = 1e-3
BATCH     = 4
DOWNSAMPLE = 4   # Spatial downsampling factor for CPU speed


class IMDDataset(Dataset):
    def __init__(self, data_np, seq_len=SEQ_LEN):
        # data_np: (T, lat, lon)  float32, NaN->0
        self.data = np.nan_to_num(data_np.astype(np.float32), nan=0.0)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]          # (seq, lat, lon)
        y = self.data[idx + self.seq_len]               # (lat, lon)
        x = np.expand_dims(x, 1)                        # (seq, 1, lat, lon)
        y = np.expand_dims(y, 0)                        # (1, lat, lon)
        return torch.tensor(x), torch.tensor(y)


def train(model, loader, epochs, lr, tag, device):
    model.to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit  = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=4)
    best_loss = float('inf')
    for ep in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred, _ = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        avg = total / len(loader)
        sched.step(avg)
        rmse = np.sqrt(avg)
        print(f"[{tag}] Epoch {ep+1:02d}/{epochs}  RMSE={rmse:.4f}  LR={opt.param_groups[0]['lr']:.6f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'{tag}.pth'))
    print(f"[{tag}] Training complete. Best checkpoint saved.")


def load_and_prep(nc_path, var, ds_factor):
    try:
        ds = xr.open_dataset(nc_path, engine='netcdf4')
        arr = ds[var].values.astype(np.float32)      # (T, lat, lon)
        # Downsample spatially
        arr = arr[:, ::ds_factor, ::ds_factor]
        # Train/val split: train on first 80%
        n_train = int(len(arr) * 0.8)
        return arr[:n_train], arr[n_train:]
    except Exception as e:
        print(f"Could not load {nc_path}: {e}")
        return None, None


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device.upper()}")

    # --- RAINFALL ---
    rain_path = os.path.join(DATA_DIR, 'IMD_Gridded_Rainfall_0.25_Real_v2.nc')
    rain_train, rain_val = load_and_prep(rain_path, 'rainfall', DOWNSAMPLE)
    if rain_train is not None:
        # log1p scale
        rain_train = np.log1p(np.maximum(0, rain_train))
        ds_rain = IMDDataset(rain_train, SEQ_LEN)
        loader_rain = DataLoader(ds_rain, batch_size=BATCH, shuffle=True, num_workers=0)
        h, w = rain_train.shape[1], rain_train.shape[2]
        model_rain = SpatioTemporalConvLSTM(1, [64, 32], (3, 3), 2)
        print(f"Rainfall grid: {h}x{w},  samples: {len(ds_rain)}")
        train(model_rain, loader_rain, EPOCHS, LR, 'climate_twin_convlstm_final', device)

    # --- MAX TEMP ---
    temp_path = os.path.join(DATA_DIR, 'IMD_Gridded_MaxTemp_1.0_Real.nc')
    temp_train, temp_val = load_and_prep(temp_path, 'max_temp', max(1, DOWNSAMPLE // 2))
    if temp_train is not None:
        # Normalise
        temp_train_n = (temp_train - 30.0) / 10.0
        ds_temp = IMDDataset(temp_train_n, SEQ_LEN)
        loader_temp = DataLoader(ds_temp, batch_size=BATCH, shuffle=True, num_workers=0)
        h, w = temp_train_n.shape[1], temp_train_n.shape[2]
        model_temp = SpatioTemporalConvLSTM(1, [64, 32], (3, 3), 2)
        print(f"MaxTemp grid: {h}x{w},  samples: {len(ds_temp)}")
        train(model_temp, loader_temp, EPOCHS, LR, 'climate_twin_convlstm_temp', device)

    print("ALL TRAINING COMPLETE.")
