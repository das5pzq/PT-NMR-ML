import gc
import json
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

MAX_GRAD_NORM = 1.0
SIGNAL_LEN = 512


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NMRSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PolarizationCNN(nn.Module):
    """1D CNN on a raw lineshape. Pooling keeps left/right peak position."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(32),
        )
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32, 128),
            nn.GELU(),
        )
        self.p_head = nn.Linear(128, 1)
        self.q_head = nn.Linear(128, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.trunk(self.features(x))
        return torch.cat([self.p_head(h), self.q_head(h)], dim=-1)


def _split_losses(criterion, y_hat, y):
    loss_p = criterion(y_hat[:, 0:1], y[:, 0:1])
    loss_q = criterion(y_hat[:, 1:2], y[:, 1:2])
    return loss_p + loss_q


def _run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_sum = 0.0
    mae_sum = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_hat = model(x)
        loss = _split_losses(criterion, y_hat, y)
        mae = F.l1_loss(y_hat, y)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        loss_sum += loss.item()
        mae_sum += mae.item()
        n_batches += 1

    n = max(n_batches, 1)
    return loss_sum / n, mae_sum / n


def standardize_traces(signals):
    """Center and scale each raw spectrum. The baseline curve stays in the trace."""
    signals = np.asarray(signals, dtype=np.float32)
    centered = signals - signals.mean(axis=1, keepdims=True)
    scale = np.maximum(centered.std(axis=1, keepdims=True), 1e-8)
    return (centered / scale).astype(np.float32)


def train_model(
    X_train, y_train, X_val, y_val,
    model_dir, performance_dir, version,
    learning_rate=3e-4, max_epochs=40,
    batch_size=256, weight_decay=1e-5,
    device=None,
):
    if device is None:
        device = _device()
    pin = device.type == "cuda"

    train_loader = DataLoader(
        NMRSequenceDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True, pin_memory=pin,
    )
    val_loader = DataLoader(
        NMRSequenceDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False, pin_memory=pin,
    )

    model = PolarizationCNN().to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}", flush=True)
    print(f"Device: {device}", flush=True)

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7,
    )

    ckpt_path = os.path.join(model_dir, "best_model_checkpoint.pth")
    best_pth_path = os.path.join(model_dir, "best_model.pth")
    start_epoch = 0
    best_val_mae = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}

    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_mae = ckpt.get("best_val_mae", float("inf"))
        if ckpt.get("history"):
            history = ckpt["history"]
        print(f"Resuming from {ckpt_path} at epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, max_epochs):
        train_loss, train_mae = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_mae = _run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        print(
            f"epoch {epoch + 1:03d}/{max_epochs} | train {train_loss:.6f} | val {val_loss:.6f} | "
            f"train_mae {train_mae:.6f} | val_mae {val_mae:.6f} | lr {lr:.2e}",
            flush=True,
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(state, best_pth_path)
            torch.save(
                {
                    "model_state_dict": state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_mae": best_val_mae,
                    "learning_rate": learning_rate,
                    "history": history,
                },
                ckpt_path,
            )

    loss_path = f"{performance_dir}/{version}_loss.csv"
    pd.DataFrame({
        "epoch": range(1, len(history["train_loss"]) + 1),
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "train_mae": history["train_mae"],
        "val_mae": history["val_mae"],
    }).to_csv(loss_path, index=False)
    print(f"Saved loss history to {loss_path}", flush=True)

    best_state = torch.load(best_pth_path, map_location=device, weights_only=True)
    model.load_state_dict(best_state)
    return model, history


def evaluate(model, X_test, y_test, test_snr, performance_dir, version, batch_size, device):
    loader = DataLoader(
        NMRSequenceDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda",
    )
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in loader:
            predictions.append(model(x.to(device)).cpu().numpy())
    y_pred = np.concatenate(predictions, axis=0)

    plt.style.use("ggplot")
    metrics = {}
    results = {"SNR": test_snr}
    for idx, name in enumerate(("P", "Q")):
        y_true = y_test[:, idx] * 100.0
        y_hat = y_pred[:, idx] * 100.0
        residuals = y_true - y_hat
        rpe = np.abs(y_hat - y_true) / np.maximum(np.abs(y_true), 1e-8) * 100

        mse = float(np.mean((y_true - y_hat) ** 2))
        mae = float(np.mean(np.abs(y_true - y_hat)))
        rmse = float(np.sqrt(mse))
        metrics[name] = {
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse,
            "Mean_RPE": float(rpe.mean()),
            "Std_RPE": float(rpe.std()),
        }
        print(f"\nTest Set Metrics ({name}):")
        print(f"  MSE:      {mse:.6f}")
        print(f"  MAE:      {mae:.6f}")
        print(f"  RMSE:     {rmse:.6f}")
        print(f"  Mean RPE: {rpe.mean():.5f}%")

        plt.figure()
        plt.hist(rpe, bins=30, alpha=0.7, edgecolor="red")
        plt.xlabel(f"{name} RPE")
        plt.ylabel("Frequency")
        plt.title(f"{name} RPE Distribution")
        plt.tight_layout()
        plt.savefig(f"{performance_dir}/{version}_{name.lower()}_rpe_histogram.png", dpi=600)
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_hat, alpha=0.5, s=1)
        lo, hi = min(y_true.min(), y_hat.min()), max(y_true.max(), y_hat.max())
        plt.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect Prediction")
        plt.xlabel(f"Actual {name} (%)")
        plt.ylabel(f"Predicted {name} (%)")
        plt.title(f"Actual vs Predicted {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{performance_dir}/{version}_{name.lower()}_actual_vs_predicted.png", dpi=600)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, residuals, alpha=0.5, s=1)
        plt.axhline(0, color="r", linestyle="--", lw=2)
        plt.xlabel(f"Actual {name} (%)")
        plt.ylabel("Residuals (%)")
        plt.title(f"{name} Residuals Plot")
        plt.tight_layout()
        plt.savefig(f"{performance_dir}/{version}_{name.lower()}_residuals.png", dpi=600)
        plt.close()

        results[f"Actual_{name}"] = y_true
        results[f"Predicted_{name}"] = y_hat
        results[f"Residuals_{name}"] = residuals
        results[f"RPE_{name}"] = rpe

    pd.DataFrame(results).to_csv(f"{performance_dir}/{version}_results.csv", index=False)
    with open(f"{performance_dir}/{version}_metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=4)

    loss_csv = f"{performance_dir}/{version}_loss.csv"
    if os.path.exists(loss_csv):
        h = pd.read_csv(loss_csv)
        plt.figure()
        plt.plot(h["train_loss"].dropna(), label="Train Loss")
        plt.plot(h["val_loss"].dropna(), label="Val Loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(f"{performance_dir}/{version}_loss.png", dpi=600)
        plt.close()


if __name__ == "__main__":
    data_path = "data/Training_Data_RGC_Period_Test.parquet"
    version = "Training_Data_RGC_Period_Test_CNN"
    performance_dir = f"Model_Performance/{version}"
    model_dir = f"Models/{version}"
    os.makedirs(performance_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    learning_rate = 3e-4
    max_epochs = 500
    batch_size = 256
    weight_decay = 1e-5

    df = pd.read_parquet(data_path)
    signal_cols = df.columns[0:SIGNAL_LEN]
    df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=1 / 3, random_state=42)
    del df, df_temp
    gc.collect()

    print("Standardizing each raw spectrum (baseline curve is kept)...")
    X_train = standardize_traces(df_train[signal_cols].to_numpy(dtype=np.float32))
    X_val = standardize_traces(df_val[signal_cols].to_numpy(dtype=np.float32))
    X_test = standardize_traces(df_test[signal_cols].to_numpy(dtype=np.float32))
    print(f"Standardized train range: {X_train.min():.3f} to {X_train.max():.3f}")

    y_train = df_train[["P", "Q"]].to_numpy(dtype=np.float32)
    y_val = df_val[["P", "Q"]].to_numpy(dtype=np.float32)
    y_test = df_test[["P", "Q"]].to_numpy(dtype=np.float32)
    test_snr = df_test["SNR"].to_numpy(dtype=np.float32)
    print(f"Number of training data points: {len(y_train)}")
    del df_train, df_val, df_test
    gc.collect()

    print("\n" + "=" * 60)
    print("Training CNN on raw lineshapes")
    print("=" * 60)

    device = _device()
    model, _history = train_model(
        X_train, y_train, X_val, y_val,
        model_dir, performance_dir, version,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        batch_size=batch_size,
        weight_decay=weight_decay,
        device=device,
    )

    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)
    evaluate(model, X_test, y_test, test_snr, performance_dir, version, batch_size, device)
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
