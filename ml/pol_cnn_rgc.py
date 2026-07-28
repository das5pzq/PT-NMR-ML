"""
Example 1D CNN for RGC polarization with baseline left in the spectra.

No offline baseline subtraction. A differentiable front-end (finite differences +
per-spectrum InstanceNorm) suppresses slow baseline energy, then a compact Conv1d
backbone predicts P only (no Area multitask).
"""

import os
import gc
import json
import random
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.core import LightningModule
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

sys.stdout.flush()
sys.stderr.flush()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Front-end: 1 = first differences (kills DC), 2 = second differences (also kills linear trend)
DIFF_ORDER = 1
P_EXCLUDE_ABS = 0.15


def _accelerator():
    if torch.cuda.is_available():
        return "cuda", torch.cuda.device_count()
    if torch.backends.mps.is_available():
        return "mps", 1
    return "cpu", 1


ACCELERATOR, N_DEVICES = _accelerator()


class NMRDataset(Dataset):
    """Raw spectra as float32; returns (1, L) for Conv1d."""

    def __init__(self, X, y):
        self.X = torch.as_tensor(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.as_tensor(np.ascontiguousarray(y, dtype=np.float32))
        if self.y.ndim == 1:
            self.y = self.y.reshape(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]


class SpectrumFrontEnd(nn.Module):
    """
    Differentiable high-pass + per-spectrum normalization.

    This is not classical wing / polynomial baseline fitting: baselines stay in
    the training data; the network learns on a representation that attenuates
    slow components and equalizes per-trace scale.
    """

    def __init__(self, diff_order=1, eps=1e-5):
        super().__init__()
        if diff_order < 1:
            raise ValueError("diff_order must be >= 1")
        self.diff_order = int(diff_order)
        self.eps = eps
        # Normalize across the frequency axis for each spectrum independently.
        self.norm = nn.InstanceNorm1d(1, affine=True, eps=eps)

    def forward(self, x):
        # x: (batch, 1, length)
        for _ in range(self.diff_order):
            x = x[..., 1:] - x[..., :-1]
        return self.norm(x)


class CNNWithFrontEnd(nn.Module):
    """
    Front-end -> compact 1D CNN -> P.

    A slim stack learns this task quickly. The paper-style Inception+residual CNN
    (~2.4M params) stayed stuck near a constant-P predictor on this RGC data.
    """

    def __init__(self, diff_order=DIFF_ORDER, channels=(128, 256, 256), pool_len=32):
        super().__init__()
        c1, c2, c3 = channels
        self.front_end = SpectrumFrontEnd(diff_order=diff_order)
        self.features = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(pool_len),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(c3 * pool_len, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.front_end(x)
        return self.head(self.features(x))


class CNNLightningModule(LightningModule):
    def __init__(self, learning_rate=1e-3, diff_order=DIFF_ORDER):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = CNNWithFrontEnd(diff_order=diff_order)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_mae", F.l1_loss(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.log("val_loss", self.criterion(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", F.l1_loss(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.log("test_loss", self.criterion(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_mae", F.l1_loss(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


class LossHistoryCallback(Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.epoch_train_loss = []
        self.epoch_val_loss = []
        self.epoch_train_mae = []
        self.epoch_val_mae = []

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        if "train_loss" in m:
            self.epoch_train_loss.append(float(m["train_loss"].cpu()))
        if "val_loss" in m:
            self.epoch_val_loss.append(float(m["val_loss"].cpu()))
        if "train_mae" in m:
            self.epoch_train_mae.append(float(m["train_mae"].cpu()))
        if "val_mae" in m:
            self.epoch_val_mae.append(float(m["val_mae"].cpu()))

    def on_fit_end(self, trainer, pl_module):
        n = max(len(self.epoch_train_loss), len(self.epoch_val_loss))
        if n == 0:
            return
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        pd.DataFrame(
            {
                "epoch": range(1, n + 1),
                "train_loss": self.epoch_train_loss + [None] * (n - len(self.epoch_train_loss)),
                "val_loss": self.epoch_val_loss + [None] * (n - len(self.epoch_val_loss)),
                "train_mae": self.epoch_train_mae + [None] * (n - len(self.epoch_train_mae)),
                "val_mae": self.epoch_val_mae + [None] * (n - len(self.epoch_val_mae)),
            }
        ).to_csv(self.save_path, index=False)
        print(f"Saved loss history to {self.save_path}")


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    model_dir,
    performance_dir,
    version,
    num_workers=4,
    learning_rate=1e-3,
    max_epochs=500,
    batch_size=128,
    diff_order=DIFF_ORDER,
):
    # Windows spawn pickles the whole dataset into each worker. Multi-GB spectrum
    # arrays hit OSError 22 (Invalid argument); Lightning then dies in teardown.
    if sys.platform == "win32" and num_workers > 0:
        print(
            f"Windows: forcing num_workers=0 (was {num_workers}) to avoid "
            "pickling large in-memory spectra into DataLoader workers."
        )
        num_workers = 0

    pin = torch.cuda.is_available()
    persistent = num_workers > 0
    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    train_loader = DataLoader(
        NMRDataset(X_train, y_train), batch_size=batch_size, shuffle=True, **loader_kw
    )
    val_loader = DataLoader(
        NMRDataset(X_val, y_val), batch_size=batch_size, shuffle=False, **loader_kw
    )
    test_loader = DataLoader(
        NMRDataset(X_test, y_test), batch_size=batch_size, shuffle=False, **loader_kw
    )

    ckpt_path = f"{model_dir}/best_model_checkpoint.ckpt"
    if os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        model = CNNLightningModule.load_from_checkpoint(ckpt_path)
    else:
        model = CNNLightningModule(
            learning_rate=learning_rate,
            diff_order=diff_order,
        )

    checkpoint_cb = ModelCheckpoint(
        dirpath=model_dir,
        filename="best_model_checkpoint",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=True,
    )
    loss_cb = LossHistoryCallback(save_path=f"{performance_dir}/{version}_loss.csv")

    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_cb, LearningRateMonitor(), loss_cb],
        logger=CSVLogger(performance_dir, name="training_log"),
        accelerator=ACCELERATOR,
        devices=N_DEVICES,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, val_loader)

    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt and os.path.isfile(best_ckpt):
        model = CNNLightningModule.load_from_checkpoint(best_ckpt)
        torch.save(model.model.state_dict(), f"{model_dir}/best_model.pth")
    else:
        torch.save(model.model.state_dict(), f"{model_dir}/best_model.pth")

    trainer.test(model, test_loader)
    return model, trainer


if __name__ == "__main__":
    data_path = "data/Training_Data_RGC_1M.parquet"
    version = "RGC_CNN_V1"
    performance_dir = f"Model_Performance/{version}"
    model_dir = f"Models/{version}"
    os.makedirs(performance_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Accelerator={ACCELERATOR} devices={N_DEVICES}")
    print(f"Front-end: diff_order={DIFF_ORDER}, InstanceNorm; backbone=slim Conv1d")
    print("Training on raw baseline-included spectra (no offline baseline subtraction).")

    df = pd.read_parquet(data_path)

    ### remove rows where P is less than -0.15 or greater than 0.15
    n_before = len(df)
    df = df[(df["P"] <= -0.05) | (df["P"] >= 0.05)].reset_index(drop=True)
    # df = df[df["P"] > 0.05].reset_index(drop=True)
    print(f"Excluded |P| < 0.05: {n_before} -> {len(df)} samples")

    signal_cols = df.columns[0:512]
    # Keep raw volts in the dataset; front-end handles high-pass + per-spectrum norm.
    meta = {
        "diff_order": DIFF_ORDER,
        "backbone": "slim_conv1d",
        "p_exclude_abs": P_EXCLUDE_ABS,
        "n_signal_bins": 512,
        "note": "No offline baseline subtraction; SpectrumFrontEnd is inside the model.",
    }
    with open(f"{performance_dir}/{version}_config.json", "w") as f:
        json.dump(meta, f, indent=4)

    df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=1 / 3, random_state=42)
    del df, df_temp
    gc.collect()

    X_train = df_train[signal_cols].values.astype("float32")
    X_val = df_val[signal_cols].values.astype("float32")
    X_test = df_test[signal_cols].values.astype("float32")
    y_train = df_train["P"].values.astype("float32").reshape(-1, 1)
    y_val = df_val["P"].values.astype("float32").reshape(-1, 1)
    y_test = df_test["P"].values.astype("float32").reshape(-1, 1)
    test_SNR = df_test["SNR"].values.astype("float32")

    print(f"Train/val/test: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    print(f"Raw signal range (train): {X_train.min():.4f} to {X_train.max():.4f}")
    print(
        f"Chance MAE (predict mean P): {np.mean(np.abs(y_train - y_train.mean())):.4f} "
        f"| Var(P)={y_train.var():.4f}"
    )

    del df_train, df_val, df_test
    gc.collect()

    num_workers = 4
    learning_rate = 1e-3
    max_epochs = 100
    batch_size = 256

    print("\n" + "=" * 60)
    print("Training RGC CNN (baseline-robust front-end)")
    print("=" * 60)

    model, trainer = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        model_dir,
        performance_dir,
        version,
        num_workers=num_workers,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        batch_size=batch_size,
        diff_order=DIFF_ORDER,
    )

    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)

    eval_workers = 0 if sys.platform == "win32" else num_workers
    test_loader = DataLoader(
        NMRDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=eval_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=eval_workers > 0,
    )

    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in test_loader:
            predictions.append(model(x.to(model.device)).cpu().numpy())

    y_pred = np.concatenate(predictions, axis=0)
    y_test_flat = y_test.flatten() * 100.0
    y_pred_flat = y_pred.flatten() * 100.0

    mse = np.mean((y_test_flat - y_pred_flat) ** 2)
    mae = np.mean(np.abs(y_test_flat - y_pred_flat))
    rmse = np.sqrt(mse)
    rpe = np.abs(y_pred_flat - y_test_flat) / np.abs(y_test_flat) * 100
    rpe_95 = rpe[rpe <= np.percentile(rpe, 95)]

    print(f"\nTest Set Metrics:")
    print(f"  MSE:      {mse:.6f}")
    print(f"  MAE:      {mae:.6f}")
    print(f"  RMSE:     {rmse:.6f}")
    print(f"  Mean RPE: {rpe.mean():.5f}%")

    plt.style.use("ggplot")

    plt.hist(rpe, bins=30, alpha=0.7, edgecolor="red")
    plt.xlabel("Polarization RPE")
    plt.ylabel("Frequency")
    plt.title("Polarization RPE Distribution")
    plt.figtext(
        0.65,
        0.8,
        f"Mean: {rpe.mean():.5f}%\nStd: {rpe.std():.5f}%",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", fc="red", ec="none", alpha=0.8),
        color="white",
    )
    plt.tight_layout()
    plt.savefig(f"{performance_dir}/{version}_rpe_histogram.png", dpi=600)
    plt.close()

    residuals = y_test_flat - y_pred_flat

    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_flat, y_pred_flat, alpha=0.5, s=1)
    lo, hi = min(y_test_flat.min(), y_pred_flat.min()), max(y_test_flat.max(), y_pred_flat.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect Prediction")
    plt.xlabel("Actual Polarization (%)")
    plt.ylabel("Predicted Polarization (%)")
    plt.title("Actual vs Predicted Polarization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{performance_dir}/{version}_actual_vs_predicted.png", dpi=600)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_flat, residuals, alpha=0.5, s=1)
    plt.axhline(0, color="r", linestyle="--", lw=2)
    plt.xlabel("Actual Polarization (%)")
    plt.ylabel("Residuals (%)")
    plt.title("Residuals Plot")
    plt.tight_layout()
    plt.savefig(f"{performance_dir}/{version}_residuals.png", dpi=600)
    plt.close()

    pd.DataFrame(
        {
            "Actual": y_test_flat,
            "Predicted": y_pred_flat,
            "Residuals": residuals,
            "RPE": rpe,
            "SNR": test_SNR,
        }
    ).to_csv(f"{performance_dir}/{version}_results.csv", index=False)

    with open(f"{performance_dir}/{version}_metrics_summary.json", "w") as f:
        json.dump(
            {
                "MSE": float(mse),
                "MAE": float(mae),
                "RMSE": float(rmse),
                "Mean_RPE": float(rpe.mean()),
                "Std_RPE": float(rpe.std()),
                "Mean_RPE_95th": float(rpe_95.mean()),
                "Std_RPE_95th": float(rpe_95.std()),
            },
            f,
            indent=4,
        )

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

    # Save a one-batch front-end preview for sanity checking
    with torch.no_grad():
        x0 = torch.from_numpy(X_test[:1]).unsqueeze(1)
        z0 = model.model.front_end(x0).squeeze().cpu().numpy()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    axes[0].plot(X_test[0], lw=1.0)
    axes[0].set_title(f"Raw input (baseline included)  P={y_test[0, 0]:.4f}")
    axes[0].set_ylabel("Signal")
    axes[1].plot(z0, lw=1.0, color="#b22222")
    axes[1].set_title("After SpectrumFrontEnd (diff + InstanceNorm)")
    axes[1].set_xlabel("Bin")
    axes[1].set_ylabel("Front-end out")
    fig.tight_layout()
    fig.savefig(f"{performance_dir}/{version}_frontend_preview.png", dpi=200)
    plt.close()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
