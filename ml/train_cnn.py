import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from pol_cnn import NMRDataset, train_model

# Edit these, then run: python ml/train_cnn.py
# DATA_PATH is relative to the directory you run from, unless you paste an absolute path.
DATA_PATH = "data/Training_Data_RGC_Period_Test.parquet"
VERSION = "CNN_RGC_Period_Test"
PERFORMANCE_DIR = f"Model_Performance/{VERSION}"
MODEL_DIR = f"Models/{VERSION}"

N_BINS = 512
SEED = 42
HOLDOUT = 0.2
TEST_OF_HOLDOUT = 1 / 3

LEARNING_RATE = 3e-4
MAX_EPOCHS = 100
BATCH_SIZE = 512


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    os.makedirs(PERFORMANCE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)
    if df.shape[1] < N_BINS + 1 or "P" not in df.columns:
        raise SystemExit(f"{DATA_PATH} needs at least {N_BINS} spectrum columns plus P")
    signal_cols = df.columns[:N_BINS]
    if "P" in signal_cols or "SNR" in signal_cols:
        raise SystemExit(f"N_BINS={N_BINS} includes a label column. Lower N_BINS.")

    df_train, df_temp = train_test_split(df, test_size=HOLDOUT, random_state=SEED)
    df_val, df_test = train_test_split(df_temp, test_size=TEST_OF_HOLDOUT, random_state=SEED)

    X_train = df_train[signal_cols].to_numpy(dtype=np.float32)
    X_val = df_val[signal_cols].to_numpy(dtype=np.float32)
    X_test = df_test[signal_cols].to_numpy(dtype=np.float32)

    # One mean and one std for the whole training array, so the lineshape stays intact.
    ps_mean = float(X_train.mean())
    ps_std = float(X_train.std())
    if ps_std < 1e-8:
        ps_std = 1.0
    X_train = (X_train - ps_mean) / ps_std
    X_val = (X_val - ps_mean) / ps_std
    X_test = (X_test - ps_mean) / ps_std
    print(f"Z-scored signal range (train): {X_train.min():.4f} to {X_train.max():.4f}")

    with open(f"{PERFORMANCE_DIR}/{VERSION}_norm.json", "w", encoding="utf-8") as f:
        json.dump({"ps_mean": ps_mean, "ps_std": ps_std, "n_bins": N_BINS}, f, indent=2)

    y_train = df_train["P"].to_numpy(dtype=np.float32).reshape(-1, 1)
    y_val = df_val["P"].to_numpy(dtype=np.float32).reshape(-1, 1)
    y_test = df_test["P"].to_numpy(dtype=np.float32).reshape(-1, 1)
    test_SNR = pd.to_numeric(df_test["SNR"], errors="coerce").to_numpy(dtype=np.float32)

    print(f"Number of training data points: {len(df_train)}")
    del df, df_temp, df_train, df_val, df_test

    print("\n" + "=" * 60)
    print("Training CNN Model")
    print("=" * 60)

    model, _history = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        MODEL_DIR, PERFORMANCE_DIR, VERSION,
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        input_length=N_BINS,
        batch_size=BATCH_SIZE,
    )

    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)

    test_loader = DataLoader(
        NMRDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in test_loader:
            predictions.append(model(x).cpu().numpy())

    y_pred = np.concatenate(predictions, axis=0)
    y_true = y_test.flatten() * 100.0
    y_hat = y_pred.flatten() * 100.0
    residuals = y_true - y_hat
    rpe = np.abs(y_hat - y_true) / np.abs(y_true) * 100

    mse = np.mean((y_true - y_hat) ** 2)
    mae = np.mean(np.abs(y_true - y_hat))
    rmse = np.sqrt(mse)

    print("\nTest Set Metrics:")
    print(f"  MSE:      {mse:.6f}")
    print(f"  MAE:      {mae:.6f}")
    print(f"  RMSE:     {rmse:.6f}")
    print(f"  Mean RPE: {rpe.mean():.5f}%")

    plt.style.use("ggplot")
    pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_hat,
        "Residuals": residuals,
        "RPE": rpe,
        "SNR": test_SNR,
    }).to_csv(f"{PERFORMANCE_DIR}/{VERSION}_results.csv", index=False)

    with open(f"{PERFORMANCE_DIR}/{VERSION}_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "MSE": float(mse),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "Mean_RPE": float(rpe.mean()),
            "Std_RPE": float(rpe.std()),
        }, f, indent=4)

    plt.hist(rpe, bins=30, alpha=0.7, edgecolor="red")
    plt.xlabel("Polarization RPE")
    plt.ylabel("Frequency")
    plt.title("Polarization RPE Distribution")
    plt.figtext(
        0.65, 0.8, f"Mean: {rpe.mean():.5f}%",
        fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="red", ec="none", alpha=0.8),
        color="white",
    )
    plt.tight_layout()
    plt.savefig(f"{PERFORMANCE_DIR}/{VERSION}_rpe_histogram.png", dpi=600)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_hat, alpha=0.5, s=1)
    lo, hi = min(y_true.min(), y_hat.min()), max(y_true.max(), y_hat.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect Prediction")
    plt.xlabel("Actual Polarization (%)")
    plt.ylabel("Predicted Polarization (%)")
    plt.title("Actual vs Predicted Polarization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PERFORMANCE_DIR}/{VERSION}_actual_vs_predicted.png", dpi=600)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, residuals, alpha=0.5, s=1)
    plt.axhline(0, color="r", linestyle="--", lw=2)
    plt.xlabel("Actual Polarization (%)")
    plt.ylabel("Residuals (%)")
    plt.title("Residuals Plot")
    plt.tight_layout()
    plt.savefig(f"{PERFORMANCE_DIR}/{VERSION}_residuals.png", dpi=600)
    plt.close()

    loss_csv = f"{PERFORMANCE_DIR}/{VERSION}_loss.csv"
    if os.path.exists(loss_csv):
        h = pd.read_csv(loss_csv)
        plt.figure()
        plt.plot(h["train_loss"].dropna(), label="Train Loss")
        plt.plot(h["val_loss"].dropna(), label="Val Loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(f"{PERFORMANCE_DIR}/{VERSION}_loss.png", dpi=600)
        plt.close()

        if "train_mae" in h.columns and "val_mae" in h.columns:
            plt.figure()
            plt.plot(h["train_mae"].dropna(), label="Train MAE")
            plt.plot(h["val_mae"].dropna(), label="Val MAE")
            plt.legend()
            plt.xlabel("Epoch")
            plt.yscale("log")
            plt.tight_layout()
            plt.savefig(f"{PERFORMANCE_DIR}/{VERSION}_mae.png", dpi=600)
            plt.close()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
