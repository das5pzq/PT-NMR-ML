import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from pol_mlp import NMRDataset, train_model

# Edit these, then run: python ml/train_mlp.py
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data_creation", "Training_Data", "Sample_vector_p45_55.parquet",
)
VERSION = "deuteron_p45_55"
PERFORMANCE_DIR = f"Model_Performance/{VERSION}"
MODEL_DIR = f"Models/{VERSION}"

N_BINS = 500
SEED = 42
HOLDOUT = 0.2
TEST_OF_HOLDOUT = 1 / 3

LEARNING_RATE = 5e-3
MAX_EPOCHS = 200
HIDDEN_DIM = 16
NUM_BLOCKS = 2
BATCH_SIZE = 128
WEIGHT_DECAY = 1e-5


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    os.makedirs(PERFORMANCE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)
    signal_cols = [str(i) for i in range(N_BINS)]
    missing = [col for col in signal_cols if col not in df.columns]
    if missing or "P" not in df.columns:
        raise SystemExit(f"{DATA_PATH} needs columns 0..{N_BINS - 1} and P")

    # signal_generator.py does not store Q. Spin-1 equilibrium alignment from P.
    p_values = df["P"].to_numpy(dtype=np.float64)
    df["Q"] = (2.0 - np.sqrt(np.maximum(0.0, 4.0 - 3.0 * p_values * p_values))).astype(np.float32)

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

    y_train = df_train[["P", "Q"]].to_numpy(dtype=np.float32)
    y_val = df_val[["P", "Q"]].to_numpy(dtype=np.float32)
    y_test = df_test[["P", "Q"]].to_numpy(dtype=np.float32)
    test_SNR = pd.to_numeric(df_test["SNR"], errors="coerce").to_numpy(dtype=np.float32)

    p_mean = float(y_train[:, 0].mean())
    p_std = float(y_train[:, 0].std())
    if p_std < 1e-8:
        p_std = 1.0
    q_mean = float(y_train[:, 1].mean())
    q_std = float(y_train[:, 1].std())
    if q_std < 1e-8:
        q_std = 1.0

    norm_stats = {
        "ps_mean": ps_mean,
        "ps_std": ps_std,
        "P_mean": p_mean,
        "P_std": p_std,
        "Q_mean": q_mean,
        "Q_std": q_std,
        "input_size": 1,
        "num_bins": N_BINS,
    }
    with open(f"{PERFORMANCE_DIR}/{VERSION}_norm.json", "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"Number of training data points: {len(df_train)}")
    del df, df_temp, df_train, df_val, df_test

    print("\n" + "=" * 60)
    print(f"Training PolMLP  bins={N_BINS} hidden={HIDDEN_DIM} blocks={NUM_BLOCKS}")
    print("=" * 60)

    model, _history = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        MODEL_DIR, PERFORMANCE_DIR, VERSION,
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        norm_stats=norm_stats,
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
    targets = ("P", "Q")
    metrics = {}

    plt.style.use("ggplot")
    results = {"SNR": test_SNR}
    for idx, name in enumerate(targets):
        y_true = y_test[:, idx] * 100.0
        y_hat = y_pred[:, idx] * 100.0
        residuals = y_true - y_hat
        rpe = np.abs(y_hat - y_true) / np.abs(y_true) * 100

        mse = np.mean((y_true - y_hat) ** 2)
        mae = np.mean(np.abs(y_true - y_hat))
        rmse = np.sqrt(mse)
        metrics[name] = {
            "MSE": float(mse),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "Mean_RPE": float(rpe.mean()),
            "Std_RPE": float(rpe.std()),
        }

        print(f"\nTest Set Metrics ({name}):")
        print(f"  MSE:      {mse:.6f}")
        print(f"  MAE:      {mae:.6f}")
        print(f"  RMSE:     {rmse:.6f}")
        print(f"  Mean RPE: {rpe.mean():.5f}%")

        plt.hist(rpe, bins=30, alpha=0.7, edgecolor="red")
        plt.xlabel(f"{name} RPE")
        plt.ylabel("Frequency")
        plt.title(f"{name} RPE Distribution")
        plt.figtext(
            0.65, 0.8, f"Mean: {rpe.mean():.5f}%",
            fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="red", ec="none", alpha=0.8),
            color="white",
        )
        plt.tight_layout()
        plt.savefig(f"{PERFORMANCE_DIR}/{VERSION}_{name.lower()}_rpe_histogram.png", dpi=600)
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
        plt.savefig(f"{PERFORMANCE_DIR}/{VERSION}_{name.lower()}_actual_vs_predicted.png", dpi=600)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, residuals, alpha=0.5, s=1)
        plt.axhline(0, color="r", linestyle="--", lw=2)
        plt.xlabel(f"Actual {name} (%)")
        plt.ylabel("Residuals (%)")
        plt.title(f"{name} Residuals Plot")
        plt.tight_layout()
        plt.savefig(f"{PERFORMANCE_DIR}/{VERSION}_{name.lower()}_residuals.png", dpi=600)
        plt.close()

        results[f"Actual_{name}"] = y_true
        results[f"Predicted_{name}"] = y_hat
        results[f"Residuals_{name}"] = residuals
        results[f"RPE_{name}"] = rpe

    pd.DataFrame(results).to_csv(f"{PERFORMANCE_DIR}/{VERSION}_results.csv", index=False)
    with open(f"{PERFORMANCE_DIR}/{VERSION}_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

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

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
