import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import json
import warnings
warnings.filterwarnings('ignore')
import gc

from pol_cnn import *

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


if __name__ == "__main__":
    data_path = "data/Training_Data_RGC_3_55_500K.parquet"
    version = 'CNN_RGC_3_55_V1'
    performance_dir = f"Model_Performance/{version}"
    model_dir = f"Models/{version}"
    os.makedirs(performance_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_parquet(data_path)
    signal_cols = df.columns[0:512]

    scaler_path = f"{performance_dir}/{version}_scaler.pkl"
    if os.path.isfile(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded existing scaler from {scaler_path}")
    else:
        scaler = MinMaxScaler()
        scaler.fit(df[signal_cols].values.astype('float32'))
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Fitted and saved new scaler to {scaler_path}")

    df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=1 / 3, random_state=42)
    del df, df_temp
    gc.collect()

    X_train = scaler.transform(df_train[signal_cols].values.astype('float32'))
    X_val = scaler.transform(df_val[signal_cols].values.astype('float32'))
    X_test = scaler.transform(df_test[signal_cols].values.astype('float32'))

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print(f"Scaled signal range (train): {X_train.min():.4f} to {X_train.max():.4f}")

    y_train = df_train["P"].values.astype('float32').reshape(-1, 1)
    y_val = df_val["P"].values.astype('float32').reshape(-1, 1)
    y_test = df_test["P"].values.astype('float32').reshape(-1, 1)
    test_SNR = df_test["SNR"].values.astype('float32')

    print(f"Number of training data points: {len(df_train)}")

    del df_train, df_val, df_test
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    learning_rate = 3e-4
    max_epochs = 100
    batch_size = 512
    input_length = len(X_train[0])

    print("\n" + "=" * 60)
    print("Training CNN Model")
    print("=" * 60)

    model, trainer = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        model_dir, performance_dir, version,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        input_length=input_length,
        batch_size=batch_size,
    )

    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)

    test_loader = DataLoader(
        NMRDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in test_loader:
            predictions.append(model(x).cpu().numpy())

    y_pred = np.concatenate(predictions, axis=0)

    y_test_flat = y_test.flatten() * 100.0
    y_pred_flat = y_pred.flatten() * 100.0
    residuals = y_test_flat - y_pred_flat
    rpe = np.abs(y_pred_flat - y_test_flat) / y_test_flat * 100

    mse = np.mean((y_test_flat - y_pred_flat) ** 2)
    mae = np.mean(np.abs(y_test_flat - y_pred_flat))
    rmse = np.sqrt(mse)

    print(f"\nTest Set Metrics:")
    print(f"  MSE:      {mse:.6f}")
    print(f"  MAE:      {mae:.6f}")
    print(f"  RMSE:     {rmse:.6f}")
    print(f"  Mean RPE: {rpe.mean():.5f}%")

    plt.style.use('ggplot')

    pd.DataFrame({
        'Actual': y_test_flat,
        'Predicted': y_pred_flat,
        'Residuals': residuals,
        'RPE': rpe,
        'SNR': test_SNR,
    }).to_csv(f"{performance_dir}/{version}_results.csv", index=False)

    with open(f"{performance_dir}/{version}_metrics_summary.json", "w") as f:
        json.dump({
            'MSE': float(mse),
            'MAE': float(mae),
            'RMSE': float(rmse),
            'Mean_RPE': float(rpe.mean()),
            'Std_RPE': float(rpe.std()),
        }, f, indent=4)

    plt.hist(rpe, bins=30, alpha=0.7, edgecolor='red')
    plt.xlabel('Polarization RPE')
    plt.ylabel('Frequency')
    plt.title('Polarization RPE Distribution')
    plt.figtext(0.65, 0.8, f"Mean: {rpe.mean():.5f}%",
                fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc='red', ec="none", alpha=0.8),
                color='white')
    plt.tight_layout()
    plt.savefig(f"{performance_dir}/{version}_rpe_histogram.png", dpi=600)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_flat, y_pred_flat, alpha=0.5, s=1)
    lo, hi = min(y_test_flat.min(), y_pred_flat.min()), max(y_test_flat.max(), y_pred_flat.max())
    plt.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Polarization (%)')
    plt.ylabel('Predicted Polarization (%)')
    plt.title('Actual vs Predicted Polarization')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{performance_dir}/{version}_actual_vs_predicted.png", dpi=600)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_flat, residuals, alpha=0.5, s=1)
    plt.axhline(0, color='r', linestyle='--', lw=2)
    plt.xlabel('Actual Polarization (%)')
    plt.ylabel('Residuals (%)')
    plt.title('Residuals Plot')
    plt.tight_layout()
    plt.savefig(f"{performance_dir}/{version}_residuals.png", dpi=600)
    plt.close()

    loss_csv = f"{performance_dir}/{version}_loss.csv"
    if os.path.exists(loss_csv):
        h = pd.read_csv(loss_csv)
        plt.figure()
        plt.plot(h['train_loss'].dropna(), label='Train Loss')
        plt.plot(h['val_loss'].dropna(), label='Val Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f"{performance_dir}/{version}_loss.png", dpi=600)
        plt.close()

        if 'train_mae' in h.columns and 'val_mae' in h.columns:
            plt.figure()
            plt.plot(h['train_mae'].dropna(), label='Train MAE')
            plt.plot(h['val_mae'].dropna(), label='Val MAE')
            plt.legend()
            plt.xlabel('Epoch')
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(f"{performance_dir}/{version}_mae.png", dpi=600)
            plt.close()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
