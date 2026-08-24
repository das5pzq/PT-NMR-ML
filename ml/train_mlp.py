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

from pol_mlp import *

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device('cuda')


if __name__ == "__main__":
    data_path = "data/Training_Data_RGC_17_34_500K.parquet"
    version = 'Training_Data_RGC_17_34_500K_V3'
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

    y_train = df_train[["P", "Q"]].values.astype('float32')
    y_val = df_val[["P", "Q"]].values.astype('float32')
    y_test = df_test[["P", "Q"]].values.astype('float32')
    test_SNR = df_test["SNR"].values.astype('float32')

    print(f"Number of training data points: {len(df_train)}")

    del df_train, df_val, df_test
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    num_workers  = 13
    learning_rate = 3e-4
    max_epochs   = 500
    hidden_dim   = 512
    batch_size   = 256
    weight_decay = 1e-5

    print("\n" + "=" * 60)
    print("Training MLP Model")
    print("=" * 60)

    model, trainer = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        model_dir, performance_dir, version,
        num_workers=num_workers,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        weight_decay=weight_decay,
    )

    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)

    test_loader = DataLoader(NMRDataset(X_test, y_test), batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                             persistent_workers=True)

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            x, _ = batch
            predictions.append(model(x).cpu().numpy())

    y_pred = np.concatenate(predictions, axis=0)
    targets = ("P", "Q")
    metrics = {}

    plt.style.use('ggplot')

    results = {'SNR': test_SNR}
    for idx, name in enumerate(targets):
        y_true = y_test[:, idx] * 100.0
        y_hat = y_pred[:, idx] * 100.0
        residuals = y_true - y_hat
        rpe = np.abs(y_hat - y_true) / np.abs(y_true) * 100

        mse = np.mean((y_true - y_hat) ** 2)
        mae = np.mean(np.abs(y_true - y_hat))
        rmse = np.sqrt(mse)
        metrics[name] = {
            'MSE': float(mse),
            'MAE': float(mae),
            'RMSE': float(rmse),
            'Mean_RPE': float(rpe.mean()),
            'Std_RPE': float(rpe.std()),
        }

        print(f"\nTest Set Metrics ({name}):")
        print(f"  MSE:      {mse:.6f}")
        print(f"  MAE:      {mae:.6f}")
        print(f"  RMSE:     {rmse:.6f}")
        print(f"  Mean RPE: {rpe.mean():.5f}%")

        plt.hist(rpe, bins=30, alpha=0.7, edgecolor='red')
        plt.xlabel(f'{name} RPE')
        plt.ylabel('Frequency')
        plt.title(f'{name} RPE Distribution')
        plt.figtext(0.65, 0.8, f"Mean: {rpe.mean():.5f}%",
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc='red', ec="none", alpha=0.8),
                    color='white')
        plt.tight_layout()
        plt.savefig(f"{performance_dir}/{version}_{name.lower()}_rpe_histogram.png", dpi=600)
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_hat, alpha=0.5, s=1)
        lo, hi = min(y_true.min(), y_hat.min()), max(y_true.max(), y_hat.max())
        plt.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel(f'Actual {name} (%)')
        plt.ylabel(f'Predicted {name} (%)')
        plt.title(f'Actual vs Predicted {name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{performance_dir}/{version}_{name.lower()}_actual_vs_predicted.png", dpi=600)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, residuals, alpha=0.5, s=1)
        plt.axhline(0, color='r', linestyle='--', lw=2)
        plt.xlabel(f'Actual {name} (%)')
        plt.ylabel('Residuals (%)')
        plt.title(f'{name} Residuals Plot')
        plt.tight_layout()
        plt.savefig(f"{performance_dir}/{version}_{name.lower()}_residuals.png", dpi=600)
        plt.close()

        results[f'Actual_{name}'] = y_true
        results[f'Predicted_{name}'] = y_hat
        results[f'Residuals_{name}'] = residuals
        results[f'RPE_{name}'] = rpe

    pd.DataFrame(results).to_csv(f"{performance_dir}/{version}_results.csv", index=False)

    with open(f"{performance_dir}/{version}_metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=4)

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

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)