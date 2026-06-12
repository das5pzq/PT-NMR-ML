from __future__ import annotations

import os
import sys
import argparse
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# =========================
# USER SETTINGS (EDIT THESE)
# =========================

SEED = 12345

# Model: set [] for linear (dumbest possilbe model)
HIDDEN: List[int] = [20, 20]      
DROPOUT = 0.0

# Data split
VAL_FRACTION = 0.15

# Training
BATCH_SIZE = 256
# Learning rate scheduler (CosineAnnealingWarmRestarts)
USE_LR_SCHEDULER = True
LR_SCHEDULER_T_0 = 100      # Number of iterations for the first restart
LR_SCHEDULER_T_MULT = 2       # A factor increases T_i after a restart
LR_SCHEDULER_ETA_MIN = 1e-6   # Minimum learning rate
MAX_EPOCHS = 5000

# Normalization
STANDARDIZE_X = True
SCALE_Y_TO_01 = True         # keeps eval script behavior identical

# Closed-form ridge (only used when HIDDEN == [])
USE_CLOSED_FORM_RIDGE = False
RIDGE_ALPHA = 1e-6           # ridge strength in standardized feature space (tune 1e-8 ... 1e-3)

# Band metric near 0.3% polarization:
# We compute A_TARGET = 0.3 * max(area_in_csv) so this always matches your data.
P_EVAL_FRACTION_OF_MAX = 0.3
BAND_REL_WIDTH = 0.20        # ±20%

# Output (KEEP NAMES SAME)
OUT_DIR = "area_model_out_TE"
BEST_MODEL_PATH = os.path.join(OUT_DIR, "best_model.pt")
SCALERS_PATH = os.path.join(OUT_DIR, "scalers.npz")
LOSS_PLOT_PATH = os.path.join(OUT_DIR, "loss_plot.png")


# =========================
# Model definition (must match your eval script)
# =========================

class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================
# Scaling helpers
# =========================

def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    sig[sig == 0] = 1.0
    return mu, sig


def standardize_apply(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return (X - mu) / sig


def minmax_fit(y: np.ndarray) -> Tuple[float, float]:
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if y_max == y_min:
        y_max = y_min + 1.0
    return y_min, y_max


def minmax_apply(y: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    return (y - y_min) / (y_max - y_min)


def minmax_invert(y01: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    return y01 * (y_max - y_min) + y_min


# =========================
# Metrics
# =========================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    resid = y_pred - y_true
    rmse = float(np.sqrt(np.mean(resid * resid)))
    mae = float(np.mean(np.abs(resid)))

    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def band_metrics(y_true: np.ndarray, y_pred: np.ndarray, center: float, rel_width: float) -> Dict[str, float]:
    lo = center * (1.0 - rel_width)
    hi = center * (1.0 + rel_width)
    mask = (y_true >= lo) & (y_true <= hi)
    n = int(np.sum(mask))
    if n == 0:
        return {"band_rmse": float("nan"), "band_rel_rmse": float("nan"), "band_n": 0}

    resid = (y_pred - y_true)[mask]
    band_rmse = float(np.sqrt(np.mean(resid * resid)))
    band_rel_rmse = band_rmse / center if center != 0 else float("nan")
    return {"band_rmse": band_rmse, "band_rel_rmse": band_rel_rmse, "band_n": n}


# =========================
# Closed-form ridge for linear model
# =========================

def fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
    """
    Fits y ≈ Xw + b with ridge penalty alpha*||w||^2 (NO penalty on bias).
    Returns (w, b).
    """
    n, d = X.shape
    ones = np.ones((n, 1), dtype=X.dtype)
    Xaug = np.hstack([X, ones])  # (n, d+1)

    A = Xaug.T @ Xaug  # (d+1, d+1)
    reg = np.zeros_like(A)
    reg[:d, :d] = alpha * np.eye(d, dtype=X.dtype)  # penalize weights only

    bvec = Xaug.T @ y  # (d+1,)
    waug = np.linalg.solve(A + reg, bvec)

    w = waug[:d]
    bias = float(waug[d])
    return w, bias


# =========================
# Main
# =========================

def load_model_and_predict(in_csv: str) -> None:
    """Load a trained model and predict on the full dataset, saving results to CSV."""
    print("Loading trained model and making predictions...")
    
    # Load data
    df = pd.read_parquet(in_csv)
    
    if "Area" not in df.columns:
        raise ValueError("Expected an 'Area' column in the CSV.")
    
    y_phys = df["Area"].to_numpy(dtype=np.float64)
    X_raw = df.iloc[:, :500].astype('float64').values
    
    # Add noise to signal before scaling (same as training)
    noise_std = 2*2.5 * 10**-5  # 5e-5
    # noise_std = 0.0
    noise = np.random.normal(0, noise_std, X_raw.shape)
    X_raw = X_raw + noise
    
    print(f"Predicting on full dataset: N={len(df)}")
    
    # Load scalers
    if not os.path.exists(SCALERS_PATH):
        raise FileNotFoundError(f"Scalers file not found: {SCALERS_PATH}")
    
    scalers = np.load(SCALERS_PATH, allow_pickle=True)
    x_mu = scalers['x_mu'] if len(scalers['x_mu']) > 0 else None
    x_sig = scalers['x_sig'] if len(scalers['x_sig']) > 0 else None
    y_min = float(scalers['y_min'][0]) if len(scalers['y_min']) > 0 else None
    y_max = float(scalers['y_max'][0]) if len(scalers['y_max']) > 0 else None
    
    # Apply scaling to full dataset
    X = X_raw.copy()
    if STANDARDIZE_X and x_mu is not None and x_sig is not None:
        X = standardize_apply(X, x_mu, x_sig)
    
    # Load model
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {BEST_MODEL_PATH}")
    
    checkpoint = torch.load(BEST_MODEL_PATH, map_location='cpu')
    input_dim = checkpoint['input_dim']
    hidden = checkpoint['hidden']
    dropout = checkpoint['dropout']
    
    model = MLPRegressor(input_dim=input_dim, hidden=hidden, dropout=dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    model.to(device)
    
    # Make predictions
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        y_pred = model(X_tensor).cpu().numpy().reshape(-1)
    
    # Convert predictions back to physical units
    if SCALE_Y_TO_01 and y_min is not None and y_max is not None:
        y_pred_phys = minmax_invert(y_pred, y_min, y_max)
    else:
        y_pred_phys = y_pred
    
    # Compute metrics
    m = compute_metrics(y_phys, y_pred_phys)
    a_max = float(np.max(y_phys))
    a_target = P_EVAL_FRACTION_OF_MAX * a_max
    bm = band_metrics(y_phys, y_pred_phys, center=a_target, rel_width=BAND_REL_WIDTH)
    rel_percent_error = ((np.abs(y_pred_phys - y_phys) / np.abs(y_phys)) * 100.0)
    
    # Print metrics
    print("Full-dataset metrics (physical area units):")
    print(f"  RMSE={m['rmse']:.6g}  MAE={m['mae']:.6g}  R2={m['r2']:.6f}")
    print(f"  Relative % Error: mean={np.mean(rel_percent_error):.5f}%, std={np.std(rel_percent_error):.5f}%")
    if bm["band_n"] > 0:
        print(f"Band around A_TARGET={a_target:.6g} (±{BAND_REL_WIDTH*100:.1f}%):")
        print(f"  band_n={bm['band_n']}  band_RMSE={bm['band_rmse']:.6g}  band_relRMSE={bm['band_rel_rmse']*100:.3f}%")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Actual_Area': y_phys,
        'Predicted_Area': y_pred_phys,
        'Residual': y_pred_phys - y_phys,
        'Relative_Percent_Error': rel_percent_error,
        'Absolute_Error': np.abs(y_pred_phys - y_phys)
    })
    
    results_path = os.path.join(OUT_DIR, "validation_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved predictions for {len(results_df)} samples to: {results_path}")
    
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description='Train or predict with area model')
    parser.add_argument('--data_file', type=str, help='Path to parquet data file')
    parser.add_argument('--reload', '--predict-only', action='store_true', 
                       help='Reload trained model and predict on the full dataset (skip training)')
    args = parser.parse_args()
    
    if args.reload:
        load_model_and_predict(args.data_file)
        return
    
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    in_csv = args.data_file
    df = pd.read_parquet(in_csv)

    if "Area" not in df.columns:
        raise ValueError("Expected an 'area' column in the CSV.")

    y_phys = df["Area"].to_numpy(dtype=np.float64)

    # ycols = [c for c in df.columns if c.startswith("y_")]
    # if len(ycols) == 0:
    #     raise ValueError("Expected columns y_000 ... y_499.")
    # ycols = sorted(ycols)
    # X_raw = df[ycols].to_numpy(dtype=np.float64)

    X_raw = df.iloc[:, :500].astype('float64').values

    # Add noise to signal before scaling
    noise_std = 2*2.5 * 10**-5  # 5e-5
    # noise_std = 0.0
    noise = np.random.normal(0, noise_std, X_raw.shape)
    X_raw = X_raw + noise

    if not np.isfinite(X_raw).all():
        raise ValueError("Non-finite values found in y_*** columns.")
    if not np.isfinite(y_phys).all():
        raise ValueError("Non-finite values found in area column.")

    a_max = float(np.max(y_phys))
    a_target = P_EVAL_FRACTION_OF_MAX * a_max
    print(f"Data: N={len(df)}, bins={X_raw.shape[1]}, area_max={a_max:.6g} -> A_TARGET={a_target:.6g}")

    # Split
    X_train, X_val, y_train_phys, y_val_phys = train_test_split(
        X_raw, y_phys, test_size=VAL_FRACTION, random_state=SEED, shuffle=True
    )

    # Fit X scaler on train only
    x_mu = x_sig = None
    if STANDARDIZE_X:
        x_mu, x_sig = standardize_fit(X_train)
        X_train = standardize_apply(X_train, x_mu, x_sig)
        X_val = standardize_apply(X_val, x_mu, x_sig)
    else:
        X_train = X_train.copy()
        X_val = X_val.copy()

    # Fit y scaler on train only (minmax) to keep eval script behavior unchanged
    y_min = y_max = None
    if SCALE_Y_TO_01:
        y_min, y_max = minmax_fit(y_train_phys)
        y_train = minmax_apply(y_train_phys, y_min, y_max)
        y_val = minmax_apply(y_val_phys, y_min, y_max)
    else:
        y_train = y_train_phys.copy()
        y_val = y_val_phys.copy()

    # Build model (eval script will reconstruct it using hidden/dropout from checkpoint)
    model = MLPRegressor(input_dim=X_train.shape[1], hidden=HIDDEN, dropout=DROPOUT)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Save scalers (SAME keys as before)
    np.savez(
        SCALERS_PATH,
        x_mu=x_mu if x_mu is not None else np.array([]),
        x_sig=x_sig if x_sig is not None else np.array([]),
        y_min=np.array([y_min]) if y_min is not None else np.array([]),
        y_max=np.array([y_max]) if y_max is not None else np.array([]),
        # ycols=np.array(ycols, dtype=object),
    )

    # =========================
    # TRAIN
    # =========================
    if (len(HIDDEN) == 0) and USE_CLOSED_FORM_RIDGE:
        # Closed-form ridge solution for linear net
        w, b = fit_ridge_closed_form(X_train, y_train, RIDGE_ALPHA)

        # Assign to the single Linear layer: model.net[0]
        lin: nn.Linear = model.net[0]  # type: ignore
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(w.reshape(1, -1)).float())
            lin.bias.copy_(torch.tensor([b], dtype=torch.float32))

        print(f"Trained linear ridge (closed-form). RIDGE_ALPHA={RIDGE_ALPHA:g}")

    else:
        # Gradient-based fallback (if you choose hidden layers later)
        # Kept simple: MSE training
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
        model.to(device)

        # Create DataLoaders
        train_dataset = data.TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float().view(-1, 1)
        )
        val_dataset = data.TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).float().view(-1, 1)
        )
        train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
        val_loader = data.DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        # Learning rate scheduler
        scheduler = None
        if USE_LR_SCHEDULER:
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                opt,
                T_0=LR_SCHEDULER_T_0,
                T_mult=LR_SCHEDULER_T_MULT,
                eta_min=LR_SCHEDULER_ETA_MIN
            )

        # Track losses
        train_losses = []
        val_losses = []

        for epoch in range(1, MAX_EPOCHS + 1):
            # Training
            model.train()
            epoch_train_loss = 0.0
            num_batches = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(X_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                opt.step()
                epoch_train_loss += loss.item()
                num_batches += 1
            avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else 0.0
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            epoch_val_loss = 0.0
            num_val_batches = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    val_pred = model(X_batch)
                    val_loss = loss_fn(val_pred, y_batch)
                    epoch_val_loss += val_loss.item()
                    num_val_batches += 1
            avg_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else 0.0
            val_losses.append(avg_val_loss)

            # Update learning rate
            if scheduler is not None:
                scheduler.step()

            if epoch % 20 == 0:
                current_lr = opt.param_groups[0]['lr']
                print(f"Epoch {epoch:03d}/{MAX_EPOCHS} train_mse={avg_train_loss:.6g} val_mse={avg_val_loss:.6g} lr={current_lr:.2e}")

        # Plot and save loss curves
        plt.figure(figsize=(10, 6))
        plt.style.use('ggplot')
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', linewidth=2)
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=24)
        plt.ylabel('Loss (MSE)', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(LOSS_PLOT_PATH, dpi=600, bbox_inches='tight')
        print(f"Saved loss plot to: {LOSS_PLOT_PATH}")
        plt.close()

        losses = pd.DataFrame({
            'train_loss': train_losses,
            'val_loss': val_losses
        })
        losses.to_csv(f"{OUT_DIR}/losses.csv", index=False)

    # =========================
    # EVAL on validation
    # =========================
    model.eval()
    with torch.no_grad():
        # Get device from model parameters
        device = next(model.parameters()).device
        y_pred_val = model(torch.from_numpy(X_val).float().to(device)).cpu().numpy().reshape(-1)

    # Convert predictions back to physical
    if SCALE_Y_TO_01 and (y_min is not None) and (y_max is not None):
        y_pred_val_phys = minmax_invert(y_pred_val, y_min, y_max)
    else:
        y_pred_val_phys = y_pred_val

    # Global metrics
    m = compute_metrics(y_val_phys, y_pred_val_phys)
    # Band metrics near target
    bm = band_metrics(y_val_phys, y_pred_val_phys, center=a_target, rel_width=BAND_REL_WIDTH)

    # Relative percent error
    rel_percent_error = ((np.abs(y_pred_val_phys - y_val_phys) / np.abs(y_val_phys)) * 100.0)
    rel_percent_error_mean = float(np.mean(rel_percent_error))
    rel_percent_error_std = float(np.std(rel_percent_error))

    print("Validation metrics (physical area units):")
    print(f"  RMSE={m['rmse']:.6g}  MAE={m['mae']:.6g}  R2={m['r2']:.6f}")
    print(f"  Relative % Error: mean={rel_percent_error_mean}%, std={rel_percent_error_std}%")
    if bm["band_n"] > 0:
        print(f"Band around A_TARGET={a_target:.6g} (±{BAND_REL_WIDTH*100:.1f}%):")
        print(f"  band_n={bm['band_n']}  band_RMSE={bm['band_rmse']:.6g}  band_relRMSE={bm['band_rel_rmse']*100:.3f}%")
    else:
        print("Band: band_n=0 (unexpected; check your area distribution)")

    # Save validation results to CSV
    results_df = pd.DataFrame({
        'Actual_Area': y_val_phys,
        'Predicted_Area': y_pred_val_phys,
        'Residual': y_pred_val_phys - y_val_phys,
        'Relative_Percent_Error': rel_percent_error,
        'Absolute_Error': np.abs(y_pred_val_phys - y_val_phys)
    })
    results_path = os.path.join(OUT_DIR, "validation_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved validation predictions to: {results_path}")

    # Save model checkpoint (SAME keys as before)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": X_train.shape[1],
            "hidden": HIDDEN,
            "dropout": DROPOUT,
        },
        BEST_MODEL_PATH,
    )

    print("Saved best model to:", BEST_MODEL_PATH)
    print("Saved scalers to:", SCALERS_PATH)


if __name__ == "__main__":
    main()