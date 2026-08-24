"""
Load a trained pol_mlp model by version name and run predictions on a test file.

Usage (from ml/):
    python test.py --version Test_MLP --data data/20_25_500K.parquet
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch

from pol_mlp import SimpleFeedForward


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_weight_path(model_dir: str) -> str:
    pth = os.path.join(model_dir, "best_model.pth")
    if os.path.isfile(pth):
        return pth

    # Prefer the newest checkpoint when training is still writing .pth
    candidates = [
        os.path.join(model_dir, name)
        for name in os.listdir(model_dir)
        if name.startswith("best_model_checkpoint") and name.endswith(".ckpt")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No model weights found in {model_dir}. "
            "Expected best_model.pth or best_model_checkpoint*.ckpt"
        )
    return max(candidates, key=os.path.getmtime)


POLARIZATION_RANGES_PCT: list[tuple[float, float]] = [
    (3, 10),
    (10, 15),
    (15, 20),
    (20, 25),
    (25, 30),
    (30, 35),
    (35, 40),
    (40, 45),
    (45, 50),
    (50, 55),
]


def _in_polarization_range(abs_p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi == POLARIZATION_RANGES_PCT[-1][1]:
        return (lo <= abs_p) & (abs_p <= hi)
    return (lo <= abs_p) & (abs_p < hi)


def _summarize(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def polarization_range_stats(
    y_true_pct: np.ndarray,
    residuals: np.ndarray,
    rpe: np.ndarray,
) -> pd.DataFrame:
    abs_p = np.abs(y_true_pct)
    rows: list[dict[str, float | int | str]] = []

    for lo, hi in POLARIZATION_RANGES_PCT:
        mask = _in_polarization_range(abs_p, lo, hi)
        n = int(mask.sum())
        resid_stats = _summarize(residuals[mask])
        rpe_stats = _summarize(rpe[mask])
        rows.append({
            "polarization_range": f"{lo}-{hi}%",
            "n_samples": n,
            "residual_mean": resid_stats["mean"],
            "residual_median": resid_stats["median"],
            "residual_std": resid_stats["std"],
            "rpe_mean": rpe_stats["mean"],
            "rpe_median": rpe_stats["median"],
            "rpe_std": rpe_stats["std"],
        })

    return pd.DataFrame(rows)


def print_polarization_range_stats(stats_df: pd.DataFrame) -> None:
    print("\nTest metrics by polarization range (|actual| %):")
    for _, row in stats_df.iterrows():
        print(f"\n  {row['polarization_range']}  (n={row['n_samples']})")
        print(
            f"    Residuals — mean: {row['residual_mean']:.6f}, "
            f"median: {row['residual_median']:.6f}, "
            f"std: {row['residual_std']:.6f}"
        )
        print(
            f"    RPE       — mean: {row['rpe_mean']:.6f}, "
            f"median: {row['rpe_median']:.6f}, "
            f"std: {row['rpe_std']:.6f}"
        )


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    if model_path.endswith(".ckpt"):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        hparams = ckpt["hyper_parameters"]
        model = SimpleFeedForward(
            input_dim=hparams["input_dim"],
            hidden_dim=hparams["hidden_dim"],
        )
        state = {
            k.replace("model.", "", 1): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(state)
    else:
        state = torch.load(model_path, map_location=device, weights_only=True)
        if "trunk.0.weight" in state:
            hidden_dim, input_dim = state["trunk.0.weight"].shape
        elif "input_proj.weight" in state:
            hidden_dim, input_dim = state["input_proj.weight"].shape
        else:
            hidden_dim, input_dim = state["net.0.weight"].shape
        model = SimpleFeedForward(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict polarization with a trained MLP (by version name)."
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Model version name (e.g. Test_MLP). Loads Models/<version> and "
             "Model_Performance/<version>/<version>_scaler.pkl",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to testing parquet file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference (default: 256)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model before timing predictions",
    )
    args = parser.parse_args()

    version = args.version
    model_dir = f"Models/{version}"
    performance_dir = f"Model_Performance/{version}"
    scaler_path = f"{performance_dir}/{version}_scaler.pkl"

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")

    device = _device()
    print(f"Device: {device}")
    print(f"Version: {version}")
    print(f"Data: {args.data}")

    # --- load data (not timed) ---
    df = pd.read_parquet(args.data)
    signal_cols = df.columns[0:512]
    X_raw = df[signal_cols].values.astype("float32")
    target_names = [name for name in ("P", "Q") if name in df.columns]
    y_true = (
        df[target_names].values.astype("float32") if target_names else None
    )
    snr = df["SNR"].values.astype("float32") if "SNR" in df.columns else None
    n_samples = len(df)
    print(f"Loaded {n_samples} samples")

    # --- load scaler (not timed) ---
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X = scaler.transform(X_raw).astype("float32")
    print(f"Loaded scaler from {scaler_path}")

    # --- load / compile model (not timed) ---
    model_path = _resolve_weight_path(model_dir)
    print(f"Loading weights from {model_path}")
    model = load_model(model_path, device)

    if args.compile:
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    X_tensor = torch.from_numpy(X).to(device)

    # Warmup so compile / CUDA kernels are not counted in predict time
    with torch.no_grad():
        _ = model(X_tensor[: min(args.batch_size, n_samples)])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    # --- prediction only (timed) ---
    predictions = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n_samples, args.batch_size):
            end = min(start + args.batch_size, n_samples)
            pred = model(X_tensor[start:end])
            predictions.append(pred.cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    # Model returns [P, Q] per sample — keep (n, 2), do not flatten.
    y_pred = torch.cat(predictions, dim=0).numpy()
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    print("\n" + "=" * 60)
    print(f"Prediction time: {elapsed:.6f} s  ({n_samples / elapsed:.1f} samples/s)")
    print("=" * 60)

    y_pred_pct = y_pred * 100.0
    pred_names = ("P", "Q")[: y_pred_pct.shape[1]]
    out: dict[str, np.ndarray] = {
        f"Predicted_{name}": y_pred_pct[:, idx]
        for idx, name in enumerate(pred_names)
    }

    if y_true is not None:
        y_true_pct = y_true * 100.0
        for idx, name in enumerate(target_names):
            if idx >= y_pred_pct.shape[1]:
                break
            y_t = y_true_pct[:, idx]
            y_h = y_pred_pct[:, idx]
            residuals = y_t - y_h
            rpe = np.abs(y_h - y_t) / np.abs(y_t) * 100.0
            mse = float(np.mean((y_t - y_h) ** 2))
            mae = float(np.mean(np.abs(y_t - y_h)))
            rmse = float(np.sqrt(mse))
            print(f"\nTest metrics ({name} %):")
            print(f"  MSE:      {mse:.6f}")
            print(f"  MAE:      {mae:.6f}")
            print(f"  RMSE:     {rmse:.6f}")
            print(f"  Mean RPE: {rpe.mean():.5f}%")

            out[f"Actual_{name}"] = y_t
            out[f"Residuals_{name}"] = residuals
            out[f"RPE_{name}"] = rpe

            # Polarization-range breakdown is only meaningful for P.
            if name == "P":
                range_stats = polarization_range_stats(y_t, residuals, rpe)
                print_polarization_range_stats(range_stats)
                range_stats_path = f"{performance_dir}/{version}_test_range_stats.csv"
                range_stats.to_csv(range_stats_path, index=False)
                print(f"\nSaved range statistics to {range_stats_path}")

    if snr is not None:
        out["SNR"] = snr

    os.makedirs(performance_dir, exist_ok=True)
    out_path = f"{performance_dir}/{version}_test_predictions.csv"
    pd.DataFrame(out).to_csv(out_path, index=False)
    print(f"\nSaved predictions to {out_path}")


if __name__ == "__main__":
    main()
