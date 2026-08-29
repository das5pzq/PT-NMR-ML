"""
Does test on sample data that was fitted to to generate data (preliminary work).
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pol_mlp import FFLightningModule

ML_DIR = Path(__file__).resolve().parent
REPO_ROOT = ML_DIR.parent
DEFAULT_VERSION = "RGC_MLP_V1"
DEFAULT_DATA = (
    REPO_ROOT
    / "data_creation"
    / "data-test"
    / "2022-09-23_00-54-02__2022-09-23_14-13-02.txt"
)
SIGNAL_KEY = "phase"
N_BINS = 512


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_model(checkpoint_path: Path, device: torch.device) -> FFLightningModule:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = FFLightningModule.load_from_checkpoint(str(checkpoint_path))
    model.eval()
    model.to(device)
    return model


def load_scaler(scaler_path: Path):
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    with scaler_path.open("rb") as f:
        return pickle.load(f)


def records_to_features(
    records: list[dict],
    scaler,
    signal_key: str = SIGNAL_KEY,
    n_bins: int = N_BINS,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    spectra = []
    exp_p = []
    meta = []

    for idx, record in enumerate(records):
        signal = record.get(signal_key)
        if signal is None:
            raise KeyError(f"Event {idx} missing '{signal_key}'")
        signal = np.asarray(signal, dtype=np.float32)
        if signal.shape[0] != n_bins:
            raise ValueError(
                f"Event {idx} has {signal.shape[0]} bins; expected {n_bins}"
            )
        spectra.append(signal)
        exp_p.append(float(record["pol"]))
        meta.append(
            {
                "event_index": idx,
                "num": record.get("num"),
                "start_time": record.get("start_time"),
                "stop_time": record.get("stop_time"),
                "cc": record.get("cc"),
                "area": record.get("area"),
            }
        )

    X = scaler.transform(np.stack(spectra, axis=0))
    y_exp = np.asarray(exp_p, dtype=np.float32)
    return X, y_exp, meta


@torch.no_grad()
def predict(model: FFLightningModule, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    preds = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start : start + batch_size]).to(device)
        preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds, axis=0).reshape(-1)


def compute_summary(exp_frac: np.ndarray, pred_frac: np.ndarray) -> dict:
    exp_pct = exp_frac * 100.0
    pred_pct = pred_frac * 100.0
    residuals = exp_pct - pred_pct

    return {
        "n_events": int(len(residuals)),
        "residual_stats_pct": {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "median": float(np.median(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
            "mean_abs": float(np.mean(np.abs(residuals))),
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "max_abs": float(np.max(np.abs(residuals))),
        },
    }


def save_results(
    output_dir: Path,
    stem: str,
    meta: list[dict],
    exp_frac: np.ndarray,
    pred_frac: np.ndarray,
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_pct = exp_frac * 100.0
    pred_pct = pred_frac * 100.0
    residuals = exp_pct - pred_pct

    rows = []
    for i, event_meta in enumerate(meta):
        rows.append(
            {
                **event_meta,
                "exp_p": exp_pct[i],
                "Predicted_pct": pred_pct[i],
                "Residual_pct": residuals[i],
            }
        )

    results_csv = output_dir / f"{stem}_results.csv"
    pd.DataFrame(rows).to_csv(results_csv, index=False)

    summary_path = output_dir / f"{stem}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"Saved per-event results to {results_csv}")
    print(f"Saved summary to {summary_path}")


def print_summary(summary: dict) -> None:
    stats = summary["residual_stats_pct"]
    print("\n" + "=" * 60)
    print(f"Data-test inference ({summary['n_events']} events)")
    print("Residuals = exp_p - Predicted_pct  (pct points)")
    print("=" * 60)
    print(f"  Mean:                {stats['mean']:.6f}")
    print(f"  Std:                 {stats['std']:.6f}")
    print(f"  Median:              {stats['median']:.6f}")
    print(f"  Min:                 {stats['min']:.6f}")
    print(f"  Max:                 {stats['max']:.6f}")
    print(f"  Mean |residual|:     {stats['mean_abs']:.6f}")
    print(f"  RMSE:                {stats['rmse']:.6f}")
    print(f"  Max |residual|:      {stats['max_abs']:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA,
        help="JSONL data-test file with 'phase' and 'pol' fields",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help="Model/version subdirectory under Models/ and Model_Performance/",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=None,
        help="Optional explicit MinMaxScaler pickle path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV/JSON outputs",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Inference device",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    version = args.version
    performance_dir = ML_DIR / "Model_Performance" / version
    model_dir = ML_DIR / "Models" / version

    checkpoint_path = args.checkpoint or (model_dir / "best_model_checkpoint.ckpt")
    scaler_path = args.scaler_path or (performance_dir / f"{version}_scaler.pkl")
    output_dir = args.output_dir or (performance_dir / "data_test")
    stem = args.data_path.stem

    device = resolve_device(args.device)
    print(f"Loading model from {checkpoint_path}")
    print(f"Loading scaler from {scaler_path}")
    print(f"Loading events from {args.data_path}")
    print(f"Using device: {device}")

    model = load_model(checkpoint_path, device)
    scaler = load_scaler(scaler_path)
    records = load_records(args.data_path)

    X, y_exp, meta = records_to_features(records, scaler, signal_key=SIGNAL_KEY)
    y_pred = predict(model, X, batch_size=args.batch_size, device=device)
    summary = compute_summary(y_exp, y_pred)

    print_summary(summary)
    save_results(output_dir, stem, meta, y_exp, y_pred, summary)


if __name__ == "__main__":
    main()
