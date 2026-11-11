"""
Does prediction on selected RGC dataset.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pol_mlp import DEFAULT_CHECKPOINT, load_model_from_checkpoint

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


def load_records(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_model(checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model, ckpt = load_model_from_checkpoint(str(checkpoint_path), device=device)
    model.eval()
    return model, ckpt


def load_scaler(scaler_path):
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    with scaler_path.open("rb") as f:
        return pickle.load(f)


def _match_bins(spectra, n_bins):
    spectra = np.asarray(spectra, dtype=np.float32)
    if spectra.shape[1] == n_bins:
        return spectra
    src = np.linspace(0.0, 1.0, spectra.shape[1])
    dst = np.linspace(0.0, 1.0, n_bins)
    return np.vstack([np.interp(dst, src, row) for row in spectra]).astype(np.float32)


def records_to_features(records, n_bins, scaler=None, ps_mean=None, ps_std=None, signal_key=SIGNAL_KEY):
    spectra = []
    exp_p = []
    meta = []

    for idx, record in enumerate(records):
        signal = record.get(signal_key)
        if signal is None:
            raise KeyError(f"Event {idx} missing '{signal_key}'")
        spectra.append(np.asarray(signal, dtype=np.float32))
        exp_p.append(record["pol"])
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

    X = _match_bins(np.stack(spectra, axis=0), n_bins)
    if ps_mean is not None:
        X = (X - np.float32(ps_mean)) / np.float32(ps_std)
    elif scaler is not None:
        X = scaler.transform(X)
    y_exp = np.asarray(exp_p, dtype=np.float32)
    return X, y_exp, meta


@torch.no_grad()
def predict(model, X, batch_size, device):
    preds = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start : start + batch_size]).to(device)
        out = model(batch).cpu().numpy()
        preds.append(out[:, 0] if out.ndim == 2 else out)
    return np.concatenate(preds, axis=0)


def compute_summary(exp_frac, pred_frac):
    exp_pct = exp_frac * 100.0
    pred_pct = pred_frac * 100.0
    residuals = exp_pct - pred_pct

    return {
        "n_events": len(residuals),
        "residual_stats_pct": {
            "mean": np.mean(residuals).item(),
            "std": np.std(residuals).item(),
            "median": np.median(residuals).item(),
            "min": np.min(residuals).item(),
            "max": np.max(residuals).item(),
            "mean_abs": np.mean(np.abs(residuals)).item(),
            "rmse": np.sqrt(np.mean(residuals ** 2)).item(),
            "max_abs": np.max(np.abs(residuals)).item(),
        },
    }


def save_results(output_dir, stem, meta, exp_frac, pred_frac, summary):
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


def print_summary(summary):
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


def parse_args():
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
        default=Path(DEFAULT_CHECKPOINT),
        help="Checkpoint path (default: ml/pol_mlp.pth)",
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


def resolve_device(device_arg):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def main():
    args = parse_args()
    version = args.version
    performance_dir = ML_DIR / "Model_Performance" / version
    checkpoint_path = args.checkpoint
    scaler_path = args.scaler_path or (performance_dir / f"{version}_scaler.pkl")
    output_dir = args.output_dir or (performance_dir / "data_test")
    stem = args.data_path.stem

    device = resolve_device(args.device)
    print(f"Loading model from {checkpoint_path}")
    print(f"Loading events from {args.data_path}")
    print(f"Using device: {device}")

    model, ckpt = load_model(checkpoint_path, device)
    net = model.model
    n_bins = int(getattr(net, "num_bins", ckpt.get("input_dim", 512)))
    stats = ckpt.get("stats") or {}
    if "ps_mean" in stats and "ps_std" in stats:
        print(
            f"Z-scoring spectra with checkpoint stats "
            f"(ps_mean={stats['ps_mean']}, ps_std={stats['ps_std']})",
        )
        scaler = None
        ps_mean, ps_std = stats["ps_mean"], stats["ps_std"]
    else:
        print(f"Loading scaler from {scaler_path}")
        scaler = load_scaler(scaler_path)
        ps_mean = ps_std = None
    records = load_records(args.data_path)

    X, y_exp, meta = records_to_features(
        records, n_bins, scaler=scaler, ps_mean=ps_mean, ps_std=ps_std, signal_key=SIGNAL_KEY,
    )
    y_pred = predict(model, X, batch_size=args.batch_size, device=device)
    summary = compute_summary(y_exp, y_pred)

    print_summary(summary)
    save_results(output_dir, stem, meta, y_exp, y_pred, summary)


if __name__ == "__main__":
    main()
