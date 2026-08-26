import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import curve_fit
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics.baseline import Baseline, DEFAULT_CIRC_CONSTS

PATH = "data-test"
OUT_YAML = "fitting/baseline_fits_single_event.yaml"
OUT_STATS_YAML = "fitting/baseline_fit_stats_single_event.yaml"
OUT_STATS_DIR = "fitting/baseline_fit_stats_single_event"
VOLTAGE_KEY = "baseline"
SPECIES = "deuteron"

REUSE_REFERENCE_FIT_FOR_ALL_EVENTS = True
FIX_CIRCUIT_PARAMS_FROM_REFERENCE = True
REFERENCE_EVENT_INDEX = 10

FIT_PARAM_NAMES = (
    "U",
    "Cknob",
    "eta",
    "trim",
    "Cstray",
    "phi_const",
    "DC_offset",
)
CIRCUIT_PARAM_NAMES = (
    "L0",
    "Rcoil",
    "R",
    "R1",
    "r",
    "alpha",
    "beta1",
    "Z_cable",
    "D",
    "M",
    "delta_C",
    "delta_phi",
    "delta_phase",
    "delta_l",
)

PARAM_NAMES = (
    "U",
    "Cknob",
    "eta",
    "trim",
    "Cstray",
    "phi_const",
    "DC_offset",
    "L0",
    "Rcoil",
    "R",
    "R1",
    "r",
    "alpha",
    "beta1",
    "Z_cable",
    "D",
    "M",
    "delta_C",
    "delta_phi",
    "delta_phase",
    "delta_l",
)

N_EXAMPLE_PLOTS = 4


def baseline_fit(f, U, Cknob, eta, trim, Cstray, phi_const, DC_offset, *circ_consts):
    return Baseline(
        f, U, Cknob, eta, trim, Cstray, phi_const, DC_offset, SPECIES, *circ_consts
    )


def load_records(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _default_fit_p0(baseline: np.ndarray) -> dict[str, float]:
    return {
        "U": 5.0,
        "Cknob": 0.404,
        "eta": 1.04e-2,
        "trim": 0.5,
        "Cstray": 1e-20,
        "phi_const": 0.0,
        "DC_offset": float(np.mean(baseline)),
    }


def _assemble_params(
    fit_params: dict[str, float], circuit_params: dict[str, float]
) -> np.ndarray:
    return np.array(
        [
            float(fit_params[name]) if name in FIT_PARAM_NAMES else float(circuit_params[name])
            for name in PARAM_NAMES
        ],
        dtype=np.float64,
    )


def fit_baseline(
    freq_mhz: np.ndarray,
    baseline: np.ndarray,
    fixed_circuit_params: dict[str, float] | None = None,
    fit_p0: dict[str, float] | None = None,
) -> tuple[np.ndarray, float, float]:
    default_fit_p0 = _default_fit_p0(baseline)
    p0 = tuple(default_fit_p0[name] for name in FIT_PARAM_NAMES) + DEFAULT_CIRC_CONSTS

    if fixed_circuit_params is not None:
        start = fit_p0 if fit_p0 is not None else default_fit_p0

        def model_free(
            f: np.ndarray,
            U: float,
            Cknob: float,
            eta: float,
            trim: float,
            Cstray: float,
            phi_const: float,
            DC_offset: float,
        ) -> np.ndarray:
            fit_params = {
                "U": U,
                "Cknob": Cknob,
                "eta": eta,
                "trim": trim,
                "Cstray": Cstray,
                "phi_const": phi_const,
                "DC_offset": DC_offset,
            }
            return baseline_fit(f, *_assemble_params(fit_params, fixed_circuit_params))

        free_p0 = tuple(start[name] for name in FIT_PARAM_NAMES)
        free_fit, _ = curve_fit(
            model_free,
            freq_mhz,
            baseline,
            p0=free_p0,
            maxfev=500000,
        )
        fit_params = dict(zip(FIT_PARAM_NAMES, free_fit, strict=True))
        params = _assemble_params(fit_params, fixed_circuit_params)
    else:
        params, _ = curve_fit(
            baseline_fit,
            freq_mhz,
            baseline,
            p0=p0,
            maxfev=500000,
        )
    return params, *_fit_errors(freq_mhz, baseline, params)


def _fit_errors(
    freq_mhz: np.ndarray,
    baseline: np.ndarray,
    params: np.ndarray,
) -> tuple[float, float]:
    residual = baseline - baseline_fit(freq_mhz, *params)
    rmse = float(np.sqrt(np.mean(residual**2)))
    peak_amp = float(np.max(np.abs(baseline)))
    nrmse = float(rmse / peak_amp) if peak_amp > 0.0 else float("inf")
    return nrmse, rmse


def _percentile_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    qs = np.percentile(values, [0, 25, 50, 75, 100])
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(qs[0]),
        "p25": float(qs[1]),
        "median": float(qs[2]),
        "p75": float(qs[3]),
        "max": float(qs[4]),
    }


def collect_fit_rows(results: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for filename, file_info in results.items():
        for event_id, event in file_info["events"].items():
            row = {
                "filename": filename,
                "event_id": int(event_id),
                "nrmse": float(event["nrmse"]),
                "rmse": float(event["rmse"]),
            }
            for name in PARAM_NAMES:
                row[name] = float(event["params"][name])
            rows.append(row)
    return rows


def summarize_fits(results: dict[str, dict], n_fitted: int, n_skipped: int) -> dict:
    rows = collect_fit_rows(results)
    if not rows:
        return {"n_files": 0, "n_events_fitted": 0, "n_events_skipped": n_skipped}

    summary: dict = {
        "n_files": len(results),
        "n_events_fitted": n_fitted,
        "n_events_skipped": n_skipped,
        "reuse_reference_fit_for_all_events": REUSE_REFERENCE_FIT_FOR_ALL_EVENTS,
        "fix_circuit_params_from_reference": (
            False if REUSE_REFERENCE_FIT_FOR_ALL_EVENTS else FIX_CIRCUIT_PARAMS_FROM_REFERENCE
        ),
        "reference_event_index": REFERENCE_EVENT_INDEX,
        "fixed_circuit_params": list(CIRCUIT_PARAM_NAMES),
        "free_fit_params": list(FIT_PARAM_NAMES),
        "nrmse": _percentile_stats(np.asarray([r["nrmse"] for r in rows], dtype=np.float64)),
        "rmse": _percentile_stats(np.asarray([r["rmse"] for r in rows], dtype=np.float64)),
        "params": {
            name: _percentile_stats(np.asarray([r[name] for r in rows], dtype=np.float64))
            for name in PARAM_NAMES
        },
        "primary_params": {
            name: _percentile_stats(np.asarray([r[name] for r in rows], dtype=np.float64))
            for name in FIT_PARAM_NAMES
        },
    }
    return summary


def print_summary(summary: dict) -> None:
    print("\n=== Baseline fit summary ===")
    ref_idx = summary.get("reference_event_index", REFERENCE_EVENT_INDEX)
    if summary.get("reuse_reference_fit_for_all_events"):
        print(
            f"mode:    reuse full fit from event {ref_idx} "
            f"(1-based event {ref_idx + 1}) for all events in each file"
        )
    elif summary.get("fix_circuit_params_from_reference"):
        print(
            f"mode:    fixed circuit params from event {ref_idx} "
            f"(1-based event {ref_idx + 1}); fit {', '.join(FIT_PARAM_NAMES)}"
        )
    print(f"files:   {summary.get('n_files', 0)}")
    print(f"fitted:  {summary.get('n_events_fitted', 0)}")
    print(f"skipped: {summary.get('n_events_skipped', 0)}")
    if nrmse := summary.get("nrmse"):
        print(f"nrmse:   median={nrmse['median']:.4g}  mean={nrmse['mean']:.4g}")
    if rmse := summary.get("rmse"):
        print(f"rmse:    median={rmse['median']:.4g}  mean={rmse['mean']:.4g}")
    if cknob := summary.get("primary_params", {}).get("Cknob"):
        print(f"Cknob:   median={cknob['median']:.4g}  std={cknob['std']:.4g}")


def plot_fit_statistics(results: dict[str, dict], out_dir: Path) -> list[Path]:
    rows = collect_fit_rows(results)
    if not rows:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    nrmse = np.asarray([r["nrmse"] for r in rows], dtype=np.float64)
    rmse = np.asarray([r["rmse"] for r in rows], dtype=np.float64)
    cknob = np.asarray([r["Cknob"] for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(nrmse, bins=50, color="steelblue", edgecolor="white", alpha=0.9)
    axes[0, 0].axvline(np.median(nrmse), color="crimson", ls="--", lw=1.2, label="median")
    axes[0, 0].set_title("NRMSE distribution")
    axes[0, 0].legend()

    axes[0, 1].hist(rmse, bins=50, color="darkorange", edgecolor="white", alpha=0.9)
    axes[0, 1].axvline(np.median(rmse), color="crimson", ls="--", lw=1.2, label="median")
    axes[0, 1].set_title("RMSE distribution")
    axes[0, 1].legend()

    axes[1, 0].hist(cknob, bins=50, color="teal", edgecolor="white", alpha=0.9)
    axes[1, 0].set_title("Cknob distribution")

    axes[1, 1].scatter(cknob, nrmse, s=8, alpha=0.35, c="slateblue")
    axes[1, 1].set_xlabel("Cknob")
    axes[1, 1].set_ylabel("NRMSE")
    axes[1, 1].set_title("NRMSE vs Cknob")

    fig.tight_layout()
    path = out_dir / "baseline_fit_statistics.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    n_primary = len(FIT_PARAM_NAMES)
    n_cols = 3
    n_rows = int(np.ceil(n_primary / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.2 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes_flat, FIT_PARAM_NAMES, strict=False):
        values = np.asarray([r[name] for r in rows], dtype=np.float64)
        ax.hist(values, bins=40, color="0.45", edgecolor="white", alpha=0.9)
        ax.axvline(np.median(values), color="crimson", ls="--", lw=1.0)
        ax.set_title(name)
    for ax in axes_flat[n_primary:]:
        ax.set_axis_off()
    fig.tight_layout()
    path = out_dir / "baseline_param_histograms.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    return written


def plot_example_fits(
    results: dict[str, dict],
    data_dir: Path,
    out_dir: Path,
    n_examples: int = N_EXAMPLE_PLOTS,
) -> Path | None:
    rows = collect_fit_rows(results)
    if not rows:
        return None

    rows_sorted = sorted(rows, key=lambda r: r["nrmse"])
    n = len(rows_sorted)
    pick_idxs = sorted({0, n // 4, n // 2, n - 1})[:n_examples]
    picks = [rows_sorted[i] for i in pick_idxs]

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        n_examples,
        2,
        figsize=(12, 3.2 * n_examples),
        sharex=True,
        gridspec_kw={"width_ratios": [2.4, 1.0]},
    )
    if n_examples == 1:
        axes = np.array([axes])

    for panel_i, row in enumerate(picks):
        ax = axes[panel_i, 0]
        axr = axes[panel_i, 1]

        records = load_records(str(data_dir / row["filename"]))
        freq_mhz = np.asarray(records[0]["freq_list"], dtype=np.float64)
        signal = np.asarray(records[row["event_id"]][VOLTAGE_KEY], dtype=np.float64)
        params = np.asarray([row[name] for name in PARAM_NAMES], dtype=np.float64)
        fitted = baseline_fit(freq_mhz, *params)
        residual = signal - fitted

        ax.plot(freq_mhz, signal, color="0.35", lw=0.8, label="data")
        ax.plot(freq_mhz, fitted, color="darkorange", lw=1.4, ls="--", label="fit")
        ax.set_ylabel("V")
        ax.set_title(
            f"evt {row['event_id']} | NRMSE={row['nrmse']:.2e} | Cknob={row['Cknob']:.4f}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3)
        if panel_i == 0:
            ax.legend(loc="upper right", fontsize=8)

        axr.plot(freq_mhz, residual, color="0.25", lw=0.7)
        axr.axhline(0.0, color="0.4", lw=0.8)
        axr.set_ylabel("res.")
        axr.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Frequency (MHz)")
    axes[-1, 1].set_xlabel("Frequency (MHz)")
    fig.tight_layout()
    path = out_dir / "baseline_fit_examples.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    data_dir = Path(PATH)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir.resolve()}")

    results: dict[str, dict] = {}
    n_fitted = 0
    n_skipped = 0
    txt_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".txt"))

    for filename in tqdm(txt_files, desc="Files"):
        records = load_records(str(data_dir / filename))
        if not records or VOLTAGE_KEY not in records[0]:
            continue

        freq_mhz = np.asarray(records[0]["freq_list"], dtype=np.float64)
        if freq_mhz.ndim != 1 or len(freq_mhz) == 0:
            continue

        file_events: dict[int, dict] = {}
        reference_circuit_params: dict[str, float] | None = None
        reference_fit_p0: dict[str, float] | None = None
        shared_params: np.ndarray | None = None

        for index, record in tqdm(enumerate(records), desc="Events", leave=False):
            if VOLTAGE_KEY not in record:
                n_skipped += 1
                continue

            signal = np.asarray(record[VOLTAGE_KEY], dtype=np.float64)
            if signal.ndim != 1 or len(signal) != len(freq_mhz) or not np.any(signal):
                n_skipped += 1
                continue

            if REUSE_REFERENCE_FIT_FOR_ALL_EVENTS:
                if index == REFERENCE_EVENT_INDEX:
                    try:
                        shared_params, nrmse, rmse = fit_baseline(
                            freq_mhz, signal
                        )
                    except Exception:
                        n_skipped += 1
                        continue
                    params = shared_params
                elif shared_params is not None:
                    params = shared_params
                    nrmse, rmse = _fit_errors(freq_mhz, signal, params)
                else:
                    n_skipped += 1
                    continue
            else:
                fixed_circuit_params = None
                fit_p0 = None
                if FIX_CIRCUIT_PARAMS_FROM_REFERENCE:
                    if index == REFERENCE_EVENT_INDEX:
                        fixed_circuit_params = None
                    elif reference_circuit_params is not None:
                        fixed_circuit_params = reference_circuit_params
                        fit_p0 = reference_fit_p0
                    else:
                        n_skipped += 1
                        continue

                try:
                    params, nrmse, rmse = fit_baseline(
                        freq_mhz,
                        signal,
                        fixed_circuit_params=fixed_circuit_params,
                        fit_p0=fit_p0,
                    )
                except Exception:
                    n_skipped += 1
                    continue

                if FIX_CIRCUIT_PARAMS_FROM_REFERENCE and index == REFERENCE_EVENT_INDEX:
                    param_dict = {
                        name: float(value)
                        for name, value in zip(PARAM_NAMES, params, strict=True)
                    }
                    reference_circuit_params = {
                        name: param_dict[name] for name in CIRCUIT_PARAM_NAMES
                    }
                    reference_fit_p0 = {
                        name: param_dict[name] for name in FIT_PARAM_NAMES
                    }

            file_events[index] = {
                "nrmse": nrmse,
                "rmse": rmse,
                "params": {
                    name: float(value) for name, value in zip(PARAM_NAMES, params, strict=True)
                },
            }
            n_fitted += 1

        if file_events:
            results[filename] = {
                "species": SPECIES,
                "n_bins": int(len(freq_mhz)),
                "freq_min_mhz": float(freq_mhz[0]),
                "freq_max_mhz": float(freq_mhz[-1]),
                "n_events_fitted": len(file_events),
                "events": file_events,
            }

    out_path = Path(OUT_YAML)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(results, f, sort_keys=False, default_flow_style=False)

    summary = summarize_fits(results, n_fitted, n_skipped)
    stats_path = Path(OUT_STATS_YAML)
    with open(stats_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, default_flow_style=False)
    print_summary(summary)
    print(f"Wrote fits to {out_path.resolve()}")
    print(f"Wrote stats to {stats_path.resolve()}")

    stats_dir = Path(OUT_STATS_DIR)
    plot_paths = plot_fit_statistics(results, stats_dir)
    if example_path := plot_example_fits(results, data_dir, stats_dir):
        plot_paths.append(example_path)
    for plot_path in plot_paths:
        print(f"Wrote plot to {plot_path.resolve()}")


if __name__ == "__main__":
    main()
