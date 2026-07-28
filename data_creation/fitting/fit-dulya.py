import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics.Lineshape import DulyaFit, QmeterGain

PATH = "data-d"
OUT_YAML = "fitting/dulya_fits.yaml"
OUT_STATS_YAML = "fitting/dulya_fit_stats.yaml"
OUT_STATS_DIR = "fitting/dulya_fit_stats"

VOLTAGE_KEY = "fitsub"

CENTER_MHZ = 32.68
HALF_WIDTH_MHZ = 0.075  # fallback if peak finding fails

EDGE_FRACTION = 0.15  # outer fraction of bins on each side for wing polynomial
POLYNOMIAL_DEGREE = 3

MIN_MODEL_SNR = 3.0
MAX_NRMSE = 0.05
REQUIRE_DOUBLET = True
MAX_NFEV = 500

# File stores pol = cc * area. DulyaFit uses sum = P * factor, so pass factor = 1/cc.
ETA_BOUNDS = (0.001, 0.3)
PHI_BOUNDS = (0.0, 2 * np.pi)
G_BOUNDS = (0.001, 0.15)
XI_BOUNDS = (-1.0, 1.0)
HALF_WIDTH_BOUNDS = (0.04, 0.12)

ETA0 = 0.0104
PHI0 = 6.1319
G0 = 0.05
XI0 = 0.0

# P and scaling_factor (file cc) are fixed; remaining params are free.
PARAM_NAMES = ("P", "scaling_factor", "eta", "phi", "g", "xi", "half_width_mhz")
N_EXAMPLE_PLOTS = 4


def dulya_model(
    freq_mhz: np.ndarray,
    p: float,
    scaling_factor: float,
    eta: float,
    phi: float,
    g: float,
    xi: float,
    half_width_mhz: float,
    center_mhz: float = CENTER_MHZ,
) -> np.ndarray:

    half_width = max(half_width_mhz, 1e-6)
    p_eff = float(p)
    if abs(p_eff) < 1e-4:
        p_eff = 1e-4 if p_eff >= 0.0 else -1e-4
    cc = float(scaling_factor)
    freq_mhz = np.asarray(freq_mhz, dtype=np.float64)
    x_eff = freq_mhz - center_mhz
    x = x_eff / half_width
    shape = DulyaFit(x, p_eff, 1.0 / cc, eta, phi, g)
    gain = QmeterGain(x_eff, half_width, xi)
    return shape * gain


def load_records(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def wing_mask(n_bins: int) -> np.ndarray:
    n_edge = max(POLYNOMIAL_DEGREE + 1, int(n_bins * EDGE_FRACTION))
    mask = np.zeros(n_bins, dtype=bool)
    mask[:n_edge] = True
    mask[-n_edge:] = True
    return mask


def detrend_wings(freq_mhz: np.ndarray, signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    coeffs = np.polyfit(freq_mhz[mask], signal[mask], deg=POLYNOMIAL_DEGREE)
    return signal - np.polyval(coeffs, freq_mhz)


def amplitude_sign(signal_detrended: np.ndarray) -> float:
    idx = int(np.argmax(np.abs(signal_detrended)))
    s = float(np.sign(signal_detrended[idx]))
    return 1.0 if s == 0.0 else s


def find_doublet_peaks(
    freq_mhz: np.ndarray,
    signal_detrended: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Peak indices after flipping so horns are positive regardless of polarity."""
    amp_sign = amplitude_sign(signal_detrended)
    profile = amp_sign * signal_detrended
    height = 0.35 * float(np.max(profile))
    distance = max(8, len(freq_mhz) // 40)
    peaks, _ = find_peaks(profile, height=height, distance=distance)
    return peaks, profile


def estimate_doublet(
    freq_mhz: np.ndarray,
    signal_detrended: np.ndarray,
) -> tuple[float, float, bool]:
    """Estimate ``(center_mhz, half_width_mhz, found_doublet)`` from the two peaks."""
    peaks, profile = find_doublet_peaks(freq_mhz, signal_detrended)
    if len(peaks) < 2:
        return CENTER_MHZ, HALF_WIDTH_MHZ, False

    top2 = np.sort(peaks[np.argsort(profile[peaks])[-2:]])
    f_lo = float(freq_mhz[top2[0]])
    f_hi = float(freq_mhz[top2[1]])
    sep = f_hi - f_lo
    if sep < 2 * HALF_WIDTH_BOUNDS[0] or sep > 2 * HALF_WIDTH_BOUNDS[1]:
        return CENTER_MHZ, HALF_WIDTH_MHZ, False
    return 0.5 * (f_lo + f_hi), 0.5 * sep, True


def _fit_dulya_once(
    freq_mhz: np.ndarray,
    signal_detrended: np.ndarray,
    p_fixed: float,
    scale_fixed: float,
    mask: np.ndarray,
    center_mhz: float,
    half_width0: float,
) -> tuple[np.ndarray, float, float]:
    """Returns ``(params, nrmse, model_snr)`` with fixed P and cc."""
    p_fixed = float(p_fixed)
    scale_fixed = float(scale_fixed)

    hw0 = float(np.clip(half_width0, *HALF_WIDTH_BOUNDS))
    p0 = np.array(
        [ETA0, PHI0, G0, XI0, hw0],
        dtype=np.float64,
    )
    lower = [
        ETA_BOUNDS[0],
        PHI_BOUNDS[0],
        G_BOUNDS[0],
        XI_BOUNDS[0],
        HALF_WIDTH_BOUNDS[0],
    ]
    upper = [
        ETA_BOUNDS[1],
        PHI_BOUNDS[1],
        G_BOUNDS[1],
        XI_BOUNDS[1],
        HALF_WIDTH_BOUNDS[1],
    ]

    sigma = float(np.std(signal_detrended[mask]))
    if sigma <= 0.0:
        raise RuntimeError("wing noise estimate is zero")

    left = (freq_mhz < center_mhz) & (np.abs(freq_mhz - center_mhz) < 2.5 * hw0)
    right = (freq_mhz >= center_mhz) & (np.abs(freq_mhz - center_mhz) < 2.5 * hw0)
    weights = np.ones_like(signal_detrended)
    if np.any(left):
        weights[left] = 2.0
    if np.any(right):
        weights[right] = 2.0

    def residuals(theta: np.ndarray) -> np.ndarray:
        model = dulya_model(
            freq_mhz,
            p_fixed,
            scale_fixed,
            *theta,
            center_mhz=center_mhz,
        )
        return weights * (signal_detrended - model) / sigma

    result = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        max_nfev=MAX_NFEV,
    )
    free = np.asarray(result.x, dtype=np.float64)
    params = np.concatenate(([p_fixed, scale_fixed], free))
    model = dulya_model(freq_mhz, *params, center_mhz=center_mhz)
    residual = signal_detrended - model
    rmse = float(np.sqrt(np.mean(residual**2)))
    peak_amp = float(np.max(np.abs(signal_detrended)))
    nrmse = float(rmse / peak_amp) if peak_amp > 0.0 else float("inf")
    model_amp = float(np.max(np.abs(model)))
    model_snr = float(model_amp / sigma) if model_amp > 0.0 else 0.0
    return params, nrmse, model_snr


def fit_dulya(
    freq_mhz: np.ndarray,
    signal: np.ndarray,
    p_fixed: float,
    scale_fixed: float,
    mask: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:

    signal_detrended = detrend_wings(freq_mhz, signal, mask)
    center_mhz, half_width0, found_doublet = estimate_doublet(
        freq_mhz, signal_detrended
    )

    if REQUIRE_DOUBLET and not found_doublet:
        raise RuntimeError("no Pake doublet detected (likely sinusoid/qcurve junk)")

    params, nrmse, model_snr = _fit_dulya_once(
        freq_mhz,
        signal_detrended,
        p_fixed,
        scale_fixed,
        mask,
        center_mhz,
        half_width0,
    )
    return params, nrmse, model_snr, center_mhz


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
                "model_snr": float(event.get("model_snr", np.nan)),
                "center_mhz": float(event["center_mhz"]),
                "pol_true": float(event["pol_true"]) if "pol_true" in event else np.nan,
            }
            for name in PARAM_NAMES:
                row[name] = float(event["params"][name])
            rows.append(row)
    return rows


def summarize_fits(results: dict[str, dict], n_fitted: int, n_skipped: int) -> dict:
    rows = collect_fit_rows(results)
    if not rows:
        return {
            "n_files": 0,
            "n_events_fitted": 0,
            "n_events_skipped": n_skipped,
        }

    nrmse = np.asarray([r["nrmse"] for r in rows], dtype=np.float64)
    p_fit = np.asarray([r["P"] for r in rows], dtype=np.float64)
    pol_true = np.asarray([r["pol_true"] for r in rows], dtype=np.float64)
    has_pol = np.isfinite(pol_true)

    summary: dict = {
        "n_files": len(results),
        "n_events_fitted": n_fitted,
        "n_events_skipped": n_skipped,
        "nrmse": _percentile_stats(nrmse),
        "model_snr": _percentile_stats(
            np.asarray([r["model_snr"] for r in rows], dtype=np.float64)
        ),
        "P": _percentile_stats(p_fit),
        "P_negative_fraction": float(np.mean(p_fit < 0.0)),
        "params": {
            name: _percentile_stats(np.asarray([r[name] for r in rows], dtype=np.float64))
            for name in PARAM_NAMES
        },
    }

    if np.any(has_pol):
        residual = p_fit[has_pol] - pol_true[has_pol]
        p_sel = p_fit[has_pol]
        t_sel = pol_true[has_pol]
        if p_sel.size > 1 and np.std(p_sel) > 0.0 and np.std(t_sel) > 0.0:
            corr = float(np.corrcoef(p_sel, t_sel)[0, 1])
        else:
            corr = float("nan")
        summary["polarization"] = {
            "n_with_pol_true": int(np.sum(has_pol)),
            "P_minus_pol_true": _percentile_stats(residual),
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "corr": corr,
        }

    return summary


def print_summary(summary: dict) -> None:
    print("\n=== Dulya fit summary ===")
    print(
        f"gates:   MAX_NRMSE={MAX_NRMSE}  MIN_MODEL_SNR={MIN_MODEL_SNR}  "
        f"REQUIRE_DOUBLET={REQUIRE_DOUBLET}"
    )
    print(f"files:   {summary.get('n_files', 0)}")
    print(f"fitted:  {summary.get('n_events_fitted', 0)}")
    print(f"skipped: {summary.get('n_events_skipped', 0)}")
    nrmse = summary.get("nrmse") or {}
    if nrmse:
        print(
            "nrmse:   "
            f"median={nrmse['median']:.4g}  "
            f"p25={nrmse['p25']:.4g}  "
            f"p75={nrmse['p75']:.4g}  "
            f"mean={nrmse['mean']:.4g}"
        )
    pol = summary.get("polarization")
    if pol:
        print(
            "P vs pol_true: "
            f"mae={pol['mae']:.4g}  "
            f"rmse={pol['rmse']:.4g}  "
            f"corr={pol['corr']:.4g}"
        )
    print(f"P < 0 fraction: {summary.get('P_negative_fraction', 0.0):.3f}")


def plot_fit_statistics(results: dict[str, dict], out_dir: Path) -> list[Path]:
    rows = collect_fit_rows(results)
    if not rows:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    nrmse = np.asarray([r["nrmse"] for r in rows], dtype=np.float64)
    p_fit = np.asarray([r["P"] for r in rows], dtype=np.float64)
    pol_true = np.asarray([r["pol_true"] for r in rows], dtype=np.float64)
    model_snr = np.asarray([r["model_snr"] for r in rows], dtype=np.float64)
    half_width = np.asarray([r["half_width_mhz"] for r in rows], dtype=np.float64)
    has_pol = np.isfinite(pol_true)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(nrmse, bins=50, color="steelblue", edgecolor="white", alpha=0.9)
    axes[0, 0].axvline(np.median(nrmse), color="crimson", ls="--", lw=1.2, label="median")
    axes[0, 0].set_xlabel("NRMSE")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("NRMSE distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(p_fit, bins=50, color="darkorange", edgecolor="white", alpha=0.9)
    axes[0, 1].set_xlabel("P")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Polarization")
    axes[0, 1].grid(True, alpha=0.3)

    if np.any(has_pol):
        axes[1, 0].scatter(
            pol_true[has_pol],
            p_fit[has_pol],
            s=8,
            alpha=0.35,
            c="teal",
            edgecolors="none",
        )
        lims = [
            float(np.nanmin([pol_true[has_pol].min(), p_fit[has_pol].min()])),
            float(np.nanmax([pol_true[has_pol].max(), p_fit[has_pol].max()])),
        ]
        axes[1, 0].plot(lims, lims, "k--", lw=1.0, label="y = x")
        axes[1, 0].set_xlabel("pol_true")
        axes[1, 0].set_ylabel("P")
        axes[1, 0].set_title("P vs pol_true")
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, "no pol_true", ha="center", va="center")
        axes[1, 0].set_axis_off()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(model_snr, nrmse, s=8, alpha=0.35, c="slateblue", edgecolors="none")
    axes[1, 1].set_xlabel("model SNR")
    axes[1, 1].set_ylabel("NRMSE")
    axes[1, 1].set_title("NRMSE vs model SNR")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Dulya fit statistics ({len(rows)} events)", fontsize=13)
    fig.tight_layout()
    path = out_dir / "dulya_fit_statistics.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, name in zip(axes.ravel(), ("eta", "phi", "g", "xi"), strict=True):
        vals = np.asarray([r[name] for r in rows], dtype=np.float64)
        ax.hist(vals, bins=40, color="seagreen", edgecolor="white", alpha=0.9)
        ax.set_xlabel(name)
        ax.set_ylabel("Count")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Nuisance parameter distributions", fontsize=13)
    fig.tight_layout()
    path = out_dir / "dulya_param_histograms.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(half_width, bins=40, color="purple", edgecolor="white", alpha=0.9)
    ax.set_xlabel("half_width_mhz")
    ax.set_ylabel("Count")
    ax.set_title("Half-width distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "dulya_half_width_hist.png"
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

    # Plot best / median / worst NRMSE events.
    rows_sorted = sorted(rows, key=lambda r: r["nrmse"])
    n = len(rows_sorted)
    pick_idxs = sorted(
        {
            0,
            n // 3,
            2 * n // 3,
            n - 1,
        }
    )[:n_examples]
    picks = [rows_sorted[i] for i in pick_idxs]

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes_flat = axes.ravel()

    for ax, row in zip(axes_flat, picks, strict=False):
        records = load_records(str(data_dir / row["filename"]))
        freq_mhz = np.asarray(records[0]["freq_list"], dtype=np.float64)
        signal = np.asarray(records[row["event_id"]][VOLTAGE_KEY], dtype=np.float64)
        mask = wing_mask(len(freq_mhz))
        detrended = detrend_wings(freq_mhz, signal, mask)
        params = np.asarray([row[name] for name in PARAM_NAMES], dtype=np.float64)
        model = dulya_model(
            freq_mhz,
            *params,
            center_mhz=row["center_mhz"],
        )

        ax.plot(freq_mhz, detrended, color="0.55", lw=0.9, label="data (detrended)")
        ax.plot(freq_mhz, model, color="darkorange", lw=1.6, label="Dulya fit")
        ax.axvline(row["center_mhz"], color="0.35", ls=":", lw=0.9)
        pol_txt = (
            f"pol_true={row['pol_true']:+.3f}"
            if np.isfinite(row["pol_true"])
            else "pol_true=?"
        )
        ax.set_title(
            f"{row['filename'][:22]}…  evt {row['event_id']}\n"
            f"P={row['P']:+.3f}  CC={row['scaling_factor']:+.3f}  "
            f"{pol_txt}  NRMSE={row['nrmse']:.3f}",
            fontsize=9,
        )
        ax.set_ylabel("Voltage (V)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    for ax in axes_flat[len(picks) :]:
        ax.set_axis_off()
    for ax in axes_flat:
        if ax.has_data():
            ax.set_xlabel("Frequency (MHz)")

    fig.suptitle("Example Dulya fits (best → worst NRMSE)", fontsize=13)
    fig.tight_layout()
    path = out_dir / "dulya_fit_examples.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    data_dir = Path(PATH)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir.resolve()}")

    results: dict[str, dict] = {}
    txt_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".txt"))
    n_fitted = 0
    n_skipped = 0

    for filename in tqdm(txt_files, desc="Fitting Dulya"):
        records = load_records(str(data_dir / filename))
        if not records or VOLTAGE_KEY not in records[0] or "freq_list" not in records[0]:
            continue

        freq_mhz = np.asarray(records[0]["freq_list"], dtype=np.float64)
        if freq_mhz.ndim != 1 or len(freq_mhz) == 0:
            print(f"skip {filename}: bad freq_list")
            continue

        mask = wing_mask(len(freq_mhz))
        file_events: dict[int, dict] = {}

        for index, record in enumerate(records):
            if VOLTAGE_KEY not in record:
                n_skipped += 1
                continue

            signal = np.asarray(record[VOLTAGE_KEY], dtype=np.float64)
            if signal.ndim != 1 or len(signal) != len(freq_mhz):
                n_skipped += 1
                continue
            if not np.any(signal):
                n_skipped += 1
                continue

            if "pol" not in record:
                n_skipped += 1
                continue
            if "cc" not in record:
                n_skipped += 1
                continue

            try:
                p_fixed = float(record["pol"])
                scale_fixed = float(record["cc"])
                params, nrmse, model_snr, center_mhz = fit_dulya(
                    freq_mhz, signal, p_fixed, scale_fixed, mask
                )
            except Exception:
                n_skipped += 1
                continue

            if model_snr < MIN_MODEL_SNR:
                n_skipped += 1
                continue
            if nrmse > MAX_NRMSE:
                n_skipped += 1
                continue

            event_entry: dict = {
                "nrmse": nrmse,
                "model_snr": float(model_snr),
                "center_mhz": float(center_mhz),
                "pol_true": p_fixed,
                "cc": float(record["cc"]),
                "params": {
                    name: float(value)
                    for name, value in zip(PARAM_NAMES, params, strict=True)
                },
            }

            file_events[index] = event_entry
            n_fitted += 1

        if not file_events:
            continue

        results[filename] = {
            "center_mhz_nominal": CENTER_MHZ,
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

    print(
        f"\nWrote {n_fitted} Dulya fits "
        f"({n_skipped} skipped) across {len(results)} files to {out_path.resolve()}"
    )

    summary = summarize_fits(results, n_fitted, n_skipped)
    stats_path = Path(OUT_STATS_YAML)
    with open(stats_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, default_flow_style=False)
    print_summary(summary)
    print(f"Wrote stats to {stats_path.resolve()}")

    stats_dir = Path(OUT_STATS_DIR)
    plot_paths = plot_fit_statistics(results, stats_dir)
    example_path = plot_example_fits(results, data_dir, stats_dir)
    if example_path is not None:
        plot_paths.append(example_path)
    for path in plot_paths:
        print(f"Wrote plot {path.resolve()}")


if __name__ == "__main__":
    main()
