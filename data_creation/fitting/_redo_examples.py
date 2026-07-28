"""Regenerate Dulya example plots using current fit gates (fast, fixed sample)."""

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("fit_dulya", ROOT / "fitting" / "fit-dulya.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

OUT = ROOT / "fitting" / "dulya_fit_stats"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data-d"

# Known good doublet file + a few spaced event ids only.
FOCUS_FILE = "2022-09-13_09-40-39__2022-09-14_14-12-34.txt"
FOCUS_EVENTS = [0, 50, 100, 150, 200, 250, 300, 350, 400, 440]
# Quick neg-amp hunt: cheap doublet prefilter first, fit at most this many candidates.
NEG_AMP_SCAN_FILES = 40
NEG_AMP_MAX_FITS = 30


def try_fit(fname: str, eid: int, recs: list[dict] | None = None) -> dict | None:
    path = DATA / fname
    if recs is None:
        if not path.exists():
            return None
        recs = [json.loads(line) for line in open(path) if line.strip()]
    if eid >= len(recs) or "basesub" not in recs[eid]:
        return None
    freq = np.asarray(recs[0]["freq_list"], dtype=np.float64)
    sig = np.asarray(recs[eid]["basesub"], dtype=np.float64)
    if sig.shape != freq.shape or not np.any(sig):
        return None
    mask = mod.wing_mask(len(freq))
    try:
        params, nrmse, snr, center = mod.fit_dulya(
            freq, sig, float(recs[eid].get("pol", 0.2)), mask
        )
    except Exception as exc:
        return {"reject": str(exc), "fname": fname, "eid": eid}
    if snr < mod.MIN_MODEL_SNR:
        return {"reject": f"model_snr={snr:.3g}", "fname": fname, "eid": eid}
    if nrmse > mod.MAX_NRMSE:
        return {"reject": f"nrmse={nrmse:.4f}", "fname": fname, "eid": eid}
    det = mod.detrend_wings(freq, sig, mask)
    pol = float(recs[eid]["pol"]) if "pol" in recs[eid] else float("nan")
    amp = mod.amplitude_sign(det)
    return {
        "fname": fname,
        "eid": eid,
        "freq": freq,
        "det": det,
        "params": params,
        "nrmse": nrmse,
        "snr": snr,
        "center": center,
        "pol": pol,
        "amp_sign": amp,
        "model": mod.dulya_model(freq, *params, center_mhz=center, amp_sign=amp),
    }


def cheap_neg_amp_candidate(freq: np.ndarray, sig: np.ndarray, mask: np.ndarray) -> bool:
    """True if negative-going and looks like a doublet (no optimizer)."""
    det = mod.detrend_wings(freq, sig, mask)
    if mod.amplitude_sign(det) >= 0.0:
        return False
    _, _, found = mod.estimate_doublet(freq, det)
    return found


def main() -> None:
    kept_all: list[dict] = []
    pos_amp_neg_pol: list[dict] = []
    neg_amp_kept: list[dict] = []

    focus_path = DATA / FOCUS_FILE
    if not focus_path.exists():
        raise FileNotFoundError(focus_path)
    focus_recs = [json.loads(line) for line in open(focus_path) if line.strip()]
    for eid in FOCUS_EVENTS:
        row = try_fit(FOCUS_FILE, eid, focus_recs)
        if row is None or "reject" in row:
            print(f"skip focus evt {eid}: {row.get('reject') if row else 'missing'}")
            continue
        kept_all.append(row)
        if row["amp_sign"] > 0 and np.isfinite(row["pol"]) and row["pol"] < 0:
            pos_amp_neg_pol.append(row)
        print(
            f"kept focus evt {eid}: nrmse={row['nrmse']:.4f} "
            f"P={row['params'][0]:+.3f} amp={row['amp_sign']:+.0f}"
        )

    # Limited neg-amp search: prefilter first, fit only a few candidates.
    n_fit_attempts = 0
    for path in sorted(DATA.glob("*.txt"))[:NEG_AMP_SCAN_FILES]:
        if len(neg_amp_kept) >= 4 or n_fit_attempts >= NEG_AMP_MAX_FITS:
            break
        recs = [json.loads(line) for line in open(path) if line.strip()]
        if not recs or "basesub" not in recs[0]:
            continue
        freq = np.asarray(recs[0]["freq_list"], dtype=np.float64)
        mask = mod.wing_mask(len(freq))
        step = max(1, len(recs) // 20)
        for eid in range(0, len(recs), step):
            if len(neg_amp_kept) >= 4 or n_fit_attempts >= NEG_AMP_MAX_FITS:
                break
            sig = np.asarray(recs[eid].get("basesub", []), dtype=np.float64)
            if sig.shape != freq.shape or not np.any(sig):
                continue
            if not cheap_neg_amp_candidate(freq, sig, mask):
                continue
            n_fit_attempts += 1
            row = try_fit(path.name, eid, recs)
            if row is None or "reject" in row:
                continue
            if row["amp_sign"] < 0:
                neg_amp_kept.append(row)
                kept_all.append(row)
                print(
                    f"kept neg-amp {path.name} evt {eid}: "
                    f"nrmse={row['nrmse']:.4f} P={row['params'][0]:+.3f}"
                )

    print(
        f"summary: kept={len(kept_all)} pos_amp_neg_pol={len(pos_amp_neg_pol)} "
        f"neg_amp_kept={len(neg_amp_kept)} neg_amp_fit_attempts={n_fit_attempts}"
    )
    if not kept_all:
        raise RuntimeError("no events passed gates in the fixed sample")

    results: dict[str, dict] = {}
    for row in kept_all:
        results.setdefault(
            row["fname"],
            {
                "center_mhz_nominal": mod.CENTER_MHZ,
                "n_bins": len(row["freq"]),
                "freq_min_mhz": float(row["freq"][0]),
                "freq_max_mhz": float(row["freq"][-1]),
                "n_events_fitted": 0,
                "events": {},
            },
        )
        entry = {
            "nrmse": row["nrmse"],
            "model_snr": float(row["snr"]),
            "center_mhz": float(row["center"]),
            "params": {
                name: float(value)
                for name, value in zip(mod.PARAM_NAMES, row["params"], strict=True)
            },
        }
        if np.isfinite(row["pol"]):
            entry["pol_true"] = float(row["pol"])
        results[row["fname"]]["events"][row["eid"]] = entry
    for file_info in results.values():
        file_info["n_events_fitted"] = len(file_info["events"])

    mod.plot_fit_statistics(results, OUT)
    print("wrote", mod.plot_example_fits(results, DATA, OUT))

    row = pos_amp_neg_pol[0] if pos_amp_neg_pol else kept_all[0]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(row["freq"], row["det"], color="0.55", lw=1.0, label="data (detrended)")
    ax.plot(row["freq"], row["model"], color="darkorange", lw=1.8, label="Dulya fit")
    ax.axvline(
        row["center"], color="0.35", ls=":", lw=1.0, label=f"center={row['center']:.3f}"
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(
        f"{row['fname']}  event {row['eid']}\n"
        f"pol_true={row['pol']:+.4f}  amp_sign={row['amp_sign']:+.0f}  "
        f"P_fit={row['params'][0]:+.4f}  NRMSE={row['nrmse']:.4f}\n"
        f"gates: doublet + NRMSE≤{mod.MAX_NRMSE}"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUT / "dulya_fit_neg_pol_example.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.ravel()
    if not neg_amp_kept:
        for ax in axes_flat:
            ax.text(
                0.5,
                0.5,
                "No negative-amplitude events\npassed doublet + NRMSE gates\n"
                f"(scanned {NEG_AMP_SCAN_FILES} files, "
                f"{n_fit_attempts} fit attempts)",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
        fig.suptitle(
            f"Negative-amplitude Dulya fits (NRMSE≤{mod.MAX_NRMSE}, doublet required)"
        )
    else:
        for ax, row in zip(axes_flat, neg_amp_kept[:4], strict=False):
            ax.plot(row["freq"], row["det"], color="0.5", lw=0.9, label="data")
            ax.plot(row["freq"], row["model"], color="darkorange", lw=1.6, label="fit")
            ax.axhline(0, color="k", lw=0.5)
            ax.axvline(row["center"], color="0.35", ls=":")
            ax.set_title(
                f"{row['fname'][:28]}… evt {row['eid']}\n"
                f"pol_true={row['pol']:+.3f} amp={row['amp_sign']:+.0f}  "
                f"P_fit={row['params'][0]:+.3f}\n"
                f"NRMSE={row['nrmse']:.3f}",
                fontsize=8,
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        for ax in axes_flat[len(neg_amp_kept[:4]) :]:
            ax.set_axis_off()
        fig.suptitle(
            f"Negative-amplitude Dulya fits (passed doublet + NRMSE≤{mod.MAX_NRMSE})"
        )
    fig.tight_layout()
    path = OUT / "dulya_fit_neg_amp_example.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    if neg_amp_kept:
        row = neg_amp_kept[0]
        ax.plot(row["freq"], row["det"], color="0.45", label="basesub detrended")
        ax.plot(
            row["freq"],
            row["model"],
            color="darkorange",
            label=f"fit P={row['params'][0]:+.3f}",
        )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(
            f"NEG amplitude (kept)\n{row['fname']}\n"
            f"evt {row['eid']}  pol_true={row['pol']:+.3f}  NRMSE={row['nrmse']:.3f}"
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No negative-amplitude events\npassed doublet + NRMSE gates",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("NEG amplitude (none kept)")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Voltage (V)")

    ax = axes[1]
    row = pos_amp_neg_pol[0] if pos_amp_neg_pol else kept_all[0]
    ax.plot(row["freq"], row["det"], color="0.45", label=f"{mod.VOLTAGE_KEY} detrended")
    ax.plot(
        row["freq"],
        row["model"],
        color="darkorange",
        label=f"fit P={row['params'][0]:+.3f}",
    )
    center = row["center"]
    amp = row["amp_sign"]
    y_a = mod.dulya_model(
        row["freq"],
        row["pol"],
        0.0025,
        0.02,
        5.0,
        0.08,
        0.0,
        0.08,
        center_mhz=center,
        amp_sign=amp,
    )
    y_b = mod.dulya_model(
        row["freq"],
        -row["pol"],
        0.0025,
        0.02,
        5.0,
        0.08,
        0.0,
        0.08,
        center_mhz=center,
        amp_sign=amp,
    )
    scale = float(np.max(np.abs(row["det"])))
    y_a = y_a * (scale / (float(np.max(np.abs(y_a))) + 1e-15))
    y_b = y_b * (scale / (float(np.max(np.abs(y_b))) + 1e-15))
    ax.plot(
        row["freq"],
        y_a,
        "--",
        color="tab:blue",
        lw=1.2,
        label="Dulya P=pol_true",
    )
    ax.plot(
        row["freq"],
        y_b,
        "--",
        color="tab:green",
        lw=1.2,
        label="Dulya P=-pol_true",
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(
        f"POS amplitude / NEG pol_true (kept)\n{row['fname']} evt{row['eid']}  "
        f"pol_true={row['pol']:+.3f}"
    )
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Frequency (MHz)")
    fig.suptitle(
        f"Sign/orientation test — NRMSE≤{mod.MAX_NRMSE} + doublet",
        fontsize=11,
    )
    fig.tight_layout()
    path = OUT / "dulya_sign_orientation_test.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
