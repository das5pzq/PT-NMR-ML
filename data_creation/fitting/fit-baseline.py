import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import curve_fit
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics.Modified_Baseline import Baseline, DEFAULT_CIRC_CONSTS

PATH = "data-d"
OUT_YAML = "fitting/baseline_fits.yaml"
SPECIES = "deuteron"

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


def baseline_fit(f, U, Cknob, eta, trim, Cstray, phi_const, DC_offset, *circ_consts):
    return Baseline(
        f, U, Cknob, eta, trim, Cstray, phi_const, DC_offset, SPECIES, *circ_consts
    )


def load_records(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def fit_baseline(
    freq_mhz: np.ndarray, baseline: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Fit Q-meter baseline. Returns ``(params, nrmse, rmse)``.

    NRMSE = RMSE / peak |baseline| (stable when absolute noise is tiny).
    """
    p0 = (
        5.0,
        0.404,
        1.04e-2,
        0.5,
        1e-20,
        0.0,
        float(np.mean(baseline)),
    ) + DEFAULT_CIRC_CONSTS

    params, _ = curve_fit(
        baseline_fit,
        freq_mhz,
        baseline,
        p0=p0,
        maxfev=500000,
    )
    residual = baseline - baseline_fit(freq_mhz, *params)
    rmse = float(np.sqrt(np.mean(residual**2)))
    peak_amp = float(np.max(np.abs(baseline)))
    nrmse = float(rmse / peak_amp) if peak_amp > 0.0 else float("inf")
    return params, nrmse, rmse


def main() -> None:
    data_dir = Path(PATH)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir.resolve()}")

    results: dict[str, dict] = {}
    txt_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".txt"))

    for filename in tqdm(txt_files, desc="Fitting baselines"):
        records = load_records(str(data_dir / filename))
        if not records or "baseline" not in records[0]:
            continue

        # Baseline is shared across events in a file — fit once from the first record.
        baseline = np.asarray(records[0]["baseline"], dtype=np.float64)
        freq_mhz = np.asarray(records[0]["freq_list"], dtype=np.float64)

        if baseline.ndim != 1 or freq_mhz.ndim != 1:
            print(f"skip {filename}: expected 1-D baseline/freq_list")
            continue
        if len(baseline) != len(freq_mhz):
            print(f"skip {filename}: baseline/freq_list length mismatch")
            continue
        if not np.any(baseline):
            print(f"skip {filename}: baseline is all zeros")
            continue

        try:
            params, nrmse, rmse = fit_baseline(freq_mhz, baseline)
        except Exception as exc:
            print(f"skip {filename}: fit failed ({exc})")
            continue

        results[filename] = {
            "species": SPECIES,
            "nrmse": nrmse,
            "rmse": rmse,
            "n_bins": int(len(freq_mhz)),
            "freq_min_mhz": float(freq_mhz[0]),
            "freq_max_mhz": float(freq_mhz[-1]),
            "params": {
                name: float(value) for name, value in zip(PARAM_NAMES, params, strict=True)
            },
        }

    out_path = Path(OUT_YAML)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(results, f, sort_keys=False, default_flow_style=False)

    print(f"\nWrote {len(results)} baseline fits to {out_path.resolve()}")


if __name__ == "__main__":
    main()
