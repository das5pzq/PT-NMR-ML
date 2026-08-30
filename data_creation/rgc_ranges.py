from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import yaml

_DATA_DIR = Path(__file__).resolve().parent
DULYA_FITS_YAML = _DATA_DIR / "fitting" / "dulya_fits_single_period.yaml"
BASELINE_FITS_YAML = _DATA_DIR / "fitting" / "baseline_fits_single_event.yaml"
DULYA_STATS_YAML = _DATA_DIR / "fitting" / "dulya_fit_stats_single_period.yaml"
BASELINE_STATS_YAML = _DATA_DIR / "fitting" / "baseline_fit_stats_single_event.yaml"

RGC_FREQ_MIN_MHZ = 32.3
RGC_FREQ_MAX_MHZ = 33.1
RGC_N_BINS = 512
CENTER_MHZ = 32.68

DULYA_SAMPLE_KEYS = (
    "P",
    "cc",
    "eta",
    "phi",
    "g",
    "xi",
    "half_width_mhz",
    "g1_amp",
    "g1_loc",
    "g1_wid",
    "g2_amp",
    "g2_loc",
    "g2_wid",
    "center_mhz",
)

BASELINE_FIT_KEYS = (
    "U",
    "Cknob",
    "baseline_eta",
    "trim",
    "Cstray",
    "phi_const",
    "DC_offset",
)

BASELINE_CIRCUIT_KEYS = (
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

BASELINE_SAMPLE_KEYS = BASELINE_FIT_KEYS + BASELINE_CIRCUIT_KEYS


def _uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _stats_ranges(stats_path: Path, param_block: str) -> Dict[str, Tuple[float, float]]:
    if not stats_path.is_file():
        return {}
    with stats_path.open("r", encoding="utf-8") as handle:
        summary = yaml.safe_load(handle) or {}
    block = summary.get(param_block, summary.get("params", {}))
    ranges: Dict[str, Tuple[float, float]] = {}
    for name, stats in block.items():
        if not isinstance(stats, dict):
            continue
        lo = float(stats["min"])
        hi = float(stats["max"])
        if lo <= hi:
            ranges[name] = (lo, hi)
    return ranges


def _fallback_dulya_ranges() -> Dict[str, Tuple[float, float]]:
    ranges = _stats_ranges(DULYA_STATS_YAML, "params")
    if "scaling_factor" in ranges and "cc" not in ranges:
        ranges["cc"] = ranges.pop("scaling_factor")
    if "cc" not in ranges:
        ranges["cc"] = (1.39, 1.39)
    ranges.setdefault("center_mhz", (CENTER_MHZ, CENTER_MHZ))
    return {key: ranges[key] for key in DULYA_SAMPLE_KEYS if key in ranges}


def _fallback_baseline_ranges() -> Dict[str, Tuple[float, float]]:
    ranges = _stats_ranges(BASELINE_STATS_YAML, "params")
    if "eta" in ranges:
        ranges["baseline_eta"] = ranges.pop("eta")
    return {key: ranges[key] for key in BASELINE_SAMPLE_KEYS if key in ranges}


def _events_from_fit_yaml(path: Path) -> Dict[int, dict]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    events: Dict[int, dict] = {}
    for file_block in payload.values():
        if not isinstance(file_block, dict):
            continue
        for event_id, event in file_block.get("events", {}).items():
            events[int(event_id)] = event
    return events


def _dulya_event_to_params(event: dict) -> dict[str, float]:
    params = event["params"]
    cc = float(event.get("cc", params.get("scaling_factor", 1.39)))
    return {
        "P": float(params["P"]),
        "cc": cc,
        "eta": float(params["eta"]),
        "phi": float(params["phi"]),
        "g": float(params["g"]),
        "xi": float(params["xi"]),
        "half_width_mhz": float(params["half_width_mhz"]),
        "g1_amp": float(params["g1_amp"]),
        "g1_loc": float(params["g1_loc"]),
        "g1_wid": float(params["g1_wid"]),
        "g2_amp": float(params["g2_amp"]),
        "g2_loc": float(params["g2_loc"]),
        "g2_wid": float(params["g2_wid"]),
        "center_mhz": float(event.get("center_mhz", CENTER_MHZ)),
    }


def _baseline_event_to_params(event: dict) -> dict[str, float]:
    params = event["params"]
    out = {
        "U": float(params["U"]),
        "Cknob": float(params["Cknob"]),
        "baseline_eta": float(params["eta"]),
        "trim": float(params["trim"]),
        "Cstray": float(params["Cstray"]),
        "phi_const": float(params["phi_const"]),
        "DC_offset": float(params["DC_offset"]),
    }
    for name in BASELINE_CIRCUIT_KEYS:
        out[name] = float(params[name])
    return out


@lru_cache(maxsize=1)
def _load_fit_pool() -> tuple[dict[str, float], ...]:
    dulya_events = _events_from_fit_yaml(DULYA_FITS_YAML)
    baseline_events = _events_from_fit_yaml(BASELINE_FITS_YAML)
    shared_ids = sorted(set(dulya_events) & set(baseline_events))
    pool: list[dict[str, float]] = []
    for event_id in shared_ids:
        merged = _dulya_event_to_params(dulya_events[event_id])
        merged.update(_baseline_event_to_params(baseline_events[event_id]))
        pool.append(merged)
    return tuple(pool)


def _sample_uniform_fallback(rng: np.random.Generator) -> dict[str, float]:
    dulya_ranges = _fallback_dulya_ranges()
    baseline_ranges = _fallback_baseline_ranges()
    params = {name: _uniform(rng, *dulya_ranges[name]) for name in dulya_ranges}
    params.update({name: _uniform(rng, *baseline_ranges[name]) for name in baseline_ranges})
    return params


def _sample_p(rng: np.random.Generator, p_range: Optional[Tuple[float, float]]) -> Optional[float]:
    if p_range is None:
        return None
    lo, hi = p_range
    if lo > hi:
        raise ValueError(f"p_range must satisfy lo <= hi, got ({lo}, {hi})")
    return float(rng.uniform(lo, hi))


def sample_rgc_params(
    rng: np.random.Generator | None = None,
    p_range: Optional[Tuple[float, float]] = None,
) -> dict[str, float]:

    if rng is None:
        rng = np.random.default_rng()

    pool = _load_fit_pool()
    if pool:
        params = dict(pool[int(rng.integers(len(pool)))])
    else:
        params = _sample_uniform_fallback(rng)

    p_override = _sample_p(rng, p_range)
    if p_override is not None:
        params["P"] = p_override

    params["Q"] = 2 - np.sqrt(4 - 3*params["P"]**2)

    return params


def rgc_frequency_mhz() -> np.ndarray:
    return np.linspace(RGC_FREQ_MIN_MHZ, RGC_FREQ_MAX_MHZ, RGC_N_BINS, dtype=np.float64)
