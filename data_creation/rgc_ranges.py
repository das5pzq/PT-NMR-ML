"""Empirical RGC Monte Carlo parameter ranges (p5–p95 from fit stats).

Dulya ranges from ``fitting/dulya_fit_stats.yaml`` / ``dulya_fits.yaml``.
Baseline ranges from ``fitting/baseline_fits.yaml``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

RGC_FREQ_MIN_MHZ = 32.3
RGC_FREQ_MAX_MHZ = 33.1
RGC_N_BINS = 512
CENTER_MHZ = 32.68

BASELINE_ETA = 0.0104
BASELINE_CSTRA = 0.0

# Lineshape / Dulya (uniform sample in [lo, hi])
DULYA_RANGES: Dict[str, Tuple[float, float]] = {
    "P": (-0.42999798, 0.47169143),
    "cc": (-14.5, -1.39),
    "eta": (0.010176152, 0.049637885),
    "phi": (4.6754515, 5.9939428),
    "g": (0.075091107, 0.092716231),
    "xi": (0.030398869, 0.15508333),
    "half_width_mhz": (0.077444836, 0.080749159),
}

# Primary baseline knobs (circuit pack fixed to DEFAULT_CIRC_CONSTS)
BASELINE_RANGES: Dict[str, Tuple[float, float]] = {
    "U": (-0.52558303, 7.1185749),
    "Cknob": (0.18480498, 0.3908962),
    "trim": (-0.20889848, 3.6620046),
    "phi_const": (-1.1715685, 25.024826),
    "DC_offset": (-5.78762, 0.091548305),
}


def _uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def sample_rgc_params(rng: np.random.Generator | None = None) -> dict[str, float]:
    """Draw one independent uniform sample of Dulya + baseline parameters."""
    if rng is None:
        rng = np.random.default_rng()

    params = {name: _uniform(rng, *bounds) for name, bounds in DULYA_RANGES.items()}
    # Keep |P| and |cc| away from DulyaFit / 1/cc singularities.
    if abs(params["P"]) < 1e-4:
        params["P"] = 1e-4 if params["P"] >= 0.0 else -1e-4
    if abs(params["cc"]) < 1e-6:
        params["cc"] = -1.39

    for name, bounds in BASELINE_RANGES.items():
        params[name] = _uniform(rng, *bounds)

    params["baseline_eta"] = BASELINE_ETA
    params["Cstray"] = BASELINE_CSTRA
    params["center_mhz"] = CENTER_MHZ
    return params


def rgc_frequency_mhz() -> np.ndarray:
    return np.linspace(RGC_FREQ_MIN_MHZ, RGC_FREQ_MAX_MHZ, RGC_N_BINS, dtype=np.float64)
