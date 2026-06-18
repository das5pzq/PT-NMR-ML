import json

import matplotlib.pyplot as plt
import numpy as np

from fit_multisite import (
    MULTISITE_MHZ_PARAM_ORDER,
    fit_chi_squared,
    fit_signal_mhz,
    freq_bounds_to_window,
    signal_model_mhz,
)

PATH = "../data/2022-07-21_00-22-36__2022-07-21_11-39-08.txt"
VOLTAGE_KEY = "basesub"
INDEX = 10

CENTER_MHZ = 32.68

EDGE_FRACTION = 0.28  # outer 28% of bins on each side (~32.3–32.52 & 32.88–33.1 MHz)
POLYNOMIAL_DEGREE = 3

# fit_multisite fit_signal defaults and bounds
SPLIT_CD0 = 0.09
SPLIT_OD0 = 0.07
SIGMA0 = 0.05
ETA_OD0 = 0.15
K0 = 0.12
XI0 = 0.0
CC0 = 1.0

P_BOUNDS = (-0.99, 0.99)
SIGMA_BOUNDS = (0.0, 0.5)
ETA_OD_BOUNDS = (0.0, 0.520)
K_BOUNDS = (0.0, 0.50)
XI_BOUNDS = (-0.15, 0.15)
CC_BOUNDS = (0.0, 1.5)

F_LO0 = CENTER_MHZ - 0.2
F_HI0 = CENTER_MHZ + 0.2

PARAM_NAMES = MULTISITE_MHZ_PARAM_ORDER


def subtract_polynomial_wings(
    freq_mhz: np.ndarray,
    signal: np.ndarray,
    *,
    edge_fraction: float = EDGE_FRACTION,
    degree: int = POLYNOMIAL_DEGREE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a polynomial to the outer wings and return the detrended signal."""
    n_bins = len(freq_mhz)
    n_edge = max(degree + 1, int(n_bins * edge_fraction))
    wing_mask = np.zeros(n_bins, dtype=bool)
    wing_mask[:n_edge] = True
    wing_mask[-n_edge:] = True

    coeffs = np.polyfit(freq_mhz[wing_mask], signal[wing_mask], deg=degree)
    polynomial = np.polyval(coeffs, freq_mhz)
    detrended = signal - polynomial
    return detrended, polynomial, wing_mask, coeffs


with open(PATH) as f:
    records = [json.loads(line) for line in f]

freq_mhz = np.asarray(records[0]["freq_list"])
signal_event = np.asarray(records[INDEX][VOLTAGE_KEY])

# ---- Subtracting out wings ----

signal_detrended, y_polynomial_fit, wing_mask, coeffs = subtract_polynomial_wings(
    freq_mhz,
    signal_event,
)
polynomial_fit_rms = float(
    np.sqrt(np.mean((signal_event[wing_mask] - y_polynomial_fit[wing_mask]) ** 2))
)
detrend_rms = float(np.sqrt(np.mean(signal_detrended**2)))

p0_guess = float(np.clip(records[INDEX].get("pol", 0.2), *P_BOUNDS))

print(f"Wing fit points: {wing_mask.sum()} / {len(freq_mhz)} bins")
print(f"Polynomial fit RMS (on edges): {polynomial_fit_rms:.6g}")
print(f"Detrended RMS (full spectrum): {detrend_rms:.6g}")
print(f"Polynomial coeffs (high→low power): {coeffs}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(freq_mhz, signal_event, label=VOLTAGE_KEY, alpha=0.85)
ax1.scatter(
    freq_mhz[wing_mask],
    signal_event[wing_mask],
    s=6,
    c="tab:green",
    alpha=0.45,
    label="wing fit points",
    zorder=3,
)
ax1.plot(
    freq_mhz,
    y_polynomial_fit,
    linestyle="--",
    color="tab:orange",
    label=f"degree-{POLYNOMIAL_DEGREE} polynomial (edge fit)",
)
ax1.set_ylabel("Voltage (V)")
ax1.set_title(f"Polynomial wing fit (event {INDEX})")
ax1.legend()
ax1.grid(True)

ax2.plot(freq_mhz, signal_detrended, color="tab:red", label="detrended")
ax2.axhline(0.0, color="black", linewidth=0.8)
ax2.set_xlabel("Frequency (MHz)")
ax2.set_ylabel("Voltage (V)")
ax2.set_title(f"After polynomial wing subtraction (RMS = {detrend_rms:.4g})")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# ---- Multi-site Dulya fit (fit_multisite) ----

_freq_lo = F_LO0
_freq_hi = F_HI0

p0 = {
    "P": p0_guess,
    "amp": 1.0,
    "f_lo_mhz": F_LO0,
    "f_hi_mhz": F_HI0,
    "cc": CC0,
    "split_cd": SPLIT_CD0,
    "split_od": SPLIT_OD0,
    "sigma": SIGMA0,
    "eta_od": ETA_OD0,
    "K": K0,
    "xi": XI0,
    "b0": 0.0,
    "b1": 0.0,
    "b2": 0.0,
    "b3": 0.0,
}
bounds = {
    "P": P_BOUNDS,
    "amp": (-np.inf, np.inf),
    "f_lo_mhz": (_freq_lo, _freq_hi),
    "f_hi_mhz": (_freq_lo, _freq_hi),
    "cc": CC_BOUNDS,
    "sigma": SIGMA_BOUNDS,
    "eta_od": ETA_OD_BOUNDS,
    "K": K_BOUNDS,
    "xi": XI_BOUNDS,
    "b0": (-np.inf, np.inf),
    "b1": (0.0, 0.0),
    "b2": (0.0, 0.0),
    "b3": (0.0, 0.0),
}
fixed = {
    "split_cd": SPLIT_CD0,
    "split_od": SPLIT_OD0,
    "b1": 0.0,
    "b2": 0.0,
    "b3": 0.0,
}

# signal_detrended = signal_detrended[::-1]
# signal_detrended *= -1.0


fit = fit_signal_mhz(
    freq_mhz,
    signal_detrended,
    p0=p0,
    bounds=bounds,
    fixed=fixed,
    wd=16.35,
    exact_intensity=False,
    nphi=64,
    loss="soft_l1",
    max_nfev=100_000,
)

params = fit.params
fitted_signal = signal_model_mhz(
    freq_mhz,
    **params,
    wd=16.35,
    exact_intensity=False,
    nphi=64,
)
fit_chi2 = fit_chi_squared(signal_detrended, fitted_signal)
resonance_center_mhz, bracket_split_mhz = freq_bounds_to_window(
    params["f_lo_mhz"],
    params["f_hi_mhz"],
)

print("\nFitted parameters:")
for name in PARAM_NAMES:
    value = params[name]
    if name in fixed:
        print(f"  {name:16s} = {value:.6g}  (fixed)")
    else:
        lo, hi = bounds[name]
        print(f"  {name:16s} = {value:.6g}  (bounds {lo:.4g} – {hi:.4g})")

print(f"Chi-squared: {fit_chi2:.6g}")

if "pol" in records[INDEX]:
    print(f"P_true: {records[INDEX]['pol']:.6g}")

print(f"Derived center: {resonance_center_mhz:.6g} MHz")
print(f"Derived bracket split: {bracket_split_mhz:.6g} MHz (from f_lo/f_hi)")

plt.figure(figsize=(10, 8))
plt.plot(freq_mhz, signal_detrended, label="detrended", alpha=0.85)
plt.plot(freq_mhz, fitted_signal, "--", label="multi-site Dulya fit")
plt.axvline(params["f_lo_mhz"], color="tab:gray", ls=":", alpha=0.7, label="f_lo")
plt.axvline(params["f_hi_mhz"], color="tab:gray", ls="--", alpha=0.7, label="f_hi")
plt.axvline(
    resonance_center_mhz - params["split_cd"],
    color="tab:blue",
    ls=":",
    alpha=0.7,
    label="C-D peaks",
)
plt.axvline(
    resonance_center_mhz + params["split_cd"],
    color="tab:blue",
    ls=":",
    alpha=0.7,
)
plt.axvline(
    resonance_center_mhz - params["split_od"],
    color="tab:orange",
    ls="--",
    alpha=0.7,
    label="O-D peaks",
)
plt.axvline(
    resonance_center_mhz + params["split_od"],
    color="tab:orange",
    ls="--",
    alpha=0.7,
)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Voltage (V)")
plt.title(f"Multi-site Dulya fit (event {INDEX}, χ² = {fit_chi2:.4g})")
plt.legend()
plt.grid(True)
plt.show()
