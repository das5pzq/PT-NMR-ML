import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from physics.Lineshape import DulyaFit

PATH = "data/2022-07-21_00-22-36__2022-07-21_11-39-08.txt"
VOLTAGE_KEY = "basesub"
INDEX = 10

CENTER_MHZ = 32.68
HALF_WIDTH_MHZ = 0.6  # initial guess: f_lo = center - half_width, f_hi = center + half_width

EDGE_FRACTION = 0.28  # outer 28% of bins on each side (~32.3–32.52 & 32.88–33.1 MHz)
POLYNOMIAL_DEGREE = 3

# DulyaFit parameters: P, scaling_factor, eta, phi, g, f_lo_mhz, f_hi_mhz
P_BOUNDS = (-0.99, 0.99)
SCALING_BOUNDS = (1e-4, 0.05)
ETA_BOUNDS = (0.001, 0.5)
PHI_BOUNDS = (0.0, 2 * np.pi)
G_BOUNDS = (0.001, 1.0)

### Frequency bounds for modeling lineshape ###

F_LO_BOUNDS = (32.0, 33.0)
F_HI_BOUNDS = (32.4, 33.3)

SCALING0 = 0.009
ETA0 = 0.0104
PHI0 = 6.1319
G0 = 0.2
F_LO0 = CENTER_MHZ - HALF_WIDTH_MHZ
F_HI0 = CENTER_MHZ + HALF_WIDTH_MHZ

PARAM_NAMES = ("P", "scaling_factor", "eta", "phi", "g", "f_lo_mhz", "f_hi_mhz")


def dulya_model(
    freq_mhz: np.ndarray,
    p: float,
    scaling_factor: float,
    eta: float,
    phi: float,
    g: float,
    f_lo_mhz: float,
    f_hi_mhz: float,
) -> np.ndarray:
    f_lo = min(f_lo_mhz, f_hi_mhz)
    f_hi = max(f_lo_mhz, f_hi_mhz)
    center = 0.5 * (f_lo + f_hi)
    half_width = max(0.5 * (f_hi - f_lo), 1e-6)
    x = (np.asarray(freq_mhz, dtype=np.float64) - center) / half_width
    return DulyaFit(x, p, scaling_factor, eta, phi, g)

with open(PATH) as f:
    records = [json.loads(line) for line in f]

freq_mhz = np.asarray(records[0]["freq_list"])
signal_event = np.asarray(records[INDEX][VOLTAGE_KEY])

# ---- Subtracting out wings ----

n_bins = len(freq_mhz)
n_edge = max(POLYNOMIAL_DEGREE + 1, int(n_bins * EDGE_FRACTION))
wing_mask = np.zeros(n_bins, dtype=bool)
wing_mask[:n_edge] = True
wing_mask[-n_edge:] = True

coeffs = np.polyfit(freq_mhz[wing_mask], signal_event[wing_mask], deg=POLYNOMIAL_DEGREE)
y_polynomial_fit = np.polyval(coeffs, freq_mhz)
signal_detrended = signal_event - y_polynomial_fit
polynomial_fit_rms = float(np.sqrt(np.mean((signal_event[wing_mask] - y_polynomial_fit[wing_mask]) ** 2)))
detrend_rms = float(np.sqrt(np.mean(signal_detrended**2)))

p0_guess = float(np.clip(records[INDEX].get("pol", 0.2), *P_BOUNDS))
f_lo0 = float(np.clip(CENTER_MHZ - HALF_WIDTH_MHZ, *F_LO_BOUNDS))
f_hi0 = float(np.clip(CENTER_MHZ + HALF_WIDTH_MHZ, *F_HI_BOUNDS))
if f_lo0 >= f_hi0:
    f_lo0, f_hi0 = F_LO0, F_HI0

print(f"Wing fit points: {wing_mask.sum()} / {n_bins} bins")
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

# ---- Dulya lineshape fit ----

p0 = (p0_guess, SCALING0, ETA0, PHI0, G0, f_lo0, f_hi0)
lower = (
    P_BOUNDS[0],
    SCALING_BOUNDS[0],
    ETA_BOUNDS[0],
    PHI_BOUNDS[0],
    G_BOUNDS[0],
    F_LO_BOUNDS[0],
    F_HI_BOUNDS[0],
)
upper = (
    P_BOUNDS[1],
    SCALING_BOUNDS[1],
    ETA_BOUNDS[1],
    PHI_BOUNDS[1],
    G_BOUNDS[1],
    F_LO_BOUNDS[1],
    F_HI_BOUNDS[1],
)


### NOTE: Reverse the signal for fitting. ACTUAL signal is in reverse order -- might need to fix the dulya model ###
signal_detrended = signal_detrended[::-1]

params, _ = curve_fit(
    dulya_model,
    freq_mhz,
    signal_detrended,
    p0=p0,
    bounds=(lower, upper),
    maxfev=100_000,
)

fitted_signal = dulya_model(freq_mhz, *params)
fit_rms = float(np.sqrt(np.mean((signal_detrended - fitted_signal) ** 2)))

print("\nFitted parameters:")
for name, value, bounds in zip(PARAM_NAMES, params, zip(lower, upper, strict=True), strict=True):
    print(f"  {name:16s} = {value:.6g}  (bounds {bounds[0]:.4g} – {bounds[1]:.4g})")
print(f"Fit RMS: {fit_rms:.6g}")

if "pol" in records[INDEX]:
    print(f"P_true: {records[INDEX]['pol']:.6g}")

plt.figure(figsize=(10, 8))
plt.plot(freq_mhz, signal_detrended, label="detrended", alpha=0.85)
plt.plot(freq_mhz, fitted_signal, "--", label="Dulya fit")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Voltage (V)")
plt.title(f"Dulya fit (event {INDEX}, RMS = {fit_rms:.4g})")
plt.legend()
plt.grid(True)
plt.show()
