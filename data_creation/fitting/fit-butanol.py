import json
import matplotlib.pyplot as plt
import numpy as np

from model.dulya import *
from model.baseline import *


PATH = "../data/2022-07-21_00-22-36__2022-07-21_11-39-08.txt"
VOLTAGE_KEY = "basesub"
INDEX = 10

CENTER_MHZ = 32.68

EDGE_FRACTION = 0.28  # outer 28% of bins on each side (~32.3–32.52 & 32.88–33.1 MHz)
POLYNOMIAL_DEGREE = 3


PARAMS = {
    "P": 0.4,              # deuteron vector polarization
    "amp": 0.7,            # overall signal amplitude scale
    "center": 0.0,         # frequency axis center shift
    "cc": 0.8,             # x-axis calibration (x_eff = cc * (x - center))
    "split_cd": 0.8,      # C-D site quadrupole scale 3*w_q
    "split_od": 0.8,      # O-D site quadrupole scale 3*w_q
    "sigma": 0.01,         # common dipolar linewidth width
    "eta_od": 0.01,        # O-D quadrupole asymmetry parameter
    "K": 0.5,             # O-D site fraction: (1-K)*CD + K*OD
    "xi": 0.0,             # Q-meter false-asymmetry correction
    "b0": 0.0,             # background polynomial constant term
    "b1": 0.0,             # background polynomial linear term
    "b2": 0.0,             # background polynomial quadratic term
    "b3": 0.0,             # background polynomial cubic term
    "wd": 32.68,           # deuteron Larmor frequency (MHz)
    "eta_cd": 0.01,        # C-D quadrupole asymmetry parameter
    "exact_intensity": True,  # use Dulya Eq. (24) vs weak-quadrupole approx
    "nphi": 64,            # phi steps for powder average
}

FIT_BOUNDS = {
    "P": (-0.99, -0.1),
    "amp": (-np.inf, np.inf),
    "center": (-1.0, 1.0),
    "cc": (0.0, 2.5),
    "split_cd": (1e-4, 2.0),
    "split_od": (1e-4, 2.0),
    "sigma": (0.0, 2.0),
    "eta_od": (0.0, 2.0),
    "eta_cd": (0.0, 0.00),
    "K": (0.0, 2.0),
    "xi": (-1.0, 1.0),
    "b0": (-np.inf, np.inf),
    "b1": (0.0, 0.0),
    "b2": (0.0, 0.0),
    "b3": (0.0, 0.0),
}

with open(PATH) as f:
    records = [json.loads(line) for line in f]

freq_mhz = np.asarray(records[0]["freq_list"])
signal_event = np.asarray(records[INDEX][VOLTAGE_KEY])

X_MIN = -6.0
X_MAX = 6.0

R = np.linspace(X_MIN, X_MAX, len(freq_mhz))

### subtract polynomial wings

signal_detrended, y_polynomial_fit, wing_mask, coeffs, chi_squared = subtract_polynomial_wings(
    R,
    signal_event,
    EDGE_FRACTION,
    POLYNOMIAL_DEGREE,
)

print(f"Chi-squared = {chi_squared:.4g}")

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
ax2.set_title(f"After polynomial wing subtraction (Chi-squared = {chi_squared:.4g})")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

signal_detrended = signal_detrended[::-1]

Y_ERR = np.ones_like(signal_detrended) * 1e-6



### Fitting Dulya lineshape to data ###

dulya_model = DulyaModel()
fitted_params = dulya_model.fit_dulya(R, signal_detrended, Y_ERR, PARAMS, bounds=FIT_BOUNDS, method="Powell")

print(f"===================== Fit Results =====================")
for name in dulya_model.param_keys:
    print(f"{name}: {fitted_params[name]:.6g}")
print(f"-"*40)
print(f"Chi-squared = {dulya_model.chi_squared(fitted_params, R, signal_detrended, Y_ERR):.4g}")
print(f"===============================================================")

curves = dulya_model.component_curves(fitted_params, R)

fig, (ax, axr) = plt.subplots(2, 1, figsize=(9.0, 7.5), sharex=True, height_ratios=[3.2, 1.0])

ax.plot(R, signal_detrended, "o", ms=3, alpha=0.85, label="data")
ax.plot(R, curves["total"], lw=2.2, label="total fit")
ax.plot(R, curves["cd_total"], lw=1.5, linestyle="--", label="C-D site")
ax.plot(R, curves["od_total"], lw=1.5, linestyle=":", label="O-D site")
ax.plot(R, curves["plus_total"], lw=1.2, linestyle="-.", label="plus branch")
ax.plot(R, curves["minus_total"], lw=1.2, linestyle=(0, (5, 2, 1, 2)), label="minus branch")
ax.plot(R, curves["background"], lw=1.0, alpha=0.8, label="background")
ax.set_ylabel("Signal")
ax.set_title("Dulya butanol fit")
ax.grid(alpha=0.25)
ax.legend(loc="best", ncols=2)

txt = (
    f"P = {fitted_params['P']:.5f}\n"
    f"r = {dulya_model.p_to_r(fitted_params['P']):.5f}\n"
    f"split_cd = {fitted_params['split_cd']:.5g}\n"
    f"split_od = {fitted_params['split_od']:.5g}\n"
    f"sigma = {fitted_params['sigma']:.5g}\n"
    f"eta_od = {fitted_params['eta_od']:.5g}\n"
    f"K = {fitted_params['K']:.5g}"
)
ax.text(
    0.02,
    0.98,
    txt,
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.78, lw=0.5),
)

residuals = signal_detrended - curves["total"]
axr.axhline(0.0, lw=1.0, alpha=0.8)
axr.plot(R, residuals, "o", ms=3, alpha=0.85)
axr.set_xlabel("Frequency / offset")
axr.set_ylabel("resid")
axr.grid(alpha=0.25)

plt.tight_layout()
plt.show()
