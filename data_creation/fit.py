import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from physics.Modified_Baseline import Baseline, DEFAULT_CIRC_CONSTS

PATH = "data/2022-07-21_00-22-36__2022-07-21_11-39-08.txt"
VOLTAGE_KEY = "basesub"  
INDEX = 10
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


with open(PATH) as f:
    records = [json.loads(line) for line in f]

baseline_signals = [rec["baseline"] for rec in records]
freq_mhz = np.asarray(records[0]["freq_list"])
n_bins = len(freq_mhz)

spectra = [
    {
        "freq_mhz": freq_mhz,
        "voltage": rec[VOLTAGE_KEY],
        "pol": rec["pol"],
        "area": rec["area"],
        "start_time": rec["start_time"],
    }
    for rec in records
]

print(f"{len(spectra)} spectra × {n_bins} bins")
print(f"freq range: {freq_mhz[0]:.4f} – {freq_mhz[-1]:.4f} MHz")
print(f"keys per record: {sorted(records[0].keys())}")

baseline_event = np.asarray(baseline_signals[INDEX])
signal_event = np.asarray(spectra[INDEX]["voltage"])

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
# ax1.plot(freq_mhz, baseline_event, label="baseline")
# ax1.set_xlabel("Frequency (MHz)")
# ax1.set_ylabel("Voltage (V)")
# ax1.set_title("Baseline")
# ax1.grid(True)

# ax2.plot(freq_mhz, signal_event, label="basesub")
# ax2.set_xlabel("Frequency (MHz)")
# ax2.set_ylabel("Voltage (V)")
# ax2.set_title("Signal, baseline subtracted")
# ax2.grid(True)

# plt.tight_layout()
# plt.show()

##### Baseline Fitting #####

p0 = (
    10.43,
    0.404,
    1.04e-2,
    0.5,
    1e-20,
    0.0,
    float(np.mean(baseline_event)),
) + DEFAULT_CIRC_CONSTS

params, pcov = curve_fit(
    baseline_fit,
    freq_mhz,
    baseline_event,
    p0=p0,
    maxfev=500000,
)
fitted_baseline = baseline_fit(freq_mhz, *params)
residual = baseline_event - fitted_baseline
rmse = float(np.sqrt(np.mean(residual**2)))

print("\nFitted baseline parameters:")
for name, value in zip(PARAM_NAMES, params):
    print(f"  {name:12s} = {value:.6g}")

print(f"\nRMSE = {rmse:.6g}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax1.plot(freq_mhz, baseline_event, label="data", alpha=0.8)
ax1.plot(freq_mhz, fitted_baseline, label="fit", linestyle="--")
ax1.set_ylabel("Voltage (V)")
ax1.set_title(f"Baseline fit (event {INDEX}, {SPECIES})")
ax1.legend()
ax1.grid(True)

ax2.plot(freq_mhz, residual, color="tab:red")
ax2.axhline(0.0, color="black", linewidth=0.8)
ax2.set_xlabel("Frequency (MHz)")
ax2.set_ylabel("Residual (V)")
ax2.set_title(f"Fit residual (RMSE = {rmse:.4g})")
ax2.grid(True)

plt.tight_layout()
plt.show()
