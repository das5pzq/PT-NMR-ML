# Fitting

This folder contains scripts and modules for fitting experimental NMR data so we can extract physical parameters for simulation.

## Running the scripts

Both scripts read a JSON-lines data file (one spectrum per line). Edit the settings at the **top of each script** before running:

| Setting | Meaning |
|---------|---------|
| `PATH` | Path to the data file (relative to where you run the script) |
| `INDEX` | Which event/scan in the file to fit (0 = first line) |

From the repository root:

```bash
cd data_creation
python fitting/fit-baseline.py
```

```bash
cd data_creation/fitting
python fit-dulya.py
```

Each script opens a plot when it finishes. You need Python with `numpy`, `scipy`, and `matplotlib` installed.

---

## `fit-baseline.py`

Fits the circuit baseline model to a baseline-only spectrum. Repeating this on many baselines helps you learn sensible parameter ranges for simulation.

### What to change

Open `fit-baseline.py` and look for the block labeled **Baseline Fitting** (around line 57).

**Initial guess (`p0`)** — a list of starting values the optimizer tries first. They must appear in the same order as `PARAM_NAMES`:

| Parameter | What it affects |
|-----------|-----------------|
| `U` | Drive voltage scale (sets current through the circuit) |
| `Cknob` | Main tuning capacitance |
| `eta` | Coil fill factor (inductance modulation) |
| `trim` | Transmission-line length scale |
| `Cstray` | Stray capacitance in parallel with the coil |
| `phi_const` | Constant phase offset of the detected signal |
| `DC_offset` | Vertical offset added after the baseline shape is computed |
| `L0`, `Rcoil`, `R`, `R1`, `r`, `alpha`, `beta1`, `Z_cable`, `D`, `M` | Fixed circuit constants (from `DEFAULT_CIRC_CONSTS` in `physics/Modified_Baseline.py`) |
| `delta_C`, `delta_phi`, `delta_phase`, `delta_l` | Small correction terms (usually left at 0) |

The first seven numbers in `p0` are the fitted baseline knobs; the rest come from `DEFAULT_CIRC_CONSTS`. A good starting point for `DC_offset` is the average voltage of your data — the script already does this with `np.mean(baseline_event)`.

**Bounds** — this script does not set bounds by default. If a fit wanders to unphysical values, you can add a `bounds=(lower, upper)` argument to `curve_fit`, with one min/max pair per parameter in `p0`.

**Other settings at the top:** `SPECIES` (`"deuteron"` or `"proton"`) sets the Larmor frequency used in the model.

---

## `fit-dulya.py`

Fits Dulya's Pake-doublet model to a baseline-subtracted signal. The fit runs in two steps: polynomial wing subtraction, then lineshape fitting.

### Step 1 — polynomial wing subtraction

| Setting | What it does |
|---------|--------------|
| `EDGE_FRACTION` | Fraction of frequency bins on each end used to fit a background polynomial (default 0.28 ≈ outer 28%) |
| `POLYNOMIAL_DEGREE` | Degree of that polynomial (default 3) |

Increase `EDGE_FRACTION` if the wings look poorly subtracted; decrease it if the polynomial is eating into the doublet peaks.

### Step 2 — Dulya lineshape fit

**Initial guesses (`PARAMS`)** — starting values passed to the optimizer:

| Parameter | What it affects |
|-----------|-----------------|
| `P` | Deuteron vector polarization (negative for typical PT-NMR data) |
| `amp` | Overall signal scale (sign matters if data were flipped) |
| `center` | Shifts the spectrum left/right on the frequency axis |
| `cc` | Stretches or compresses the frequency axis (`x_eff = cc * (x - center)`) |
| `split_cd` | C–D bond quadrupole splitting scale (peak separation) |
| `split_od` | O–D bond quadrupole splitting scale |
| `sigma` | Common dipolar linewidth (peak width) |
| `eta_od` | O–D quadrupole asymmetry |
| `eta_cd` | C–D quadrupole asymmetry |
| `K` | O–D site fraction: `(1-K)` is C–D, `K` is O–D |
| `xi` | Q-meter false-asymmetry correction |
| `b0`–`b3` | Residual background polynomial (constant through cubic) |
| `wd` | Deuteron Larmor frequency in MHz (fixed, not optimized) |
| `exact_intensity` | `True` = full Dulya intensity; `False` = weak-quadrupole approximation (fixed) |
| `nphi` | Powder-average resolution (fixed) |

**Bounds (`FIT_BOUNDS`)** — each entry is `(minimum, maximum)` for that parameter. The optimizer will not go outside these limits. To **fix** a parameter, set its lower and upper bound to the same value (see `b1`–`b3` and `eta_cd`, which are pinned at 0). To **free** a fixed parameter, give it a real range instead.

Only keys listed in `FIT_BOUNDS` are optimized; everything else in `PARAMS` stays at the value you set.

**Tips:** If the fit fails or looks wrong, adjust the starting values in `PARAMS` to be closer to what you expect (e.g. peak positions → `split_cd` / `split_od`, peak heights → `amp` and `P`), and widen bounds that may be too tight. Check the printed chi-squared and residual plot at the bottom of the figure.
