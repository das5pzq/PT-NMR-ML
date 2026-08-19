# Fitting NMR spectra

This folder fits experimental NMR traces so we can extract physical parameters (circuit knobs, polarization, lineshape widths, and so on). Those fitted values become the ranges used when we simulate training data.

You do not need to know the optimizer internals to use these scripts. You mainly need to:

1. Point a script at your data.
2. Give it a reasonable starting guess.
3. Look at the plot (or YAML output) and decide whether the fit looks physical.

## What “fitting” means here

A **model** is a formula that draws a spectrum from a list of numbers (parameters). Fitting means: start from a guess, then let the computer nudge those numbers until the model curve matches the measured voltage vs frequency as closely as it can.

If the starting guess is far from the truth, the fit can land on a wrong shape that still looks “okay” on a plot. When a fit fails, the first thing to change is usually the starting values, not the physics code.

## Data files

Each `.txt` file is **JSON lines**: one JSON object per line, one NMR **event** (scan) per line.

Typical keys:

| Key | What it is |
|-----|------------|
| `freq_list` | Frequency axis in MHz (usually taken from the first event) |
| `phase` | Raw Q-meter voltage (baseline shape, used by `fit-baseline.py`) |
| `basesub` | Baseline-subtracted signal (used by `fit-butanol.py` and `fit-dulya.py`) |
| `cc` | Frequency-axis calibration constant stored with the event |
| `pol` | Polarization label from the experiment (if present) |

`fit-baseline.py` and `fit-dulya.py` take a **folder** of `.txt` files. `fit-butanol.py` takes a **single file** and one event index.

## How to run the scripts

Always run from `data_creation` so the `PATH` settings and `physics/` imports resolve the same way:

```bash
cd data_creation
```

Edit the settings at the **top of the script**, then:

```bash
python fitting/fit-baseline.py
python fitting/fit-dulya.py
python fitting/fit-butanol.py
```

`fit-baseline.py` and `fit-dulya.py` write YAML plus PNG plots (they do not pop up an interactive window). `fit-butanol.py` opens matplotlib figures on screen.

---

## Shared idea: polynomial wing subtraction

Baseline-subtracted spectra still have a slow leftover slope on the **wings** (the far left and far right of the frequency axis, away from the NMR peaks).

Before fitting a lineshape, `fit-butanol.py` and `fit-dulya.py` fit a low-degree polynomial **only on those outer bins**, then subtract it from the whole spectrum. That flattens the background without using the peaks themselves.

| Setting | What it does |
|---------|--------------|
| `EDGE_FRACTION` | Fraction of bins on **each** end used as wings (e.g. `0.25` = outer 25% on the left and on the right) |
| `POLYNOMIAL_DEGREE` | Degree of that polynomial (`2` = quadratic, `3` = cubic) |

If the subtracted spectrum still tilts, increase `EDGE_FRACTION` or the polynomial degree. If the peaks look chewed away, decrease `EDGE_FRACTION` so the polynomial is not using bins that belong to the doublet.

---

## `fit-baseline.py`

Fits the Q-meter **circuit baseline** to a raw (`phase`) spectrum. Repeating this on many events gives sensible ranges for simulation (especially `U`, `Cknob`, `eta`, `trim`, `Cstray`, phase, and DC offset).

By default the script fits **every event** in every `.txt` file under `PATH` (currently `"data-test"`). It does **not** use an `INDEX` setting.

### What to change

| Setting | Meaning |
|---------|---------|
| `PATH` | Folder of `.txt` files (relative to `data_creation`) |
| `VOLTAGE_KEY` | Which voltage array to fit (`"phase"` for raw baseline) |
| `SPECIES` | `"deuteron"` or `"proton"` — sets the Larmor frequency in the model |
| `OUT_YAML` / `OUT_STATS_YAML` | Where per-event fits and summary stats are written |
| `FIX_CIRCUIT_PARAMS_FROM_REFERENCE` | If `True`, event `REFERENCE_EVENT_INDEX` is a full fit; later events keep that event’s circuit constants fixed and only refit the seven “knobs” |
| `REFERENCE_EVENT_INDEX` | 0-based event number used as that reference (0 = first line in the file) |

Starting values for the seven knobs live in `_default_fit_p0` (not a `p0` list at the top of the file). `DC_offset` starts at the mean voltage of the wing bins.

### Parameters

The first seven numbers are the ones we usually care about. The rest are circuit constants from `DEFAULT_CIRC_CONSTS` in `physics/Modified_Baseline.py`.

| Parameter | What it affects |
|-----------|-----------------|
| `U` | Drive voltage scale (sets current through the circuit) |
| `Cknob` | Main tuning capacitance |
| `eta` | Coil fill factor (how much the sample inductance modulates the coil) |
| `trim` | Transmission-line length scale |
| `Cstray` | Stray capacitance in parallel with the coil |
| `phi_const` | Constant phase offset of the detected signal |
| `DC_offset` | Vertical offset added after the baseline shape is computed |
| `L0`, `Rcoil`, `R`, `R1`, `r`, `alpha`, `beta1`, `Z_cable`, `D`, `M` | Fixed circuit constants |
| `delta_C`, `delta_phi`, `delta_phase`, `delta_l` | Small correction terms (usually left at 0) |

This script does not set optimizer bounds by default. If a fit wanders to unphysical values, add a `bounds=(lower, upper)` argument to `curve_fit` in `fit_baseline()`, with one min/max pair per fitted parameter.

The baseline is matched on the **wings only** (`EDGE_FRACTION` / `POLYNOMIAL_DEGREE` define which bins count as wings). Gray bands on the example plots mark that region.

### Outputs

- `fitting/baseline_fits_single_event.yaml` — fitted parameters per file and event
- `fitting/baseline_fit_stats_single_event.yaml` — summary (median NRMSE, parameter spreads)
- `fitting/baseline_fit_stats_single_event/` — histograms and a few example overlay plots

**NRMSE** is root-mean-square residual divided by the peak |voltage| on the wings. Smaller is better; compare events to each other rather than chasing a magic number.

---

## `fit-butanol.py`

Fits a **d-butanol** (two-site C–D / O–D) lineshape to **one** baseline-subtracted event. This is the script to use when you want to sit with a single spectrum, tweak guesses, and watch the plot.

It runs in two steps: polynomial wing subtraction, then lineshape fitting.

### What to change

| Setting | Meaning |
|---------|---------|
| `PATH` | Path to **one** JSON-lines file (relative to where you run the script) |
| `INDEX` | Which event in that file (0 = first line) |
| `VOLTAGE_KEY` | Usually `"basesub"` |
| `CENTER_MHZ` | Nominal Larmor frequency used to build the dimensionless axis |
| `PARAMS` | Starting guesses (see table below) |
| `FIT_BOUNDS` | Allowed range for each fitted parameter |

### Step 1 — polynomial wing subtraction

Same idea as above (`EDGE_FRACTION`, `POLYNOMIAL_DEGREE`). The first figure shows the wing points, the polynomial, and the detrended spectrum. Chi-squared for that polynomial is printed in the terminal.

### Step 2 — d-butanol lineshape fit

**Initial guesses (`PARAMS`)** — starting values passed to the optimizer:

| Parameter | What it affects |
|-----------|-----------------|
| `P` | Deuteron vector polarization |
| `amp` | Overall signal scale (sign matters if the data were flipped) |
| `center` | Shifts the spectrum left/right on the frequency axis |
| `cc` | Stretches or compresses the frequency axis (`x_eff = cc * (x - center)`) |
| `split_cd` | C–D bond quadrupole splitting (peak separation for that site) |
| `split_od` | O–D bond quadrupole splitting |
| `sigma` | Common dipolar linewidth (peak width) |
| `eta_od` | O–D quadrupole asymmetry |
| `eta_cd` | C–D quadrupole asymmetry |
| `K` | O–D site fraction: `(1 - K)` is C–D, `K` is O–D |
| `xi` | Q-meter false-asymmetry correction |
| `b0`–`b3` | Residual background polynomial (constant through cubic) |
| `wd` | Deuteron Larmor frequency in MHz (fixed, not optimized) |
| `exact_intensity` | `True` = full d-butanol intensity; `False` = weak-quadrupole approximation (fixed) |
| `nphi` | Powder-average resolution (fixed) |

**Bounds (`FIT_BOUNDS`)** — each entry is `(minimum, maximum)`. The optimizer will not go outside these limits.

- To **fix** a parameter, set lower and upper to the same value (see `b1`–`b3` and `eta_cd`, which are pinned at 0).
- To **free** a fixed parameter, give it a real range instead.

Only keys listed in `FIT_BOUNDS` are optimized; everything else in `PARAMS` stays at the value you set.

**Tips:** If the fit fails or looks wrong, move `PARAMS` closer to what you see (peak positions → `split_cd` / `split_od`, heights → `amp` and `P`) and widen bounds that are too tight. Check the printed chi-squared and the residual panel at the bottom of the figure.

---

## `fit-dulya.py`

Fits Dulya’s **Pake doublet** (with powder averaging) to every baseline-subtracted event in a folder. Use this when you want batch fits and parameter statistics, not a single interactive plot.

The pipeline is the same idea as `fit-butanol.py` (detrend wings, then fit a lineshape), but the model and the parameter list are different.

### What to change

| Setting | Meaning |
|---------|---------|
| `PATH` | Folder of `.txt` files (relative to `data_creation`) |
| `VOLTAGE_KEY` | Usually `"basesub"` |
| `CENTER_MHZ` | Nominal center frequency if peak finding fails |
| `HALF_WIDTH_MHZ` | Fallback half-separation of the two Pake peaks (MHz) |
| `OUT_YAML` / `OUT_STATS_YAML` | Per-event fits and summary stats |
| `REQUIRE_DOUBLET` | Skip events where two peaks cannot be found |
| `MIN_MODEL_SNR` / `MAX_NRMSE` | Quality gates: events that fail these are skipped |
| `P0`, `XI0`, `G_AMP0`, … | Starting guesses for the **free** parameters |

`eta` (`ETA_FIXED`) and dipolar broadening `g` (`G_FIXED`) are held at values from a reference fit. The frequency calibration `scaling_factor` is **not** fitted: each event uses the `cc` stored in the data file. Keep that constant across a run if you can.

Events also need a `"pol"` field (used as `pol_true` in the YAML). Events missing `basesub`, `pol`, or `cc` are skipped.

### Step 1 — polynomial wing subtraction

Same as in `fit-butanol.py` (`EDGE_FRACTION`, `POLYNOMIAL_DEGREE`).

### Step 2 — Pake doublet fit (with powder averaging)

The model maps frequency to a dimensionless axis

`x = (freq_mhz - center) / half_width_mhz`

then evaluates Dulya’s powder-averaged Pake shape, multiplies by a Q-meter gain curve (`xi`), and optionally dips two narrow Gaussians on the peaks so the model is not systematically too tall at the horns.

**What is fitted vs held fixed**

| Role | Names |
|------|--------|
| Fitted | `P`, `xi`, `half_width_mhz`, and the six Gaussian knobs (`g1_amp`, `g1_loc`, `g1_wid`, `g2_amp`, `g2_loc`, `g2_wid`) |
| Taken from the data file | `scaling_factor` (`cc`) |
| Held at constants in the script | `eta`, `phi`, `g` |

| Parameter | What it affects |
|-----------|-----------------|
| `P` | Deuteron vector polarization |
| `scaling_factor` (`cc`) | Frequency-axis calibration from the file; do not fit it unless you have a reason |
| `eta` | Coil fill factor (fixed in this script) |
| `phi` | Extra phase in the Dulya lineshape (fixed at `PHI_FIXED`, usually 0) |
| `g` | Dipolar broadening (fixed in this script) |
| `xi` | Q-meter false-asymmetry correction |
| `half_width_mhz` | Half the frequency gap between the two Pake peaks; sets the x-axis scale. Bounds: `HALF_WIDTH_BOUNDS` |
| `g1_amp` / `g2_amp` | Depth of the Gaussian “shave” on the left / right peak (`G_AMP`) |
| `g1_loc` / `g2_loc` | Location of those Gaussians on the dimensionless x-axis (near −1 and +1) |
| `g1_wid` / `g2_wid` | Width of those Gaussians (`G_WID`) |
| `POWDER_NPHI` | Number of φ steps in the powder average (larger = smoother and slower) |

Bounds for the Gaussians are intentionally tight so they stay on the peaks and do not eat the shoulders.

### Outputs

- `fitting/dulya_fits_single_period.yaml` — fitted parameters per file and event
- `fitting/dulya_fit_stats_single_period.yaml` — summary (NRMSE, `P` vs `pol_true` if available)
- `fitting/dulya_fit_stats_single_period/` — statistics plots and a few example overlays

If many events are skipped, check `REQUIRE_DOUBLET`, `MIN_MODEL_SNR`, and `MAX_NRMSE`, and look at a raw `basesub` trace to see whether the doublet is actually visible.

---

## If a fit looks wrong

1. Plot the data first and confirm you are using the right voltage key (`phase` vs `basesub`).
2. Move starting guesses toward what you see (center, peak separation, amplitude sign).
3. Loosen bounds that sit on the fitted value (the optimizer is stuck against a wall).
4. Tighten bounds only when the fit is running to clearly unphysical numbers.
5. For batch scripts, open the example PNGs and the YAML for one bad event instead of retuning everything at once.
