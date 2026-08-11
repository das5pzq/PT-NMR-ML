# Data Creation

This folder turns **physics models** and **real NMR measurements** into **synthetic training datasets** (Parquet files) for the ML scripts in `ml/`.

## What lives here

| Path | Purpose |
|------|---------|
| `physics/` | Core physics: Dulya lineshapes, circuit baseline model, proton/deuteron signal generation |
| `fitting/` | Fit real spectra to extract parameters (see [fitting/README.md](fitting/README.md)) |
| `signal_generator.py` | Monte Carlo generator for Spin-1 (deuteron) and Spin-1/2 (proton) spectra |
| `Create_Training_Data.py` | Command-line wrapper around `signal_generator.py` |
| `signal_generator_rgc.py` | RGC-specific generator that samples from empirical fit results |
| `Create_Training_Data_RGC.py` | Command-line wrapper for the RGC generator |
| `rgc_ranges.py` | Parameter ranges loaded from fit YAML files (used by the RGC generator) |
| `Create_Data.slurm` | SLURM template for large parallel data-generation jobs |
| `Create_Data_RGC.slurm` | SLURM template for RGC-style generation at scale |

## Typical workflow

1. **(Optional) Fit real data** in `fitting/` to learn realistic parameter ranges.
2. **Generate synthetic spectra** with one of the create scripts below.
3. **Train a model** in `ml/` using the output Parquet file.

---

## Generate training data (general Monte Carlo)

The simplest way to try things locally is `Create_Training_Data.py`. Run it from this directory:

```bash
cd data_creation
python Create_Training_Data.py --num_samples 100 --output_dir Training_Data
```

Useful flags:

| Flag | Meaning | Default |
|------|---------|---------|
| `--mode` | `deuteron` or `proton` | `deuteron` |
| `--polarization_type` | `vector` or `tensor` | `vector` |
| `--num_samples` | Spectra per run | `10` |
| `--add_noise` | `1` to add Gaussian noise | `0` |
| `--baseline` | `1` to include a circuit baseline | `1` |
| `--output_dir` | Where Parquet shards are written | `Training_Data` |
| `--job_id` | Suffix for the output filename (useful on clusters) | none |

Example with noise and a job id:

```bash
python Create_Training_Data.py \
  --num_samples 500 \
  --add_noise 1 \
  --job_id 1 \
  --output_dir Training_Data
```

Each run writes a file like `Training_Data/Sample_<job_id>.parquet`. Columns include the frequency-domain signal bins plus labels such as polarization `P`.

---

## Generate RGC-style data (from fitted parameters)

For deuteron vector data matched to RGC experiment fits, use `Create_Training_Data_RGC.py`. It draws random parameters from the pools in `fitting/dulya_fits_single_period.yaml` and `fitting/baseline_fits_single_event.yaml` (via `rgc_ranges.py`):

```bash
cd data_creation
python Create_Training_Data_RGC.py --num_samples 1000 --add_noise 1 --output_dir Training_Data_RGC
```

Optional: override the polarization range with `--p_min` and `--p_max` (both required if you use either).

---

## Fit experimental spectra

The `fitting/` subfolder has interactive scripts for single-spectrum fits. Start there if you want to understand the models before generating data:

```bash
cd data_creation/fitting
python fit-baseline.py    # circuit baseline on a baseline-only scan
python fit-dulya.py       # Dulya Pake doublet on a signal spectrum
```

Both scripts show a plot when done. Edit the `PATH` and `INDEX` settings at the top of each script to point at your data file and event number. Full parameter guides: [fitting/README.md](fitting/README.md).

---

## Run on a SLURM cluster

For large datasets, edit the configuration block at the top of `Create_Data.slurm` or `Create_Data_RGC.slurm` (paths, sample counts, array size, account), then submit from `data_creation/`:

```bash
sbatch Create_Data.slurm
# or
sbatch Create_Data_RGC.slurm
```

The general SLURM script runs many array tasks in parallel and submits a follow-up merge job to combine `Sample_*.parquet` shards into one file. Check `job_logs/` for output.

---

## Physics modules (`physics/`)

- **`Lineshape.py`** — Dulya doublet model, vector/tensor lineshapes, proton signal helper.
- **`Modified_Baseline.py`** — Lumped-element circuit baseline used in fits and simulation.

These are imported by the generators and fitting code; you usually do not run them directly.
