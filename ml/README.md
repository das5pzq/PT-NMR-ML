# ML

Training scripts that learn to predict NMR quantities (mainly **polarization** or **signal area**) from frequency-domain lineshapes. Each script reads a **Parquet** dataset produced by `data_creation/`, trains a model, and saves checkpoints under `Models/` and metrics/plots under `Model_Performance/`.

## Before you start

1. Have a Parquet training file ready (from `data_creation/Create_Training_Data.py` or similar).
2. Open the script you want to run and set **`data_path`** (or pass **`--data_file`** where supported) to that file.

Run all commands from the **`ml/`** directory unless noted otherwise.

---

## Scripts

| Script | What it predicts | Notes |
|--------|------------------|-------|
| `pol_mlp.py` | Spin-1 **polarization** | Simple 2-layer MLP; good starting point. Edit `data_path` and `version` in the `__main__` block. |
| `train_mlp.py` | Same as `pol_mlp.py` | Alternate training entry point with its own `data_path` / `version` settings. |
| `test_pol_mlp.py` | — | Load a trained checkpoint and run inference on a real JSON-lines data file. |
| `pol_cnn.py` | Spin-1 **polarization** | CNN from the paper (residual blocks). Set `POLARIZATION_RANGE` to `LOW_POL` or `HIGH_POL` in `main()`. |
| `area.py` | Spin-1/2 **signal area** | Small MLP (or linear ridge if `HIDDEN = []`). Supports `--data_file` and `--reload`. |
| `dae.py` | — | **Denoising autoencoder**: reconstructs clean lineshapes from noisy input. |

Model definitions live in the same files as the training loops (`pol_mlp.py` defines `FFLightningModule`, etc.).

---

## Run locally

**Polarization MLP** (edit paths first):

```bash
cd ml
python pol_mlp.py
```

**Area model** with explicit data path:

```bash
python area.py --data_file path/to/your_data.parquet
```

**CNN, DAE, test script** — same pattern: set paths in the script, then:

```bash
python pol_cnn.py
python dae.py
python test_pol_mlp.py --version RGC_MLP_V1 --data-path ../data_creation/data-test/your_file.txt
```

### Expected data format

Parquet files should contain:

- **Signal columns** — typically 500 or 512 consecutive columns with the frequency-domain voltage (the scripts select the first N columns as `signal_cols`).
- **Target column** — e.g. `P` for polarization scripts or `Area` for `area.py`.

Check the `data_path` variable in each script before running.

### Outputs

After training you will see folders like:

```
ml/Models/<version>/          # saved checkpoints
ml/Model_Performance/<version>/   # scalers, loss plots, metrics
```

`area.py` writes to `area_model_out_TE/` by default instead.

---

## Submit a SLURM job

`train.slurm` is a template for GPU clusters. Before submitting:

1. Edit `#SBATCH` lines (partition, GPU, account, email).
2. Update `module load` / Apptainer lines for your site.
3. Replace `sample_script.py` with your script (e.g. `pol_cnn.py`).

```bash
cd ml
sbatch train.slurm
```

Check progress:

```bash
squeue -u $USER
tail -f training.out
```

Cancel with `scancel <job_id>`.

---

## Which script should I use?

| Goal | Start here |
|------|------------|
| First time / quick experiment | `pol_mlp.py` |
| Low polarization (roughly TE–2%) | `pol_cnn.py` with `LOW_POL` |
| Higher polarization (roughly 2–60%) | `pol_mlp.py` or `pol_cnn.py` with `HIGH_POL` |
| Spin-1/2 area instead of polarization | `area.py` |
| Denoise spectra | `dae.py` |
| Test a trained MLP on real data | `test_pol_mlp.py` |
