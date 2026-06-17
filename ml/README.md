# ML scripts

Training scripts for NMR polarization and signal models. Each script reads a parquet dataset, trains a model, and writes checkpoints and metrics under `Models/` and `Model_Performance/`.

## Scripts

| Script | Purpose |
|--------|---------|
| `area.py` | Predicts signal **area** from a 500-bin lineshape for **Spin-1/2** data. Uses a small MLP (or linear ridge regression if `HIDDEN = []`). Supports `--data_file` and `--reload` for prediction-only runs. |
| `pol_cnn.py` | Predicts **polarization** from lineshape for **Spin-1 non-cubic** materials using the paper's CNN (residual blocks, optional SE block for low-polarization). Edit `data_path`, `version`, and `POLARIZATION_RANGE` at the top of `main()`. |
| `pol_mlp.py` | Simpler 2-layer MLP for **Spin-1** polarization, aimed at the higher-polarization range (roughly 2–60%). Edit `data_path` and `version` in the `__main__` block. |
| `dae.py` | **Denoising autoencoder** that reconstructs clean Spin-1 lineshapes from noisy input. Edit `version` and the parquet path in the `__main__` block. |

## Run locally

From the repo root (or `ml/`), with your data file in place:

```bash
python ml/area.py --data_file path/to/data.parquet
python ml/pol_cnn.py
python ml/pol_mlp.py
python ml/dae.py
```

Most scripts expect parquet files with 500 signal columns plus target columns (`Area`, `P`, etc.). Check the `data_path` / `--data_file` setting in each script before running.

## Submit a SLURM job

`train.slurm` is a template batch script. Before submitting:

1. Edit the `#SBATCH` lines for your cluster (partition, GPU type, account, email).
2. Set the `module load` lines to match your site's PyTorch/Apptainer modules.
3. Replace `sample_script.py` with the script you want to run (e.g. `pol_cnn.py`).
4. Add any script arguments after the filename if needed (e.g. `area.py --data_file data.parquet`).

Submit from the directory that contains the job script and your data:

```bash
cd ml
sbatch train.slurm
```

Check status and output:

```bash
squeue -u $USER          # job status
tail -f training.out     # stdout (name set by #SBATCH --output)
tail -f training.err     # stderr
```

Cancel a job with `scancel <job_id>`.
