# PT-NMR-ML

Code for **Polarized Target Nuclear Magnetic Resonance (PT-NMR)** measurements with deep learning. This repo has two main parts:

1. **`data_creation/`** — build physics-based models, fit real spectra, and generate synthetic training data.
2. **`ml/`** — train neural networks to predict polarization (or related quantities) from NMR lineshapes.

## How the pieces fit together

```
Real NMR data  →  fit parameters  →  Monte Carlo simulation  →  parquet dataset  →  train ML model
                 (data_creation/fitting)  (data_creation/)              (ml/)
```

You do not need to run every step. For example, you can train on an existing parquet file without generating new data, or run the fitting scripts alone to study a single spectrum.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

**Generate a small test dataset** (from the repo root):

```bash
cd data_creation
python Create_Training_Data.py --num_samples 100 --output_dir Training_Data
```

**Train a polarization model** (after you have a parquet file):

```bash
cd ml
# Edit data_path at the bottom of pol_mlp.py, then:
python pol_mlp.py
```

See the folder READMEs for full details:

- [data_creation/README.md](data_creation/README.md) — data generation, physics models, fitting
- [ml/README.md](ml/README.md) — training scripts and cluster jobs

## Directory overview

| Folder | What it does |
|--------|----------------|
| [`data_creation/`](data_creation/) | Synthetic data generation and experimental fitting |
| [`data_creation/physics/`](data_creation/physics/) | Lineshape and baseline physics models |
| [`data_creation/fitting/`](data_creation/fitting/) | Scripts to fit baseline and Dulya models to real data |
| [`ml/`](ml/) | MLP, CNN, and denoising-autoencoder training |
