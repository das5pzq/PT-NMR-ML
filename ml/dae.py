from typing import Tuple

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar, StochasticWeightAveraging
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import os
import pickle
import gc 
from sklearn.preprocessing import MinMaxScaler

from stable_minmax import StableMinMaxScaler

sys.stdout.flush()
sys.stderr.flush()

NUM_EPOCHS = 500 
BATCH_SIZE = 64
LEARNING_RATE = 1e-2
DEFAULT_NOISE_FACTOR = 3 * 2.690506959957014e-05
LOSS_PROG_BAR_FORMAT = ".8f"

if torch.backends.mps.is_available():
    device = torch.device("mps")
    trainer_accelerator = "mps"
else:
    device = torch.device("cpu")
    trainer_accelerator = "cpu"


class DenoisingAutoencoder(nn.Module):
    """
    Simple DAE: input_dim -> bottleneck -> input_dim.
    Input: noisy signal (normalized); output: clean signal (normalized).
    """

    def __init__(self, input_dim: int = 500, hidden_dims: Tuple[int, ...] = (256, 128, 64, 32, 16)):
        super().__init__()
        self.input_dim = input_dim

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                ]
            )
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.bottleneck_dim = hidden_dims[-1]

        layers = []
        for h in reversed(hidden_dims[:-1]):
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class AE(L.LightningModule):
    def __init__(
        self,
        noise_factor=DEFAULT_NOISE_FACTOR,
        scaler=None,
        input_dim: int = 500,
        hidden_dims: Tuple[int, ...] = (256, 128, 64, 32, 16),
    ):
        super(AE, self).__init__()
        # Gaussian noise std in raw spectrum units (same units as parquet columns before scaling).
        self.noise_factor = noise_factor
        self.scaler = scaler
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.net = DenoisingAutoencoder(input_dim=input_dim, hidden_dims=hidden_dims)

    def add_noise(self, x):
        """Additive Gaussian noise; std = ``noise_factor`` in the same units as ``x``."""
        noise = torch.randn_like(x) * self.noise_factor
        return x + noise

    def noisy_scaled_batch(self, x_clean_scaled: torch.Tensor) -> torch.Tensor:
        """Inverse-scale clean inputs, add noise in raw units, then scale (training/eval pipeline)."""
        x_phys = self.unscale_with_scaler(x_clean_scaled)
        x_noisy_phys = self.add_noise(x_phys)
        return self.scale_with_scaler(x_noisy_phys)

    def scale_with_scaler(self, x):
        if self.scaler is None:
            return x
        device = x.device
        x_np = x.detach().cpu().numpy()
        x_scaled_np = self.scaler.transform(x_np)
        return torch.tensor(x_scaled_np, dtype=x.dtype, device=device)

    def unscale_with_scaler(self, x):
        if self.scaler is None:
            return x
        device = x.device
        x_np = x.detach().cpu().numpy()
        x_unscaled_np = self.scaler.inverse_transform(x_np)
        return torch.tensor(x_unscaled_np, dtype=x.dtype, device=device)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        x = x.view(x.size(0), -1)
        x_clean_scaled = x.clone()
        x_noisy_scaled = self.noisy_scaled_batch(x_clean_scaled)

        decoded = self.forward(x_noisy_scaled)

        train_loss = nn.functional.mse_loss(decoded, x_clean_scaled)

        self.log('train_loss', train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        x = x.view(x.size(0), -1)
        x_clean_scaled = x.clone()
        x_noisy_scaled = self.noisy_scaled_batch(x_clean_scaled)

        decoded = self.forward(x_noisy_scaled)

        val_loss = nn.functional.mse_loss(decoded, x_clean_scaled)

        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, _, _ = batch
        x = x.view(x.size(0), -1)
        x_clean_scaled = x.clone()
        x_noisy_scaled = self.noisy_scaled_batch(x_clean_scaled)

        decoded = self.forward(x_noisy_scaled)

        test_loss = nn.functional.mse_loss(decoded, x_clean_scaled)

        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=max(1e-8, LEARNING_RATE * 1e-4)
        )
        return [optimizer], [scheduler]


if __name__ == '__main__':
    SEED = 42

    torch.set_default_dtype(torch.float32)
    L.seed_everything(SEED, workers=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    version = 'Exp2_DAE_V1' 

    performance_dir = f"Model_Performance/{version}"  
    model_dir = f"Models/{version}"  
    os.makedirs(performance_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_parquet("Exp2_DAE.parquet")

    X = df.drop(columns=["P", 'SNR', 'Area']).values
    area = df["Area"].values
    P = df["P"].values 
    SNR = df["SNR"].values  

    P = P.reshape(-1, 1)
    area = area.reshape(-1, 1)

    X_Scaler = StableMinMaxScaler(range_floor_relative=1e-4)
    X_Scaler.fit(X)
    X = X_Scaler.transform(X)
    P_Scaler = MinMaxScaler()
    P = P_Scaler.fit_transform(P)
    Area_Scaler = MinMaxScaler()
    area = Area_Scaler.fit_transform(area)

    with open(f"{performance_dir}/{version}_scaler_X.pkl", 'wb') as f:
        pickle.dump(X_Scaler, f)
    with open(f"{performance_dir}/{version}_scaler_P.pkl", 'wb') as f:
        pickle.dump(P_Scaler, f)
    with open(f"{performance_dir}/{version}_scaler_area.pkl", 'wb') as f:
        pickle.dump(Area_Scaler, f)

    print(f"Shape of X: {X.shape}")
    print(f"Range of X: {X.min():.4f} to {X.max():.4f}")
    print(f"Range of P: {P.min():.4f} to {P.max():.4f}")
    print(f"Range of area: {area.min():.4f} to {area.max():.4f}")

    dataset = data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(P, dtype=torch.float32), torch.tensor(area, dtype=torch.float32))

    del df, X, P, area
    gc.collect()

    train_dataset, val_dataset, test_dataset = data.random_split(dataset, [0.80, 0.10, 0.10])

    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=13, persistent_workers=True)
    val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=13, persistent_workers=True)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=13, persistent_workers=True)

    rich_progress = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_format=LOSS_PROG_BAR_FORMAT,
        )
    )

    trainer = L.Trainer(
        max_epochs=NUM_EPOCHS, 
        callbacks=[
            rich_progress,
            EarlyStopping(monitor='val_loss', patience=20, mode='min'),
            ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, save_last=True, 
                          dirpath=model_dir, filename='best_model-{epoch:02d}-{val_loss:.8f}'),
            # Early SWA + warm restarts made val_loss sit near the ~1/12 "mean predictor" floor in practice.
            StochasticWeightAveraging(swa_lrs=min(LEARNING_RATE, 1e-3), swa_epoch_start=max(10, int(NUM_EPOCHS * 0.75)))
        ],
        logger=TensorBoardLogger(save_dir=performance_dir, name=version),
        enable_progress_bar=True,
        devices=1,
        accelerator=trainer_accelerator,
        precision='32',  
        deterministic=True
    )


    with trainer.init_module():
        model = AE(noise_factor=DEFAULT_NOISE_FACTOR, scaler=X_Scaler)
        
    trainer.fit(model, train_loader, val_loader)

    final_model_path = f"{model_dir}/{version}_final_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    state_dict_path = f"{model_dir}/{version}_state_dict.pth"
    torch.save(model.state_dict(), state_dict_path)
    print(f"Model state dict saved to: {state_dict_path}")

    test_results = trainer.test(model, test_loader)

    print("\nCollecting test predictions for analysis...")
    model_to_eval = trainer.model
    if hasattr(model_to_eval, 'module'):
        model_to_eval = model_to_eval.module

    model_to_eval.eval()
    model_to_eval = model_to_eval.to(device)
    test_X_actual = []
    test_X_noisy = []
    test_X_predicted = []
    test_P = []
    test_area = []
    test_SNR = []
    test_reconstruction_errors = []

    with torch.no_grad():
        for batch in test_loader:
            x, p, area_batch = batch
            x = x.to(device)
            x_clean_scaled = x.clone().view(x.size(0), -1)

            x_noisy_scaled = model_to_eval.noisy_scaled_batch(x_clean_scaled)

            decoded_scaled = model_to_eval.forward(x_noisy_scaled)
            
            x_clean_unscaled = model_to_eval.unscale_with_scaler(x_clean_scaled)
            x_noisy_unscaled = model_to_eval.unscale_with_scaler(x_noisy_scaled)
            decoded_unscaled = model_to_eval.unscale_with_scaler(decoded_scaled)

            x_clean_np = x_clean_unscaled.cpu().numpy()
            x_noisy_np = x_noisy_unscaled.cpu().numpy()
            decoded_np = decoded_unscaled.cpu().numpy()
            
            reconstruction_error = np.mean((decoded_np - x_clean_np) ** 2, axis=1)
            
            test_X_actual.append(x_clean_np)
            test_X_noisy.append(x_noisy_np)
            test_X_predicted.append(decoded_np)
            test_P.append(p.numpy())
            test_area.append(area_batch.numpy())
            test_reconstruction_errors.append(reconstruction_error)

    test_indices = test_dataset.indices
    test_SNR_values = SNR[test_indices]

    test_X_actual = np.concatenate(test_X_actual, axis=0)
    test_X_noisy = np.concatenate(test_X_noisy, axis=0)
    test_X_predicted = np.concatenate(test_X_predicted, axis=0)
    test_P = np.concatenate(test_P, axis=0)
    test_area = np.concatenate(test_area, axis=0)
    test_reconstruction_errors = np.concatenate(test_reconstruction_errors, axis=0)

    test_X_actual_unscaled = test_X_actual
    test_X_noisy_unscaled = test_X_noisy
    test_X_predicted_unscaled = test_X_predicted

    noise_mag_phys = np.mean(np.abs(test_X_noisy_unscaled - test_X_actual_unscaled))
    print(f"\nNoise verification (physical units): mean |noisy - clean| = {noise_mag_phys:.6f}")
    print(f"Noise std (raw spectrum units, before scaling): {model_to_eval.noise_factor}")
    # example plot with noisy input shown
    plt.figure(figsize=(16, 12))
    plt.style.use('ggplot')
    plt.plot(test_X_actual_unscaled[0], label='Actual (Clean)', color='red', linewidth=2)
    plt.plot(test_X_noisy_unscaled[0], label='Noisy Input', color='orange', linewidth=1.5, alpha=0.7)
    plt.plot(test_X_predicted_unscaled[0], label='Predicted (Denoised)', color='blue', linewidth=2)
    plt.xlabel('Frequency [MHz]', fontsize=18, fontfamily='Times New Roman')
    plt.ylabel('Signal [$C_E$ mV]', fontsize=18, fontfamily='Times New Roman')
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3, color='lightgray', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{performance_dir}/{version}_reconstruction_example.pdf", dpi=1200)
    plt.close()
    print(f"Reconstruction example saved to {performance_dir}/{version}_reconstruction_example.pdf")

    mse_per_sample_unscaled = np.mean((test_X_predicted_unscaled - test_X_actual_unscaled) ** 2, axis=1)
    mae_per_sample = np.mean(np.abs(test_X_predicted_unscaled - test_X_actual_unscaled), axis=1)
    rmse_per_sample = np.sqrt(mse_per_sample_unscaled)

    signal_magnitude = np.sqrt(np.sum(test_X_actual_unscaled ** 2, axis=1))
    reconstruction_error_magnitude = np.sqrt(np.sum((test_X_predicted_unscaled - test_X_actual_unscaled) ** 2, axis=1))
    rre = (reconstruction_error_magnitude / (signal_magnitude + 1e-10)) * 100

    mean_residual = np.mean(test_X_predicted_unscaled - test_X_actual_unscaled)
    std_residual = np.std(test_X_predicted_unscaled - test_X_actual_unscaled)
    print(f"Mean residual: {mean_residual:.6e}")
    print(f"Std residual: {std_residual:.6e}")

    # rre_90 = rre[rre < np.percentile(rre, 90)]

    print("\nSaving results...")
    results_data = {
        'Reconstruction_MSE': mse_per_sample_unscaled,
        'Reconstruction_MAE': mae_per_sample,
        'Reconstruction_RMSE': rmse_per_sample,
        'RRE': rre,
        'SNR': test_SNR_values,
        'Polarization': test_P.flatten(),
        'Area': test_area.flatten(),
        'Mean_Residual': mean_residual,
        'Std_Residual': std_residual,
    }


    results = pd.DataFrame(results_data)
    results.to_csv(f"{performance_dir}/{version}_results.csv", index=False)
    print(f"Results saved to {performance_dir}/{version}_results.csv")

    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(10, 6))
    plt.hist(rre, bins=30, alpha=0.7, edgecolor='darkblue')
    plt.xlabel('Relative Reconstruction Error (RRE)')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error Distribution')
    plt.figtext(0.65, 0.8, f"Mean: {rre.mean():.5f}%\nStd Dev: {rre.std():.5f}%", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", fc='blue', ec="none", alpha=0.8),
                color='white')
    plt.tight_layout()
    plt.savefig(f"{performance_dir}/{version}_rre_histogram.png", dpi=600)
    plt.close()
    print(f"RRE histogram saved to {performance_dir}/{version}_rre_histogram.png")
    