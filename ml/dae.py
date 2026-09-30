import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import os
import pickle
import gc
from sklearn.preprocessing import MinMaxScaler

sys.stdout.flush()
sys.stderr.flush()

NUM_EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 1e-2
DEFAULT_NOISE_FACTOR = 3 * 2.690506959957014e-05
MAX_GRAD_NORM = 1.0
EARLY_STOP_PATIENCE = 20


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _device()


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim=500, hidden_dims=(256, 128, 64, 32, 16)):
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

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class AE(nn.Module):
    def __init__(
        self,
        noise_factor=DEFAULT_NOISE_FACTOR,
        scaler=None,
        input_dim=500,
        hidden_dims=(256, 128, 64, 32, 16),
    ):
        super(AE, self).__init__()
        self.noise_factor = noise_factor
        self.scaler = scaler
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.net = DenoisingAutoencoder(input_dim=input_dim, hidden_dims=hidden_dims)

    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_factor
        return x + noise

    def noisy_scaled_batch(self, x_clean_scaled):
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


def _clone_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        x, _, _ = batch
        x = x.to(device).view(x.size(0), -1)
        x_clean_scaled = x.clone()
        x_noisy_scaled = model.noisy_scaled_batch(x_clean_scaled)

        optimizer.zero_grad()
        decoded = model(x_noisy_scaled)
        loss = loss_fn(decoded, x_clean_scaled)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            x, _, _ = batch
            x = x.to(device).view(x.size(0), -1)
            x_clean_scaled = x.clone()
            x_noisy_scaled = model.noisy_scaled_batch(x_clean_scaled)
            decoded = model(x_noisy_scaled)
            loss = loss_fn(decoded, x_clean_scaled)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches if n_batches > 0 else 0.0


if __name__ == '__main__':
    from stable_minmax import StableMinMaxScaler

    SEED = 42

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    version = 'Exp2_DAE_V1'

    performance_dir = f"Model_Performance/{version}"
    model_dir = f"Models/{version}"
    os.makedirs(performance_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    device = DEVICE
    print(f'Using device: {device}', flush=True)

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

    print(f"Shape of X: {X.shape}", flush=True)
    print(f"Range of X: {X.min():.4f} to {X.max():.4f}", flush=True)
    print(f"Range of P: {P.min():.4f} to {P.max():.4f}", flush=True)
    print(f"Range of area: {area.min():.4f} to {area.max():.4f}", flush=True)

    dataset = data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(P, dtype=torch.float32),
        torch.tensor(area, dtype=torch.float32),
    )

    del df, X, P, area
    gc.collect()

    train_dataset, val_dataset, test_dataset = data.random_split(dataset, [0.80, 0.10, 0.10])

    train_loader = data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=13, persistent_workers=True
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=13, persistent_workers=True
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=13, persistent_workers=True
    )

    model = AE(noise_factor=DEFAULT_NOISE_FACTOR, scaler=X_Scaler).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_trainable:,}', flush=True)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=max(1e-8, LEARNING_RATE * 1e-4)
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state = None
    history = {'train_loss': [], 'val_loss': []}

    print('Training DAE model...', flush=True)
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(
            f'Epoch {epoch + 1}: train={train_loss:.6f} val={val_loss:.6f} '
            f'lr={scheduler.get_last_lr()[0]:.2e}',
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = _clone_state_dict(model)
            torch.save(
                {
                    'model_state_dict': best_state,
                    'noise_factor': model.noise_factor,
                    'input_dim': model.input_dim,
                    'hidden_dims': model.hidden_dims,
                    'best_val_loss': best_val_loss,
                    'epoch': epoch + 1,
                },
                f"{model_dir}/best_model_checkpoint.pth",
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f'Early stop at epoch {epoch + 1} (best val {best_val_loss:.6f})', flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
    }).to_csv(f"{performance_dir}/{version}_loss.csv", index=False)

    final_model_path = f"{model_dir}/{version}_final_model.pth"
    torch.save(
        {
            'model_state_dict': _clone_state_dict(model),
            'noise_factor': model.noise_factor,
            'input_dim': model.input_dim,
            'hidden_dims': model.hidden_dims,
            'best_val_loss': best_val_loss,
        },
        final_model_path,
    )
    print(f"\nFinal model saved to: {final_model_path}", flush=True)

    state_dict_path = f"{model_dir}/{version}_state_dict.pth"
    torch.save(model.state_dict(), state_dict_path)
    print(f"Model state dict saved to: {state_dict_path}", flush=True)

    test_loss = evaluate(model, test_loader, loss_fn, device)
    print(f'\nTest MSE (normalized): {test_loss:.6f}', flush=True)

    print("\nCollecting test predictions for analysis...", flush=True)
    model.eval()
    model = model.to(device)
    test_X_actual = []
    test_X_noisy = []
    test_X_predicted = []
    test_P = []
    test_area = []
    test_reconstruction_errors = []

    with torch.no_grad():
        for batch in test_loader:
            x, p, area_batch = batch
            x = x.to(device)
            x_clean_scaled = x.clone().view(x.size(0), -1)

            x_noisy_scaled = model.noisy_scaled_batch(x_clean_scaled)
            decoded_scaled = model.forward(x_noisy_scaled)

            x_clean_unscaled = model.unscale_with_scaler(x_clean_scaled)
            x_noisy_unscaled = model.unscale_with_scaler(x_noisy_scaled)
            decoded_unscaled = model.unscale_with_scaler(decoded_scaled)

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
    print(f"\nNoise verification (physical units): mean |noisy - clean| = {noise_mag_phys:.6f}", flush=True)
    print(f"Noise std (raw spectrum units, before scaling): {model.noise_factor}", flush=True)
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
    print(f"Reconstruction example saved to {performance_dir}/{version}_reconstruction_example.pdf", flush=True)

    mse_per_sample_unscaled = np.mean((test_X_predicted_unscaled - test_X_actual_unscaled) ** 2, axis=1)
    mae_per_sample = np.mean(np.abs(test_X_predicted_unscaled - test_X_actual_unscaled), axis=1)
    rmse_per_sample = np.sqrt(mse_per_sample_unscaled)

    signal_magnitude = np.sqrt(np.sum(test_X_actual_unscaled ** 2, axis=1))
    reconstruction_error_magnitude = np.sqrt(np.sum((test_X_predicted_unscaled - test_X_actual_unscaled) ** 2, axis=1))
    rre = (reconstruction_error_magnitude / (signal_magnitude + 1e-10)) * 100

    mean_residual = np.mean(test_X_predicted_unscaled - test_X_actual_unscaled)
    std_residual = np.std(test_X_predicted_unscaled - test_X_actual_unscaled)
    print(f"Mean residual: {mean_residual:.6e}", flush=True)
    print(f"Std residual: {std_residual:.6e}", flush=True)

    print("\nSaving results...", flush=True)
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
    print(f"Results saved to {performance_dir}/{version}_results.csv", flush=True)

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
    print(f"RRE histogram saved to {performance_dir}/{version}_rre_histogram.png", flush=True)
    print('Done.', flush=True)
