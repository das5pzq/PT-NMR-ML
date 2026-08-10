import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import json
import warnings
warnings.filterwarnings('ignore')
import gc
import sys

sys.stdout.flush()
sys.stderr.flush()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def _accelerator():
    if torch.cuda.is_available():
        return 'cuda', torch.cuda.device_count()
    if torch.backends.mps.is_available():
        return 'mps', 1
    return 'cpu', 1

ACCELERATOR, N_DEVICES = _accelerator()

class NMRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleFeedForward(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class FFLightningModule(LightningModule):
    def __init__(
        self,
        input_dim=512,
        hidden_dim=256,
        learning_rate=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleFeedForward(
            input_dim,
            hidden_dim,
        )
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def _unpack_batch(self, batch):
        x, y = batch
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self._unpack_batch(batch)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mae', F.l1_loss(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._unpack_batch(batch)
        y_hat = self(x)
        self.log('val_loss', self.criterion(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', F.l1_loss(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = self._unpack_batch(batch)
        y_hat = self(x)
        self.log('test_loss', self.criterion(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_mae', F.l1_loss(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-7
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}


def _find_resume_checkpoint(model_dir):
    """Return (path, is_lightning_ckpt) preferring best validation weights."""
    best_ckpt = os.path.join(model_dir, "best_model_checkpoint.ckpt")
    if os.path.isfile(best_ckpt):
        return best_ckpt, True
    best_pth = os.path.join(model_dir, "best_model.pth")
    if os.path.isfile(best_pth):
        return best_pth, False
    return None, False


def _load_or_create_model(
    model_dir,
    input_dim,
    hidden_dim,
    learning_rate,
    resume=True,
    full_resume=False,
):
    if not resume:
        return FFLightningModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
        ), None

    ckpt_path, is_lightning_ckpt = _find_resume_checkpoint(model_dir)
    if ckpt_path is None:
        print("No existing model found. Building new model...")
        return FFLightningModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
        ), None

    fit_ckpt_path = ckpt_path if (full_resume and is_lightning_ckpt) else None
    if ckpt_path.endswith(".pth"):
        print(f"Loading best model weights from {ckpt_path}")
        model = FFLightningModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
        )
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.model.load_state_dict(state_dict)
    else:
        resume_kind = "full training state" if fit_ckpt_path else "best weights"
        print(f"Resuming from {ckpt_path} ({resume_kind})")
        model = FFLightningModule.load_from_checkpoint(ckpt_path, map_location='cuda', weights_only=False)
        model.learning_rate = learning_rate

    return model, fit_ckpt_path


def _load_prior_loss_history(save_path):
    if not os.path.isfile(save_path):
        return None
    history = pd.read_csv(save_path)
    if history.empty:
        return None
    return history


class LossHistoryCallback(Callback):
    def __init__(self, save_path, prior_history=None):
        super().__init__()
        self.save_path = save_path
        self.epoch_train_loss = []
        self.epoch_val_loss = []
        self.epoch_train_mae = []
        self.epoch_val_mae = []
        if prior_history is not None:
            for col, buf in (
                ("train_loss", self.epoch_train_loss),
                ("val_loss", self.epoch_val_loss),
                ("train_mae", self.epoch_train_mae),
                ("val_mae", self.epoch_val_mae),
            ):
                if col in prior_history.columns:
                    buf.extend(prior_history[col].dropna().tolist())

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        if 'train_loss' in m: self.epoch_train_loss.append(float(m['train_loss'].cpu()))
        if 'val_loss'   in m: self.epoch_val_loss.append(float(m['val_loss'].cpu()))
        if 'train_mae'  in m: self.epoch_train_mae.append(float(m['train_mae'].cpu()))
        if 'val_mae'    in m: self.epoch_val_mae.append(float(m['val_mae'].cpu()))

    def on_fit_end(self, trainer, pl_module):
        n = max(
            len(self.epoch_train_loss),
            len(self.epoch_val_loss),
            len(self.epoch_train_mae),
            len(self.epoch_val_mae),
        )
        if n == 0:
            return
        pd.DataFrame({
            'epoch': range(1, n + 1),
            'train_loss': self.epoch_train_loss + [None] * (n - len(self.epoch_train_loss)),
            'val_loss':   self.epoch_val_loss   + [None] * (n - len(self.epoch_val_loss)),
            'train_mae':  self.epoch_train_mae  + [None] * (n - len(self.epoch_train_mae)),
            'val_mae':    self.epoch_val_mae    + [None] * (n - len(self.epoch_val_mae)),
        }).to_csv(self.save_path, index=False)
        print(f"Saved loss history to {self.save_path}")


class SetLearningRateCallback(Callback):
    """Apply module learning_rate after Lightning restores a checkpoint."""

    def on_fit_start(self, trainer, pl_module):
        if not trainer.optimizers:
            return
        for param_group in trainer.optimizers[0].param_groups:
            param_group['lr'] = pl_module.learning_rate


def train_model(X_train, y_train, X_val, y_val, X_test, y_test,
                model_dir, performance_dir, version,
                num_workers=4, learning_rate=1e-3, max_epochs=500,
                hidden_dim=256, batch_size=256,
                resume=True, full_resume=False):

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        NMRDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        NMRDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        NMRDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=num_workers > 0,
    )

    input_dim = X_train.shape[1]
    loss_history_path = f"{performance_dir}/{version}_loss.csv"
    prior_loss_history = _load_prior_loss_history(loss_history_path) if resume else None

    model, fit_ckpt_path = _load_or_create_model(
        model_dir,
        input_dim,
        hidden_dim,
        learning_rate,
        resume=resume,
        full_resume=full_resume,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=model_dir,
            filename='best_model_checkpoint',
            monitor='val_loss',
            save_top_k=1,
            mode='min',
            save_last=True,
        ),
        LearningRateMonitor(),
        LossHistoryCallback(save_path=loss_history_path, prior_history=prior_loss_history),
    ]
    if fit_ckpt_path is not None:
        callbacks.append(SetLearningRateCallback())

    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=CSVLogger(performance_dir, name='training_log'),
        accelerator=ACCELERATOR,
        devices=N_DEVICES,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=fit_ckpt_path)

    checkpoint_cb = callbacks[0]
    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt and os.path.isfile(best_ckpt):
        best_module = FFLightningModule.load_from_checkpoint(best_ckpt, weights_only=False)
        torch.save(best_module.model.state_dict(), f"{model_dir}/best_model.pth")
        model = best_module
    else:
        torch.save(model.model.state_dict(), f"{model_dir}/best_model.pth")

    trainer.test(model, test_loader)
    return model, trainer


if __name__ == "__main__":
    data_path = "data/Training_Data_RGC_3_55_500K.parquet"
    version = 'RGC_MLP_3_55_V4'
    performance_dir = f"Model_Performance/{version}"
    model_dir = f"Models/{version}"
    os.makedirs(performance_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_parquet(data_path)
    signal_cols = df.columns[0:512]

    scaler_path = f"{performance_dir}/{version}_scaler.pkl"
    if os.path.isfile(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded existing scaler from {scaler_path}")
    else:
        scaler = MinMaxScaler()
        scaler.fit(df[signal_cols].values.astype('float32'))
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Fitted and saved new scaler to {scaler_path}")

    df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=1 / 3, random_state=42)
    del df, df_temp
    gc.collect()

    X_train = scaler.transform(df_train[signal_cols].values.astype('float32'))
    X_val = scaler.transform(df_val[signal_cols].values.astype('float32'))
    X_test = scaler.transform(df_test[signal_cols].values.astype('float32'))

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print(f"Scaled signal range (train): {X_train.min():.4f} to {X_train.max():.4f}")

    y_train = df_train["P"].values.astype('float32').reshape(-1, 1)
    y_val = df_val["P"].values.astype('float32').reshape(-1, 1)
    y_test = df_test["P"].values.astype('float32').reshape(-1, 1)
    test_SNR = df_test["SNR"].values.astype('float32')

    print(f"Number of training data points: {len(df_train)}")

    del df_train, df_val, df_test
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    num_workers  = 13
    learning_rate = 3e-4
    max_epochs   = 500
    hidden_dim   = 256
    batch_size   = 512

    print("\n" + "=" * 60)
    print("Training MLP Model")
    print("=" * 60)

    resume_training = True
    full_resume = False  # set True to also restore optimizer/epoch state from best_model_checkpoint.ckpt

    model, trainer = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        model_dir, performance_dir, version,
        num_workers=num_workers,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        resume=resume_training,
        full_resume=full_resume,
    )

    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)

    test_loader = DataLoader(NMRDataset(X_test, y_test), batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                             persistent_workers=True)

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            x, _ = batch
            predictions.append(model(x).cpu().numpy())

    y_pred = np.concatenate(predictions, axis=0)
    y_test_flat = y_test.flatten() * 100.0
    y_pred_flat = y_pred.flatten() * 100.0

    mse  = np.mean((y_test_flat - y_pred_flat) ** 2)
    mae  = np.mean(np.abs(y_test_flat - y_pred_flat))
    rmse = np.sqrt(mse)
    rpe  = np.abs(y_pred_flat - y_test_flat) / y_test_flat * 100

    print(f"\nTest Set Metrics:")
    print(f"  MSE:      {mse:.6f}")
    print(f"  MAE:      {mae:.6f}")
    print(f"  RMSE:     {rmse:.6f}")
    print(f"  Mean RPE: {rpe.mean():.5f}%")

    plt.style.use('ggplot')

    plt.hist(rpe, bins=30, alpha=0.7, edgecolor='red')
    plt.xlabel('Polarization RPE')
    plt.ylabel('Frequency')
    plt.title('Polarization RPE Distribution')
    plt.figtext(0.65, 0.8, f"Mean: {rpe.mean():.5f}%",
                fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc='red', ec="none", alpha=0.8),
                color='white')
    plt.tight_layout()
    plt.savefig(f"{performance_dir}/{version}_rpe_histogram.png", dpi=600)
    plt.close()

    residuals = y_test_flat - y_pred_flat

    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_flat, y_pred_flat, alpha=0.5, s=1)
    lo, hi = min(y_test_flat.min(), y_pred_flat.min()), max(y_test_flat.max(), y_pred_flat.max())
    plt.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Polarization (%)')
    plt.ylabel('Predicted Polarization (%)')
    plt.title('Actual vs Predicted Polarization')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{performance_dir}/{version}_actual_vs_predicted.png", dpi=600)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_flat, residuals, alpha=0.5, s=1)
    plt.axhline(0, color='r', linestyle='--', lw=2)
    plt.xlabel('Actual Polarization (%)')
    plt.ylabel('Residuals (%)')
    plt.title('Residuals Plot')
    plt.tight_layout()
    plt.savefig(f"{performance_dir}/{version}_residuals.png", dpi=600)
    plt.close()

    pd.DataFrame({
        'Actual': y_test_flat, 'Predicted': y_pred_flat,
        'Residuals': residuals, 'RPE': rpe, 'SNR': test_SNR,
    }).to_csv(f"{performance_dir}/{version}_results.csv", index=False)

    with open(f"{performance_dir}/{version}_metrics_summary.json", "w") as f:
        json.dump({
            'MSE': float(mse), 'MAE': float(mae), 'RMSE': float(rmse),
            'Mean_RPE': float(rpe.mean()), 'Std_RPE': float(rpe.std()),
        }, f, indent=4)

    loss_csv = f"{performance_dir}/{version}_loss.csv"
    if os.path.exists(loss_csv):
        h = pd.read_csv(loss_csv)
        plt.figure()
        plt.plot(h['train_loss'].dropna(), label='Train Loss')
        plt.plot(h['val_loss'].dropna(), label='Val Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f"{performance_dir}/{version}_loss.png", dpi=600)
        plt.close()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
