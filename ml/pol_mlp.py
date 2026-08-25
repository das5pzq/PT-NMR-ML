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
    """Shared trunk (2 hidden layers + LayerNorm) with separate P/Q heads."""

    def __init__(
        self,
        input_dim,
        hidden_dim=256,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.p_head = nn.Linear(hidden_dim, 1)
        self.q_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.trunk(x)
        p = self.p_head(h)
        q = self.q_head(h)
        return torch.cat([p, q], dim=-1)


class FFLightningModule(LightningModule):
    def __init__(
        self,
        input_dim=512,
        hidden_dim=256,
        learning_rate=1e-3,
        max_epochs=500,
        weight_decay=1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleFeedForward(
            input_dim,
            hidden_dim,
        )
        # L1 directly optimizes MAE; Huber with small delta was mostly quadratic.
        self.criterion = nn.L1Loss()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)


    def _split_losses(self, y_hat, y):
        loss_p = self.criterion(y_hat[:, 0:1], y[:, 0:1])
        loss_q = self.criterion(y_hat[:, 1:2], y[:, 1:2])
        return loss_p + loss_q, loss_p, loss_q

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss, loss_p, loss_q = self._split_losses(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss_p', loss_p, on_step=False, on_epoch=True)
        self.log('train_loss_q', loss_q, on_step=False, on_epoch=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss, loss_p, loss_q = self._split_losses(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss_p', loss_p, on_step=False, on_epoch=True)
        self.log('val_loss_q', loss_q, on_step=False, on_epoch=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss, loss_p, loss_q = self._split_losses(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_loss_p', loss_p, on_step=False, on_epoch=True)
        self.log('test_loss_q', loss_q, on_step=False, on_epoch=True)
        self.log('test_mae', mae, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=1e-7
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }


def _load_or_create_model(
    model_dir,
    input_dim,
    hidden_dim,
    learning_rate,
    max_epochs,
    weight_decay=1e-5,
):

    ckpt_path = os.path.join(model_dir, "best_model_checkpoint.ckpt")

    if not ckpt_path or not os.path.isfile(ckpt_path):
        print("No existing model found. Building new model...")
        return FFLightningModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            weight_decay=weight_decay,
        ), None

    print(f"Resuming from {ckpt_path})")
    model = FFLightningModule.load_from_checkpoint(
        ckpt_path,
        map_location='cuda',
        weights_only=False,
        max_epochs=max_epochs,
    )
    model.learning_rate = learning_rate
    model.max_epochs = max_epochs
    model.weight_decay = weight_decay

    return model, ckpt_path


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

    def on_fit_start(self, trainer, pl_module):
        if not trainer.optimizers:
            return
        for param_group in trainer.optimizers[0].param_groups:
            param_group['lr'] = pl_module.learning_rate


def train_model(X_train, y_train, X_val, y_val, X_test, y_test,
                model_dir, performance_dir, version,
                learning_rate=1e-3, max_epochs=500,
                hidden_dim=256, batch_size=256, weight_decay=1e-5,
                ):

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        NMRDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True, pin_memory=pin,
    )
    val_loader = DataLoader(
        NMRDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False, pin_memory=pin,
    )
    test_loader = DataLoader(
        NMRDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False, pin_memory=pin,
    )

    input_dim = X_train.shape[1]
    loss_history_path = f"{performance_dir}/{version}_loss.csv"
    prior_loss_history = _load_prior_loss_history(loss_history_path)

    model, fit_ckpt_path = _load_or_create_model(
        model_dir,
        input_dim,
        hidden_dim,
        learning_rate,
        max_epochs,
        weight_decay,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=model_dir,
            filename='best_model_checkpoint',
            monitor='val_mae',
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