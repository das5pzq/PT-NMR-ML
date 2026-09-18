import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule
import random
import warnings
warnings.filterwarnings('ignore')
import sys

POLARIZATION_RANGE = "HIGH_POL"  # Options: HIGH_POL (2% - 60), LOW_POL (TE - 2%)
USE_SE_BLOCK = POLARIZATION_RANGE == "HIGH_POL"

sys.stdout.flush()

sys.stderr.flush()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def _accelerator_and_device():
    if torch.cuda.is_available():
        return "cuda", torch.cuda.device_count(), torch.device("cuda")
    if torch.backends.mps.is_available():
        return "mps", 1, torch.device("mps")
    return "cpu", 1, torch.device("cpu")


ACCELERATOR, N_DEVICES, device = _accelerator_and_device()


class NMRDataset(Dataset):
    """Buffers X, y as NumPy float32; __getitem__ uses torch.from_numpy (no full tensor copy)."""

    def __init__(self, X, y):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.float32)
        if self.X.ndim != 2:
            raise ValueError("X must be 2D (N, length)")
        if self.y.ndim == 1:
            self.y = self.y.reshape(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # (length,) -> (1, length) for Conv1d; row is a view, no copy until collate stacks batches.
        x_row = torch.from_numpy(self.X[idx]).unsqueeze(0)
        y_row = torch.from_numpy(self.y[idx])
        return x_row, y_row

class InceptionBlock(nn.Module):
    """
    Inception block with four parallel Conv1D layers (kernel sizes 1, 3, 5, 3 + max pool) -> concatenate
    """
    def __init__(self, c1, c2, c3, c4):
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.LazyConv1d(c1, kernel_size=1),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.LazyConv1d(c2[0], kernel_size=1),
            nn.ReLU(),
            nn.LazyConv1d(c2[1], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.LazyConv1d(c3[0], kernel_size=1),
            nn.ReLU(),
            nn.LazyConv1d(c3[1], kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.LazyConv1d(c4, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            dim=1,
        )

class ResidualBlock(nn.Module):

    """Residual block with two Conv1D layers and batch normalization"""

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        
        # Skip connection - identity if same channels, otherwise 1x1 conv

        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.conv1(x))
        out = self.conv2(out) 
        out = self.bn(out)
        out = out + residual
        return out

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block
    """
    def __init__(self, channels, reduction=2):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction) 
        self.fc2 = nn.Linear(channels // reduction, channels) 
    
    def forward(self, x):

        # x shape: (batch, channels, length) = (batch, channels, length)
        
        # Global average pooling
        y = self.global_pool(x).squeeze(-1)  # (batch, channels)
        
        y = F.relu(self.fc1(y))  # (batch, channels // reduction)
        y = torch.sigmoid(self.fc2(y))  # (batch, channels)
        
        # Reshape to (batch, channels, 1) for broadcasting with (batch, channels, length)
        y = y.unsqueeze(-1)  # (batch, channels, 1)
        return x * y  # (batch, channels, length) * (batch, channels, 1) -> (batch, channels, length)

class CNNArchitectureModel(nn.Module):
    """
    Architecture:
    1. Inception block (4 parallel Conv1D layers (kernel sizes 1, 3, 5, 3 + max pool)) -> concatenate
    2. Residual blocks (2 Conv1D layers each)
    3. Optional SE (Squeeze-and-Excitation) block (LOW_POL only)
    4. Global average pooling -> FC + ReLU -> output
    """
    def __init__(self, input_length=190, num_residual_blocks=3, use_se_block=USE_SE_BLOCK):
        super(CNNArchitectureModel, self).__init__()
        self.use_se_block = use_se_block

        c1 = 64
        c2 = (32, 32 * 3)  # 32 channels, 96 filters
        c3 = (32, 32 * 5)  # 32 channels, 160 filters
        c4 = 32
        channels = c1 + c2[1] + c3[1] + c4  # 64 + 96 + 160 + 32 = 352

        self.inception_block = InceptionBlock(c1, c2, c3, c4)

        self.residual_blocks = nn.ModuleList(
            ResidualBlock(channels, channels) for _ in range(num_residual_blocks)
        )

        self.se_block = SEBlock(channels, reduction=2) if use_se_block else None

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch, 1, length)
        x = self.inception_block(x)

        for residual_block in self.residual_blocks:
            x = residual_block(x)

        if self.se_block is not None:
            x = self.se_block(x)

        x = self.global_pool(x)
        x = x.flatten(1)
        x = F.relu(self.fc(x))
        return self.output(x)

class CNNLightningModule(LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        input_length=500,
        num_residual_blocks=3,
        use_se_block=USE_SE_BLOCK,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = CNNArchitectureModel(
            input_length=input_length,
            num_residual_blocks=num_residual_blocks,
            use_se_block=use_se_block,
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        mae = F.l1_loss(y_hat, y)
    
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_mae', mae, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-5
        )

        # ReduceLROnPlateau never raises the LR, so val_mae declines monotonically
        # instead of spiking on every warm-restart cycle.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=20,
            min_lr=1e-5,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }


class LossHistoryCallback(Callback):
    """Records epoch train_loss and val_loss and saves to CSV at end of training."""
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.epoch_train_loss = []
        self.epoch_val_loss = []
        self.epoch_train_mae = []
        self.epoch_val_mae = []

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if 'train_loss' in metrics:
            self.epoch_train_loss.append(float(metrics['train_loss'].cpu()))
        if 'val_loss' in metrics:
            self.epoch_val_loss.append(float(metrics['val_loss'].cpu()))
        if 'train_mae' in metrics:
            self.epoch_train_mae.append(float(metrics['train_mae'].cpu()))
        if 'val_mae' in metrics:
            self.epoch_val_mae.append(float(metrics['val_mae'].cpu()))

    def on_fit_end(self, trainer, pl_module):
        if not self.epoch_train_loss and not self.epoch_val_loss:
            return
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        n = max(len(self.epoch_train_loss), len(self.epoch_val_loss))
        # Pad shorter list so we have one row per epoch
        train_loss = self.epoch_train_loss + [None] * (n - len(self.epoch_train_loss))
        val_loss = self.epoch_val_loss + [None] * (n - len(self.epoch_val_loss))
        train_mae = self.epoch_train_mae + [None] * (n - len(self.epoch_train_mae))
        val_mae = self.epoch_val_mae + [None] * (n - len(self.epoch_val_mae))
        history_df = pd.DataFrame({
            'epoch': range(1, n + 1),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_mae': train_mae,
            'val_mae': val_mae,
        })
        history_df.to_csv(self.save_path, index=False)
        print(f"Saved loss and validation loss to {self.save_path}")


def train_model(X_train, y_train, X_val, y_val, X_test, y_test, 
                model_dir, performance_dir, version, 
                learning_rate=1e-3, max_epochs=2000, input_length=190, batch_size=256):
    
    pin_memory = torch.cuda.is_available()

    train_dataset = NMRDataset(X_train, y_train)
    val_dataset = NMRDataset(X_val, y_val)
    test_dataset = NMRDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=pin_memory, 
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=pin_memory, 
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=pin_memory, 
    )
    
    checkpoint_path = f"{model_dir}/best_model_checkpoint.ckpt"
    model_path = f"{model_dir}/best_model.ckpt"
    
    if os.path.exists(checkpoint_path):
        print(f"Loading existing model checkpoint from {checkpoint_path}")
        model = CNNLightningModule.load_from_checkpoint(checkpoint_path)
        print("Model loaded successfully. Continuing training...")
    elif os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = CNNLightningModule.load_from_checkpoint(model_path)
        print("Model loaded successfully. Continuing training...")
    else:
        print("No existing model found. Building new model...")
        model = CNNLightningModule(learning_rate=learning_rate, input_length=input_length)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename='best_model_checkpoint',
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        save_last=True
    )
    
    loss_history_path = f"{performance_dir}/{version}_loss.csv"
    loss_history_callback = LossHistoryCallback(save_path=loss_history_path)
    
    csv_logger = CSVLogger(performance_dir, name='training_log')
    
    trainer_kwargs = {
        'max_epochs': max_epochs,
        'callbacks': [
            # EarlyStopping(monitor='val_loss', patience=50, mode='min'),
            checkpoint_callback,
            LearningRateMonitor(),
            loss_history_callback,
        ],
        'logger': csv_logger,
        'enable_progress_bar': True
    }
    
    trainer_kwargs["accelerator"] = ACCELERATOR
    trainer_kwargs["devices"] = N_DEVICES
    trainer_kwargs['gradient_clip_val'] = 1.0
    
    trainer = Trainer(**trainer_kwargs)
    
    trainer.fit(model, train_loader, val_loader)
    
    best_pth_path = f"{model_dir}/best_model.pth"
    best_ckpt = checkpoint_callback.best_model_path
    if best_ckpt and os.path.isfile(best_ckpt):
        best_module = CNNLightningModule.load_from_checkpoint(best_ckpt)
        torch.save(best_module.model.state_dict(), best_pth_path)
        print(f"Saved best model (lowest val_loss) weights to {best_pth_path}")
    else:
        torch.save(model.model.state_dict(), best_pth_path)
        print(
            f"Saved model weights to {best_pth_path} "
            f"(no best checkpoint path; using weights at end of training)"
        )

    # Reload best weights for test/evaluation when a best checkpoint exists
    if best_ckpt and os.path.isfile(best_ckpt):
        model = CNNLightningModule.load_from_checkpoint(best_ckpt)

    trainer.test(model, test_loader)
    
    return model, trainer