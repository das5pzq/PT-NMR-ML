import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import warnings
warnings.filterwarnings('ignore')
import sys

POLARIZATION_RANGE = "HIGH_POL"  # Options: HIGH_POL (2% - 60), LOW_POL (TE - 2%)
USE_SE_BLOCK = POLARIZATION_RANGE == "HIGH_POL"
MAX_GRAD_NORM = 1.0

sys.stdout.flush()
sys.stderr.flush()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _device()


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
        self.input_length = input_length

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


class CNNLightningModule(nn.Module):
    """Thin wrapper around CNNArchitectureModel (no Lightning)."""

    def __init__(
        self,
        learning_rate=1e-3,
        input_length=500,
        num_residual_blocks=3,
        use_se_block=USE_SE_BLOCK,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.input_length = input_length
        self.num_residual_blocks = num_residual_blocks
        self.use_se_block = use_se_block
        self.model = CNNArchitectureModel(
            input_length=input_length,
            num_residual_blocks=num_residual_blocks,
            use_se_block=use_se_block,
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)


def _clone_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _materialize_lazy(model, input_length, device):
    """Run a dummy forward so LazyConv1d layers get real parameter tensors."""
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 1, input_length, device=device)
        model(dummy)
    model.train()


def _checkpoint_path(model_dir):
    return os.path.join(model_dir, "best_model_checkpoint.pth")


def _legacy_checkpoint_path(model_dir):
    return os.path.join(model_dir, "best_model_checkpoint.ckpt")


def save_checkpoint(path, *, model, optimizer, scheduler, epoch, best_val_loss,
                    learning_rate, input_length, num_residual_blocks, use_se_block,
                    history):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": _clone_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "learning_rate": learning_rate,
            "input_length": input_length,
            "num_residual_blocks": num_residual_blocks,
            "use_se_block": use_se_block,
            "history": history,
        },
        path,
    )


def load_model_from_checkpoint(path, device=None):
    if device is None:
        device = DEVICE
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if "model_state_dict" in ckpt:
        input_length = ckpt.get("input_length", 500)
        module = CNNLightningModule(
            learning_rate=ckpt.get("learning_rate", 1e-3),
            input_length=input_length,
            num_residual_blocks=ckpt.get("num_residual_blocks", 3),
            use_se_block=ckpt.get("use_se_block", USE_SE_BLOCK),
        ).to(device)
        _materialize_lazy(module, input_length, device)
        module.model.load_state_dict(ckpt["model_state_dict"])
        return module, ckpt

    # Legacy Lightning checkpoint
    hparams = ckpt.get("hyper_parameters", {})
    input_length = hparams.get("input_length", 500)
    module = CNNLightningModule(
        learning_rate=hparams.get("learning_rate", 1e-3),
        input_length=input_length,
        num_residual_blocks=hparams.get("num_residual_blocks", 3),
        use_se_block=hparams.get("use_se_block", USE_SE_BLOCK),
    ).to(device)
    _materialize_lazy(module, input_length, device)
    state = {
        k.replace("model.", "", 1): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    module.model.load_state_dict(state)
    return module, ckpt


def _run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_sum = 0.0
    mae_sum = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        mae = F.l1_loss(y_hat, y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        loss_sum += loss.item()
        mae_sum += mae.item()
        n_batches += 1

    n = max(n_batches, 1)
    return loss_sum / n, mae_sum / n


def train_model(X_train, y_train, X_val, y_val, X_test, y_test,
                model_dir, performance_dir, version,
                learning_rate=1e-3, max_epochs=2000, input_length=190, batch_size=256,
                device=None):

    if device is None:
        device = DEVICE
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        NMRDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        NMRDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        NMRDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    ckpt_path = _checkpoint_path(model_dir)
    legacy_ckpt = _legacy_checkpoint_path(model_dir)
    legacy_model = f"{model_dir}/best_model.ckpt"
    resume_path = None
    for candidate in (ckpt_path, legacy_ckpt, legacy_model):
        if os.path.isfile(candidate):
            resume_path = candidate
            break

    resume_ckpt = None
    if resume_path is not None:
        print(f"Loading existing model checkpoint from {resume_path}", flush=True)
        model, resume_ckpt = load_model_from_checkpoint(resume_path, device=device)
        model.learning_rate = learning_rate
        print("Model loaded successfully. Continuing training...", flush=True)
    else:
        print("No existing model found. Building new model...", flush=True)
        model = CNNLightningModule(
            learning_rate=learning_rate,
            input_length=input_length,
        ).to(device)
        _materialize_lazy(model, input_length, device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_trainable:,}', flush=True)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-5,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20,
        min_lr=1e-5,
    )

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
    }
    start_epoch = 0
    best_val_loss = float('inf')
    best_state = _clone_state_dict(model.model)

    if resume_ckpt is not None and "model_state_dict" in resume_ckpt:
        if resume_ckpt.get("optimizer_state_dict") is not None:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            except (ValueError, KeyError) as exc:
                print(f"Could not restore optimizer state ({exc}); continuing with fresh optimizer.", flush=True)
        if resume_ckpt.get("scheduler_state_dict") is not None:
            try:
                scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
            except (ValueError, KeyError) as exc:
                print(f"Could not restore scheduler state ({exc}); continuing with fresh scheduler.", flush=True)
        for pg in optimizer.param_groups:
            pg['lr'] = learning_rate
        start_epoch = int(resume_ckpt.get("epoch", 0))
        best_val_loss = float(resume_ckpt.get("best_val_loss", float('inf')))
        if resume_ckpt.get("history"):
            for col, values in resume_ckpt["history"].items():
                if col in history and values:
                    history[col] = list(values)

    best_pth_path = f"{model_dir}/best_model.pth"
    loss_history_path = f"{performance_dir}/{version}_loss.csv"

    for epoch in range(start_epoch, max_epochs):
        train_loss, train_mae = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_mae = _run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)

        print(
            f'epoch {epoch + 1:03d}/{max_epochs} | train {train_loss:.6f} | val {val_loss:.6f} | '
            f'train_mae {train_mae:.6f} | val_mae {val_mae:.6f} | lr {lr:.2e}',
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = _clone_state_dict(model.model)
            save_checkpoint(
                ckpt_path,
                model=model.model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_val_loss=best_val_loss,
                learning_rate=learning_rate,
                input_length=input_length,
                num_residual_blocks=model.num_residual_blocks,
                use_se_block=model.use_se_block,
                history=history,
            )
            torch.save(best_state, best_pth_path)
            print(f"Saved best model (lowest val_loss) weights to {best_pth_path}", flush=True)

    os.makedirs(os.path.dirname(loss_history_path) or ".", exist_ok=True)
    n = max(len(history['train_loss']), 1)
    pd.DataFrame({
        'epoch': range(1, n + 1),
        'train_loss': history['train_loss'] + [None] * (n - len(history['train_loss'])),
        'val_loss': history['val_loss'] + [None] * (n - len(history['val_loss'])),
        'train_mae': history['train_mae'] + [None] * (n - len(history['train_mae'])),
        'val_mae': history['val_mae'] + [None] * (n - len(history['val_mae'])),
    }).to_csv(loss_history_path, index=False)
    print(f"Saved loss and validation loss to {loss_history_path}", flush=True)

    if best_state is not None:
        model.model.load_state_dict(best_state)
    if not os.path.isfile(best_pth_path):
        torch.save(model.model.state_dict(), best_pth_path)
        print(
            f"Saved model weights to {best_pth_path} "
            f"(no best checkpoint path; using weights at end of training)",
            flush=True,
        )

    test_loss, test_mae = _run_epoch(model, test_loader, criterion, device, optimizer=None)
    print(f'test | loss {test_loss:.6f} | mae {test_mae:.6f}', flush=True)

    return model, history
