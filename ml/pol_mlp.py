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

sys.stdout.flush()
sys.stderr.flush()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

MAX_GRAD_NORM = 1.0


def _device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = _device()


class NMRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
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


class FFLightningModule(nn.Module):
    def __init__(
        self,
        input_dim=512,
        hidden_dim=256,
        learning_rate=1e-3,
        max_epochs=500,
        weight_decay=1e-5,
    ):
        super().__init__()
        self.model = SimpleFeedForward(input_dim, hidden_dim)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return self.model(x)


def _split_losses(criterion, y_hat, y):
    loss_p = criterion(y_hat[:, 0:1], y[:, 0:1])
    loss_q = criterion(y_hat[:, 1:2], y[:, 1:2])
    return loss_p + loss_q, loss_p, loss_q


def _clone_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_checkpoint(path, *, model, optimizer, scheduler, epoch, best_val_mae,
                    input_dim, hidden_dim, learning_rate, max_epochs, weight_decay,
                    history):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": _clone_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_mae": best_val_mae,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "weight_decay": weight_decay,
            "history": history,
        },
        path,
    )


def load_model_from_checkpoint(path, device=None):
    if device is None:
        device = DEVICE
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if "model_state_dict" in ckpt:
        module = FFLightningModule(
            input_dim=ckpt.get("input_dim", 512),
            hidden_dim=ckpt.get("hidden_dim", 256),
            learning_rate=ckpt.get("learning_rate", 1e-3),
            max_epochs=ckpt.get("max_epochs", 500),
            weight_decay=ckpt.get("weight_decay", 1e-5),
        )
        module.model.load_state_dict(ckpt["model_state_dict"])
        module.to(device)
        return module, ckpt

    hparams = ckpt.get("hyper_parameters", {})
    module = FFLightningModule(
        input_dim=hparams.get("input_dim", 512),
        hidden_dim=hparams.get("hidden_dim", 256),
        learning_rate=hparams.get("learning_rate", 1e-3),
        max_epochs=hparams.get("max_epochs", 500),
        weight_decay=hparams.get("weight_decay", 1e-5),
    )
    state = {
        k.replace("model.", "", 1): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    module.model.load_state_dict(state)
    module.to(device)
    return module, ckpt


def _load_or_create_model(
    model_dir,
    input_dim,
    hidden_dim,
    learning_rate,
    max_epochs,
    weight_decay=1e-5,
    device=None,
):
    if device is None:
        device = DEVICE

    ckpt_path = os.path.join(model_dir, "best_model_checkpoint.pth")
    legacy_path = os.path.join(model_dir, "best_model_checkpoint.ckpt")
    resume_path = ckpt_path if os.path.isfile(ckpt_path) else (
        legacy_path if os.path.isfile(legacy_path) else None
    )

    if resume_path is None:
        print("No existing model found. Building new model...", flush=True)
        model = FFLightningModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            weight_decay=weight_decay,
        ).to(device)
        return model, None

    print(f"Resuming from {resume_path}", flush=True)
    model, ckpt = load_model_from_checkpoint(resume_path, device=device)
    model.learning_rate = learning_rate
    model.max_epochs = max_epochs
    model.weight_decay = weight_decay
    return model, ckpt


def _load_prior_loss_history(save_path):
    if not os.path.isfile(save_path):
        return None
    history = pd.read_csv(save_path)
    return None if history.empty else history


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
        loss, _, _ = _split_losses(criterion, y_hat, y)
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
                learning_rate=1e-3, max_epochs=500,
                hidden_dim=256, batch_size=256, weight_decay=1e-5,
                device=None):

    if device is None:
        device = DEVICE
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

    model, resume_ckpt = _load_or_create_model(
        model_dir,
        input_dim,
        hidden_dim,
        learning_rate,
        max_epochs,
        weight_decay,
        device=device,
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_trainable:,}', flush=True)

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-7
    )

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
    }
    if prior_loss_history is not None:
        for col in history:
            if col in prior_loss_history.columns:
                history[col].extend(prior_loss_history[col].dropna().tolist())

    start_epoch = 0
    best_val_mae = float('inf')
    best_state = _clone_state_dict(model.model)

    if resume_ckpt is not None and "model_state_dict" in resume_ckpt:
        if resume_ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if resume_ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        for pg in optimizer.param_groups:
            pg['lr'] = learning_rate
        start_epoch = resume_ckpt.get("epoch", 0)
        best_val_mae = resume_ckpt.get("best_val_mae", float('inf'))
        if resume_ckpt.get("history"):
            for col, values in resume_ckpt["history"].items():
                if col in history and values:
                    history[col] = list(values)

    ckpt_path = os.path.join(model_dir, "best_model_checkpoint.pth")
    best_pth_path = f"{model_dir}/best_model.pth"

    for epoch in range(start_epoch, max_epochs):
        train_loss, train_mae = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_mae = _run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )
        scheduler.step()
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

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = _clone_state_dict(model.model)
            save_checkpoint(
                ckpt_path,
                model=model.model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_val_mae=best_val_mae,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                weight_decay=weight_decay,
                history=history,
            )
            torch.save(best_state, best_pth_path)

    pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_mae': history['train_mae'],
        'val_mae': history['val_mae'],
    }).to_csv(loss_history_path, index=False)
    print(f"Saved loss history to {loss_history_path}", flush=True)

    model.model.load_state_dict(best_state)
    torch.save(best_state, best_pth_path)

    test_loss, test_mae = _run_epoch(model, test_loader, criterion, device, optimizer=None)
    print(f'test | loss {test_loss:.6f} | mae {test_mae:.6f}', flush=True)

    return model, history
