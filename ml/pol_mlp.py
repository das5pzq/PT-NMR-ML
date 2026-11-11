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
# Floor for the relative loss, in physical polarization units.
REL_LOSS_EPS = 1e-4


def _device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = _device()
DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pol_mlp.pth")


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


class ResidualBlock(nn.Module):
    """Two Linear-LayerNorm-GELU layers plus an identity skip. The last linear starts at zero so the block is identity at init."""

    def __init__(self, dim):
        super().__init__()
        if int(dim) < 1:
            raise ValueError(f'dim must be positive, got {dim}')
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x):
        hidden = self.act(self.ln1(self.fc1(x)))
        return x + self.ln2(self.fc2(hidden))


class PolMLP(nn.Module):
    """Spectrum in, P and Q out. ``[B, bins]`` or ``[B, bins, channels]`` → ``[B, 2]``.

    With ``out_stats`` set, P and Q are physical fractions. Otherwise they are
    the network's z-scored outputs.
    """

    def __init__(self, input_size=1, hidden=16, num_blocks=2, num_bins=500):
        super().__init__()
        if int(input_size) < 1:
            raise ValueError(f'input_size must be positive, got {input_size}')
        if int(hidden) < 1:
            raise ValueError(f'hidden must be positive, got {hidden}')
        if int(num_blocks) < 1:
            raise ValueError(f'num_blocks must be positive, got {num_blocks}')
        if int(num_bins) < 1:
            raise ValueError(f'num_bins must be positive, got {num_bins}')
        self.input_size = int(input_size)
        self.hidden = int(hidden)
        self.num_blocks = int(num_blocks)
        self.num_bins = int(num_bins)
        self.out_stats = None
        in_dim = self.input_size * self.num_bins
        self.stem = nn.Sequential(nn.Linear(in_dim, self.hidden), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualBlock(self.hidden) for _ in range(self.num_blocks)])
        self.decoder = nn.Linear(self.hidden, self.num_bins)
        self.head_p = nn.Linear(self.num_bins, 1)
        self.head_q = nn.Linear(self.num_bins, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in (self.stem, self.blocks):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.zeros_(layer.bias)
        for block in self.blocks:
            nn.init.zeros_(block.fc2.weight)
            nn.init.zeros_(block.fc2.bias)
        for layer in (self.decoder, self.head_p, self.head_q):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        if x.ndim == 2:
            x = x.reshape(x.shape[0], self.num_bins, self.input_size)
        if x.ndim != 3 or x.shape[-1] != self.input_size or x.shape[1] != self.num_bins:
            raise ValueError(
                f'PolMLP expects input [B, {self.num_bins}, {self.input_size}], got shape {tuple(x.shape)}'
            )
        recon = self.decoder(self.blocks(self.stem(x.flatten(1))))
        pred_p = self.head_p(recon).squeeze(-1)
        pred_q = self.head_q(recon).squeeze(-1)
        if self.out_stats is not None:
            p_mean, p_std, q_mean, q_std = self.out_stats
            pred_p = pred_p * p_std + p_mean
            pred_q = pred_q * q_std + q_mean
        return torch.stack((pred_p, pred_q), dim=-1)


class FFLightningModule(nn.Module):
    def __init__(
        self,
        input_dim=512,
        hidden_dim=256,
        learning_rate=1e-3,
        max_epochs=500,
        weight_decay=1e-5,
        model=None,
    ):
        super().__init__()
        self.model = SimpleFeedForward(input_dim, hidden_dim) if model is None else model
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return self.model(x)


def relative_weighted_loss(pred, true, eps=REL_LOSS_EPS):
    """Mean absolute relative error: |pred - true| / (|true| + eps).

    P and Q are already physical fractions, so there is no mean/std
    denormalization before the ratio.
    """
    return torch.mean(torch.abs(pred - true) / (torch.abs(true) + eps))


def _split_losses(y_hat, y):
    loss_p = relative_weighted_loss(y_hat[:, 0:1], y[:, 0:1])
    loss_q = relative_weighted_loss(y_hat[:, 1:2], y[:, 1:2])
    return loss_p + loss_q, loss_p, loss_q


def _clone_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_checkpoint(path, *, model, optimizer, scheduler, epoch, best_val_loss,
                    input_dim, hidden_dim, learning_rate, max_epochs, weight_decay,
                    history):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": _clone_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "weight_decay": weight_decay,
            "history": history,
        },
        path,
    )


def _is_pol_mlp_checkpoint(ckpt):
    state = ckpt.get("model_state_dict")
    return isinstance(state, dict) and "stem.0.weight" in state and "head_p.weight" in state


def _pol_mlp_from_checkpoint(ckpt):
    stats = ckpt.get("stats") or {}
    model = PolMLP(
        input_size=int(ckpt.get("input_size", stats.get("input_size", 1))),
        hidden=int(ckpt.get("hidden", 16)),
        num_blocks=int(ckpt.get("num_blocks", 2)),
        num_bins=int(ckpt.get("num_bins", stats.get("num_bins", 500))),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    if all(key in stats for key in ("P_mean", "P_std", "Q_mean", "Q_std")):
        model.out_stats = (
            float(stats["P_mean"]),
            float(stats["P_std"]),
            float(stats["Q_mean"]),
            float(stats["Q_std"]),
        )
    return model


def load_model_from_checkpoint(path=None, device=None):
    if device is None:
        device = DEVICE
    if path is None:
        path = DEFAULT_CHECKPOINT
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if _is_pol_mlp_checkpoint(ckpt):
        model = _pol_mlp_from_checkpoint(ckpt)
        module = FFLightningModule(
            input_dim=model.num_bins * model.input_size,
            hidden_dim=model.hidden,
            model=model,
        )
        module.to(device)
        return module, ckpt

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


def _run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_sum = 0.0
    mae_sum = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss, _, _ = _split_losses(y_hat, y)
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

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    # First cycle decays within ~20 epochs, then restarts with longer cycles.
    # A single cosine over max_epochs leaves the rate unchanged while val_mae stalls.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-6,
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
    best_val_loss = float('inf')
    best_state = _clone_state_dict(model.model)

    if resume_ckpt is not None and "model_state_dict" in resume_ckpt:
        if resume_ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        sched_state = resume_ckpt.get("scheduler_state_dict")
        if sched_state is not None and "T_0" in sched_state:
            scheduler.load_state_dict(sched_state)
        else:
            for pg in optimizer.param_groups:
                pg['lr'] = learning_rate
            if sched_state is not None:
                print(
                    "Checkpoint scheduler is not cosine warm restarts; starting a new schedule.",
                    flush=True,
                )
        start_epoch = resume_ckpt.get("epoch", 0)
        # Older checkpoints stored unweighted L1 as best_val_mae; that scale
        # is not comparable to the relative P+Q loss.
        best_val_loss = resume_ckpt.get("best_val_loss", float('inf'))
        if resume_ckpt.get("history"):
            for col, values in resume_ckpt["history"].items():
                if col in history and values:
                    history[col] = list(values)

    ckpt_path = os.path.join(model_dir, "best_model_checkpoint.pth")
    best_pth_path = f"{model_dir}/best_model.pth"

    for epoch in range(start_epoch, max_epochs):
        train_loss, train_mae = _run_epoch(
            model, train_loader, device, optimizer=optimizer
        )
        val_loss, val_mae = _run_epoch(
            model, val_loader, device, optimizer=None
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

    test_loss, test_mae = _run_epoch(model, test_loader, device, optimizer=None)
    print(f'test | loss {test_loss:.6f} | mae {test_mae:.6f}', flush=True)

    return model, history


def _check_pol_mlp():
    fresh = PolMLP(input_size=1, hidden=8, num_blocks=2, num_bins=20)
    block_in = torch.randn(4, 8)
    with torch.no_grad():
        assert torch.allclose(fresh.blocks[0](block_in), block_in)
        raw = fresh(torch.randn(4, 20))
    assert tuple(raw.shape) == (4, 2) and torch.isfinite(raw).all()
    loaded = _pol_mlp_from_checkpoint({
        "model_state_dict": {k: v.clone() for k, v in fresh.state_dict().items()},
        "input_size": 1,
        "hidden": 8,
        "num_blocks": 2,
        "num_bins": 20,
        "stats": {"P_mean": 0.1, "P_std": 2.0, "Q_mean": -0.3, "Q_std": 4.0},
    })
    x = torch.randn(4, 20)
    with torch.no_grad():
        got = loaded(x)
        expect = fresh(x) * torch.tensor([2.0, 4.0]) + torch.tensor([0.1, -0.3])
    assert torch.allclose(got, expect)
    module, _ = load_model_from_checkpoint(DEFAULT_CHECKPOINT, device=torch.device("cpu"))
    assert isinstance(module.model, PolMLP)
    pred = module(torch.zeros(2, module.model.num_bins))
    assert tuple(pred.shape) == (2, 2) and torch.isfinite(pred).all()


if __name__ == "__main__":
    _check_pol_mlp()
