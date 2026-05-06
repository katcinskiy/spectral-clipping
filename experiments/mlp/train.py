import os
import sys
from collections import OrderedDict
from itertools import cycle

import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clipping import (  # noqa: E402
    SpectralClipperConst,
    SpectralClipperEMA,
    SpectralClipperQuantile,
)
from noise import add_lowrank_noise_to_gradients  # noqa: E402


class MultiplicationDataset(Dataset):
    def __init__(self, num_samples: int, input_dim: int):
        self.X = torch.randn(num_samples, input_dim)
        self.y = (self.X[:, 0] * self.X[:, 1] * self.X[:, 2]).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = OrderedDict()
        prev = input_dim
        for i, h in enumerate(hidden_dims, start=1):
            layers[f"W_{i}"] = nn.Linear(prev, h, bias=False)
            layers[f"ReLU_{i}"] = nn.ReLU()
            prev = h
        layers[f"W_{len(hidden_dims) + 1}"] = nn.Linear(prev, output_dim, bias=False)
        self.network = nn.Sequential(layers)

    def forward(self, x):
        return self.network(x)


def build_optimizer(cfg, model):
    if cfg.optimizer.type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.training.lr, momentum=cfg.optimizer.momentum)
    if cfg.optimizer.type == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    raise ValueError(f"Unknown optimizer.type={cfg.optimizer.type}")


def build_clip_fn(cfg, model):
    if not cfg.clipping.enabled:
        return None
    if cfg.clipping.type == "norm_clip":
        max_norm = float(cfg.clipping.max_norm)
        return lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    if cfg.clipping.type == "spectral_clip":
        mode = cfg.clipping.spectral_clip_mode
        if mode == "const":
            clipper = SpectralClipperConst(model, threshold=cfg.clipping.threshold)
        elif mode == "ema":
            clipper = SpectralClipperEMA(model, ema_coef=cfg.clipping.ema_coef)
        elif mode == "quantile":
            clipper = SpectralClipperQuantile(
                model, q=cfg.clipping.quantile_q, window_size=cfg.clipping.quantile_window_size,
            )
        else:
            raise ValueError(f"Unknown spectral_clip_mode={mode}")
        return lambda: clipper.clip(model)
    raise ValueError(f"Unknown clipping.type={cfg.clipping.type}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=cfg.wandb.project, entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    full = MultiplicationDataset(cfg.dataset.num_samples, cfg.model.input_dim)
    train_size = int(0.8 * len(full))
    train_ds, val_ds = random_split(full, [train_size, len(full) - train_size])
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False)
    train_iter = cycle(train_loader)

    model = MLP(cfg.model.input_dim, cfg.model.hidden_dims, cfg.model.output_dim).to(device)
    optimizer = build_optimizer(cfg, model)
    clip_fn = build_clip_fn(cfg, model)
    criterion = nn.MSELoss()

    log_freq = int(cfg.training.get("log_freq", 25))
    loss_hist: list[float] = []
    min_loss = float("inf")

    for step in tqdm(range(cfg.training.total_steps), desc="Training"):
        model.train()
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()

        if cfg.noise.enabled and torch.rand(1).item() < cfg.noise.prob:
            add_lowrank_noise_to_gradients(model, cfg.noise, device, cfg.noise.noise_rank)

        if clip_fn is not None:
            clip_fn()
        optimizer.step()

        loss_hist.append(loss.item())
        if torch.isnan(loss).any():
            raise RuntimeError("Training produced NaN loss")

        if (step + 1) % log_freq == 0 or step < 20:
            train_loss = sum(loss_hist) / len(loss_hist)
            min_loss = min(min_loss, min(loss_hist))
            wandb.log({"train/loss": train_loss}, step=step)
            loss_hist = []

        if (step + 1) % cfg.training.eval_steps == 0:
            model.eval()
            with torch.no_grad():
                val_loss = sum(
                    criterion(model(xv.to(device)), yv.to(device)).item() for xv, yv in val_loader
                ) / len(val_loader)
            wandb.log({"validation/loss": val_loss}, step=step)
            print(f"[Validation] step {step + 1}: val_loss={val_loss:.4f}")

    wandb.summary["train/min_loss"] = min_loss
    wandb.finish()


if __name__ == "__main__":
    main()
