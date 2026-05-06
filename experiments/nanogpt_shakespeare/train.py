import os
import sys
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for path in (_REPO_ROOT, _HERE):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from clipping import (  # noqa: E402
    clip_singular_values_randomized_svd,
    clip_singular_values_svd,
)
from data import build_dataloaders  # noqa: E402
from model import GPT  # noqa: E402
from muon import Muon  # noqa: E402


def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizers(cfg, model):
    adam_params, matrix_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (adam_params if "emb" in name.lower() or p.ndim == 1 else matrix_params).append(p)

    lr = cfg.optimizer.learning_rate
    optimizers = [torch.optim.Adam(adam_params, lr=lr, betas=(0.9, 0.95), eps=1e-8)]
    if cfg.optimizer.optimizer_name == "sgdm":
        optimizers.append(torch.optim.SGD(
            matrix_params, lr=lr, momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay,
        ))
    elif cfg.optimizer.optimizer_name == "muon":
        optimizers.append(Muon(
            matrix_params, learning_rate=lr, momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay,
        ))
    else:
        raise ValueError(f"optimizer_name must be 'sgdm' or 'muon', got {cfg.optimizer.optimizer_name}")
    return optimizers, matrix_params


def build_lr_schedulers(cfg, optimizers, total_steps):
    if not bool(cfg.train.use_lr_scheduler):
        return []
    warmup = max(1, int(cfg.train.warmup_steps))
    out = []
    for opt in optimizers:
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0 / warmup, total_iters=warmup)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, total_steps - warmup), eta_min=float(cfg.train.min_lr),
        )
        out.append(torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warm, cos], milestones=[warmup],
        ))
    return out


def build_clipper(cfg, model, matrix_params):
    spectral = cfg.optimizer.get("spectral_clip", None)
    if spectral and bool(spectral.enabled):
        if str(spectral.mode) != "const":
            raise ValueError("Only spectral_clip.mode='const' is supported here.")
        threshold = float(spectral.const_threshold)
        rank = int(spectral.r_max)
        rp, ri = int(spectral.randomized_p), int(spectral.randomized_n_iter)

        @torch.no_grad()
        def _clip():
            for p in matrix_params:
                if p.grad is None:
                    continue
                grad = p.grad.data
                shape, dtype = grad.shape, grad.dtype
                grad_2d = grad.reshape(grad.shape[0], -1)
                if rank <= 0 or rank >= min(grad_2d.shape):
                    clipped, _ = clip_singular_values_svd(grad_2d.float(), threshold)
                else:
                    clipped, _ = clip_singular_values_randomized_svd(
                        grad_2d, threshold, rank=rank, p=rp, n_iter=ri,
                    )
                p.grad.data = clipped.to(dtype).reshape(shape)
        return _clip

    max_norm = float(cfg.train.global_clip_value)
    if max_norm == 0.0:
        return None
    return lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


@hydra.main(config_path="configs", config_name="train_nanogpt_shakespeare_sgdm", version_base="1.2")
def main(cfg: DictConfig):
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    if cfg.train.seed is not None:
        seed_everything(int(cfg.train.seed))

    train_loader, val_loader, vocab_size = build_dataloaders(cfg)
    OmegaConf.update(cfg, "dataset.vocab_size", vocab_size, force_add=True)
    OmegaConf.update(cfg, "model.vocab_size", vocab_size, force_add=True)

    model = GPT(cfg.model).to(device)
    if bool(getattr(cfg.train, "torch_compile", False)):
        model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.2f}M params")

    optimizers, matrix_params = build_optimizers(cfg, model)
    schedulers = build_lr_schedulers(cfg, optimizers, total_steps=int(cfg.train.train_steps))
    clip_fn = build_clipper(cfg, model, matrix_params)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    wandb_run = None
    if bool(getattr(cfg.train, "wandb_enabled", True)) and cfg.train.wandb_project:
        import wandb
        kwargs = dict(
            project=str(cfg.train.wandb_project),
            name=str(cfg.train.wandb_name),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        if cfg.train.wandb_entity:
            kwargs["entity"] = str(cfg.train.wandb_entity)
        wandb_run = wandb.init(**kwargs)

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    accum_steps = int(cfg.train.gradient_accumulation_steps)
    total_steps = int(cfg.train.train_steps)

    from tqdm.auto import tqdm
    print("Training...")
    for step in tqdm(range(total_steps)):
        model.train()
        loss_acc = 0.0
        for _ in range(accum_steps):
            x, y = next(train_iter)
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out.view(-1, out.size(-1)), y.view(-1)) / accum_steps
            loss_acc += loss.detach().item()
            loss.backward()

        if clip_fn is not None:
            clip_fn()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()

        for opt in optimizers:
            opt.step()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        for sch in schedulers:
            sch.step()

        if wandb_run is not None:
            log = {"Train Step Loss": loss_acc, "grad_norm": grad_norm}
            for i, opt in enumerate(optimizers):
                log[f"lr_{i}"] = opt.param_groups[0]["lr"]
            wandb_run.log(log, step=step)

        if (step + 1) % int(cfg.train.val_iters) == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(int(cfg.train.val_steps)):
                    x, y = next(val_iter)
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    val_loss += criterion(out.view(-1, out.size(-1)), y.view(-1)).item()
            val_loss /= int(cfg.train.val_steps)
            print(f"step {step + 1}: val_loss={val_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log({"Test Loss": val_loss}, step=step)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
