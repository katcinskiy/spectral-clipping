from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_from_disk
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments


def is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def resolve_attn_implementation(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        from transformers.utils import is_flash_attn_2_available
        if is_flash_attn_2_available():
            return "flash_attention_2"
    except Exception:
        pass
    return "sdpa" if torch.cuda.is_available() else "eager"


def load_local_train_dataset(tokenized_dataset_path: Path) -> Dataset:
    return load_from_disk(str(tokenized_dataset_path))["train"]


def split_muon_params(model: torch.nn.Module, exclude_names: list[str]):
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and not any(e in name for e in exclude_names):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


class NativeMuonWithAdamW(torch.optim.Optimizer):
    """torch.optim.Muon for 2D weights + AdamW for embeddings/biases/LN."""

    def __init__(
        self,
        *,
        muon_params, adamw_params,
        lr, weight_decay, momentum,
        adamw_lr_ratio, adamw_betas, adamw_eps, adamw_weight_decay,
        muon_kwargs=None,
    ):
        muon_kwargs = muon_kwargs or {}
        self.muon_opt = torch.optim.Muon(
            muon_params, lr=lr, weight_decay=weight_decay, momentum=momentum, **muon_kwargs,
        )
        self.adamw_opt = torch.optim.AdamW(
            adamw_params, lr=lr * adamw_lr_ratio, betas=adamw_betas,
            eps=adamw_eps, weight_decay=adamw_weight_decay,
        )
        super().__init__(
            [{"params": muon_params}, {"params": adamw_params}], defaults={},
        )
        self.param_groups = self.muon_opt.param_groups + self.adamw_opt.param_groups

    def step(self, closure=None):
        loss = self.muon_opt.step(closure=closure)
        self.adamw_opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.muon_opt.zero_grad(set_to_none=set_to_none)
        self.adamw_opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"muon": self.muon_opt.state_dict(), "adamw": self.adamw_opt.state_dict()}

    def load_state_dict(self, state_dict):
        self.muon_opt.load_state_dict(state_dict["muon"])
        self.adamw_opt.load_state_dict(state_dict["adamw"])
        self.param_groups = self.muon_opt.param_groups + self.adamw_opt.param_groups


def build_optimizer(cfg: DictConfig, model: torch.nn.Module):
    opt_type = str(cfg.optimizer.type).lower()
    params = [p for p in model.parameters() if p.requires_grad]
    lr = float(cfg.training.learning_rate)
    wd = float(cfg.training.weight_decay)

    if opt_type == "sgdm":
        if is_main_process():
            print(f"[optimizer] SGDM params: {sum(p.numel() for p in params):,}")
        return torch.optim.SGD(
            params, lr=lr, momentum=float(cfg.optimizer.sgdm.momentum), weight_decay=wd,
        )

    if opt_type == "adam":
        if is_main_process():
            print(f"[optimizer] Adam params: {sum(p.numel() for p in params):,}")
        return torch.optim.AdamW(
            params, lr=lr,
            betas=(float(cfg.optimizer.adam.beta1), float(cfg.optimizer.adam.beta2)),
            eps=float(cfg.optimizer.adam.eps), weight_decay=wd,
        )

    if opt_type != "muon":
        raise ValueError(f"Unsupported optimizer.type={opt_type}. Use one of: muon, sgdm, adam.")

    muon_params, adamw_params = split_muon_params(
        model, exclude_names=[str(n) for n in cfg.optimizer.muon.exclude_names],
    )
    if is_main_process():
        m = sum(p.numel() for p in muon_params)
        a = sum(p.numel() for p in adamw_params)
        print(f"[optimizer] MUON params: {m:,}; AdamW fallback params: {a:,}")

    optional_kwargs: dict[str, Any] = {}
    for key in ("nesterov", "ns_steps", "eps", "adjust_lr_fn"):
        if key in cfg.optimizer.muon:
            v = cfg.optimizer.muon.get(key)
            optional_kwargs[key] = None if v is None else v

    return NativeMuonWithAdamW(
        muon_params=muon_params, adamw_params=adamw_params,
        lr=lr, weight_decay=wd,
        momentum=float(cfg.optimizer.muon.momentum),
        adamw_lr_ratio=float(cfg.optimizer.muon.adamw_lr_ratio),
        adamw_betas=(float(cfg.optimizer.muon.adamw_beta1), float(cfg.optimizer.muon.adamw_beta2)),
        adamw_eps=float(cfg.optimizer.muon.adamw_eps),
        adamw_weight_decay=wd,
        muon_kwargs=optional_kwargs,
    )


def build_training_arguments(cfg, max_steps, gradient_accumulation_steps, run_name=None):
    sig = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": cfg.training.output_dir,
        "max_steps": int(max_steps),
        "per_device_train_batch_size": int(cfg.training.per_device_train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "learning_rate": float(cfg.training.learning_rate),
        "weight_decay": float(cfg.training.weight_decay),
        "lr_scheduler_type": str(cfg.training.lr_scheduler_type),
        "warmup_steps": int(cfg.training.warmup_steps),
        "logging_steps": int(cfg.training.logging_steps),
        "save_steps": int(cfg.training.save_steps),
        "save_total_limit": int(cfg.training.save_total_limit),
        "max_grad_norm": float(cfg.clipping.max_grad_norm),
        "bf16": bool(cfg.training.bf16),
        "dataloader_num_workers": int(cfg.training.dataloader_num_workers),
        "dataloader_pin_memory": bool(cfg.training.dataloader_pin_memory),
        "gradient_checkpointing": bool(cfg.training.gradient_checkpointing),
        "remove_unused_columns": False,
        "logging_first_step": True,
        "report_to": ["wandb"] if bool(cfg.wandb.enabled) else [],
    }
    if run_name is not None:
        kwargs["run_name"] = run_name
    if cfg.training.seed is not None:
        kwargs["seed"] = int(cfg.training.seed)
        kwargs["data_seed"] = int(cfg.training.seed)
    if "save_strategy" in sig:
        kwargs["save_strategy"] = str(cfg.training.save_strategy)
    if "dataloader_persistent_workers" in sig:
        kwargs["dataloader_persistent_workers"] = bool(cfg.training.dataloader_persistent_workers)
    if "dataloader_prefetch_factor" in sig and int(cfg.training.dataloader_num_workers) > 0:
        kwargs["dataloader_prefetch_factor"] = int(cfg.training.dataloader_prefetch_factor)
    if "ddp_find_unused_parameters" in sig:
        kwargs["ddp_find_unused_parameters"] = bool(cfg.training.ddp_find_unused_parameters)
    if "torch_compile" in sig:
        kwargs["torch_compile"] = bool(cfg.training.torch_compile)
    if "gradient_checkpointing_kwargs" in sig:
        kwargs["gradient_checkpointing_kwargs"] = {
            "use_reentrant": bool(cfg.training.gradient_checkpointing_use_reentrant)
        }

    if cfg.distributed.fsdp is not None:
        kwargs["fsdp"] = str(cfg.distributed.fsdp)
    if cfg.distributed.fsdp_config is not None:
        kwargs["fsdp_config"] = OmegaConf.to_container(cfg.distributed.fsdp_config, resolve=True)
    if cfg.distributed.deepspeed is not None:
        kwargs["deepspeed"] = str(cfg.distributed.deepspeed)

    return TrainingArguments(**kwargs)
