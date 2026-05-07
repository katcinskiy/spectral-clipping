from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    set_seed,
)

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for path in (_REPO_ROOT, _HERE):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from clipping import SpectralClipperConst  # noqa: E402
from utils import (  # noqa: E402
    build_optimizer,
    build_training_arguments,
    is_main_process,
    load_local_train_dataset,
    resolve_attn_implementation,
)


class ClippingTrainer(Trainer):
    """Hooks into accelerator.clip_grad_norm_ to switch between norm and spectral clipping."""

    def __init__(self, *, clipping_mode, spectral_tau, spectral_rank,
                 spectral_randomized_p, spectral_randomized_n_iter, **kwargs):
        super().__init__(**kwargs)
        self.clipping_mode = clipping_mode
        self.spectral_tau = float(spectral_tau)
        self.spectral_rank = int(spectral_rank)
        self.spectral_randomized_p = int(spectral_randomized_p)
        self.spectral_randomized_n_iter = int(spectral_randomized_n_iter)
        self._original_clip = self.accelerator.clip_grad_norm_
        self.accelerator.clip_grad_norm_ = self._clip_hook
        self._spectral_clipper: SpectralClipperConst | None = None

    def _model_for_clipper(self):
        return self.model_wrapped if self.model_wrapped is not None else self.model

    def _ensure_spectral_clipper(self):
        if self._spectral_clipper is None:
            self._spectral_clipper = SpectralClipperConst(
                self._model_for_clipper(),
                threshold=self.spectral_tau,
                method="randomized_svd",
                rank=self.spectral_rank,
                randomized_p=self.spectral_randomized_p,
                randomized_n_iter=self.spectral_randomized_n_iter,
            )

    def _clip_hook(self, parameters, max_norm, *args, **kwargs):
        params = [p for p in parameters if p is not None and p.grad is not None]
        if not params:
            return torch.tensor(0.0, device=self.model.device)

        if self.clipping_mode == "norm":
            self._original_clip(params, max_norm, *args, **kwargs)
        elif self.clipping_mode == "spectral":
            self._ensure_spectral_clipper()
            self._spectral_clipper.clip(self._model_for_clipper())
        else:
            raise ValueError(f"Unsupported clipping_mode: {self.clipping_mode}")

        total_sq = torch.zeros((), device=params[0].grad.device, dtype=torch.float32)
        for p in params:
            g = p.grad.detach().float()
            total_sq += torch.sum(g * g)
        return torch.sqrt(total_sq).to(params[0].grad.device)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    if is_main_process():
        print(OmegaConf.to_yaml(cfg))

    if not bool(cfg.wandb.enabled):
        os.environ["WANDB_DISABLED"] = "true"
    if bool(cfg.wandb.enabled) and cfg.wandb.mode is not None:
        os.environ["WANDB_MODE"] = str(cfg.wandb.mode)

    if cfg.training.seed is not None:
        set_seed(int(cfg.training.seed))

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_micro = int(cfg.training.per_device_train_batch_size) * world_size
    if cfg.training.gradient_accumulation_steps is None:
        grad_accum = math.ceil(int(cfg.training.target_global_batch_size) / global_micro)
    else:
        grad_accum = int(cfg.training.gradient_accumulation_steps)

    effective_global = global_micro * grad_accum
    tokens_per_step = effective_global * int(cfg.data.sequence_length)
    max_steps = math.ceil(int(cfg.training.max_train_tokens) / tokens_per_step)

    if is_main_process():
        print(f"[batching] world_size={world_size} grad_accum={grad_accum} "
              f"effective_global={effective_global} tokens/step={tokens_per_step} "
              f"max_steps={max_steps}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_name, trust_remote_code=bool(cfg.model.trust_remote_code), use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must have a pad_token or eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = resolve_attn_implementation(str(cfg.model.attn_implementation))
    if is_main_process():
        print(f"[model] attention implementation: {attn_impl}")

    model_config = AutoConfig.from_pretrained(
        cfg.model.base_config_name, trust_remote_code=bool(cfg.model.trust_remote_code),
    )
    model_config.use_cache = False
    if model_config.pad_token_id is None:
        model_config.pad_token_id = tokenizer.pad_token_id

    if bool(cfg.model.init_from_pretrained):
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.pretrained_model_name, config=model_config,
            trust_remote_code=bool(cfg.model.trust_remote_code),
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16 if bool(cfg.training.bf16) else None,
        )
    else:
        model = AutoModelForCausalLM.from_config(
            model_config, trust_remote_code=bool(cfg.model.trust_remote_code),
            attn_implementation=attn_impl,
        )

    if is_main_process():
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[model] trainable parameters: {n:,}")

    if cfg.data.tokenized_dataset_path is None:
        raise ValueError(
            "data.tokenized_dataset_path must be set. "
            "Run prepare_dataset.py first."
        )
    train_dataset = load_local_train_dataset(Path(str(cfg.data.tokenized_dataset_path)))
    if is_main_process():
        rows = len(train_dataset)
        slots = rows * int(cfg.data.sequence_length)
        print(f"[data] rows={rows:,}  token_slots={slots:,}")
        if int(cfg.training.max_train_tokens) > slots:
            print("[data] warning: max_train_tokens > one full pass; Trainer will repeat the data.")

    if bool(cfg.wandb.enabled) and is_main_process() and wandb.run is None:
        wandb.init(
            project=str(cfg.wandb.project),
            entity=cfg.wandb.entity,
            tags=[str(t) for t in cfg.wandb.tags],
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    training_args = build_training_arguments(
        cfg=cfg, max_steps=max_steps, gradient_accumulation_steps=grad_accum,
    )
    optimizer = build_optimizer(cfg, model)
    trainer = ClippingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        optimizers=(optimizer, None),
        clipping_mode=str(cfg.clipping.mode),
        spectral_tau=float(cfg.spectral.tau),
        spectral_rank=int(cfg.spectral.rank),
        spectral_randomized_p=int(cfg.spectral.randomized_p),
        spectral_randomized_n_iter=int(cfg.spectral.randomized_n_iter),
    )
    trainer.train()

    if is_main_process() and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
