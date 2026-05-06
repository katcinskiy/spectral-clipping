# Derived from https://github.com/KellerJordan/cifar10-airbench.

import os
import sys
import uuid
from math import ceil

import hydra
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from omegaconf import DictConfig, OmegaConf
from torch import nn

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clipping import (  # noqa: E402
    clip_singular_values_randomized_svd,
    clip_singular_values_svd,
)

torch.backends.cudnn.benchmark = True

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))


def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    if r <= 2:
        for sy in range(-r, r + 1):
            for sx in range(-r, r + 1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r + sy: r + sy + crop_size, r + sx: r + sx + crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size + 2 * r), device=images.device, dtype=images.dtype)
        for s in range(-r, r + 1):
            mask = shifts[:, 0] == s
            images_tmp[mask] = images[mask, :, r + s: r + s + crop_size, :]
        for s in range(-r, r + 1):
            mask = shifts[:, 1] == s
            images_out[mask] = images_tmp[mask, :, :, r + s: r + s + crop_size]
    return images_out


class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            torch.save({
                "images": torch.tensor(dset.data),
                "labels": torch.tensor(dset.targets),
                "classes": dset.classes,
            }, data_path)
        data = torch.load(data_path, map_location=torch.device("cuda"))
        self.images, self.labels = data["images"], data["labels"]
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {}
        self.epoch = 0
        self.aug = aug or {}
        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def __len__(self):
        return len(self.images) // self.batch_size if self.drop_last else ceil(len(self.images) / self.batch_size)

    def __iter__(self):
        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,) * 4, "reflect")

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]
        if self.aug.get("flip", False) and self.epoch % 2 == 1:
            images = images.flip(-1)

        self.epoch += 1
        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i * self.batch_size: (i + 1) * self.batch_size]
            yield images[idxs], self.labels[idxs]


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.dirac_(self.weight.data[: self.weight.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.activ(self.norm1(self.pool(self.conv1(x))))
        return self.activ(self.norm2(self.conv2(x)))


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = (64, 256, 256)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size ** 2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths[0]),
            ConvGroup(widths[0], widths[1]),
            ConvGroup(widths[1], widths[2]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths[2], 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.half()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()
        flat = patches.view(len(patches), -1)
        eigvals, eigvecs = torch.linalg.eigh((flat.T @ flat) / len(flat), UPLO="U")
        scaled = eigvecs.T.reshape(-1, c, h, w) / torch.sqrt(eigvals.view(-1, 1, 1, 1) + eps)
        self.whiten.weight.data[:] = torch.cat((scaled, -scaled))

    def forward(self, x, whiten_bias_grad=True):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x).view(len(x), -1)
        return self.head(x) / x.size(-1)


def evaluate(model, loader, tta_level=0):
    def basic(x): return model(x).clone()
    def mirror(x): return 0.5 * model(x) + 0.5 * model(x.flip(-1))

    def mirror_translate(x):
        logits = mirror(x)
        padded = F.pad(x, (1,) * 4, "reflect")
        logits_t = torch.stack([mirror(padded[:, :, 0:32, 0:32]),
                                mirror(padded[:, :, 2:34, 2:34])]).mean(0)
        return 0.5 * logits + 0.5 * logits_t

    fn = [basic, mirror, mirror_translate][tta_level]
    model.eval()
    with torch.no_grad():
        logits = torch.cat([fn(x) for x in loader.normalize(loader.images).split(2000)])
    return (logits.argmax(1) == loader.labels).float().mean().item()


@torch.no_grad()
def _spectral_clip_const(p, threshold, method, rank, randomized_p, randomized_n_iter):
    grad = p.grad.data
    shape, dtype = grad.shape, grad.dtype
    grad_2d = grad.reshape(grad.shape[0], -1)
    if method == "svd":
        if grad_2d.dtype in (torch.float16, torch.bfloat16):
            grad_2d = grad_2d.float()
        clipped, _ = clip_singular_values_svd(grad_2d, threshold)
    else:
        clipped, _ = clip_singular_values_randomized_svd(
            grad_2d, threshold, rank=rank, p=randomized_p, n_iter=randomized_n_iter,
        )
    p.grad.data = clipped.to(dtype).reshape(shape)


def build_clipper(cfg, filter_params):
    if not cfg.clipping.enabled:
        return None
    if cfg.clipping.type == "norm_clip":
        max_norm = float(cfg.clipping.max_norm)
        return lambda: torch.nn.utils.clip_grad_norm_(filter_params, max_norm)
    if cfg.clipping.type == "spectral_clip":
        if cfg.clipping.spectral_clip_mode != "const":
            raise ValueError("Only spectral_clip_mode='const' is supported here.")
        method = str(cfg.clipping.spectral_clip_method)
        threshold = float(cfg.clipping.threshold)
        rank = int(cfg.clipping.r_max)
        rp, ri = int(cfg.clipping.randomized_p), int(cfg.clipping.randomized_n_iter)

        def _clip():
            for p in filter_params:
                if p.grad is not None:
                    _spectral_clip_const(p, threshold, method, rank, rp, ri)
        return _clip
    raise ValueError(f"Unknown clipping.type={cfg.clipping.type}")


def train_one_run(run, model, cfg: DictConfig):
    batch_size = int(cfg.training.batch_size)
    bias_lr = float(cfg.optimizer.bias_lr)
    head_lr = float(cfg.optimizer.head_lr)
    wd = float(cfg.optimizer.wd_base) * batch_size
    use_fused = bool(cfg.optimizer.get("fused", True)) and not bool(cfg.clipping.enabled)

    test_loader = CifarLoader(cfg.data.path, train=False, batch_size=int(cfg.evaluation.batch_size))
    train_loader = CifarLoader(
        cfg.data.path, train=True, batch_size=batch_size,
        aug=dict(flip=bool(cfg.training.augmentation.flip),
                 translate=int(cfg.training.augmentation.translate)),
    )
    if run == "warmup":
        train_loader.labels = torch.randint(
            0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device,
        )

    total_steps = ceil(float(cfg.training.total_train_epochs) * len(train_loader))
    whiten_bias_steps = ceil(float(cfg.training.whiten_bias_epochs) * len(train_loader))

    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]

    head_opt = torch.optim.SGD(
        [
            dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
            dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
            dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
        ],
        momentum=float(cfg.optimizer.sgd_head_momentum),
        nesterov=bool(cfg.optimizer.sgd_head_nesterov),
        fused=use_fused,
    )
    filter_opt = torch.optim.SGD(
        filter_params,
        lr=float(cfg.optimizer.filter_lr),
        momentum=float(cfg.optimizer.filter_momentum),
        nesterov=bool(cfg.optimizer.filter_nesterov),
        fused=use_fused,
    )
    optimizers = [head_opt, filter_opt]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    clip = build_clipper(cfg, filter_params)
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    elapsed = 0.0

    def tic(): starter.record()
    def toc():
        nonlocal elapsed
        ender.record()
        torch.cuda.synchronize()
        elapsed += 1e-3 * starter.elapsed_time(ender)

    model.reset()
    step = 0
    epoch_train_accs, epoch_val_accs = [], []

    tic()
    model.init_whiten(train_loader.normalize(train_loader.images[:5000]))
    toc()

    for epoch in range(ceil(total_steps / len(train_loader))):
        tic()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_steps))
            F.cross_entropy(
                outputs, labels,
                label_smoothing=float(cfg.training.label_smoothing),
                reduction="sum",
            ).backward()

            if clip is not None:
                clip()

            for group in head_opt.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / max(1, whiten_bias_steps))
            for group in head_opt.param_groups[1:] + filter_opt.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / max(1, total_steps))

            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_steps:
                break
        toc()

        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_acc = evaluate(model, test_loader, tta_level=int(cfg.evaluation.val_tta_level))
        epoch_train_accs.append(train_acc)
        epoch_val_accs.append(val_acc)
        print(
            f"run={run if run is not None else '':>3}  epoch={epoch:>3}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  time={elapsed:.2f}s"
        )

    tic()
    tta_val_acc = evaluate(model, test_loader, tta_level=int(cfg.evaluation.final_tta_level))
    toc()
    print(f"run={run if run is not None else '':>3}  final  tta_val_acc={tta_val_acc:.4f}  time={elapsed:.2f}s")

    return {
        "tta_val_acc": float(tta_val_acc),
        "val_acc": float(val_acc),
        "time_seconds": float(elapsed),
        "epoch_train_accs": epoch_train_accs,
        "epoch_val_accs": epoch_val_accs,
    }


@hydra.main(config_path="configs", config_name="sgdm_no_clip", version_base=None)
def run_experiment(cfg: DictConfig):
    if cfg.experiment.seed is not None:
        torch.manual_seed(int(cfg.experiment.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg.experiment.seed))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    wandb_run = None
    if bool(cfg.wandb.enabled):
        import wandb
        kwargs = dict(project=str(cfg.wandb.project), config=OmegaConf.to_container(cfg, resolve=True))
        if cfg.wandb.entity is not None:
            kwargs["entity"] = str(cfg.wandb.entity)
        if cfg.wandb.run_name:
            kwargs["name"] = str(cfg.wandb.run_name).strip()
        if cfg.wandb.tags:
            kwargs["tags"] = [str(t) for t in cfg.wandb.tags]
        wandb_run = wandb.init(**kwargs)

    model = CifarNet().to(str(cfg.experiment.device)).to(memory_format=torch.channels_last)
    if cfg.experiment.compile_model:
        model.compile(mode=str(cfg.experiment.compile_mode))

    if cfg.experiment.warmup:
        train_one_run("warmup", model, cfg)

    mini_results = [train_one_run(run, model, cfg) for run in range(int(cfg.experiment.num_runs))]
    accs = torch.tensor([x["tta_val_acc"] for x in mini_results], dtype=torch.float32)
    mean_acc, std_acc = float(accs.mean()), float(accs.std(unbiased=False))
    print(f"Mean: {mean_acc:.4f}    Std: {std_acc:.4f}")

    if wandb_run is not None:
        min_epochs = min(len(x["epoch_train_accs"]) for x in mini_results)
        train_mat = torch.tensor([x["epoch_train_accs"][:min_epochs] for x in mini_results])
        val_mat = torch.tensor([x["epoch_val_accs"][:min_epochs] for x in mini_results])
        for e in range(min_epochs):
            wandb_run.log({
                "epoch": e,
                "avg/train_acc": float(train_mat.mean(0)[e]),
                "avg/val_acc": float(val_mat.mean(0)[e]),
            })
        wandb_run.summary["sweep/mean_tta_val_acc"] = mean_acc

    log_dir = os.path.join(str(cfg.experiment.log_dir), str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    torch.save(
        dict(accs=accs, mini_results=mini_results, config=OmegaConf.to_container(cfg, resolve=True)),
        os.path.join(log_dir, "log.pt"),
    )
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    run_experiment()
