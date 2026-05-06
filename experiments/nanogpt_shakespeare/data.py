import os
import pickle

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, IterableDataset


_TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def prepare_shakespeare_char(data_dir: str) -> int:
    """Build train.bin / val.bin / meta.pkl in `data_dir` if missing. Returns vocab_size."""
    os.makedirs(data_dir, exist_ok=True)
    train_bin = os.path.join(data_dir, "train.bin")
    meta_pkl = os.path.join(data_dir, "meta.pkl")

    if os.path.exists(train_bin) and os.path.exists(meta_pkl):
        with open(meta_pkl, "rb") as f:
            return pickle.load(f)["vocab_size"]

    input_path = os.path.join(data_dir, "input.txt")
    if not os.path.exists(input_path):
        with open(input_path, "w") as f:
            f.write(requests.get(_TINY_SHAKESPEARE_URL).text)

    with open(input_path, "r") as f:
        data = f.read()

    chars = sorted(set(data))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}

    n = len(data)
    train_ids = np.array([stoi[c] for c in data[: int(0.9 * n)]], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in data[int(0.9 * n):]], dtype=np.uint16)
    train_ids.tofile(train_bin)
    val_ids.tofile(os.path.join(data_dir, "val.bin"))

    with open(meta_pkl, "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "stoi": stoi,
                     "itos": {i: ch for ch, i in stoi.items()}}, f)
    return vocab_size


class ShakespeareCharDataset(IterableDataset):
    def __init__(self, data_path: str, block_size: int, seed: int):
        self.data_path = data_path
        self.block_size = block_size
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info is not None else (0, 1)
        data = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        step = 0
        while True:
            rng = np.random.default_rng(self.seed + worker_id + step * num_workers)
            idx = rng.integers(0, len(data) - self.block_size)
            x = torch.from_numpy(data[idx: idx + self.block_size].astype(np.int64))
            y = torch.from_numpy(data[idx + 1: idx + 1 + self.block_size].astype(np.int64))
            step += 1
            yield x, y


def build_dataloaders(cfg):
    data_dir = os.path.join(cfg.dataset.dataset_path, cfg.dataset.dataset_name)
    vocab_size = prepare_shakespeare_char(data_dir)
    seed = int(cfg.train.seed or 0)
    train_ds = ShakespeareCharDataset(
        os.path.join(data_dir, "train.bin"), cfg.model.block_size, seed=seed,
    )
    val_ds = ShakespeareCharDataset(
        os.path.join(data_dir, "val.bin"), cfg.model.block_size, seed=seed,
    )
    return (
        DataLoader(train_ds, batch_size=cfg.train.batch_size, num_workers=4),
        DataLoader(val_ds, batch_size=cfg.train.test_batch_size, num_workers=4,
                   pin_memory=True, persistent_workers=True),
        vocab_size,
    )
