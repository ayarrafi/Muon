"""CIFAR-10 loaded from the local pickle files in cifar-10-batches-py/."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DATA_DIR = Path(__file__).resolve().parent / "cifar-10-batches-py"

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _unpickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def _load_split(train: bool) -> tuple[np.ndarray, np.ndarray]:
    cache = DATA_DIR / ("train.npz" if train else "test.npz")
    if cache.exists():
        d = np.load(cache)
        return d["x"], d["y"]
    files = [DATA_DIR / f"data_batch_{i}" for i in range(1, 6)] if train else [DATA_DIR / "test_batch"]
    images, labels = [], []
    for f in files:
        d = _unpickle(f)
        images.append(d[b"data"])
        labels.extend(d[b"labels"])
    x = np.concatenate(images, axis=0).reshape(-1, 3, 32, 32).astype(np.uint8)
    y = np.array(labels, dtype=np.int64)
    np.savez(cache, x=x, y=y)
    return x, y


class CIFAR10(Dataset):
    def __init__(self, train: bool, augment: bool):
        self.images, self.labels = _load_split(train)
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.images[idx].transpose(1, 2, 0)  # CHW uint8 -> HWC for PIL
        return self.transform(img), int(self.labels[idx])


def build_loaders(batch_size: int, num_workers: int = 4, augment: bool = True) -> tuple[DataLoader, DataLoader]:
    train_set = CIFAR10(train=True, augment=augment)
    test_set = CIFAR10(train=False, augment=False)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=512,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader
