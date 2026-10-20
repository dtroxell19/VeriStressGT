"""Real labeled datasets for the realistic trained constructors, downsampled to IMG x IMG.

All are 28x28 grayscale via torchvision (MNIST / FashionMNIST / KMNIST), adaptive-avg-pooled to
IMG x IMG in [0,1] (the pixel domain used by the [0,1]-clamped MILP + VNNLIB). Cached per dataset.
"""
from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.nn.functional as F

from .contractive_cnn import IMG

# name -> (torchvision class, is_color, uses split= arg instead of train=)
_LOADERS = {
    "mnist": ("MNIST", False, False),
    "fashion_mnist": ("FashionMNIST", False, False),
    "kmnist": ("KMNIST", False, False),
    "svhn": ("SVHN", True, True),        # Street-View House Numbers, 32x32 color, 10 classes
    "stl10": ("STL10", True, True),      # natural images, 96x96 color, 10 classes (uncommon)
}


def _to_gray(X: torch.Tensor, size: int) -> torch.Tensor:
    if X.shape[1] == 3:                    # color -> luminance grayscale
        X = (0.2989 * X[:, 0] + 0.5870 * X[:, 1] + 0.1140 * X[:, 2]).unsqueeze(1)
    return F.adaptive_avg_pool2d(X, (size, size))   # -> (N,1,size,size) in [0,1]


def load_dataset(name: str, data_dir: str, size: int, max_per_split: int = 20000):
    """((Xtr,ytr),(Xte,yte)) as `size` x `size` grayscale in [0,1]. Color -> luminance. Cached.

    `size` = 8 for the CNN constructors (MILP tractability); larger (e.g. 32) for MEAP (an MLP whose
    MILP size is input-independent), giving real-resolution real-data accuracy.
    """
    if name not in _LOADERS:
        raise ValueError(f"unknown dataset {name}; choices {list(_LOADERS)}")
    cache = os.path.join(data_dir, f"{name}_{size}x{size}_cache.pt")
    if os.path.exists(cache):
        d = torch.load(cache)
        return (d["Xtr"], d["ytr"]), (d["Xte"], d["yte"])
    from torchvision import datasets, transforms
    cls_name, is_color, uses_split = _LOADERS[name]
    cls = getattr(datasets, cls_name)
    tf = transforms.ToTensor()
    if uses_split:
        tr = cls(data_dir, split="train", download=True, transform=tf)
        te = cls(data_dir, split="test", download=True, transform=tf)
    else:
        tr = cls(data_dir, train=True, download=True, transform=tf)
        te = cls(data_dir, train=False, download=True, transform=tf)

    def down(ds):
        n = min(len(ds), max_per_split)
        X = torch.stack([ds[i][0] for i in range(n)])
        y = torch.tensor([int(ds[i][1]) for i in range(n)])
        return _to_gray(X, size), y
    (Xtr, ytr), (Xte, yte) = down(tr), down(te)
    torch.save({"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte}, cache)
    return (Xtr, ytr), (Xte, yte)


def load_dataset_8x8(name: str, data_dir: str, max_per_split: int = 20000):
    return load_dataset(name, data_dir, IMG, max_per_split)
