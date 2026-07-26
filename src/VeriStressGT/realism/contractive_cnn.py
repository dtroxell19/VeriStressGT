"""Constructor A: TRAINED Deep-Contractive CNN.

A genuine multiclass MNIST classifier `f_k(x) = b_k + w_k . Phi(x)` with a real learned conv feature
extractor Phi and an ordinary (uncentered) linear head. During training each conv block is projected
to a target induced-l-inf norm <= lam, giving a rigorous whole-network Lipschitz constant
`L_Phi = prod_l L_l`. The native certificate at (x0, y, eps) is, for every competitor k != y,

    mu_yk(x) >= mu_yk(x0) - eps * ||w_y - w_k||_1 * L_Phi.

Robust iff all these lower bounds exceed the acceptance floor. (This bound is sound but loose, hence
the HARDNESS gate falls back to exact MILP for near-boundary labels — see hardness_gate.py.)

Inputs are 8x8 avg-pooled MNIST so the exact MILP / abcrown remain tractable (see DECISIONS.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_bounds import conv_induced_inf_norm, linear_induced_inf_norm

IMG = 8  # downsampled resolution


class ContractiveCNN(nn.Module):
    def __init__(self, depth: int = 2, channels: int = 4, kernel: int = 3, num_classes: int = 10):
        super().__init__()
        pad = kernel // 2
        convs: List[nn.Module] = []
        c_in = 1
        for _ in range(depth):
            convs += [nn.Conv2d(c_in, channels, kernel, padding=pad), nn.ReLU()]
            c_in = channels
        self.features = nn.Sequential(*convs)
        self.flatten = nn.Flatten()
        self.head = nn.Linear(channels * IMG * IMG, num_classes)

    def forward(self, x):
        if x.dim() != 4:
            x = x.view(-1, 1, IMG, IMG)
        return self.head(self.flatten(self.features(x)))

    # -- induced-norm control ------------------------------------------------
    @torch.no_grad()
    def project_contraction(self, lam: float) -> None:
        """Scale each conv weight so its induced-l-inf norm <= lam (rigorous per-layer bound)."""
        for m in self.features:
            if isinstance(m, nn.Conv2d):
                n = conv_induced_inf_norm(m.weight)
                if n > lam:
                    m.weight.mul_(lam / n)

    def lipschitz_phi(self) -> float:
        L = 1.0
        for m in self.features:
            if isinstance(m, nn.Conv2d):
                L *= conv_induced_inf_norm(m.weight)
        return L


def load_mnist_8x8(data_dir: str):
    import os
    cache = os.path.join(data_dir, f"mnist_{IMG}x{IMG}_cache.pt")
    if os.path.exists(cache):
        d = torch.load(cache)
        return (d["Xtr"], d["ytr"]), (d["Xte"], d["yte"])
    from torchvision import datasets, transforms
    tf = transforms.ToTensor()
    tr = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    te = datasets.MNIST(data_dir, train=False, download=True, transform=tf)

    def down(ds):
        X = torch.stack([ds[i][0] for i in range(len(ds))])          # (N,1,28,28) in [0,1]
        X = F.adaptive_avg_pool2d(X, (IMG, IMG))                      # (N,1,IMG,IMG), stays in [0,1]
        y = torch.tensor([ds[i][1] for i in range(len(ds))])
        return X, y
    (Xtr, ytr), (Xte, yte) = down(tr), down(te)
    torch.save({"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte}, cache)
    return (Xtr, ytr), (Xte, yte)


@dataclass
class TrainResult:
    model: ContractiveCNN
    test_acc: float
    baseline_acc: float
    L_phi: float
    test_X: torch.Tensor
    test_y: torch.Tensor


def _train_one(model, Xtr, ytr, Xte, yte, epochs, lam, project: bool):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.CrossEntropyLoss()
    n = Xtr.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            lossf(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
            if project:
                model.project_contraction(lam)
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    return acc


def train_contractive(data_dir: str, depth=2, channels=4, lam=1.2, epochs=6,
                      seed=0, data=None) -> TrainResult:
    torch.manual_seed(seed)
    (Xtr, ytr), (Xte, yte) = data if data is not None else load_mnist_8x8(data_dir)
    model = ContractiveCNN(depth, channels)
    acc = _train_one(model, Xtr, ytr, Xte, yte, epochs, lam, project=True)
    # size-matched unconstrained baseline
    torch.manual_seed(seed)
    base = ContractiveCNN(depth, channels)
    bacc = _train_one(base, Xtr, ytr, Xte, yte, epochs, lam, project=False)
    return TrainResult(model=model.double(), test_acc=acc, baseline_acc=bacc,
                       L_phi=model.double().lipschitz_phi(), test_X=Xte.double(), test_y=yte)


@torch.no_grad()
def native_certificate(model: ContractiveCNN, x0: torch.Tensor, y: int, eps: float) -> Dict:
    """Rigorous whole-network Lipschitz-margin certificate (float64, exact weights)."""
    model = model.double().eval()
    x0 = x0.double().view(1, 1, IMG, IMG)
    L_phi = model.lipschitz_phi()
    W = model.head.weight.double()          # (C, F)
    b = model.head.bias.double()            # (C,)
    logits0 = model(x0).reshape(-1)
    per_k = {}
    L_ana = float("inf")
    for k in range(W.shape[0]):
        if k == y:
            continue
        mu0 = float(logits0[y] - logits0[k])
        w1 = float((W[y] - W[k]).abs().sum().item())
        lb = mu0 - eps * w1 * L_phi
        per_k[int(k)] = {"mu_at_x0": mu0, "w_diff_l1": w1, "native_lb": lb}
        L_ana = min(L_ana, lb)
    return {"L_phi": L_phi, "analytical_lower_bound": L_ana,
            "mu_at_x0_min": float(min(v["mu_at_x0"] for v in per_k.values())),
            "per_competitor": per_k, "label": int(y), "epsilon": float(eps)}
