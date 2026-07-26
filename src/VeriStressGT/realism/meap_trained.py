"""Constructor C: TRAINED MEAP classifier (MLP).

Genuine MNIST classifier. Trained MLP prefix -> scalar responses s_p(x) -> MEAP features
r_p = gamma_p + |s_p| -> ordinary trained head f_k = beta_k + sum_p a_kp r_p.

|s| = relu(s) + relu(-s), so the whole thing is a ReLU MLP:
    Linear(in->hid) -> ReLU -> Linear(hid->2P) [= [W_s; -W_s], produces [s; -s]] -> ReLU
    -> head_combined(2P->K) [weight [A, A], bias beta + A gamma]   (folds r = gamma + relu(s)+relu(-s)).

Native certificate (tightness study): bound s_p over the box via the prefix l-inf Lipschitz, then
r_p in [gamma_p + dist(0,[l,u]), gamma_p + max(|l|,|u|)], interval-propagate to a margin lower bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_bounds import linear_induced_inf_norm
from .contractive_cnn import load_mnist_8x8, IMG


class MEAPNet(nn.Module):
    def __init__(self, hidden=32, num_pairs=16, num_classes=10, input_hw=IMG):
        super().__init__()
        self.P = num_pairs
        self.input_hw = input_hw
        self.IN = input_hw * input_hw
        self.lin1 = nn.Linear(self.IN, hidden)
        self.Ws = nn.Linear(hidden, num_pairs)     # produces s
        self.gamma_raw = nn.Parameter(torch.full((num_pairs,), -1.0))  # gamma = softplus > 0
        self.A = nn.Parameter(0.05 * torch.randn(num_classes, num_pairs))
        self.beta = nn.Parameter(torch.zeros(num_classes))

    def gamma(self):
        return F.softplus(self.gamma_raw)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        h = F.relu(self.lin1(x))
        s = self.Ws(h)
        r = self.gamma() + s.abs()                 # gamma + |s|
        return self.beta + r @ self.A.t()

    @torch.no_grad()
    def to_modules(self) -> List[nn.Module]:
        P = self.P
        l1 = nn.Linear(self.IN, self.lin1.out_features).double()
        l1.weight.copy_(self.lin1.weight.double()); l1.bias.copy_(self.lin1.bias.double())
        # Linear(hid -> 2P) = [W_s; -W_s], biases [b_s; -b_s]  -> [s; -s]
        stack = nn.Linear(self.lin1.out_features, 2 * P).double()
        stack.weight.copy_(torch.cat([self.Ws.weight, -self.Ws.weight], 0).double())
        stack.bias.copy_(torch.cat([self.Ws.bias, -self.Ws.bias], 0).double())
        # head_combined(2P -> K): weight [A, A], bias beta + A gamma
        head = nn.Linear(2 * P, self.A.shape[0]).double()
        head.weight.copy_(torch.cat([self.A, self.A], dim=1).double())
        head.bias.copy_((self.beta + self.A @ self.gamma()).double())
        return [nn.Flatten(), l1, nn.ReLU(), stack, nn.ReLU(), head]

    def lipschitz_s(self) -> float:
        return linear_induced_inf_norm(self.lin1.weight) * linear_induced_inf_norm(self.Ws.weight)

    @torch.no_grad()
    def project_lipschitz(self, lam: float) -> None:
        """Project prefix linears to induced-inf-norm <= lam, so L_s stays small -> tight native cert."""
        for lin in (self.lin1, self.Ws):
            n = linear_induced_inf_norm(lin.weight)
            if n > lam:
                lin.weight.mul_(lam / n)


def certified_radius_native(model, x0, y, cert_fn, floor=1e-7, hi=1.0) -> float:
    """Largest eps for which the NATIVE certificate proves robustness (analytic, no MILP).
    Binary search on eps for the L_ana=floor crossing (L_ana is decreasing in eps)."""
    if cert_fn(model, x0, y, 0.0)["analytical_lower_bound"] <= floor:
        return 0.0
    lo, h = 0.0, hi
    if cert_fn(model, x0, y, h)["analytical_lower_bound"] > floor:
        return h  # certifies out to hi
    for _ in range(40):
        mid = 0.5 * (lo + h)
        if cert_fn(model, x0, y, mid)["analytical_lower_bound"] > floor:
            lo = mid
        else:
            h = mid
    return lo


class _ExportWrapper(nn.Module):
    def __init__(self, modules): super().__init__(); self.seq = nn.Sequential(*modules)
    def forward(self, x): return self.seq(x.view(x.shape[0], -1))


@dataclass
class TrainResult:
    model: MEAPNet
    export_model: nn.Module
    test_acc: float
    baseline_acc: float
    test_X: torch.Tensor
    test_y: torch.Tensor


def train_meap(data_dir, hidden=32, num_pairs=16, epochs=6, seed=0, data=None, input_hw=IMG,
               project_lam=None) -> TrainResult:
    torch.manual_seed(seed)
    (Xtr, ytr), (Xte, yte) = data if data is not None else load_mnist_8x8(data_dir)
    input_hw = int(Xtr.shape[-1])          # infer from data (supports full-res MEAP)
    IN = input_hw * input_hw
    Xtr = Xtr.reshape(Xtr.shape[0], -1); Xte = Xte.reshape(Xte.shape[0], -1)
    model = MEAPNet(hidden, num_pairs, input_hw=input_hw)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3); lossf = nn.CrossEntropyLoss()
    n = Xtr.shape[0]
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i+256]; opt.zero_grad(); lossf(model(Xtr[idx]), ytr[idx]).backward(); opt.step()
            if project_lam is not None:
                model.project_lipschitz(project_lam)
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    # matched baseline: ordinary MLP in->hid->K
    torch.manual_seed(seed)
    base = nn.Sequential(nn.Flatten(), nn.Linear(IN, hidden), nn.ReLU(), nn.Linear(hidden, 10))
    bo = torch.optim.Adam(base.parameters(), lr=2e-3)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i+256]; bo.zero_grad(); lossf(base(Xtr[idx]), ytr[idx]).backward(); bo.step()
    with torch.no_grad():
        bacc = (base(Xte).argmax(1) == yte).float().mean().item()
    md = model.double()
    return TrainResult(model=md, export_model=_ExportWrapper(md.to_modules()).double().eval(),
                       test_acc=acc, baseline_acc=bacc, test_X=Xte.double(), test_y=yte)


@torch.no_grad()
def native_certificate(model: MEAPNet, x0: torch.Tensor, y: int, eps: float) -> Dict:
    model = model.double().eval()
    P = model.P
    g = model.gamma().double()
    h0 = F.relu(model.lin1(x0.double().view(1, -1)))
    s0 = model.Ws(h0).reshape(-1)                 # (P,)
    Ls = model.lipschitz_s()
    s_lo, s_hi = s0 - Ls * eps, s0 + Ls * eps
    # r_p = gamma + |s|, |s| in [dist(0,[lo,hi]), max(|lo|,|hi|)]
    absmin = torch.where((s_lo <= 0) & (s_hi >= 0), torch.zeros_like(s_lo),
                         torch.minimum(s_lo.abs(), s_hi.abs()))
    absmax = torch.maximum(s_lo.abs(), s_hi.abs())
    r_lo = g + absmin; r_hi = g + absmax           # (P,)
    A = model.A.double()
    logits0 = model(x0.double().view(1, -1)).reshape(-1)
    L_ana = float("inf"); per_k = {}
    for k in range(A.shape[0]):
        if k == y:
            continue
        alpha = A[y] - A[k]
        lb = float((torch.clamp(alpha, min=0) * r_lo + torch.clamp(alpha, max=0) * r_hi).sum()
                   + (model.beta[y] - model.beta[k]))
        per_k[int(k)] = {"native_lb": lb, "mu_at_x0": float(logits0[y] - logits0[k])}
        L_ana = min(L_ana, lb)
    return {"analytical_lower_bound": L_ana, "L_s": Ls,
            "mu_at_x0_min": float(min(v["mu_at_x0"] for v in per_k.values())),
            "per_competitor": per_k, "label": int(y), "epsilon": float(eps)}
