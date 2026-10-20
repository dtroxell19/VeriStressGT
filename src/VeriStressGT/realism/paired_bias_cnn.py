"""Constructor B: TRAINED Paired-Bias CNN.

A genuine MNIST classifier. Real trained conv prefix -> shared paired conv producing s_i(x) ->
paired-difference features d_i = relu(s_i + b_i) - relu(s_i + c_i), b_i > c_i -> ordinary trained
head f_k = beta_k + sum_i a_ki d_i.

MILP-encodable sequential form (for the exact hardness gate + ONNX):
    backbone(conv,relu)* -> Conv2d(2P, weight-shared, bias=[b;c]) -> ReLU -> Flatten -> Linear([A,-A]).
Since d_i = u1_i - u2_i and f = A d = [A,-A] [u1;u2], the head reads the paired differences.

Native coupled certificate (tightness study; MILP assigns the actual label near the boundary):
bound s_i over the box via the prefix l-inf Lipschitz, use monotonicity/boundedness of d_i in
[0, b_i-c_i], and interval-propagate to a rigorous margin lower bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_bounds import conv_induced_inf_norm
from .contractive_cnn import load_mnist_8x8, IMG


class PairedBiasCNN(nn.Module):
    def __init__(self, depth=2, channels=4, num_pairs=8, kernel=3, num_classes=10):
        super().__init__()
        pad = kernel // 2
        self.P, self.channels, self.pad, self.kernel = num_pairs, channels, pad, kernel
        convs: List[nn.Module] = []
        c_in = 1
        for _ in range(depth):
            convs += [nn.Conv2d(c_in, channels, kernel, padding=pad), nn.ReLU()]
            c_in = channels
        self.backbone = nn.Sequential(*convs)
        # shared paired conv weight (P output filters, applied twice)
        self.pc_weight = nn.Parameter(torch.empty(num_pairs, channels, kernel, kernel))
        nn.init.kaiming_normal_(self.pc_weight, nonlinearity="relu")
        self.base = nn.Parameter(torch.zeros(num_pairs))     # bias center per pair
        self.gap = nn.Parameter(torch.full((num_pairs,), -1.0))  # b-c = softplus(gap) > 0
        self.A = nn.Parameter(0.01 * torch.randn(num_classes, num_pairs * IMG * IMG))

    def _bc(self):
        g = F.softplus(self.gap)
        return self.base + g / 2, self.base - g / 2   # b_i > c_i

    def forward(self, x):
        if x.dim() != 4:
            x = x.view(-1, 1, IMG, IMG)
        h = self.backbone(x)
        W2 = torch.cat([self.pc_weight, self.pc_weight], 0)      # (2P,C,k,k) shared
        b, c = self._bc()
        u = F.relu(F.conv2d(h, W2, torch.cat([b, c]), padding=self.pad))  # (N,2P,H,W)
        headW = torch.cat([self.A, -self.A], dim=1)              # (K, 2P*H*W)
        return u.flatten(1) @ headW.t()

    # -- concrete module list for exact MILP dense conversion + ONNX --------
    @torch.no_grad()
    def to_modules(self) -> List[nn.Module]:
        b, c = self._bc()
        pconv = nn.Conv2d(self.channels, 2 * self.P, self.kernel, padding=self.pad).double()
        pconv.weight.copy_(torch.cat([self.pc_weight, self.pc_weight], 0).double())
        pconv.bias.copy_(torch.cat([b, c]).double())
        head = nn.Linear(2 * self.P * IMG * IMG, self.A.shape[0]).double()
        head.weight.copy_(torch.cat([self.A, -self.A], dim=1).double())
        head.bias.zero_()
        return list(self.backbone) + [pconv, nn.ReLU(), nn.Flatten(), head]

    def lipschitz_s(self) -> float:
        """Sound l-inf Lipschitz of the prefix x -> s (backbone then shared paired conv response)."""
        L = 1.0
        for m in self.backbone:
            if isinstance(m, nn.Conv2d):
                L *= conv_induced_inf_norm(m.weight)
        L *= conv_induced_inf_norm(self.pc_weight)  # response uses shared weight (no bias)
        return L

    @torch.no_grad()
    def project_lipschitz(self, lam: float) -> None:
        """Project prefix convs to induced-inf-norm <= lam so L_s stays small -> tight native cert."""
        for m in self.backbone:
            if isinstance(m, nn.Conv2d):
                n = conv_induced_inf_norm(m.weight)
                if n > lam:
                    m.weight.mul_(lam / n)
        n = conv_induced_inf_norm(self.pc_weight)
        if n > lam:
            self.pc_weight.mul_(lam / n)


class _ExportWrapper(nn.Module):
    """Standard-module version (for torch.onnx via to_modules)."""
    def __init__(self, modules): super().__init__(); self.seq = nn.Sequential(*modules)
    def forward(self, x):
        if x.dim() != 4:
            x = x.view(-1, 1, IMG, IMG)
        return self.seq(x)


@dataclass
class TrainResult:
    model: PairedBiasCNN
    export_model: nn.Module
    test_acc: float
    baseline_acc: float
    test_X: torch.Tensor
    test_y: torch.Tensor


def train_paired(data_dir, depth=2, channels=4, num_pairs=8, epochs=8, seed=0, data=None,
                 project_lam=None) -> TrainResult:
    torch.manual_seed(seed)
    (Xtr, ytr), (Xte, yte) = data if data is not None else load_mnist_8x8(data_dir)
    model = PairedBiasCNN(depth, channels, num_pairs)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.CrossEntropyLoss()
    n = Xtr.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad(); lossf(model(Xtr[idx]), ytr[idx]).backward(); opt.step()
            if project_lam is not None:
                model.project_lipschitz(project_lam)
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    # baseline: ordinary small CNN, matched size
    from .contractive_cnn import ContractiveCNN, _train_one
    torch.manual_seed(seed)
    base = ContractiveCNN(depth, channels)
    bacc = _train_one(base, Xtr, ytr, Xte, yte, epochs, lam=1e9, project=False)
    md = model.double()
    return TrainResult(model=md, export_model=_ExportWrapper(md.to_modules()).double().eval(),
                       test_acc=acc, baseline_acc=bacc, test_X=Xte.double(), test_y=yte)


@torch.no_grad()
def native_certificate(model: PairedBiasCNN, x0: torch.Tensor, y: int, eps: float) -> Dict:
    """Rigorous coupled-interval whole-net margin lower bound (float64)."""
    model = model.double().eval()
    x0 = x0.double().view(1, 1, IMG, IMG)
    P = model.P
    b, c = (t.double() for t in model._bc())
    # nominal s at x0 (shared response, no bias)
    h0 = model.backbone(x0)
    s0 = F.conv2d(h0, model.pc_weight.double(), None, padding=model.pad).reshape(P, IMG * IMG)  # (P, HW)
    Ls = model.lipschitz_s()
    s_lo, s_hi = s0 - Ls * eps, s0 + Ls * eps
    # d_i(s) = relu(s+b)-relu(s+c) is monotone nondecreasing in s -> interval [d(s_lo), d(s_hi)]
    def d(s, bi, ci):
        return torch.clamp(s + bi.view(P, 1), min=0) - torch.clamp(s + ci.view(P, 1), min=0)
    d_lo = d(s_lo, b, c).reshape(-1)   # (P*HW,)
    d_hi = d(s_hi, b, c).reshape(-1)
    A = model.A.double()               # (K, P*HW)
    K = A.shape[0]
    logits0 = model(x0).reshape(-1)
    L_ana = float("inf"); per_k = {}
    for k in range(K):
        if k == y:
            continue
        alpha = A[y] - A[k]            # (P*HW,)
        # min over box of sum alpha_j d_j  = sum(alpha+ * d_lo + alpha- * d_hi)
        lb = float((torch.clamp(alpha, min=0) * d_lo + torch.clamp(alpha, max=0) * d_hi).sum())
        per_k[int(k)] = {"native_lb": lb, "mu_at_x0": float(logits0[y] - logits0[k])}
        L_ana = min(L_ana, lb)
    return {"analytical_lower_bound": L_ana, "L_s": Ls,
            "mu_at_x0_min": float(min(v["mu_at_x0"] for v in per_k.values())),
            "per_competitor": per_k, "label": int(y), "epsilon": float(eps)}
