"""Backbone-agnostic Paired-Bias HEAD for the scale / arbitrary-architecture demo.

The paired-bias certificate needs only two things of the upstream network Psi: (1) the
head hard-zeros every non-label logit, and (2) each pair contributes a non-negative
monotone gap. So it composes with ANY frozen backbone Psi (a real, downsampling,
heterogeneous CNN), at ANY input scale, with O(1) ground-truth cost:

    phi        = Flatten(Psi(x))                      # arbitrary backbone features (dim F)
    s_i        = w_i . phi                            # SHARED linear weight for the pair
    f_label(x) = margin + sum_i scale*(ReLU(s_i+b_i) - ReLU(s_i+c_i)),   b_i > c_i
    f_k(x)     = 0   (k != label)

Because ReLU is monotone and b_i > c_i, every pair term is >= 0, so f_label >= margin > 0
= f_k for EVERY x and every Psi -> analytically robust, no solver, cost independent of the
backbone size. Biases are centered at s_i(x0) so each pair straddles 0 near x0 (both ReLUs
unstable) -> a genuine stress test, exactly as in the MILP/MEAP families.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class PairedBiasHeadNet(nn.Module):
    """Frozen backbone Psi + linear paired-bias head with a hard-zeroed readout."""

    def __init__(self, backbone: nn.Module, feat_dim: int, in_shape: Tuple[int, int, int],
                 num_pairs: int, num_classes: int, label: int, margin: float):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.in_shape = in_shape                 # (C, H, W)
        self.num_pairs = num_pairs
        self.num_classes = num_classes
        self.label = label
        self.margin = float(margin)
        self.flatten = nn.Flatten()
        self.paired = nn.Linear(feat_dim, 2 * num_pairs, bias=True)
        nn.init.kaiming_normal_(self.paired.weight, nonlinearity="relu")
        self.relu = nn.ReLU()
        self.fc = nn.Linear(2 * num_pairs, num_classes, bias=True)
        self._enforce_weight_sharing()
        self._setup_output_layer()

    def _enforce_weight_sharing(self):
        with torch.no_grad():
            P = self.num_pairs
            self.paired.weight.data[P:] = self.paired.weight.data[:P].clone()

    def _setup_output_layer(self):
        P = self.num_pairs
        scale = 1.0 / P
        with torch.no_grad():
            self.fc.weight.zero_(); self.fc.bias.zero_()
            for i in range(P):
                self.fc.weight[self.label, i] = +scale        # +ReLU(s_i + b_i)
                self.fc.weight[self.label, i + P] = -scale     # -ReLU(s_i + c_i)
            self.fc.bias[self.label] = self.margin

    @torch.no_grad()
    def features(self, x):
        if x.dim() != 4:
            x = x.view(-1, *self.in_shape)
        return self.flatten(self.backbone(x))

    def forward(self, x):
        if x.dim() != 4:
            x = x.view(-1, *self.in_shape)
        phi = self.flatten(self.backbone(x))
        s = self.paired(phi)                                   # (batch, 2P), s[:, i]==s[:, i+P] pre-bias
        return self.fc(self.relu(s))


@torch.no_grad()
def set_paired_biases(model: PairedBiasHeadNet, x0: torch.Tensor, delta: float):
    """Center each pair at s_i(x0): b_i = -s_i(x0)+delta, c_i = -s_i(x0)-delta (b_i>c_i).

    Backbone-agnostic: uses only a forward pass of the frozen Psi at x0 (no IBP), so it
    works for any architecture. Near x0 the pair straddles 0 (both ReLUs unstable)."""
    model.eval()
    P = model.num_pairs
    phi = model.features(x0.view(1, *model.in_shape))         # (1, F)
    w = model.paired.weight.data[:P]                          # (P, F) shared
    s0 = (phi @ w.T).reshape(-1)                              # (P,) pre-bias s_i(x0)
    b = model.paired.bias.data
    b[:P] = -s0 + delta
    b[P:] = -s0 - delta
    assert bool((b[:P] - b[P:] > 0).all()), "b_i > c_i violated"


@torch.no_grad()
def certify(model: PairedBiasHeadNet, x0: torch.Tensor):
    """Return the analytic GT dict; asserts f_label(x0) correct and margin>0 structurally.

    The certificate (margin >= self.margin, f_k=0) holds by construction for all x; here we
    also report the empirical margin at x0 and the fraction of pairs that are unstable near x0.
    """
    model.eval()
    logits = model(x0.view(1, *model.in_shape)).reshape(-1)
    others = torch.cat([logits[:model.label], logits[model.label + 1:]])
    emp_margin = float(logits[model.label] - others.max())
    # unstable pairs near x0: both ReLU(s+b), ReLU(s+c) straddle 0 => s in (-b, -c) region
    phi = model.features(x0.view(1, *model.in_shape))
    P = model.num_pairs
    s0 = (phi @ model.paired.weight.data[:P].T).reshape(-1)
    b_i = model.paired.bias.data[:P]; c_i = model.paired.bias.data[P:]
    unstable = float(((s0 + b_i > 0) & (s0 + c_i < 0)).float().mean())
    return {
        "construction": "cnn.paired_bias_head",
        "ground_truth_source": "paired_bias_certificate",
        "is_robust": True,
        "certified_margin_lb": model.margin,
        "empirical_margin_x0": emp_margin,
        "num_pairs": P,
        "unstable_pair_fraction_x0": unstable,
    }
