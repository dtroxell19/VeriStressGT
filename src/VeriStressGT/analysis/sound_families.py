"""Per-family rigorous re-certification of VeriStress-GT ground-truth margins.

Each function reads the EXACT shipped fp32 weights from the instance's ONNX
initializers, converts them to `fractions.Fraction` (exact dyadic), verifies the
family's analytic certificate, and returns a FamilyResult with a certified margin
lower bound and whether it is provably > 0. See `sound_backends` for the arithmetic
and the plan/REPORT for the per-family certificate arguments.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional

import numpy as np
import onnx
from onnx import numpy_helper

from . import sound_backends as SB

FR = Fraction


@dataclass
class FamilyResult:
    arithmetic: str
    certified_lb: object                 # Fraction or float (interval lower)
    certified_positive: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    notes: str = ""


# --------------------------------------------------------------------------- #
# Shared I/O
# --------------------------------------------------------------------------- #
def load_inits(onnx_path: str) -> Dict[str, np.ndarray]:
    """name -> exact fp32 array for every initializer in the graph."""
    m = onnx.load(onnx_path)
    return {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}


def parse_vnnlib_box(vnnlib_path: str):
    """Return (lo, hi, label) from a box-robustness VNNLIB.

    lo/hi are exact-Fraction arrays over the input coords; label is the reference
    class Y_label such that robustness == (Y_label > Y_k for all k != label).
    """
    txt = open(vnnlib_path).read()
    n_x = len(set(re.findall(r"declare-const X_(\d+)", txt)))
    lo = [None] * n_x
    hi = [None] * n_x
    for i, v in re.findall(r"\(assert \(>= X_(\d+) ([-\d.eE+]+)\)\)", txt):
        lo[int(i)] = Fraction(float(v))
    for i, v in re.findall(r"\(assert \(<= X_(\d+) ([-\d.eE+]+)\)\)", txt):
        hi[int(i)] = Fraction(float(v))
    refs = re.findall(r"\(>= Y_(\d+) Y_(\d+)\)", txt)
    label = int(refs[0][1]) if refs else 0
    return (np.array(lo, dtype=object), np.array(hi, dtype=object), label)


def _center_eps(lo, hi):
    x0 = np.array([(a + b) / 2 for a, b in zip(lo, hi)], dtype=object)
    return x0


# --------------------------------------------------------------------------- #
# Exact-Fraction MLP forward
# --------------------------------------------------------------------------- #
def frac_mlp_forward(layers, x):
    """layers = [(W,b), 'relu', (W,b), ...]  (W,b exact Fraction). Returns logits."""
    h = x
    for L in layers:
        if L == "relu":
            h = SB.frac_relu(h)
        else:
            W, b = L
            h = W @ h + b
    return h


# --------------------------------------------------------------------------- #
# Family A: Paired-Biases (cnn.cnn_paired_bias)
# --------------------------------------------------------------------------- #
def recert_paired_bias(onnx_path: str, vnnlib_path: str) -> FamilyResult:
    W = load_inits(onnx_path)
    pc_w = SB.to_frac_array(W["paired_conv_weight"])   # (2P, cin, k, k)
    pc_b = SB.to_frac_array(W["paired_conv_bias"])     # (2P,)
    fc_w = SB.to_frac_array(W["fc_weight"])            # (K, 2P*HW)
    fc_b = SB.to_frac_array(W["fc_bias"])              # (K,)
    _, _, label = parse_vnnlib_box(vnnlib_path)

    twoP = pc_w.shape[0]
    P = twoP // 2
    K, total = fc_w.shape
    HW = total // twoP
    checks = {}

    # (1) conv weight-sharing: channels [P:2P] == [:P] exactly
    checks["weight_sharing"] = bool(np.all(pc_w[P:] == pc_w[:P]))

    # (2) fc label row is the +/-scale paired pattern; (3) other rows/biases zero
    row = fc_w[label]
    ok_pattern = True
    for i in range(P):
        pos = row[i * HW:(i + 1) * HW]
        neg = row[(i + P) * HW:(i + P + 1) * HW]
        # each pos entry >= 0 and its paired neg entry == -pos (exact float negation)
        if not (np.all(np.array([p >= 0 for p in pos])) and np.all(pos == -neg)):
            ok_pattern = False
            break
    checks["fc_paired_pattern"] = bool(ok_pattern)
    other = np.array([k for k in range(K) if k != label])
    checks["other_logits_zero"] = bool(
        np.all(fc_w[other] == FR(0)) and np.all(fc_b[other] == FR(0)))

    # (5) b_i > c_i  (paired_conv_bias[:P] - [P:] > 0)
    checks["pair_gap_positive"] = bool(np.all(np.array([g > 0 for g in (pc_b[:P] - pc_b[P:])])))

    # certified lb = fc.bias[label]  (f_label >= margin, f_k = 0)
    lb = fc_b[label]
    positive = all(checks.values()) and lb > 0
    return FamilyResult("exact_rational", lb, bool(positive), checks,
                        notes=f"P={P}, HW={HW}, lb=fc_bias[label]")


# --------------------------------------------------------------------------- #
# Family B: MEAP (mlp_relu.meap)
# --------------------------------------------------------------------------- #
def _selector_pattern_ok(M, ones_cols):
    """M is (rows, cols) Fraction; row r must be e_{ones_cols[r]} (1 at that col, 0 else)."""
    rows, cols = M.shape
    if len(ones_cols) != rows:
        return False
    for r in range(rows):
        for c in range(cols):
            want = FR(1) if c == ones_cols[r] else FR(0)
            if M[r, c] != want:
                return False
    return True


def recert_meap(onnx_path: str, vnnlib_path: str) -> FamilyResult:
    W = load_inits(onnx_path)
    _, _, label = parse_vnnlib_box(vnnlib_path)
    pW = SB.to_frac_array(W["pairs.W_T_for_gemm"])   # (2P, in)
    pb = SB.to_frac_array(W["pairs.b"])              # (2P,)
    twoP = pW.shape[0]
    P = twoP // 2
    checks = {}

    # (1) first-layer pairing: row 2p+1 == -row 2p
    checks["pair_negation"] = bool(np.all(pW[1::2] == -pW[0::2]))
    # (2) all pair biases equal gamma (single value) and > 0
    gamma = pb[0]
    checks["biases_equal_gamma"] = bool(np.all(pb == gamma))
    checks["gamma_positive"] = bool(gamma > 0)

    # (3) min/max tree constant matrices are exactly the intended selectors / -I
    sel_even = SB.to_frac_array(W["sel.W_even_T_for_gemm"])       # (P, 2P)
    sel_odd = SB.to_frac_array(W["sel.W_odd_T_for_gemm"])         # (P, 2P)
    sel_odd_neg = SB.to_frac_array(W["sel.W_odd_neg_T_for_gemm"])
    checks["sel_even"] = _selector_pattern_ok(sel_even, [2 * p for p in range(P)])
    checks["sel_odd"] = _selector_pattern_ok(sel_odd, [2 * p + 1 for p in range(P)])
    checks["sel_odd_neg"] = bool(np.all(sel_odd_neg == -sel_odd))
    # min-tree levels: neg_*.Wneg == -I
    tree_ok = True
    for name, arr in W.items():
        if re.search(r"mintree\.neg_\d+\.Wneg", name):
            M = SB.to_frac_array(arr)
            n = M.shape[0]
            if not np.all(M == -np.eye(n, dtype=object)):
                tree_ok = False
    checks["neg_identity"] = bool(tree_ok)
    # output embed: one-hot at label (out.W_T_for_gemm is (num_classes, 1))
    outW = SB.to_frac_array(W["out.W_T_for_gemm"])
    exp = np.zeros(outW.shape, dtype=object)
    exp[label, 0] = FR(1)
    checks["output_onehot"] = bool(np.all(outW == exp))

    lb = gamma
    positive = all(checks.values()) and lb > 0
    return FamilyResult("exact_rational", lb, bool(positive), checks,
                        notes=f"P={P}, lb=gamma (f_label>=gamma pointwise)")


# --------------------------------------------------------------------------- #
# Family C: Constant-on-Box (mlp_relu.embedded_projection)
# --------------------------------------------------------------------------- #
def _downstream_layers(W):
    """Ordered [(Wl,bl),'relu',...] for downstream.net.* (biases default 0)."""
    idx = sorted(int(m.group(1)) for k in W for m in [re.match(r"downstream\.net\.(\d+)\.weight", k)] if m)
    layers = []
    for j, li in enumerate(idx):
        Wl = SB.to_frac_array(W[f"downstream.net.{li}.weight"])
        bkey = f"downstream.net.{li}.bias"
        bl = SB.to_frac_array(W[bkey]) if bkey in W else np.array([FR(0)] * Wl.shape[0], dtype=object)
        layers.append((Wl, bl))
        if j != len(idx) - 1:
            layers.append("relu")
    return layers


def recert_embedded_projection(onnx_path: str, vnnlib_path: str) -> FamilyResult:
    """Rigorous exact-rational IBP through projection + downstream.

    The projection is *designed* to be constant over the box (each embedded ReLU
    inactive), but at the default eta=0 the shipped fp32 alpha_i can exceed lo_i by
    ~1 ULP, so the ReLU is barely active and the projection is not EXACTLY constant.
    Rather than assert exact constancy, we propagate the box through the whole
    network in exact-rational IBP: the projection collapses it to a ~1-ULP-wide
    interval around c, so downstream IBP stays tight and the margin lower bound is
    margin(c) minus a negligible slack -> still provably > 0.
    """
    W = load_inits(onnx_path)
    lo, hi, label = parse_vnnlib_box(vnnlib_path)
    l1w = SB.to_frac_array(W["projection.linear1.weight"])
    l1b = SB.to_frac_array(W["projection.linear1.bias"])
    l2w = SB.to_frac_array(W["projection.linear2.weight"])
    l2b = SB.to_frac_array(W["projection.linear2.bias"])
    c = l2b
    d = l1w.shape[1]
    checks = {}

    # structural sanity (interpretability; not load-bearing for the IBP certificate)
    struct = all(l1w[2 * i, i] == FR(-1) and l1w[2 * i + 1, i] == FR(1) for i in range(d))
    checks["proj_l1_structure"] = bool(struct)
    l2ok = all(l2w[i, 2 * i] == FR(1) and l2w[i, 2 * i + 1] == FR(1) for i in range(d))
    checks["proj_l2_structure"] = bool(l2ok)
    # how tight does the projection collapse the box? (report; not a pass/fail)
    plo, phi = SB.frac_ibp([(l1w, l1b), "relu", (l2w, l2b)], lo, hi)
    proj_width = max(phi[i] - plo[i] for i in range(d))

    # rigorous certificate: exact-rational IBP margin lower bound over the full net
    layers = [(l1w, l1b), "relu", (l2w, l2b)] + _downstream_layers(W)
    lb = SB.frac_ibp_margin_lb(layers, lo, hi, label)
    checks["ibp_margin_positive"] = bool(lb > 0)
    positive = struct and l2ok and lb > 0
    return FamilyResult("exact_rational", lb, bool(positive), checks,
                        notes=f"exact IBP over full net; projection box-width={float(proj_width):.2e}")


# --------------------------------------------------------------------------- #
# Family D: Input-Corner Stress (mlp_relu.corners) — concave -> exact corner min
# --------------------------------------------------------------------------- #
def recert_corners(onnx_path: str, vnnlib_path: str, max_corner_bits: int = 20) -> FamilyResult:
    W = load_inits(onnx_path)
    lo, hi, label = parse_vnnlib_box(vnnlib_path)
    sel = SB.to_frac_array(W["selector.weight"])          # (r, d) = [I_r|0]
    Ah = SB.to_frac_array(W["hinge.weight"])              # (m, r)
    bh = SB.to_frac_array(W["hinge.bias"])                # (m,)
    C = SB.to_frac_array(W["competitor_layer.weight"])    # (K, m), >= 0
    cb = SB.to_frac_array(W["competitor_layer.bias"])     # (K,) = -beta
    Wout = SB.to_frac_array(W["output_layer.weight"])     # (num_classes, K)
    bout = SB.to_frac_array(W["output_layer.bias"])       # (num_classes,)
    checks = {}

    # concavity justification: competitor weights nonnegative
    checks["competitor_nonneg"] = bool(np.all(np.array([v >= 0 for v in C.reshape(-1)])))
    # active coords = columns the selector reads (its 1-positions)
    active = sorted({int(j) for r in range(sel.shape[0])
                     for j in range(sel.shape[1]) if sel[r, j] != 0})
    r = len(active)
    x0 = _center_eps(lo, hi)
    if r > max_corner_bits:
        return FamilyResult("exact_rational", None, False, checks,
                            notes=f"active_dim={r} too large for corner enumeration")

    def net_margin(xvec):
        xa = sel @ xvec
        h = SB.frac_relu(Ah @ xa + bh)
        z = C @ h + cb
        logits = Wout @ z + bout
        return SB.frac_margin(list(logits), label)

    # min over 2^r active corners (concave -> min at a corner)
    worst = None
    for mask in range(1 << r):
        x = x0.copy()
        for bit, coord in enumerate(active):
            x[coord] = hi[coord] if (mask >> bit) & 1 else lo[coord]
        mgn = net_margin(x)
        worst = mgn if worst is None else min(worst, mgn)
    checks["corner_min_computed"] = True
    positive = checks["competitor_nonneg"] and worst is not None and worst > 0
    return FamilyResult("exact_rational", worst, bool(positive), checks,
                        notes=f"active_dim={r}, exact min over {1 << r} corners (concave margin)")


# --------------------------------------------------------------------------- #
# Family E: Deep-Contractive CNN (cnn.deep_contractive_cnn)
# --------------------------------------------------------------------------- #
def recert_deep_contractive(onnx_path: str, vnnlib_path: str, cert_json: dict = None) -> FamilyResult:
    if cert_json is None:
        import os, json as _json
        cj = os.path.join(os.path.dirname(onnx_path), "certificate.json")
        if os.path.exists(cj):
            cert_json = _json.load(open(cj))
    """Structural exact certificate (primary) + rigorous Lipschitz recheck (secondary).

    The network ends ...Conv -> ReLU -> Flatten -> Gemm with fc.weight[label] all
    = 1/flat_dim > 0 and every other output row/bias zero. Post-ReLU features are
    >= 0 and the readout is positive, so f_label(x) = B + w_out.Phi(x) >= B and
    f_k = 0 -> margin >= B = fc.bias[label], POINTWISE and exactly. That is the
    rigorous certificate (lb = B). Separately we recompute the paper's Lipschitz
    perturbation bound with SOUND spectral upper bounds to check whether the shipped
    convs actually meet the design contraction lambda^depth.
    """
    m = onnx.load(onnx_path)
    ops = [n.op_type for n in m.graph.node]
    W = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    _, _, label = parse_vnnlib_box(vnnlib_path)
    fc_w = SB.to_frac_array(W["core_fc_weight"])
    fc_b = SB.to_frac_array(W["core_fc_bias"])
    K = fc_w.shape[0]
    checks = {}

    # structural certificate
    checks["ends_in_relu_before_readout"] = (ops[-2:] == ["Flatten", "Gemm"] and "Relu" in ops[:-2]
                                             and ops[[i for i, o in enumerate(ops) if o == "Flatten"][0] - 1] == "Relu")
    checks["readout_row_nonneg"] = bool(np.all(np.array([v >= 0 for v in fc_w[label]])))
    other = np.array([k for k in range(K) if k != label])
    checks["other_logits_zero"] = bool(np.all(fc_w[other] == FR(0)) and np.all(fc_b[other] == FR(0)))
    lb = fc_b[label]                 # margin >= B pointwise
    positive = all(checks.values()) and lb > 0

    # Secondary (informational only, NOT part of the certificate): the paper's cert
    # uses the l2 contraction lambda^depth. A cheap sound bound sqrt(||W||_1*||W||_inf)
    # on a conv's l2 operator norm is very loose (it ignores the spatial/circulant
    # structure the power-iteration certificate relies on), so it cannot re-verify the
    # l2 contraction and is not used for positivity. Positivity is established purely
    # structurally (margin >= B), which needs no contraction assumption.
    lip_note = ""
    if cert_json and "core_input_proj_weight" in W:
        D = int(cert_json["depth"])
        lip_note = (f"; paper's l2-contraction cert not re-verified here (cheap induced-norm bound is "
                    f"loose for conv operators); positivity is structural and contraction-independent, depth={D}")
    return FamilyResult("exact_rational", lb, bool(positive), checks,
                        notes=f"structural: margin>=B={float(lb):.4g}{lip_note}")


# --------------------------------------------------------------------------- #
# Generic MLP-layer extraction from an ONNX Gemm/Relu chain
# --------------------------------------------------------------------------- #
def mlp_layers_from_onnx(onnx_path):
    """Ordered [(W,b) Fraction, 'relu', ...] for a Gemm/(MatMul+Add)/Relu MLP.

    Gemm weight is (out,in) with transB=1 (nn.Linear), so forward = W@x+b. Skips
    Flatten/Reshape/Identity. Raises on unsupported ops.
    """
    m = onnx.load(onnx_path)
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    layers = []
    for node in m.graph.node:
        op = node.op_type
        if op == "Gemm":
            W = SB.to_frac_array(inits[node.input[1]])
            if not any(a.name == "transB" and a.i for a in node.attribute):
                W = W.T
            bn = node.input[2] if len(node.input) > 2 else None
            b = SB.to_frac_array(inits[bn]) if bn and bn in inits \
                else np.array([FR(0)] * W.shape[0], dtype=object)
            layers.append((W, b))
        elif op == "Relu":
            layers.append("relu")
        elif op in ("Flatten", "Reshape", "Identity", "Dropout"):
            pass
        else:
            raise ValueError(f"mlp_layers_from_onnx: unsupported op {op}")
    return layers


# --------------------------------------------------------------------------- #
# Family I: MILP exact-radius (mlp_relu.milp) — complete check
# --------------------------------------------------------------------------- #
def recert_milp(onnx_path: str, vnnlib_path: str, max_splits: int = 25) -> FamilyResult:
    """Re-certify robustness at the shipped eps. Exact-rational IBP first; if
    inconclusive (near-boundary eps_frac->1), escalate to an exact-rational complete
    check that branches on unstable ReLUs (tiny net) so every leaf is linear. Does NOT
    re-derive r*."""
    layers = mlp_layers_from_onnx(onnx_path)
    lo, hi, label = parse_vnnlib_box(vnnlib_path)
    lb = SB.frac_ibp_margin_lb(layers, lo, hi, label)
    if lb > 0:
        return FamilyResult("exact_rational", lb, True, {"ibp_positive": True},
                            notes="exact-rational IBP margin > 0")
    # box branch-and-bound is SOUND (over-approximation) so True => certified; a
    # non-closing result is INCONCLUSIVE (box relaxation can't track the input-space
    # halfspace of a ReLU split -> spurious infeasible leaves), NOT a robustness failure.
    ok, n_leaves = _complete_check(layers, lo, hi, label, max_splits)
    if ok:
        return FamilyResult("complete_check", None, True,
                            {"ibp_positive": False, "box_bab_robust": True},
                            notes=f"exact-rational box branch-and-bound closed ({n_leaves} leaves)")
    return FamilyResult("inconclusive", None, False,
                        {"ibp_positive": False, "box_bab_robust": False},
                        notes="exact-rational IBP + box-BaB too loose for this near-boundary MILP "
                              "instance; robustness at eps<r* rests on the exact-radius MILP (complete)")


def _complete_check(layers, lo, hi, label, max_splits):
    """Exact-rational branch-and-bound: branch on the PRE-activation sign of each
    unstable ReLU (staying on the same layer until all its unstable neurons are split),
    so every leaf is fully linear and its exact IBP margin is a true bound. Sound; leaves
    that still have unsplit unstable neurons at the budget fall back to the ReLU relaxation
    (may fail to close -> reported)."""
    import sys
    sys.setrecursionlimit(100000)
    leaves = [0]

    def rec(lo, hi, li, splits):
        while li < len(layers):
            L = layers[li]
            if L == "relu":
                unstable = [j for j in range(len(lo)) if lo[j] < 0 < hi[j]]
                if unstable and splits < max_splits:
                    j = unstable[0]
                    lo_i, hi_i = lo.copy(), hi.copy(); hi_i[j] = FR(0)   # inactive: z_j <= 0
                    lo_a, hi_a = lo.copy(), hi.copy(); lo_a[j] = FR(0)   # active:   z_j >= 0
                    return (rec(lo_i, hi_i, li, splits + 1)
                            and rec(lo_a, hi_a, li, splits + 1))
                lo, hi = SB.frac_interval_relu(lo, hi)
            else:
                W, b = L
                lo, hi = SB.frac_interval_linear(lo, hi, W, b)
            li += 1
        leaves[0] += 1
        others = [hi[k] for k in range(len(hi)) if k != label]
        return (lo[label] - max(others)) > 0

    return rec(lo, hi, 0, 0), leaves[0]


# --------------------------------------------------------------------------- #
# Families F/G: Attention (linear_dominance, fixed_pattern) — interval enclosure
# --------------------------------------------------------------------------- #
def recert_attention(onnx_path: str, vnnlib_path: str, prec_bits: int = 200) -> FamilyResult:
    """Rigorous mpmath.iv enclosure of the attention forward over the box.

    Dominant-Key is matmul/relu/reshape (no softmax); Fixed-Order adds a real
    softmax. Direct interval enclosure of the margin over B_eps(x0) with outward
    rounding -> lower endpoint > 0 certifies over the continuum, with no Lipschitz
    looseness (this is what closes the razor-thin margin_slack~1.0001 fixed-order set).
    """
    from . import iv_onnx
    lo, hi, label = parse_vnnlib_box(vnnlib_path)
    mlb, _ = iv_onnx.interval_margin_lb(onnx_path, lo, hi, label, prec_bits)
    return FamilyResult("interval_mpmath", mlb, bool(mlb > 0),
                        {"interval_enclosure_positive": bool(mlb > 0)},
                        notes=f"direct mpmath.iv box enclosure (prec={prec_bits} bits)")


# --------------------------------------------------------------------------- #
# Family H: Polynomial (polynomial.algebraic_boundary) — interval enclosure
# --------------------------------------------------------------------------- #
def recert_polynomial(onnx_path: str, vnnlib_path: str, prec_bits: int = 200,
                      subdiv_dims: int = 0, subdiv_k: int = 2) -> FamilyResult:
    """Rigorous mpmath.iv enclosure of the (nonconvex) polynomial margin over the box.

    Poly instances ship no ONNX (regenerated from args). We regenerate deterministically
    and interval-evaluate mu(x) = sum_j alpha_j (W1_j.x + b1_j)^degree + (b2_0-b2_1) over
    the committed box: the input->preactivation map is linear (tight) and the only
    nonlinearity is an elementwise integer power (iv encloses exactly per neuron), so the
    lower endpoint > 0 is a rigorous certificate despite nonconvexity. Optional bisection
    of the `subdiv_dims` widest input coords (k parts each) tightens the dependency slack.
    """
    import json as _json
    import os
    import tempfile
    from .soundness_audit import regenerate
    from . import iv_onnx

    inst_dir = os.path.dirname(vnnlib_path)
    meta = _json.load(open(os.path.join(inst_dir, "meta.json")))
    lo, hi, label = parse_vnnlib_box(vnnlib_path)
    deg = int(meta["args"].get("degree", 0))
    with tempfile.TemporaryDirectory() as t:
        onnx_re, _, _ = regenerate(meta, t)
        if not subdiv_dims:
            mlb, _ = iv_onnx.interval_margin_lb(onnx_re, lo, hi, label, prec_bits)
        else:
            widths = [(hi[i] - lo[i], i) for i in range(len(lo))]
            widths.sort(reverse=True)
            dims = [i for _, i in widths[:subdiv_dims]]
            mlb = _subdiv_min_margin(onnx_re, lo, hi, label, dims, subdiv_k, prec_bits)
    return FamilyResult("interval_mpmath", mlb, bool(mlb > 0),
                        {"interval_enclosure_positive": bool(mlb > 0)},
                        notes=f"degree={deg}, nonconvex shipped; iv box enclosure"
                              + (f" +{subdiv_k}^{subdiv_dims} subdivision" if subdiv_dims else ""))


def recert_polynomial_convex(onnx_path: str, vnnlib_path: str, degree: int = None) -> FamilyResult:
    """EXACT certificate for a convex (Prop-11) polynomial instance.

    mu(x) = sum_j alpha_j (W1_j.x + b1_j)^degree + (b2_0 - b2_1), alpha_j = W2[0,j]-W2[1,j].
    If every alpha_j >= 0 and degree is even, mu is convex, so over the l_inf box of radius
    eps:  min mu(x) >= mu(x0) - eps*||grad mu(x0)||_1  (convex first-order bound, dual norm).
    All quantities are exact rationals from the shipped fp32 params -> a machine-checked proof.
    """
    W = load_inits(onnx_path)
    lo, hi, label = parse_vnnlib_box(vnnlib_path)
    W1 = SB.to_frac_array(W["W1"]); b1 = SB.to_frac_array(W["b1"])
    W2 = SB.to_frac_array(W["W2"]); b2 = SB.to_frac_array(W["b2"])
    x0 = _center_eps(lo, hi)
    eps = (hi[0] - lo[0]) / 2
    if degree is None:
        import json as _json, os
        degree = int(_json.load(open(os.path.join(os.path.dirname(vnnlib_path), "meta.json")))["args"]["degree"])
    alpha = W2[0] - W2[1]                       # (hidden,)
    checks = {}
    checks["degree_even"] = (degree % 2 == 0)
    checks["alpha_nonneg"] = bool(np.all(np.array([a >= 0 for a in alpha])))
    s = W1 @ x0 + b1                            # exact preactivations
    mu_x0 = sum(alpha[j] * s[j] ** degree for j in range(len(alpha))) + (b2[0] - b2[1])
    coeff = np.array([alpha[j] * degree * s[j] ** (degree - 1) for j in range(len(alpha))], dtype=object)
    grad = W1.T @ coeff                         # exact gradient of mu at x0
    grad_l1 = sum(abs(g) for g in grad)
    lb = mu_x0 - eps * grad_l1                  # convex first-order lower bound over the box
    positive = checks["degree_even"] and checks["alpha_nonneg"] and lb > 0
    return FamilyResult("exact_rational", lb, bool(positive), checks,
                        notes=f"convex Prop-11, degree={degree}: lb=mu(x0)-eps*||grad||_1")


def _subdiv_min_margin(onnx_path, lo, hi, label, dims, k, prec_bits):
    """Min interval margin lower bound over a k-way bisection grid on `dims`."""
    from fractions import Fraction
    from . import iv_onnx
    grids = []
    for i in dims:
        step = (hi[i] - lo[i]) / k
        grids.append([(lo[i] + step * j, lo[i] + step * (j + 1)) for j in range(k)])
    worst = None
    import itertools
    for combo in itertools.product(*grids):
        slo = np.array(lo, dtype=object); shi = np.array(hi, dtype=object)
        for (a, b), i in zip(combo, dims):
            slo[i] = a; shi[i] = b
        mlb, _ = iv_onnx.interval_margin_lb(onnx_path, slo, shi, label, prec_bits)
        worst = mlb if worst is None else min(worst, mlb)
    return worst


# --------------------------------------------------------------------------- #
# Dispatch by construction id
# --------------------------------------------------------------------------- #
FAMILY_FUNCS = {
    "cnn.cnn_paired_bias": recert_paired_bias,
    "mlp_relu.meap": recert_meap,
    "mlp_relu.embedded_projection": recert_embedded_projection,
    "mlp_relu.corners": recert_corners,
    "cnn.deep_contractive_cnn": recert_deep_contractive,
    "attention.linear_dominance": recert_attention,
    "attention.fixed_pattern": recert_attention,
    "mlp_relu.milp.exact_radius": recert_milp,
    "polynomial.algebraic_boundary": recert_polynomial,
}
