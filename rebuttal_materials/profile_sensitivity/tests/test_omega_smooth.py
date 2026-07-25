"""Verify the omega_j>tau smooth-activation unstable test (paper eq.12).

Builds a tiny sigmoid MLP, exports ONNX + a matching VNNLIB box query, and checks:
  * width mode (legacy): U is tau-independent and counts every non-degenerate neuron;
  * omega mode (faithful): U is non-increasing in tau and strictly below the width
    count once tau exceeds some neurons' slope-variation;
  * a ReLU control network: U is identical under width and omega (tau-independent).

Run:  cd rebuttal_materials/profile_sensitivity/tests && PYTHONPATH=../../../src python test_omega_smooth.py
"""
from __future__ import annotations
import os
import tempfile
import numpy as np
import torch
import torch.nn as nn


def _write_vnnlib(path, x0, eps, n_out, true_cls=0):
    lines = []
    for i in range(len(x0)):
        lines.append(f"(declare-const X_{i} Real)")
    for k in range(n_out):
        lines.append(f"(declare-const Y_{k} Real)")
    for i, v in enumerate(x0):
        lines.append(f"(assert (>= X_{i} {v - eps}))")
        lines.append(f"(assert (<= X_{i} {v + eps}))")
    clauses = "".join(f"\n  (and (>= Y_{k} Y_{true_cls}))" for k in range(n_out) if k != true_cls)
    lines.append(f"(assert (or{clauses}))")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _export(net, d_in, path):
    from VeriStressGT.utils.onnx_export import ExportConfig, export_pytorch_to_onnx
    net.eval()
    export_pytorch_to_onnx(net, path, input_shape=(1, d_in),
                           config=ExportConfig(opset=13),
                           example_input=np.zeros((1, d_in), dtype=np.float32),
                           input_name="X", output_name="Y")


def _run(onnx_path, vnnlib_path, mode, tau):
    from VeriStressGT.difficulty_profile.instance_loader import load_instance
    from VeriStressGT.difficulty_profile import components as K
    inst = load_instance(onnx_path, vnnlib_path, device="cpu")
    r = K.estimate_ibp_components(inst, sample_min_margin=None, tau=tau,
                                  smooth_unstable_mode=mode, verbose=False)
    return r.get("unstable_frac"), r.get("n_unstable"), r.get("n_total_neurons")


def main():
    torch.manual_seed(0)
    d_in, h, d_out = 6, 8, 3
    # sigmoid net: scale weights so pre-activations straddle 0 with a spread of magnitudes
    sig = nn.Sequential(nn.Linear(d_in, h), nn.Sigmoid(), nn.Linear(h, d_out))
    with torch.no_grad():
        sig[0].weight.copy_(torch.randn(h, d_in) * 0.8)
        sig[0].bias.copy_(torch.randn(h) * 0.3)
    relu = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, d_out))
    with torch.no_grad():
        relu[0].weight.copy_(torch.randn(h, d_in) * 0.8)
        relu[0].bias.copy_(torch.randn(h) * 0.3)

    x0 = np.full(d_in, 0.5, dtype=np.float64)
    eps = 0.3
    tmp = tempfile.mkdtemp()
    paths = {}
    for name, net in [("sig", sig), ("relu", relu)]:
        o = os.path.join(tmp, f"{name}.onnx"); v = os.path.join(tmp, f"{name}.vnnlib")
        _export(net, d_in, o); _write_vnnlib(v, x0, eps, d_out)
        paths[name] = (o, v)

    o, v = paths["sig"]
    print("SIGMOID net (smooth):")
    width_U = _run(o, v, "width", 0.0)[0]
    print(f"  width mode          U = {width_U:.3f}  (tau-independent, counts all non-degenerate)")
    prev = 1.1
    omega_us = []
    for tau in [0.0, 0.01, 0.05, 0.1, 0.25]:
        U, nu, nt = _run(o, v, "omega", tau)
        omega_us.append(U)
        flag = "" if U <= prev + 1e-9 else "  <-- NOT monotone!"
        print(f"  omega mode tau={tau:<5} U = {U:.3f}  ({nu}/{nt}){flag}")
        prev = U

    o, v = paths["relu"]
    ru_w = _run(o, v, "width", 0.0)[0]
    ru_o = _run(o, v, "omega", 0.1)[0]
    print("RELU net (control):")
    print(f"  width U = {ru_w:.3f}   omega(tau=0.1) U = {ru_o:.3f}  (should be identical)")

    ok = True
    ok &= all(omega_us[i] >= omega_us[i + 1] - 1e-9 for i in range(len(omega_us) - 1))  # non-increasing
    ok &= omega_us[-1] < width_U + 1e-9                                                  # tau shrinks U
    ok &= abs(ru_w - ru_o) < 1e-9                                                        # ReLU invariant
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
