"""ONNX/VNNLIB export + parity for the trained realism constructors."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def export_onnx(model: torch.nn.Module, onnx_path: str, input_shape: Tuple[int, ...]) -> None:
    import onnx
    model = model.float().eval()
    dummy = torch.zeros(1, *input_shape, dtype=torch.float32)
    torch.onnx.export(model, dummy, onnx_path, export_params=True, opset_version=11,
                      do_constant_folding=False, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}}, dynamo=False)
    onnx.checker.check_model(onnx.load(onnx_path))


def parity_max_err(model: torch.nn.Module, onnx_path: str, x0: torch.Tensor, eps: float,
                   input_shape, domain=(0.0, 1.0), n_rand: int = 16) -> float:
    """Max abs logit error torch(float64) vs ONNXRuntime over x0 + random box points."""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    ishape = [d if isinstance(d, int) and d > 0 else 1 for d in sess.get_inputs()[0].shape]
    model = model.double().eval()
    x0f = x0.double().reshape(-1)
    lo = torch.clamp(x0f - eps, *domain); hi = torch.clamp(x0f + eps, *domain)
    pts = [x0f] + [lo + torch.rand_like(x0f) * (hi - lo) for _ in range(n_rand)]
    mx = 0.0
    for xf in pts:
        t = model(xf.reshape(1, *input_shape)).reshape(-1).detach().numpy()
        o = sess.run(None, {iname: xf.reshape(ishape).float().numpy()})[0].reshape(-1)
        mx = max(mx, float(np.abs(t - o).max()))
    return mx


def write_vnnlib(x0: torch.Tensor, eps: float, label: int, num_outputs: int, out_path: str,
                 domain=(0.0, 1.0)) -> None:
    x0f = x0.double().reshape(-1).numpy()
    lo_d, hi_d = domain
    with open(out_path, "w") as f:
        for i in range(x0f.size):
            f.write(f"(declare-const X_{i} Real)\n")
        for i in range(num_outputs):
            f.write(f"(declare-const Y_{i} Real)\n")
        for i in range(x0f.size):
            f.write(f"(assert (>= X_{i} {max(float(x0f[i]-eps), lo_d)}))\n")
            f.write(f"(assert (<= X_{i} {min(float(x0f[i]+eps), hi_d)}))\n")
        f.write("(assert (or\n")
        for k in range(num_outputs):
            if k != label:
                f.write(f"  (and (>= Y_{k} Y_{label}))\n")
        f.write("))\n")


def write_single_instance_benchmark(out_dir: Path, inst_id: str, model, x0, eps, label,
                                    num_outputs, input_shape, gt: dict, domain=(0.0, 1.0)) -> dict:
    """Export ONNX+VNNLIB+meta into a verify_benchmark-compatible 1-instance benchmark dir."""
    inst = out_dir / "instances" / inst_id
    inst.mkdir(parents=True, exist_ok=True)
    onnx_path = inst / "model.onnx"
    export_onnx(model, str(onnx_path), input_shape)
    perr = parity_max_err(model, str(onnx_path), x0, eps, input_shape, domain)
    write_vnnlib(x0, eps, label, num_outputs, str(inst / "spec.vnnlib"), domain)
    meta = {"id": inst_id, "construction": gt.get("construction", "realism"),
            "paths": {"onnx": "model.onnx", "vnnlib": "spec.vnnlib", "meta": "meta.json"},
            "is_robust": bool(gt.get("is_robust", True)), "parity_max_err": perr, "gt": gt}
    (inst / "meta.json").write_text(json.dumps(meta, indent=2))
    manifest = {"name": out_dir.name, "n_instances": 1,
                "instances": [{"id": inst_id, "construction": meta["construction"],
                               "paths": {"onnx": f"instances/{inst_id}/model.onnx",
                                         "vnnlib": f"instances/{inst_id}/spec.vnnlib",
                                         "meta": f"instances/{inst_id}/meta.json"},
                               "is_robust": meta["is_robust"], "gt": gt}]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"onnx": str(onnx_path), "parity_max_err": perr}
