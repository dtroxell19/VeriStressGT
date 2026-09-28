"""Ground-truth labels for a Thrust 1 benchmark, and witness-certified SAT twins.

UNSAT labels come from the constructors' analytic certificates (and, for the MILP family, the
exact-radius MILP), exactly as in the paper. SAT labels need no trust in any verifier: a SAT
instance ships a concrete witness x in the box whose margin is negative under both a float64
re-evaluation of the network and onnxruntime float32 inference.

SAT twins reuse a robust instance's network and center and scale the radius up. A bracket search
with the reference verifier locates the smallest scale k at which a counterexample exists, and
two twins are written:
  <id>_sat_near  eps = k_hi * eps0, just past the robustness threshold (hard to expose bugs)
  <id>_sat_far   eps = 2 * k_hi * eps0 (easy)
MEAP and paired-bias CNNs are robust for every radius, so they have no SAT twin.

Usage:
  python -m VeriStressGT.soundness.ground_truth --bench thrust1_bench --twins milp_ corners_ dc_cnn_
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from VeriStressGT.soundness.graph import DTYPE, UnsupportedOp
from VeriStressGT.soundness.refverify import load_problem, verify_box
from VeriStressGT.utils.make_box_vnnlib import make_box_vnnlib

ANALYTIC_CERT = {
    "mlp_relu.meap": "analytic: paired preactivations sum to 2*gamma (Prop. A.2)",
    "mlp_relu.milp.exact_radius": "exact-radius MILP: eps < r*",
    "mlp_relu.corners": "analytic: convex competitor logits maximised at box corners (Prop. 3)",
    "mlp_relu.embedded_projection": "analytic: network constant on the box",
    "cnn.deep_contractive_cnn": "analytic: Lipschitz contraction (Prop. 4)",
    "cnn.cnn_paired_bias": "analytic: ReLU monotonicity, global margin (Prop. 5)",
    "attention.fixed_pattern": "analytic: fixed score ordering + Lipschitz head (Prop. 7)",
    "attention.linear_dominance": "analytic: dominant key + Lipschitz head (Prop. 10)",
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def certify_witness(onnx_path: str, graph, C: np.ndarray, x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Tuple[bool, float, float]:
    """Witness is valid iff it lies in the box and violates the spec in float64 AND float32 (ORT)."""
    import onnxruntime as ort
    if not (np.all(x >= lo) and np.all(x <= hi)):
        return False, np.inf, np.inf
    xt = torch.tensor(x, dtype=DTYPE).reshape((1,) + graph.input_shape)
    m64 = float((graph.forward(xt) @ torch.tensor(C, dtype=DTYPE).T).min())
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    y32 = sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float32).reshape((1,) + graph.input_shape)})[0].reshape(-1)
    m32 = float((C @ y32.astype(np.float64)).min())
    return (m64 < 0 and m32 < 0), m64, m32


def _status(graph, C, center, radius, budget) -> Dict:
    return verify_box(graph, C, center, radius, "none", budget)


def find_threshold(graph, C, center, radius, *, budget: float = 20.0, bisections: int = 7,
                   k_max: float = 512.0) -> Optional[Tuple[float, float, Dict]]:
    """Bracket the radius scale where the instance flips from robust to non-robust.

    Returns (k_lo, k_hi, sat_result_at_k_hi) or None if no counterexample was found up to k_max.
    k_lo is the largest scale not shown SAT (proved robust or undecided).
    """
    k_lo, k_hi, hit = 1.0, None, None
    k = 2.0
    while k <= k_max:
        out = _status(graph, C, center, radius * k, budget)
        if out["status"] == "sat":
            k_hi, hit = k, out
            break
        k_lo, k = k, k * 2
    if k_hi is None:
        return None
    for _ in range(bisections):
        k = float(np.sqrt(k_lo * k_hi))
        out = _status(graph, C, center, radius * k, budget)
        if out["status"] == "sat":
            k_hi, hit = k, out
        else:
            k_lo = k
    return k_lo, k_hi, hit


def _write_twin(bench: Path, base: Dict, suffix: str, eps_scale: float, graph, spec, onnx_src: Path,
                witness: np.ndarray, bracket, budget: float) -> Optional[Dict]:
    center, radius = spec.center, spec.radius * eps_scale
    lo, hi = center - radius, center + radius
    ok, m64, m32 = certify_witness(str(onnx_src), graph, spec.C, witness, lo, hi)
    if not ok:  # re-search for a witness at this radius
        out = _status(graph, spec.C, center, radius, budget)
        if out["status"] != "sat":
            return None
        witness = np.asarray(out["witness"])
        ok, m64, m32 = certify_witness(str(onnx_src), graph, spec.C, witness, lo, hi)
        if not ok:
            return None
    iid = f"{base['id']}_sat_{suffix}"
    d = bench / "instances" / iid
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(onnx_src, d / "model.onnx")
    eps = float(radius.max())
    if not np.allclose(radius, eps):
        raise ValueError("non-uniform box radius; twin generation expects an l_inf ball")
    make_box_vnnlib(center=center.astype(np.float64), eps=eps, out=str(d / "spec.vnnlib"),
                    num_outputs=spec.C.shape[1], label=spec.label)
    # Re-certify against the box exactly as written to disk (text round-trip of the bounds).
    from VeriStressGT.soundness.spec import parse_vnnlib
    written = parse_vnnlib(str(d / "spec.vnnlib"), spec.C.shape[1])
    witness = np.clip(witness, written.lo, written.hi)
    ok, m64, m32 = certify_witness(str(onnx_src), graph, spec.C, witness, written.lo, written.hi)
    if not ok:
        shutil.rmtree(d)
        return None
    (d / "witness.json").write_text(json.dumps({"x": witness.tolist(), "margin_f64": m64, "margin_f32": m32}))
    xt = torch.tensor(witness, dtype=DTYPE).reshape((1,) + graph.input_shape)
    viol = [spec.disjunct_classes[i] for i, v in enumerate((graph.forward(xt) @ torch.tensor(spec.C, dtype=DTYPE).T)[0]) if float(v) < 0]
    gt = {
        "label": "sat",
        "certificate": "witness (float64 and onnxruntime float32 margins < 0)",
        "witness": "instances/%s/witness.json" % iid,
        "witness_margin_f64": m64,
        "witness_margin_f32": m32,
        "violated_classes": viol,
        "last_disjunct_class": spec.disjunct_classes[-1],
        "base_instance": base["id"],
        "eps_scale": eps_scale,
        "threshold_bracket": list(bracket),
    }
    meta = {
        "id": iid, "construction": base["construction"], "seed": base.get("seed"),
        "variant": f"sat_{suffix}",
        "paths": {"onnx": f"instances/{iid}/model.onnx", "vnnlib": f"instances/{iid}/spec.vnnlib",
                  "meta": f"instances/{iid}/meta.json"},
        "sha256": {"onnx": _sha(d / "model.onnx"), "vnnlib": _sha(d / "spec.vnnlib")},
        "args": {**base.get("args", {}), "epsilon_scale_vs_base": eps_scale},
        "ground_truth": gt,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def random_cnn_sat(bench: Path, n: int, budget: float, seed0: int = 0) -> List[Dict]:
    """Witness-only SAT instances on small random CNNs (no analytic UNSAT counterpart).

    The constructed CNN families are robust at every radius we can search (paired-bias) or up to
    enormous radii (contractive), so CNN counterexamples come from random conv nets instead.
    """
    import torch.nn as nn
    from VeriStressGT.utils.onnx_export import export_pytorch_to_onnx

    # Two sizes: "rcnn" (200 ReLUs) and "tcnn" (tiny, 32 ReLUs) whose near-threshold twins stay
    # within reach of exact methods, so they separate "cannot decide" from "decides wrongly".
    archs = {
        "rcnn": ((2, 5, 5), lambda: nn.Sequential(nn.Conv2d(2, 4, 3, padding=1), nn.ReLU(),
                                                  nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(),
                                                  nn.Flatten(), nn.Linear(4 * 5 * 5, 5)),
                 "conv(2->4,3x3)-relu-conv(4->4,3x3)-relu-fc(100->5)"),
        "tcnn": ((1, 4, 4), lambda: nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.ReLU(),
                                                  nn.Conv2d(2, 2, 3, stride=2, padding=1), nn.ReLU(),
                                                  nn.Flatten(), nn.Linear(2 * 2 * 2, 4)),
                 "conv(1->2,3x3)-relu-conv(2->2,3x3,s2)-relu-fc(8->4)"),
    }
    tmp = bench / "_tmp_rcnn"
    tmp.mkdir(parents=True, exist_ok=True)
    out: List[Dict] = []
    for (fam, (in_shape, make, arch_desc)), i in itertools.product(archs.items(), range(n)):
        seed = seed0 + i
        torch.manual_seed(seed)
        net = make()
        with torch.no_grad():
            for m in net:
                if isinstance(m, nn.Conv2d):
                    m.bias.uniform_(-0.5, 0.5)
        x0 = np.random.default_rng(seed).uniform(-1, 1, size=(1,) + in_shape).astype(np.float32)
        with torch.no_grad():
            y = net(torch.tensor(x0))[0]
        label, n_out = int(torch.argmax(y)), int(y.numel())
        onnx_p = tmp / f"{fam}_{seed}.onnx"
        export_pytorch_to_onnx(net, str(onnx_p), (1,) + in_shape, example_input=x0)
        eps0 = 0.02
        for _ in range(8):  # shrink until the base radius is not already SAT
            make_box_vnnlib(center=x0.reshape(-1).astype(np.float64), eps=eps0, out=str(tmp / "base.vnnlib"),
                            num_outputs=n_out, label=label)
            graph, spec = load_problem(str(onnx_p), str(tmp / "base.vnnlib"))
            if _status(graph, spec.C, spec.center, spec.radius, budget)["status"] != "sat":
                break
            eps0 /= 4
        found = find_threshold(graph, spec.C, spec.center, spec.radius, budget=budget, bisections=10)
        if found is None:
            continue
        k_lo, k_hi, hit = found
        base = {"id": f"{fam}_{seed}", "construction": "cnn.random_witness", "seed": seed,
                "args": {"arch": arch_desc, "epsilon_base": eps0}}
        print(f"[twin] {base['id']}: threshold scale in ({k_lo:.5g}, {k_hi:.5g}]", flush=True)
        near = _write_twin(bench, base, "near", k_hi, graph, spec, onnx_p, np.asarray(hit["witness"]), (k_lo, k_hi), budget)
        far_hit = _status(graph, spec.C, spec.center, spec.radius * 2 * k_hi, budget)
        far = None
        if far_hit["status"] == "sat":
            far = _write_twin(bench, base, "far", 2 * k_hi, graph, spec, onnx_p, np.asarray(far_hit["witness"]), (k_lo, k_hi), budget)
        out += [t for t in (near, far) if t is not None]
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def label_bench(bench: Path, twin_prefixes: List[str], budget: float, n_random_cnn: int = 0) -> Dict:
    manifest = json.loads((bench / "manifest.json").read_text())
    base_instances = [i for i in manifest["instances"] if "_sat_" not in i["id"]]
    for stale in (bench / "instances").glob("*_sat_*"):   # regenerate twins from scratch
        shutil.rmtree(stale)
    for inst in base_instances:
        inst["ground_truth"] = {"label": "unsat",
                                "certificate": ANALYTIC_CERT.get(inst["construction"], "analytic")}
        meta_p = bench / inst["paths"]["meta"]
        meta = json.loads(meta_p.read_text())
        meta["ground_truth"] = inst["ground_truth"]
        meta_p.write_text(json.dumps(meta, indent=2))

    twins: List[Dict] = []
    for inst in base_instances:
        if not any(inst["id"].startswith(p) for p in twin_prefixes):
            continue
        onnx_p = bench / inst["paths"]["onnx"]
        try:
            graph, spec = load_problem(str(onnx_p), str(bench / inst["paths"]["vnnlib"]))
        except UnsupportedOp as e:
            print(f"[twin] {inst['id']}: skip ({e})", flush=True)
            continue
        found = find_threshold(graph, spec.C, spec.center, spec.radius, budget=budget)
        if found is None:
            print(f"[twin] {inst['id']}: no counterexample up to k_max -- no SAT twin", flush=True)
            continue
        k_lo, k_hi, hit = found
        print(f"[twin] {inst['id']}: threshold scale in ({k_lo:.4g}, {k_hi:.4g}]", flush=True)
        near = _write_twin(bench, inst, "near", k_hi, graph, spec, onnx_p, np.asarray(hit["witness"]),
                           (k_lo, k_hi), budget)
        far_hit = _status(graph, spec.C, spec.center, spec.radius * 2 * k_hi, budget)
        far = None
        if far_hit["status"] == "sat":
            far = _write_twin(bench, inst, "far", 2 * k_hi, graph, spec, onnx_p,
                              np.asarray(far_hit["witness"]), (k_lo, k_hi), budget)
        for t in (near, far):
            if t is not None:
                twins.append(t)
                print(f"[twin]   wrote {t['id']} (witness margin f64={t['ground_truth']['witness_margin_f64']:.3e})", flush=True)

    if n_random_cnn:
        twins += random_cnn_sat(bench, n_random_cnn, budget)

    manifest["instances"] = base_instances + [
        {k: t[k] for k in ("id", "construction", "seed", "paths", "sha256", "args", "ground_truth", "variant")}
        for t in twins
    ]
    (bench / "manifest.json").write_text(json.dumps(manifest, indent=2))
    n_sat = sum(1 for i in manifest["instances"] if i["ground_truth"]["label"] == "sat")
    print(f"[ground_truth] {len(manifest['instances'])} instances: {len(manifest['instances']) - n_sat} UNSAT, {n_sat} SAT")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--twins", nargs="*", default=["milp_", "corners_", "dc_cnn_"],
                    help="Instance-id prefixes that get SAT twins.")
    ap.add_argument("--budget", type=float, default=20.0, help="Reference-verifier budget per bracket step (s).")
    ap.add_argument("--random_cnn", type=int, default=0, help="Number of random-CNN witness-SAT networks to add.")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    label_bench(Path(args.bench), args.twins, args.budget, args.random_cnn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
