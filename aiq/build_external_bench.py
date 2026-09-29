"""Build an external (non-VeriStressGT) benchmark in the Thrust 1 format, labelled only by what an
external benchmark can actually know.

VNN-COMP benchmarks ship no ground truth. The only labels available without our constructions are
counterexamples: an instance is known SAT if some attack finds an input in the box that violates the
spec. Robustness (UNSAT) can never be established this way. So each instance gets:

  * label "sat" + witness.json, if a strong PGD attack (5 seeds x 16 restarts x 150 steps) finds a
    counterexample that certifies in float64 AND onnxruntime float32 (same check as thrust1_bench);
  * label "unknown" otherwise.

    B=~/vnncomp2022_benchmarks/benchmarks    # github.com/ChristopherBrix/vnncomp2022_benchmarks
    python aiq/build_external_bench.py --source src/VeriStressGT/benchmarks/vnncomp_mnist_fc \
        --onnx_root $B/mnist_fc --out ext_mnist_fc_bench --per_network 10
    python aiq/build_external_bench.py --source src/VeriStressGT/benchmarks/oval21 \
        --onnx_root $B/oval21 --out ext_oval21_bench

The output runs through the Thrust 1 runner unchanged (aiq/thrust1_runner.py --bench_dir <out>).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from VeriStressGT.soundness.ground_truth import certify_witness  # noqa: E402
from VeriStressGT.soundness.graph import DTYPE, UnsupportedOp  # noqa: E402
from VeriStressGT.soundness.refverify import load_problem, pgd_attack  # noqa: E402

ATTACK_SEEDS = 5


def _onnx_rel(inst: dict, meta: dict) -> str:
    """Path of the instance's network inside the VNN-COMP benchmark dir (e.g. onnx/mnist-net_256x2.onnx)."""
    return (inst.get("vnncomp") or {}).get("onnx_rel") or meta.get("source_onnx") or ""


def _network_key(inst: dict) -> str:
    """The network an instance belongs to (for picking a balanced subset)."""
    v = inst.get("vnncomp") or {}
    return v.get("onnx_rel") or (inst.get("args") or {}).get("onnx_basename") or inst["id"]


def _attack(onnx_p: Path, vnnlib_p: Path):
    """Certified counterexample from PGD, or None."""
    try:
        graph, spec = load_problem(str(onnx_p), str(vnnlib_p))
    except UnsupportedOp:
        return None, "unsupported"
    lo = torch.tensor(spec.lo, dtype=DTYPE).reshape((1,) + graph.input_shape)
    hi = torch.tensor(spec.hi, dtype=DTYPE).reshape((1,) + graph.input_shape)
    C = torch.tensor(spec.C, dtype=DTYPE)
    for seed in range(ATTACK_SEEDS):
        x, margin = pgd_attack(graph, C, lo, hi, seed=seed)
        if margin <= 0:
            xs = x.detach().numpy().reshape(-1).astype(np.float64)
            ok, m64, m32 = certify_witness(str(onnx_p), graph, spec.C, xs, spec.lo, spec.hi)
            if ok:
                return {"x": xs.tolist(), "margin_f64": m64, "margin_f32": m32, "attack_seed": seed}, "sat"
    return None, "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="Ingested VNN-COMP benchmark dir (manifest.json).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--onnx_root", default=None,
                    help="VNN-COMP benchmark dir holding onnx/ (the ingested benchmarks do not commit .onnx).")
    ap.add_argument("--per_network", type=int, default=0,
                    help="Keep N evenly spaced instances of each network (0 = all).")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)

    src, out = Path(args.source), Path(args.out)
    manifest = json.loads((src / "manifest.json").read_text())
    by_net = defaultdict(list)
    for inst in manifest["instances"]:
        by_net[_network_key(inst)].append(inst)
    def pick(insts):
        if not args.per_network or len(insts) <= args.per_network:
            return insts
        return [insts[round(k * len(insts) / args.per_network)] for k in range(args.per_network)]
    chosen = [i for net in sorted(by_net) for i in pick(by_net[net])]

    if out.exists():
        shutil.rmtree(out)
    kept = []
    for inst in chosen:
        iid = inst["id"]
        d = out / "instances" / iid
        d.mkdir(parents=True)
        onnx_src = src / inst["paths"]["onnx"]
        if not onnx_src.exists():
            src_meta = json.loads((src / inst["paths"]["meta"]).read_text()) if inst["paths"].get("meta") else {}
            if not args.onnx_root:
                raise SystemExit(f"{onnx_src} missing: pass --onnx_root <VNN-COMP benchmark dir>")
            onnx_src = Path(args.onnx_root).expanduser() / _onnx_rel(inst, src_meta)
        shutil.copy(onnx_src, d / "model.onnx")
        shutil.copy(src / inst["paths"]["vnnlib"], d / "spec.vnnlib")
        witness, label = _attack(d / "model.onnx", d / "spec.vnnlib")
        gt = {"label": label if label == "sat" else "unknown",
              "certificate": "PGD counterexample (float64 and onnxruntime float32 margins < 0)"
              if label == "sat" else "none: an external benchmark cannot certify robustness"}
        if witness:
            (d / "witness.json").write_text(json.dumps(witness))
            gt["witness"] = f"instances/{iid}/witness.json"
        meta = {k: v for k, v in inst.items() if k not in ("paths", "sha256")}
        meta["network"] = _network_key(inst)
        meta["ground_truth"] = gt
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        kept.append({"id": iid, "construction": inst.get("construction", "external"), "seed": inst.get("seed"),
                     "paths": {"onnx": f"instances/{iid}/model.onnx", "vnnlib": f"instances/{iid}/spec.vnnlib",
                               "meta": f"instances/{iid}/meta.json"},
                     "network": meta["network"], "ground_truth": gt})
        print(f"{iid:28s} {meta['network'][:40]:40s} {gt['label']}", flush=True)

    (out / "manifest.json").write_text(json.dumps(
        {"name": out.name, "source": str(src), "external": True, "instances": kept}, indent=2))
    n_sat = sum(k["ground_truth"]["label"] == "sat" for k in kept)
    print(f"{out}: {len(kept)} instances from {len(by_net)} networks, {n_sat} certified SAT, "
          f"{len(kept) - n_sat} unknown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
