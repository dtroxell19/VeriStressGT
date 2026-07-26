"""Numerical-soundness audit of VeriStress-GT constructors.

Rebuttal question (Reviewer WvAC Q5 / AC bullet 1): as constructed networks get
deep/extreme, can accumulated float32 rounding in the forward pass silently flip
an analytically-positive margin negative?

This script re-evaluates each constructed instance under float64 and compares it to
float32, at the nominal point and over the profiler's own "hard" stress points
(worst-case corner, PGD, boundary/face/corner samples). The exported ONNX graphs are
`*.onnx`-gitignored, so instances are regenerated in-memory from the exact args stored
in each `meta.json` (the driver is deterministic for all constructors except the
deep-contractive CNN, whose spectral-norm power-iteration is unseeded).

float64 forward: ONNX Runtime's CPU EP has no float64 Conv kernel and fuses MatMul into
a float32-only op, so we use `onnx.reference.ReferenceEvaluator` (pure-numpy, dtype-faithful,
no graph fusion) for BOTH precisions. This isolates dtype as the only variable; ORT-float32
is additionally reported as a shipped-engine cross-check.

Usage:
    PYTHONPATH=src python -m VeriStressGT.analysis.soundness_audit \
        --benchmarks benchmarks/sweep_all benchmarks/polynomial_stress_22 \
        --out-dir benchmarks/_soundness --n-samples 128
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto

FLOAT, DOUBLE = TensorProto.FLOAT, TensorProto.DOUBLE

# Cap (seconds) on the MILP constructor's Gurobi re-solve during audit regeneration.
_MILP_TIME_CAP = 8.0


# --------------------------------------------------------------------------- #
# float32 -> float64 ONNX conversion
# --------------------------------------------------------------------------- #
def onnx_to_float64(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return a deep copy of `model` with every float32 tensor promoted to float64."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph

    new_inits = []
    for init in g.initializer:
        if init.data_type == FLOAT:
            arr = numpy_helper.to_array(init).astype(np.float64)
            new_inits.append(numpy_helper.from_array(arr, init.name))
        else:
            new_inits.append(init)
    del g.initializer[:]
    g.initializer.extend(new_inits)

    for vi in list(g.input) + list(g.output) + list(g.value_info):
        tt = vi.type.tensor_type
        if tt.elem_type == FLOAT:
            tt.elem_type = DOUBLE

    for node in g.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == FLOAT:
                    arr = numpy_helper.to_array(attr.t).astype(np.float64)
                    attr.t.CopyFrom(numpy_helper.from_array(arr))
        elif node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == FLOAT:
                    attr.i = DOUBLE
    return m


# --------------------------------------------------------------------------- #
# Forward evaluation (pure-numpy, dtype-faithful)
# --------------------------------------------------------------------------- #
def _input_shape(model: onnx.ModelProto) -> List[int]:
    dims = model.graph.input[0].type.tensor_type.shape.dim
    return [d.dim_value if d.dim_value > 0 else 1 for d in dims]


class _Evaluator:
    """Wraps onnx.reference.ReferenceEvaluator for one model at one dtype."""

    def __init__(self, onnx_path: str, dtype: str):
        from onnx.reference import ReferenceEvaluator

        model = onnx.load(onnx_path)
        self.dtype = np.float64 if dtype == "fp64" else np.float32
        if dtype == "fp64":
            model = onnx_to_float64(model)
        self.base_shape = _input_shape(model)
        self.iname = model.graph.input[0].name
        self.sess = ReferenceEvaluator(model)

    def logits(self, X: np.ndarray) -> np.ndarray:
        """X: (n, d) -> (n, C). Loops rows (fixed batch dim in most exported graphs)."""
        X = np.asarray(X, dtype=self.dtype)
        if X.ndim == 1:
            X = X[None, :]
        outs = []
        for row in X:
            xr = row.reshape(self.base_shape).astype(self.dtype)
            out = self.sess.run(None, {self.iname: xr})[0]
            outs.append(np.asarray(out).reshape(-1))
        return np.stack(outs, axis=0)


def margins(logits: np.ndarray, label: int) -> np.ndarray:
    """Per-row min-margin f_label - max_{k != label} f_k."""
    logits = np.asarray(logits, dtype=np.float64)
    y = logits[:, label]
    other = logits.copy()
    other[:, label] = -np.inf
    return y - np.max(other, axis=1)


# --------------------------------------------------------------------------- #
# Instance regeneration + stress sampling
# --------------------------------------------------------------------------- #
def regenerate(meta: Dict[str, Any], out_dir: str):
    """Re-run the constructor from stored args; returns (onnx_path, vnnlib_path, ret)."""
    from VeriStressGT.registry.constructions import discover_constructions

    global _CONS
    try:
        _CONS
    except NameError:
        _CONS = discover_constructions()

    args = dict(meta["args"])
    onnx_path = os.path.join(out_dir, "model.onnx")
    vnnlib_path = os.path.join(out_dir, "spec.vnnlib")
    args["onnx_path"] = onnx_path
    args["vnnlib_path"] = vnnlib_path
    # The MILP constructor re-solves a Gurobi MILP only to set epsilon = eps_frac * r*.
    # The network weights are seed-deterministic and independent of the solve; for a
    # dtype-gap audit on a shallow MLP the precise r* is immaterial, so cap the solve
    # time to keep near-boundary (eps_frac->1) instances from stalling for minutes each.
    if "time_limit" in args and args.get("time_limit", 0) and float(args["time_limit"]) > _MILP_TIME_CAP:
        args["time_limit"] = _MILP_TIME_CAP
    ns = argparse.Namespace(**args)
    ret = _CONS[meta["construction"]].run(ns)
    return onnx_path, vnnlib_path, (ret if isinstance(ret, dict) else {})


def stress_points(onnx_path: str, vnnlib_path: str, n: int) -> Tuple[np.ndarray, int, float]:
    """Reuse the profiler's stress sampler; returns (X (m,d), label, epsilon). Row 0 is x0."""
    import torch
    from VeriStressGT.difficulty_profile.instance_loader import load_instance
    from VeriStressGT.difficulty_profile.components import _sample_stress_points

    inst = load_instance(onnx_path, vnnlib_path, device="cpu")
    x0 = inst.x0.detach().reshape(1, -1)
    try:
        Xs = _sample_stress_points(inst, n, n_pgd_steps=20)
        X = torch.cat([x0, Xs.detach()], dim=0)
    except Exception:
        X = x0
    return X.cpu().numpy(), int(inst.true_class), float(inst.epsilon)


# --------------------------------------------------------------------------- #
# Per-instance audit
# --------------------------------------------------------------------------- #
def load_profile_reference(bench_dir: Path) -> Dict[str, Dict[str, float]]:
    """instance_id -> {margin_nominal, margin_sample_min, ibp_margin_lb} from difficulty_profiles.json."""
    pf = bench_dir / "difficulty_profiles.json"
    ref: Dict[str, Dict[str, float]] = {}
    if not pf.exists():
        return ref
    data = json.loads(pf.read_text())
    for rec in data.get("instances", []):
        iid = rec.get("instance_id") or rec.get("id")
        if iid is None:
            continue
        ref[str(iid)] = {
            k: rec.get(k) for k in ("margin_nominal", "margin_sample_min", "ibp_margin_lb")
        }
    return ref


def audit_instance(meta_path: Path, bench_dir: Path, prof_ref: Dict[str, Dict[str, float]],
                   n_samples: int) -> Dict[str, Any]:
    meta = json.loads(meta_path.read_text())
    iid = meta["id"]
    rec: Dict[str, Any] = {
        "instance_id": iid,
        "construction": meta["construction"],
        "benchmark": bench_dir.name,
    }
    with tempfile.TemporaryDirectory(prefix=f"audit_{iid}_") as tmp:
        onnx_path, vnnlib_path, ret = regenerate(meta, tmp)
        rec["claimed_margin"] = ret.get("gt_margin_lower_bound", ret.get("margin"))
        # Prefer the COMMITTED spec.vnnlib for x0/epsilon/box: network weights are
        # seed-deterministic and independent of any constructor solve, so the shipped box
        # (e.g. MILP's true eps from the original full Gurobi solve) is the faithful one and
        # avoids depending on the time-capped regeneration solve.
        committed_vnnlib = meta_path.parent / "spec.vnnlib"
        box_vnnlib = str(committed_vnnlib) if committed_vnnlib.exists() else vnnlib_path
        rec["used_committed_vnnlib"] = committed_vnnlib.exists()
        X, label, eps = stress_points(onnx_path, box_vnnlib, n_samples)
        rec["label"] = label
        rec["epsilon"] = eps
        rec["n_points"] = int(X.shape[0])

        ev32 = _Evaluator(onnx_path, "fp32")
        ev64 = _Evaluator(onnx_path, "fp64")
        m32 = margins(ev32.logits(X), label)
        m64 = margins(ev64.logits(X), label)

        # shipped-engine cross-check on x0 (ORT float32 vs reference float32)
        rec["ort_ref_agree_x0"] = _ort_ref_agreement(onnx_path, X[0], label, float(m32[0]))

    rec["margin_x0_fp32"] = float(m32[0])
    rec["margin_x0_fp64"] = float(m64[0])
    rec["min_margin_fp32"] = float(np.min(m32))
    rec["min_margin_fp64"] = float(np.min(m64))
    rec["max_abs_delta"] = float(np.max(np.abs(m32 - m64)))
    ref = prof_ref.get(iid, {})
    rec["profile_margin_nominal"] = ref.get("margin_nominal")
    rec["profile_margin_sample_min"] = ref.get("margin_sample_min")
    denom = abs(rec["min_margin_fp64"]) if abs(rec["min_margin_fp64"]) > 0 else 1.0
    rec["max_rel_delta"] = rec["max_abs_delta"] / denom
    rec["sign_flip"] = bool(np.any((m32 > 0) != (m64 > 0)))
    rec["all_positive_fp32"] = bool(np.all(m32 > 0))
    rec["all_positive_fp64"] = bool(np.all(m64 > 0))
    return rec


def _ort_ref_agreement(onnx_path: str, x0: np.ndarray, label: int, ref_fp32_margin: float) -> Optional[float]:
    """|reference-fp32 margin - ORT-fp32 margin| at x0 (engine cross-check). None if ORT unavailable."""
    try:
        import onnxruntime as ort

        model = onnx.load(onnx_path)
        shape = _input_shape(model)
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        out = sess.run(None, {iname: x0.astype(np.float32).reshape(shape)})[0]
        m = margins(np.asarray(out).reshape(1, -1), label)[0]
        return float(abs(ref_fp32_margin - m))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", nargs="+", required=True,
                    help="Benchmark dirs each containing instances/*/meta.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="Audit only the first N instances (smoke test).")
    ap.add_argument("--only", nargs="*", default=None, help="Restrict to these instance ids.")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    n_done = 0
    for bench in args.benchmarks:
        bench_dir = Path(bench)
        prof_ref = load_profile_reference(bench_dir)
        metas = sorted((bench_dir / "instances").glob("*/meta.json"))
        for mp in metas:
            iid = json.loads(mp.read_text())["id"]
            if args.only and iid not in args.only:
                continue
            if args.limit is not None and n_done >= args.limit:
                break
            try:
                rec = audit_instance(mp, bench_dir, prof_ref, args.n_samples)
                records.append(rec)
                flag = "  SIGN-FLIP!" if rec["sign_flip"] else ""
                print(f"[{rec['benchmark']}/{iid}] "
                      f"m_x0 fp32={rec['margin_x0_fp32']:.3e} fp64={rec['margin_x0_fp64']:.3e} "
                      f"min fp64={rec['min_margin_fp64']:.3e} maxΔ={rec['max_abs_delta']:.2e}{flag}")
            except Exception as e:
                errors.append({"instance": iid, "error": repr(e), "traceback": traceback.format_exc()})
                print(f"[{bench_dir.name}/{iid}] ERROR: {e!r}")
            n_done += 1

    (out_dir / "soundness_audit.json").write_text(json.dumps(records, indent=2))
    _write_csv(records, out_dir / "soundness_audit.csv")
    if errors:
        (out_dir / "soundness_audit_errors.json").write_text(json.dumps(errors, indent=2))

    _print_summary(records, errors)
    return 0


def _write_csv(records: List[Dict[str, Any]], path: Path) -> None:
    if not records:
        path.write_text("")
        return
    cols = list(records[0].keys())
    lines = [",".join(cols)]
    for r in records:
        lines.append(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols))
    path.write_text("\n".join(lines))


def _print_summary(records: List[Dict[str, Any]], errors: List[Dict[str, str]]) -> None:
    print("\n" + "=" * 70)
    print(f"SOUNDNESS AUDIT SUMMARY: {len(records)} instances, {len(errors)} errors")
    if not records:
        return
    n_flip = sum(1 for r in records if r["sign_flip"])
    n_pos64 = sum(1 for r in records if r["all_positive_fp64"])
    max_absd = max(r["max_abs_delta"] for r in records)
    max_reld = max(r["max_rel_delta"] for r in records)
    worst = max(records, key=lambda r: r["max_rel_delta"])
    print(f"  sign flips (fp32 vs fp64)        : {n_flip} / {len(records)}")
    print(f"  all-samples-positive under fp64  : {n_pos64} / {len(records)}")
    print(f"  max abs fp32-fp64 margin gap     : {max_absd:.3e}")
    print(f"  max relative gap (|Δ|/|margin|)  : {max_reld:.3e}")
    print(f"  worst relative-gap instance      : {worst['benchmark']}/{worst['instance_id']} "
          f"({worst['construction']}) rel={worst['max_rel_delta']:.2e}")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
