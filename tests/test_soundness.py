"""Tests for the Thrust 1 soundness package (loader, bounds, ground truth, verifiers, scoring).

Run: PYTHONPATH=src python tests/test_soundness.py      (or: PYTHONPATH=src pytest tests/test_soundness.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from VeriStressGT.soundness.bounds import BoundOpts, spec_lower_bounds
from VeriStressGT.soundness.graph import DTYPE, UnsupportedOp, load_graph
from VeriStressGT.soundness.ground_truth import certify_witness
from VeriStressGT.soundness.refverify import load_problem, verify
from VeriStressGT.soundness.scoring import majority_labels, score

BENCH = Path(__file__).resolve().parents[1] / "thrust1_bench"


def _inst(iid: str):
    d = BENCH / "instances" / iid
    return str(d / "model.onnx"), str(d / "spec.vnnlib")


def test_loader_refuses_attention():
    try:
        load_graph(_inst("ld_01")[0])
    except UnsupportedOp:
        print("ok loader refuses bilinear attention")
        return
    raise AssertionError("attention model should be unsupported")


def test_bounds_are_sound():
    """CROWN lower bounds never exceed the margin at sampled points (MLP, residual-style, CNN)."""
    torch.manual_seed(0)
    for iid in ("milp_s0_b", "meap_02", "corners_03", "pb_cnn_02", "tcnn_1_sat_near"):
        graph, spec = load_problem(*_inst(iid))
        shape = (1,) + graph.input_shape
        c = torch.tensor(spec.center, dtype=DTYPE).reshape(shape)
        r = torch.tensor(spec.radius, dtype=DTYPE).reshape(shape)
        C = torch.tensor(spec.C, dtype=DTYPE)
        lb, _ = spec_lower_bounds(graph, C, c, r, BoundOpts())
        X = c + r * (2 * torch.rand((3000,) + graph.input_shape, dtype=DTYPE) - 1)
        X = torch.cat([X, c + r * torch.sign(torch.randn((1000,) + graph.input_shape, dtype=DTYPE))])
        m = graph.forward(X) @ C.T
        assert bool((m >= lb - 1e-9).all()), f"unsound bound on {iid}"
    print("ok CROWN bounds sound on 5 instances")


def test_every_sat_witness_certifies():
    """Every SAT label in the shipped benchmark is backed by a valid counterexample."""
    manifest = json.loads((BENCH / "manifest.json").read_text())
    n = 0
    for inst in manifest["instances"]:
        gt = inst["ground_truth"]
        if gt["label"] != "sat":
            continue
        onnx_p, vnn_p = _inst(inst["id"])
        graph, spec = load_problem(onnx_p, vnn_p)
        x = np.asarray(json.loads((BENCH / gt["witness"]).read_text())["x"])
        ok, m64, m32 = certify_witness(onnx_p, graph, spec.C, x, spec.lo, spec.hi)
        assert ok, f"{inst['id']}: witness does not certify (f64={m64}, f32={m32})"
        n += 1
    assert n > 0
    print(f"ok {n} SAT witnesses certify in float64 and float32")


def test_reference_and_a_mutant():
    assert verify(*_inst("milp_s0_a"), "none", 30)["status"] == "unsat"
    assert verify(*_inst("milp_s1_a_sat_far"), "none", 30)["status"] == "sat"
    # disjunct_last: just past the threshold only class 2 is violated; the last disjunct (class 3)
    # is provable, so reporting the last disjunct's status gives a false UNSAT
    assert verify(*_inst("milp_s1_a_sat_near"), "disjunct_last", 30)["status"] == "unsat"
    print("ok reference correct; disjunct_last mutant returns a false UNSAT")


def test_scoring():
    gt = {"a": "unsat", "b": "sat", "c": "sat"}
    verdicts = {
        "ref": {"a": "unsat", "b": "sat", "c": "timeout"},
        "real": {"a": "unsat", "b": "timeout", "c": "timeout"},
        "bug": {"a": "unsat", "b": "unsat", "c": "unsat"},   # false UNSAT on b and c
    }
    assert majority_labels(verdicts, ["ref", "bug"], ["a", "b", "c"]) == {"a": "unsat", "b": None, "c": "unsat"}
    res = score(verdicts, gt, planted={"bug": "false_unsat"}, controls=["ref"], others=["real"])
    s = res["summary"]
    assert s["ground_truth"]["detected"] == ["bug"] and s["ground_truth"]["controls_flagged"] == []
    # one-buggy pool: b is a 1-1 tie (unresolved), c is decided only by the buggy verifier -> missed
    assert s["majority_one_buggy"]["missed"] == ["bug"]
    print("ok scoring: ground truth flags the bug, majority vote misses it")


if __name__ == "__main__":
    torch.set_num_threads(1)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all soundness tests passed")
    sys.exit(0)
