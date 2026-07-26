"""Generate the convex Prop-11 polynomial companion set (analytically-grounded GT).

Poly ONNX are gitignored (regenerable from args), so run this once before
`sound_recertify` to (re)create the companion model.onnx / spec.vnnlib under
poly_convex/instances/. Deterministic (fixed seeds).

Run: PYTHONPATH=src python rebuttal_materials/sound_arithmetic/gen_poly_convex.py
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

from VeriStressGT.registry.constructions import discover_constructions

OUT = Path(__file__).resolve().parent / "poly_convex" / "instances"
DEGREES = [2, 4, 6, 8, 10, 12]
SEEDS = [1, 2]


def main():
    c = discover_constructions()["polynomial.algebraic_boundary"]
    OUT.mkdir(parents=True, exist_ok=True)
    for deg in DEGREES:
        for seed in SEEDS:
            iid = f"polyconv_d{deg}_s{seed}"
            d = OUT / iid
            d.mkdir(exist_ok=True)
            p = argparse.ArgumentParser(); c.add_args(p); ns = p.parse_args([])
            ns.seed = seed; ns.degree = deg; ns.input_dim = 40; ns.hidden_dim = 40
            ns.certified_convex = True; ns.eps_normal = 0.02; ns.delta_prime = 0.005
            ns.onnx_path = str(d / "model.onnx"); ns.vnnlib_path = str(d / "spec.vnnlib")
            c.run(ns)
            json.dump({"id": iid, "construction": "polynomial.algebraic_boundary",
                       "args": {"degree": deg, "certified_convex": True, "seed": seed}},
                      open(d / "meta.json", "w"))
            print("wrote", iid)


if __name__ == "__main__":
    main()
