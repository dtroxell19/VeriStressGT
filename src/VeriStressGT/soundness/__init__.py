"""Thrust 1: soundness evaluation with ground-truth labels.

- ``graph``:        strict ONNX -> piecewise-linear DAG (refuses ops it cannot model exactly)
- ``spec``:         VNNLIB box + disjunctive output-constraint parser
- ``bounds``:       IBP and CROWN-style linear bound propagation on the DAG
- ``refverify``:    in-house reference verifier (attack + bound-based BaB) with injectable bugs
- ``bugs``:         catalogue of planted bugs (algorithmic mutants and output-level wrappers)
- ``ground_truth``: witness-certified SAT instance generation
- ``scoring``:      ground-truth vs majority-vote bug detection
"""
