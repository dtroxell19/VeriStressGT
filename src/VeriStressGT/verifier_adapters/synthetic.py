"""Adapter for the in-house reference verifier and its planted-bug mutants (Thrust 1).

    python -m VeriStressGT.cli.verify_benchmark --benchmark thrust1_bench --verifier synthetic \
        --out_dir runs/synthetic_tol_unsat --timeout 60 --synthetic_bug tol_unsat

``--synthetic_bug none`` is the sound reference; see ``VeriStressGT.soundness.bugs`` for the rest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from .common import authoritative_status_from_text

VERIFIER_NAME = "synthetic"

_REFVERIFY = Path(__file__).resolve().parents[1] / "soundness" / "refverify.py"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--synthetic_bug", default="none", help="Planted bug to enable ('none' = sound reference).")
    p.add_argument("--synthetic_budget", type=float, default=55.0,
                   help="Internal time budget (s); keep below verify_benchmark --timeout to get UNKNOWN, not TIMEOUT.")


def build_cmd(args: argparse.Namespace, onnx_path: str, vnnlib_path: str, workdir: Path) -> List[str]:
    # Run by path so this checkout's code is used even if another copy of the package is installed.
    return [sys.executable, str(_REFVERIFY), "--onnx", onnx_path, "--vnnlib", vnnlib_path,
            "--bug", args.synthetic_bug, "--time_budget", str(args.synthetic_budget)]


def parse_result(stdout: str, stderr: str, rc: int) -> str | None:
    return authoritative_status_from_text(stdout)
