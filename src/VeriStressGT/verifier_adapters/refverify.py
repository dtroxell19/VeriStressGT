"""Adapter for the in-house reference verifier and its planted-bug mutants (soundness evaluation).

    python -m VeriStressGT.cli.verify_benchmark --benchmark src/VeriStressGT/benchmarks/soundness_gt \
        --verifier refverify --out_dir runs/refverify_tol_unsat --timeout 60 --refverify_bug tol_unsat

``--refverify_bug none`` is the sound reference; see ``VeriStressGT.soundness.bugs`` for the rest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from .common import authoritative_status_from_text

VERIFIER_NAME = "refverify"

_REFVERIFY = Path(__file__).resolve().parents[1] / "soundness" / "refverify.py"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--refverify_bug", default="none", help="Planted bug to enable ('none' = sound reference).")
    p.add_argument("--refverify_budget", type=float, default=55.0,
                   help="Internal time budget (s); keep below verify_benchmark --timeout to get UNKNOWN, not TIMEOUT.")


def build_cmd(args: argparse.Namespace, onnx_path: str, vnnlib_path: str, workdir: Path) -> List[str]:
    # Run by path so this checkout's code is used even if another copy of the package is installed.
    return [sys.executable, str(_REFVERIFY), "--onnx", onnx_path, "--vnnlib", vnnlib_path,
            "--bug", args.refverify_bug, "--time_budget", str(args.refverify_budget)]


def parse_result(stdout: str, stderr: str, rc: int) -> str | None:
    return authoritative_status_from_text(stdout)
