from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import re

from .common import (
    normalize_status_from_text,
    authoritative_status_from_text,
    _TRACEBACK_RE,
)

VERIFIER_NAME = "abcrown"
CONDA_ENV_VAR = "ABCROWN_CONDA_ENV"
CONDA_ENV_DEFAULT = "alpha-beta-crown"

def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--abcrown_config",
        required=True,
        help="Path to ABCROWN yaml config (passed to VeriStressGT-verify --config_path).",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for alpha-beta-CROWN (default: cpu).",
    )


def build_cmd(args: argparse.Namespace, onnx_path: str, vnnlib_path: str, workdir: Path) -> List[str]:
    cfg = str(Path(args.abcrown_config).expanduser().resolve())
    return [
        "VeriStressGT-verify",
        "--method", "abcrown-vnncomp2024",
        "--onnx", onnx_path,
        "--vnnlib", vnnlib_path,
        "--config_path", cfg,
        "--device", args.device,
        # Hardcode a large value so abcrown's internal BaB timer never fires.
        # The subprocess wall-clock kill in verify_benchmark (--timeout) is
        # the only control point, and it correctly receives --timeout flag
        "--timeout", "999999",
    ]


def parse_result(stdout: str, stderr: str, rc: int) -> str | None:
    # 1) Trust ABCROWN's authoritative "Result: <token>" line (stdout beats stderr).
    #    This still beats a nonzero conda rc (abcrown prints "Result: timeout" then exits nonzero).
    auth = authoritative_status_from_text(stdout) or authoritative_status_from_text(stderr)
    if auth is not None:
        return auth

    # 2) No authoritative verdict AND the run crashed (nonzero exit or a Python traceback)
    #    -> report ERROR, never a loose keyword SAT. A crash is not a counterexample; treating
    #    it as SAT would fabricate a soundness failure on a provably-robust instance.
    combined = (stdout or "") + "\n" + (stderr or "")
    if rc != 0 or _TRACEBACK_RE.search(combined):
        return None  # finalize_status -> ERROR (rc != 0)

    # 3) Clean exit but no explicit Result line -> conservative loose fallback (rarely needed).
    return normalize_status_from_text(stdout) or normalize_status_from_text(stderr)