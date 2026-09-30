"""Commands the AIQ evaluation cards run as their pipeline node (veristressgt-thrust1, ...).

MAGNET resolves a card's `executable` against the directory `magnet evaluate` is launched from, so
`python aiq/thrust1_runner.py` only works from the repository root. These installed commands find
the checkout this package is installed from (`pip install -e .`, done by scripts/bootstrap.sh) and
run the runner there, so `magnet evaluate /path/to/card.yaml` works from any directory.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/VeriStressGT/cli/aiq_cards.py -> repo root


def _run(script: str) -> None:
    path = REPO_ROOT / "aiq" / script
    if not path.exists():
        raise SystemExit(f"{path} not found: install VeriStressGT from its git checkout with `pip install -e .`")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.stdout.reconfigure(line_buffering=True)
    sys.path.insert(0, str(path.parent))            # the runners import aiq/verifier_args.py
    sys.argv = [str(path)] + sys.argv[1:]
    runpy.run_path(str(path), run_name="__main__")


def thrust1() -> None:
    _run("thrust1_runner.py")


def thrust2() -> None:
    _run("thrust2_runner.py")


def mini_sweep() -> None:
    _run("mini_sweep_runner.py")
