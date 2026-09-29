"""Shared per-verifier command-line extras for the AIQ runners (Thrust 1 and Thrust 2)."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent

PYRAT_BASE = ["--pyrat_domains", "con_z", "--pyrat_device", "cpu", "--pyrat_library", "torch",
              "--pyrat_split_relu", "--no-pyrat_split", "--pyrat_split_heuristic", "better"]


def conda_env_python(env: str) -> str:
    """Absolute path of a conda env's python (falls back to plain 'python' if not found)."""
    base = None
    try:
        base = subprocess.run(["conda", "info", "--base"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
        if exe:
            base = str(Path(exe).resolve().parents[1])
    if base:
        py = Path(base) / "envs" / env / "bin" / "python"
        if py.exists():
            return str(py)
    return "python"


def verifier_extra(abcrown_config: Path, *, falsify: bool) -> Dict[str, List[str]]:
    """Extra args per verifier. ``falsify`` turns on pyrat's counterexample search (Thrust 1)."""
    return {
        "abcrown": ["--abcrown_config", str(abcrown_config)],
        # --check/--attack enable pyrat's counterexample search (off by default); must come last
        "pyrat": PYRAT_BASE + (["--pyrat_extra", "--check", "both", "--attack", "pgd"] if falsify else []),
        "neuralsat": ["--neuralsat_python",
                      conda_env_python(os.environ.get("NEURALSAT_CONDA_ENV", "neuralsat"))],
        "marabou": ["--marabou_bin", str(REPO_ROOT / "scripts" / "marabou_env.sh")],
    }
