"""Shared paths / constants / canonical mapping for the Difficulty-Profile
sensitivity study.

All paths resolve relative to the repo root (never hardcode a home directory).
Analysis-layer helpers (leak-free transforms, the frozen 1725-row table) are
reused from the predictive-modeling module so the correlation matrix and the
unified Difficulty Index are computed on exactly the same numbers reported there.
"""
from __future__ import annotations
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (repo-root-relative)
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[3]
PS_ROOT = REPO_ROOT / "rebuttal_materials" / "profile_sensitivity"
CONFIG_PATH = PS_ROOT / "config" / "experiment.json"
RESULTS = PS_ROOT / "results"
PLOTS = PS_ROOT / "plots"
LOGS = PS_ROOT / "logs"

MERGED_RECORDS = REPO_ROOT / "rebuttal_materials" / "merged_records_constructed_and_real.json"
BENCH_ROOT = REPO_ROOT / "src" / "VeriStressGT" / "benchmarks"

# make the core profiler importable
CORE_SRC = REPO_ROOT / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

# Reuse the predictive-modeling analysis layer (TRANSFORM, load_rows, ...) WITHOUT
# a module-name clash: both packages have a `common.py`, so load PM's by file path
# under a distinct name.
PM_SRC = REPO_ROOT / "rebuttal_materials" / "predictive_modeling" / "src"


def _load_pm_common():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pm_common", str(PM_SRC / "common.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pm = _load_pm_common()


def load_rows():
    """The frozen 1725-row canonical table (delegates to predictive_modeling)."""
    return pm.load_rows()


def apply_transforms(df, cols):
    return pm.apply_transforms(df, cols)


TRANSFORM = pm.TRANSFORM
PROFILE_FEATURES = pm.PROFILE_FEATURES
POS_STATUS = pm.POS_STATUS
EXCLUDE_STATUSES = pm.EXCLUDE_STATUSES

# --------------------------------------------------------------------------- #
# Canonical 5-component mapping: profile-dict field -> canonical name.
# (Matches predictive_modeling PROFILE_MAP semantics; keys are the fields that
# DifficultyProfile.to_dict() emits.)
# --------------------------------------------------------------------------- #
CANON5 = {
    "margin_sample_min": "margin_hat_min",
    "ibp_relative_gap": "g_ibp",
    "unstable_frac": "unstable_fraction",
    "A_tau_local_log": "a_tau",
    "effective_grad_dim_mean": "d_eff",
}
COMPONENTS = ["margin_hat_min", "g_ibp", "unstable_fraction", "a_tau", "d_eff"]

# components that need the IBP pass (an ONNX graph); N/A for reconstructed poly.
IBP_COMPONENTS = {"g_ibp", "unstable_fraction"}

VERIFIERS = ["abcrown", "neuralsat", "marabou", "nnenum", "pyrat"]
SYNTHETIC_BENCHES = {"sweep_all", "polynomial_stress_22"}
ESTABLISHED_BENCHES = {"vnncomp_mnist_fc", "oval21"}


def domain_of(benchmark: str) -> str:
    return "established" if benchmark in ESTABLISHED_BENCHES else "synthetic"


def extract_components(profile_dict) -> dict:
    """Pull the canonical 5 components out of a DifficultyProfile.to_dict()."""
    return {canon: profile_dict.get(field) for field, canon in CANON5.items()}


def instance_paths(benchmark: str, instance_id: str):
    """Return (onnx_rel, vnnlib_rel): REPO_ROOT-relative paths (never absolute, so no
    username leaks into committed CSVs), or ("" , vnnlib_rel) when the benchmark has no
    exported ONNX (polynomial_stress_22). Resolve with ``resolve_path``."""
    d = BENCH_ROOT / benchmark / "instances" / instance_id
    onnx = d / "model.onnx"
    vnnlib = d / "spec.vnnlib"
    rel = lambda p: str(p.relative_to(REPO_ROOT)) if p.exists() else ""
    return rel(onnx), rel(vnnlib)


def resolve_path(rel: str):
    """Absolute path from a REPO_ROOT-relative path string ('' -> None)."""
    if not rel or (isinstance(rel, float)):
        return None
    return str(REPO_ROOT / rel)


def load_config() -> dict:
    import json
    with open(CONFIG_PATH) as f:
        return json.load(f)


def ensure_dirs():
    for p in (RESULTS, PLOTS, LOGS, RESULTS / "correlation"):
        p.mkdir(parents=True, exist_ok=True)
