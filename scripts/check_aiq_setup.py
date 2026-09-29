#!/usr/bin/env python3
"""Check that the environment can run the AIQ evaluation cards (run from the repo root, in the
VeriStressGT env):

    python scripts/check_aiq_setup.py

Checks MAGNET and gurobipy, then runs each real verifier on one UNSAT and one SAT instance of
thrust1_bench exactly as the Thrust 1 card does, and prints the verdicts next to the labels.
Exit code 0 only if every verifier returns both labels correctly.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "aiq"))
from verifier_args import conda_env_python, verifier_env_defaults, verifier_extra  # noqa: E402

VERIFIERS = ["abcrown", "pyrat", "nnenum", "neuralsat", "marabou"]
PROBE = {"milp_s3_a": "unsat", "milp_s3_a_sat_far": "sat"}   # all five verifiers decide both


def _version(py: str, dist: str, fallback_module: str = "") -> str:
    code = ("import importlib.metadata as m\n"
            f"try: print(m.version({dist!r}))\n"
            "except Exception:\n"
            f"    import importlib.util, pathlib; s = importlib.util.find_spec({fallback_module or dist!r})\n"
            "    f = pathlib.Path(s.origin).parent / 'SOURCE_VERSION' if s else None\n"
            "    print(f.read_text().strip() + ' (source build)' if f and f.exists() else 'unknown')")
    try:
        out = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=120)
        return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "not installed"
    except Exception as e:  # noqa: BLE001
        return f"error ({e})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verifiers", nargs="*", default=VERIFIERS)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    ok = True
    print("== Python packages (this env) ==")
    for dist in ("aiq-magnet", "gurobipy", "onnxruntime", "torch"):
        v = _version(sys.executable, dist)
        print(f"  {dist:12s} {v}")
        if dist == "aiq-magnet" and v == "not installed":
            ok = False
    print("  (gurobipy is only needed to rebuild or scale the benchmark)")

    env = {**os.environ, **verifier_env_defaults(), "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": str(REPO_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    abc_dir = Path(env["ABCROWN_VNNCOMP2024_DIR"])
    if not (abc_dir / "complete_verifier").exists():
        print(f"  ! alpha-beta-CROWN submodule missing at {abc_dir}")

    marabou_py = os.environ.get("MARABOU_PYTHON") or conda_env_python(os.environ.get("MARABOU_CONDA_ENV", "marabou"))
    mv = _version(marabou_py, "maraboupy")
    print(f"  maraboupy    {mv}  [{marabou_py}]")
    if not mv.startswith("2."):
        print("  ! Marabou should be 2.0.0 (the version the recorded results use); see AIQ_README.md")

    bench = REPO_ROOT / "thrust1_bench"
    extra = verifier_extra(REPO_ROOT / "src/VeriStressGT/configs/abcrown_thrust1.yaml", falsify=True)
    print(f"\n== Real verifiers on {list(PROBE)} (timeout {args.timeout:.0f}s) ==")
    with tempfile.TemporaryDirectory() as tmp:
        for v in args.verifiers:
            out = Path(tmp) / v
            cmd = [sys.executable, "-m", "VeriStressGT.cli.verify_benchmark", "--benchmark", str(bench),
                   "--verifier", v, "--out_dir", str(out), "--timeout", str(args.timeout), "--overwrite",
                   "--instances", *PROBE, *extra.get(v, [])]
            subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True)
            got = {}
            res = out / "results.jsonl"
            if res.exists():
                for line in res.read_text().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        got[r["instance_id"]] = str(r.get("status", "?")).lower()
            good = all(got.get(i) == lab for i, lab in PROBE.items())
            ok &= good
            shown = ", ".join(f"{i}={got.get(i, 'missing')}" for i in PROBE)
            print(f"  {'OK  ' if good else 'FAIL'} {v:10s} {shown}")
            if not good:
                logs = sorted((out / "logs").glob("*.stderr.txt")) if (out / "logs").exists() else []
                tail = logs[0].read_text().strip().splitlines()[-3:] if logs else ["(no logs)"]
                for t in tail:
                    print(f"         {t[:150]}")
    print("\nAll checks passed." if ok else "\nSome checks failed; see AIQ_README.md (Setup).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
