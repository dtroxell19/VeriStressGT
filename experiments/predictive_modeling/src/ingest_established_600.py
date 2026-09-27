"""Fold the 600 s established re-run back into the canonical row table.

Reads server_reruns/est600/<bench>_<verifier>/results.jsonl (produced by
rerun_established_600.sh), overwrites the ESTABLISHED rows' raw_status / raw_time /
budget_s with the 600 s outcomes (profiles and size features are a-priori and
unchanged; synthetic rows are untouched), and writes
data/processed/instance_verifier_rows_600.parquet (+ .csv).

Then run the CV against it with:
  PM_ROWS=../data/processed/instance_verifier_rows_600.parquet \
    python run_all.py --repeats 20 --horizon 600 --n-jobs 50
"""
from __future__ import annotations
import json, glob, os
import numpy as np
import pandas as pd

import common as C

RERUN_DIR = C.PM_ROOT / "server_reruns" / "est600"


def _norm_status(rec) -> str:
    if rec.get("timed_out"):
        return "TIMEOUT"
    s = (rec.get("status") or rec.get("parse_result_status") or "").strip().upper()
    mapping = {
        "UNSAT": "UNSAT", "HOLDS": "UNSAT", "ROBUST": "UNSAT", "SAFE": "UNSAT",
        "SAT": "SAT", "VIOLATED": "SAT", "UNSAFE": "SAT", "COUNTEREXAMPLE": "SAT",
        "TIMEOUT": "TIMEOUT", "TIMED_OUT": "TIMEOUT",
        "ERROR": "ERROR", "UNKNOWN": "UNKNOWN",
    }
    return mapping.get(s, "UNKNOWN" if s else "ERROR")


def load_rerun(budget=600.0):
    """(benchmark, instance_id, verifier) -> (status, time, budget)."""
    out = {}
    files = glob.glob(str(RERUN_DIR / "*" / "results.jsonl"))
    if not files:
        raise FileNotFoundError(f"no results.jsonl under {RERUN_DIR} — run/rsync the server re-run first")
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            v = r.get("verifier")
            bench = r.get("benchmark")
            iid = str(r.get("instance_id"))
            if not (v and bench and iid):
                continue
            status = _norm_status(r)
            t = r.get("wall_time_s", r.get("wall_time", np.nan))
            t = float(t) if isinstance(t, (int, float)) else np.nan
            if status == "TIMEOUT" and not np.isfinite(t):
                t = budget
            out[(bench, iid, v)] = (status, t, budget)
    return out


def build(budget=600.0):
    df = C.load_rows().copy()
    rerun = load_rerun(budget)
    print(f"loaded {len(rerun)} re-run outcomes from {RERUN_DIR}")

    before = {v: df[(df.verifier == v) & (df.domain == "established")]["raw_status"].value_counts().to_dict()
              for v in C.VERIFIERS}

    n_updated = 0
    for i, row in df.iterrows():
        if row["domain"] != "established":
            continue
        key = (row["benchmark"], str(row["instance_id"]), row["verifier"])
        if key in rerun:
            st, t, b = rerun[key]
            df.at[i, "raw_status"] = st
            df.at[i, "raw_time"] = t
            df.at[i, "budget_s"] = b
            n_updated += 1
    print(f"updated {n_updated} established rows to the {int(budget)}s re-run")

    after = {v: df[(df.verifier == v) & (df.domain == "established")]["raw_status"].value_counts().to_dict()
             for v in C.VERIFIERS}
    print("\nestablished status BEFORE -> AFTER (per verifier):")
    for v in C.VERIFIERS:
        print(f"  {v:10} {before[v]}  ->  {after[v]}")

    out_pq = C.DATA_PROCESSED / "instance_verifier_rows_600.parquet"
    out_csv = C.DATA_PROCESSED / "instance_verifier_rows_600.csv"
    df.to_parquet(out_pq)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_pq}")
    print("Now: PM_ROWS=%s python run_all.py --repeats 20 --horizon 600 --n-jobs 50" % out_pq)
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=600.0)
    build(ap.parse_args().budget)
