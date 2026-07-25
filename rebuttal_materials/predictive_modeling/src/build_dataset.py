"""Audit inputs, build the canonical (instance x verifier) row table, recover
network groups, infer per-(verifier,benchmark) timeout budgets, and freeze a
hashed analysis manifest.

Run:  python build_dataset.py
Outputs under rebuttal_materials/predictive_modeling/{data,results}/.
"""
from __future__ import annotations
import json
import collections
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
from extract_network_features import features_from_onnx, features_from_poly_args


def _local_onnx(benchmark: str, instance_id: str) -> Path:
    return C.BENCH_ROOT / benchmark / "instances" / instance_id / "model.onnx"


def build():
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    C.DATA_MANIFESTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "audit").mkdir(parents=True, exist_ok=True)

    records = json.load(open(C.MERGED_RECORDS))
    source_manifest = {
        "merged_records": {
            "path": str(C.MERGED_RECORDS.relative_to(C.REPO_ROOT)),
            "sha256": C.sha256_of(C.MERGED_RECORDS),
            "n_records": len(records),
        },
        "onnx": {},
    }

    # ------------------------------------------------------------------ #
    # 1. per-instance network features (+ group id, +onnx hash)
    # ------------------------------------------------------------------ #
    inst_rows = []
    feat_cache = {}
    excluded = []  # (level, id, verifier, reason)
    for r in records:
        bench = r["benchmark"]
        iid = r["instance_id"]
        key = (bench, iid)
        if key in feat_cache:
            continue
        onnx_p = _local_onnx(bench, iid)
        if onnx_p.exists():
            feat = features_from_onnx(str(onnx_p))
            feat["feat_source"] = "onnx"
            source_manifest["onnx"][f"{bench}/{iid}"] = feat.get("_onnx_sha256")
        elif bench == "polynomial_stress_22":
            feat = features_from_poly_args(r.get("args", {}))
            feat["feat_source"] = "poly_args"
        else:
            feat = None
        feat_cache[key] = feat

    # ------------------------------------------------------------------ #
    # network group id: synthetic -> per instance; established -> onnx hash
    # ------------------------------------------------------------------ #
    def group_id(bench, iid, feat):
        if C.domain_of(bench) == "established" and feat and feat.get("_onnx_sha256"):
            return "net_" + feat["_onnx_sha256"][:16]
        return f"{bench}::{iid}"

    # ------------------------------------------------------------------ #
    # 2. infer per-(verifier,benchmark) timeout budget from TIMEOUT rows
    # ------------------------------------------------------------------ #
    budget = {}
    for v in C.VERIFIERS:
        for bench in set(r["benchmark"] for r in records):
            ts = [r.get(f"{v}_time") for r in records
                  if r["benchmark"] == bench and r.get(f"{v}_status") == C.POS_STATUS
                  and isinstance(r.get(f"{v}_time"), (int, float))]
            if ts:
                budget[(v, bench)] = float(np.median(ts))
            else:
                # fallback: max recorded time in the cell
                alt = [r.get(f"{v}_time") for r in records
                       if r["benchmark"] == bench and isinstance(r.get(f"{v}_time"), (int, float))]
                budget[(v, bench)] = float(max(alt)) if alt else np.nan

    # round budgets to a clean grid for horizon comparisons
    def clean_budget(b):
        if not np.isfinite(b):
            return np.nan
        grid = np.array([60, 120, 180, 240, 300, 360, 480, 600, 720])
        return float(grid[np.argmin(np.abs(grid - b))])

    budget_clean = {k: clean_budget(v) for k, v in budget.items()}

    # ------------------------------------------------------------------ #
    # 3. canonical long table: one row per (instance, verifier)
    # ------------------------------------------------------------------ #
    rows = []
    for r in records:
        bench = r["benchmark"]
        iid = r["instance_id"]
        feat = feat_cache[(bench, iid)]
        if feat is None:
            excluded.append(("instance", f"{bench}/{iid}", "*", "no_onnx_and_not_poly"))
            continue
        gid = group_id(bench, iid, feat)
        base = {
            "instance_id": iid,
            "benchmark": bench,
            "domain": C.domain_of(bench),
            "constructor": r.get("construction_short"),
            "network_group": gid,
            "feat_source": feat["feat_source"],
        }
        # profile features (canonical names), preserve raw
        prof_missing = 0
        for src, dst in C.PROFILE_MAP.items():
            val = r.get(src)
            base[dst] = float(val) if isinstance(val, (int, float)) else np.nan
            if not isinstance(val, (int, float)):
                prof_missing += 1
        base["profile_missing_ct"] = prof_missing
        # size/type features
        for k in C.SIZE_NUMERIC + C.ARCH_ONEHOT + C.OP_FLAGS:
            base[k] = feat.get(k, np.nan)
        base["arch_type"] = feat.get("_arch")
        base["size_missing_ct"] = int(sum(
            1 for k in C.SIZE_NUMERIC if not np.isfinite(float(feat.get(k, np.nan)))
        ))

        for v in C.VERIFIERS:
            status = r.get(f"{v}_status")
            t = r.get(f"{v}_time")
            row = dict(base)
            row["verifier"] = v
            row["raw_status"] = status
            row["raw_time"] = float(t) if isinstance(t, (int, float)) else np.nan
            row["raw_outcome"] = r.get(f"{v}_outcome")
            row["budget_s"] = budget_clean.get((v, bench), np.nan)
            rows.append(row)

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # 4. audit
    # ------------------------------------------------------------------ #
    audit = {}
    audit["n_rows"] = int(len(df))
    audit["n_unique_instances"] = int(df["instance_id"].nunique())
    audit["n_unique_networks"] = int(df["network_group"].nunique())
    audit["rows_per_domain"] = df["domain"].value_counts().to_dict()
    audit["rows_per_benchmark"] = df["benchmark"].value_counts().to_dict()
    audit["rows_per_constructor"] = df["constructor"].value_counts().to_dict()
    audit["rows_per_verifier"] = df["verifier"].value_counts().to_dict()
    audit["status_per_verifier"] = {
        v: df[df.verifier == v]["raw_status"].value_counts().to_dict() for v in C.VERIFIERS
    }
    audit["groups_per_domain"] = df.groupby("domain")["network_group"].nunique().to_dict()
    audit["groups_per_benchmark"] = df.groupby("benchmark")["network_group"].nunique().to_dict()
    audit["budgets_per_verifier_benchmark"] = {
        f"{v}|{b}": budget_clean[(v, b)] for (v, b) in sorted(budget_clean)
        if b in set(df.benchmark)
    }
    audit["profile_missing_rows"] = int((df["profile_missing_ct"] > 0).sum())
    audit["size_missing_rows"] = int((df["size_missing_ct"] > 0).sum())
    audit["feat_source_counts"] = df.groupby("instance_id")["feat_source"].first().value_counts().to_dict()
    audit["arch_type_counts"] = (
        df.drop_duplicates("instance_id")["arch_type"].value_counts().to_dict()
    )

    # duplicate id check
    dup = df.duplicated(subset=["instance_id", "verifier", "benchmark"]).sum()
    audit["duplicate_instance_verifier_rows"] = int(dup)

    # ------------------------------------------------------------------ #
    # 5. write outputs
    # ------------------------------------------------------------------ #
    df.to_parquet(C.DATA_PROCESSED / "instance_verifier_rows.parquet")
    df.to_csv(C.DATA_PROCESSED / "instance_verifier_rows.csv", index=False)
    pd.DataFrame(excluded, columns=["level", "id", "verifier", "reason"]).to_csv(
        C.RESULTS / "audit" / "excluded_rows.csv", index=False
    )
    json.dump(audit, open(C.RESULTS / "audit" / "coverage_audit.json", "w"), indent=2, default=str)
    json.dump(source_manifest, open(C.DATA_MANIFESTS / "source_manifest.json", "w"), indent=2, default=str)

    # frozen manifest (hashes of the frozen objects)
    frozen = {
        "row_table_sha256": C.sha256_of(C.DATA_PROCESSED / "instance_verifier_rows.parquet"),
        "row_table_csv_sha256": C.sha256_of(C.DATA_PROCESSED / "instance_verifier_rows.csv"),
        "n_rows": int(len(df)),
        "n_networks": int(df["network_group"].nunique()),
        "profile_features": C.PROFILE_FEATURES,
        "size_features": C.SIZE_FEATURES,
        "verifiers": C.VERIFIERS,
        "cv": C.load_config()["cv"],
        "budgets": {f"{v}|{b}": budget_clean[(v, b)] for (v, b) in budget_clean
                    if b in set(df.benchmark)},
    }
    frozen["frozen_manifest_sha256"] = C.sha256_str(json.dumps(frozen, sort_keys=True, default=str))
    json.dump(frozen, open(C.DATA_MANIFESTS / "frozen_analysis_manifest.json", "w"), indent=2, default=str)

    return df, audit, frozen


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    df, audit, frozen = build()
    print("=== BUILD COMPLETE ===")
    print(f"rows={audit['n_rows']}  instances={audit['n_unique_instances']}  networks={audit['n_unique_networks']}")
    print("rows_per_domain:", audit["rows_per_domain"])
    print("groups_per_domain:", audit["groups_per_domain"])
    print("groups_per_benchmark:", audit["groups_per_benchmark"])
    print("arch_type_counts:", audit["arch_type_counts"])
    print("feat_source_counts:", audit["feat_source_counts"])
    print("frozen sha:", frozen["frozen_manifest_sha256"][:16])
