"""Stratified representative subset for the Difficulty-Profile sensitivity study.

We re-profile a *representative* subset (the reviewers asked for a "brief"
analysis) rather than all 345 instances.  The subset is chosen deterministically
to span:
  * all constructor families (11),
  * all four architectures (MLP / CNN / ATTENTION / POLYNOMIAL),
  * all four benchmarks/datasets (sweep_all, polynomial_stress_22,
    vnncomp_mnist_fc, oval21), and
  * the full empirical difficulty range (easy / medium / hard terciles, where
    difficulty = fraction of the 5 verifiers that time out).

Output: results/subset.csv (one row per selected instance).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import common as C


def _difficulty_bucket(d: float) -> str:
    if d <= 0.0:
        return "easy"
    if d <= 0.5:
        return "medium"
    return "hard"


def _instance_table() -> pd.DataFrame:
    """One row per instance with family, arch, benchmark, difficulty."""
    df = C.load_rows()
    diff = (df.assign(is_to=(df.raw_status == "TIMEOUT").astype(float))
              .groupby("instance_id")["is_to"].mean())
    meta = df.drop_duplicates("instance_id").set_index("instance_id")
    out = pd.DataFrame({
        "instance_id": meta.index,
        "benchmark": meta["benchmark"].values,
        "family": meta["constructor"].values,
        "arch": meta["arch_type"].values,
        "difficulty": diff.reindex(meta.index).values,
    })
    out["tercile"] = out["difficulty"].map(_difficulty_bucket)
    return out.reset_index(drop=True)


def _pick_spread(sub: pd.DataFrame, k: int, rng: np.random.RandomState) -> pd.DataFrame:
    """Pick k rows from a family, spread across difficulty terciles then difficulty."""
    if k >= len(sub):
        return sub
    # round-robin over terciles present, taking evenly-spaced picks within each
    picks = []
    groups = {t: g.sort_values("difficulty") for t, g in sub.groupby("tercile")}
    order = [t for t in ("hard", "medium", "easy") if t in groups]  # prioritize hard
    ti = 0
    while len(picks) < k and any(len(groups[t]) for t in order):
        t = order[ti % len(order)]
        g = groups[t]
        if len(g):
            # take the row nearest an evenly-spaced quantile of the remaining group
            idx = g.index[len(g) // 2]
            picks.append(idx)
            groups[t] = g.drop(idx)
        ti += 1
    return sub.loc[picks]


def select_subset(n_target: int = 48, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    tab = _instance_table()
    n_total = len(tab)
    fam_sizes = tab.groupby("family").size()

    # proportional allocation with a floor of 2 per family
    alloc = {f: max(2, int(round(n_target * s / n_total))) for f, s in fam_sizes.items()}

    chosen = []
    for fam, k in alloc.items():
        sub = tab[tab.family == fam]
        chosen.append(_pick_spread(sub, min(k, len(sub)), rng))
    picked = pd.concat(chosen).drop_duplicates("instance_id")

    # trim toward n_target by dropping from the most over-allocated families,
    # but never drop the last representative of an arch, benchmark, or tercile.
    def _protected_ids(df):
        keep = set()
        for col in ("arch", "benchmark", "tercile", "family"):
            for _, g in df.groupby(col):
                keep.add(g.iloc[0].instance_id)
        return keep

    while len(picked) > n_target:
        prot = _protected_ids(picked)
        # candidate = a droppable row from the currently largest family
        fam_counts = picked.groupby("family").size().sort_values(ascending=False)
        dropped = False
        for fam in fam_counts.index:
            cand = picked[(picked.family == fam) & (~picked.instance_id.isin(prot))]
            if len(cand):
                picked = picked.drop(cand.index[-1])
                dropped = True
                break
        if not dropped:
            break  # everything remaining is protected

    picked = picked.sort_values(["family", "difficulty"]).reset_index(drop=True)

    # resolve paths + onnx availability
    onnx, vnnlib, has_onnx = [], [], []
    for _, r in picked.iterrows():
        o, v = C.instance_paths(r.benchmark, r.instance_id)
        onnx.append(o or "")
        vnnlib.append(v or "")
        has_onnx.append(bool(o))
    picked["onnx_path"] = onnx
    picked["vnnlib_path"] = vnnlib
    picked["has_onnx"] = has_onnx
    picked["domain"] = picked["benchmark"].map(C.domain_of)
    return picked


def main():
    C.ensure_dirs()
    sub = select_subset()
    out = C.RESULTS / "subset.csv"
    sub.to_csv(out, index=False)
    print(f"selected {len(sub)} instances -> {out}")
    print("  archs      :", dict(sub.arch.value_counts()))
    print("  benchmarks :", dict(sub.benchmark.value_counts()))
    print("  terciles   :", dict(sub.tercile.value_counts()))
    print("  families   :", dict(sub.family.value_counts()))
    print("  has_onnx   :", dict(sub.has_onnx.value_counts()))
    miss_v = sub[sub.vnnlib_path == ""]
    if len(miss_v):
        print("  WARNING missing vnnlib:", list(miss_v.instance_id))
    return sub


if __name__ == "__main__":
    main()
