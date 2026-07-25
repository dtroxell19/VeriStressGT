"""Consolidate per-instance out-of-fold predictions (spec sec 11).

Produces:
  results/predictions/out_of_fold_predictions.csv   (one row per instance x verifier
      x scenario x repeat x fold, with size-only / profile-only / augmented probs)
  results/predictions/instance_verifier_repeat_avg.csv  (compact repeat-averaged)
"""
from __future__ import annotations
import glob, os
import numpy as np
import pandas as pd

import common as C


def build():
    rowtab = C.load_rows()
    rowtab["censored"] = (rowtab.raw_status == "TIMEOUT").astype(int)
    key = rowtab.set_index(["instance_id", "verifier"])[["raw_time", "censored", "raw_status", "budget_s"]]

    long_frames = []
    for f in glob.glob(str(C.RESULTS / "cv" / "*.oof.parquet")):
        tag = os.path.basename(f).replace(".oof.parquet", "")
        scn, v = tag.split("__")
        oof = pd.read_parquet(f)
        piv = oof.pivot_table(index=["repeat", "fold", "instance_id", "network_group",
                                     "benchmark", "constructor", "domain", "arch_type", "y"],
                              columns="feature_set", values="p")
        thr = oof.pivot_table(index=["repeat", "fold", "instance_id"], columns="feature_set", values="thr")
        piv = piv.reset_index()
        piv["verifier"] = v; piv["scenario"] = scn
        # attach thresholds for S and SD
        thr = thr.reset_index().rename(columns={"S": "thr_S", "SD": "thr_SD"})
        piv = piv.merge(thr[["repeat", "fold", "instance_id", "thr_S", "thr_SD"]],
                        on=["repeat", "fold", "instance_id"], how="left")
        long_frames.append(piv)
    if not long_frames:
        print("no oof parquet yet"); return None
    long = pd.concat(long_frames, ignore_index=True)
    # predicted classes
    for fs in ["S", "SD"]:
        if fs in long:
            long[f"pred_{fs}"] = (long[fs] >= long[f"thr_{fs}"]).astype(int)
    long = long.rename(columns={"S": "p_size", "D": "p_profile_only", "SD": "p_augmented",
                                "y": "y_timeout"})
    # observed runtime / censoring
    long = long.merge(key.reset_index(), on=["instance_id", "verifier"], how="left")

    cols = ["instance_id", "network_group", "verifier", "scenario", "repeat", "fold",
            "y_timeout", "raw_time", "censored", "raw_status", "budget_s",
            "p_size", "p_profile_only", "p_augmented", "thr_S", "thr_SD",
            "pred_S", "pred_SD", "domain", "benchmark", "constructor", "arch_type"]
    long = long[[c for c in cols if c in long.columns]]
    (C.RESULTS / "predictions").mkdir(parents=True, exist_ok=True)
    long.to_csv(C.RESULTS / "predictions" / "out_of_fold_predictions.csv", index=False)

    compact = (long.groupby(["instance_id", "verifier", "scenario"])
                   .agg(y_timeout=("y_timeout", "first"),
                        p_size=("p_size", "mean"), p_profile_only=("p_profile_only", "mean"),
                        p_augmented=("p_augmented", "mean"),
                        domain=("domain", "first"), benchmark=("benchmark", "first"),
                        constructor=("constructor", "first"), arch_type=("arch_type", "first"))
                   .reset_index())
    compact.to_csv(C.RESULTS / "predictions" / "instance_verifier_repeat_avg.csv", index=False)
    print(f"wrote {len(long)} OOF rows, {len(compact)} compact rows")
    return long


if __name__ == "__main__":
    build()
