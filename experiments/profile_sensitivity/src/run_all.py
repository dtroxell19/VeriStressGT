"""Driver for the Difficulty-Profile sensitivity study.

Stages:
  subset    -> results/subset.csv                     (stratified instance list)
  driver    -> results/axis_*.csv, sensitivity_long   (recompute sweeps)
  analysis  -> results/*.csv                           (stability, CoV, dependence, index)
  plots     -> plots/*.png
  all       -> subset, driver, analysis, plots

Examples:
  python run_all.py --stage subset
  python run_all.py --stage driver --axes all --n-jobs 100
  python run_all.py --smoke                 # fast local sanity path
"""
from __future__ import annotations
import argparse
import pandas as pd

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["subset", "driver", "analysis", "plots", "all"])
    ap.add_argument("--axes", default="all")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = C.load_config()
    C.ensure_dirs()

    if args.smoke:
        import subset as S
        import driver as D
        if not (C.RESULTS / "subset.csv").exists():
            S.main()
        sub = pd.read_csv(C.RESULTS / "subset.csv")
        sm = cfg["smoke"]
        sub = sub[sub.has_onnx].head(sm["n_instances"]).reset_index(drop=True)
        cfg = dict(cfg); cfg["axes"] = dict(cfg["axes"])
        cfg["axes"]["N"] = dict(cfg["axes"]["N"], grid=sm["N_grid"], seeds=sm["seeds"])
        cfg["axes"]["seed"] = dict(cfg["axes"]["seed"], seeds=sm["seeds"])
        D.run(sub, cfg, sm["axes"], n_jobs=args.n_jobs,
              timeout_s=cfg.get("component_timeout_s", 300))
        return

    if args.stage in ("subset", "all"):
        import subset as S
        S.main()

    if args.stage in ("driver", "all"):
        import driver as D
        sub = pd.read_csv(C.RESULTS / "subset.csv")
        axes = (["N", "dist", "atau", "eta", "U", "seed"]
                if args.axes == "all" else args.axes.split(","))
        D.run(sub, cfg, axes, n_jobs=args.n_jobs,
              timeout_s=cfg.get("component_timeout_s", 300),
              exclude_benchmarks=tuple(cfg.get("exclude_benchmarks_recompute", [])))

    if args.stage in ("analysis", "all"):
        import analysis as A
        A.main()

    if args.stage in ("plots", "all"):
        import plots as P
        P.main()


if __name__ == "__main__":
    main()
