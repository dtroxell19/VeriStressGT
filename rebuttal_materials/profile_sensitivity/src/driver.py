"""Sensitivity driver: sweep the Difficulty-Profile estimator knobs one axis at a
time around a production center-point, over the stratified subset.

Each job loads its instance once and calls only the sub-estimator(s) the axis
touches, then reports the canonical components that move on that axis.  Output is
tidy long form (one row per instance x setting x seed x component).

Jobs run in separate processes (loky) so the estimators' torch/numpy *global* RNG
seeding is isolated and reproducible; each worker pins torch to 1 thread to avoid
oversubscribing a many-core host.
"""
from __future__ import annotations
import argparse
import signal
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import common as C


# ------------------------------------------------------------------ estimators
def _stress_preset(name):
    """(weights, include_worstcase) for a sampling-distribution preset."""
    presets = {
        "uniform_only":    ((1.0, 0.0, 0.0, 0.0, 0.0), False),
        "boundary_only":   ((0.0, 0.5, 0.5, 0.0, 0.0), False),
        "current_mixture": (None, True),
        "pgd_heavy":       ((0.10, 0.10, 0.10, 0.10, 0.60), True),
    }
    return presets[name]


class _Timeout(Exception):
    pass


def _alarm(_s, _f):
    raise _Timeout()


def run_job(job: dict) -> list:
    """Execute one sweep cell; return a list of long-form component rows."""
    import torch
    torch.set_num_threads(1)
    from VeriStressGT.difficulty_profile.instance_loader import load_instance
    from VeriStressGT.difficulty_profile import components as K

    base = {k: job[k] for k in (
        "instance_id", "benchmark", "family", "arch", "domain", "axis",
        "n_samples", "dist_preset", "atau_width", "proj_dim", "eta",
        "u_mode", "u_tau", "seed")}

    def emit(component, value, na_reason="", wall=0.0):
        r = dict(base); r.update(component=component, value=value,
                                 na_reason=na_reason, wall_time_s=round(wall, 3))
        return r

    onnx_abs = C.resolve_path(job["onnx_path"])
    vnnlib_abs = C.resolve_path(job["vnnlib_path"])
    if not onnx_abs:
        # No committed network to load (polynomial_stress_22): recompute N/A.
        return [emit(c, np.nan, na_reason="no_committed_onnx") for c in job["report"]]

    rows = []
    weights, worstcase = _stress_preset(job["dist_preset"])
    seed = int(job["seed"])
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(job["timeout_s"]))
    try:
        inst = load_instance(onnx_abs, vnnlib_abs, device="cpu")
        gen = ibp = atau = None
        margin_min = None

        if "generic" in job["estimators"]:
            t0 = time.time()
            gen = K.estimate_generic_components(
                inst, n_samples=int(job["n_samples"]), verbose=False,
                seed=seed, eta=float(job["eta"]),
                stress_weights=weights, stress_include_worstcase=worstcase)
            gt = time.time() - t0
            margin_min = gen.get("margin_sample_min")
            if "margin_hat_min" in job["report"]:
                rows.append(emit("margin_hat_min", margin_min, wall=gt))
            if "d_eff" in job["report"]:
                rows.append(emit("d_eff", gen.get("effective_grad_dim_mean"), wall=gt))

        if "ibp" in job["estimators"]:
            t0 = time.time()
            ibp = K.estimate_ibp_components(
                inst, sample_min_margin=margin_min, eta=float(job["eta"]),
                tau=float(job["u_tau"]), smooth_unstable_mode=job["u_mode"],
                verbose=False)
            it = time.time() - t0
            if "unstable_fraction" in job["report"]:
                rows.append(emit("unstable_fraction", ibp.get("unstable_frac"), wall=it))
            if "g_ibp" in job["report"]:
                rows.append(emit("g_ibp", ibp.get("ibp_relative_gap"), wall=it))

        if "atau" in job["estimators"]:
            t0 = time.time()
            atau = K.estimate_local_region_count(
                inst, n_samples=int(job["atau_n_samples"]),
                projection_dim=int(job["proj_dim"]),
                quantize_width=float(job["atau_width"]), seed=seed, verbose=False)
            at = time.time() - t0
            if "a_tau" in job["report"]:
                rows.append(emit("a_tau", atau.get("A_tau_local_log"), wall=at))
    except _Timeout:
        rows = [emit(c, np.nan, na_reason="timeout") for c in job["report"]]
    except Exception as e:
        rows = [emit(c, np.nan, na_reason=f"{type(e).__name__}: {e}"[:180]) for c in job["report"]]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return rows


# ------------------------------------------------------------------ job builder
def _center(cfg):
    return cfg["center_point"]


def build_jobs(subset: pd.DataFrame, cfg: dict, axes, timeout_s: int) -> list:
    c0 = _center(cfg)
    jobs = []

    def base_row(r):
        return dict(
            instance_id=r.instance_id, benchmark=r.benchmark, family=r.family,
            arch=r.arch, domain=r.domain, onnx_path=r.onnx_path,
            vnnlib_path=r.vnnlib_path, timeout_s=timeout_s,
            # center defaults; axes override
            n_samples=c0["n_samples"], dist_preset=c0["dist_preset"],
            atau_width=c0["atau_width"], proj_dim=c0["atau_proj"],
            eta=c0["eta"], u_mode=c0["u_mode"], u_tau=c0["u_tau"],
            atau_n_samples=min(1024, c0["n_samples"] * 2))

    for _, r in subset.iterrows():
        for axis in axes:
            a = cfg["axes"][axis]
            if axis == "N":
                for N in a["grid"]:
                    for s in a["seeds"]:
                        j = base_row(r); j.update(axis="N", n_samples=N, seed=s,
                                                  atau_n_samples=N,
                                                  estimators=a["estimators"], report=a["report"])
                        jobs.append(j)
            elif axis == "dist":
                for preset in a["grid"]:
                    for s in a["seeds"]:
                        j = base_row(r); j.update(axis="dist", dist_preset=preset, seed=s,
                                                  estimators=a["estimators"], report=a["report"])
                        jobs.append(j)
            elif axis == "atau":
                for w in a["widths"]:
                    for p in a["projs"]:
                        for s in a["seeds"]:
                            j = base_row(r); j.update(axis="atau", atau_width=w, proj_dim=p, seed=s,
                                                      estimators=a["estimators"], report=a["report"])
                            jobs.append(j)
            elif axis == "eta":
                for e in a["grid"]:
                    for s in a["seeds"]:
                        j = base_row(r); j.update(axis="eta", eta=e, seed=s,
                                                  estimators=a["estimators"], report=a["report"])
                        jobs.append(j)
            elif axis == "U":
                for mode in a["modes"]:
                    for tau in a["taus"]:
                        for s in a["seeds"]:
                            j = base_row(r); j.update(axis="U", u_mode=mode, u_tau=tau, seed=s,
                                                      estimators=a["estimators"], report=a["report"])
                            jobs.append(j)
            elif axis == "seed":
                for s in a["seeds"]:
                    j = base_row(r); j.update(axis="seed", seed=s,
                                              estimators=a["estimators"], report=a["report"])
                    jobs.append(j)
    return jobs


LONG_COLS = ["instance_id", "benchmark", "family", "arch", "domain", "axis",
             "n_samples", "dist_preset", "atau_width", "proj_dim", "eta",
             "u_mode", "u_tau", "seed", "component", "value", "na_reason", "wall_time_s"]


def run(subset: pd.DataFrame, cfg: dict, axes, n_jobs=-1, timeout_s=300):
    C.ensure_dirs()
    jobs = build_jobs(subset, cfg, axes, timeout_s)
    print(f"built {len(jobs)} jobs over axes {axes} on {len(subset)} instances", flush=True)
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(run_job)(j) for j in jobs)
    rows = [r for sub in results for r in sub]
    long = pd.DataFrame(rows)[LONG_COLS]
    for axis in axes:
        sub = long[long.axis == axis]
        sub.to_csv(C.RESULTS / f"axis_{axis}.csv", index=False)
    # merge with any existing per-axis files into the consolidated long table
    all_axis = []
    for f in sorted(C.RESULTS.glob("axis_*.csv")):
        all_axis.append(pd.read_csv(f))
    if all_axis:
        pd.concat(all_axis, ignore_index=True).to_csv(C.RESULTS / "sensitivity_long.csv", index=False)
    print(f"wrote {len(long)} component rows -> results/axis_*.csv + sensitivity_long.csv", flush=True)
    return long


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", default="all")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = C.load_config()
    subset = pd.read_csv(C.RESULTS / "subset.csv")

    if args.smoke:
        sm = cfg["smoke"]
        subset = subset[subset.has_onnx].head(sm["n_instances"]).reset_index(drop=True)
        # shrink the N grid + seeds for a fast local sanity run
        cfg = dict(cfg)
        cfg["axes"] = dict(cfg["axes"])
        cfg["axes"]["N"] = dict(cfg["axes"]["N"], grid=sm["N_grid"], seeds=sm["seeds"])
        cfg["axes"]["seed"] = dict(cfg["axes"]["seed"], seeds=sm["seeds"])
        axes = sm["axes"]
    else:
        axes = (["N", "dist", "atau", "eta", "U", "seed"]
                if args.axes == "all" else args.axes.split(","))
    run(subset, cfg, axes, n_jobs=args.n_jobs, timeout_s=cfg.get("component_timeout_s", 300))


if __name__ == "__main__":
    main()
