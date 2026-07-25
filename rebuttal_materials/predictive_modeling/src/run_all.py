"""Resumable driver for the predictive-modeling experiment.

Execution order (spec sec 14):
  1. (build_dataset.py is run separately / first)
  2. Primary repeated grouped nested-CV, all feature sets, one pass per cell:
       Scenario A (synthetic), B (established, exploratory), C (combined)
  3. Cross-domain transfer (Scenario D)
  4. Correlations
Everything checkpoints per cell to results/cv/. Re-running skips completed cells.

Usage:
  python run_all.py --repeats 20            # full
  python run_all.py --repeats 3 --smoke     # fast validation
  python run_all.py --only-scenario A --only-verifier abcrown
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
import models as M
import evaluate as E
import grouped_splits as GS


def _save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))


def scenario_frames(df):
    return {
        "A_synthetic": df[df.domain == "synthetic"],
        "B_established": df[df.domain == "established"],
        "C_combined": df,
    }


def run_primary(df, cfg, feats, only_scn=None, only_ver=None, n_jobs=-1):
    cvdir = C.RESULTS / "cv"
    cvdir.mkdir(parents=True, exist_ok=True)
    frames = scenario_frames(df)
    index = []
    for scn, fr in frames.items():
        if only_scn and not scn.startswith(only_scn):
            continue
        for v in C.VERIFIERS:
            if only_ver and v != only_ver:
                continue
            tag = f"{scn}__{v}"
            summ_path = cvdir / f"{tag}.summary.json"
            if summ_path.exists():
                print(f"[skip] {tag} (done)")
                index.append(tag)
                continue
            cell = fr[fr.verifier == v]
            print(f"[run ] {tag}  rows={len(cell)}", flush=True)
            t0 = time.time()
            try:
                oof, meta = E.run_cell(cell, feats, cfg, label=tag, n_jobs=n_jobs)
            except Exception as exc:
                import traceback
                print(f"[ERR ] {tag}: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                _save({"meta": {"label": tag, "error": f"{type(exc).__name__}: {exc}"},
                       "summary": None}, summ_path)
                continue
            meta["elapsed_s"] = round(time.time() - t0, 1)
            if len(oof):
                oof.to_parquet(cvdir / f"{tag}.oof.parquet")
                summ, _ = E.summarize(oof, list(feats))
                out = {"meta": meta, "summary": summ}
            else:
                out = {"meta": meta, "summary": None}
            _save(out, summ_path)
            a = (out["summary"]["feature_sets"]["SD"]["auc"]["median"]
                 if out["summary"] else None)
            print(f"[done] {tag}  H={meta['horizon_s']}s n={meta['n']} base={meta['base_rate']:.2f} "
                  f"elig={meta['eligible']} SD_AUC={a}  ({meta['elapsed_s']}s)", flush=True)
            index.append(tag)
    return index


# --------------------------------------------------------------------------- #
# Scenario D: cross-domain transfer
# --------------------------------------------------------------------------- #
def run_transfer(df, cfg, feats_keys=("S", "SD"), n_jobs=-1):
    from sklearn.metrics import roc_auc_score, brier_score_loss
    fs = M.feature_sets()
    outdir = C.RESULTS / "transfer"
    outdir.mkdir(parents=True, exist_ok=True)
    grid = cfg["grid"]
    results = {}
    for v in C.VERIFIERS:
        cell = df[df.verifier == v]
        # common horizon over union (per verifier) = combined-cell horizon
        H, _, _ = GS.choose_horizon(cell.reset_index(drop=True))
        for direction, src_dom, tgt_dom in [("synth_to_estab", "synthetic", "established"),
                                             ("estab_to_synth", "established", "synthetic")]:
            tag = f"{v}__{direction}"
            path = outdir / f"{tag}.json"
            if path.exists():
                results[tag] = json.load(open(path)); print(f"[skip] transfer {tag}"); continue
            src = cell[cell.domain == src_dom].reset_index(drop=True)
            tgt = cell[cell.domain == tgt_dom].reset_index(drop=True)

            def lab(fr):
                return np.array([GS.label_at_horizon(s, t, b, H) for s, t, b in
                                 zip(fr.raw_status, fr.raw_time, fr.budget_s)])
            ys, yt = lab(src), lab(tgt)
            ms, mt = np.isfinite(ys), np.isfinite(yt)
            src, ys = src[ms].reset_index(drop=True), ys[ms].astype(int)
            tgt, yt = tgt[mt].reset_index(drop=True), yt[mt].astype(int)
            gs = src.network_group.to_numpy(); gt = tgt.network_group.to_numpy()
            entry = {"horizon_s": H, "n_src": int(len(src)), "n_tgt": int(len(tgt)),
                     "tgt_base_rate": float(yt.mean()) if len(yt) else None,
                     "n_tgt_groups": int(len(np.unique(gt)))}
            if len(np.unique(yt)) < 2 or len(np.unique(ys)) < 2 or len(yt) < 20:
                entry["verdict"] = "BLOCKED_INSUFFICIENT_TARGET"
                _save(entry, path); results[tag] = entry; continue

            preds = {}
            for fk in feats_keys:
                cols = fs[fk]
                Xs = M.build_design_matrix(src, cols); Xt = M.build_design_matrix(tgt, cols)
                # tune on source with grouped inner CV (seed-averaged params via one GridSearch)
                inner = E._inner_cv(ys, gs, cfg["cv"]["inner_folds"])
                from sklearn.model_selection import GridSearchCV
                g = GridSearchCV(M.make_pipeline(), M.param_grid(grid), scoring="roc_auc",
                                 cv=inner, n_jobs=n_jobs, refit=True, error_score=np.nan)
                g.fit(Xs, ys, groups=gs)
                pt = g.best_estimator_.predict_proba(Xt)[:, 1]
                preds[fk] = pt
                entry[f"{fk}_auc"] = float(roc_auc_score(yt, pt))
                entry[f"{fk}_brier"] = float(brier_score_loss(yt, pt))
            # group bootstrap over target networks for paired dAUC
            if "S" in preds and "SD" in preds:
                rng = np.random.RandomState(7); uniq = np.unique(gt); boots = []
                point = roc_auc_score(yt, preds["SD"]) - roc_auc_score(yt, preds["S"])
                for _ in range(2000):
                    samp = rng.choice(uniq, len(uniq), replace=True)
                    idx = np.concatenate([np.where(gt == gg)[0] for gg in samp])
                    if len(np.unique(yt[idx])) < 2: continue
                    boots.append(roc_auc_score(yt[idx], preds["SD"][idx])
                                 - roc_auc_score(yt[idx], preds["S"][idx]))
                boots = np.array(boots)
                entry["dauc_point"] = float(point)
                entry["dauc_lo"] = float(np.percentile(boots, 2.5)) if len(boots) else None
                entry["dauc_hi"] = float(np.percentile(boots, 97.5)) if len(boots) else None
            _save(entry, path); results[tag] = entry
            print(f"[done] transfer {tag}: tgt_n={entry['n_tgt']} S_auc={entry.get('S_auc')} "
                  f"SD_auc={entry.get('SD_auc')}", flush=True)
    _save(results, outdir / "transfer_summary.json")
    return results


def run_correlations(df):
    outdir = C.RESULTS / "coefficients"; outdir.mkdir(parents=True, exist_ok=True)
    inst = df.drop_duplicates("instance_id")
    num = C.PROFILE_FEATURES + [c for c in C.SIZE_NUMERIC]
    for dom in ["all", "synthetic", "established"]:
        sub = inst if dom == "all" else inst[inst.domain == dom]
        tsub = C.apply_transforms(sub, [c for c in num if c in sub])
        corr = tsub.corr(method="spearman")
        corr.to_csv(outdir / f"spearman_{dom}.csv")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only-scenario", default=None)
    ap.add_argument("--only-verifier", default=None)
    ap.add_argument("--skip-transfer", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    cfg = C.load_config()
    if args.repeats:
        cfg["cv"]["outer_repeats"] = args.repeats
    if args.smoke:
        cfg["cv"]["outer_repeats"] = cfg["run"]["smoke_repeats"]

    df = C.load_rows()
    feats = M.feature_sets()

    print(f"=== PRIMARY CV (repeats={cfg['cv']['outer_repeats']}) ===", flush=True)
    run_primary(df, cfg, feats, args.only_scenario, args.only_verifier, args.n_jobs)
    run_correlations(df)
    if not args.skip_transfer:
        print("=== TRANSFER (Scenario D) ===", flush=True)
        run_transfer(df, cfg, n_jobs=args.n_jobs)
    print("=== run_all complete ===", flush=True)


if __name__ == "__main__":
    main()
