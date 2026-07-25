"""Timeout labeling at a horizon, common-horizon selection, and repeated
group-aware stratified splits (with an explicit no-leakage guarantee)."""
from __future__ import annotations
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

import common as C

HORIZON_GRID = [60, 120, 180, 240, 300, 360, 480, 600]


def label_at_horizon(status, time, budget, horizon):
    """Timeout target at `horizon` seconds (spec sec 4.1/4.2).
    Returns 1 (timeout), 0 (solved before horizon), or np.nan (excluded).

    A *conclusively solved* instance is labelable at ANY horizon regardless of the
    run's nominal budget (a solve in 8 s is a valid negative at H=600). Only a true
    TIMEOUT needs budget >= H, because otherwise we cannot know whether it would have
    solved between its budget and H. (Fixes a bug where an all-solved/all-error
    sub-benchmark spuriously dragged the common horizon down via budget rounding.)"""
    if status in C.EXCLUDE_STATUSES:
        return np.nan                      # ERROR/UNKNOWN/missing/other
    if status in C.NEG_STATUSES:           # conclusive UNSAT / SAT -> valid at any horizon
        if not np.isfinite(time):
            return np.nan
        return 1.0 if time >= horizon - 1e-6 else 0.0   # solved-after-horizon => timeout@H
    if status == C.POS_STATUS:             # TIMEOUT: only labelable if the run reached H
        if budget is None or not np.isfinite(budget) or budget < horizon - 1e-6:
            return np.nan
        return 1.0
    return np.nan


def choose_horizon(cell_df, retain=0.85, min_class=10):
    """Largest grid horizon retaining >= `retain` of rows with both classes
    present. Returns (horizon, labels(np.array aligned to cell_df), n_dropped)."""
    n = len(cell_df)
    best = None
    for H in sorted(HORIZON_GRID, reverse=True):
        y = np.array([label_at_horizon(s, t, b, H) for s, t, b in
                      zip(cell_df["raw_status"], cell_df["raw_time"], cell_df["budget_s"])])
        mask = np.isfinite(y)
        if mask.sum() < retain * n:
            continue
        yv = y[mask]
        if (yv == 1).sum() < min_class or (yv == 0).sum() < min_class:
            continue
        best = (H, y, int((~mask).sum()))
        break
    if best is None:
        # fall back to the smallest horizon (keeps the most rows) even if underpowered
        H = HORIZON_GRID[0]
        y = np.array([label_at_horizon(s, t, b, H) for s, t, b in
                      zip(cell_df["raw_status"], cell_df["raw_time"], cell_df["budget_s"])])
        best = (H, y, int((~np.isfinite(y)).sum()))
    return best


def repeated_group_splits(y, groups, n_splits, n_repeats, seed_base, min_splits_fallback=(5, 4, 3)):
    """Yield (repeat_idx, fold_idx, train_idx, test_idx) for repeated stratified
    group k-fold. Falls back to fewer splits (then plain GroupKFold) if the
    stratified splitter can't honor the class/group constraints."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    for rep in range(n_repeats):
        seed = seed_base + rep
        k = None
        for cand in min_splits_fallback:
            if cand <= n_groups:
                k = cand
                break
        if k is None:
            k = max(2, min(n_splits, n_groups))
        splitter = None
        folds = None
        try:
            sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            folds = list(sgkf.split(np.zeros(len(y)), y, groups))
            # verify both classes appear in every train & test fold
            ok = all(len(np.unique(y[tr])) == 2 and len(np.unique(y[te])) == 2
                     for tr, te in folds)
            if not ok:
                raise ValueError("class missing in a fold")
            splitter = "stratified_group"
        except Exception:
            gkf = GroupKFold(n_splits=k)
            folds = list(gkf.split(np.zeros(len(y)), y, groups))
            splitter = "group_kfold"
        for fi, (tr, te) in enumerate(folds):
            yield rep, fi, tr, te, k, splitter


def assert_no_group_leakage(groups, train_idx, test_idx):
    g = np.asarray(groups)
    inter = set(g[train_idx]) & set(g[test_idx])
    if inter:
        raise AssertionError(f"group leakage: {len(inter)} groups in both train and test")
