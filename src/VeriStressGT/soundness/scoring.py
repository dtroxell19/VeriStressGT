"""Bug detection: ground-truth labels vs. majority-vote labels.

A verifier is *flagged* by a labelling if any of its definitive verdicts (SAT/UNSAT) contradicts
that labelling's label for the instance. TIMEOUT/UNKNOWN/ERROR/UNSUPPORTED never flag.

Majority vote (the VNN-COMP-style baseline): among the pool's definitive verdicts on an instance,
the label with more votes wins; a tie leaves the instance unresolved (no one is flagged on it); an
instance nobody decides is assumed robust. Two pools are scored:
  * one_buggy: the candidate joined by every non-planted verifier (sound controls + real tools) --
               a competition with a single faulty entrant;
  * full:      every verifier at once.

A third labelling, labels_plus_majority, is the best an external benchmark can do: its own labels
where it has them (e.g. certified counterexamples), majority vote everywhere else. On a fully
labelled benchmark it equals ground_truth. ``summary["coverage"]`` counts how each definitive
verdict can be judged: by a label, only by majority vote (correctness unverifiable), or not at all.

Detection rates are computed over *exercised* planted bugs only: a planted verifier whose verdicts
are identical to its base verifier's on every instance (e.g. a crash-handling fault on a verifier
that never crashed) gave the benchmark nothing to catch and is listed as ``not_exercised``.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional

DEFINITIVE = ("sat", "unsat")


def majority_labels(verdicts: Dict[str, Dict[str, str]], pool: Iterable[str],
                    instance_ids: List[str]) -> Dict[str, Optional[str]]:
    pool = list(pool)
    out: Dict[str, Optional[str]] = {}
    for iid in instance_ids:
        votes = Counter(verdicts[v].get(iid) for v in pool if verdicts[v].get(iid) in DEFINITIVE)
        if not votes:
            out[iid] = "unsat"          # nobody decided: assumed robust
        elif votes["sat"] == votes["unsat"]:
            out[iid] = None             # tie: unresolved
        else:
            out[iid] = votes.most_common(1)[0][0]
    return out


def disagreements(verdicts: Dict[str, str], labels: Dict[str, Optional[str]]) -> List[Dict]:
    out = []
    for iid, s in verdicts.items():
        lab = labels.get(iid)
        if s in DEFINITIVE and lab is not None and s != lab:
            out.append({"instance_id": iid, "verdict": s, "label": lab,
                        "kind": "false_unsat" if s == "unsat" else "false_sat"})
    return out


def score(verdicts: Dict[str, Dict[str, str]], ground_truth: Dict[str, str],
          planted: Dict[str, str], controls: List[str], others: List[str],
          base_of: Optional[Dict[str, str]] = None) -> Dict:
    """
    verdicts:     verifier -> instance -> status (lower-case)
    ground_truth: instance -> "sat" | "unsat" | None (unknown: nothing is flagged on it, and its
                  verdicts are left out of scoring accuracy, counted as "unknown_truth")
    planted:      buggy verifier -> how its bug fails ("false_unsat" / "false_sat")
    controls:     verifiers known to be sound (must never be flagged)
    others:       unmodified real verifiers (soundness unknown a priori)
    base_of:      planted verifier -> the unmodified verifier it was derived from. A planted bug
                  only counts as detected on instances where its verdict differs from the base's;
                  disagreements it merely inherits from the base are reported as "inherited".
    """
    base_of = base_of or {}
    iids = sorted(ground_truth)
    all_v = list(planted) + controls + others
    honest = controls + others
    per: Dict[str, Dict] = {}

    def attributable(v: str, ev: List[Dict]) -> List[Dict]:
        b = base_of.get(v)
        return [e for e in ev if b is None or verdicts[b][e["instance_id"]] != verdicts[v][e["instance_id"]]]

    full_mv = majority_labels(verdicts, all_v, iids)
    combined = {iid: (ground_truth[iid] if ground_truth[iid] is not None else full_mv[iid]) for iid in iids}
    for v in all_v:
        gt_all = disagreements(verdicts[v], ground_truth)
        gt_bad = attributable(v, gt_all)
        pool = honest + ([v] if v in planted else [])
        mv1 = majority_labels(verdicts, pool, iids)
        mv1_bad = attributable(v, disagreements(verdicts[v], mv1))
        mvf_bad = attributable(v, disagreements(verdicts[v], full_mv))
        comb_bad = attributable(v, disagreements(verdicts[v], combined))
        n_def = sum(1 for s in verdicts[v].values() if s in DEFINITIVE)
        b = base_of.get(v)
        per[v] = {
            "role": "planted" if v in planted else ("control" if v in controls else "real"),
            "fails_as": planted.get(v),
            "exercised": (v in planted) and (b is None or any(verdicts[v][i] != verdicts[b][i] for i in iids)),
            "definitive": n_def,
            "gt_flagged": bool(gt_bad), "gt_evidence": gt_bad,
            "gt_inherited": [e for e in gt_all if e not in gt_bad],
            "mv_one_buggy_flagged": bool(mv1_bad), "mv_one_buggy_evidence": mv1_bad,
            "mv_full_flagged": bool(mvf_bad), "mv_full_evidence": mvf_bad,
            "labels_mv_flagged": bool(comb_bad), "labels_mv_evidence": comb_bad,
            # MV flags that are wrong w.r.t. ground truth: MV blames a verdict that is actually correct
            "mv_one_buggy_wrongful": [e for e in mv1_bad if verdicts[v][e["instance_id"]] == ground_truth[e["instance_id"]]],
            "mv_full_wrongful": [e for e in mvf_bad if verdicts[v][e["instance_id"]] == ground_truth[e["instance_id"]]],
        }

    def rate(keys, flag):
        keys = list(keys)
        return (sum(per[k][flag] for k in keys) / len(keys)) if keys else None

    exercised = [v for v in planted if per[v]["exercised"]]
    summary = {}
    for method, flag in (("ground_truth", "gt_flagged"), ("majority_one_buggy", "mv_one_buggy_flagged"),
                         ("majority_full", "mv_full_flagged"), ("labels_plus_majority", "labels_mv_flagged")):
        summary[method] = {
            "detection_rate": rate(exercised, flag),
            "detected": sorted(v for v in exercised if per[v][flag]),
            "missed": sorted(v for v in exercised if not per[v][flag]),
            "not_exercised": sorted(v for v in planted if not per[v]["exercised"]),
            "control_false_flag_rate": rate(controls, flag),
            "controls_flagged": sorted(v for v in controls if per[v][flag]),
            "real_flagged": sorted(v for v in others if per[v][flag]),
        }
    summary["majority_full"]["sound_verifiers_penalized"] = sorted(
        v for v in honest if per[v]["mv_full_wrongful"])
    summary["majority_one_buggy"]["sound_verifiers_penalized"] = sorted(
        v for v in honest if per[v]["mv_one_buggy_wrongful"])
    mv_label_errors = [iid for iid in iids if full_mv[iid] is not None and ground_truth[iid] is not None
                       and full_mv[iid] != ground_truth[iid]]
    summary["majority_full"]["label_errors"] = mv_label_errors
    summary["majority_full"]["unresolved"] = [iid for iid in iids if full_mv[iid] is None]

    cov = {"verdicts": 0, "judged_by_labels": 0, "judged_by_majority_only": 0, "unjudged": 0}
    for v in all_v:
        for iid, s in verdicts[v].items():
            if s not in DEFINITIVE:
                continue
            cov["verdicts"] += 1
            if ground_truth[iid] is not None:
                cov["judged_by_labels"] += 1
            elif full_mv[iid] is not None:
                cov["judged_by_majority_only"] += 1
            else:
                cov["unjudged"] += 1
    summary["coverage"] = cov

    # Scoring accuracy: how often a labelling judges a definitive verdict correctly. Every
    # definitive verdict of every verifier is one judgment; its true correctness comes from the
    # certificates (ground_truth). A labelling can wrongly accuse (calls a correct verdict wrong),
    # wrongly acquit (calls a wrong verdict correct), or leave it unscored (tie).
    one_buggy_labels = {v: majority_labels(verdicts, honest + ([v] if v in planted else []), iids)
                        for v in all_v}
    labellings = {
        "ground_truth": lambda v: ground_truth,
        "majority_one_buggy": lambda v: one_buggy_labels[v],
        "majority_full": lambda v: full_mv,
    }
    for method, labels_for in labellings.items():
        acc = {"judgments": 0, "correctly_scored": 0, "wrongful_accusations": 0,
               "wrongful_acquittals": 0, "unscored": 0, "unknown_truth": 0}
        for v in all_v:
            labels = labels_for(v)
            for iid, s in verdicts[v].items():
                if s not in DEFINITIVE:
                    continue
                if ground_truth[iid] is None:
                    acc["unknown_truth"] += 1
                    continue
                acc["judgments"] += 1
                truly_correct = s == ground_truth[iid]
                lab = labels.get(iid)
                if lab is None:
                    acc["unscored"] += 1
                elif (s == lab) == truly_correct:
                    acc["correctly_scored"] += 1
                elif truly_correct:
                    acc["wrongful_accusations"] += 1
                else:
                    acc["wrongful_acquittals"] += 1
        acc["scoring_accuracy"] = acc["correctly_scored"] / acc["judgments"] if acc["judgments"] else None
        summary[method]["scoring"] = acc
    return {"per_verifier": per, "summary": summary, "majority_full_labels": full_mv}
