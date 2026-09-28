"""Catalogue of planted verifier bugs for the Thrust 1 evaluation.

Two tiers:
  * MUTANT_BUGS  -- one defect injected into the in-house reference verifier (``refverify``); the
                    wrong answers come from real computation on the instance.
  * WRAPPER_BUGS -- output-level faults applied to a real verifier's recorded verdicts, modelling
                    pipeline bugs (result parsing, timeout handling, backend-specific paths).

Each bug is modelled on a documented failure class. ``fails_as`` says which wrong verdict it can
produce: a false UNSAT (claims robustness that does not hold -- the dangerous direction) or a
false SAT (a spurious counterexample).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

SOUND_CONTROLS: Dict[str, str] = {
    "none": "Reference verifier: attack + CROWN + exact MILP / ReLU-split BaB. Sound.",
    "ibp_only": "Reference with IBP bounds only. Sound but weaker (more UNKNOWN).",
}

MUTANT_BUGS: Dict[str, Dict[str, str]] = {
    "ibp_sign": {
        "fails_as": "false_unsat",
        "description": "Interval matmul computes [min(W lo, W hi), max(W lo, W hi)] instead of splitting "
                       "W into positive/negative parts; intervals are too narrow whenever weights have mixed signs.",
        "models": "classic bound-propagation implementation error",
    },
    "relu_no_intercept": {
        "fails_as": "false_unsat",
        "description": "Upper linear relaxation of an unstable ReLU drops its intercept -u*l/(u-l), so the "
                       "relaxation under-approximates ReLU on the unstable interval.",
        "models": "unsound relaxation (cf. Zombori et al. 2021; SoundnessBench)",
    },
    "conv_bias_dropped": {
        "fails_as": "false_unsat",
        "description": "Conv biases are ignored in bound propagation and the MILP encoding (but not in the "
                       "attack's forward pass). Only affects CNNs.",
        "models": "architecture-specific backend bug",
    },
    "tol_unsat": {
        "fails_as": "false_unsat",
        "description": "Declares a sub-problem verified when its lower bound exceeds -1e-3 instead of 0.",
        "models": "numerical tolerance too loose on the proof side (paper, Sec. 4)",
    },
    "tol_sat": {
        "fails_as": "false_sat",
        "description": "Accepts candidate counterexamples whose margin is below +1e-3 and skips the float64 "
                       "re-check, so near-boundary robust instances get spurious counterexamples.",
        "models": "numerical tolerance on the falsification side (paper, Sec. 4: eps = 0.999 r*)",
    },
    "disjunct_last": {
        "fails_as": "false_unsat",
        "description": "Checks each output disjunct separately and returns the status of the last one "
                       "checked instead of OR-aggregating them.",
        "models": "the disjunctive-spec MIP presolve bug reported in the paper (Appendix D)",
    },
    "input_clip01": {
        "fails_as": "false_unsat",
        "description": "Silently clips the input box to [0, 1] (assumes normalised pixels), verifying a "
                       "subset of the specified region.",
        "models": "input-domain preprocessing assumption",
    },
    "eps_half": {
        "fails_as": "false_unsat",
        "description": "Computes the l_inf radius as (hi - lo)/4 instead of (hi - lo)/2, so only half of "
                       "the specified radius is verified.",
        "models": "specification-parsing off-by-factor bug",
    },
    "weights_fp16": {
        "fails_as": "false_unsat",
        "description": "Loads the network weights in half precision and verifies that copy, i.e. the "
                       "quantized deployment model rather than the network in the specification.",
        "models": "precision mismatch between the verified and the specified model",
    },
    "hwc_layout": {
        "fails_as": "false_unsat",
        "description": "Reads the flat VNNLIB input box as HWC and transposes it to CHW, scrambling the "
                       "per-coordinate bounds of multi-channel image inputs.",
        "models": "tensor-layout (channels-first vs channels-last) conversion bug",
    },
    "bab_any_row": {
        "fails_as": "false_unsat",
        "description": "Treats a sub-domain as verified once ANY output disjunct is proved, instead of all "
                       "of them (conjunction/disjunction confusion in branch-and-bound pruning).",
        "models": "logic error in multi-disjunct pruning",
    },
    "label_off_by_one": {
        "fails_as": "false_sat",
        "description": "Builds the specification rows against class (label + 1) mod C, verifying "
                       "robustness of the wrong class.",
        "models": "0- vs 1-based class-index bug",
    },
}


def _is_conv(inst: Dict) -> bool:
    return str(inst.get("construction", "")).startswith("cnn.")


def _is_attention(inst: Dict) -> bool:
    return str(inst.get("construction", "")).startswith("attention.")


# Wrapper bugs: fn(status, instance) -> new status (statuses are lower-case: sat/unsat/timeout/
# unknown/error/unsupported/missing). ``None`` means "leave unchanged".
WrapperFn = Callable[[str, Dict], Optional[str]]

WRAPPER_BUGS: Dict[str, Dict] = {
    "timeout_as_unsat": {
        "fails_as": "false_unsat",
        "description": "Treats TIMEOUT/UNKNOWN as verified: 'no counterexample found' => robust.",
        "models": "the soundness assumption benchmark labelling makes for unresolved instances",
        "fn": lambda s, inst: "unsat" if s in ("timeout", "unknown") else None,
    },
    "crash_as_sat": {
        "fails_as": "false_sat",
        "description": "A crashed run is parsed as SAT by a loose keyword match on the log.",
        "models": "the abcrown result-parsing bug found and fixed during the rebuttal",
        "fn": lambda s, inst: "sat" if s == "error" else None,
    },
    "conv_sat_to_unsat": {
        "fails_as": "false_unsat",
        "description": "On CNN instances, genuine counterexamples are discarded and reported as UNSAT.",
        "models": "backend-specific soundness bug in convolution handling",
        "fn": lambda s, inst: "unsat" if (s == "sat" and _is_conv(inst)) else None,
    },
    "attention_unsat_to_sat": {
        "fails_as": "false_sat",
        "description": "On attention instances, proofs are replaced by spurious counterexamples.",
        "models": "numerical overflow in softmax / bilinear bound code",
        "fn": lambda s, inst: "sat" if (s == "unsat" and _is_attention(inst)) else None,
    },
    "cache_collision": {
        "fails_as": "false_unsat",
        "description": "Results are cached by network file alone, ignoring the specification: every later "
                       "query on the same network returns the first cached verdict.",
        "models": "incremental-verification cache keyed on the model hash only",
        "apply": lambda statuses, instances: _cache_collision(statuses, instances),
    },
    "flaky": {
        "fails_as": "false_unsat",
        "description": "Nondeterministic parallel reductions flip a definitive verdict on a pseudo-random "
                       "~15% of instances (deterministic per instance id, so runs are reproducible).",
        "models": "run-to-run nondeterminism (e.g. unordered GPU reductions) near decision thresholds",
        "fn": lambda s, inst: ({"sat": "unsat", "unsat": "sat"}.get(s) if _hash_frac(inst["id"]) < 0.15 else None),
    },
}


def _hash_frac(key: str) -> float:
    import hashlib
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _cache_collision(statuses: Dict[str, str], instances: Dict[str, Dict]) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    out = {}
    for iid in sorted(statuses):
        key = instances[iid].get("sha256", {}).get("onnx") or iid
        s = statuses[iid]
        if key in cache:
            out[iid] = cache[key]
        else:
            out[iid] = s
            if s in ("sat", "unsat"):
                cache[key] = s
    return out


def apply_wrapper(bug: str, statuses: Dict[str, str], instances: Dict[str, Dict]) -> Dict[str, str]:
    if "apply" in WRAPPER_BUGS[bug]:  # stateful wrappers see the whole benchmark
        return WRAPPER_BUGS[bug]["apply"](statuses, instances)
    fn: WrapperFn = WRAPPER_BUGS[bug]["fn"]
    out = {}
    for iid, s in statuses.items():
        new = fn(s, instances[iid])
        out[iid] = s if new is None else new
    return out
