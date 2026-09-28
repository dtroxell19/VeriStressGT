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
}


def apply_wrapper(bug: str, statuses: Dict[str, str], instances: Dict[str, Dict]) -> Dict[str, str]:
    fn: WrapperFn = WRAPPER_BUGS[bug]["fn"]
    out = {}
    for iid, s in statuses.items():
        new = fn(s, instances[iid])
        out[iid] = s if new is None else new
    return out
