"""Parser for the VNNLIB robustness specs this repo writes (``make_box_vnnlib``).

Input:  per-coordinate box constraints ``(assert (>= X_i v))`` / ``(assert (<= X_i v))``.
Output: a disjunction of conjunctions of ``(>= Y_a Y_b)`` / ``(<= Y_a Y_b)`` atoms describing the
        *violation* set. For robustness specs every disjunct is a single atom ``Y_j >= Y_label``.

The property holds (UNSAT) iff no point in the box satisfies any disjunct. Each disjunct d is
encoded as a row ``c_d`` with ``c_d . y <= 0`` <=> violation, so the property is ``min_d c_d . y > 0``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import numpy as np

_X_RE = re.compile(r"\(\s*(>=|<=)\s+X_(\d+)\s+([-+0-9.eE]+)\s*\)")
_Y_RE = re.compile(r"\(\s*(>=|<=)\s+Y_(\d+)\s+Y_(\d+)\s*\)")


@dataclass
class Spec:
    lo: np.ndarray            # [n_in] input lower bounds (flat)
    hi: np.ndarray            # [n_in] input upper bounds (flat)
    C: np.ndarray             # [n_disjuncts, n_out]; violation of disjunct d  <=>  C[d] . y <= 0
    disjunct_classes: List[int]   # competing class of each disjunct (for diagnostics / bugs)
    label: int

    @property
    def center(self) -> np.ndarray:
        return (self.lo + self.hi) / 2

    @property
    def radius(self) -> np.ndarray:
        return (self.hi - self.lo) / 2


def parse_vnnlib(path: str, num_outputs: int) -> Spec:
    text = open(path).read()
    decl_x = len(re.findall(r"declare-const\s+X_\d+", text))
    lo = np.full(decl_x, -np.inf)
    hi = np.full(decl_x, np.inf)
    for op, i, v in _X_RE.findall(text):
        i, v = int(i), float(v)
        if op == ">=":
            lo[i] = max(lo[i], v)
        else:
            hi[i] = min(hi[i], v)
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError(f"unbounded input coordinate in {path}")

    out_part = text[text.find("(assert (or"):] if "(assert (or" in text else text
    rows, classes, labels = [], [], set()
    for chunk in re.split(r"\(\s*and\s+", out_part)[1:]:
        atoms = _Y_RE.findall(chunk.split(")")[0] + ")")
        if len(atoms) != 1:
            raise ValueError("only single-atom disjuncts are supported")
        op, a, b = atoms[0]
        a, b = int(a), int(b)
        j, l = (a, b) if op == ">=" else (b, a)          # violation: Y_j >= Y_l
        c = np.zeros(num_outputs)
        c[l] += 1.0
        c[j] -= 1.0                                      # c . y = Y_l - Y_j <= 0  <=> violation
        rows.append(c)
        classes.append(j)
        labels.add(l)
    if not rows:
        raise ValueError(f"no output disjuncts parsed from {path}")
    if len(labels) != 1:
        raise ValueError(f"not a single-label robustness spec: {labels}")
    return Spec(lo, hi, np.stack(rows), classes, labels.pop())
