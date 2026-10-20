#!/usr/bin/env python3
"""In-process drop-in replacement for the compiled `Marabou` CLI binary.

The verify_benchmark marabou adapter (verifier_adapters/marabou.py) invokes a compiled Marabou
executable as:

    <bin> <network.onnx> <property.vnnlib> --timeout=N --num-workers=N \
          --blas-threads=N --verbosity=N --summary-file PATH

That C++ binary is not always available (e.g. the source build cannot fetch Boost on a
network-restricted host). This script accepts the SAME CLI but solves *in process* using the
maraboupy Python bindings (MarabouCore.so from the `maraboupy` wheel), which parse .vnnlib natively
(MarabouCore.loadProperty -> VnnLibParser) and solve via MarabouCore.solve. No compiled Marabou binary
is required.

Point the adapter at this file:  --marabou_bin /path/to/marabou_maraboupy.py  (or export MARABOU_BIN).
It prints a lowercase `sat` / `unsat` line (matched by the adapter) and `timeout reached` on timeout.
"""
import argparse
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("network")                       # .onnx
    p.add_argument("prop")                           # .vnnlib
    p.add_argument("--timeout", type=int, default=0)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=1)
    p.add_argument("--blas-threads", dest="blas_threads", type=int, default=1)
    p.add_argument("--verbosity", type=int, default=0)
    p.add_argument("--summary-file", dest="summary_file", type=str, default="")
    # Tolerate any extra flags the adapter or Marabou binary might pass.
    args, _unknown = p.parse_known_args()

    # BLAS threads must be set before numpy/maraboupy import to take effect.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max(1, args.blas_threads)))
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, args.blas_threads)))

    try:
        from maraboupy import Marabou
    except Exception as e:  # maraboupy not importable
        print("error: could not import maraboupy: %s" % e)
        return 2

    # solve(filename=...) redirects fd 1 into the summary file, and Marabou 2.0 does not restore it
    # when preprocessing already decides the query. Keep a copy of the real stdout for the verdict.
    sys.stdout.flush()
    real_stdout = os.dup(1)
    try:
        net = Marabou.read_onnx(args.network)
        options = Marabou.createOptions(
            timeoutInSeconds=int(args.timeout),
            numWorkers=int(args.num_workers),
            verbosity=int(args.verbosity),
        )
        # Redirect Marabou's internal chatter to the summary file so stdout stays clean; parse .vnnlib
        # via propertyFilename (MarabouCore.loadProperty dispatches to the VnnLibParser for .vnnlib).
        redirect = args.summary_file or ""
        import inspect
        if "propertyFilename" in inspect.signature(net.solve).parameters:
            exit_code, _vals, _stats = net.solve(
                filename=redirect, verbose=False, options=options, propertyFilename=args.prop
            )
        else:  # maraboupy 1.x (the only PyPI wheel on some platforms): encode the property ourselves
            exit_code = _solve_legacy(net, args.prop, redirect, options)
    except Exception as e:
        sys.stdout.flush()
        os.dup2(real_stdout, 1)
        print("error: marabou solve failed: %s" % e)
        return 2

    sys.stdout.flush()
    os.dup2(real_stdout, 1)
    os.close(real_stdout)

    ec = str(exit_code).strip()
    # Emit a line the adapter's parse_result recognizes (it matches ^unsat$ / ^sat$, and timeout /
    # error language). Also echo the raw code for the logs.
    if ec == "unsat":
        print("unsat")
    elif ec == "sat":
        print("sat")
    elif ec == "TIMEOUT":
        print("timeout reached")
    else:
        # UNKNOWN / ERROR / QUIT_REQUESTED
        print("error: marabou returned %s" % ec)
    print("marabou_exit_code=%s" % ec)
    return 0


def _parse_robustness_vnnlib(path):
    """Input box + disjunction of single-atom (>= Y_a Y_b) / (<= Y_a Y_b) violation constraints."""
    import re
    text = open(path).read()
    lo, hi = {}, {}
    for op, i, v in re.findall(r"\(\s*(>=|<=)\s+X_(\d+)\s+([-+0-9.eE]+)\s*\)", text):
        (lo if op == ">=" else hi)[int(i)] = float(v)
    out_part = text[text.find("(assert (or"):] if "(assert (or" in text else text
    disjuncts = []
    for chunk in re.split(r"\(\s*and\s+", out_part)[1:]:
        atoms = re.findall(r"\(\s*(>=|<=)\s+Y_(\d+)\s+Y_(\d+)\s*\)", chunk.split(")")[0] + ")")
        if len(atoms) != 1:
            raise ValueError("unsupported vnnlib: multi-atom disjunct")
        op, a_, b_ = atoms[0]
        big, small = (int(a_), int(b_)) if op == ">=" else (int(b_), int(a_))
        disjuncts.append((big, small))                     # violation: Y_big >= Y_small
    if not disjuncts:
        raise ValueError("unsupported vnnlib: no output disjunction")
    return lo, hi, disjuncts


def _solve_legacy(net, prop_path, redirect, options):
    """maraboupy 1.x: set the input box, add the violation disjunction, map the result to an exit code.

    solve() returns (vals, stats): vals is non-empty iff a satisfying assignment (a counterexample)
    was found; stats.hasTimedOut() distinguishes a timeout from a completed UNSAT search."""
    from maraboupy import MarabouCore
    lo, hi, disjuncts = _parse_robustness_vnnlib(prop_path)
    xs = [v for v in __import__("numpy").array(net.inputVars[0]).flatten()]
    ys = [v for v in __import__("numpy").array(net.outputVars[0] if isinstance(net.outputVars, list) else net.outputVars).flatten()]
    for i, var in enumerate(xs):
        net.setLowerBound(var, lo[i])
        net.setUpperBound(var, hi[i])
    disj = []
    for big, small in disjuncts:                           # Y_big - Y_small >= 0
        eq = MarabouCore.Equation(MarabouCore.Equation.GE)
        eq.addAddend(1.0, ys[big])
        eq.addAddend(-1.0, ys[small])
        eq.setScalar(0.0)
        disj.append([eq])
    net.addDisjunctionConstraint(disj)
    res = net.solve(filename=redirect, verbose=False, options=options)
    if len(res) == 3:
        return res[0]
    vals, stats = res
    if vals:
        return "sat"
    if stats is not None and stats.hasTimedOut():
        return "TIMEOUT"
    return "unsat"


if __name__ == "__main__":
    raise SystemExit(main())
