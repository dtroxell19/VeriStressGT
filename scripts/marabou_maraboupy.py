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
        exit_code, _vals, _stats = net.solve(
            filename=redirect, verbose=False, options=options, propertyFilename=args.prop
        )
    except Exception as e:
        print("error: marabou solve failed: %s" % e)
        return 2

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


if __name__ == "__main__":
    raise SystemExit(main())
