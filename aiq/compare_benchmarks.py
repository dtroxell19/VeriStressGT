"""Compare benchmarks by which planted bugs they can expose (Thrust 1 pool on each benchmark).

    python aiq/compare_benchmarks.py VeriStressGT=<run>/results.json mnist_fc=<run>/results.json \
        oval21=<run>/results.json --out benchmark_comparison.md

Each results.json comes from aiq/thrust1_runner.py run with the same verifier pool on a different
benchmark. Per planted bug and benchmark:

  fired      the bug changed at least one verdict of the verifier it was planted in. A bug that never
             fires on a benchmark cannot be caught there by any labelling.
  labels     caught by the benchmark's own labels: analytic certificates + witnesses (VeriStressGT),
             or only attack-certified counterexamples (external benchmarks, aiq/build_external_bench.py)
  MV-1 / MV  caught by majority vote (one buggy verifier in the pool / the whole pool)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())["result"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="name=path/to/results.json")
    ap.add_argument("--out", default=None, help="Write the markdown report here (default: stdout).")
    args = ap.parse_args(argv)

    runs = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        runs[name] = _load(path)
    names = list(runs)
    planted = sorted({v for r in runs.values() for v, d in r["per_verifier"].items() if d["role"] == "planted"})

    lines = ["## Summary", "",
             "| Benchmark | Instances (UNSAT / SAT / no label) | Bugs that fire | Caught by the benchmark's labels "
             "| false-UNSAT bugs caught by labels | Caught by majority vote (1 buggy / full) "
             "| Sound verifiers blamed by full-pool vote |",
             "|---|---|---|---|---|---|---|"]
    for n, r in runs.items():
        per = r["per_verifier"]
        s = r["summary"]
        fired = [v for v in planted if per.get(v, {}).get("exercised")]
        fu = [v for v in planted if per.get(v, {}).get("fails_as") == "false_unsat"]
        fu_caught = [v for v in fu if per[v]["gt_flagged"] and per[v]["exercised"]]
        gt_c = [v for v in fired if per[v]["gt_flagged"]]
        mv1 = [v for v in fired if per[v]["mv_one_buggy_flagged"]]
        mvf = [v for v in fired if per[v]["mv_full_flagged"]]
        penal = s["majority_full"].get("sound_verifiers_penalized", [])
        unl = r.get("n_unlabelled", 0)
        lines.append(f"| {n} | {r['n_instances']} ({r['n_unsat']} / {r['n_sat']} / {unl}) | "
                     f"{len(fired)}/{len(planted)} | **{len(gt_c)}/{len(planted)}** | {len(fu_caught)}/{len(fu)} | "
                     f"{len(mv1)} / {len(mvf)} | {len(penal)}{': ' + ', '.join(penal) if penal else ''} |")
    lines += ["", "Rates are out of all planted bugs, so a bug that never fires counts as missed: on that "
              "benchmark nothing can catch it.", ""]

    def cell(r, v):
        d = r["per_verifier"].get(v)
        if d is None:
            return "n/a"
        if not d["exercised"]:
            return "-"
        marks = [m for m, k in (("L", "gt_flagged"), ("M1", "mv_one_buggy_flagged"), ("M", "mv_full_flagged")) if d[k]]
        return " ".join(marks) if marks else "fired, missed"

    lines += ["## Per planted bug", "",
              "`-` never fired · `L` caught by the benchmark's labels · `M1` / `M` caught by majority vote "
              "(one buggy / full pool) · `fired, missed` changed verdicts that nothing could expose", "",
              "| Planted bug | Fails as | " + " | ".join(names) + " |",
              "|---|---|" + "---|" * len(names)]
    for v in planted:
        fa = next((r["per_verifier"][v]["fails_as"] for r in runs.values() if v in r["per_verifier"]), "")
        lines.append(f"| `{v}` | {fa} | " + " | ".join(cell(runs[n], v) for n in names) + " |")

    lines += ["", "## Real verifiers flagged by each benchmark's labels", ""]
    for n, r in runs.items():
        real = [v for v, d in r["per_verifier"].items() if d["role"] == "real"]
        fl = {v: sorted({e["instance_id"] for e in r["per_verifier"][v]["gt_evidence"]}) for v in real
              if r["per_verifier"][v]["gt_flagged"]}
        lines.append(f"- **{n}:** " + ("; ".join(f"{v} ({', '.join(i)})" for v, i in fl.items()) or "none"))

    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"Wrote {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
