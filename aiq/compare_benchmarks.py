"""Compare benchmarks by which planted bugs they can expose (Thrust 1 pool on each benchmark).

    python aiq/compare_benchmarks.py VeriStressGT=<run>/results.json mnist_fc=<run>/results.json \
        oval21=<run>/results.json --out benchmark_comparison.md

Each results.json comes from aiq/thrust1_runner.py run with the same verifier pool on a different
benchmark. Per planted bug and benchmark:

  fired      the bug changed at least one verdict of the verifier it was planted in. A bug that never
             fires on a benchmark cannot be caught there by any labelling.
  labels     caught by the benchmark's own labels: analytic certificates + witnesses (VeriStressGT),
             or only attack-certified counterexamples (external benchmarks, aiq/build_external_bench.py)
  labels+MV  caught by its own labels, with majority vote on the instances it has no label for: the
             best an external benchmark can do (VNN-COMP style). Equals labels on VeriStressGT.

Coverage: how each benchmark can judge the pool's definitive verdicts: by its labels (certain), only
by majority vote (correctness unverifiable), or not at all (a tied vote).
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

    def pct(a, b):
        return f"{a}/{b} ({a / b:.0%})" if b else "n/a"

    lines = ["## Bugs caught", "",
             "| Benchmark | Instances (UNSAT / SAT / no label) | Bugs that fire | Caught by own labels "
             "| Caught by own labels + majority vote |",
             "|---|---|---|---|---|"]
    for n, r in runs.items():
        per, s = r["per_verifier"], r["summary"]
        fired = [v for v in planted if per.get(v, {}).get("exercised")]
        lab = [v for v in fired if per[v]["gt_flagged"]]
        lmv = [v for v in fired if per[v].get("labels_mv_flagged")]
        unl = r.get("n_unlabelled", 0)
        lines.append(f"| {n} | {r['n_instances']} ({r['n_unsat']} / {r['n_sat']} / {unl}) | "
                     f"{len(fired)}/{len(planted)} | **{len(lab)}/{len(planted)}** | {len(lmv)}/{len(planted)} |")
    lines += ["", "Out of all planted bugs: a bug that never fires on a benchmark cannot be caught there. "
              "Majority-vote catches on unlabelled instances are known to be right only because the bugs "
              "were planted.", "",
              "## How verdicts can be judged", "",
              "| Benchmark | SAT/UNSAT verdicts | Judged by own labels (certain) "
              "| Judged only by majority vote (unverifiable) | Not judged (tied vote) |",
              "|---|---|---|---|---|"]
    for n, r in runs.items():
        c = r["summary"]["coverage"]
        lines.append(f"| {n} | {c['verdicts']} | **{pct(c['judged_by_labels'], c['verdicts'])}** | "
                     f"{pct(c['judged_by_majority_only'], c['verdicts'])} | {pct(c['unjudged'], c['verdicts'])} |")
    lines.append("")

    def cell(r, v):
        d = r["per_verifier"].get(v)
        if d is None:
            return "n/a"
        if not d["exercised"]:
            return "-"
        marks = [m for m, k in (("L", "gt_flagged"), ("L+M", "labels_mv_flagged")) if d.get(k)]
        return " ".join(marks) if marks else "fired, missed"

    lines += ["## Per planted bug", "",
              "`-` never fired · `L` caught by the benchmark's labels · `L+M` caught by its labels plus "
              "majority vote · `fired, missed` changed verdicts that nothing could expose", "",
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
