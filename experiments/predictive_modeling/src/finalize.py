"""Post-CV finalization: run after run_all.py completes.
Chains interpretation, runtime regression, per-instance predictions, report
tables, and figures. Each step is guarded so one failure doesn't block the rest.
"""
from __future__ import annotations
import traceback
import pandas as pd

import common as C


def _step(name, fn):
    print(f"\n=== {name} ===", flush=True)
    try:
        fn()
        print(f"[ok] {name}", flush=True)
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


def main():
    df = C.load_rows()
    cells = [(v, dom) for v in C.VERIFIERS for dom in ["synthetic", "combined"]]

    import interpret, runtime_regression, predictions, report, plots

    _step("interpret: final coefficients", lambda: interpret.final_coefficients(df, cells))
    _step("interpret: interactions", lambda: interpret.interaction_model(df, cells))
    _step("interpret: shallow trees", lambda: interpret.shallow_trees(df, cells))
    _step("runtime regression", lambda: runtime_regression.run_runtime(
        df, C.load_config(), C.RESULTS / "runtime" / "runtime_regression.csv"))
    _step("per-instance predictions", predictions.build)
    _step("report tables", lambda: (report.primary_table(), report.transfer_table(),
                                    report.ablation_summary(), report.coefficient_table(),
                                    report.per_subgroup_breakdown(), report.verdict_summary()))
    _step("plots", plots.main)
    print("\n=== finalize complete ===")


if __name__ == "__main__":
    main()
