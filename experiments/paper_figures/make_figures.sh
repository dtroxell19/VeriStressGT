#!/usr/bin/env bash
# Rebuild the paper's Figure 2 (runtime outcomes per constructor x verifier) and Figure 3
# (timeout AUC per profile component; timeout rate over U x d_eff) from the committed
# benchmarks, Difficulty Profiles and runs_standardized results.
#
#   bash experiments/paper_figures/make_figures.sh            # both variants
#   VARIANTS=convex bash experiments/paper_figures/make_figures.sh
#
# Variants (the 225-instance VeriStress-GT suite plus MNIST_fc and oval21):
#   paper   sweep_all + polynomial_stress_22 (the 22 numerically screened polynomial nets),
#           i.e. the published figures.
#   convex  sweep_all + polynomial_convex_certified (22 analytically certified convex
#           polynomial nets) replacing polynomial_stress_22. Its runs are abcrown and NeuralSAT
#           (GPU server) and nnenum; Marabou and PyRAT have no convex results and count as errors.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src" MPLBACKEND=Agg
PY="${PYTHON:-python}"
B=src/VeriStressGT/benchmarks
RS=src/VeriStressGT/runs_standardized
HERE=experiments/paper_figures

for variant in ${VARIANTS:-paper convex}; do
  case "$variant" in
    paper)  poly=polynomial_stress_22;        poly_runs="abcrown_poly marabou_poly neuralsat_poly nnenum_poly pyrat_poly" ;;
    convex) poly=polynomial_convex_certified; poly_runs="abcrown_poly_convex neuralsat_poly_convex nnenum_poly_convex" ;;
    *) echo "unknown variant: $variant" >&2; exit 1 ;;
  esac
  runs=()
  for v in abcrown_activation_veristress marabou_veristress neuralsat_veristress nnenum_veristress pyrat_veristress \
           abcrown_mnist marabou_mnist neuralsat_mnist nnenum_mnist pyrat_mnist \
           abcrown_oval21 marabou_oval21 neuralsat_oval21 nnenum_oval21 pyrat_oval21 $poly_runs; do
    runs+=("$RS/$v")
  done
  work="$(mktemp -d)"
  echo "=== $variant: $poly ==="
  "$PY" src/VeriStressGT/analysis/analyze_profile_updated_components.py \
    --benchmarks "$B/sweep_all" "$B/$poly" "$B/oval21" "$B/vnncomp_mnist_fc" \
    --runs "${runs[@]}" --out-dir "$work/fig2" --skip-models > "$work/fig2.log"
  "$PY" src/VeriStressGT/analysis/make_targeted_timeout_heatmaps_with_auc.py \
    --merged-records "$work/fig2/merged_records.json" --out-dir "$work/fig3" --bins 5 > "$work/fig3.log"

  out="$HERE/$variant"
  mkdir -p "$out"
  cp "$work/fig2/construction_verifier_heatmap.png"               "$out/figure2_runtime_outcomes.png"
  cp "$work/fig2/outcome_summary_by_construction_verifier.csv"    "$out/figure2_outcomes_by_constructor.csv"
  cp "$work/fig2/merged_records.json"                             "$out/merged_records.json"
  cp "$work/fig3/fig_pooled_oriented_timeout_auc_scorecard.png"   "$out/figure3a_timeout_auc.png"
  cp "$work/fig3/fig_pooled_oriented_timeout_auc_scorecard.pdf"   "$out/figure3a_timeout_auc.pdf"
  cp "$work/fig3/table_pooled_oriented_timeout_auc.csv"           "$out/figure3a_timeout_auc.csv"
  cp "$work/fig3/fig_timeout_heatmap_U_by_d_eff.png"              "$out/figure3b_U_by_d_eff.png"
  cp "$work/fig3/fig_timeout_heatmap_U_by_d_eff.pdf"              "$out/figure3b_U_by_d_eff.pdf"
  cp "$work/fig3/debug_timeout_heatmap_U_by_d_eff_bins.csv"       "$out/figure3b_U_by_d_eff_bins.csv"
  rm -rf "$work"
  echo "  -> $out"
done
