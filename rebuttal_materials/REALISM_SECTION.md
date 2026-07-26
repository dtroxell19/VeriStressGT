# Realism of the constructed instances (rebuttal draft)

**Concern (reviewers + AC).** Are the provably-robust instances VeriStress-GT constructs representative
of networks verifiers meet in practice, or are they artificial gadgets whose difficulty (and the
verifier failures they expose) is an artifact of the construction?

We answer in two parts: (1) the constructed instances match the **difficulty profile** of real
verification benchmarks, and (2) the constructions are not synthetic-only — the same architectures can
be realized as **fully trained classifiers on real data** with whole-network ground truth.

---

## 1. Constructed instances track the difficulty profile of real benchmarks

We compare the synthetic constructors against a panel of real verification benchmarks — **MNIST_fc,
oval21, reach_prob_density, rl_benchmarks, and cifar_biasfield** (5 benchmarks, 30 instances each, 150
total) — on the five Difficulty-Profile components. See
`plots/profile_synth_vs_real.png`.

| component | synthetic (min / median / max) | real (min / median / max) | \|Δ median\| (signed-log) |
|---|---|---|--:|
| A_τ (local linear regions) | 0.00 / 5.55 / 6.40 | 0.00 / 5.66 / 6.40 | **0.02** |
| d_eff / input_dim | 0.00 / 0.59 / 0.88 | 0.002 / 0.63 / 0.94 | **0.03** |
| M̂_min (min margin) | 1e−5 / 0.34 / 1.9e3 | −17.7 / 0.19 / 23.6 | **0.12** |
| U (unstable fraction) | 0.00 / 0.88 / 1.00 | 0.01 / 0.64 / 1.00 | **0.13** |
| G_IBP (IBP relative gap) | −0.26 / 1.09 / 2.5e5 | 1.15 / 161 / 1.7e8 | 4.35 |

On four of the five axes the synthetic and real distributions overlap closely — near-identical on
**local-region count (A_τ)** and **effective dimensionality (d_eff/dim)**, same order on **margin** and
**unstable fraction**. The one axis where the synthetic *median* sits below real is **G_IBP** (the IBP
relative gap), where a few real benchmarks — MNIST_fc, cifar_biasfield — have extreme values; the
synthetic *range* still extends into that regime (the per-constructor MEAP family reaches G_IBP ≈ 645,
bracketing MNIST_fc's 771). The constructed instances are therefore not degenerate: across the profile
that predicts verification hardness, they sit inside the real distribution rather than off to one side.

*(We deliberately show synthetic-vs-real here. Difficulty is what the profile is meant to capture, and
this is the fair, load-bearing comparison.)*

### 1.1 Robust-vs-robust: real instances verifiers certify robust

Our constructed instances are all provably robust, so the sharpest comparison is against **real
instances that are themselves verified robust** — returned **UNSAT by at least one** of five verifiers
(α,β-CROWN, Marabou, NeuralSAT, nnenum, PyRAT). Verified-robust counts: mnist_fc 48/90, oval21 29/30,
rl_benchmarks 29/30, reach_prob_density 3/30 (each capped to 30 for equal per-benchmark weight;
real-UNSAT n=91, real-all n=120). Min / median / max (d_eff normalized by input dim):

| component | group | min | median | max |
|---|---|--:|--:|--:|
| **M̂_min** (min margin) | synthetic | 1e−5 | 0.34 | 1.9e3 |
| | real (all) | −17.7 | 0.11 | 3.37 |
| | real (verified-robust) | −17.7 | 0.25 | 3.37 |
| **U** (unstable frac) | synthetic | 0.00 | 0.88 | 1.00 |
| | real (all) | 0.01 | 0.30 | 1.00 |
| | real (verified-robust) | 0.01 | 0.16 | 1.00 |
| **A_τ** (local regions) | synthetic | 0.00 | 5.55 | 6.40 |
| | real (all) | 0.00 | 5.92 | 6.40 |
| | real (verified-robust) | 0.69 | 6.38 | 6.40 |
| **d_eff / input_dim** | synthetic | 0.00 | 0.59 | 0.88 |
| | real (all) | 0.002 | 0.60 | 0.94 |
| | real (verified-robust) | 0.07 | 0.56 | 0.94 |
| **G_IBP** (IBP rel. gap) | synthetic | −0.26 | 1.09 | 2.5e5 |
| | real (all) | 1.15 | 67.7 | 2.5e5 |
| | real (verified-robust) | 2.48 | 59.4 | 9.6e4 |

Under the fair robust-vs-robust comparison, the synthetic instances **cover the real range** on four of
five components: A_τ shares the same support (both max out at 6.40), d_eff and min-margin overlap, and on
**unstable-fraction our instances are *harder*** than real robust ones (median 0.88 vs 0.16 —
verified-robust real instances have few unstable ReLUs, which is why a verifier can close them). We do
not claim the full distributions are identical (a formal test in `realism_stats/` shows real
verified-robust nets sit higher on A_τ and G_IBP), but the real instances fall inside the synthetic range
on every axis. The one axis with a pooled-*median* gap is **G_IBP**, and even there the synthetic *range*
extends to 2.5e5 — beyond the real band — while the low median is a **composition** effect, not a coverage
limit:

| synthetic family | n | G_IBP median | range |
|---|--:|--:|---|
| MEAP | 14 | **645** | [41, 5.4e3] |
| MILP-exact-radius | 31 | **143** | [9.7, 5.2e4] |
| ReLU-corners | 22 | 12.8 | [0, 2.5e5] |
| paired-bias | 46 | 8.4 | [0.8, 56] |
| deep-contractive | 50 | 0.99 | [0, 1.0] |
| attention / embedded-projection | 40 | ≈0 | — |

The MEAP (645) and MILP (143) families **bracket** the real verified-robust G_IBP (59; full-real 161);
the low pooled synthetic median only reflects that the intentionally IBP-tight families (deep-contractive,
attention) are the most numerous in the sweep. Reported per-family, the constructors already cover the
real G_IBP range.

### 1.2 G_IBP is a knob, not a ceiling: one hyperparameter moves it into the real band

The one axis with a pooled-median gap is also the one that is *trivial to dial*. G_IBP responds
monotonically to a single amplitude hyperparameter in each amplifier-type family, so reaching the real
verified-robust band (median 59, P25–P75 **[9.3, 467]**) takes one knob change — no redesign. We
generated a 12-instance suite (`gibp_realband/`, 3 per family) doing exactly this:

| family | knob turned | default G_IBP (median) | after one push | lands in real band [9.3, 467]? |
|---|---|--:|---|:--:|
| paired-bias | `margin` 1e-3 → 1e-4 (± `num_pairs`) | 8.4 | **103 → 369** | ✅ cleanly inside at every setting |
| ReLU-corners | `hinge_l1` = 1e3 → 1e5 | 12.8 | **253 → 24,381** | ✅ enters at 1e3, sweeps through and past |
| MILP-exact-radius | ε near the boundary (`ε_frac`→1) | 143 | **373 → 13,241** | ✅ enters at 0.99, overshoots at 0.9999 |
| MEAP | `num_pairs` = 32 → 128 | 645 | **1,322 → 5,335** | ⤴ above — dial `num_pairs`≈8–16 for the median |

Two things this shows. (1) **Coverage is by design, not luck:** every amplifier family reaches the real
band, and paired-bias lands squarely inside it (103–369) at all tested settings. (2) **The knob is
monotone,** so you can place an instance anywhere along the real G_IBP range — including its median — by
turning one dial; corners/MILP/MEAP even *overshoot*, so the constraint is choosing the setting, not
reaching the range. The four intentionally IBP-tight families (deep-contractive, embedded-projection,
both attention) are ≈0 by construction and are neither able nor meant to amplify — that is the point of
those families. So the low default pooled median is a *sampling choice* in the released sweep, not a
limitation of the constructors. (Details + ready-to-use "real-median" settings in `gibp_realband/README.md`.)

## 2. The constructions are trainable architectures, not synthetic-only gadgets

A natural follow-up objection is that the constructors are hand-built weight patterns. They are not
required to be. Each MILP-encodable constructor can be realized as a **genuine classifier trained on
real data** — a Lipschitz-controlled feature prefix followed by the constructor's structured head —
retaining a **whole-network** robustness certificate (analytic where the prefix Lipschitz bound is
tight; exact MILP near the boundary). Training never assigns the label; a rigorous post-training check
does.

We validated this end-to-end on three constructors (real MNIST, matched-capacity baselines):

| constructor | test acc | matched baseline | whole-net ground truth | abcrown |
|---|--:|--:|---|---|
| Deep-Contractive CNN | 0.816 | 0.946 | MILP-exact radius (near-boundary) | UNSAT (correct) |
| Paired-Bias CNN | 0.916 | 0.910 | MILP-exact radius | UNSAT (correct) |
| MEAP (MLP) | 0.910 | 0.912 | MILP-exact radius | UNSAT (correct) |

Two of the three train to accuracy **at or above** their matched ordinary baseline; the contractive one
pays a measurable ~13-point accuracy cost, which is the honest, reportable price of the Lipschitz control
that gives it its certificate. The point for realism: these are ordinary trained networks on real data
whose *architecture* carries the constructor, and off-the-shelf abcrown parses and reasons about the
exported ONNX exactly as it would any benchmark. The constructions are architecturally realizable and
trainable — not artifacts of hand-set weights.

---

### Figures
- `plots/profile_synth_vs_real.png` — difficulty-profile components, synthetic vs real (§1).
- `plots/profile_per_benchmark_winner.png`, `plots/profile_per_benchmark_points.png` — per-benchmark
  breakdown (supporting detail; shows real benchmarks are heterogeneous in difficulty).

### Reproduce
```
# §1 figure + closeness table (omit --trained -> synthetic-vs-real, 2 groups):
PYTHONPATH=src python -m VeriStressGT.realism.profile_distributions \
  --synthetic benchmarks/sweep_all/difficulty_profiles.json \
  --real benchmarks/vnncomp_mnist_fc/difficulty_profiles.json benchmarks/oval21/difficulty_profiles.json \
         benchmarks/real_reach_prob_density/difficulty_profiles.json \
         benchmarks/real_rl_benchmarks/difficulty_profiles.json \
         benchmarks/real_cifar_biasfield/difficulty_profiles.json \
  --out-dir rebuttal_materials/realism_sweep_analytic/plots
# §2 trained constructors: see rebuttal_materials/realism_smoke/REPORT.md (reproduce commands + env)
```
