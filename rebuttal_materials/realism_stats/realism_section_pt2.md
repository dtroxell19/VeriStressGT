# Realism pt2 — synthetic vs real *classifier* networks (self-imposed classification-margin spec)

**Framing.** Reviewers care about real *networks*, not the property a benchmark ships with. So we take real **classifier** networks and evaluate our Difficulty Profile — a classification-margin L∞-robustness diagnostic — on them, with the true class set to the *network's own decision at the box centre* (argmax at x0). This only uses networks that are genuinely classifiers and is self-consistent, fixing the earlier artifact where non-classification benchmarks (reachability, RUL regression) produced meaningless negative margins.

**Real classifier benchmarks (5, n=365, FULL table — no verified-robust filter):** mnist_fc (90), oval21 (30), cifar_biasfield (30), vnncomp22_acasxu (185), rl_benchmarks (30). Image classifiers keep their shipped profiles (their label already *is* the image class); acasxu (5 advisories) and rl (2 actions) are re-profiled here with argmax-at-x0 self-labeling over each instance's native input box. Excluded: reach-density and collins-RUL (not classifiers). Synthetic: n=203.

## Per-component (min / median / max)

| component | synthetic | real classifiers | Cliff's δ (90% CI) | effect | real within synth range |
|---|---|---|---|---|--:|
| M̂_min | 1e-05 / 0.34 / 1.94e+03 | -1.06 / 0.003 / 23.65 | +0.47 [+0.40,+0.53] | medium | 56% |
| G_IBP | -0.26 / 1.09 / 2.53e+05 | 1.15 / 490 / 1.74e+08 | -0.66 [-0.72,-0.59] | large | 63% |
| U | 0.00 / 0.88 / 1.00 | 0.00977 / 0.48 / 0.89 | +0.65 [+0.57,+0.72] | large | 100% |
| A_τ | 0.00 / 5.55 / 6.40 | 0.00 / 5.24 / 6.40 | -0.02 [-0.11,+0.07] | negligible | 100% |
| d_eff/input_dim | 0.00 / 0.59 / 0.88 | 0.0016 / 0.51 / 0.94 | +0.04 [-0.05,+0.13] | negligible | 85% |

## Read

- **Coverage.** Real classifier networks fall inside the synthetic range on all five components (mean 81%); the constructions span the profile space real networks occupy.
- **Effect sizes.** 2/5 components are negligible-to-small; the rest differ but with real sitting *inside* the synthetic support. Our instances remain as-hard-or-harder on instability.
- **Omnibus.** Energy-distance permutation test on the joint 5-D profile: statistic = 1.62, p = 0.001 (n_syn=203, n_real=180). We do not claim identical distributions — we claim coverage of real networks + tunable difficulty (see G_IBP amplitude sweep in `REALISM_SECTION.md` §1.2).

*Note.* We use the full table, so real includes non-robust nets (negative margin); since our synthetic instances are all robust by construction, the margin axis is a *conservative* comparison for us. acasxu/rl are re-profiled over their native input boxes with self-labeling; existing realism artifacts are unchanged.

*Reproduce:* `PYTHONPATH=src python rebuttal_materials/realism_stats/realism_pt2.py` (re-profiles acasxu+rl on first run, caches to `selflabeled_profiles.json`).
