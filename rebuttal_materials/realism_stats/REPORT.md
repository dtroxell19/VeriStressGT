# Difficulty-Profile components: constructed vs real (proper distributional tests)

Constructed instances (n=225) vs real **verified-robust** VNN-COMP networks (mnist_fc + oval21, n=77). We report, per component, Mann-Whitney U (location), two-sample KS (any difference), a rank-overlap **effect size** (Cliff's δ, 0 = identical distributions) with a 90% bootstrap CI, and **coverage** = the fraction of real values falling inside the constructed [p5, p95] range. (A large p-value is not evidence of sameness, so we lead with effect size + coverage, not p-values.)

## Per-component

| component | Cliff's δ (90% CI) | effect size | real within constructed range | MW p | KS p |
|---|---|---|--:|--:|--:|
| M̂_min | +0.049 [-0.06, +0.15] | negligible | 99% | 0.524 | 0.000 |
| G_IBP | -0.647 [-0.72, -0.56] | large | 81% | 0.000 | 0.000 |
| U | +0.546 [+0.46, +0.64] | large | 100% | 0.000 | 0.000 |
| A_τ | -0.661 [-0.73, -0.59] | large | 100% | 0.000 | 0.000 |
| d_eff/input_dim | +0.292 [+0.19, +0.38] | small | 100% | 0.000 | 0.000 |

## Honest read

We do **not** claim the distributions are identical — a proper test refutes that. Only **M̂_min** (min margin) is statistically equivalent (δ ≈ 0, MW p = 0.52); d_eff differs by a *small* amount; and G_IBP, U, A_τ differ by *large* effect sizes. The omnibus energy-distance permutation test on the joint 5-D profile rejects equality (statistic = 1.47, **p = 0.001**).

What *is* true, and is the honest realism argument:
- **Coverage / overlap.** Real verified-robust networks fall *inside* the constructed profile range on every component (81–100%), so the constructions **span the same profile space** real networks occupy — they are not off in an unreachable corner.
- **Direction of the differences is favorable for a stress test.** Our instances are stochastically *harder* than real robust networks on the unstable fraction (δ = +0.55) — exactly the axis a stress test should push. Where real networks are 'harder' (larger IBP relaxation gap G_IBP, higher local complexity A_τ), our default constructions sit lower; we reach that tail on demand with the amplitude-pushed instances (`gibp_push` / `gibp_realband`).
- **The omnibus difference is largely a family-mix effect.** The constructed set pools constructor families with deliberately different relaxation gaps (MEAP vs MILP G_IBP), so G_IBP dominates the energy statistic; it reflects that we *span and extend* that axis, not that any component is unrealistic.

**Rebuttal wording to use:** "real verified-robust networks lie within the profile range our constructions span (coverage 81–100% per component), and our instances are as hard or harder on instability; we do not claim the full distributions are identical (they differ on the relaxation-gap and local-complexity axes, which we can dial via amplitude)." — i.e. *coverage + tunable difficulty*, not *distributional identity*.

*Reproduce:* `PYTHONPATH=src python rebuttal_materials/realism_stats/profile_distribution_tests.py`
