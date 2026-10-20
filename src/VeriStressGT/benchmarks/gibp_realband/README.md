# gibp_realband

A small benchmark (12 instances) showing that our constructions can be made as loose under interval
bound propagation (IBP) as real networks are, by turning a single knob.

## Why this exists

One of the Difficulty Profile components is the IBP relative gap, G_IBP: how much looser the IBP
bound on the margin is than the margin actually observed inside the input box. Real verified-robust
networks have large gaps. On the `vnncomp_mnist_fc` and `oval21` networks that a verifier proved
robust (77 instances), G_IBP has a median of 267, and the middle half of them lie between 42 and
995.

Pooled over `sweep_all`, our instances have a median G_IBP of about 1. That is mostly a matter of
which families are in the mix. Four families are built to be tight under IBP: deep-contractive CNNs,
embedded projection, and both attention constructions. Their G_IBP is close to zero by design, and
they account for 90 of the 203 instances. The other four families have an amplitude knob that drives
G_IBP up. Two of them already reach the real range with the settings used in `sweep_all`: MILP
(median 143) and MEAP (median 645). The other two start lower: paired-bias, the largest family in
`sweep_all`, at 8.4, and corners at 12.8.

This benchmark takes those four families and turns their knob up, three settings each.

## The instances

| Family | Instance | Setting | G_IBP |
|---|---|---|--:|
| Paired-bias CNN | `pb_hi1` | 32 pairs, 4 backbone layers, margin 1e-3 | 103 |
| | `pb_hi2` | 64 pairs, 4 backbone layers, margin 1e-3 | 122 |
| | `pb_hi3` | 64 pairs, 4 backbone layers, margin 1e-4, δ = 0.01 | 369 |
| Corners | `rc_hi1` | `hinge_l1` = 1e3 | 253 |
| | `rc_hi2` | `hinge_l1` = 1e4 | 2,378 |
| | `rc_hi3` | `hinge_l1` = 1e5 | 24,381 |
| MILP exact radius | `milp_hi1` | hidden width 20, ε = 0.99 r* | 373 |
| | `milp_hi3` | hidden width 20, ε = 0.9999 r* | 727 |
| | `milp_hi2` | hidden width 20, ε = 0.999 r* | 13,241 |
| MEAP | `meap_hi1` | 32 pairs | 1,322 |
| | `meap_hi2` | 64 pairs | 2,672 |
| | `meap_hi3` | 128 pairs | 5,335 |

Six of the twelve land inside the real middle range (42 to 995): all three paired-bias instances,
`rc_hi1`, `milp_hi1` and `milp_hi3`. The rest overshoot it. That is deliberate: the instances
document how far each knob goes, not just where it crosses into the real range. Within each family,
G_IBP grows as the knob is turned up. The exception is MILP, where the instance at 0.999 r* has a
larger gap than the one at 0.9999 r*. Each MILP instance is a different random network (seeds 0, 1
and 2), so the three are not a clean sweep of one network.

If you want instances near the real median rather than at the extremes, these settings are a good
starting point: paired-bias with 32 pairs, 4 backbone layers and margin 1e-3; corners with
`hinge_l1` = 1000; MILP at hidden width 20 with ε close to r*; MEAP with about 16 pairs (G_IBP
around 645 in `sweep_all`).

All twelve instances are robust by construction, like every other instance in this repo. G_IBP
values are in `difficulty_profiles.json`, computed with `VeriStressGT.difficulty_profile`.

## Regenerating

The spec is `src/VeriStressGT/configs/gibp_realband.yaml`. This reproduces every `model.onnx` and
`spec.vnnlib` byte for byte (the MILP instances need gurobipy):

```bash
PYTHONPATH=src python -m VeriStressGT.cli.create_benchmark \
  --spec src/VeriStressGT/configs/gibp_realband.yaml \
  --out_dir ./gibp_realband_regen
```
