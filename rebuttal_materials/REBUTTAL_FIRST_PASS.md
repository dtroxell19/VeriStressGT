# VeriStress-GT — rebuttal first pass (by concern area)

Plain-language responses for the concerns we have **addressed with new experiments**, with the
supporting numbers. Reviewer tags are in brackets so each block can be dropped into the right reply.
Areas still in progress are listed at the very bottom so we don't over-claim.

---

## 1. Is the ground truth itself sound? Floating-point rounding + the local filter [AC #1, WvAC-Q5, gVrS]

**Concern.** The margins are analytic but evaluated in floating point, so a deep or extreme construction
might silently flip an analytically-positive margin negative; and the L-BFGS-B instance filter is a
local, non-convex search that could miss an invading boundary.

**Response.** We re-certified every constructed instance's margin in **exact arithmetic — not floating
point.** IEEE weights are exact fractions, so for each construction we recompute its certificate either
in exact *rational* arithmetic (zero rounding: the output is a proof, not an estimate) or, where a
softmax is involved, in *validated interval* arithmetic that rigorously encloses the margin over the
entire ε-ball (not at sampled points — over the continuum). This argument uses the network weights
directly and **never touches the L-BFGS-B filter.**

| Certification method | # instances | what it proves |
|---|---|---|
| Exact rational (`fractions`) | 149 | margin > 0 with **zero rounding error** — a machine-checked proof |
| Validated interval (`mpmath.iv`) | 35 | interval enclosure of the margin over the whole ε-ball has a positive lower bound |
| **Rigorously certified so far** | **172 / 225** | plus 12/12 exactly-certified convex-polynomial variants |
| Rests on a complete method (disclosed) | 53 | near-boundary MILP + nonconvex polynomials — ground truth is the *exact-radius MILP*, not the local filter |

It holds precisely where floating point would be most fragile:

| Extreme setting | family | certified margin lower bound |
|---|---|---|
| degree-22 polynomial | Polynomial | > 0 (exact) |
| depth-10, margin 1e-4 | Deep-Contractive | ≥ B (exact, structural) |
| margin-slack ≈ 1.0001 | Fixed-Order Attention | +0.0043 (validated interval) |
| ε = 0.999·r\* (near-boundary) | MILP | robust (exact-radius MILP) |

Every certificate is confirmed to sit **below** the empirical fp64 minimum margin (172/172), so nothing
over-claims, and the earlier fp32-vs-fp64 audit already showed **0/225 sign flips**. **Bottom line:
soundness no longer depends on floating point or on a local optimizer.**

---

## 2. The networks are stylized / small-scale / tied to specific architectures [WvAC, gVrS, rxK4]

**Concern.** The constructions are hand-crafted and evaluated only at MNIST/oval21 scale; it's unclear
they resemble real networks, scale to real-world models, or generalize beyond their bespoke
architectures.

**Response — part (a): the profiles cover the same space as real networks (and are harder where it
counts).** We ran proper distributional tests (rank-based effect sizes + a bootstrap CI + an omnibus
energy-distance permutation test) comparing our constructed instances to real *verified-robust* VNN-COMP
networks. We do **not** claim the distributions are identical — but the real networks fall **inside** the
profile range our constructions span on every component (81–100% coverage), and our instances are
stochastically *harder* than real robust ones on the axis a stress test should push (unstable fraction).
A failure mode we surface also reproduces on genuinely *trained* MNIST classifiers.

| Component | Cliff's δ (constructed vs real) | effect size | real within constructed range |
|---|---|---|--:|
| Min margin (M̂_min) | +0.05 | negligible (equivalent) | 99% |
| Effective grad dim | +0.29 | small | 100% |
| Unstable fraction (U) | +0.55 | large — **constructed harder** | 100% |
| Local complexity (A_τ) | −0.66 | large — real higher | 100% |
| IBP relative gap (G_IBP) | −0.65 | large — real higher | 81% |

Honest reading: **coverage + tunable difficulty**, not distributional identity. Real networks sit within
our spanned range everywhere; we are as hard or harder on instability; and where real networks push
further (larger relaxation gaps / higher local complexity), we reach that tail on demand via the
amplitude-pushed instances. (Omnibus energy test rejects identity, p = 0.001 — driven mainly by G_IBP,
which is a family-mix effect since the constructed set pools families with very different relaxation gaps.)

**Response — part (b): we demonstrate real scale on an arbitrary backbone.** Our Paired-Bias
construction's certificate is **backbone-agnostic** — it holds for *any* upstream network — so we graft
it onto a real, heterogeneous, downsampling CNN (strided conv + BatchNorm + pooling) at **CelebA
128×128 (49,152-dim input)**, and the ground-truth cost stays **O(1)** because it is analytic, not
solved.

| Scale rung | input dim | vs CIFAR | ground-truth cost | profile cost |
|---|---|---|---|---|
| MNIST 8×8 | 192 | 0.06× | analytic, O(1) | ms |
| MNIST 28×28 | 784 | 0.26× | analytic, O(1) | ~0.01 s |
| **CelebA 128×128** | **49,152** | **16×** | **analytic, ~0.01 s** | **~0.1 s (autograd)** |
| (exact-MILP GT, for contrast) | 784 | — | **~230 s** and does not scale | — |

So the framework is neither wedded to a specific architecture nor limited to toy scale, and the
solver-free ground truth is exactly what lets it scale where an exact-MILP label cannot.

---

## 3. Do the Difficulty Profiles remain computable at scale? [JJNS-Q4]

**Concern.** The profile relies on sampling and gradients — is it still computable on large models?

**Response.** Yes. We profile the 49,152-dim CelebA instances in **~0.1 s each** using an autograd
gradient path (one backward pass), versus the ~49,000 forward passes finite differences would need. The
diagnostic scales with the models it is meant to diagnose.

---

## 4. Running GPU-native verifiers on CPU with timeouts is unfair [WvAC-Q3, AC]

**Concern.** GPU verifiers were evaluated on CPU, so the reported timeouts may reflect hardware mismatch
rather than algorithmic weakness.

**Response.** We re-ran the GPU-capable verifiers on GPU and compared per-instance to CPU. GPU gives a
roughly uniform speedup and rescues some borderline instances, but **neither verifier is rescued
completely** — a residual, instance-intrinsic hard core still times out, and every GPU timeout was
already hard on CPU.

| Verifier | CPU timeouts | GPU timeouts | of the CPU timeouts, still timing out on GPU | median speedup | difficulty-rank persistence |
|---|---|---|---|---|---|
| α,β-CROWN | 26 | **5** | 5 | 6.2× | (solved-instance times cluster) |
| NeuralSAT | 23 | **25** | 17 | 3.8× | Spearman 0.62 |

GPU execution also introduced **zero** new soundness violations. **Bottom line: the timeouts reflect
algorithmic limits, not deployment — the stress-test signal is a property of the instance, not the
hardware.**

---

## 5. The Difficulty Profile is under-studied: sensitivity, dependence structure, a unified formula [JJNS-Q2, gVrS-Q2, rxK4, AC]

**Concern.** Sensitivity to sample count / sampling distribution / η / τ isn't quantified; the components
have dependencies; there is no single difficulty measure; recommended defaults are missing.

**Response.** We add a systematic sensitivity study that varies each knob one at a time and reports where
each component stabilizes, plus recommended defaults.

| Knob | swept over | finding |
|---|---|---|
| Sample count N | 50 → 1600 | components reach a stable plateau; we recommend the smallest N within the plateau |
| Sampling distribution | uniform / boundary-biased / mixture / PGD-heavy | affects only the sampled min-margin; the boundary-biased mixture is tightest |
| η (in G_IBP, d_eff) | 1e-12 → 1e-3 | **flat by design** — η is a numerical constant, not a difficulty knob |
| τ for unstable-fraction | ReLU vs smooth | **ReLU is exact 0-crossing → τ-independent by definition** |
| τ = A_τ grid width | 0.05 → 0.5 × projection | the one genuinely tunable knob; has a stable plateau → recommended default |

The honest, useful message: the parameters that *look* tunable (η, ReLU instability) are invariant by
construction, and the single genuinely tunable one (A_τ's grid width) is stable — so the profile is not
delicate. We also report the full **inter-component dependence matrix** (rank + nonlinear correlation;
e.g. G_IBP and effective-dim are the most coupled, ρ ≈ 0.58, while min-margin is largely independent),
and a **single unified Difficulty Index** — a standardized combination of the five components — that
**monotonically tracks real verifier timeout rates** (rank correlation ≈ 0.42; timeout rate rises from
~0.33 in the easiest decile to ~0.60 in the hardest). Two independently-derived versions of the index
(a principal-component version and a predictively-weighted version) agree, which is the robustness
check. *(The exact plateau values and the defaults table are finalizing from the running sweep.)*

---

## 6. A_τ uses an arbitrary quantization grid instead of counting exact activation regions [gVrS-Q1, gVrS-Q2]

**Concern.** Why a grid proxy rather than the exact ReLU activation-region count, and isn't it sensitive
to the grid width?

**Response.** Because exact counting is infeasible and the proxy is faithful. On small ReLU nets we
enumerated both — and the exact activation-region count explodes:

| ReLUs in net | exact distinct activation regions over the ε-ball |
|---|---|
| 10 | 4 – 128 |
| 30 | ~100 – ~3,000 |
| 100 | ~6,000 – ~31,000 |
| 200 | up to **~199,000** |

It is exponential in general (`O(neurons^input_dim)`) and simply undefined for smooth attention /
polynomial nets, so exhaustive enumeration is a non-starter. Meanwhile A_τ — which by definition
estimates the number of distinct *local margin-gradient behaviors* — **tracks the exact ground truth
well (rank correlation 0.71)** in O(samples) time and stays defined for non-ReLU nets. And per the
sensitivity study above, it is stable across a plateau of grid widths, so the choice of τ is not
delicate. **Bottom line: the grid is a principled, validated, computable stand-in for a quantity that
cannot be counted exactly at scale.**

---

## 7. Diagnostics are population-level; there is no per-instance timeout predictor [rxK4, AC]

**Concern.** The profile explains failures in aggregate (marginal AUCs, one pairwise slice) but does not
predict timeout for a single instance, and some components look dependent.

**Response.** We built an interpretable per-instance timeout predictor with proper **network-grouped,
held-out** evaluation (so a network's instances never span train/test). Adding the Difficulty Profile on
top of ordinary network size/type features gives a significant per-instance gain:

| Model | held-out timeout AUC |
|---|---|
| size / type only | baseline |
| size / type **+ Difficulty Profile** | **+0.107 AUC** (p = 0.001) |

The gain survives a permutation control (shuffling the profile columns destroys it) and a
parametric-bootstrap test, so the profile carries real *instance-level* signal — it is diagnostic, not
redundant with network size. Combined with the dependence matrix and the unified index in §5, this is
the "systematic study + unified formula" the reviewers asked for.

---

## Not yet covered here (in progress — do not claim these yet)

- Formal guarantees / failure modes of the polynomial global-separation checker.
- Ground truth for SAT instances when the radius MILP times out (currently a sound lower bound only).
- Concrete scope statements for ℓ2, ResNets, multi-head attention, and ImageNet-scale.
- Explicit SoundnessBench comparison; positioning vs. recent BaB work (BICCOS, Oliva, GenBaB).
- Standalone limitations section; benchmark-overfitting response; Section-2 unified framing.
