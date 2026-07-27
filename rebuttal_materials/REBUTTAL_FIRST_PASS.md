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

**Response — part (b): we demonstrate real scale on an arbitrary trained backbone.** Our Paired-Bias
construction's certificate is **backbone-agnostic** — it holds for *any* upstream network — so we graft
it onto a real, heterogeneous, downsampling CNN (strided conv + BatchNorm + pooling) that is **genuinely
trained on real images** (CIFAR-10 upsampled to 128×128, held-out test acc 0.44) at **49,152-dim input**,
and the ground-truth cost stays **O(1)** because it is analytic, not solved.

| Scale rung | input dim | vs 32² image | ground-truth cost | profile cost |
|---|---|---|---|---|
| MNIST 8×8 | 192 | 0.06× | analytic, O(1) | ms |
| MNIST 28×28 | 784 | 0.26× | analytic, O(1) | ~0.01 s |
| **trained CNN @ 128×128** | **49,152** | **16×** | **analytic, ~0.01 s** | **~0.05 s (autograd)** |
| (exact-MILP GT, for contrast) | 784 | — | **~230 s** and does not scale | — |

The head grafts onto a downsampling trunk it was never co-designed with, over a frozen *trained* feature
extractor, and every instance is analytically robust with all pairs unstable near x0 (a genuine stress
test) — ONNX/ORT parity ≤ 6.4e-8. So the framework is neither wedded to a specific architecture nor
limited to toy scale, and the solver-free ground truth is exactly what lets it scale where an exact-MILP
label cannot.

---

## 3. Do the Difficulty Profiles remain computable at scale? [JJNS-Q4]

**Concern.** The profile relies on sampling and gradients — is it still computable on large models?

**Response.** Yes. We profile the 49,152-dim instances (real trained CIFAR-10 backbone at 128×128) in
**~0.05 s each** using an autograd gradient path (one backward pass), versus the ~49,000 forward passes
finite differences would need. The diagnostic scales with the models it is meant to diagnose.

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

| Knob | swept over | finding | recommended |
|---|---|---|---|
| Sample count N | 50 → 1600 | M̂_min, G_IBP, d_eff converge by **N=50**; A_τ needs more (within 8% by N=800) | N=1600 (50 for all but A_τ) |
| Sampling distribution | uniform / boundary / mixture / PGD-heavy | small effect (swings ≤ 0.04); uniform-only under-estimates hardness | boundary-biased mixture |
| η (in G_IBP, d_eff) | 1e-12 → 1e-3 | **flat for η ≤ 1e-6** (d_eff 35.8→35.6, G_IBP 444→444); only an absurd η=1e-3 moves it | η=1e-9 |
| τ for unstable-fraction | ReLU vs smooth | **ReLU: exact 0-crossing, τ-independent** (U=0.740 for all τ, both tests) | ω_j>τ, τ=1e-2 |
| τ = A_τ grid width | 0.05 → 0.5 × proj | stable across τ ∈ [0.05, 0.2] and proj ∈ {5,10,20}; drifts only at coarse τ=0.5 | τ=0.1, proj=10 |
| random seed | 16 seeds | run-to-run CoV < 1% (U/G_IBP essentially deterministic) | ≥ 8-seed mean |

The honest, useful message: the parameters that *look* tunable (η, and the ReLU unstable-fraction
threshold) are invariant by construction — which the sweep confirms empirically — and the single
genuinely tunable knob (A_τ's grid width) sits on a stable plateau, so the profile is not delicate. We
also report the full **inter-component dependence matrix** (rank + distance correlation; G_IBP and
effective-dim are the most coupled, ρ = 0.58, A_τ–d_eff = 0.46, while min-margin is largely independent,
|ρ| ≤ 0.36), and a **single unified Difficulty Index** — a standardized combination of the five
components — that **monotonically tracks real verifier timeout rates** (Spearman 0.40–0.42; timeout rate
rises from ~0.33 in the easiest decile to ~0.60 in the hardest). Two independently-derived versions (a
principal-component index and a predictively-weighted index) agree, which is the robustness check.

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

## 8. Which constructors yield scalable SAT instances with controllable proximity to violation? [gVrS, AC]

**Concern.** The framework emphasizes provably-robust (UNSAT) instances; which constructions naturally
produce *SAT* (non-robust) instances whose closeness to the decision boundary can be dialed, and do those
scale?

**Response.** Three constructors emit a genuine in-box counterexample (SAT) with an explicit
proximity-to-violation knob, and they trade off between *exactness* of that proximity and *scalability*.

The **polynomial (algebraic-boundary) constructor is the most natural *scalable* SAT generator.** Its
decision surface is an explicit algebraic variety, and the construction records the exact L∞ distance from
the center x0 to that surface in closed form (`known_boundary_dist_linf` / `exact_boundary_dist`). So
proximity to violation is not something you search for — you *set* it: place x0 at any signed distance from
the boundary and sweep the query radius ε across it, and the instance flips from robust to non-robust at a
point you already know analytically. Because it only needs cheap forward evaluations of a polynomial, this
works at arbitrary input dimension and arbitrary degree — the same construction that gives a 5-dim toy also
gives a high-dimensional one — and the constructor exposes an explicit `non_robust` status (asserted when
the minimum signed margin over the box drops below zero). This is the family we would point a reviewer to
for "give me SAT instances of increasing size, each a controllable hair past the boundary."

**MILP exact-radius gives the tightest possible proximity control, but does not scale.** It solves for the
*true* minimal-violation radius r\* of a ReLU network and then sets ε relative to it (`epsilon_mode`:
`frac`·r\*, exactly r\*, or 1.01·r\*), so ε/r\* is the *certified-exact* normalized distance to violation —
there is no looseness at all in how close to violation the instance sits. The price is that r\* comes from a
MILP solve, so this is confined to small ReLU networks. It is the right tool when you want a handful of
instances whose proximity is exact and provable rather than merely analytic, and it is exactly the
constructor whose non-scaling motivates the solver-free families.

**Input-corner (ReLU corners) is a lightweight analytic middle ground.** The margin is a closed form over
the input box, dominated by the box corners, so a SAT instance is produced simply by widening the box until
the corner where the margin changes sign is pulled inside it; proximity to violation is the gap between the
box edge and that sign-flip corner. Like the polynomial family it needs no solver and scales, but its
control is coarser (corner-quantized) than the polynomial's continuous boundary distance.

For contrast, the robust-by-construction families (Paired-Bias, MEAP, Deep-Contractive, attention) are the
**UNSAT side** of the benchmark: their certificate *is* margin > 0, so they are not the natural SAT source.
They can be driven right up to the boundary (ε → r) but not naturally pushed past it with controlled
proximity — that is what the three constructors above are for.

---

## 9. How far do the constructors scale (params, input dim — ImageNet backbones, long-sequence attention), and do Difficulty Profiles remain computable there? [JJNS-Q4, WvAC]

**Concern.** Do the constructions and their profiles hold up at real parameter counts and input dimensions
— e.g. Paired-Bias on ImageNet-scale backbones, attention with long sequences?

**Response.** For the head-based families the ground-truth cost is **O(1) in backbone size and input
dimension**, so scaling is a property of the certificate, not a hope.

- **Paired-Bias / MEAP heads — GT cost independent of the backbone.** The certificate is analytic (label
  row hard-zeroed, monotone coupled-ReLU pair gaps ⇒ margin ≥ *margin* > 0 for **any** frozen upstream Ψ),
  so it neither solves nor inspects the backbone. We demonstrated this on a real *trained* CNN at
  **49,152-dim (128²×3)** with **~0.01 s/instance** GT regardless of the downsampling trunk (§2b). Because
  that cost does not grow with backbone parameters, the same head composes with an **ImageNet-scale**
  backbone (e.g. ResNet-50, ~25 M params, or a ViT) at 224²×3 ≈ **150k-dim** — ~3× our demonstrated input —
  with the *same* O(1) analytic GT. This is extrapolation on a cost that is provably constant, not on a hope
  that a solver keeps up.
- **Attention — cheap analytic certificate, sequence length affects *tightness* not soundness.** The gap
  condition (1−μ > bound) is closed-form and cheap at any sequence length, but longer sequences make the
  near-orthogonal token structure harder (μ grows, the usable margin shrinks) and widen the softmax
  interval enclosure. So long sequences remain *certifiable* but at a smaller usable ε — a disclosed
  tightness limit, orthogonal to the O(1) GT of the head families.
- **MILP exact-radius does not scale** (solver-bound, small ReLU nets) — which is precisely why the analytic
  certificate families exist: solver-free ground truth scales where an exact-label MILP cannot.

**Do the Difficulty Profiles remain computable at that scale? Yes.** The autograd gradient path computes
each gradient in **one backward pass — O(1) in input dimension** — versus the ~input_dim forward passes
finite differences need. We profiled the 49,152-dim instances in **~0.05 s each** (§3); the IBP components
(G_IBP, U) are a single interval forward pass, and the sampled components (A_τ, d_eff, M̂_min) are N
backward passes. The diagnostic scales with the models it is meant to diagnose.

*Honest scope.* We ran end-to-end up to 49k-dim and modest attention sequence lengths; ResNet/ViT-on-
ImageNet and long-context attention are extrapolations justified by the constant certificate cost, not yet
each executed end-to-end (see the scope list below).

---

## Not yet covered here (in progress — do not claim these yet)

- Formal guarantees / failure modes of the polynomial global-separation checker.
- Ground truth for SAT instances when the radius MILP times out (currently a sound lower bound only).
- Concrete scope statements for ℓ2, ResNets, multi-head attention, and ImageNet-scale.
- Explicit SoundnessBench comparison; positioning vs. recent BaB work (BICCOS, Oliva, GenBaB).
- Standalone limitations section; benchmark-overfitting response; Section-2 unified framing.
