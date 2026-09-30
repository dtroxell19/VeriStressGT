> **Phase 1 submission summary (PDF):** [docs/aiq_phase1_summary.pdf](docs/aiq_phase1_summary.pdf)
> covers the approach, the comparison with existing benchmarks, expected results, BAA alignment
> and scalability.

## AIQ Flow

**Phase 1 submission card: `cards/thrust1_soundness.yaml`** (other cards are for
self-evaluation).

### Setup
Needs git and a C/C++ toolchain; conda is installed if missing. Tested on macOS
```bash
git clone https://github.com/dtroxell19/VeriStressGT.git
cd VeriStressGT
git checkout evaluation-card
bash scripts/bootstrap.sh            # envs: VeriStressGT, alpha-beta-crown, pyrat, nnenum, neuralsat, marabou
conda activate VeriStressGT
python scripts/check_aiq_setup.py    # runs every real verifier on one UNSAT + one SAT instance
```
`bootstrap.sh` installs MAGNET pinned to `v0.1.0` into the `VeriStressGT`
env, and each verifier into its own conda env. The runners locate envs by name, and the
α-β-CROWN checkout by default, so no environment variables are needed.

**Marabou** is pinned to 2.0.0 (`scripts/install_marabou.sh`). PyPI has 2.0.0 wheels only for
Linux x86_64 and Intel macOS; on Apple Silicon and Linux aarch64 the script builds the `v2.0.0`
tag from source (30-60 min). The Intel wheel crashes under Rosetta on macOS 14, so it is not used.

**Gurobi.** `gurobipy` (pip, free size-limited license) is needed only to rebuild or scale the
benchmark; the committed `thrust1_bench/` runs without it.

**macOS + MAGNET v0.1.0.** Its job scripts run `chmod +x -- <file>`, which macOS `chmod` rejects,
so `magnet evaluate` fails on macOS unless a `chmod` that accepts `--` is first on `PATH`
(e.g. GNU coreutils)

### Run
```bash
magnet evaluate cards/thrust1_soundness.yaml
```
**Expected:** `RESULT: VERIFIED`, and in the top-level `verdict.json` both metrics at 1.0
("Scoring accuracy (ground truth)", "Buggy verifiers caught (ground truth)"). Takes about 30 min on a laptop
CPU.

**Metrics** (declared with `define_metric`, reported in the top-level `verdict.json`):
`gt_scoring_accuracy` (fraction of all definitive verdicts that ground truth judges correctly;
the BAA accuracy metric) and `gt_detection_rate` (fraction of exercised planted bugs caught).

### Scaling
Set `scale: N` in the card's `algo_params`. The runner builds (once) a benchmark with `N`
re-seeded copies of every base network, each with its analytic certificate, plus `N` times the
random and attack-hard CNN SAT instances, in `thrust1_bench_xN/`, and runs the same pool on it.
Every label is re-derived from construction, so the claim should hold unchanged at any scale.

### Results (Thrust 1)

Reference run on a MacBook Pro (Apple Silicon, 8 GB, CPU only), MAGNET v0.1.0, all five real
verifiers: **VERIFIED** in 32 min.

| Metric | Result |
|---|---|
| Scoring accuracy (BAA metric) | **2025/2025 verdicts (100%)** |
| Correct verdicts wrongly rejected / wrong verdicts wrongly accepted | 0 / 0 |
| Buggy verifiers caught | **39/39 fired bugs (100%)** (42 planted; 3 never fired) |
| Sound verifiers wrongly flagged | 0 |

Ground-truth SAT labels are re-certified from their witnesses on every run, so the 100% is
checked, not assumed.

**Compared with existing benchmarks** (same pool on VNN-COMP `mnist_fc`, 30/90 instances, and
`oval21`, each scored with the best labels it has; details in
[experiments/external_benchmarks](experiments/external_benchmarks/README.md)):

| Benchmark | Bugs that fire | Caught by own labels | Caught by own labels + majority vote | Verdicts judged with certainty |
|---|---|---|---|---|
| VeriStressGT | 39/42 | **39/42** | 39/42 | **100%** |
| `mnist_fc` (30/90) | 19/42 | 6/42 | 14/42 | 21% |
| `oval21` | 19/42 | 2/42 | 9/42 | 7% |

Real finding: unmodified Marabou 2.0.0 gives wrong answers on provably robust and provably
non-robust instances. Benchmark, verifier pool, scoring rules and all findings:
[docs/thrust1_details.md](docs/thrust1_details.md).

### Evaluation cards

| Card | Thrust | Metric | Details |
|------|--------|--------|---------|
| `cards/thrust1_soundness.yaml` | 1 (**submission**) | Scoring accuracy and buggy verifiers caught, with ground-truth labels | [docs/thrust1_details.md](docs/thrust1_details.md) |
| `cards/thrust2_timeout_prediction.yaml` | 2 | Per-verifier timeout-prediction AUC (target >= 0.7) | [docs/thrust2_timeout_prediction.md](docs/thrust2_timeout_prediction.md) |
| `cards/evaluation.yaml` | 2 (supporting) | Mini sweep: per-verifier correctness + per-component timeout AUC | [docs/mini_sweep.md](docs/mini_sweep.md) |
