# UCLA TA1 Phase 1 evaluation submission

**Evaluation card:** `cards/thrust1_soundness.yaml`
**Repository:** https://github.com/dtroxell19/VeriStressGT (public), branch `evaluation-card`
**Summary:** `docs/aiq_phase1_summary.pdf` (approach, BAA metric alignment, expected results, scalability)

## Setup
Linux x86_64 or macOS; needs git, conda (installed if missing) and a C/C++ toolchain.
```bash
git clone https://github.com/dtroxell19/VeriStressGT.git
cd VeriStressGT
git checkout evaluation-card
bash scripts/bootstrap.sh            # conda envs: VeriStressGT, alpha-beta-crown, pyrat, nnenum, neuralsat, marabou
conda activate VeriStressGT
python scripts/check_aiq_setup.py    # each real verifier on one UNSAT + one SAT instance; all should print OK
```
`bootstrap.sh` installs MAGNET v0.1.0 and pins Marabou 2.0.0 (PyPI wheel on Linux x86_64). No
environment variables are needed. All code, models and the benchmark are in the public repository.

## Run
From the repository root:
```bash
magnet evaluate cards/thrust1_soundness.yaml
```
About 30 min on a laptop CPU; no GPU needed.

## Expected result
- `RESULT: VERIFIED`
- Metrics in the top-level `verdict.json`: "Scoring accuracy (ground truth)" = 1.0 (the BAA metric,
  target >= 0.95) and "Buggy verifiers caught (ground truth)" = 1.0.
- Verdict totals (about 2,025 definitive verdicts, 39 of 42 planted bugs firing) can shift slightly
  with machine speed; Marabou 2.0.0 is nondeterministic, so which of its known false SAT answers
  appear can vary. Neither changes the verdict or the metrics.

## Scaling
Set `scale: N` under the card's `algo_params` to build and run an N-times-larger benchmark
(fresh seeds, labels re-derived from construction). Expected ground-truth accuracy stays 1.0.

## Troubleshooting
- A real verifier not working stops the run with an error naming it; `python scripts/check_aiq_setup.py`
  shows which and why. `allow_missing_verifiers: True` in `algo_params` scores without it.
- Full details: `AIQ_README.md`.

## Contact and availability
David Troxell (davidtroxell@g.ucla.edu). Unavailable for two weeks after submission, back before
preliminary results on 10/30.
