# GPU verifier runs on the synthetic benchmark — did any verdict flip?

**Experiment.** A reviewer asked for GPU (not just CPU) verifier runs. We re-ran the two
GPU-capable verifiers, **α,β-CROWN (abcrown)** and **NeuralSAT**, on all **203** constructed
synthetic instances on GPU (CUDA), 600 s budget. Every instance is **provably robust**, so the
correct verdict is **UNSAT**; a **SAT** verdict would be an unsound "counterexample" (a soundness
flip). The ground truth itself is numerically sound — the float64 audit finds **0 / 225 sign flips**
(see `soundness_audit.json`), so any SAT here would be the *verifier's* error, not ours.

## The only direction that matters
Ground truth is UNSAT for every instance, so the only possible soundness flip is
**robust (UNSAT) → SAT**. We report how many occurred.

## Result — no soundness flips introduced on GPU

| Verifier (GPU) | UNSAT (correct) | TIMEOUT | Error/Unknown | **SAT (soundness flip)** |
|---|---|---|---|---|
| abcrown   | 182 | 5  | 16 | **0** |
| NeuralSAT | 154 | 25 | 23 | **1** (`milp7`) |

- **abcrown: 0 flips.** No instance flipped robust→SAT. (16 runs did not produce a verdict — they
  ended in runtime errors; these were initially *mis-reported* as SAT by a result-parsing bug, now
  fixed so a crash can never masquerade as a counterexample. They are counted as errors, not flips.)
- **NeuralSAT: 1 flip (`milp7`).** NeuralSAT reports a counterexample on this single instance. It is
  **not GPU-induced**: the same instance is already SAT under NeuralSAT on **CPU**, so moving to GPU
  changed nothing here. (The instance is float64-robust; the returned point warrants a float64
  counterexample re-check, tracked with the CPU soundness analysis.)

## Bottom line
Moving abcrown and NeuralSAT to GPU produced **zero new soundness flips**: abcrown flipped **0 / 203**
robust instances to SAT, and NeuralSAT's only SAT (`milp7`, 1 / 203) is a pre-existing CPU result, not
a GPU artifact. GPU execution did not make either verifier unsound on the constructed benchmark.

*Data:* `rebuttal_materials/runs_server_1/sweep_all_{abcrown,neuralsat}_gpu/results.jsonl`.
*(Raw abcrown SAT count before the parser fix was 16; all 16 were runtime errors with no counterexample,
now classified as errors.)*
