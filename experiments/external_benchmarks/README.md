# Which planted bugs can an external benchmark catch?

The Thrust 1 verifier pool (2 sound controls, 42 planted bugs, 5 real verifiers with counterexample
search, 60 s timeout) run on VeriStressGT's `thrust1_bench` and on two VNN-COMP benchmarks.
External benchmarks ship no ground truth, so they get the only labels available without our
constructions: SAT where a strong PGD attack finds a counterexample that certifies in float64 and
onnxruntime float32, and no label elsewhere (robustness cannot be certified from outside).

- `mnist_fc`: 30 of its 90 instances, 10 evenly spaced per network (256x2/4/6, both eps 0.03 and
  0.05), to keep the laptop run under 2 h
- `oval21`: all 30 instances (3 CIFAR CNNs)

> **Subset note.** The `mnist_fc` numbers below use 30 of 90 instances. More instances can only
> make more bugs fire and add certified counterexamples, so this subset may understate what full
> `mnist_fc` can catch. The reproduce script defaults to every instance
> (`bash scripts/run_external_comparison.sh`, about 4-5 h); `PER_NETWORK=10` reproduces the run
> reported here.

Each benchmark is scored with the best labels it has: its own labels, plus majority vote on the
instances it has no label for (VNN-COMP style). Majority-vote catches on unlabelled instances are
known to be right here only because we planted the bugs; on a real benchmark nobody can tell
whether the flagged verifier or the majority is wrong, and on VeriStressGT majority vote blamed
three sound verifiers.

**Real finding:** unmodified Marabou 2.0.0 answers UNSAT on `oval21` instance `vb_oval21_0014`
(`cifar_wide_kw`, image 1909, eps 0.0034), which is SAT: the certified counterexample gives class 5 a lead of
0.0027 over the true class 3. All six other verifiers answer SAT.

## Reproduce
```bash
# every instance (default; mnist_fc 90 + oval21 30)
bash scripts/run_external_comparison.sh <thrust1 card run>/results.json
# the 30-instance mnist_fc subset reported here
PER_NETWORK=10 bash scripts/run_external_comparison.sh <thrust1 card run>/results.json
```
Needs the VNN-COMP 2022 benchmark checkout for the `.onnx` files (`VNNCOMP_DIR`, default
`~/vnncomp2022_benchmarks/benchmarks`). Steps: `aiq/build_external_bench.py` (labels),
`aiq/thrust1_runner.py` (verifier pool), `aiq/compare_benchmarks.py` (tables).
Laptop run times for the reported run (MacBook Pro, Apple Silicon): mnist_fc (30) 67 min, oval21 (30) 104 min.

## Bugs caught

| Benchmark | Instances (UNSAT / SAT / no label) | Bugs that fire | Caught by own labels | Caught by own labels + majority vote |
|---|---|---|---|---|
| VeriStressGT | 50 (24 / 26 / 0) | 39/42 | **39/42** | 39/42 |
| mnist_fc (30/90) | 30 (0 / 5 / 25) | 19/42 | **6/42** | 14/42 |
| oval21 | 30 (0 / 1 / 29) | 19/42 | **2/42** | 9/42 |

Out of all planted bugs: a bug that never fires on a benchmark cannot be caught there. Majority-vote catches on unlabelled instances are known to be right only because the bugs were planted.

## How verdicts can be judged

| Benchmark | SAT/UNSAT verdicts | Judged by own labels (certain) | Judged only by majority vote (unverifiable) | Not judged (tied vote) |
|---|---|---|---|---|
| VeriStressGT | 2025 | **2025/2025 (100%)** | 0/2025 (0%) | 0/2025 (0%) |
| mnist_fc (30/90) | 953 | **199/953 (21%)** | 754/953 (79%) | 0/953 (0%) |
| oval21 | 729 | **49/729 (7%)** | 680/729 (93%) | 0/729 (0%) |

## Per planted bug

`-` never fired · `L` caught by the benchmark's labels · `L+M` caught by its labels plus majority vote · `M1` / `M` caught by majority vote alone (one buggy / full pool) · `fired, missed` changed verdicts that nothing could expose

| Planted bug | Fails as | VeriStressGT | mnist_fc (30/90) | oval21 |
|---|---|---|---|---|
| `abcrown+attention_unsat_to_sat` | false_sat | - | - | - |
| `abcrown+cache_collision` | false_unsat | L L+M M1 M | - | - |
| `abcrown+conv_sat_to_unsat` | false_unsat | L L+M M1 M | - | - |
| `abcrown+crash_as_sat` | false_sat | L L+M M1 M | - | - |
| `abcrown+flaky` | false_unsat | L L+M M1 M | L L+M M1 M | L+M M1 M |
| `abcrown+timeout_as_unsat` | false_unsat | L L+M M1 M | L+M M1 M | fired, missed |
| `marabou+attention_unsat_to_sat` | false_sat | L L+M M1 M | - | - |
| `marabou+cache_collision` | false_unsat | L L+M M1 M | - | - |
| `marabou+conv_sat_to_unsat` | false_unsat | L L+M M1 M | - | - |
| `marabou+crash_as_sat` | false_sat | L L+M M | - | L+M M1 M |
| `marabou+flaky` | false_unsat | L L+M M1 M | L+M M1 M | L+M M1 M |
| `marabou+timeout_as_unsat` | false_unsat | L L+M M1 M | L L+M M1 M | fired, missed |
| `mutant:bab_any_row` | false_unsat | L L+M M1 M | fired, missed | fired, missed |
| `mutant:conv_bias_dropped` | false_unsat | L L+M M1 M | fired, missed | fired, missed |
| `mutant:disjunct_last` | false_unsat | L L+M M1 M | fired, missed | L L+M M1 M |
| `mutant:eps_half` | false_unsat | L L+M M1 M | L L+M M1 M | L L+M M1 M |
| `mutant:hwc_layout` | false_unsat | L L+M M1 M | - | L+M M1 M |
| `mutant:ibp_sign` | false_unsat | L L+M M1 M | L+M M1 M | fired, missed |
| `mutant:input_clip01` | false_unsat | L L+M M1 M | - | L+M M1 M |
| `mutant:label_off_by_one` | false_sat | L L+M M1 M | L+M M1 M | L+M M1 M |
| `mutant:relu_no_intercept` | false_unsat | L L+M M1 M | L+M M1 M | fired, missed |
| `mutant:tol_sat` | false_sat | L L+M M1 M | fired, missed | - |
| `mutant:tol_unsat` | false_unsat | L L+M M1 M | - | - |
| `mutant:weights_fp16` | false_unsat | L L+M M1 M | - | fired, missed |
| `neuralsat+attention_unsat_to_sat` | false_sat | L L+M M1 M | - | - |
| `neuralsat+cache_collision` | false_unsat | L L+M M1 M | - | - |
| `neuralsat+conv_sat_to_unsat` | false_unsat | L L+M M1 M | - | - |
| `neuralsat+crash_as_sat` | false_sat | L L+M M | L+M M | - |
| `neuralsat+flaky` | false_unsat | L L+M M1 M | L L+M M1 M | L+M M1 M |
| `neuralsat+timeout_as_unsat` | false_unsat | L L+M M1 M | fired, missed | fired, missed |
| `nnenum+attention_unsat_to_sat` | false_sat | - | - | - |
| `nnenum+cache_collision` | false_unsat | L L+M M1 M | - | - |
| `nnenum+conv_sat_to_unsat` | false_unsat | L L+M M1 M | - | - |
| `nnenum+crash_as_sat` | false_sat | L L+M M1 M | - | - |
| `nnenum+flaky` | false_unsat | L L+M M1 M | L+M M1 M | - |
| `nnenum+timeout_as_unsat` | false_unsat | L L+M M1 M | L L+M M1 M | fired, missed |
| `pyrat+attention_unsat_to_sat` | false_sat | L L+M M1 M | - | - |
| `pyrat+cache_collision` | false_unsat | L L+M M1 M | - | - |
| `pyrat+conv_sat_to_unsat` | false_unsat | L L+M M1 M | - | - |
| `pyrat+crash_as_sat` | false_sat | - | - | - |
| `pyrat+flaky` | false_unsat | L L+M M1 M | L L+M M1 M | - |
| `pyrat+timeout_as_unsat` | false_unsat | L L+M M1 M | L+M M1 M | fired, missed |

## Real verifiers flagged by each benchmark's labels

- **VeriStressGT:** marabou (meap_02, meap_03, milp_s2_a)
- **mnist_fc (30/90):** none
- **oval21:** marabou (vb_oval21_0014)
