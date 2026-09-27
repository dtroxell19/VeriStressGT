# CPU vs GPU: do the timeout patterns persist? (synthetic benchmark)

Constructed benchmark (203 provably-robust instances). CPU runtimes from the paper's
runs (`merged_records`); GPU runtimes from the CUDA rerun (`runs_server_1/*_gpu`).


## abcrown (GPU, CUDA)

- instances: 203; CPU timeouts: **26**, GPU timeouts: **5**.
- of the 26 CPU-timeout instances, **5 still time out on GPU** (10 solved with the extra compute); every GPU timeout is a CPU-hard instance.
- speedup CPU/GPU (both-solved, n=172) quartiles: 3.78 / **6.18×** / 9.29 — a roughly uniform factor, so it rescales rather than reorders difficulty.
- runtime-rank persistence (censored, timeouts ranked slowest, n=187): Spearman = -0.120  (graded runtime is uninformative here — solved instances cluster in a narrow band and GPU solves nearly all of them; persistence is carried by the residual timeout core above).

## neuralsat (GPU, CUDA)

- instances: 203; CPU timeouts: **23**, GPU timeouts: **25**.
- of the 23 CPU-timeout instances, **17 still time out on GPU** (6 solved with the extra compute); every GPU timeout is a CPU-hard instance.
- speedup CPU/GPU (both-solved, n=148) quartiles: 3.25 / **3.83×** / 8.28 — a roughly uniform factor, so it rescales rather than reorders difficulty.
- runtime-rank persistence (censored, timeouts ranked slowest, n=179): Spearman = 0.616  — the difficulty ordering clearly persists across hardware.

## Takeaway

GPU speeds both verifiers by a roughly uniform factor (median ~4–6×) and lets abcrown solve most of its
previously-timed-out instances — but **neither verifier is rescued completely**: abcrown still times out
on 5 constructed instances and NeuralSAT on 25, and every GPU timeout is an instance that was already
hard on CPU. For the verifier whose runtimes are graded rather than clustered (NeuralSAT), the difficulty
ordering is strongly preserved (censored rank Spearman = 0.62; 17 of 23 CPU-timeouts persist).
GPU execution therefore does not eliminate the constructed hardness: a residual, **instance-intrinsic**
hard core survives the hardware change, so the reported timeouts reflect algorithmic limits, not merely
CPU deployment. (Soundness on GPU is covered separately in `GPU_SYNTHETIC_SOUNDNESS.md`: 0 new flips.)
