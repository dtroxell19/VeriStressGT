# VeriStress-GT

Benchmark framework for neural network verification. Generates provably robust (ONNX, VNNLIB) pairs with known ground-truth properties, then runs third-party verifiers on them in a standardized way.

## Setup

**Requires:** Python 3.9+, git. Optional: CUDA GPU, [Gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/) (for MILP construction), MATLAB (for NNV verifier).

```bash
git clone --recursive https://github.com/dtroxell19/VeriStressGT.git
cd VeriStressGT
bash scripts/bootstrap.sh
conda activate VeriStressGT
source .env
cd src/VeriStressGT
```

The bootstrap script handles everything: installs Miniconda if needed, pulls submodules, installs Python packages (`VeriStressGT[generate,milp,analysis]`), creates verifier conda environments, and writes your `.env`. Marabou 2.0.0 is built from source on Apple Silicon (30–60 min); skip it with `SKIP_MARABOU=1 bash scripts/bootstrap.sh`. If you just want instance generation without verifier setup: `pip install -e '.[generate]'`

Optional extras: `milp` (gurobipy), `analysis` (sound re-certification, soundness evaluation, plots), `experiments` (everything under `experiments/`), `dev`, `all`.

**Run `VeriStressGT-doctor` at any time to check what's working and how to fix installation**

## Usage

**Generate a benchmark:**
Define instances in a YAML spec (see `src/VeriStressGT/configs/` for examples). Then generate:

```bash
python3 -m VeriStressGT.cli.create_benchmark \
  --spec configs/test_benchmark.yaml \
  --out_dir ./benchmarks/test_benchmark \
  --overwrite
```

**Run a verifier on it** (no env switching needed — conda isolation is automatic):

```bash
python3 -m VeriStressGT.cli.verify_benchmark \
  --benchmark ./benchmarks/test_benchmark \
  --verifier abcrown \
  --out_dir ./runs/abcrown_run \
  --timeout 300 \
  --abcrown_config configs/abcrown_basic.yaml \
  --overwrite
```

Results are written to `results.jsonl` and `summary.json`/`summary.csv`. Each instance gets stdout/stderr logs in the `logs/` subdirectory. On timeout the verifier's whole process group is killed; `--max_memory_gb` caps its memory. Use `configs/abcrown_gpu.yaml` for GPU runs of α-β-CROWN.

To run a different verifier on the same benchmark:

```bash
python -m VeriStressGT.cli.verify_benchmark \
  --benchmark ./benchmarks/my_benchmark \
  --verifier neuralsat \
  --out_dir ./runs/neuralsat_run \
  --timeout 300
```

Results go to `results.jsonl` and `summary.csv`.

## Shipped benchmarks

Under `src/VeriStressGT/benchmarks/`, each with a `manifest.json` and `instances/<id>/{model.onnx, spec.vnnlib, meta.json}`:

| Benchmark | Instances | Contents |
|---|---|---|
| `sweep_all` | 203 | All constructor families across their difficulty knobs (spec: `configs/sweep_all.yaml`) |
| `polynomial_stress_22` | 22 | Polynomial networks (spec: `configs/polynomial_stress_22.yaml`) |
| `gibp_realband` | 12 | Amplifier-type families pushed into the IBP-gap band of real verified-robust networks ([README](src/VeriStressGT/benchmarks/gibp_realband/README.md)) |
| `soundness_gt` | 50 | 24 analytic-UNSAT + 26 witness-certified SAT instances for soundness evaluation |
| `vnncomp_mnist_fc`, `oval21`, `cifar2020`, `vnncomp22_acasxu` | 90 / 30 / 203 / 186 | VNN-COMP 2022 comparison sets: specs, metadata and difficulty profiles; the ONNX models come from the VNN-COMP 2022 benchmark repository (each manifest records its source) |

`src/VeriStressGT/runs_standardized/` holds standardized verifier runs (results and logs) on these benchmarks.

## Available constructions

| Name | Type | What it stresses |
|------|------|-----------------|
| `mlp_relu.embedded_projection` | MLP | Baseline provably-robust via input projection |
| `mlp_relu.meap` | MLP | Mutually exclusive activation patterns |
| `mlp_relu.corners` | MLP | Convex corner-based certificate |
| `mlp_relu.milp.exact_radius` | MLP | MILP-computed exact robustness radius |
| `cnn.cnn_paired_bias` | CNN | Exploits CROWN's independent neuron relaxation |
| `cnn.deep_contractive_cnn` | CNN | Easy for CROWN, hard for CDCL verifiers |
| `attention.fixed_pattern` | Attention | Softmax attention with pattern stability certificate |
| `attention.linear_dominance` | Attention | Linear attention with key-dominance certificate |
| `polynomial.algebraic_boundary` | Polynomial Net | Small margin. `certified_convex=True` gives an analytic certificate (`configs/polynomial_convex_certified.yaml`); the nonconvex mode labels robustness with a numerical multi-start screen (`configs/polynomial_nonconvex_screen.yaml`) |

`src/VeriStressGT/realism/` adds trained MNIST-family classifiers (contractive CNN, paired-bias CNN, MEAP MLP, and a paired-bias head for arbitrary backbones) whose architecture keeps these certificates valid after training.

## Ground truth you can check

- **Sound re-certification.** `VeriStressGT.analysis.sound_backends` / `sound_families` recompute each instance's certified margin from its fp32 weights in exact-rational or validated-interval arithmetic; see [`experiments/sound_arithmetic`](experiments/sound_arithmetic/REPORT.md). `python -m VeriStressGT.analysis.soundness_audit` compares fp32 and fp64 forward passes.
- **Soundness evaluation.** `VeriStressGT-soundness` runs a verifier pool — real verifiers, a sound in-house reference verifier (`--verifier refverify`), and planted-bug mutants — on `soundness_gt` and reports which bugs the labels expose. See [docs/soundness_evaluation.md](docs/soundness_evaluation.md).
- **Difficulty profiles.** `python -m VeriStressGT.difficulty_profile.cli` computes the five-component Difficulty Profile; opt-in flags (`--eta`, `--atau-width`, `--u-tau`, `--u-mode`, `--seed`, ...) expose its hyperparameters without changing the defaults.

## Experiments

[`experiments/`](experiments/README.md) holds the supporting studies: timeout prediction from the Difficulty Profile, profile sensitivity, A_τ vs exact activation regions, realism against real networks, scale / arbitrary-architecture, CPU vs GPU, and sound re-certification. Install with `pip install -e '.[experiments]'`.

## Verifier setup

All handled by `bootstrap.sh`, or individually via `make install-<verifier>`:

| Verifier | GPU | Setup |
|----------|-----|-------|
| α-β-CROWN | ✓ | `make install-abcrown` |
| NeuralSAT | ✓ | `make install-neuralsat` (own conda env `neuralsat`; `.env` sets `NEURALSAT_CONDA_ENV`) |
| nnenum | ✗ | `make install-nnenum` |
| Marabou | ✗ | `make install-marabou-py` (maraboupy 2.0.0 in conda env `marabou`; `.env` sets `MARABOU_BIN=scripts/marabou_env.sh`), or `make install-marabou` (compiled binary, requires cmake) |
| NNV | ✗ | `make install-nnv` (requires MATLAB) |
| PyRAT | ✗ | `make install-pyrat` |

## Adding a new constructor

Place a module under `src/VeriStressGT/robust_constructions/` with:

```python
CONSTRUCTION_NAME = "category.my_method"

def add_args(parser):
    parser.add_argument("--my_param", type=float, default=1.0)

def run(args):
    # Build model, export to args.onnx_path and args.vnnlib_path
    ...
```

Discovery is automatic. Subdirectories need an `__init__.py`.

## Repo structure

```
src/VeriStressGT/
    cli/                    create_benchmark, verify_benchmark, doctor, soundness_eval
    robust_constructions/   mlp_relu/, cnn/, attention/, polynomial/
    realism/                trained constructors with certificates
    difficulty_profile/     Difficulty Profile computation
    analysis/               sound re-certification, soundness audit, paper analysis scripts
    soundness/              reference verifier, planted bugs, ground-truth labelling, scoring
    verifier_adapters/      per-verifier CLI adapters
    verifiers/              git submodules for each verifier
    benchmarks/             shipped benchmarks (see above)
    configs/                benchmark specs and verifier configs
    runs_standardized/      standardized verifier runs (results + logs)
    utils/                  ONNX export + VNNLIB generation
experiments/                supporting studies (see experiments/README.md)
docs/                       soundness evaluation guide
tests/                      unit tests
```
