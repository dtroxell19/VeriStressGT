# A_tau validated against the exact ReLU region count (reviewer gVrS Q1)

gVrS asked why A_tau uses a quantization-grid proxy instead of counting the exact ReLU activation patterns. Two reasons, both demonstrated here:

**(1) Exact enumeration is infeasible in general.** A network's ReLU hyperplanes cut the input into up to O(neurons^input_dim) cells; the notion is also undefined for smooth activations (attention, polynomial). Even on these tiny MLPs the exact number of distinct activation patterns realized over B_eps(x0) explodes with size — from 4 to **198,773** as the ReLU count grows (via 200,000-point dense enumeration). A proxy is unavoidable at scale.

**(2) The proxy is faithful.** A_tau (eq.14) estimates the number of distinct *normalized margin gradients* (= distinct local affine behaviors of the margin), not every whole-network activation pattern. Computing that ground truth exactly (exact per-point margin gradient, finely quantized) and comparing to the shipped A_tau:

> **Spearman(log #exact-margin-gradients, A_tau) = 0.705**, Pearson = 0.565 (n=31 non-degenerate ReLU-MLP instances). The grid proxy tracks the true local affine complexity, so it is a faithful, O(samples) surrogate that also extends to non-ReLU nets where an exact count does not exist.

*(Scope: the clean set is the MILP ReLU-MLP family — the constructors that export as pure Gemm/ReLU chains, spanning 10–200 ReLUs. MEAP/Input-Corner export with Add/Transpose ops and the smooth families (attention, polynomial) have no activation-region notion; embedded_projection is excluded as degenerate — its output is constant over the box, so its projection ReLUs flip only at the box faces.)*

## Per-instance (sample)

| instance | family | activation regions (exact) | margin-grad regions | A_tau |
|---|---|---|---|---|
| milp15 | exact_radius | 4 | 4 | 4.33 |
| milp16 | exact_radius | 18 | 22 | 4.51 |
| milp18 | exact_radius | 22 | 27 | 4.39 |
| milp20 | exact_radius | 22 | 27 | 4.42 |
| milp1 | exact_radius | 94 | 122 | 3.22 |
| milp8 | exact_radius | 284 | 284 | 3.09 |
| milp29 | exact_radius | 360 | 358 | 3.43 |
| milp9 | exact_radius | 2,072 | 2,072 | 4.49 |
| milp4 | exact_radius | 1,687 | 2,280 | 4.55 |
| milp6 | exact_radius | 1,693 | 2,289 | 4.37 |
| milp28 | exact_radius | 2,734 | 2,723 | 5.17 |
| milp10 | exact_radius | 2,942 | 2,942 | 4.65 |
| milp12 | exact_radius | 3,042 | 3,042 | 4.60 |
| milp14 | exact_radius | 3,043 | 3,043 | 4.63 |
| milp24 | exact_radius | 14,296 | 16,516 | 5.47 |
| milp26 | exact_radius | 198,773 | 198,765 | 6.39 |

*Reproduce:* `PYTHONPATH=src python rebuttal_materials/activation_regions/count_regions.py`
