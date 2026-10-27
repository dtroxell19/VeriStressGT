"""One-off correction of G_IBP for the attention instances in results/axis_{N,dist,eta,seed}.csv.

These rows were produced before the Difficulty Profile's IBP handled MatMul of two varying tensors
(attention Q.K^T and A.V), so their IBP lower bound was too tight. The bound is deterministic per
instance, so from the published G = (m - lb_old) / (m + eta) we recover the sampled minimum margin
m = (lb_old + G * eta) / (1 - G) and recompute G with the fixed bound. Every other sampled value
is kept as published (the sweep driver is not bit-reproducible for softmax attention).

Already applied to the committed CSVs; running it again would correct twice. Run it only on the
pre-fix CSVs, from experiments/profile_sensitivity:  python src/fix_attention_gibp.py
"""
from pathlib import Path
import pandas as pd

# ibp_margin_lb per instance: before / after the IBP fix (VeriStressGT.difficulty_profile)
LB_OLD = {"ld9": 0.614675983786583, "ld10": 0.4261612892150879, "ld11": 0.5172526240348816,
          "fp9": 1.7875699996948242, "fp10": 0.630872368812561}
LB_NEW = {"ld9": 0.40530063211917877, "ld10": 0.2843800112605095, "ld11": 0.2852822542190552,
          "fp9": 1.724870353937149, "fp10": 0.5773493647575378}

for axis in ["N", "dist", "eta", "seed"]:
    path = Path("results") / f"axis_{axis}.csv"
    df = pd.read_csv(path, float_precision="round_trip")
    sel = df.instance_id.isin(list(LB_OLD)) & (df.component == "g_ibp")
    g, eta = df.loc[sel, "value"], df.loc[sel, "eta"].astype(float)
    inst = df.loc[sel, "instance_id"]
    m = (inst.map(LB_OLD) + g * eta) / (1 - g)
    assert (m > 0).all(), axis
    df.loc[sel, "value"] = ((m - inst.map(LB_NEW)) / (m + eta)).values
    df.to_csv(path, index=False)
    print(f"axis {axis}: {int(sel.sum())} g_ibp rows corrected")
