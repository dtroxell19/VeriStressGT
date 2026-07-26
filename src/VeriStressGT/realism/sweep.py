"""Sweep runner for the realistic trained constructors (A/B/C).

Config-driven, multi-dataset, multi-hyperparameter. For each (constructor, dataset, hyperparameter
combo): train a genuine classifier, then for a few correctly-classified test inputs compute the exact
MILP radius r* (OPTIMAL only) and emit near-boundary instances at eps = frac * r*. Every emitted
instance is a trained realistic net with an exact-MILP whole-network ground-truth label
(hardness_class MILP_GROUND_TRUTH), plus the native-certificate tightness recorded.

Output: a verify_benchmark-compatible benchmark dir (manifest.json + instances/) — the "realistic nets"
analogue of sweep_all. Use fewer than 225 instances.

Usage:
    PYTHONPATH=src python -m VeriStressGT.realism.sweep --config configs/realism_sweep.yaml \
        --out-dir rebuttal_materials/realism_sweep --data-dir /tmp/realism_data
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

from .contractive_cnn import IMG, train_contractive, native_certificate as cert_A
from .paired_bias_cnn import train_paired, native_certificate as cert_B
from .meap_trained import train_meap, native_certificate as cert_C
from .data import load_dataset_8x8
from .hardness_gate import (
    contractive_to_layers, sequential_to_layers, exact_radius_milp, witness_min_margin, classify,
)
from .export import export_onnx, parity_max_err, write_vnnlib
from ..robust_constructions.mlp_relu.milp.exact_radius import forward_layers


def _sha(p: Path) -> str:
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


# per-constructor adapters
def _handler(kind: str):
    if kind == "contractive_cnn":
        return dict(train=lambda dd, data, hp, seed: train_contractive(dd, data=data, seed=seed, **hp),
                    layers=lambda res: contractive_to_layers(res.model, IMG),
                    export=lambda res: res.model, cert=lambda res, x0, y, e: cert_A(res.model, x0, y, e))
    if kind == "paired_bias_cnn":
        return dict(train=lambda dd, data, hp, seed: train_paired(dd, data=data, seed=seed, **hp),
                    layers=lambda res: sequential_to_layers(list(res.export_model.seq), (1, IMG, IMG)),
                    export=lambda res: res.export_model, cert=lambda res, x0, y, e: cert_B(res.model, x0, y, e))
    if kind == "meap":
        return dict(train=lambda dd, data, hp, seed: train_meap(dd, data=data, seed=seed, **hp),
                    layers=lambda res: sequential_to_layers(list(res.export_model.seq), (IMG * IMG,)),
                    export=lambda res: res.export_model, cert=lambda res, x0, y, e: cert_C(res.model, x0, y, e))
    raise ValueError(kind)


def _grid(g: Dict) -> List[Dict]:
    keys = list(g.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*[g[k] for k in keys])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these constructor kinds")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out_dir); (out / "instances").mkdir(parents=True, exist_ok=True)
    datasets = cfg["datasets"]
    rfr = cfg.get("robust_fracs", [0.9, 0.99]); nfr = cfg.get("nonrobust_fracs", [1.01])
    n_inputs = int(cfg.get("instances_per_model", 2)); seed = int(cfg.get("seed", 0))

    data_cache: Dict[str, tuple] = {}
    manifest_instances: List[Dict] = []
    model_records: List[Dict] = []
    inst_ctr = 0

    for kind, spec in cfg["constructors"].items():
        if args.only and kind not in args.only:
            continue
        H = _handler(kind); combos = _grid(spec["grid"])
        epochs = int(spec.get("epochs", 6))
        milp = spec.get("milp", {}); rmax = float(milp.get("rmax", 0.2))
        tlim = float(milp.get("time_limit", 60)); gap = float(milp.get("mip_gap", 1e-3))
        in_shape = (IMG * IMG,) if kind == "meap" else (1, IMG, IMG)
        for ds in datasets:
            if ds not in data_cache:
                data_cache[ds] = load_dataset_8x8(ds, args.data_dir)
            data = data_cache[ds]
            _, (Xte, yte) = data
            for ci, hp in enumerate(combos):
                t0 = time.time()
                res = H["train"](args.data_dir, data, hp, seed)
                em = H["export"](res); layers = H["layers"](res)
                mrec = {"kind": kind, "dataset": ds, "hp": hp, "test_acc": res.test_acc,
                        "baseline_acc": res.baseline_acc, "train_s": round(time.time() - t0, 1),
                        "n_optimal": 0, "n_emitted": 0}
                model = res.model
                # pick correctly-classified test inputs
                picked = []
                for i in range(len(yte)):
                    xi = Xte[i].reshape(1, 1, IMG, IMG).double()
                    if int(model(xi).argmax()) == int(yte[i]):
                        picked.append(i)
                    if len(picked) >= n_inputs:
                        break
                for i in picked:
                    x0 = Xte[i].reshape(1, IMG, IMG).double()
                    r = exact_radius_milp(layers, x0.reshape(-1).numpy(), Rmax=rmax,
                                          time_limit=tlim, mip_gap=gap, domain=(0, 1))
                    if r["status"] != "OPTIMAL" or not r["r_star"] or not np.isfinite(r["r_star"]):
                        continue
                    mrec["n_optimal"] += 1; rstar = float(r["r_star"]); y = int(yte[i])
                    for frac in list(rfr) + list(nfr):
                        eps = frac * rstar; robust = frac < 1.0
                        cert = H["cert"](res, x0, y, eps)
                        wit = witness_min_margin(em, x0.reshape(1, 1, IMG, IMG), eps, y, domain=(0, 1))
                        hc, src = classify(cert["analytical_lower_bound"], r_star=rstar, eps=eps,
                                           U_wit=wit, floor=1e-7)
                        iid = f"{kind[:4]}_{ds[:4]}_c{ci}_i{i}_f{frac}"
                        idir = out / "instances" / iid; idir.mkdir(parents=True, exist_ok=True)
                        export_onnx(em, str(idir / "model.onnx"), in_shape)
                        perr = parity_max_err(em, str(idir / "model.onnx"), x0, eps, in_shape)
                        write_vnnlib(x0, eps, y, 10, str(idir / "spec.vnnlib"))
                        gt = {"kind": kind, "dataset": ds, "hp": hp, "label": y, "epsilon": eps,
                              "r_star": rstar, "epsilon_frac": frac, "is_robust": robust,
                              "hardness_class": hc, "label_source": src, "parity_max_err": perr,
                              "analytical_lower_bound": cert["analytical_lower_bound"],
                              "certificate_gap": (rstar - eps if robust else eps - rstar) - cert["analytical_lower_bound"],
                              "test_acc": res.test_acc, "baseline_acc": res.baseline_acc}
                        (idir / "meta.json").write_text(json.dumps(
                            {"id": iid, "construction": f"realism.{kind}", "is_robust": robust,
                             "paths": {"onnx": "model.onnx", "vnnlib": "spec.vnnlib", "meta": "meta.json"},
                             "gt": gt}, indent=2))
                        manifest_instances.append(
                            {"id": iid, "construction": f"realism.{kind}", "is_robust": robust,
                             "paths": {"onnx": f"instances/{iid}/model.onnx",
                                       "vnnlib": f"instances/{iid}/spec.vnnlib",
                                       "meta": f"instances/{iid}/meta.json"}, "gt": gt})
                        mrec["n_emitted"] += 1; inst_ctr += 1
                model_records.append(mrec)
                print(f"[{kind}/{ds}/c{ci} {hp}] acc={res.test_acc:.3f} base={res.baseline_acc:.3f} "
                      f"opt={mrec['n_optimal']} emitted={mrec['n_emitted']} ({mrec['train_s']}s)", flush=True)

    (out / "manifest.json").write_text(json.dumps(
        {"name": cfg.get("name", "realism_sweep"), "n_instances": len(manifest_instances),
         "datasets": datasets, "instances": manifest_instances}, indent=2))
    (out / "sweep_summary.json").write_text(json.dumps({"models": model_records,
                                                        "n_instances": len(manifest_instances)}, indent=2))
    print(f"\nDONE: {len(model_records)} models, {len(manifest_instances)} instances -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
