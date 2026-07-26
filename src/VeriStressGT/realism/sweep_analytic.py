"""MILP-FREE sweep for the realistic trained constructors.

Each constructor uses a LIPSCHITZ-CONTROLLED prefix + its structured head (contractive: linear head;
paired-bias / MEAP: structured head). Because the prefix Lipschitz constant is small and known, the
native whole-network certificate is TIGHT and yields a positive certified radius eps_cert ANALYTICALLY
-- no MILP. Instances are emitted at eps = frac * eps_cert, all robust by the analytic certificate
(hardness_class = TIGHT_NATIVE, label_source = native_certificate).

The induced-norm budget lambda is the accuracy<->difficulty knob (small lambda: robust, tighter cert,
lower accuracy; large lambda: higher accuracy, cert further inside the true boundary). We sweep it.

Output: a verify_benchmark-compatible benchmark of trained realistic nets on real-world data, built
WITHOUT any verifier or MILP -- so difficulty profiles can be compared to synthetic + real before any
verification.

Usage:
    PYTHONPATH=src python -m VeriStressGT.realism.sweep_analytic --config configs/realism_sweep_analytic.yaml \
        --out-dir rebuttal_materials/realism_sweep_analytic --data-dir /tmp/realism_data
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

from .contractive_cnn import IMG, train_contractive, native_certificate as cert_A
from .paired_bias_cnn import train_paired, native_certificate as cert_B
from .meap_trained import train_meap, native_certificate as cert_C, certified_radius_native
from .data import load_dataset
from .export import export_onnx, parity_max_err, write_vnnlib


def _handler(kind: str):
    if kind == "contractive_cnn":
        return dict(train=lambda dd, data, hp, seed: train_contractive(dd, data=data, seed=seed, **hp),
                    export=lambda res: res.model, cert=cert_A)
    if kind == "paired_bias_cnn":
        return dict(train=lambda dd, data, hp, seed: train_paired(dd, data=data, seed=seed, **hp),
                    export=lambda res: res.export_model, cert=cert_B)
    if kind == "meap":
        return dict(train=lambda dd, data, hp, seed: train_meap(dd, data=data, seed=seed, **hp),
                    export=lambda res: res.export_model, cert=cert_C)
    raise ValueError(kind)


def _grid(g: Dict) -> List[Dict]:
    keys = list(g.keys())
    return [dict(zip(keys, v)) for v in itertools.product(*[g[k] for k in keys])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out_dir); (out / "instances").mkdir(parents=True, exist_ok=True)
    fracs = cfg.get("robust_fracs", [0.5, 0.9, 0.99]); n_inputs = int(cfg.get("instances_per_model", 3))
    seed = int(cfg.get("seed", 0)); min_eps = float(cfg.get("min_eps_cert", 1e-4))
    data_cache: Dict[str, tuple] = {}
    manifest: List[Dict] = []; models: List[Dict] = []

    for kind, spec in cfg["constructors"].items():
        if args.only and kind not in args.only:
            continue
        H = _handler(kind); cert_fn = H["cert"]
        in_shape = (1, IMG, IMG)
        for ds in cfg["datasets"]:
            if ds not in data_cache:
                data_cache[ds] = load_dataset(ds, args.data_dir, IMG)
            data = data_cache[ds]; _, (Xte, yte) = data
            fixed = {k: v for k, v in spec.items() if k != "grid"}
            for ci, combo in enumerate(_grid(spec["grid"])):
                hp = {**fixed, **combo}
                res = H["train"](args.data_dir, data, hp, seed); em = H["export"](res)
                model = res.model
                picked = [i for i in range(len(yte))
                          if int(model(Xte[i].reshape(1, 1, IMG, IMG).double()).argmax()) == int(yte[i])][:n_inputs]
                n_emit = 0
                for i in picked:
                    x0 = Xte[i].reshape(1, IMG, IMG).double(); y = int(yte[i])
                    eps_cert = certified_radius_native(model, x0, y, cert_fn)   # ANALYTIC, no MILP
                    if eps_cert <= min_eps:
                        continue
                    for frac in fracs:
                        eps = float(frac) * eps_cert
                        iid = f"{kind[:4]}_{ds[:4]}_c{ci}_i{i}_f{frac}"
                        idir = out / "instances" / iid; idir.mkdir(parents=True, exist_ok=True)
                        export_onnx(em, str(idir / "model.onnx"), in_shape)
                        perr = parity_max_err(em, str(idir / "model.onnx"), x0, eps, in_shape)
                        write_vnnlib(x0, eps, y, 10, str(idir / "spec.vnnlib"))
                        gt = {"kind": kind, "dataset": ds, "hp": hp, "label": y, "epsilon": eps,
                              "eps_cert": eps_cert, "epsilon_frac": frac, "is_robust": True,
                              "hardness_class": "TIGHT_NATIVE", "label_source": "native_certificate",
                              "parity_max_err": perr, "test_acc": res.test_acc, "baseline_acc": res.baseline_acc}
                        (idir / "meta.json").write_text(json.dumps(
                            {"id": iid, "construction": f"realism.{kind}", "is_robust": True,
                             "paths": {"onnx": "model.onnx", "vnnlib": "spec.vnnlib", "meta": "meta.json"},
                             "gt": gt}, indent=2))
                        manifest.append({"id": iid, "construction": f"realism.{kind}", "is_robust": True,
                                         "paths": {"onnx": f"instances/{iid}/model.onnx",
                                                   "vnnlib": f"instances/{iid}/spec.vnnlib",
                                                   "meta": f"instances/{iid}/meta.json"}, "gt": gt})
                        n_emit += 1
                models.append({"kind": kind, "dataset": ds, "hp": hp, "test_acc": res.test_acc,
                               "baseline_acc": res.baseline_acc, "n_emitted": n_emit})
                print(f"[{kind}/{ds}/c{ci} {hp}] acc={res.test_acc:.3f} base={res.baseline_acc:.3f} "
                      f"emitted={n_emit}", flush=True)

    (out / "manifest.json").write_text(json.dumps(
        {"name": cfg.get("name", "realism_sweep_analytic"), "n_instances": len(manifest),
         "datasets": cfg["datasets"], "instances": manifest}, indent=2))
    (out / "sweep_summary.json").write_text(json.dumps({"models": models, "n_instances": len(manifest)}, indent=2))
    print(f"\nDONE: {len(models)} models, {len(manifest)} instances (MILP-free) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
