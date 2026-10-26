"""IBP soundness in the Difficulty Profile: the interval bounds must contain every value the
network actually takes on the input box, including for products of two varying tensors
(attention scores Q K^T and the weighted sum A V).

Run: PYTHONPATH=src pytest tests/test_profile_ibp.py   (or: PYTHONPATH=src python tests/test_profile_ibp.py)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from VeriStressGT.difficulty_profile.components import _interval_matmul_both, estimate_ibp_components
from VeriStressGT.difficulty_profile.instance_loader import load_instance

LD4 = Path(__file__).resolve().parents[1] / "src" / "VeriStressGT" / "benchmarks" / "sweep_all" / "instances" / "ld4"


def test_interval_matmul_both_encloses_samples():
    rng = np.random.default_rng(0)
    a_lo = rng.normal(size=(2, 4, 5)); a_hi = a_lo + rng.uniform(0, 1, size=a_lo.shape)
    b_lo = rng.normal(size=(2, 5, 3)); b_hi = b_lo + rng.uniform(0, 1, size=b_lo.shape)
    lo, hi = _interval_matmul_both(a_lo, a_hi, b_lo, b_hi)
    for _ in range(2000):
        a = a_lo + rng.uniform(size=a_lo.shape) * (a_hi - a_lo)
        b = b_lo + rng.uniform(size=b_lo.shape) * (b_hi - b_lo)
        y = a @ b
        assert np.all(y >= lo - 1e-9) and np.all(y <= hi + 1e-9)
    print("ok interval matmul of two varying operands encloses 2000 samples")


def test_attention_ibp_lower_bound_is_sound():
    """On ld4 (dominant-key attention) the IBP margin lower bound must not exceed any margin
    sampled inside the box. Before the fix it did (0.2197 vs. a sampled 0.1750)."""
    inst = load_instance(str(LD4 / "model.onnx"), str(LD4 / "spec.vnnlib"))
    lb = estimate_ibp_components(inst, verbose=False)["ibp_margin_lb"]
    import onnxruntime as ort
    sess = ort.InferenceSession(str(LD4 / "model.onnx"), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [1] + [d if isinstance(d, int) else 1 for d in inp.shape[1:]]
    x0 = inst.x0.detach().cpu().numpy().reshape(-1)
    eps = float(inst.epsilon)
    rng = np.random.default_rng(0)
    xs = x0 + eps * rng.uniform(-1, 1, size=(4000, x0.size))
    xs = np.concatenate([xs, x0 + eps * np.sign(rng.normal(size=(1000, x0.size)))])
    y = inst.true_class
    margins = []
    for x in xs.astype(np.float32):
        out = sess.run(None, {inp.name: x.reshape(shape)})[0].reshape(-1)
        margins.append(out[y] - np.max(np.delete(out, y)))
    assert lb <= min(margins) + 1e-5, (lb, min(margins))
    print(f"ok ld4 IBP lower bound {lb:.4f} <= sampled min margin {min(margins):.4f}")


if __name__ == "__main__":
    test_interval_matmul_both_encloses_samples()
    test_attention_ibp_lower_bound_is_sound()
    print("all IBP tests passed")
