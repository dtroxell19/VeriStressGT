"""Trained ("realistic") constructors with sound certificates.

Real classifiers (MNIST / FashionMNIST / KMNIST via torchvision) whose architecture keeps the
paper's ground-truth certificates valid after training: deep-contractive CNN, paired-bias CNN,
MEAP MLP, and a backbone-agnostic paired-bias head. ``common_bounds`` holds the shared sound
bounds, ``hardness_gate`` the witness / exact-MILP tightness check, ``export`` ONNX + VNNLIB export.
Needs the ``experiments`` extra (torchvision).
"""
