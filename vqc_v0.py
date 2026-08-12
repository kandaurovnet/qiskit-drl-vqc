"""
Minimal VQC (Variational Quantum Circuit) using Qiskit and PyTorch.

Specifications:
- 4 qubits and 4 input features
- A single RY data-encoding layer without data re-uploading
- Trainable RZ and RY layers
- Selectable linear or circular CX entanglement
- A TorchConnector wrapper that makes the QNN directly trainable as a
  PyTorch nn.Module

This is the minimal teaching example. The Q-network actually used by the
CartPole agent lives in vqc.py, which adds data re-uploading, two observables,
and trainable input/output scaling.

Two gate-ordering choices below are load-bearing, not stylistic. Both were
verified by comparing finite-difference gradients on a direct statevector
simulation against the gradients TorchConnector reports:

- Entanglement must be circular. A purely linear CX chain runs q0 -> q1 ->
  q2 -> q3, so qubit 0 is never a target. Measuring Z on qubit 0 then leaves
  the backward light cone unable to reach the upper qubits: x[2] and x[3]
  have literally no effect on the output, and 11 of the 16 trainable
  parameters have exactly zero gradient.
- Each layer applies RZ before RY, so the last rotation before measurement
  does not commute with the observable. RZ commutes with any Z-type
  observable, so a trailing RZ is a dead parameter.

Two parameters (theta[8], theta[9]) remain dead even after both fixes. That is
structural: with a single local observable, the final layer's rotations on
qubits outside its light cone cannot matter. vqc.py resolves it by using two
observables with complementary light cones.

Installation:
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
"""

import sys

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

torch.manual_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
N_QUBITS = 4
N_LAYERS = 2                 # Set to 1 to test a shallower circuit.
ENTANGLE_MODE = "circular"   # Either "linear" or "circular"; see module docstring.
CIRCUIT_IMAGE = "vqc_v0_circuit.png"


def build_vqc_circuit(n_qubits: int, n_layers: int, entangle_mode: str):
    """Build a VQC with RY encoding and trainable RZ-RY entangling layers."""
    x_params = ParameterVector("x", n_qubits)
    # Use an ASCII name so qc.draw() also works in Windows cp950 terminals.
    theta_params = ParameterVector("theta", n_layers * n_qubits * 2)

    qc = QuantumCircuit(n_qubits)

    # 1. Apply one data-encoding layer without data re-uploading.
    for i in range(n_qubits):
        qc.ry(x_params[i], i)

    # 2. Apply trainable RZ and RY rotations followed by entanglement. RY comes
    #    last so the final rotation does not commute with the Z observable.
    idx = 0
    for _ in range(n_layers):
        for i in range(n_qubits):
            qc.rz(theta_params[idx], i)
            idx += 1
            qc.ry(theta_params[idx], i)
            idx += 1
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        if entangle_mode == "circular":
            qc.cx(n_qubits - 1, 0)

    return qc, x_params, theta_params


class VQCModel(nn.Module):
    """Wrap an EstimatorQNN as a standard PyTorch module."""

    def __init__(self, qnn: EstimatorQNN):
        super().__init__()
        init_weights = np.random.uniform(-np.pi, np.pi, size=len(qnn.weight_params))
        self.qlayer = TorchConnector(qnn, initial_weights=init_weights)

    def forward(self, x):
        raw = self.qlayer(x)      # Pauli-Z expectation value in [-1, 1].
        return (raw + 1) / 2      # Map the output to [0, 1] as a probability.


def main():
    # ---------------- Build the circuit ----------------
    qc, x_params, theta_params = build_vqc_circuit(N_QUBITS, N_LAYERS, ENTANGLE_MODE)
    circuit_text = str(qc.draw())
    terminal_encoding = sys.stdout.encoding or "utf-8"
    print(circuit_text.encode(terminal_encoding, errors="replace").decode(terminal_encoding))

    # Keep the documented circuit figure in sync with the code that built it.
    qc.draw("mpl", filename=CIRCUIT_IMAGE, fold=-1)
    print(f"Circuit diagram written to {CIRCUIT_IMAGE}\n")

    observable = SparsePauliOp.from_list([("IIIZ", 1)])  # Measure qubit 0.

    qnn = EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=list(x_params),
        weight_params=list(theta_params),
    )

    model = VQCModel(qnn)

    # ---------------- Generate a synthetic dataset ----------------
    # Assign label 1 when sum(x) > 0, otherwise label 0. This simple linearly
    # separable task verifies that the circuit can learn.
    N = 40
    X = np.random.uniform(-np.pi, np.pi, size=(N, N_QUBITS))
    Y = (X.sum(axis=1) > 0).astype(np.float32)

    X_train = torch.tensor(X, dtype=torch.float32)
    Y_train = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

    # ---------------- Training loop ----------------
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.BCELoss()

    n_epochs = 30
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        preds = model(X_train)
        loss = loss_fn(preds, Y_train)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            acc = ((preds > 0.5).float() == Y_train).float().mean().item()
            print(f"Epoch {epoch:2d} | Loss: {loss.item():.4f} | Acc: {acc:.2f}")


if __name__ == "__main__":
    main()
