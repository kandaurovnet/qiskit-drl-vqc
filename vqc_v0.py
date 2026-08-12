"""
Minimal VQC (Variational Quantum Circuit) using Qiskit and PyTorch.

Specifications:
- 4 qubits and 4 input features
- A single RY data-encoding layer without data re-uploading
- Trainable RY and RZ layers
- Selectable linear or circular CX entanglement
- A TorchConnector wrapper that makes the QNN directly trainable as a
  PyTorch nn.Module

Installation:
    pip install qiskit qiskit-machine-learning qiskit-aer torch --break-system-packages
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
ENTANGLE_MODE = "linear"     # Either "linear" or "circular".


def build_vqc_circuit(n_qubits: int, n_layers: int, entangle_mode: str):
    """Build a VQC with RY encoding and trainable RY-RZ entangling layers."""
    x_params = ParameterVector("x", n_qubits)
    # Use an ASCII name so qc.draw() also works in Windows cp950 terminals.
    theta_params = ParameterVector("theta", n_layers * n_qubits * 2)

    qc = QuantumCircuit(n_qubits)

    # 1. Apply one data-encoding layer without data re-uploading.
    for i in range(n_qubits):
        qc.ry(x_params[i], i)

    # 2. Apply trainable RY and RZ rotations followed by entanglement.
    idx = 0
    for _ in range(n_layers):
        for i in range(n_qubits):
            qc.ry(theta_params[idx], i)
            idx += 1
            qc.rz(theta_params[idx], i)
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
