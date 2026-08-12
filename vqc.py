"""
VQC Q-network for CartPole-v1, built with Qiskit and PyTorch.

This is the quantum half of the DRL+VQC agent. It exposes a single
``nn.Module`` that maps a batch of CartPole states to a batch of Q-values:

    (batch, 4) observations  ->  (batch, 2) Q-values

That is the only seam the DQN half needs, which also means the execution
backend can change later without touching any RL code.

Architecture (following Skolik, Jerbi & Dunjko, arXiv:2103.15084):

- 4 qubits, one per CartPole observation feature.
- Hadamard layer, then ``N_LAYERS`` blocks of
  ``[RZ, RY] per qubit -> circular CZ ring -> data re-uploading encoding``.
- A final ``[RZ, RY]`` variational layer.
- Two observables, ``Z0Z1`` and ``Z2Z3``, one per action.
- Trainable input scaling ``lambda`` and output scaling ``w``.

Four of these choices are load-bearing, and all four were verified by
finite-difference gradients on a direct statevector simulation:

1. *Circular* entanglement. A purely linear CX/CZ chain runs q0 -> q1 -> q2 ->
   q3, so qubit 0 is never a target and the backward light cone of a local
   observable cannot reach the upper qubits.
2. Each rotation pair applies RZ *before* RY, so the last rotation before
   measurement does not commute with the Z-type observable. A trailing RZ is
   always a dead parameter.
3. *Two* observables rather than one. With a single local observable the final
   layer's rotations on out-of-light-cone qubits are structurally dead. Z0Z1
   and Z2Z3 have complementary light cones, so their union covers all four
   qubits and every parameter carries gradient for at least one output.
4. Trainable output scaling. CartPole Q-values under gamma=0.99 reach ~100,
   but an expectation value is confined to [-1, 1]. Without ``w`` the network
   cannot represent its own targets. This is the most common reason a naive
   VQC-DQN silently fails to learn.

Measured dead-parameter counts (3 seeds; a parameter counts as dead only if
its gradient vanishes for *every* output):

    linear CX,   RY->RZ, obs Z0            16 params, 11 dead
    circular CX, RY->RZ, obs Z0            16 params,  5 dead
    circular CX, RZ->RY, obs Z0            16 params,  2 dead
    circular CZ, RZ->RY, obs Z0Z1 + Z2Z3   32 params,  4 dead   (RZ last)
    circular CZ, RZ->RY, obs Z0Z1 + Z2Z3   32 params,  0 dead   (RY last)

Installation:
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
"""

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import BackendEstimatorV2, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
N_QUBITS = 4        # One per CartPole observation feature.
N_LAYERS = 3        # Variational blocks; each re-uploads the input.
N_ACTIONS = 2       # CartPole: push left / push right.
CIRCUIT_IMAGE = "vqc_circuit.png"
TRANSPILED_IMAGE = "vqc_circuit_transpiled.png"


def build_qnetwork_circuit(n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS):
    """Build the data re-uploading VQC.

    Returns ``(circuit, encoding_params, weight_params)``. The encoding
    parameters are ordered ``layer-major`` (layer 0 qubit 0, layer 0 qubit 1,
    ...) so a flattened ``(batch, n_layers, n_qubits)`` tensor lines up with
    them directly.
    """
    # One encoding angle per qubit per layer, because the input is re-uploaded
    # in every layer rather than encoded once at the start.
    x_params = ParameterVector("x", n_layers * n_qubits)
    # RZ + RY per qubit per layer, plus one final rotation layer.
    theta_params = ParameterVector("theta", (n_layers + 1) * n_qubits * 2)

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))

    t = iter(range(len(theta_params)))
    for layer in range(n_layers):
        # Variational rotations. RY last so the final rotation of the circuit
        # does not commute with a Z-type observable.
        for q in range(n_qubits):
            qc.rz(theta_params[next(t)], q)
            qc.ry(theta_params[next(t)], q)

        # Circular entanglement, so qubit 0 is inside the light cone too.
        for q in range(n_qubits - 1):
            qc.cz(q, q + 1)
        qc.cz(n_qubits - 1, 0)

        # Data re-uploading: re-encode the (scaled) input every layer.
        for q in range(n_qubits):
            qc.rx(x_params[layer * n_qubits + q], q)

    # Final variational layer, again ending on RY.
    for q in range(n_qubits):
        qc.rz(theta_params[next(t)], q)
        qc.ry(theta_params[next(t)], q)

    return qc, x_params, theta_params


def build_observables(n_qubits: int = N_QUBITS):
    """One observable per action: Z0Z1 and Z2Z3.

    Qiskit Pauli strings are little-endian, so qubit 0 is the *rightmost*
    character: ``Z0Z1`` is ``"IIZZ"`` and ``Z2Z3`` is ``"ZZII"``.
    """
    if n_qubits != 4:
        raise ValueError("build_observables currently assumes 4 qubits")
    return [
        SparsePauliOp.from_list([("IIZZ", 1.0)]),  # Z0 Z1 -> action 0
        SparsePauliOp.from_list([("ZZII", 1.0)]),  # Z2 Z3 -> action 1
    ]


class VQCQNetwork(nn.Module):
    """Maps CartPole states to Q-values through a variational quantum circuit.

    Three trainable groups, which the optimizer should treat separately (the
    reference implementation uses a much larger learning rate for the output
    weights than for the circuit parameters):

    - ``qlayer``      the circuit rotation angles (theta)
    - ``input_scale`` per-layer, per-feature input scaling (lambda)
    - ``output_scale`` per-action output scaling (w)
    """

    def __init__(
        self,
        n_qubits: int = N_QUBITS,
        n_layers: int = N_LAYERS,
        seed: int | None = None,
        backend=None,
        shots: int | None = None,
        optimization_level: int = 1,
    ):
        """
        Args:
            backend: ``None`` runs on the exact statevector simulator. Pass a
                Qiskit backend (``FakeManilaV2()``, or a real
                ``QiskitRuntimeService`` backend) to run there instead; the
                circuit is transpiled for it and the observables are laid out
                to match.
            shots: ``None`` means exact expectation values, which is only
                possible on the simulator. Any integer adds sampling noise with
                standard error ~1/sqrt(shots). Ignored-as-exact is impossible on
                hardware, so a backend without ``shots`` defaults to 1024.
            optimization_level: transpiler optimization level, hardware only.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        qc, x_params, theta_params = build_qnetwork_circuit(n_qubits, n_layers)
        observables = build_observables(n_qubits)

        # precision is the estimator's target standard error; 0.0 means exact.
        precision = 0.0 if shots is None else 1.0 / np.sqrt(shots)

        if backend is None:
            estimator = StatevectorEstimator(seed=seed)
        else:
            pm = generate_preset_pass_manager(
                optimization_level=optimization_level, backend=backend, seed_transpiler=seed
            )
            qc = pm.run(qc)
            # EstimatorQNN transpiles the circuit but leaves user-supplied
            # observables alone, so they must be mapped onto the physical
            # qubits by hand or they act on the wrong wires.
            observables = [obs.apply_layout(qc.layout) for obs in observables]
            estimator = BackendEstimatorV2(backend=backend)
            if shots is None:
                precision = 1.0 / np.sqrt(1024)  # hardware can never be exact

        self.backend = backend
        self.circuit = qc

        qnn = EstimatorQNN(
            circuit=qc,
            observables=observables,
            input_params=list(x_params),
            weight_params=list(theta_params),
            # Gradients must reach the input parameters, because the trainable
            # input scaling lambda is applied on the torch side.
            input_gradients=True,
            # The library default is 0.015625, which injects sampling noise into
            # every expectation value even on the simulator.
            default_precision=precision,
            estimator=estimator,
        )

        rng = np.random.default_rng(seed)
        init_theta = rng.uniform(-np.pi, np.pi, size=len(theta_params))
        self.qlayer = TorchConnector(qnn, initial_weights=init_theta)

        # Input scaling, one per (layer, feature). Starts at 1.0 so the initial
        # encoding is plain arctan of the observation.
        self.input_scale = nn.Parameter(torch.ones(n_layers, n_qubits))
        # Output scaling, one per action. Lets Q-values leave [-1, 1].
        self.output_scale = nn.Parameter(torch.ones(N_ACTIONS))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """``(batch, n_qubits)`` observations -> ``(batch, N_ACTIONS)`` Q-values."""
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # arctan bounds CartPole's unbounded velocity features into the range
        # where a rotation angle is still informative.
        encoded = torch.arctan(state)                        # (B, n_qubits)
        angles = self.input_scale.unsqueeze(0) * encoded.unsqueeze(1)  # (B, L, n_qubits)
        angles = angles.reshape(state.shape[0], -1)          # (B, L * n_qubits)

        expectations = self.qlayer(angles)                   # (B, N_ACTIONS), in [-1, 1]
        return expectations * self.output_scale              # unbounded Q-values

    def parameter_groups(self, lr_theta=1e-3, lr_input=1e-3, lr_output=1e-1):
        """Optimizer parameter groups with per-group learning rates."""
        return [
            {"params": self.qlayer.parameters(), "lr": lr_theta},
            {"params": [self.input_scale], "lr": lr_input},
            {"params": [self.output_scale], "lr": lr_output},
        ]


def main():
    torch.manual_seed(0)
    model = VQCQNetwork(seed=0)

    print(model.circuit.draw())
    print()

    # Keep the documented circuit figures in sync with the code that built them.
    model.circuit.draw("mpl", filename=CIRCUIT_IMAGE, fold=-1)
    print(f"Circuit diagram written to {CIRCUIT_IMAGE}")

    # The same circuit as a real device would actually run it: native basis
    # gates, physical qubit mapping, and SWAPs for missing connectivity.
    try:
        from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    except ImportError:
        print("(install qiskit-ibm-runtime to also render the transpiled circuit)\n")
    else:
        hw = VQCQNetwork(seed=0, backend=FakeManilaV2(), shots=1024)
        hw.circuit.draw("mpl", filename=TRANSPILED_IMAGE, fold=38, idle_wires=False)
        print(f"Transpiled diagram written to {TRANSPILED_IMAGE} "
              f"({model.circuit.num_qubits} qubits depth {model.circuit.depth()} -> "
              f"{hw.circuit.num_qubits} qubits depth {hw.circuit.depth()})\n")

    n_theta = sum(p.numel() for p in model.qlayer.parameters())
    print(f"circuit rotation params (theta) : {n_theta}")
    print(f"input scaling params (lambda)   : {model.input_scale.numel()}")
    print(f"output scaling params (w)       : {model.output_scale.numel()}")

    # A batch of CartPole-like observations: [x, x_dot, theta, theta_dot].
    states = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, -1.2, 0.05, 0.8],
            [-2.4, 3.0, -0.2, -2.5],
        ],
        dtype=torch.float32,
    )

    q = model(states)
    print(f"\nQ-values shape: {tuple(q.shape)}")
    print(q.detach().numpy())


if __name__ == "__main__":
    main()
