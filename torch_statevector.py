"""
Execute a Qiskit QuantumCircuit as a batched statevector in PyTorch.

Qiskit's ``EstimatorQNN`` differentiates by parameter shift, costing
``2 * n_params`` circuit evaluations per sample. For a 46-parameter circuit at
batch 64 that is ~15.7 s per gradient step, which makes DQN training take weeks.

For a handful of qubits the state is a small complex vector (16 amplitudes at
n=4), so the circuit is just a sequence of small matmuls. Running it in torch
lets autograd produce the gradient in one backward pass instead of 88 circuit
evaluations — ~1900x faster, and exact rather than approximate.

This module knows nothing about any particular ansatz. It *reads* a Qiskit
circuit and replays its gate list, so the circuit definition stays wherever it
was written and cannot drift out of sync with this executor.

Conventions match Qiskit exactly:
    RZ(t) = diag(exp(-i t/2), exp(+i t/2))
    RY(t) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
    RX(t) = [[cos(t/2), -i sin(t/2)], [-i sin(t/2), cos(t/2)]]
    CZ    = diag(1, 1, 1, -1)
with little-endian qubit ordering, so qubit 0 is the least significant bit.

Supported gates: h, rz, ry, rx, cz. Anything else raises NotImplementedError
rather than silently producing a wrong answer.
"""

import numpy as np
import torch
from qiskit.circuit import Parameter


def _rz(t):
    one = torch.ones_like(t)
    zero = torch.zeros_like(t).to(torch.complex64)
    return torch.stack([
        torch.stack([torch.polar(one, -t / 2), zero], -1),
        torch.stack([zero, torch.polar(one, t / 2)], -1)], -2)


def _ry(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    return torch.stack([torch.stack([c, -s], -1),
                        torch.stack([s, c], -1)], -2).to(torch.complex64)


def _rx(t):
    c = torch.cos(t / 2).to(torch.complex64)
    s = -1j * torch.sin(t / 2).to(torch.complex64)
    return torch.stack([torch.stack([c, s], -1),
                        torch.stack([s, c], -1)], -2)


ROTATIONS = {"rz": _rz, "ry": _ry, "rx": _rx}
H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)


def apply_1q(psi, gate, qubit, n_qubits):
    """Contract a (2,2) or (batch,2,2) gate against one qubit of (batch, 2**n).

    The state is reshaped so the target qubit is its own axis, the gate is
    contracted against it, then it is flattened back.
    """
    batch = psi.shape[0]
    psi = psi.reshape(batch, 2 ** (n_qubits - qubit - 1), 2, 2 ** qubit)
    eq = "gh,bihk->bigk" if gate.dim() == 2 else "bgh,bihk->bigk"
    return torch.einsum(eq, gate, psi).reshape(batch, 2 ** n_qubits)


def compile_circuit(qc, weight_params, input_params):
    """Flatten a QuantumCircuit into a list of torch-executable ops.

    Rotation angles are resolved once to an index into either the weight vector
    or the per-sample input vector, so the compiled program is reused by every
    forward pass.

    Args:
        qc: the circuit to compile.
        weight_params: ordered trainable parameters (indexed into ``weights``).
        input_params: ordered data parameters (indexed into ``angles``).

    Returns:
        A list of tuples: ``("h", qubit)``, ``("diag", phases)``, or
        ``(rotation_name, qubit, "weight"|"input", index)``.
    """
    weight_index = {p: i for i, p in enumerate(weight_params)}
    input_index = {p: i for i, p in enumerate(input_params)}
    n = qc.num_qubits
    ops = []
    for inst in qc.data:
        name = inst.operation.name
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        if name == "h":
            ops.append(("h", qubits[0]))
        elif name == "cz":
            # CZ is diagonal, so the whole gate is an elementwise phase flip
            # on the basis states where both qubits are 1.
            c, t = qubits
            idx = torch.arange(2 ** n)
            both = (((idx >> c) & 1) & ((idx >> t) & 1)).bool()
            ops.append(("diag", torch.where(both, -1.0, 1.0)))
        elif name in ROTATIONS:
            param = inst.operation.params[0]
            if not isinstance(param, Parameter):
                raise ValueError(f"{name} angle must be a bare Parameter, got {param!r}")
            if param in weight_index:
                ops.append((name, qubits[0], "weight", weight_index[param]))
            elif param in input_index:
                ops.append((name, qubits[0], "input", input_index[param]))
            else:
                raise ValueError(f"parameter {param} is neither a weight nor an input")
        else:
            raise NotImplementedError(
                f"gate {name!r} has no torch equivalent; add one here or use the Qiskit backend")
    return ops


def compile_observables(observables, n_qubits):
    """Turn I/Z-only SparsePauliOps into diagonal vectors over the basis states.

    A Z-type observable is diagonal, so its expectation value is just a weighted
    sum of measurement probabilities — no matrix needed.

    Returns a ``(n_observables, 2**n)`` tensor.
    """
    diags = []
    for obs in observables:
        total = torch.zeros(2 ** n_qubits)
        for pauli, coeff in zip(obs.paulis, obs.coeffs):
            label = str(pauli)[::-1]        # little-endian: reverse to index by qubit
            if set(label) - {"I", "Z"}:
                raise NotImplementedError(
                    f"only I/Z observables are supported, got {pauli}")
            idx = torch.arange(2 ** n_qubits)
            parity = torch.zeros_like(idx)
            for q, ch in enumerate(label):
                if ch == "Z":
                    parity = parity ^ ((idx >> q) & 1)
            total = total + float(coeff.real) * torch.where(parity.bool(), -1.0, 1.0)
        diags.append(total)
    return torch.stack(diags)


def run(ops, observables, weights, angles, n_qubits):
    """Execute a compiled circuit and return expectation values.

    Args:
        ops: output of ``compile_circuit``.
        observables: output of ``compile_observables``, ``(n_obs, 2**n)``.
        weights: 1-D tensor of trainable angles.
        angles: ``(batch, n_input_params)`` per-sample data angles.
        n_qubits: qubit count.

    Returns:
        ``(batch, n_obs)`` expectation values.
    """
    batch = angles.shape[0]
    device = angles.device
    # Start in |0...0>; any Hadamard layer is part of the circuit itself.
    psi = torch.zeros(batch, 2 ** n_qubits, dtype=torch.complex64, device=device)
    psi[:, 0] = 1.0

    for op in ops:
        if op[0] == "h":
            psi = apply_1q(psi, H.to(device), op[1], n_qubits)
        elif op[0] == "diag":
            psi = psi * op[1].to(device).to(torch.complex64)
        else:
            name, qubit, source, idx = op
            angle = weights[idx] if source == "weight" else angles[:, idx]
            psi = apply_1q(psi, ROTATIONS[name](angle), qubit, n_qubits)

    prob = psi.real ** 2 + psi.imag ** 2
    return prob @ observables.T.to(device)
