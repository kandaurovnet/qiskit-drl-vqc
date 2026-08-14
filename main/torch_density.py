"""
Differentiable noisy quantum simulation in PyTorch, calibrated from a real device.

``torch_statevector`` runs an ideal circuit as a pure state. Noise makes states
mixed, which a statevector cannot represent, so this module tracks a density
matrix instead: ``rho`` of shape ``(batch, 2**n, 2**n)``. At 5 qubits that is a
32x32 matrix -- 1024 numbers -- so it stays cheap, and every operation is a
torch matmul, so autograd still supplies gradients in one backward pass. That is
what makes noisy training feasible at all: parameter shift on a noisy simulator
is ~1.5 min per gradient step, i.e. 34 days for a full run.

Fidelity is structural rather than fitted. The Kraus operators are taken
directly from ``qiskit_aer.noise.NoiseModel.from_backend(...)``, so the channels
applied here *are* Aer's channels for that device -- depolarizing, thermal
relaxation (T1/T2) and readout error, with the device's own calibration data.
Nothing is hand-tuned.

Two details matter for honesty:

- Noise is applied to the *transpiled* circuit (native gates, physical qubit
  layout, SWAPs inserted for missing connectivity), not the logical one. On
  FakeManilaV2 the CartPole ansatz goes from depth 24 to depth 185, and CX error
  (0.88%) is ~50x the single-qubit error, so most of the damage comes from
  routing. Applying noise to the logical circuit would understate it hugely.
- Readout error is applied to the measurement probabilities at the end, since
  the observables are diagonal.

``validate_against_aer()`` checks this reproduces Aer, and should be run
whenever the circuit or device changes.
"""

import numpy as np
import torch
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.quantum_info import Kraus

# Gate matrices for the transpiled (native) gate set, plus the logical set.
_SQ2 = 1 / np.sqrt(2)
STATIC_GATES = {
    "x":  np.array([[0, 1], [1, 0]], dtype=complex),
    "sx": 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex),
    "h":  _SQ2 * np.array([[1, 1], [1, -1]], dtype=complex),
    "id": np.eye(2, dtype=complex),
}


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


# ---------------------------------------------------------------------------
# Parameter expressions
# ---------------------------------------------------------------------------
# Transpilation rewrites angles: a bare `theta[7]` becomes e.g. `pi + theta[7]`,
# and rotations get merged. Those expressions are affine in the parameters, so
# each is reduced once to (offset, {param: coefficient}) and evaluated with
# torch ops at run time -- keeping the graph differentiable.

def affine_terms(expr):
    """Reduce a ParameterExpression to (offset, {parameter: coefficient})."""
    if not isinstance(expr, ParameterExpression):
        return float(expr), {}
    params = list(expr.parameters)
    if not params:
        return float(expr), {}
    coeffs = {}
    for p in params:
        grad = expr.gradient(p)
        if isinstance(grad, ParameterExpression) and grad.parameters:
            raise NotImplementedError(
                f"angle {expr} is not affine in {p}; torch_density supports affine angles only")
        coeffs[p] = float(grad)
    offset = float(expr.bind({p: 0.0 for p in params}))
    return offset, coeffs


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def compile_noisy_circuit(qc, weight_params, input_params, noise_model=None):
    """Flatten a (transpiled) circuit into ops, each optionally followed by noise.

    Returns a list of dicts describing gates and the Kraus channel that Aer
    would apply after them.
    """
    weight_index = {p: i for i, p in enumerate(weight_params)}
    input_index = {p: i for i, p in enumerate(input_params)}
    local = noise_model._local_quantum_errors if noise_model is not None else {}

    ops = []
    for inst in qc.data:
        name = inst.operation.name
        qubits = tuple(qc.find_bit(q).index for q in inst.qubits)
        if name in ("barrier", "measure", "delay"):
            continue

        op = {"name": name, "qubits": qubits}
        if name == "rz":
            # RZ is diagonal, so rho -> D rho D* is elementwise and needs no
            # axis permutation. On IBM devices RZ is also virtual (zero
            # duration, no error), so it carries no Kraus channel. It is over
            # half of a transpiled circuit, which makes this the single most
            # valuable special case.
            idx = np.arange(2 ** qc.num_qubits)
            op["diag_mask"] = torch.tensor((idx >> qubits[0]) & 1, dtype=torch.bool)
        if name in ROTATIONS:
            offset, coeffs = affine_terms(inst.operation.params[0])
            terms = []
            for p, c in coeffs.items():
                if p in weight_index:
                    terms.append(("weight", weight_index[p], c))
                elif p in input_index:
                    terms.append(("input", input_index[p], c))
                else:
                    raise ValueError(f"parameter {p} is neither a weight nor an input")
            op.update(kind="rotation", offset=offset, terms=terms)
        else:
            # Take the matrix from Qiskit rather than transcribing it, so the
            # operator convention matches the Kraus operators exactly.
            try:
                matrix = inst.operation.to_matrix()
            except (AttributeError, TypeError) as exc:
                raise NotImplementedError(
                    f"gate {name!r} has no matrix for torch_density") from exc
            op.update(kind="static",
                      matrix=torch.tensor(np.asarray(matrix), dtype=torch.complex64))

        # The Kraus channel Aer would apply after this gate on these qubits.
        err = local.get(name, {}).get(qubits)
        if err is not None:
            K = np.stack(Kraus(err.to_quantumchannel()).data)
            if op["kind"] == "static":
                # Fold the gate into the channel: applying K_i @ U once is half
                # the axis permutations of applying U and then K_i.
                K = K @ np.asarray(matrix)
                op.pop("matrix")
                op["kind"] = "channel"
            op["kraus"] = torch.tensor(K, dtype=torch.complex64)
        ops.append(op)
    return ops


def _two_qubit_matrix(name):
    if name == "cx":
        m = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    elif name == "cz":
        m = np.diag([1, 1, 1, -1]).astype(complex)
    else:  # ecr
        m = _SQ2 * np.array([[0, 1, 0, 1j], [1, 0, -1j, 0],
                             [0, 1j, 0, 1], [-1j, 0, 1, 0]], dtype=complex)
    return torch.tensor(m, dtype=torch.complex64)


def readout_matrix(noise_model, n_qubits):
    """Confusion matrix over full bitstrings, or None if the device has no readout error.

    Applied to the measurement probability vector at the end, which is valid
    because the observables here are diagonal.
    """
    if noise_model is None or not noise_model._local_readout_errors:
        return None
    per_qubit = {}
    for qubits, ro in noise_model._local_readout_errors.items():
        per_qubit[qubits[0] if isinstance(qubits, tuple) else qubits] = np.array(ro.probabilities)
    dim = 2 ** n_qubits
    M = np.ones((dim, dim))
    idx = np.arange(dim)
    for q in range(n_qubits):
        A = per_qubit.get(q, np.eye(2))
        true_bit = (idx[None, :] >> q) & 1     # column: true outcome
        meas_bit = (idx[:, None] >> q) & 1     # row: measured outcome
        M = M * A[true_bit, meas_bit]
    return torch.tensor(M, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _perm_for(qubits, n):
    """Permutation bringing `qubits`' row and column axes to the front of each side.

    rho is viewed as 2n single-qubit axes (n row, n column). This groups the
    target row axes and target column axes together so a channel becomes one
    einsum over a (2**k x 2**k) block, and returns the inverse permutation too.
    """
    axes = _axes_for(qubits, n)
    rest = [i for i in range(n) if i not in axes]
    # +1 for the batch axis; column axes live n positions further along.
    fwd = ([0] + [a + 1 for a in axes] + [r + 1 for r in rest]
           + [a + 1 + n for a in axes] + [r + 1 + n for r in rest])
    inv = [0] * len(fwd)
    for new_pos, old_pos in enumerate(fwd):
        inv[old_pos] = new_pos
    return fwd, inv


def _apply_channel(rho, ops, qubits, n, unitary=False):
    """Apply a Kraus channel (or a single unitary) to `qubits` of rho.

    ``ops`` is (m, 2**k, 2**k) Kraus operators, or (2**k, 2**k) / (batch, ...)
    for a plain unitary. The whole Kraus sum is one einsum rather than a Python
    loop over operators -- a CX error channel has 16 operators, and looping over
    them with separate permutes was ~25x slower than the batched contraction.
    """
    batch = rho.shape[0]
    k = len(qubits)
    d, rest = 2 ** k, 2 ** (n - k)
    fwd, inv = _perm_for(qubits, n)

    rho = rho.reshape([batch] + [2] * (2 * n))
    rho = rho.permute(fwd).reshape(batch, d, rest, d, rest)

    if unitary:
        if ops.dim() == 2:                       # same gate for every sample
            rho = torch.einsum("tu,buavc,sv->btasc", ops, rho, ops.conj())
        else:                                    # per-sample rotation angles
            rho = torch.einsum("btu,buavc,bsv->btasc", ops, rho, ops.conj())
    else:
        rho = torch.einsum("ptu,buavc,psv->btasc", ops, rho, ops.conj())

    rho = rho.reshape([batch] + [2] * (2 * n)).permute(inv)
    return rho.reshape(batch, 2 ** n, 2 ** n)


def _axes_for(qubits, n):
    """Tensor axes for `qubits`, ordered to match Qiskit's operator convention.

    Little-endian: qubit q occupies axis (n - 1 - q). Within a multi-qubit
    operator Qiskit treats the *first* listed qubit as the least significant
    bit, so the gathered axes are reversed -- the first qubit must end up last,
    i.e. least significant, in the 2**k block index. Getting this backwards
    silently transposes control and target on every CX and mismatches the Kraus
    operators, which is not visible on symmetric gates like CZ.
    """
    return [n - 1 - q for q in reversed(qubits)]


def run_noisy(ops, observables, weights, angles, n_qubits, readout=None):
    """Execute a compiled noisy circuit; returns (batch, n_obs) expectation values."""
    batch = angles.shape[0]
    device = angles.device
    rho = torch.zeros(batch, 2 ** n_qubits, 2 ** n_qubits,
                      dtype=torch.complex64, device=device)
    rho[:, 0, 0] = 1.0                       # |0...0><0...0|

    for op in ops:
        if op["kind"] == "rotation":
            angle = torch.full((batch,), op["offset"], device=device)
            for source, idx, coeff in op["terms"]:
                angle = angle + coeff * (weights[idx] if source == "weight"
                                         else angles[:, idx])
            if "diag_mask" in op:
                # Diagonal gate: rho -> d rho d*, elementwise, no permutation.
                mask = op["diag_mask"].to(device)
                phase = torch.polar(torch.ones_like(angle), angle / 2)   # e^{+i t/2}
                d = torch.where(mask.unsqueeze(0), phase.unsqueeze(1),
                                phase.conj().unsqueeze(1))               # (batch, 2**n)
                rho = rho * d.unsqueeze(2) * d.conj().unsqueeze(1)
                continue
            U = ROTATIONS[op["name"]](angle)
        elif op["kind"] == "channel":
            # Gate already folded into the Kraus operators at compile time.
            rho = _apply_channel(rho, op["kraus"].to(device), op["qubits"], n_qubits)
            continue
        else:
            U = op["matrix"].to(device)
        rho = _apply_channel(rho, U, op["qubits"], n_qubits, unitary=True)
        if "kraus" in op:
            rho = _apply_channel(rho, op["kraus"].to(device), op["qubits"], n_qubits)

    prob = torch.diagonal(rho, dim1=-2, dim2=-1).real    # measurement distribution
    if readout is not None:
        prob = prob @ readout.to(device).T               # apply readout confusion
    return prob @ observables.T.to(device)
