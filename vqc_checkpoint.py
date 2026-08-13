#!/usr/bin/env python3
"""Load a trained VQC checkpoint into any execution backend.

Training runs on ``backend="torch"`` because the Qiskit parameter-shift path is
~1900x slower, but finite-shot and hardware evaluation must run on a Qiskit
backend. Those two backends store the same 32 circuit angles under different
state_dict keys, so a plain ``load_state_dict`` fails:

    torch  backend : input_scale, output_scale, theta,         _obs
    qiskit backend : input_scale, output_scale, qlayer.weight, qlayer._weights

``theta`` and ``qlayer.weight`` are the same numbers; ``_obs`` is a buffer the
torch executor precomputes and the Qiskit path does not use. ``qlayer._weights``
is the same tensor object as ``qlayer.weight`` -- state_dict does not
deduplicate shared parameters, so it appears twice and must be written twice.

Loading is strict on both sides: every target key must be filled and every
source key must be consumed. A tolerant load would silently leave the circuit
angles at their random initialization, which looks like a working model and
evaluates like an untrained one.

    from vqc_checkpoint import load_vqc
    model = load_vqc("results_q3seed/quantum_s2_policy.pt",
                     backend="qiskit", shots=1024)

Self-test (asserts the converted model reproduces the torch model exactly):

    python vqc_checkpoint.py results_q3seed/quantum_s2_policy.pt
"""

import argparse

import torch

from vqc import VQCQNetwork

# Circuit angles, per backend. Both entries of the Qiskit pair alias one tensor.
_ANGLE_KEYS = {"torch": ("theta",), "qiskit": ("qlayer.weight", "qlayer._weights")}
# Buffers the torch executor precomputes; absent from every other backend.
_TORCH_ONLY = ("_obs",)
_SHARED = ("input_scale", "output_scale")


def infer_shape(state_dict):
    """Recover ``(n_layers, n_qubits)`` from the checkpoint itself.

    Nothing in the .pt file records the architecture, but ``input_scale`` is one
    trainable scale per (layer, qubit), so its shape is exactly that pair.
    """
    return tuple(state_dict["input_scale"].shape)


def convert_state_dict(state_dict, model):
    """Map a checkpoint onto ``model``'s key layout.

    Starts from the model's own state_dict so backend-specific buffers (the
    torch executor's precomputed ``_obs``) come from the freshly built circuit
    rather than from a checkpoint that may not carry them, then overwrites
    exactly the tensors that training produced.
    """
    src_angles = _ANGLE_KEYS["torch" if "theta" in state_dict else "qiskit"]
    missing = [k for k in (*_SHARED, src_angles[0]) if k not in state_dict]
    if missing:
        raise KeyError(f"checkpoint is missing {missing}; not a VQCQNetwork state_dict")
    unused = set(state_dict) - set(_SHARED) - set(src_angles) - set(_TORCH_ONLY)
    if unused:
        raise KeyError(f"unrecognized keys in checkpoint: {sorted(unused)}")

    angles = state_dict[src_angles[0]]
    out = dict(model.state_dict())
    all_angle_keys = set(_ANGLE_KEYS["torch"]) | set(_ANGLE_KEYS["qiskit"])
    for key in out:
        if key in all_angle_keys:
            if out[key].shape != angles.shape:
                raise ValueError(
                    f"angle count mismatch: checkpoint has {tuple(angles.shape)}, "
                    f"model expects {tuple(out[key].shape)} -- wrong n_layers?"
                )
            out[key] = angles.clone()
        elif key in _SHARED:
            out[key] = state_dict[key].clone()
    return out


def load_vqc(checkpoint_path, backend="torch", shots=None, **kwargs):
    """Build a VQCQNetwork on ``backend`` and load ``checkpoint_path`` into it.

    ``n_qubits``/``n_layers`` come from the checkpoint unless overridden. Extra
    kwargs (e.g. ``optimization_level``) pass through to VQCQNetwork.
    """
    sd = torch.load(checkpoint_path, map_location="cpu")
    n_layers, n_qubits = infer_shape(sd)

    model = VQCQNetwork(
        n_qubits=kwargs.pop("n_qubits", n_qubits),
        n_layers=kwargs.pop("n_layers", n_layers),
        backend=backend,
        shots=shots,
        **kwargs,
    )
    # A hardware backend transpiles the circuit but keeps the same ParameterVector,
    # so the angle count -- and therefore this mapping -- is unchanged.
    model.load_state_dict(convert_state_dict(sd, model))
    model.eval()
    return model


def check_roundtrip(checkpoint_path, batch=8, seed=0, rtol=1e-5):
    """Assert the Qiskit-loaded model matches the torch-loaded one.

    Same trained weights on two executors must give the same Q-values, so any
    difference here is a conversion bug rather than a modelling choice. Uses
    ``shots=None`` because sampling noise would mask exactly that.

    The tolerance is *relative*: training drives ``output_scale`` to ~120 so it
    can express CartPole's Q-values, which amplifies the float32 disagreement
    between the two executors by the same factor. An absolute bound tuned on a
    freshly initialized model (output_scale = 1) fails here for that reason
    alone, with nothing actually wrong.
    """
    tc = load_vqc(checkpoint_path, backend="torch")
    qk = load_vqc(checkpoint_path, backend="qiskit", shots=None)

    # Observations arrive already normalized to [-1, 1] by cartpole_dqn.
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(batch, tc.n_qubits, generator=g, dtype=torch.float32) * 2 - 1
    with torch.no_grad():
        a, b = tc(x), qk(x)
    scale = float(a.abs().max())
    diff = float((a - b).abs().max()) / max(scale, 1e-9)
    assert diff < rtol, f"backends disagree by {diff:.2e} relative (rtol {rtol:.0e})"
    return diff


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()

    sd = torch.load(args.checkpoint, map_location="cpu")
    n_layers, n_qubits = infer_shape(sd)
    n_params = sum(v.numel() for k, v in sd.items() if k != "_obs")
    print(f"{args.checkpoint}: {n_layers} layers x {n_qubits} qubits, {n_params} trainable")

    diff = check_roundtrip(args.checkpoint, batch=args.batch)
    print(f"torch vs qiskit max relative |dQ| = {diff:.2e}  -> conversion is exact")
