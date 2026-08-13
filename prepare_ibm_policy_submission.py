#!/usr/bin/env python3
"""Prepare a trained two-state VQC evaluation for IBM Runtime.

This script connects to IBM Quantum to select and transpile for a backend, but
it never creates an Estimator and never submits a Runtime job.  Its output is a
reviewable manifest containing the exact checkpoint, states, parameter order,
parameter values, circuit metrics, and planned execution settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ibm_backend_inference import (
    ACCOUNT_NAME,
    DEFAULT_BACKEND,
    OPTIMIZATION_LEVEL,
    SEED_TRANSPILER,
    TARGET_PRECISION,
    prepare_vqc_for_backend,
)


DEFAULT_CHECKPOINT = Path("results/quantum_policy.pt")
DEFAULT_PREVIEW = Path("artifacts/ibm/validation/ibm_policy_preview.json")
DEFAULT_OUTPUT = Path("artifacts/ibm/validation/ibm_policy_submission_preview.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_validated_preview(preview_path: Path, checkpoint_path: Path) -> dict:
    if not preview_path.is_file():
        raise FileNotFoundError(f"Policy preview not found: {preview_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    preview = json.loads(preview_path.read_text(encoding="utf-8"))
    if preview.get("status") != "preview":
        raise RuntimeError("Policy preview status is not 'preview'.")
    if preview.get("qpu_job_submitted") is not False:
        raise RuntimeError("Expected a local-only preview with no submitted QPU job.")
    validation = preview.get("torch_qiskit_exact_validation", {})
    if validation.get("status") != "passed":
        raise RuntimeError("Torch/Qiskit exact validation has not passed.")

    selected = preview.get("selected_states", [])
    if len(selected) != 2:
        raise RuntimeError(f"Expected exactly two selected states, got {len(selected)}.")
    if {state.get("exact_action") for state in selected} != {0, 1}:
        raise RuntimeError("Selected states must include one state for each action.")

    actual_hash = _sha256(checkpoint_path)
    if preview.get("checkpoint_sha256") != actual_hash:
        raise RuntimeError("Checkpoint SHA-256 does not match the validated policy preview.")
    return preview


def bind_trained_policy(prepared, preview: dict, checkpoint_path: Path) -> tuple[list, list]:
    """Create one parameter row per state in transpiled-circuit order."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    theta = checkpoint["theta"].detach().cpu().numpy().astype(float)
    input_scale = checkpoint["input_scale"].detach().cpu().numpy().astype(float)
    output_scale = checkpoint["output_scale"].detach().cpu().numpy().astype(float)

    if theta.shape != (len(prepared.weight_parameters),):
        raise RuntimeError(
            f"Theta shape {theta.shape} does not match "
            f"{len(prepared.weight_parameters)} circuit weights."
        )
    expected_scale_shape = (len(prepared.input_parameters) // 4, 4)
    if input_scale.shape != expected_scale_shape:
        raise RuntimeError(
            f"Input-scale shape {input_scale.shape} does not match {expected_scale_shape}."
        )
    if output_scale.shape != (2,) or not np.isfinite(output_scale).all():
        raise RuntimeError(f"Invalid output scale: {output_scale}")

    parameter_names = [str(parameter) for parameter in prepared.circuit.parameters]
    parameter_rows = []
    for index, state in enumerate(preview["selected_states"]):
        normalized = np.asarray(state["normalized_observation"], dtype=float)
        if normalized.shape != (4,) or not np.isfinite(normalized).all():
            raise RuntimeError(f"Invalid normalized state {index}: {normalized}")

        input_angles = (input_scale * normalized[None, :]).reshape(-1)
        values_by_parameter = {
            **dict(zip(prepared.input_parameters, input_angles, strict=True)),
            **dict(zip(prepared.weight_parameters, theta, strict=True)),
        }
        row = np.asarray(
            [values_by_parameter[parameter] for parameter in prepared.circuit.parameters],
            dtype=float,
        )
        if row.shape != (44,) or not np.isfinite(row).all():
            raise RuntimeError(f"Invalid bound parameter row {index}: shape={row.shape}")
        parameter_rows.append([float(value) for value in row])

    return parameter_names, parameter_rows


def create_submission_preview(
    *,
    checkpoint_path: Path,
    preview_path: Path,
    output_path: Path,
    backend_name: str | None,
    account_name: str,
    target_precision: float,
) -> dict:
    if target_precision <= 0:
        raise ValueError("Target precision must be greater than zero.")

    preview = load_validated_preview(preview_path, checkpoint_path)
    prepared = prepare_vqc_for_backend(
        backend_name,
        account_name=account_name,
        seed_transpiler=SEED_TRANSPILER,
        optimization_level=OPTIMIZATION_LEVEL,
    )
    parameter_names, parameter_rows = bind_trained_policy(
        prepared, preview, checkpoint_path
    )
    status = prepared.backend.status()

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "submission_preview",
        "qpu_job_submitted": False,
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": preview["checkpoint_sha256"],
        "source_preview": preview_path.as_posix(),
        "backend": {
            "name": prepared.backend.name,
            "num_qubits": prepared.backend.num_qubits,
            "operational": bool(status.operational),
            "pending_jobs_at_preparation": int(status.pending_jobs),
        },
        "execution_plan": {
            "primitive": "qiskit_ibm_runtime.EstimatorV2",
            "mode": "job",
            "target_precision": target_precision,
            "pubs": 1,
            "parameter_sets": len(parameter_rows),
            "observables_per_parameter_set": len(prepared.observables),
            "states": len(parameter_rows),
        },
        "transpilation": {
            "optimization_level": OPTIMIZATION_LEVEL,
            "seed_transpiler": SEED_TRANSPILER,
            "original_depth": prepared.original_depth,
            "isa_depth": prepared.transpiled_depth,
            "two_qubit_gates": prepared.two_qubit_gates,
            "input_parameters": len(prepared.input_parameters),
            "weight_parameters": len(prepared.weight_parameters),
            "mapped_observables": len(prepared.observables),
        },
        "parameter_binding": {
            "order": parameter_names,
            "values": parameter_rows,
            "values_per_state": len(parameter_names),
        },
        "output_scale": [
            float(value)
            for value in torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )["output_scale"]
        ],
        "selected_states": preview["selected_states"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, but do not submit, a two-state IBM policy evaluation."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--account", default=ACCOUNT_NAME)
    parser.add_argument("--target-precision", type=float, default=TARGET_PRECISION)
    args = parser.parse_args()

    manifest = create_submission_preview(
        checkpoint_path=args.checkpoint,
        preview_path=args.preview,
        output_path=args.output,
        backend_name=args.backend.strip() or None,
        account_name=args.account,
        target_precision=args.target_precision,
    )
    backend = manifest["backend"]
    plan = manifest["execution_plan"]
    transpilation = manifest["transpilation"]
    print(f"Backend: {backend['name']} ({backend['num_qubits']} qubits)")
    print(f"Pending jobs: {backend['pending_jobs_at_preparation']}")
    print(
        f"Circuit depth: {transpilation['original_depth']} -> "
        f"{transpilation['isa_depth']}"
    )
    print(f"Two-qubit gates: {transpilation['two_qubit_gates']}")
    print(
        f"Planned execution: {plan['states']} states, "
        f"{plan['observables_per_parameter_set']} observables, "
        f"target precision {plan['target_precision']}"
    )
    print(f"Submission preview saved to: {args.output}")
    print("No Estimator was created and no QPU job was submitted.")


if __name__ == "__main__":
    main()
