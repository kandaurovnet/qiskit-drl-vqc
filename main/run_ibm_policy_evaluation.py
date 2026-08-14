#!/usr/bin/env python3
"""Submit or resume the trained two-state VQC evaluation on IBM Quantum."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

from ibm_backend_inference import ACCOUNT_NAME, prepare_vqc_for_backend
from prepare_ibm_policy_submission import (
    DEFAULT_CHECKPOINT,
    DEFAULT_PREVIEW,
    bind_trained_policy,
    load_validated_preview,
)


DEFAULT_MANIFEST = Path("artifacts/ibm/validation/ibm_policy_submission_preview.json")
RESULTS_DIRECTORY = Path("artifacts/ibm/core")


def _safe_job_id(job_id: str) -> str:
    return "".join(c for c in job_id if c.isalnum() or c in "-_")


def _result_path(job_id: str) -> Path:
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIRECTORY / f"ibm_policy_evaluation_{_safe_job_id(job_id)}.json"


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "submission_preview":
        raise RuntimeError("Expected a validated submission_preview manifest.")
    if manifest.get("qpu_job_submitted") is not False:
        raise RuntimeError("The preview manifest unexpectedly records a QPU job.")
    plan = manifest.get("execution_plan", {})
    if plan.get("states") != 2 or plan.get("observables_per_parameter_set") != 2:
        raise RuntimeError("Expected exactly two states and two observables.")
    return manifest


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def complete_result(metadata: dict, result_path: Path, runtime_result) -> dict:
    evs = np.asarray(runtime_result[0].data.evs, dtype=float)
    if evs.shape == (2,):
        metadata.update(
            {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "partial_broadcast_result",
                "runtime_expectations": [float(v) for v in evs],
                "broadcast_pairing": [
                    "state_0_observable_0",
                    "state_1_observable_1",
                ],
                "result_complete": False,
                "note": (
                    "The two parameter sets and two observables were paired "
                    "element-wise. This job is retained as a partial result and "
                    "must not be used for action comparison."
                ),
            }
        )
        save_json(result_path, metadata)
        return metadata
    if evs.shape != (2, 2) or not np.isfinite(evs).all():
        raise RuntimeError(f"Expected Runtime expectation shape (2, 2), got {evs.shape}.")

    output_scale = np.asarray(metadata["output_scale"], dtype=float)
    hardware_q_values = evs * output_scale[None, :]
    states = []
    for index, (expectations, q_values, reference) in enumerate(
        zip(evs, hardware_q_values, metadata["selected_states"], strict=True)
    ):
        exact_q_values = np.asarray(reference["qiskit_exact_q_values"], dtype=float)
        hardware_action = int(np.argmax(q_values))
        exact_action = int(reference["exact_action"])
        states.append(
            {
                "state_index": index,
                "evaluation_seed": reference["evaluation_seed"],
                "step": reference["step"],
                "runtime_expectations": [float(v) for v in expectations],
                "runtime_q_values": [float(v) for v in q_values],
                "runtime_action": hardware_action,
                "exact_qiskit_q_values": [float(v) for v in exact_q_values],
                "exact_action": exact_action,
                "action_agreement": hardware_action == exact_action,
                "absolute_q_value_error": [
                    float(v) for v in np.abs(q_values - exact_q_values)
                ],
            }
        )

    metadata.update(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "runtime_results": states,
            "action_agreement_count": sum(s["action_agreement"] for s in states),
            "action_agreement_rate": float(
                np.mean([s["action_agreement"] for s in states])
            ),
            "max_absolute_q_value_error": max(
                max(s["absolute_q_value_error"]) for s in states
            ),
        }
    )
    save_json(result_path, metadata)
    return metadata


def submit(manifest_path: Path, checkpoint_path: Path, preview_path: Path) -> tuple[dict, Path]:
    manifest = load_manifest(manifest_path)
    preview = load_validated_preview(preview_path, checkpoint_path)
    if manifest["checkpoint_sha256"] != preview["checkpoint_sha256"]:
        raise RuntimeError("Manifest and policy preview use different checkpoints.")

    backend_name = manifest["backend"]["name"]
    prepared = prepare_vqc_for_backend(backend_name, account_name=ACCOUNT_NAME)
    parameter_names, parameter_rows = bind_trained_policy(
        prepared, preview, checkpoint_path
    )
    if parameter_names != manifest["parameter_binding"]["order"]:
        raise RuntimeError("Current transpiled parameter order differs from the manifest.")
    np.testing.assert_allclose(
        parameter_rows,
        manifest["parameter_binding"]["values"],
        rtol=0.0,
        atol=0.0,
        err_msg="Current trained-policy bindings differ from the manifest.",
    )

    precision = float(manifest["execution_plan"]["target_precision"])
    estimator = EstimatorV2(mode=prepared.backend)
    # Estimator V2 broadcasts observables against parameter-set batch axes.
    # Shapes (1, 2) and (2, 1, 44) create the required Cartesian product:
    # two states by two observables, producing expectation shape (2, 2).
    observable_grid = [list(prepared.observables)]
    parameter_grid = np.asarray(parameter_rows, dtype=float)[:, None, :]
    job = estimator.run(
        [(prepared.circuit, observable_grid, parameter_grid)],
        precision=precision,
    )
    job_id = job.job_id()
    result_path = _result_path(job_id)
    metadata = {
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "submitted",
        "qpu_job_submitted": True,
        "backend": prepared.backend.name,
        "job_id": job_id,
        "target_precision": precision,
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": preview["checkpoint_sha256"],
        "source_preview": preview_path.as_posix(),
        "source_submission_preview": manifest_path.as_posix(),
        "output_scale": manifest["output_scale"],
        "selected_states": preview["selected_states"],
        "original_depth": prepared.original_depth,
        "isa_depth": prepared.transpiled_depth,
        "two_qubit_gates": prepared.two_qubit_gates,
    }
    save_json(result_path, metadata)
    print(f"IBM Runtime job submitted: {job_id}", flush=True)
    print(f"Submission metadata saved to: {result_path}", flush=True)
    print("Waiting for the queued job to complete...", flush=True)
    return complete_result(metadata, result_path, job.result()) , result_path


def resume(job_id: str, account_name: str = ACCOUNT_NAME) -> tuple[dict, Path]:
    result_path = _result_path(job_id)
    if not result_path.is_file():
        raise FileNotFoundError(f"Submission metadata not found: {result_path}")
    metadata = json.loads(result_path.read_text(encoding="utf-8"))
    if metadata.get("job_id") != job_id:
        raise RuntimeError("Saved metadata job ID does not match the requested job.")
    if metadata.get("status") == "completed":
        return metadata, result_path
    service = QiskitRuntimeService(name=account_name)
    print(f"Retrieving IBM Runtime job: {job_id}", flush=True)
    return complete_result(metadata, result_path, service.job(job_id).result()), result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--resume", metavar="JOB_ID")
    args = parser.parse_args()

    if args.resume:
        result, path = resume(args.resume)
    else:
        result, path = submit(args.manifest, args.checkpoint, args.preview)

    print(f"Status: {result['status']}")
    if result["status"] == "completed":
        print(f"Action agreement: {result['action_agreement_count']}/2")
        print(f"Maximum absolute Q-value error: {result['max_absolute_q_value_error']:.6f}")
    else:
        print(result["note"])
    print(f"Final results saved to: {path}")


if __name__ == "__main__":
    main()
