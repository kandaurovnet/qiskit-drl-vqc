"""Prepare the CartPole VQC for an IBM Quantum backend.

Running it first completes Q8.2 without using QPU time.  It then offers an
explicit confirmation dialog for the Q8.3 Runtime Estimator smoke test.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable

import numpy as np
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

from vqc import build_observables, build_qnetwork_circuit


ACCOUNT_NAME = "cartpole-vqc"
DEFAULT_BACKEND = "ibm_marrakesh"
SEED_TRANSPILER = 42
OPTIMIZATION_LEVEL = 3
TARGET_PRECISION = 0.1
SMOKE_TEST_OBSERVATION = (0.05, -0.10, 0.02, 0.15)
RESULTS_DIRECTORY = Path(__file__).with_name("artifacts") / "ibm" / "validation"


@dataclass(frozen=True)
class PreparedVQC:
    """Artifacts needed by the later Q8.3 Runtime Estimator submission."""

    backend: object
    circuit: object
    observables: tuple[SparsePauliOp, ...]
    input_parameters: tuple[Parameter, ...]
    weight_parameters: tuple[Parameter, ...]
    original_depth: int
    transpiled_depth: int
    two_qubit_gates: int


@dataclass(frozen=True)
class SmokeTestInputs:
    """One frozen model input expressed in circuit parameter order."""

    observation: tuple[float, ...]
    parameter_values: tuple[float, ...]
    output_scale: tuple[float, ...]
    local_expectations: tuple[float, ...]
    local_q_values: tuple[float, ...]


def _select_backend(service: QiskitRuntimeService, backend_name: str | None):
    """Select a named operational QPU, or the currently least-busy QPU."""
    if backend_name:
        backend = service.backend(backend_name)
        status = backend.status()
        if not status.operational:
            raise RuntimeError(f"Backend {backend_name!r} is not operational.")
        if getattr(backend.configuration(), "simulator", False):
            raise RuntimeError(f"Backend {backend_name!r} is a simulator, not a QPU.")
        if backend.num_qubits < 4:
            raise RuntimeError(f"Backend {backend_name!r} has fewer than four qubits.")
        return backend

    return service.least_busy(
        simulator=False,
        operational=True,
        min_num_qubits=4,
    )


def prepare_vqc_for_backend(
    backend_name: str | None = DEFAULT_BACKEND,
    *,
    account_name: str = ACCOUNT_NAME,
    seed_transpiler: int = SEED_TRANSPILER,
    optimization_level: int = OPTIMIZATION_LEVEL,
) -> PreparedVQC:
    """Build and validate an ISA-compatible VQC without submitting a job."""
    service = QiskitRuntimeService(name=account_name)
    backend = _select_backend(service, backend_name)

    return prepare_vqc_for_target(
        backend,
        seed_transpiler=seed_transpiler,
        optimization_level=optimization_level,
    )


def prepare_vqc_for_target(
    backend,
    *,
    seed_transpiler: int = SEED_TRANSPILER,
    optimization_level: int = OPTIMIZATION_LEVEL,
) -> PreparedVQC:
    """Prepare the VQC for an already selected real or fake backend."""

    circuit, input_parameters, weight_parameters = build_qnetwork_circuit()
    original_depth = circuit.depth()
    expected_parameters = set(input_parameters) | set(weight_parameters)

    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    isa_circuit = pass_manager.run(circuit)

    # Transpilation must preserve the Parameter objects so later binding by
    # explicit parameter-to-value dictionaries cannot silently change order.
    actual_parameters = set(isa_circuit.parameters)
    if actual_parameters != expected_parameters:
        missing = sorted(str(p) for p in expected_parameters - actual_parameters)
        extra = sorted(str(p) for p in actual_parameters - expected_parameters)
        raise RuntimeError(
            "Transpilation changed the VQC parameter set. "
            f"Missing={missing}, extra={extra}"
        )

    mapped_observables = tuple(
        observable.apply_layout(isa_circuit.layout)
        for observable in build_observables()
    )
    if len(mapped_observables) != 2:
        raise RuntimeError("Expected exactly two action observables.")
    if any(observable.num_qubits != isa_circuit.num_qubits for observable in mapped_observables):
        raise RuntimeError("Observable layout does not match the ISA circuit width.")

    two_qubit_gates = sum(
        1 for instruction in isa_circuit.data if instruction.operation.num_qubits == 2
    )

    return PreparedVQC(
        backend=backend,
        circuit=isa_circuit,
        observables=mapped_observables,
        input_parameters=tuple(input_parameters),
        weight_parameters=tuple(weight_parameters),
        original_depth=original_depth,
        transpiled_depth=isa_circuit.depth(),
        two_qubit_gates=two_qubit_gates,
    )


def _format_summary(prepared: PreparedVQC) -> str:
    status = prepared.backend.status()
    return (
        "Q8.2 preparation successful.\n\n"
        f"Backend: {prepared.backend.name}\n"
        f"Backend qubits: {prepared.backend.num_qubits}\n"
        f"Pending jobs: {status.pending_jobs}\n"
        f"Original circuit depth: {prepared.original_depth}\n"
        f"ISA circuit depth: {prepared.transpiled_depth}\n"
        f"Two-qubit gates: {prepared.two_qubit_gates}\n"
        f"Input parameters: {len(prepared.input_parameters)} (expected 12)\n"
        f"Weight parameters: {len(prepared.weight_parameters)} (expected 32)\n"
        f"Mapped observables: {len(prepared.observables)} (expected 2)\n\n"
        "No Runtime job was submitted and no QPU execution time was used."
    )


def build_smoke_test_inputs(
    prepared: PreparedVQC,
    observation: tuple[float, ...] = SMOKE_TEST_OBSERVATION,
    *,
    seed: int = 42,
) -> SmokeTestInputs:
    """Create fixed parameters and verify them with exact local inference."""
    if len(observation) != 4:
        raise ValueError("The CartPole observation must contain four values.")

    # Match VQCQNetwork's initialized parameters without constructing its QNN.
    # The smoke-test input and output scales both initialize to one.
    encoded = np.arctan(np.asarray(observation, dtype=float))
    input_angles = np.tile(encoded, 3)
    theta_values = np.random.default_rng(seed).uniform(
        -np.pi, np.pi, size=len(prepared.weight_parameters)
    )
    output_scale = np.ones(2, dtype=float)

    values_by_parameter = {
        **dict(zip(prepared.input_parameters, input_angles, strict=True)),
        **dict(zip(prepared.weight_parameters, theta_values, strict=True)),
    }
    parameter_values = tuple(
        float(values_by_parameter[parameter])
        for parameter in prepared.circuit.parameters
    )

    # Simulate the original four logical qubits.  The transpiled IBM circuit
    # spans every physical backend wire (156 on ibm_marrakesh), which is not a
    # tractable statevector even though only four physical qubits are active.
    local_circuit, local_inputs, local_weights = build_qnetwork_circuit()
    local_values_by_parameter = {
        **dict(zip(local_inputs, input_angles, strict=True)),
        **dict(zip(local_weights, theta_values, strict=True)),
    }
    local_parameter_values = tuple(
        float(local_values_by_parameter[parameter])
        for parameter in local_circuit.parameters
    )
    local_estimator = StatevectorEstimator(seed=seed)
    local_result = local_estimator.run(
        [(local_circuit, build_observables(), [local_parameter_values])]
    ).result()[0]
    local_expectations = tuple(
        float(value) for value in np.asarray(local_result.data.evs).reshape(-1)
    )
    if len(local_expectations) != 2 or not np.isfinite(local_expectations).all():
        raise RuntimeError(f"Unexpected local Estimator result: {local_expectations}")

    output_values = tuple(float(value) for value in output_scale)
    local_q_values = tuple(
        expectation * scale
        for expectation, scale in zip(local_expectations, output_values, strict=True)
    )
    return SmokeTestInputs(
        observation=tuple(float(value) for value in observation),
        parameter_values=parameter_values,
        output_scale=output_values,
        local_expectations=local_expectations,
        local_q_values=local_q_values,
    )


def _result_path(job_id: str) -> Path:
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    safe_job_id = "".join(character for character in job_id if character.isalnum() or character in "-_")
    return RESULTS_DIRECTORY / f"ibm_smoke_test_{safe_job_id}.json"


def run_runtime_smoke_test(
    prepared: PreparedVQC,
    inputs: SmokeTestInputs,
    *,
    target_precision: float = TARGET_PRECISION,
    on_submitted: Callable[[str, Path], None] | None = None,
) -> tuple[dict, Path]:
    """Submit one confirmed Runtime job and save reproducibility metadata."""
    if target_precision <= 0:
        raise ValueError("Target precision must be greater than zero.")

    estimator = EstimatorV2(mode=prepared.backend)
    job = estimator.run(
        [(prepared.circuit, list(prepared.observables), [inputs.parameter_values])],
        precision=target_precision,
    )
    job_id = job.job_id()
    result_path = _result_path(job_id)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "submitted",
        "backend": prepared.backend.name,
        "job_id": job_id,
        "target_precision": target_precision,
        "observation": inputs.observation,
        "seed": 42,
        "original_depth": prepared.original_depth,
        "isa_depth": prepared.transpiled_depth,
        "two_qubit_gates": prepared.two_qubit_gates,
        "local_expectations": inputs.local_expectations,
        "local_q_values": inputs.local_q_values,
    }
    result_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if on_submitted is not None:
        on_submitted(job_id, result_path)

    runtime_result = job.result()[0]
    expectations = tuple(
        float(value) for value in np.asarray(runtime_result.data.evs).reshape(-1)
    )
    if len(expectations) != 2 or not np.isfinite(expectations).all():
        raise RuntimeError(f"Unexpected Runtime Estimator result: {expectations}")

    q_values = tuple(
        expectation * scale
        for expectation, scale in zip(expectations, inputs.output_scale, strict=True)
    )
    metadata.update(
        {
            "status": "completed",
            "runtime_expectations": expectations,
            "runtime_q_values": q_values,
            "selected_action": int(np.argmax(q_values)),
        }
    )
    result_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata, result_path


def main() -> None:
    """Run Q8.2 through dialogs so no terminal input is required."""
    # Imported here, not at module scope: prepare_vqc_for_backend() is called
    # headlessly by run_experiment.py --arms ibm, and a Python built without
    # Tk would otherwise make this whole module unimportable.
    import tkinter as tk
    from tkinter import messagebox, simpledialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        selected_name = simpledialog.askstring(
            "IBM backend preparation",
            "Backend name (leave blank to automatically choose the least busy QPU):",
            initialvalue=DEFAULT_BACKEND,
            parent=root,
        )
        if selected_name is None:
            return

        prepared = prepare_vqc_for_backend(selected_name.strip() or None)
        messagebox.showinfo(
            "IBM backend preparation",
            _format_summary(prepared),
            parent=root,
        )

        inputs = build_smoke_test_inputs(prepared)
        should_submit = messagebox.askyesno(
            "Submit IBM Runtime smoke test?",
            "Q8.2 is complete.\n\n"
            f"Backend: {prepared.backend.name}\n"
            f"Observation: {inputs.observation}\n"
            f"Target precision: {TARGET_PRECISION}\n"
            f"Local Q-values: {tuple(round(v, 6) for v in inputs.local_q_values)}\n\n"
            "Selecting Yes will submit a real IBM Quantum job, use QPU "
            "execution time, and might require waiting in the queue.\n\n"
            "Submit the job now?",
            parent=root,
        )
        if not should_submit:
            messagebox.showinfo(
                "No job submitted",
                "Q8.2 remains complete. No Runtime job was submitted.",
                parent=root,
            )
            return

        def show_submitted(job_id: str, result_path: Path) -> None:
            messagebox.showinfo(
                "IBM Runtime job submitted",
                f"Job ID: {job_id}\n\n"
                "The job was submitted successfully. After closing this "
                "message, the program will wait for the queued job to finish.\n\n"
                f"Submission metadata saved to:\n{result_path}",
                parent=root,
            )

        metadata, result_path = run_runtime_smoke_test(
            prepared,
            inputs,
            on_submitted=show_submitted,
        )
        messagebox.showinfo(
            "IBM Runtime smoke test complete",
            f"Backend: {metadata['backend']}\n"
            f"Job ID: {metadata['job_id']}\n"
            f"Expectation values: {metadata['runtime_expectations']}\n"
            f"Q-values: {metadata['runtime_q_values']}\n"
            f"Selected action: {metadata['selected_action']}\n\n"
            f"Results saved to:\n{result_path}",
            parent=root,
        )
    except Exception as error:
        messagebox.showerror(
            "IBM backend preparation failed",
            f"{type(error).__name__}:\n\n{error}",
            parent=root,
        )
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
