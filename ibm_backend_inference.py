"""Prepare the CartPole VQC for an IBM Quantum backend.

This module implements milestone Q8.2 only.  Running it connects to IBM
Quantum, selects a QPU, transpiles the parameterized VQC to that QPU's ISA,
maps both observables to the transpiled layout, and reports validation data.
It does not submit a Runtime job or consume QPU execution time.
"""

from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox, simpledialog

from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService

from vqc import build_observables, build_qnetwork_circuit


ACCOUNT_NAME = "cartpole-vqc"
DEFAULT_BACKEND = "ibm_marrakesh"
SEED_TRANSPILER = 42
OPTIMIZATION_LEVEL = 3


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


def main() -> None:
    """Run Q8.2 through dialogs so no terminal input is required."""
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
