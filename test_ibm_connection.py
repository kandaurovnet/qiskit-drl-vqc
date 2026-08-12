"""Test an IBM Quantum account and select an operational QPU."""

import tkinter as tk
from tkinter import messagebox

from qiskit_ibm_runtime import QiskitRuntimeService


ACCOUNT_NAME = "cartpole-vqc"


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        service = QiskitRuntimeService(name=ACCOUNT_NAME)
        backends = service.backends(
            simulator=False,
            operational=True,
            min_num_qubits=4,
        )
        if not backends:
            messagebox.showwarning(
                "No QPU available",
                "The account connected successfully, but no operational QPU "
                "with at least four qubits was found.",
                parent=root,
            )
            return

        backend_lines = []
        for backend in backends:
            status = backend.status()
            backend_lines.append(
                f"{backend.name}: {backend.num_qubits} qubits, "
                f"{status.pending_jobs} pending jobs"
            )

        selected = service.least_busy(
            simulator=False,
            operational=True,
            min_num_qubits=4,
        )
        selected_status = selected.status()
        summary = (
            "Connection successful.\n\n"
            f"Selected backend: {selected.name}\n"
            f"Qubits: {selected.num_qubits}\n"
            f"Pending jobs: {selected_status.pending_jobs}\n\n"
            "Accessible operational QPUs:\n"
            + "\n".join(backend_lines)
        )
        messagebox.showinfo("IBM Quantum connection", summary, parent=root)
    except Exception as error:
        messagebox.showerror(
            "Connection failed",
            f"{type(error).__name__}:\n\n{error}",
            parent=root,
        )
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
