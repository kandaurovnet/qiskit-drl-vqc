"""Save IBM Quantum credentials using graphical input dialogs."""

import tkinter as tk
from tkinter import messagebox, simpledialog

from qiskit_ibm_runtime import QiskitRuntimeService


ACCOUNT_NAME = "cartpole-vqc"


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        api_key = simpledialog.askstring(
            "IBM Quantum account setup",
            "Paste your IBM Cloud API key:",
            show="*",
            parent=root,
        )
        if not api_key or not api_key.strip():
            messagebox.showwarning("Setup cancelled", "No API key was entered.", parent=root)
            return

        crn = simpledialog.askstring(
            "IBM Quantum account setup",
            "Paste your IBM Quantum instance CRN:",
            parent=root,
        )
        if not crn or not crn.strip():
            messagebox.showwarning("Setup cancelled", "No instance CRN was entered.", parent=root)
            return

        QiskitRuntimeService.save_account(
            channel="ibm_quantum_platform",
            token=api_key.strip(),
            instance=crn.strip(),
            name=ACCOUNT_NAME,
            set_as_default=True,
            overwrite=True,
        )
        messagebox.showinfo(
            "Setup complete",
            f'IBM Quantum account "{ACCOUNT_NAME}" was saved successfully.',
            parent=root,
        )
    except Exception as error:
        messagebox.showerror(
            "Setup failed",
            f"{type(error).__name__}:\n\n{error}",
            parent=root,
        )
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
