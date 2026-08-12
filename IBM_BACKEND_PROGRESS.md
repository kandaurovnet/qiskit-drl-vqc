# IBM Backend Integration Progress

This document summarizes the current IBM Quantum backend integration work for
the CartPole VQC-QDQN project. Hardware execution is intentionally paused at
the team's request. No Runtime Estimator job has been submitted so far.

## Scope and execution policy

IBM Quantum hardware is intended for inference and validation only. Full QDQN
training remains on the exact local estimator because hardware training would
require a large number of queued circuit and gradient evaluations.

The intended execution flow is:

```text
trained or frozen VQC parameters
        |
selected CartPole observations
        |
IBM-backend ISA transpilation and observable layout
        |
Runtime EstimatorV2 inference
        |
Q(left), Q(right), and selected action
```

Only the connection and backend-preparation portions of this flow have been
completed. Real QPU execution is deferred.

## Q8.1: Account setup and backend connection

**Status: Completed**

The following work is complete:

- Created and configured an IBM Cloud API key and IBM Quantum service
  instance.
- Saved the credentials locally through `QiskitRuntimeService.save_account()`
  under the account name `cartpole-vqc`.
- Kept the API key and instance CRN out of source files and Git history.
- Added `setup_ibm_account.py`, which uses graphical dialogs to save account
  credentials without requiring terminal input.
- Added `test_ibm_connection.py`, which lists accessible operational QPUs and
  selects a least-busy backend.
- Confirmed access to operational IBM Quantum processors.
- Successfully selected `ibm_marrakesh`, a 156-qubit QPU.

The accessible QPUs observed during connection testing included:

- `ibm_pittsburgh`
- `ibm_boston`
- `ibm_fez`
- `ibm_miami`
- `ibm_marrakesh`
- `ibm_kingston`

Relevant commit:

```text
888b10a Add IBM Quantum account setup and backend connection test
```

Querying backend information and selecting a backend does not submit a Runtime
job or consume QPU execution time.

## Q8.2: ISA circuit and observable preparation

**Status: Completed**

The project now contains `ibm_backend_inference.py`, which prepares the
existing VQC architecture for an IBM backend without submitting a job. It:

1. Loads the locally saved `cartpole-vqc` account.
2. Uses a requested backend or selects a least-busy operational QPU.
3. Builds the same parameterized circuit used by `VQCQNetwork`.
4. Generates a preset pass manager for the selected backend.
5. Transpiles the VQC with a fixed transpiler seed.
6. Applies the transpiled layout to both action observables, `Z0Z1` and
   `Z2Z3`.
7. Verifies that all input and circuit-weight parameters survive
   transpilation.
8. Reports the original and ISA circuit depths and the two-qubit gate count.

The successful preparation result for `ibm_marrakesh` was:

```text
Backend: ibm_marrakesh
Backend qubits: 156
Original circuit depth: 24
ISA circuit depth: 93
Two-qubit gates: 34
Input parameters: 12 / 12
Weight parameters: 32 / 32
Mapped observables: 2 / 2
```

These results confirm that:

- all 12 data re-uploading input parameters are preserved;
- all 32 variational circuit weights are preserved;
- both action observables use the transpiled physical-qubit layout; and
- the circuit is prepared in the target backend's ISA.

Relevant commit:

```text
7eea57c Prepare VQC for IBM backend ISA execution
```

Q8.2 performs account access, backend queries, transpilation, and validation
only. It does not consume QPU execution time.

## Q8.3: Runtime Estimator smoke test

**Status: Deferred**

The planned smoke test will use one fixed CartPole-like observation and one
fixed set of VQC parameters to verify the complete execution path:

```text
CartPole observation
        |
input-angle and VQC-weight binding
        |
IBM Runtime EstimatorV2
        |
two expectation values
        |
two scaled Q-values and a selected action
```

When resumed, the test should record:

- backend name;
- Runtime job ID;
- target precision;
- circuit depth and two-qubit gate count;
- local exact expectation values and Q-values;
- hardware expectation values and Q-values; and
- the action selected by each execution path.

The team has decided not to submit this real-hardware job yet. Consequently:

- no Runtime job ID exists;
- no QPU execution time has been consumed;
- no hardware expectation values have been collected; and
- no local-versus-hardware comparison is reported.

## Q8.4: Local-versus-hardware validation

**Status: Not started**

After Q8.3 is approved and completed, the same frozen observation and model
parameters will be evaluated locally and on IBM hardware. The comparison will
verify parameter ordering, expectation values, scaled Q-values, and action
agreement. This step is required before evaluating a larger set of trained
CartPole states on hardware.

## Current milestone summary

| Item | Status |
|---|---|
| Q8.1 IBM account and backend connection | Completed and pushed |
| Q8.2 ISA transpilation and observable mapping | Completed and pushed |
| Q8.3 Runtime Estimator smoke test | Deferred; no job submitted |
| Q8.4 Local-versus-hardware comparison | Not started |

The current branch demonstrates that the project can authenticate with IBM
Quantum, discover operational QPUs, select `ibm_marrakesh`, and produce a
validated ISA circuit with correctly mapped observables. Work can resume from
Q8.3 later without repeating account setup or backend preparation.

