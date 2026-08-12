# Quantum Part Plan and Progress

This document tracks the quantum-model work for the CartPole VQC-QDQN project.
It reflects the current implementation on branch `quantum-part--seanliu`.

## Current Status

The CartPole VQC Q-network is implemented in `vqc.py`. It already includes
data re-uploading and maps a batch of four-dimensional CartPole observations to
two differentiable Q-values. The next quantum milestone is to add a separate
IBM Runtime inference path that can execute the trained circuit on an IBM
backend without changing the reinforcement-learning interface.

| Phase | Task | Status | Evidence / Deliverable |
|---|---|---|---|
| Q0 | Build a basic four-qubit VQC prototype | **Completed** | `vqc_v0.py` |
| Q1 | Connect Qiskit to PyTorch | **Completed** | `EstimatorQNN` and `TorchConnector` |
| Q2 | Verify basic learning and gradients | **Completed** | Reproducible synthetic training and gradient checks |
| Q3 | Build the CartPole quantum Q-network | **Completed** | Batched `[batch, 4] -> [batch, 2]` forward pass in `vqc.py` |
| Q4 | Add CartPole observation encoding and scaling | **Completed** | Bounded `arctan` encoding and trainable input scaling |
| Q5 | Add data re-uploading | **Completed** | Scaled `RX` encoding is repeated in every variational block |
| Q6 | Design gradient-safe circuit and readout | **Completed** | Circular CZ ring, `RZ-RY` order, and two complementary observables |
| Q7 | Document model architecture and parameter count | **Completed** | `vqc_documentation.md`; 46 trainable parameters |
| Q8 | Connect to an IBM backend | **Next** | Runtime service, backend selection, ISA transpilation, and one inference job |
| Q9 | Add finite-shot and hardware evaluation | **Planned** | Exact/noisy/hardware action-agreement results |
| Q10 | Integrate trained parameters with backend inference | **Planned** | Frozen trained VQC evaluated on representative CartPole states |

## Completed VQC Architecture

The production CartPole model is `VQCQNetwork` in `vqc.py`:

```text
(batch, 4) CartPole observations
        |
bounded arctan encoding + trainable input scaling
        |
[RZ-RY -> circular CZ -> RX data re-uploading] x 3
        |
final RZ-RY layer
        |
measure Z0Z1 and Z2Z3
        |
trainable output scaling
        |
(batch, 2) Q-values
```

The current trainable parameter count is:

| Parameter group | Count |
|---|---:|
| Circuit rotations (`theta`) | 32 |
| Input scaling (`lambda`) | 12 |
| Output scaling (`w`) | 2 |
| **Total** | **46** |

### Data re-uploading is complete

Data re-uploading is not a future task. `build_qnetwork_circuit()` creates one
set of input parameters for every layer and qubit:

```python
x_params = ParameterVector("x", n_layers * n_qubits)
```

Every variational block encodes the scaled observation again:

```python
for q in range(n_qubits):
    qc.rx(x_params[layer * n_qubits + q], q)
```

With three layers and four features, the state is uploaded three times through
12 encoding angles. The trainable `input_scale` parameters allow the model to
learn a different scale for every layer-feature pair.

## Next Milestone: IBM Backend Connection

The first backend milestone is inference only. Full QDQN training will remain
on the exact local estimator because hardware training would require too many
queued circuit and gradient evaluations.

The IBM execution path should be kept separate from `VQCQNetwork.forward()`:

```text
trained VQCQNetwork state_dict
        |
selected CartPole states
        |
bind input-scale, circuit, and output-scale parameters
        |
transpile to an IBM backend ISA circuit
        |
Runtime EstimatorV2
        |
hardware expectation values
        |
Q(left), Q(right) and selected action
```

### Q8.1 Install and configure Runtime

Add `qiskit-ibm-runtime` to `requirements.txt` and verify:

- `QiskitRuntimeService()` can load the user's saved account.
- The correct NTU/IBM instance is selected when required.
- At least one operational non-simulator backend is visible.
- No API token is stored in the repository or committed to Git.

Completion criterion:

```text
Print the accessible backend names and select one backend successfully.
```

### Q8.2 Prepare the VQC for the backend

IBM Runtime V2 requires an ISA-compatible circuit and observables. The backend
adapter must:

1. Build the same parameterized circuit used by `VQCQNetwork`.
2. Generate a preset pass manager for the selected backend.
3. Transpile the circuit with a fixed `seed_transpiler`.
4. Apply the transpiled layout to both observables.
5. Preserve and verify the parameter ordering used by the model.
6. Record transpiled depth and two-qubit gate count.

Completion criterion:

```text
The transpiled circuit passes backend compatibility checks, and both
observables use the transpiled circuit layout.
```

### Q8.3 Run a minimal Runtime Estimator job

Before loading trained RL parameters, run one small smoke test:

- Use one CartPole-like observation.
- Use a known or initialized set of VQC parameters.
- Submit the two observables through `EstimatorV2` in job mode.
- Retrieve two expectation values.
- Apply the model's output scaling to produce two Q-values.
- Save the backend name, job ID, target precision, and results.

Completion criterion:

```text
One IBM Runtime job returns two finite expectation values and two finite
Q-values for the same VQC architecture used locally.
```

### Q8.4 Validate local-versus-hardware parameter binding

For the same state and frozen model parameters:

1. Evaluate the model with the current exact local estimator.
2. Evaluate the bound circuit with Runtime EstimatorV2.
3. Confirm that both paths use the same encoding angles and `theta` ordering.
4. Compare Q-values and selected actions.

This validation is essential because a parameter-order mismatch can produce a
valid hardware result for the wrong circuit without raising an error.

## Planned Hardware Robustness Evaluation

After the classical team produces a trained model checkpoint:

1. Load and freeze the trained `state_dict`.
2. Collect representative CartPole evaluation states.
3. Evaluate the states with exact local expectation values.
4. Evaluate them with finite-shot or noisy simulation.
5. Evaluate a manageable subset on IBM hardware.
6. Compare Q-values and actions across execution modes.

Recommended state groups:

- Easy states with a large Q-value margin.
- Difficult states where the two Q-values are close.
- Left-leaning and right-leaning pole states.
- States near episode termination boundaries.

Primary metrics:

```text
action agreement = hardware action == exact action
Q-value error     = hardware Q-value - exact Q-value
Q-value margin    = abs(Q_left - Q_right)
```

Report at least:

- Exact-versus-hardware action agreement.
- Mean absolute Q-value error.
- Whether action errors concentrate at small Q-value margins.
- Backend name, job IDs, target precision, circuit depth, and two-qubit gates.

## Immediate Work Order

1. Add `qiskit-ibm-runtime` to the project requirements.
2. Create a separate IBM backend inference module or script.
3. Verify account and NTU/IBM instance access without exposing credentials.
4. Select an operational backend with at least four qubits.
5. Transpile the VQC and map `Z0Z1` and `Z2Z3` to the resulting layout.
6. Verify model-to-circuit parameter ordering.
7. Submit one minimal EstimatorV2 smoke-test job.
8. Save the job metadata and compare its output with local exact inference.
9. Once a trained checkpoint exists, run the planned robustness evaluation.

## Definition of the Next Handoff

The IBM-backend milestone is complete when the branch contains a documented,
reproducible inference path that:

- Uses the same VQC architecture and parameter values as `VQCQNetwork`.
- Connects through `QiskitRuntimeService` without storing credentials in Git.
- Selects an IBM backend and produces an ISA circuit and mapped observables.
- Submits an `EstimatorV2` job for both action observables.
- Converts the returned expectations into two Q-values.
- Records sufficient metadata to reproduce and compare the hardware result.

