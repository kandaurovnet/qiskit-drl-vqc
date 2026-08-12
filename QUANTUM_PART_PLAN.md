# Quantum Part Plan and Progress

This document tracks the quantum-model work for the CartPole VQC-QDQN project.
The quantum component is responsible for converting a batch of four-dimensional
CartPole observations into two trainable Q-values, one for each action.

## Current Status

Branch: `quantum-part--seanliu`

The initial VQC prototype and the first CartPole quantum Q-network are complete.
The current `vqc.py` maps batched four-dimensional observations to two
differentiable Q-values, uses data re-uploading, and exposes PyTorch-compatible
parameters. The next milestone is integration testing with the CartPole DQN and
adding a non-re-uploading configuration for a controlled ablation study.

| Phase | Task | Status | Deliverable / Completion Criterion |
|---|---|---|---|
| Q0 | Build a basic four-qubit VQC | **Completed** | Four input features, two trainable layers, and selectable CX entanglement |
| Q1 | Connect Qiskit to PyTorch | **Completed** | Differentiable `EstimatorQNN` wrapped with `TorchConnector` |
| Q2 | Verify basic training and gradients | **Completed** | Reproducible synthetic classification training with decreasing loss |
| Q3 | Convert the prototype into a two-output quantum Q-network | **Completed** | Batched `[batch, 4] -> [batch, 2]` forward pass with two observables |
| Q4 | Add CartPole observation encoding and scaling | **Completed** | Bounded `arctan` encoding with trainable per-layer input scaling |
| Q5 | Provide the QDQN integration interface | **In progress** | Forward and optimizer interface complete; target-copy and save/load tests remain |
| Q6 | Add data re-uploading | **In progress** | Re-uploading model complete; equivalent basic-model switch remains |
| Q7 | Report model complexity | **In progress** | Parameter groups documented; automated depth and two-qubit gate report remains |
| Q8 | Evaluate finite-shot robustness | **Planned** | Action agreement and Q-value error across multiple shot counts |
| Q9 | Evaluate noise and IBM hardware inference | **Planned** | Exact/noisy/hardware comparison on representative CartPole states |

## Completed Work

### Q0: Basic VQC prototype

The current implementation in `vqc_v0.py` contains:

- Four qubits and four input parameters.
- One `RY(x)` data-encoding stage.
- Two trainable layers.
- Trainable `RY(theta)` and `RZ(theta)` gates on every qubit in each layer.
- Linear or circular CX entanglement.
- Sixteen trainable circuit parameters.
- A Pauli-Z expectation-value readout on qubit 0.

### Q1: Qiskit-PyTorch integration

The prototype uses `EstimatorQNN` and `TorchConnector` to expose the VQC as a
PyTorch `nn.Module`. It supports:

- PyTorch forward propagation.
- Automatic differentiation through `loss.backward()`.
- Parameter updates with the Adam optimizer.
- Reproducible parameter initialization using fixed NumPy and PyTorch seeds.

### Q2: Minimal learning verification

The model was trained on a synthetic four-dimensional binary-classification
task. The test verifies that the circuit, connector, gradient path, and
optimizer work together. In the recorded run, the loss decreased from `0.6617`
to `0.5208`, and training accuracy reached a maximum of `82%`.

This result validates the implementation pipeline, but it is not yet a
CartPole or reinforcement-learning result.

## Completed CartPole Quantum Q-Network

The original `vqc_v0.py` prototype produces one probability-like output:

```text
[batch, 4] -> [batch, 1]
```

The new `VQCQNetwork` in `vqc.py` now produces the two unrestricted Q-values
required by the CartPole DQN:

```text
[batch, 4] -> [batch, 2]
                  |      |
                  |      +-- Q(state, right)
                  +--------- Q(state, left)
```

The `[0, 1]` classification mapping is not used in the Q-network. The new model
measures `Z0Z1` and `Z2Z3`, then applies trainable output scaling:

```text
z0 = <Z0 Z1>
z1 = <Z2 Z3>

Q_left  = scale_left  * z0
Q_right = scale_right * z1
```

Verified or implemented behavior:

- Batched CartPole-like observations produce two Q-values per state.
- `input_gradients=True` allows gradients to reach the trainable input scales.
- The circuit design was checked for structurally dead parameters with
  finite-difference gradients.
- Circular CZ entanglement and complementary observables give all four qubits
  a path to at least one output.

Remaining integration tests:

- Test input shapes `[1, 4]`, `[8, 4]`, and `[32, 4]` automatically.
- Verify a complete optimizer step against a small Q-value target.
- Verify exact online-to-target copying and save/load equivalence.

## Planned Quantum Model Interface

The classical DQN code should be able to use the quantum model like a normal
PyTorch network:

```python
model = VQCQNetwork(
    n_qubits=4,
    n_layers=3,
    seed=0,
)

q_values = model(states)  # states: [batch, 4], q_values: [batch, 2]
```

The implementation must also support:

```python
optimizer = torch.optim.Adam(model.parameters())
target_model.load_state_dict(model.state_dict())
torch.save(model.state_dict(), path)
model.load_state_dict(torch.load(path))
```

## Completed CartPole Observation Encoding

The current model maps all four observations to bounded angles with `arctan`,
then multiplies each feature by a trainable scale for every upload layer:

```python
encoded = torch.arctan(states)
angles = input_scale * encoded
```

The preprocessing implementation must:

- Support both individual states and batches.
- Produce no `NaN` or infinite values.
- Use the same transformation during training, simulation, and hardware tests.
- Count the trainable input scales as model parameters.

## Data Re-uploading Experiment

The data-re-uploading Q-network is implemented. A matching basic configuration
still needs to be added so the two architectures can be compared fairly.

Basic VQC:

```text
Encode(x) -> Trainable layer 1 -> Trainable layer 2 -> Measure
```

Data-re-uploading VQC:

```text
Encode(x) -> Trainable layer 1 -> Encode(x) -> Trainable layer 2 -> Measure
```

For a fair ablation study, both configurations will use the same:

- Number of qubits.
- Number of trainable layers.
- Number of trainable circuit parameters.
- Entanglement pattern.
- Observables and output head.
- Initialization seed and DQN training settings.

The data inputs are not trainable parameters, so adding a second encoding stage
does not increase the reported trainable parameter count.

## Model Complexity Report

Each model configuration will report at least:

| Metric | Basic VQC | Data-re-uploading VQC |
|---|---:|---:|
| Qubits | 4 | 4 |
| Trainable circuit parameters | To be matched | 32 |
| Trainable input scales | To be matched | 12 |
| Output-scale parameters | 2 | 2 |
| Total trainable parameters | To be matched | 46 |
| Data-upload stages | 1 | 2 |
| Circuit depth | To be measured | To be measured |
| Two-qubit gate count | To be measured | To be measured |

Parameter efficiency will be treated as a model-compression comparison, not as
evidence of quantum advantage. Quantum and classical parameters do not have
equal execution costs.

## Finite-Shot and Noise Evaluation

Full QDQN training will initially use an exact simulator. After training, the
model parameters will be frozen and evaluated under:

1. Exact expectation values.
2. 64 shots.
3. 256 shots.
4. 1,024 shots.
5. A realistic noisy simulation.
6. IBM quantum hardware, subject to access and queue availability.

Primary robustness metrics:

- Average CartPole evaluation reward.
- Q-value estimation error relative to exact simulation.
- Action agreement with the exact model.
- Performance degradation versus shot count.
- Relationship between action errors and the Q-value margin.

Hardware execution is intended for inference validation on representative
states. Full reinforcement-learning training will not be performed on the QPU.

## Immediate Work Order

1. Add automated batch-shape, gradient, optimizer, copy, and save/load tests.
2. Add a non-re-uploading configuration with a controlled parameter budget.
3. Automate total parameter, circuit-depth, and two-qubit gate reporting.
4. Integrate `VQCQNetwork` with the classical CartPole DQN pipeline.
5. Confirm that reward improves under exact-simulator training.
6. Compare the basic and data-re-uploading models under identical DQN settings.
7. Run finite-shot and noisy inference after successful exact-simulator training.
8. Perform IBM hardware inference on selected evaluation states.

## Definition of the Next Handoff

The next quantum-part handoff is complete when the branch provides a reusable
four-qubit `QuantumQNetwork` that:

- Accepts batched CartPole observations with shape `[batch, 4]`.
- Returns two differentiable Q-values with shape `[batch, 2]`.
- Works with a standard PyTorch optimizer.
- Can be copied into a target network and saved or loaded with `state_dict`.
- Supports matched basic and data-re-uploading circuit configurations.
- Reports its trainable parameter count and circuit complexity.

