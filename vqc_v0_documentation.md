# VQC v0 Architecture and Execution Results

`vqc_v0.py` is a minimal Variational Quantum Circuit (VQC) example built with Qiskit and PyTorch. It wraps a parameterized quantum circuit as a PyTorch model and trains it on a synthetic binary classification task.

It is a teaching example. The Q-network used by the CartPole agent lives in `vqc.py`, which adds data re-uploading, two observables, and trainable input/output scaling.

> **Revision note.** The first version of this circuit used a linear CX chain and applied `RY` before `RZ`. Both choices were bugs: 11 of the 16 trainable parameters had exactly zero gradient and two of the four input features could not affect the output at all. Section 5 explains the diagnosis; the results in section 4 are from the corrected circuit.

## 1. Model Specifications

- Number of qubits: 4
- Number of input features: 4
- Data encoding: one `RY(x[i])` gate on each qubit
- Number of trainable layers: 2
- Trainable gates per layer: `RZ(theta)` then `RY(theta)`
- Entanglement pattern: circular CX chain (`q0 -> q1 -> q2 -> q3 -> q0`)
- Total number of trainable parameters: 16, of which 14 carry gradient
- Output observable: Pauli-Z expectation value on qubit 0
- Optimizer: Adam with a learning rate of 0.1
- Loss function: Binary Cross-Entropy Loss
- Training duration: 30 epochs
- Circuit image: regenerated as `vqc_v0_circuit.png` whenever the program runs

## 2. Program Architecture

### 2.1 Dependencies and Random Seeds

The program uses NumPy to generate data and initialize parameters, PyTorch to manage the model, loss function, and optimization process, and Qiskit to construct the parameterized quantum circuit.

The NumPy and PyTorch random seeds are both set to `0`, making the generated dataset, initial weights, and training results reproducible.

### 2.2 Hyperparameters

```python
N_QUBITS = 4
N_LAYERS = 2
ENTANGLE_MODE = "circular"
CIRCUIT_IMAGE = "vqc_v0_circuit.png"
```

`ENTANGLE_MODE` can be set to either `linear` or `circular`. Circular mode adds a CX gate from the final qubit back to the first qubit in every trainable layer.

`CIRCUIT_IMAGE` defines the output path used by `qc.draw("mpl", ...)`. The diagram embedded in this document is therefore generated directly by the same circuit-construction function used for training.

**Circular is required, not a preference.** A linear chain runs `q0 -> q1 -> q2 -> q3`, so qubit 0 is only ever a control and never a target. Because the output observable is measured on qubit 0, its backward light cone never reaches the upper qubits. Setting `linear` reproduces the original bug.

### 2.3 VQC Construction

The `build_vqc_circuit()` function creates two parameter groups:

- `x_params`: four parameters representing the input data.
- `theta_params`: the trainable model weights, with `2 layers x 4 qubits x 2 rotations = 16` parameters.

The circuit first applies `RY(x[i])` gates to encode the input features into the quantum state. Each trainable layer then applies `RZ` and `RY` rotations, followed by CX gates that create entanglement between the qubits.

The rotation order matters. `RZ` commutes with any Z-type observable, so an `RZ` placed last in the circuit has exactly zero gradient. Applying `RY` last ensures the final rotation does not commute with the measurement.

### 2.4 PyTorch Model Wrapper

`VQCModel` inherits from `torch.nn.Module`. It uses `TorchConnector` to convert the `EstimatorQNN` into a quantum layer that participates in PyTorch automatic differentiation.

The Pauli-Z expectation value produced by the quantum circuit is in the range `[-1, 1]`. The forward pass maps it to `[0, 1]`:

```python
return (raw + 1) / 2
```

The mapped value can be used directly as a binary classification probability.

### 2.5 Observable

```python
observable = SparsePauliOp.from_list([("IIIZ", 1)])
```

Qiskit Pauli strings use little-endian qubit ordering. Therefore, `IIIZ` represents a Pauli-Z measurement on qubit 0.

### 2.6 Synthetic Dataset

The program generates 40 four-dimensional samples. Each feature is drawn uniformly from `[-pi, pi]`. Labels are assigned using the following rule:

```text
sum(x) > 0  -> label 1
sum(x) <= 0 -> label 0
```

This is a simple linearly separable task designed to verify that the VQC and PyTorch integration, forward pass, and backward pass work correctly.

### 2.7 Training Process

Each epoch performs the following steps:

1. Clear the gradients from the previous iteration.
2. Pass all 40 training samples through the VQC.
3. Calculate the Binary Cross-Entropy Loss.
4. Use `loss.backward()` to calculate gradients for the quantum-circuit parameters.
5. Use Adam to update the `theta` parameters.
6. Display the loss and accuracy every five epochs.

## 3. Quantum Circuit Diagram

![VQC v0 quantum circuit](vqc_v0_circuit.png)

The circuit proceeds from left to right as follows:

1. Encode the input with `RY(x[0...3])` gates.
2. Apply the first trainable `RZ(theta) + RY(theta)` rotation layer.
3. Apply the first circular CX chain: `q0 -> q1 -> q2 -> q3 -> q0`.
4. Apply the second trainable `RZ(theta) + RY(theta)` rotation layer.
5. Apply the second circular CX chain.
6. Measure the Pauli-Z expectation value on qubit 0 as the model output.

Running `python vqc_v0.py` regenerates this figure, so it cannot drift from the code.

## 4. Execution Output

Running the current circular `RZ -> RY` model for 30 epochs (Qiskit 2.5.1, Qiskit Machine Learning 0.9.0, and PyTorch 2.13.0) produced the following reproducible output:

```text
Epoch  0 | Loss: 0.6741 | Acc: 0.57
Epoch  5 | Loss: 0.6180 | Acc: 0.65
Epoch 10 | Loss: 0.5576 | Acc: 0.73
Epoch 15 | Loss: 0.4796 | Acc: 0.80
Epoch 20 | Loss: 0.4650 | Acc: 0.77
Epoch 25 | Loss: 0.4722 | Acc: 0.77
Epoch 29 | Loss: 0.4595 | Acc: 0.80
```

For comparison, the original buggy circuit plateaued at loss `0.5208`:

```text
Epoch  0 | Loss: 0.6617 | Acc: 0.62     <- original, linear CX + RY->RZ
Epoch 10 | Loss: 0.5252 | Acc: 0.80
Epoch 29 | Loss: 0.5208 | Acc: 0.77
```

## 5. Result Interpretation

### 5.1 What the original run was actually showing

The original circuit had two independent bugs, both confirmed by finite-difference gradients on a direct statevector simulation and then reproduced through Qiskit's own `TorchConnector` gradients:

| circuit | dead parameters | inputs with zero influence |
|---|---|---|
| linear CX, `RY -> RZ` (original) | 11 / 16 | `x[2]`, `x[3]` |
| circular CX, `RZ -> RY` (current) | 2 / 16 | none |

The measured qubit could not see half of its own input. Since the labelling rule is `sum(x) > 0`, a model restricted to `x[0]` and `x[1]` has a Bayes-optimal accuracy of **75.0%**. The original run reported 77–82% on 40 training samples, which is that ceiling plus small-sample variance.

So the original conclusion — that the model "learned part of the classification rule" — was measuring a circuit that had saturated the only two features it could see. The plateau was architectural, not an artifact of dataset size or learning rate.

### 5.2 What the corrected run shows

- The loss now falls from `0.6741` to `0.4595`, well below the `0.5208` floor of the buggy circuit, and it is still decreasing at epoch 29.
- All four input features now influence the output, and 14 of 16 parameters carry gradient.
- Accuracy reaches `80%`, which is above the 75.0% blind ceiling.

Accuracy improves much less than loss does, and that is expected. Without data re-uploading, a single `RY(x)` encoding layer makes the expectation value a degree-1 trigonometric polynomial in each feature. The best accuracy any such model can reach on this task is about **78.2%**:

| model class | ceiling on `sum(x) > 0` |
|---|---|
| sees only `x[0]`, `x[1]` (original bug) | 75.0% |
| degree-1 trig, no re-uploading (current) | 78.2% |
| with degree-2 harmonics (re-uploading) | 85.5% |

The corrected circuit is therefore performing at its architectural limit rather than below it. Raising that limit requires data re-uploading, which is what `vqc.py` implements.

### 5.3 Remaining known limitation

Two parameters (`theta[8]`, `theta[9]`) still have zero gradient. This is structural rather than a bug: with a single local observable, the final layer's rotations on qubits outside its light cone cannot influence the output. `vqc.py` resolves it by measuring two observables, `Z0Z1` and `Z2Z3`, whose light cones together cover all four qubits — that model has zero dead parameters across all 46.

The reported accuracy is training accuracy on 40 samples and should not be read as generalization performance.

## 6. Running the Program

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python vqc_v0.py
```

This prints the circuit, regenerates `vqc_v0_circuit.png`, and trains for 30 epochs.

Qiskit displays a message indicating that it is automatically creating a gradient function. This is informational, not an execution error.

## 7. Relationship to `vqc.py`

`vqc.py` is the Q-network for the CartPole agent. It keeps the two fixes described above and adds what CartPole additionally requires:

| | `vqc_v0.py` | `vqc.py` |
|---|---|---|
| purpose | teaching example, binary classification | DQN Q-network |
| encoding | one `RY` layer | `RX` re-uploading in every layer |
| entanglement | circular CX | circular CZ |
| observables | `Z0` (1 output) | `Z0Z1`, `Z2Z3` (2 outputs, one per action) |
| output range | `[0, 1]` via `(raw + 1) / 2` | unbounded, via trainable output scaling |
| input scaling | none | trainable, per layer per feature |
| dead parameters | 2 / 16 | 0 / 46 |

The output range is the critical difference for reinforcement learning. CartPole Q-values under `gamma = 0.99` reach roughly 100, while an expectation value is confined to `[-1, 1]`. Without trainable output scaling the network cannot represent its own targets.
