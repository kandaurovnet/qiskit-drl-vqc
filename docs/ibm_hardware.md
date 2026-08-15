# Running on real IBM Quantum hardware

Part of [Hybrid Quantum-Classical DQN for CartPole-v1](../README.md). The local
simulator arms need none of this — it is only for replaying a trained
checkpoint on a real QPU. All `cd` paths below are relative to the repo root.


The `ibm` arm loads the already-trained `quantum` checkpoint and replays one
greedy episode on IBM hardware, one Runtime Estimator job per environment step.
It is deployment evidence, not a fifth training result — one capped episode
isn't comparable to the local arms' full evaluation — which is why it's opt-in
(`--arms ibm`) and requires exactly one seed. It does **not** train on
hardware; the checkpoint's weights never change.

Everything below uses IBM's public, free-tier **IBM Quantum Platform**. No paid
plan is required to run a handful of jobs.

## 1. Get credentials

1. Create an account at IBM Quantum Platform and open the dashboard.
2. Copy your **IBM Cloud API key**.
3. Copy your **IBM Quantum instance CRN** (the free "open plan" instance works).

Never commit these to source control — the setup below stores them outside the
repo, in Qiskit's own account store.

## 2. Install the Runtime client

```bash
pip install qiskit-ibm-runtime>=0.42   # already included via requirements.txt
```

## 3. Save the account locally

Either run the GUI helper (a couple of dialog boxes, no terminal typing of
secrets):

```bash
cd tools
python setup_ibm_account.py
```

or save it directly in Python:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="<your IBM Cloud API key>",
    instance="<your instance CRN>",
    name="cartpole-vqc",   # this project's account name; the scripts expect it
    set_as_default=True,
    overwrite=True,
)
```

This writes to Qiskit's local account file (outside the repo), not to any file
this project tracks — nothing secret ever touches Git.

## 4. Verify the connection

```bash
cd tools
python test_ibm_connection.py
```

Lists every operational QPU your account can reach (min 4 qubits) and picks the
least-busy one. Status queries only — no Runtime job, no QPU time.

## 5. (Optional) Validate the circuit against the backend, free

```bash
cd main
python ibm_backend_inference.py
```

Builds the same parameterized circuit `VQCQNetwork` uses, transpiles it to the
selected backend's native gate set (ISA), and confirms every input and weight
parameter survives transpilation and that both action observables (`Z0Z1`,
`Z2Z3`) map onto the physical qubit layout correctly. Still zero QPU time — a
dry run that catches wiring bugs before you pay for hardware. It then offers to
submit one real confirmation job if you want to go further.

## 6. Run the hardware evaluation

```bash
cd main
python run_experiment.py --arms ibm --seeds 10000 --ibm-max-steps 10 \
  --ibm-backend ibm_marrakesh --ibm-account cartpole-vqc \
  --ibm-checkpoint results/quantum_policy.pt \
  --ibm-output results/ibm_cartpole_run.json
```

Loads `results/quantum_policy.pt` (train a `quantum` arm first, or use a
committed checkpoint under `results_drl/`), submits one Runtime `EstimatorV2`
job per environment step, selects `argmax(Q)`, and steps the real `CartPole-v1`
environment with that action. It writes:

- `results/ibm_cartpole_policy.pt` — copy of the frozen Torch checkpoint
- `results/ibm_cartpole_run.json` — per-step job IDs, Q-values, and the
  local-exact-vs-hardware agreement check
- `results/ibm_cartpole_evaluation.png` — plotted alongside `torch_exact`
  Q-values, since IBM Runtime doesn't run an optimizer or produce a training
  loss

Progress is saved after every submitted and completed step, so an interrupted
run (queue wait, connection drop) resumes without resubmitting the pending job:

```bash
python run_experiment.py --arms ibm --seeds 10000 --ibm-max-steps 10 \
  --ibm-output results/ibm_cartpole_run.json --ibm-resume
```

Pass a distinct `--ibm-output` per run — it refuses to overwrite an existing
file, so a longer run doesn't clobber a shorter completed one.

Because this is one capped episode while the local arms use many full
evaluation episodes, its reward is not a fair ranking against the other arms —
`run_experiment.py` prints a note to that effect whenever `ibm` is included.

## A more rigorous alternative: the manifest-reviewed pipeline

`run_ibm_cartpole.py` (used above) is the fast path. For a fully auditable
trail — useful if you want to review exactly what will be submitted *before*
any QPU time is spent — there's a four-stage pipeline instead, each stage
writing a reviewable JSON manifest under `artifacts/ibm/`:

```text
evaluate_ibm_policy.py            # picks 2 high-margin states locally, zero IBM connection
        |
prepare_ibm_policy_submission.py  # connects, transpiles, writes a submission manifest — no job yet
        |
run_ibm_policy_evaluation.py      # submits/resumes the Runtime job from that reviewed manifest
        |
summarize_ibm_policy_evaluation.py  # turns the completed result into a report-ready summary
```

Each script's docstring explains its exact inputs/outputs; run with `--help`
for full flag lists. The pipeline writes its manifests under `artifacts/ibm/`,
split into `core/` / `validation/` / `archive/`.

