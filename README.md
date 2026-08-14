# Hybrid Quantum-Classical DQN for CartPole-v1

A Deep Q-Network agent for `CartPole-v1` where the Q-function approximator is
**swappable**: either a classical PyTorch MLP or a parameterized quantum
circuit (Qiskit `EstimatorQNN` wrapped in a `TorchConnector`, trained through a
fast PyTorch statevector backend). Everything else — replay memory,
epsilon-greedy exploration, target network, Huber loss, the training loop — is
shared, so the two agents are compared on equal footing, and the same trained
circuit can be replayed on real IBM Quantum hardware.

Built for the 2026 NTU "Scaling for Quantum Advantage and Beyond" hackathon.
Everything here is open source (MIT) — the IBM hardware section below is
written so anyone can clone the repo and reproduce the hardware run with their
own free IBM Quantum account.

## Why this structure

The whole point is a fair comparison, which means the *only* thing that
differs between the classical and quantum runs is the network itself. So there
is exactly one integration point, in `main/cartpole_dqn.py`:

```python
def build_q_network(agent: str) -> nn.Module:
    if agent == "classical":
        return ClassicalQNetwork()
    elif agent == "quantum":
        return VQCQNetwork()   # EstimatorQNN + TorchConnector
```

Any network satisfying this contract drops in with **no other changes**:

| | Requirement |
|---|---|
| Type | `torch.nn.Module` |
| Input | `(batch, 4)` float32, observations already normalized to `[-1, 1]` |
| Output | `(batch, 2)` — one Q-value per action |
| Gradients | must flow to `.parameters()` (the loop calls `.backward()`) |

Verify a network satisfies it before wiring anything up:

```bash
cd main
python test_interface.py
```

### Observation normalization is shared, deliberately

`normalize_obs()` clips and scales the four CartPole observations to roughly
`[-1, 1]`. Cart position and pole angle are physically bounded, but both
velocities are not — near failure the env returns large values.

This lives in the shared wrapper, not inside either network, because the
quantum circuit's **angle encoding requires bounded inputs** (an unbounded
value would wrap around the Bloch sphere and alias to a different state), and
the classical net trains more stably with the same treatment. Putting it in
one place keeps the comparison honest.

## Setup

```bash
pip install -r requirements.txt
```

The classical baseline only needs `torch`, `gymnasium`, `numpy`, `matplotlib`.
The quantum agent additionally needs `qiskit` and `qiskit-machine-learning`.
Real IBM hardware additionally needs `qiskit-ibm-runtime` (see below).

> **Note:** this uses `gymnasium`, not the legacy `gym`. The APIs differ —
> `reset()` returns `(obs, info)` and `step()` returns five values
> `(obs, reward, terminated, truncated, info)`.

All the commands below assume you have `cd main` first — the scripts import
each other as siblings (`import cartpole_dqn`, `from vqc import ...`), so they
must run from inside `main/`.

## Quickstart: one agent, one run

```bash
python cartpole_dqn.py --agent classical --tag _zoo
python cartpole_dqn.py --agent quantum --tag _quantum
```

Useful flags: `--seed` (default 0), `--total-steps`, `--lr`, `--train-freq`,
`--gradient-steps`, `--target-update-every-steps`, `--eps-end`, `--double-dqn`,
`--tag`, `--out-dir`. Always pass a distinct `--tag` so runs write to separate
files instead of overwriting each other.

Each run writes `results/<agent><tag>_run.json` (per-episode rewards and mean
loss, wall-clock, parameter count, seed, greedy-eval results, solved flag),
`results/<agent><tag>_training.png`, and `results/<agent><tag>_policy.pt`.

Animate a trained checkpoint:

```bash
python watch.py --checkpoint results/classical_zoo_policy.pt --out results/cartpole_solved.gif
```

## The four experiment arms

`run_experiment.py` is the actual benchmark tool — it trains several network
configurations back to back, on matched step budgets, and reports the median
and spread across seeds. This is what the results in `results_drl/` came from.

```bash
python run_experiment.py                     # 4 arms, 1 seed
python run_experiment.py --seeds 0 1 2 3 4   # averaged over 5 seeds
python run_experiment.py --smoke             # wiring check, ~3 min
python run_experiment.py --arms quantum      # one arm only
```

### Why four arms, not two

The obvious comparison is "classical vs. quantum," but the default classical
baseline (`hidden=(32, 32)` = 1,282 parameters) is far wider than CartPole's
4-in/2-out interface needs. Beating it with a ~70-parameter quantum circuit
would say more about the baseline being oversized than about quantum
efficiency. So the benchmark trains four networks under identical conditions:

| Arm | Network | Params (n-layers=5) | What it shows |
|---|---|---:|---|
| `classical` | MLP, hidden=(32,32) | 1,282 | The conventional, deliberately oversized baseline |
| `classical-small` | MLP, hidden=(10,) | 72 | The **honest** like-for-like comparison — matched parameter count to the quantum arm |
| `quantum` | VQC, exact statevector | 70 | The quantum circuit, trained on an exact (noise-free) simulator |
| `quantum-noisy` | VQC + finite-shot sampling | 70 | The same circuit under measurement noise, i.e. what shot noise costs on top |
| `ibm` *(opt-in)* | trained `quantum` checkpoint | 70 | Deployment evidence: the frozen policy replayed on a real IBM QPU |

`classical` vs. `classical-small` shows how much of the "classical baseline"
result is just parameter count. `classical-small` vs. `quantum` is the fair
fight. `quantum` vs. `quantum-noisy` isolates the cost of hardware-realistic
sampling noise, still on a simulator (no QPU time spent). `ibm` is the only
arm that doesn't train — see the next section.

### Matched step budgets, on purpose

Early stopping is **off** by default (`cartpole_dqn.train()` normally stops as
soon as a run's greedy policy clears the solve bar). For a single run that's
fine; for a four-way comparison it would let arms train for different numbers
of steps, so a weaker score might only mean that arm quit sooner. Pass
`--stop-on-eval` if you want sample-efficiency (steps-to-solve) instead of
quality-at-a-fixed-budget — the summary prints mean step count per arm either
way, so an unmatched comparison stays visible.

### Other flags worth knowing

- `--n-layers` (default 5) — VQC depth; also changes the quantum/`classical-small`
  parameter count quoted above.
- `--quantum-backend torch|qiskit` — `torch` (default) is the fast statevector
  path (~8 ms/gradient step); `qiskit` is ~1900x slower and only practical on
  `--smoke`, to confirm the two agree.
- `--noisy-shots` (default 1024) — finite-sampling noise for `quantum-noisy`.
- `--reward-shaping` / `--no-reward-shaping` — dense centering/velocity shaping
  on top of the native reward, applied identically to every arm.
- `--out-dir` — each arm's artifacts, plus a combined `benchmark.json` /
  `benchmark_curves.png` and a solved-policy `.gif` per arm.

Treat a single-seed result as indicative, not conclusive: the committed
classical runs include both a solve (greedy 500.0) and a failure (greedy
111.4) at identical settings. Pass several `--seeds` to get the median and
spread that a real comparison should rest on.

## Connecting to real IBM Quantum hardware (open source setup)

The `ibm` arm loads the already-trained `quantum` checkpoint and replays one
greedy episode on IBM hardware, one Runtime Estimator job per environment
step. It is deployment evidence, not a fifth training result — one capped
episode isn't comparable to the local arms' full evaluation — which is why
it's opt-in (`--arms ibm`) instead of part of the default set, and requires
exactly one seed.

Everything below uses IBM's public, free-tier **IBM Quantum Platform**. No
paid plan is required to run a handful of jobs.

### 1. Get credentials

1. Create an account at IBM Quantum Platform and open the dashboard.
2. Copy your **IBM Cloud API key**.
3. Copy your **IBM Quantum instance CRN** (the free "open plan" instance
   works).

Never commit these to source control — the setup below stores them outside
the repo, in Qiskit's own account store.

### 2. Install the Runtime client

```bash
pip install qiskit-ibm-runtime>=0.42   # already included via requirements.txt
```

### 3. Save the account locally

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

This writes to Qiskit's local account file (outside the repo), not to any
file this project tracks — nothing secret ever touches Git.

### 4. Verify the connection

```bash
cd tools
python test_ibm_connection.py
```

This lists every operational QPU your account can reach (min 4 qubits) and
picks the least-busy one. It only queries backend status — no Runtime job is
submitted, no QPU time is spent.

### 5. (Optional) Validate the circuit against the backend, without spending QPU time

```bash
cd main
python ibm_backend_inference.py
```

This builds the same parameterized circuit `VQCQNetwork` uses, transpiles it
to the selected backend's native gate set (ISA), and confirms every input and
weight parameter survives transpilation and that both action observables
(`Z0Z1`, `Z2Z3`) map onto the physical qubit layout correctly. Still zero QPU
time — it's a dry run that catches wiring bugs before you pay for hardware
time. It then offers to submit one real confirmation job if you want to go
further.

### 6. Run the hardware evaluation

```bash
cd main
python run_experiment.py --arms ibm --seeds 10000 --ibm-max-steps 10 \
  --ibm-backend ibm_marrakesh --ibm-account cartpole-vqc \
  --ibm-checkpoint results/quantum_policy.pt \
  --ibm-output results/ibm_cartpole_run.json
```

This loads `results/quantum_policy.pt` (a checkpoint produced by the `quantum`
arm — train one first, or use a committed checkpoint under `results_drl/`),
submits one Runtime `EstimatorV2` job per environment step, selects
`argmax(Q)`, and steps the real `CartPole-v1` environment with that action. It
writes the familiar artifact triplet:

- `results/ibm_cartpole_policy.pt` — copy of the frozen Torch checkpoint
- `results/ibm_cartpole_run.json` — per-step job IDs, Q-values, and the
  local-exact-vs-hardware agreement check
- `results/ibm_cartpole_evaluation.png` — plotted alongside `torch_exact`
  Q-values, since IBM Runtime doesn't run an optimizer or produce a training
  loss

It does **not** train on IBM hardware — the checkpoint's weights never
change; hardware is inference-only.

Progress is saved after every submitted and completed step, so an interrupted
run (queue wait, connection drop) resumes without resubmitting the pending
job:

```bash
python run_experiment.py --arms ibm --seeds 10000 --ibm-max-steps 10 \
  --ibm-output results/ibm_cartpole_run.json --ibm-resume
```

Pass a distinct `--ibm-output` per run — it refuses to overwrite an existing
file, so a longer run doesn't clobber a shorter completed one.

Because this is one capped episode while the local arms use many full
evaluation episodes, its reward is not a fair ranking against the other arms
— `run_experiment.py` prints a note to that effect whenever `ibm` is included.

### A more rigorous alternative: the manifest-reviewed pipeline

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
for full flag lists. `artifacts/ibm/README.md` explains the `core/` /
`validation/` / `archive/` split of what these stages produce.

## Results

The classical baseline **solves CartPole-v1**:

| Metric | Value |
|---|---|
| Greedy evaluation (100 unseen seeds) | **500.0** mean — min 500, max 500, zero episodes below 500 |
| Solve bar | 475 — cleared |
| Environment steps to solve | 50,077 |
| Wall clock (CPU, no GPU) | 83 s |
| Parameters | 67,586 |

Reproduce with `python cartpole_dqn.py --agent classical --tag _zoo` (seed 0).

Full four-arm sweeps across multiple VQC depths (`--n-layers` 1–5, with and
without reward shaping) are committed under `results_drl/nlayer<N>[-no-rwshp]/`
— each directory has a `benchmark.json` (full per-arm/per-seed summary) and
`benchmark_curves.png` (median training curve with inter-seed spread). Per-seed
checkpoints, per-seed plots, and solved GIFs are regeneratable and gitignored;
rerun `run_experiment.py` to recreate them.

### Solved is judged on the greedy policy, not training reward

CartPole-v1 counts as solved at mean reward 475 over 100 episodes. This repo
evaluates that on the **greedy** policy (ε=0) after training, as SB3 and
RL-Baselines3-Zoo do — not on the training reward curve.

That distinction is not pedantic. Training reward is collected with
exploration still active, and because updates happen in bursts of 128
gradient steps the policy changes discontinuously between logged episodes. In
one run the training log read ~10 while the actual policy scored 468 at
ε=0.04. Judging by the training curve would have been badly wrong in both
directions.

### Hyperparameters

The defaults are the [RL-Baselines3-Zoo tuned CartPole-v1 DQN
config](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml),
ported rather than hand-tuned. The structurally important part is the update
schedule: **128 gradient steps every 256 environment steps, against a target
network synced every 10 steps** — fit hard against a frozen target, then
refresh. An earlier version did one gradient step per environment step against
a 20k buffer, which overfits the network to the narrow band of states the
current policy happens to visit; its greedy policy scored 111 while its
exploring policy scored 353.

## Repository layout

| Path | Purpose |
|---|---|
| `main/cartpole_dqn.py` | The agent, training loop, and the `build_q_network` seam |
| `main/vqc.py` | The VQC Q-network: circuit, encoding, observables, PyTorch wrapper |
| `main/torch_statevector.py` | Fast exact statevector executor used during training |
| `main/torch_density.py` | Differentiable noisy (density-matrix) executor, calibrated from a real device |
| `main/run_experiment.py` | The four-arm benchmark harness described above |
| `main/run_qiskit_sim.py` | Rehearses a trained checkpoint through actual Qiskit simulators |
| `main/eval_shots.py` | Measures the cost of finite-shot sampling on a trained policy |
| `main/watch.py` | Renders a trained checkpoint as an animated GIF |
| `main/vqc_checkpoint.py` | Loads a checkpoint into any execution backend |
| `main/ibm_backend_inference.py` | Transpiles/validates the VQC for an IBM backend, no QPU time |
| `main/run_ibm_cartpole.py` | Fast-path hardware evaluation (used by `run_experiment.py --arms ibm`) |
| `main/{evaluate_ibm_policy,prepare_ibm_policy_submission,run_ibm_policy_evaluation,summarize_ibm_policy_evaluation}.py` | The manifest-reviewed hardware pipeline |
| `main/test_interface.py` | Contract check — run before dropping in a new network |
| `tools/setup_ibm_account.py` | GUI helper to save IBM Quantum credentials locally |
| `tools/test_ibm_connection.py` | GUI helper to verify the saved account and list QPUs |
| `tools/build_ibm_backend_report.py` | Builds a report from collected IBM artifacts |
| `requirements.txt` | Pinned, verified-working dependency versions |
| `docs/architecture.md` | Detailed VQC circuit design and gradient-safety rationale |
| `docs/ibm_integration_log.md` | Narrative log of the IBM integration milestones |
| `results_drl/` | Committed four-arm sweep results (JSON + plots) |
| `artifacts/ibm/` | IBM hardware run manifests and evidence |
| `circuit_docs/` | Rendered circuit diagrams |
| `legacy/` | Earlier teaching version of the VQC (`vqc_v0.py`) |

## Two bugs worth knowing about

Both are easy to write, both silently break training, and neither raises an
error. They are documented here because they cost us most of a day.

**1. Target-network syncing measured in episodes.** In CartPole reward equals
episode length, so as the policy improves episodes get *longer*. An
episode-based sync interval therefore stretches the target network's staleness
in proportion to how well the agent is doing — maximum staleness at maximum
Q-value magnitude. This produced a textbook divergence: reward climbed to a
mean of ~200, then collapsed to 60 and never recovered. Sync on **steps**.

**2. Treating truncation as termination.** gymnasium ends a CartPole episode
with two distinct flags: `terminated` (the pole fell — a real MDP terminal
state) and `truncated` (the 500-step time limit — the pole is still balanced).
Storing `done = terminated or truncated` in the replay buffer zeroes the
bootstrap target for exactly the agent's best states, teaching it that a
perfectly balanced pole is worthless. Store **`terminated` only**. The replay
field is named `terminated` here so the distinction is hard to reintroduce.

## License

MIT — see `LICENSE`.
