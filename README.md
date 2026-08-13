# Hybrid Quantum-Classical DQN for CartPole-v1

A Deep Q-Network agent for `CartPole-v1` where the Q-function approximator is
**swappable**: either a classical PyTorch MLP or a parameterized quantum circuit
(Qiskit `EstimatorQNN` wrapped in a `TorchConnector`). Everything else — replay
memory, epsilon-greedy exploration, target network, Huber loss, the training
loop — is shared, so the two agents are compared on equal footing.

Built for the 2026 NTU "Scaling for Quantum Advantage and Beyond" hackathon.

## Why this structure

The whole point is a fair comparison, which means the *only* thing that differs
between the classical and quantum runs is the network itself. So there is
exactly one integration point:

```python
def build_q_network(agent: str) -> nn.Module:
    if agent == "classical":
        return ClassicalQNetwork()
    elif agent == "quantum":
        return QuantumQNetwork()   # your EstimatorQNN + TorchConnector
```

Any network satisfying this contract drops in with **no other changes**:

| | Requirement |
|---|---|
| Type | `torch.nn.Module` |
| Input | `(batch, 4)` float32, observations already normalized to `[-1, 1]` |
| Output | `(batch, 2)` — one Q-value per action |
| Gradients | must flow to `.parameters()` (the loop calls `.backward()`) |

Verify your network satisfies it before wiring anything up:

```bash
python test_interface.py
```

### Observation normalization is shared, deliberately

`normalize_obs()` clips and scales the four CartPole observations to roughly
`[-1, 1]`. Cart position and pole angle are physically bounded, but both
velocities are not — near failure the env returns large values.

This lives in the shared wrapper, not inside either network, because the
quantum circuit's **angle encoding requires bounded inputs** (an unbounded
value would wrap around the Bloch sphere and alias to a different state), and
the classical net trains more stably with the same treatment. Putting it in one
place keeps the comparison honest.

## Setup

```bash
pip install -r requirements.txt
```

The classical baseline only needs `torch`, `gymnasium`, `numpy`, `matplotlib`.
The quantum agent additionally needs `qiskit` and `qiskit-machine-learning`.

> **Note:** this uses `gymnasium`, not the legacy `gym`. The APIs differ —
> `reset()` returns `(obs, info)` and `step()` returns five values
> `(obs, reward, terminated, truncated, info)`.

## Running

```bash
python cartpole_dqn.py --agent classical --tag _zoo
```

```bash
python cartpole_dqn.py --agent quantum --tag _quantum
```

The same experiment orchestrator also exposes an evaluation-only IBM hardware
path for the already-trained VQC checkpoint:

```bash
python run_experiment.py --agents ibm --seeds 10000 --ibm-max-steps 10 \
  --ibm-output results/ibm_cartpole_run.json
```

This runs one greedy CartPole episode capped at 10 sequential environment
steps and writes the familiar artifact triplet:

- `results/ibm_cartpole_policy.pt`
- `results/ibm_cartpole_run.json`
- `results/ibm_cartpole_evaluation.png`

It does **not** train on IBM
hardware: it loads `results/quantum_policy.pt`, submits one Runtime Estimator
job per step, selects `argmax(Q)`, and advances the same `CartPole-v1`
environment. The IBM policy artifact is therefore an exact copy of the frozen
Torch-trained checkpoint. Its plot is deliberately named `evaluation.png`, not
`training.png`, because IBM Runtime does not run the optimizer or produce a
training loss. If waiting is interrupted, resume without duplicating the
pending job:

```bash
python run_experiment.py --agents ibm --seeds 10000 --ibm-max-steps 10 \
  --ibm-output results/ibm_cartpole_run.json --ibm-resume
```

Because the IBM demonstration is one capped episode while the local agents use
many full evaluation episodes, its reward is not a fair performance ranking.
The committed 10-step run is complete. A longer, resumable 50-step run uses a
separate output stem so it cannot overwrite the completed artifacts:

```bash
python run_experiment.py --agents ibm --seeds 0 --ibm-max-steps 50 \
  --ibm-output results/ibm_cartpole_50step_run.json
```

After completion, its associated files are
`results/ibm_cartpole_50step_policy.pt` and
`results/ibm_cartpole_50step_evaluation.png`. Progress is saved after every
submitted and completed step; resume with the same arguments plus
`--ibm-resume`. IBM results provide hardware deployment and robustness
evidence, while Classical and Quantum Torch remain the training comparison.

Useful flags: `--seed` (default 0), `--total-steps`, `--lr`, `--train-freq`,
`--gradient-steps`, `--target-update-every-steps`, `--eps-end`, `--double-dqn`,
`--tag`, `--out-dir`. Always pass a distinct `--tag` so runs write to separate
files instead of overwriting each other.

Each run writes `results/<agent><tag>_run.json` (per-episode rewards and mean
loss, wall-clock, parameter count, seed, greedy-eval results, solved flag),
`results/<agent><tag>_training.png`, and `results/<agent><tag>_policy.pt`.

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

### Solved is judged on the greedy policy, not training reward

CartPole-v1 counts as solved at mean reward 475 over 100 episodes. This repo
evaluates that on the **greedy** policy (ε=0) after training, as SB3 and
RL-Baselines3-Zoo do — not on the training reward curve.

That distinction is not pedantic. Training reward is collected with exploration
still active, and because updates happen in bursts of 128 gradient steps the
policy changes discontinuously between logged episodes. In one run the training
log read ~10 while the actual policy scored 468 at ε=0.04. Judging by the
training curve would have been badly wrong in both directions.

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

| File | Purpose |
|---|---|
| `cartpole_dqn.py` | The agent, training loop, and the `build_q_network` seam |
| `test_interface.py` | Contract check — run before dropping in a new network |
| `requirements.txt` | Pinned, verified-working dependency versions |
| `drl.py` | Reference DQN from the forked quantum-maze project |
| `module2_mdp_cartpole_demo.ipynb` | MDP / CartPole teaching notebook |
| `results/` | Run artifacts (JSON + plots) |

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
