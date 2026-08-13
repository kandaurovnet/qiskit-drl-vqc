#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CartPole-v1 DQN — classical baseline with a pluggable Q-network interface.

Design goal: the quantum team can drop in their own Q-network (a Qiskit
EstimatorQNN wrapped in TorchConnector, or anything else) as long as it
implements the interface below. Nothing else in this file needs to change.

    class QNetwork(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            # x: (batch, 4) float32, normalized observations
            # returns: (batch, 2) Q-values, one per action
            ...

Usage:
    python cartpole_dqn.py --agent classical --episodes 600
"""

import argparse
import json
import os
import random
import time
from collections import deque, namedtuple

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Replay memory (generic, reusable as-is)
# ---------------------------------------------------------------------------

# `terminated` is the true MDP terminal flag (pole fell / cart left track) -- NOT
# gymnasium's `truncated` time-limit flag. Only `terminated` may cut bootstrapping.
# `discount` is the bootstrap multiplier for next_state: gamma**k where k is the
# length of the (possibly n-step) reward window this transition summarizes.
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "terminated", "discount"))


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ---------------------------------------------------------------------------
# Observation normalization
# ---------------------------------------------------------------------------
# CartPole-v1 observations: [cart position, cart velocity, pole angle, pole
# angular velocity]. Position and angle are physically bounded; velocity and
# angular velocity are not (the env can return large values near failure).
# We clip + scale everything to roughly [-1, 1] so a quantum angle-encoding
# circuit (which expects bounded inputs) and the classical net both train on
# comparable input ranges.

OBS_HIGH = np.array([2.4, 3.0, 0.2095, 3.0], dtype=np.float32)  # position, vel, angle(rad), ang-vel


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    clipped = np.clip(obs, -OBS_HIGH, OBS_HIGH)
    return (clipped / OBS_HIGH).astype(np.float32)


def shape_reward(norm_obs: np.ndarray, base_reward: float, terminated: bool) -> float:
    """Dense shaping: reward being centered, upright and slow, not merely alive.

    CartPole-v1's native reward is +1 every step no matter how close to
    failure the state is, so an agent teetering at the track edge gets the
    same signal as one sitting dead-center -- nothing pushes it toward the
    safer states it's easier to keep balancing from. The deviation terms
    (inputs already normalized to roughly [-1, 1] by normalize_obs) supply
    that gradient; the terminal penalty makes falling explicitly costly
    rather than merely an early end to the +1 stream.

    Measured against the classical agent, 3 seeds, 80k steps: 3/3 solved at
    greedy 500.0 with no post-success collapse, and the greedy policy holds
    mean |x| ~= 0.11 / mean |x_dot| ~= 0.11, versus ~0.66 / ~0.40 for a
    position+angle-only penalty and ~0.72 / ~0.30 for no shaping at all.
    The velocity term is what buys the steadiness; the terminal penalty is
    what removes the collapse.

    Two consequences worth knowing:

    - This is NOT potential-based (Ng et al. 1999), so it does not preserve
      the optimal policy of the underlying MDP -- deliberately, since a
      centered-and-slow balance is the behaviour we want. train() keeps
      ep_reward and the solve check on the TRUE env reward, so "solved"
      still means solved on CartPole's own terms.
    - Terminal transitions bootstrap from zero, so their target is the raw
      reward: Q now spans roughly [-200, +100] rather than [0, +100]. That
      breaks the assumption vqc.py documents for VQCQNetwork.output_scale
      ("Q-values reach ~100"), which starts at 1.0 and must now stretch
      ~3x further. Check the quantum agent before assuming this transfers.
    """
    x, x_dot, theta, _theta_dot = norm_obs
    return (base_reward
            - 1.0 * abs(float(x))
            - 1.0 * abs(float(theta))
            - 5.0 * abs(float(x_dot))
            - (200.0 if terminated else 0.0))


# ---------------------------------------------------------------------------
# Classical Q-network (baseline / reference implementation of the interface)
# ---------------------------------------------------------------------------

class ClassicalQNetwork(nn.Module):
    def __init__(self, n_inputs: int = 4, n_actions: int = 2, hidden=(32, 32)):
        super().__init__()
        dims = [n_inputs, *hidden, n_actions]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


# Construction options for the quantum network, e.g. {"n_layers": 5} or
# {"backend": "qiskit"}. run_experiment.py sets this; defaults are vqc.py's.
QUANTUM_KWARGS: dict = {}
# Construction options for the classical network, e.g. {"hidden": (6,)} to get a
# parameter-matched baseline instead of the oversized default.
CLASSICAL_KWARGS: dict = {}


def build_q_network(agent: str, seed: int | None = None) -> nn.Module:
    """Factory returning a network that satisfies the interface above.

    ``seed`` is a fallback for VQCQNetwork's circuit-parameter init, used only
    when QUANTUM_KWARGS doesn't already specify one. VQCQNetwork draws its
    initial theta from its own `np.random.default_rng(seed)`, not from the
    global `np.random.seed()` train() already sets -- so without this, two
    runs with the same `--seed` still start from different, OS-entropy-drawn
    circuit weights and can converge to different local optima. run_experiment
    .py always sets QUANTUM_KWARGS["seed"] itself; the CLI entry point below
    does not, which is exactly the gap this closes.
    """
    if agent == "classical":
        return ClassicalQNetwork(**CLASSICAL_KWARGS)
    elif agent == "quantum":
        # Imported lazily so the classical baseline does not require qiskit.
        from vqc import VQCQNetwork
        kwargs = {"seed": seed, **QUANTUM_KWARGS}
        return VQCQNetwork(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def select_action(state: torch.Tensor, policy_net: nn.Module, eps: float, n_actions: int) -> int:
    if random.random() < eps:
        return random.randrange(n_actions)
    with torch.no_grad():
        return int(policy_net(state.unsqueeze(0)).argmax(dim=1).item())


def optimize_step(policy_net, target_net, optimizer, memory, batch_size,
                  double_dqn: bool = False):
    if len(memory) < batch_size:
        return None
    batch = Transition(*zip(*memory.sample(batch_size)))

    state_batch = torch.stack(batch.state).to(DEVICE)
    action_batch = torch.tensor(batch.action, dtype=torch.long, device=DEVICE).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE)
    terminated_batch = torch.tensor(batch.terminated, dtype=torch.float32, device=DEVICE)
    discount_batch = torch.tensor(batch.discount, dtype=torch.float32, device=DEVICE)

    non_final_mask = ~terminated_batch.bool()
    non_final_next_states = torch.stack(
        [s for s, t in zip(batch.next_state, batch.terminated) if not t]
    ).to(DEVICE) if any(not t for t in batch.terminated) else None

    q_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    next_q_values = torch.zeros(batch_size, device=DEVICE)
    if non_final_next_states is not None:
        with torch.no_grad():
            if double_dqn:
                # Decouple action selection from evaluation: the policy net picks
                # the action, the target net scores it. Plain DQN's max() takes
                # both from the same noisy net, so positive errors compound.
                best = policy_net(non_final_next_states).argmax(1, keepdim=True)
                next_q_values[non_final_mask] = (
                    target_net(non_final_next_states).gather(1, best).squeeze(1)
                )
            else:
                next_q_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_q_values = reward_batch + discount_batch * next_q_values

    loss = F.smooth_l1_loss(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss.item()


def evaluate(policy_net: nn.Module, episodes: int = 20, seed: int = 10_000):
    """Run the greedy policy (no exploration) and return per-episode rewards."""
    env = gym.make("CartPole-v1")
    rewards = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total = 0.0
        for _ in range(500):
            state = torch.tensor(normalize_obs(obs), device=DEVICE)
            with torch.no_grad():
                action = int(policy_net(state.unsqueeze(0)).argmax(dim=1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    env.close()
    return rewards


def train(
    agent: str = "classical",
    episodes: int = 2000,
    # Defaults below are the RL-Baselines3-Zoo tuned CartPole-v1 DQN config
    # (hyperparams/dqn.yml), which solves in ~50k env steps. Ported rather than
    # hand-tuned: our own sweeps kept refuting single-knob hypotheses.
    total_steps: int = 50_000,
    batch_size: int = 64,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.04,
    exploration_fraction: float = 0.16,
    target_update_every_steps: int = 10,
    # Polyak coefficient for the target sync. 1.0 is a hard copy (the default,
    # matching the RL-Zoo config). Smaller values blend the target toward the
    # policy instead: target <- tau*policy + (1-tau)*target. The burst schedule
    # makes this matter -- the target is frozen for a whole gradient_steps burst
    # and then jumps, so the policy can drift far between syncs.
    tau: float = 1.0,
    # Rainbow-style n-step returns (arXiv:2512.05946 uses them on a ring-topology
    # VQC like ours). Each stored transition sums n rewards and bootstraps from
    # the state n steps ahead with gamma**n, shortening the chain of bootstrapped
    # estimates a value error must cross. 1 = classic single-step targets.
    n_step: int = 1,
    train_freq: int = 256,
    gradient_steps: int = 128,
    learning_starts: int = 1000,
    # DQN on CartPole oscillates: a run can reach a strong policy and then train
    # past it. The committed classical_base run peaked at best_mean_100 352.9 but
    # finished at greedy 111.4. Evaluating periodically and keeping the best
    # weights recovers that policy instead of discarding it -- the same thing
    # SB3's EvalCallback(best_model_save_path=...) does in stable_dqn.py.
    eval_every_steps: int = 5000,
    eval_episodes: int = 5,
    replay_capacity: int = 100_000,
    lr: float = 2.3e-3,
    # Only used by networks exposing parameter_groups() (the quantum net).
    # None resolves below to a shaping-dependent default.
    lr_output: float | None = None,
    solve_reward: float = 475.0,
    solve_window: int = 100,
    out_dir: str = "results",
    seed: int = 0,
    double_dqn: bool = True,
    # Helps the classical net a lot (206 vs 581 episodes to solve, same
    # quality); a multi-seed ablation showed it's a wash-to-slight-negative for
    # the quantum agent (mean 452.9 vs 472.7, more seed variance). Pass
    # reward_shaping=False (CLI: --no-reward-shaping) for quantum runs.
    reward_shaping: bool = True,
    tag: str = "",
    name: str = "",
):
    # `name` prefixes every output file. It defaults to agent+tag, but callers
    # that run several variants of one agent (e.g. two classical widths) need to
    # name runs themselves, or both would write to the same files.
    # Shaping with a terminal penalty widens the Q range from ~[0,100] to
    # ~[-200,100], and VQCQNetwork.output_scale starts at 1.0 and has to stretch
    # to cover it. At lr_output=0.1 it stalls short (measured: 95-220) and the
    # agent fails; at 0.3 it reaches 320-430 and solves 3/3 (mean 491.5, std
    # 11.2) versus 1/3 (mean 408.6, std 63.5). Unshaped runs keep 0.1, which is
    # what the narrower range was calibrated for.
    if lr_output is None:
        lr_output = 0.3 if reward_shaping else 0.1

    name = name or f"{agent}{tag}"
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    n_actions = env.action_space.n

    policy_net = build_q_network(agent, seed=seed).to(DEVICE)
    target_net = build_q_network(agent, seed=seed).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # A quantum network needs per-group learning rates: its output scaling has
    # to grow from 1 to ~100 to reach CartPole-sized Q-values, which a single
    # small lr cannot do in any reasonable number of steps. Networks that
    # expose parameter_groups() get to choose; everything else is unchanged.
    # lr_output matters more once shaping widens the Q range: with the terminal
    # penalty, targets span ~[-200, +100], and a run whose output_scale stalls
    # near 100 cannot represent them (measured: the one seed that stalled at
    # 98.8 was the one that failed to solve).
    if hasattr(policy_net, "parameter_groups"):
        optimizer = optim.Adam(policy_net.parameter_groups(
            lr_theta=lr, lr_input=lr, lr_output=lr_output))
    else:
        optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(replay_capacity)

    episode_rewards = []
    episode_losses = []
    steps_done = 0
    best_eval = -float("inf")
    best_state = None
    eval_history = []          # (step, greedy mean) for the training plot
    tic = time.time()

    def push_window(buf, k):
        """Store one n-step transition summarizing the first k entries of buf."""
        k = min(k, len(buf))
        s0, a0 = buf[0][0], buf[0][1]
        reward_n = sum(gamma ** i * buf[i][2] for i in range(k))
        next_n, term_n = buf[k - 1][3], buf[k - 1][4]
        memory.push(s0, a0, next_n, reward_n, term_n, gamma ** k)

    for ep in range(episodes):
        obs, _info = env.reset()
        nstep_buf = deque()
        state = torch.tensor(normalize_obs(obs), device=DEVICE)
        ep_reward = 0.0
        ep_loss_sum = 0.0
        ep_loss_count = 0

        for _t in range(500):  # CartPole-v1 max episode length
            # Linear epsilon decay over exploration_fraction of the step budget,
            # measured in environment steps (SB3 semantics), not episodes.
            frac = min(1.0, steps_done / max(1, exploration_fraction * total_steps))
            eps = eps_start + frac * (eps_end - eps_start)

            action = select_action(state, policy_net, eps, n_actions)
            obs, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            norm_next_obs = normalize_obs(obs)
            next_state = torch.tensor(norm_next_obs, device=DEVICE)
            # Shaping only changes what the optimizer is told to chase; ep_reward
            # (logging, solve-window check) always accumulates the true env
            # reward, so `solve_reward=475` keeps meaning the same thing whether
            # or not shaping is on.
            train_reward = shape_reward(norm_next_obs, reward, terminated) if reward_shaping else reward
            # Store `terminated`, NOT `done`. gymnasium truncates CartPole at 500
            # steps, but a truncated state is not terminal -- the pole is still
            # balanced, so its value must keep bootstrapping. Storing `done` here
            # teaches the net that a perfectly balanced pole is worthless, and
            # the damage grows the more often the agent succeeds.
            nstep_buf.append((state, action, train_reward, next_state, terminated))
            if len(nstep_buf) >= n_step:
                push_window(nstep_buf, n_step)
                nstep_buf.popleft()
            state = next_state
            steps_done += 1

            # Burst updates: every train_freq steps, do gradient_steps updates
            # against a freshly synced target. Updating once per env step against
            # a small buffer overfits the network to the narrow band of states
            # the current policy visits.
            if steps_done >= learning_starts and steps_done % train_freq == 0:
                for _ in range(gradient_steps):
                    loss = optimize_step(policy_net, target_net, optimizer, memory,
                                         batch_size, double_dqn)
                    if loss is not None:
                        ep_loss_sum += loss
                        ep_loss_count += 1

            # Sync on step count, not episode count: episode length grows with
            # policy quality, so an episode-based interval silently stretches
            # target staleness exactly when Q-values are largest.
            if steps_done % target_update_every_steps == 0:
                if tau >= 1.0:
                    target_net.load_state_dict(policy_net.state_dict())
                else:
                    with torch.no_grad():
                        for tp, pp in zip(target_net.parameters(),
                                          policy_net.parameters()):
                            tp.mul_(1.0 - tau).add_(pp, alpha=tau)

            # Snapshot the best greedy policy seen, not merely the last one.
            if steps_done >= learning_starts and steps_done % eval_every_steps == 0:
                score = float(np.mean(evaluate(policy_net, episodes=eval_episodes,
                                               seed=seed + 50_000)))
                eval_history.append((steps_done, score))
                if score > best_eval:
                    best_eval = score
                    best_state = {k: v.detach().clone()
                                  for k, v in policy_net.state_dict().items()}

            if done:
                break

        # Tail windows at episode end: each remaining head still becomes a
        # transition, over however many steps it actually saw (gamma**k stored
        # per transition, so shorter windows bootstrap correctly).
        while nstep_buf:
            push_window(nstep_buf, n_step)
            nstep_buf.popleft()

        episode_rewards.append(ep_reward)
        episode_losses.append(ep_loss_sum / ep_loss_count if ep_loss_count else 0.0)

        if ep % 20 == 0:
            recent = episode_rewards[-solve_window:]
            avg = sum(recent) / len(recent)
            print(f"[{name}] episode {ep:4d}  reward {ep_reward:6.1f}  "
                  f"avg({len(recent)}) {avg:6.1f}  loss {episode_losses[-1]:7.4f}  eps {eps:.3f}")

        if len(episode_rewards) >= solve_window:
            avg = sum(episode_rewards[-solve_window:]) / solve_window
            if avg >= solve_reward:
                print(f"[{name}] SOLVED at episode {ep}, {steps_done} steps "
                      f"(avg reward {avg:.1f} over {solve_window} eps)")
                break

        if steps_done >= total_steps:
            print(f"[{name}] reached step budget ({steps_done}/{total_steps}) at episode {ep}")
            break

    elapsed = time.time() - tic
    env.close()

    # Restore the best checkpoint before the final measurement, so the reported
    # score is the best policy the run actually found rather than wherever the
    # oscillation happened to leave it.
    if best_state is not None:
        policy_net.load_state_dict(best_state)
        print(f"[{name}] restored best checkpoint (greedy {best_eval:.1f} "
              f"on {eval_episodes} eps, from {len(eval_history)} evaluations)")

    # Training reward is measured *with* exploration noise still on. Evaluate the
    # greedy policy separately so a good policy masked by epsilon is visible.
    greedy = evaluate(policy_net, episodes=solve_window, seed=seed + 10_000)
    greedy_mean = float(np.mean(greedy))
    print(f"[{name}] greedy eval (eps=0), {solve_window} episodes: mean {greedy_mean:.1f} "
          f"min {min(greedy):.0f} max {max(greedy):.0f}  "
          f"-> {'SOLVED' if greedy_mean >= solve_reward else 'not solved'} "
          f"(bar {solve_reward:.0f})")

    torch.save(policy_net.state_dict(), os.path.join(out_dir, f"{name}_policy.pt"))

    run_data = {
        "agent": agent,
        "episode_rewards": episode_rewards,
        "episode_losses": episode_losses,
        "elapsed_sec": elapsed,
        "episodes_run": len(episode_rewards),
        "params": sum(p.numel() for p in policy_net.parameters()),
        "seed": seed,
        "double_dqn": double_dqn,
        "tau": tau,
        "n_step": n_step,
        "reward_shaping": reward_shaping,
        "eps_end": eps_end,
        # Solved is judged on the GREEDY policy, as SB3/RL-Zoo do. Training reward
        # is collected with exploration still on and, with burst updates, lags the
        # policy badly -- it is a progress trace, not a performance measure.
        "solved": greedy_mean >= solve_reward,
        "greedy_eval_rewards": greedy,
        "greedy_eval_mean": greedy_mean,
        "eval_history": eval_history,
        "best_eval_during_training": best_eval if best_state is not None else None,
        "best_mean_100": max(
            sum(episode_rewards[i:i + solve_window]) / solve_window
            for i in range(max(1, len(episode_rewards) - solve_window + 1))
        ) if len(episode_rewards) >= solve_window else None,
    }
    json_path = os.path.join(out_dir, f"{name}_run.json")
    with open(json_path, "w") as f:
        json.dump(run_data, f)
    print(f"Saved run data to {json_path}")

    plot_rewards(episode_rewards, episode_losses, name, out_dir)
    return run_data


def plot_rewards(episode_rewards, episode_losses, agent, out_dir, mean_window=100):
    fig, (ax_r, ax_l) = plt.subplots(2, 1, sharex=True, figsize=(7, 7), dpi=150)

    ax_r.plot(episode_rewards, alpha=0.4, label="episode reward")
    if len(episode_rewards) >= mean_window:
        means = np.convolve(episode_rewards, np.ones(mean_window) / mean_window, mode="valid")
        ax_r.plot(range(mean_window - 1, len(episode_rewards)), means, label=f"mean ({mean_window})")
    ax_r.axhline(475, color="green", linestyle="--", alpha=0.5, label="solved threshold")
    ax_r.set_ylabel("Reward")
    ax_r.set_title(f"CartPole-v1 training — {agent}")
    ax_r.legend()

    ax_l.plot(episode_losses, alpha=0.6, color="tab:red", label="mean loss / episode")
    ax_l.set_yscale("log")
    ax_l.set_xlabel("Episode")
    ax_l.set_ylabel("Huber loss (log)")
    ax_l.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, f"{agent}_training.png")
    plt.savefig(path)
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["classical", "quantum"], default="classical")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2.3e-3)
    parser.add_argument("--lr-output", type=float, default=None,
                        help="learning rate for VQCQNetwork.output_scale (quantum only); default 0.3 with --reward-shaping, else 0.1")
    parser.add_argument("--eps-end", type=float, default=0.04)
    parser.add_argument("--train-freq", type=int, default=256)
    parser.add_argument("--gradient-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every-steps", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--target-update-every-steps", type=int, default=10)
    parser.add_argument("--n-step", type=int, default=1,
                        help="n-step return window (1 = classic DQN targets)")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="target sync: 1.0 = hard copy, <1 = Polyak blend")
    parser.add_argument("--double-dqn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward-shaping", action=argparse.BooleanOptionalAction, default=True,
                        help="penalize cart/pole deviation from center, on top of the "
                             "native +1/step reward (see shape_reward). Helps the classical "
                             "agent; use --no-reward-shaping for quantum (see train()'s "
                             "docstring comment on reward_shaping)")
    parser.add_argument("--tag", type=str, default="",
                        help="suffix for output filenames, e.g. --tag _ddqn")
    args = parser.parse_args()

    train(
        agent=args.agent,
        episodes=args.episodes,
        out_dir=args.out_dir,
        seed=args.seed,
        lr=args.lr,
        lr_output=args.lr_output,
        total_steps=args.total_steps,
        eps_end=args.eps_end,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        batch_size=args.batch_size,
        eval_every_steps=args.eval_every_steps,
        eval_episodes=args.eval_episodes,
        target_update_every_steps=args.target_update_every_steps,
        tau=args.tau,
        n_step=args.n_step,
        double_dqn=args.double_dqn,
        reward_shaping=args.reward_shaping,
        tag=args.tag,
    )
