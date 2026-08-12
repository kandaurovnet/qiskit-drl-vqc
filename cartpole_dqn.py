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
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "terminated"))


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


# ---------------------------------------------------------------------------
# Classical Q-network (baseline / reference implementation of the interface)
# ---------------------------------------------------------------------------

class ClassicalQNetwork(nn.Module):
    def __init__(self, n_inputs: int = 4, n_actions: int = 2, hidden=(256, 256)):
        super().__init__()
        dims = [n_inputs, *hidden, n_actions]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


def build_q_network(agent: str) -> nn.Module:
    """Factory. Quantum team: add an 'quantum' branch here that returns your
    TorchConnector-wrapped EstimatorQNN. Must satisfy the interface above."""
    if agent == "classical":
        return ClassicalQNetwork()
    elif agent == "quantum":
        raise NotImplementedError(
            "Plug in your Qiskit EstimatorQNN + TorchConnector network here."
        )
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


def optimize_step(policy_net, target_net, optimizer, memory, batch_size, gamma,
                  double_dqn: bool = False):
    if len(memory) < batch_size:
        return None
    batch = Transition(*zip(*memory.sample(batch_size)))

    state_batch = torch.stack(batch.state).to(DEVICE)
    action_batch = torch.tensor(batch.action, dtype=torch.long, device=DEVICE).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE)
    terminated_batch = torch.tensor(batch.terminated, dtype=torch.float32, device=DEVICE)

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

    expected_q_values = reward_batch + gamma * next_q_values

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
    train_freq: int = 256,
    gradient_steps: int = 128,
    learning_starts: int = 1000,
    replay_capacity: int = 100_000,
    lr: float = 2.3e-3,
    solve_reward: float = 475.0,
    solve_window: int = 100,
    out_dir: str = "results",
    seed: int = 0,
    double_dqn: bool = False,
    tag: str = "",
):
    name = f"{agent}{tag}"
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    n_actions = env.action_space.n

    policy_net = build_q_network(agent).to(DEVICE)
    target_net = build_q_network(agent).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(replay_capacity)

    episode_rewards = []
    episode_losses = []
    steps_done = 0
    tic = time.time()

    for ep in range(episodes):
        obs, _info = env.reset()
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

            next_state = torch.tensor(normalize_obs(obs), device=DEVICE)
            # Store `terminated`, NOT `done`. gymnasium truncates CartPole at 500
            # steps, but a truncated state is not terminal -- the pole is still
            # balanced, so its value must keep bootstrapping. Storing `done` here
            # teaches the net that a perfectly balanced pole is worthless, and
            # the damage grows the more often the agent succeeds.
            memory.push(state, action, next_state, reward, terminated)
            state = next_state
            steps_done += 1

            # Burst updates: every train_freq steps, do gradient_steps updates
            # against a freshly synced target. Updating once per env step against
            # a small buffer overfits the network to the narrow band of states
            # the current policy visits.
            if steps_done >= learning_starts and steps_done % train_freq == 0:
                for _ in range(gradient_steps):
                    loss = optimize_step(policy_net, target_net, optimizer, memory,
                                         batch_size, gamma, double_dqn)
                    if loss is not None:
                        ep_loss_sum += loss
                        ep_loss_count += 1

            # Sync on step count, not episode count: episode length grows with
            # policy quality, so an episode-based interval silently stretches
            # target staleness exactly when Q-values are largest.
            if steps_done % target_update_every_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

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
        "eps_end": eps_end,
        # Solved is judged on the GREEDY policy, as SB3/RL-Zoo do. Training reward
        # is collected with exploration still on and, with burst updates, lags the
        # policy badly -- it is a progress trace, not a performance measure.
        "solved": greedy_mean >= solve_reward,
        "greedy_eval_rewards": greedy,
        "greedy_eval_mean": greedy_mean,
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
    parser.add_argument("--eps-end", type=float, default=0.04)
    parser.add_argument("--train-freq", type=int, default=256)
    parser.add_argument("--gradient-steps", type=int, default=128)
    parser.add_argument("--target-update-every-steps", type=int, default=10)
    parser.add_argument("--double-dqn", action="store_true")
    parser.add_argument("--tag", type=str, default="",
                        help="suffix for output filenames, e.g. --tag _ddqn")
    args = parser.parse_args()

    train(
        agent=args.agent,
        episodes=args.episodes,
        out_dir=args.out_dir,
        seed=args.seed,
        lr=args.lr,
        total_steps=args.total_steps,
        eps_end=args.eps_end,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_every_steps=args.target_update_every_steps,
        double_dqn=args.double_dqn,
        tag=args.tag,
    )
