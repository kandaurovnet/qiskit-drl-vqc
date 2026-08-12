#!/usr/bin/env python3
"""
Orchestrator: run the classical and quantum CartPole-v1 agents and compare them.

The two halves meet at exactly one place — ``cartpole_dqn.build_q_network()``
returns either the classical MLP or ``vqc.VQCQNetwork``, and the DQN loop is
otherwise identical. This script drives both under matched settings and writes a
side-by-side comparison, so any difference is attributable to the network rather
than to the training code.

    python run_experiment.py                      # both agents, default budget
    python run_experiment.py --agents quantum     # quantum only
    python run_experiment.py --smoke              # 2-minute wiring check
    python run_experiment.py --seeds 0 1 2        # repeat over seeds

Runtime note: the quantum network defaults to the torch statevector backend
(~8 ms per gradient step). The Qiskit parameter-shift path is ~1900x slower and
is not viable for a full run; use ``--quantum-backend qiskit`` only on a smoke
test, to confirm the two agree.
"""

import argparse
import json
import os
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cartpole_dqn
from cartpole_dqn import train


def run_one(agent, seed, args):
    """Train a single agent at a single seed and return its run_data."""
    tag = f"_s{seed}" if len(args.seeds) > 1 else ""
    print(f"\n{'=' * 70}\n{agent.upper()}  seed={seed}\n{'=' * 70}")

    if agent == "quantum":
        cartpole_dqn.QUANTUM_KWARGS = {
            "n_layers": args.n_layers,
            "backend": args.quantum_backend,
            "seed": seed,
        }

    tic = time.time()
    run = train(
        agent=agent,
        episodes=args.episodes,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        lr=args.lr_classical if agent == "classical" else args.lr_quantum,
        seed=seed,
        out_dir=args.out_dir,
        double_dqn=args.double_dqn,
        tag=tag,
    )
    run["wall_sec"] = time.time() - tic
    run["seed"] = seed
    return run


def summarize(results):
    """Aggregate runs per agent into a printable comparison table."""
    rows = []
    for agent, runs in results.items():
        greedy = [r["greedy_eval_mean"] for r in runs]
        rows.append({
            "agent": agent,
            "params": runs[0]["params"],
            "seeds": len(runs),
            "greedy_mean": statistics.mean(greedy),
            "greedy_spread": (min(greedy), max(greedy)),
            "solved": sum(r["solved"] for r in runs),
            "episodes": statistics.mean(r["episodes_run"] for r in runs),
            "wall_sec": statistics.mean(r["wall_sec"] for r in runs),
        })
    return rows


def print_summary(rows, solve_reward=475.0):
    print(f"\n{'=' * 78}\nRESULTS\n{'=' * 78}")
    print(f"{'agent':<12}{'params':>8}{'greedy eval':>14}{'range':>18}"
          f"{'solved':>9}{'episodes':>10}{'wall':>9}")
    print("-" * 78)
    for r in rows:
        lo, hi = r["greedy_spread"]
        print(f"{r['agent']:<12}{r['params']:>8}{r['greedy_mean']:>14.1f}"
              f"{f'{lo:.0f} - {hi:.0f}':>18}{f'{r['solved']}/{r['seeds']}':>9}"
              f"{r['episodes']:>10.0f}{r['wall_sec']:>8.0f}s")
    print("-" * 78)
    print(f"'solved' = greedy policy mean >= {solve_reward:.0f} over the eval episodes.")

    if len(rows) == 2:
        c = next((r for r in rows if r["agent"] == "classical"), None)
        q = next((r for r in rows if r["agent"] == "quantum"), None)
        if c and q:
            print(f"\nParameter ratio: classical uses {c['params'] / q['params']:.0f}x "
                  f"more parameters ({c['params']} vs {q['params']}).")


def plot_comparison(results, out_dir, window=100):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    colors = {"classical": "tab:blue", "quantum": "tab:purple"}

    for agent, runs in results.items():
        # Seeds can stop at different episode counts; pad to the longest with
        # the final value so the mean curve does not jump when a run ends.
        longest = max(len(r["episode_rewards"]) for r in runs)
        padded = [r["episode_rewards"] + [r["episode_rewards"][-1]] *
                  (longest - len(r["episode_rewards"])) for r in runs]
        curve = np.mean(padded, axis=0)
        if len(curve) >= window:
            smooth = np.convolve(curve, np.ones(window) / window, mode="valid")
            xs = range(window - 1, len(curve))
        else:
            smooth, xs = curve, range(len(curve))
        label = f"{agent} ({runs[0]['params']} params)"
        ax.plot(xs, smooth, label=label, color=colors.get(agent))

    ax.axhline(475, color="green", linestyle="--", alpha=0.5, label="solved (475)")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward (mean of {window})")
    ax.set_title("CartPole-v1: classical DQN vs DRL+VQC")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison.png")
    plt.savefig(path)
    print(f"\nSaved comparison plot to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+", default=["classical", "quantum"],
                   choices=["classical", "quantum"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--total-steps", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-freq", type=int, default=256)
    p.add_argument("--gradient-steps", type=int, default=128)
    # The classical default is the RL-Zoo tuned value. The quantum network wants
    # a smaller rate on the circuit angles; its output scaling is separately
    # driven at 0.1 by VQCQNetwork.parameter_groups().
    p.add_argument("--lr-classical", type=float, default=2.3e-3)
    p.add_argument("--lr-quantum", type=float, default=1e-3)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--quantum-backend", default="torch",
                   help="'torch' (fast, default) or 'qiskit' (exact, ~1900x slower)")
    p.add_argument("--double-dqn", action="store_true")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--smoke", action="store_true",
                   help="Short run to verify wiring end to end.")
    args = p.parse_args()

    if args.smoke:
        # Step budget must be the binding limit, and must clear learning_starts
        # (1000) or no gradient step ever runs and the check proves nothing.
        args.total_steps = 4000
        print("SMOKE MODE: 4k steps — checks wiring, not performance.\n")

    os.makedirs(args.out_dir, exist_ok=True)
    results = {a: [run_one(a, s, args) for s in args.seeds] for a in args.agents}

    rows = summarize(results)
    print_summary(rows)
    plot_comparison(results, args.out_dir)

    path = os.path.join(args.out_dir, "comparison.json")
    with open(path, "w") as f:
        json.dump({"config": vars(args), "summary": rows}, f, indent=2)
    print(f"Saved summary to {path}")


if __name__ == "__main__":
    main()
