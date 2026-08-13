#!/usr/bin/env python3
"""
Benchmark: classical DQN vs DRL+VQC on CartPole-v1, over multiple seeds.

The two halves meet at exactly one place — ``cartpole_dqn.build_q_network()``
returns either a classical MLP or ``vqc.VQCQNetwork``, and the DQN loop is
otherwise identical. Any difference is therefore attributable to the network.

Four arms, because the obvious two-way comparison is misleading. The default
classical baseline (hidden=(32,32) = 1,282 parameters) is far wider than
CartPole's 4-in/2-out interface needs, so beating it with a ~70-parameter
circuit says more about the baseline being oversized than about quantum
efficiency. The ``classical-small`` arm (hidden=(10,) = 72 parameters) is the
honest like-for-like comparison against the quantum arm at the default
--n-layers 5 (70 parameters). ``quantum-noisy`` adds finite-shot sampling on
top, to show what the same circuit costs under measurement noise.

    python run_experiment.py                     # 4 arms, 1 seed
    python run_experiment.py --seeds 0 1 2 3 4   # averaged over 5 seeds
    python run_experiment.py --smoke             # wiring check, ~3 min
    python run_experiment.py --arms quantum      # one arm only

Treat a single-seed result as indicative, not conclusive: the committed
classical runs include both a solve (greedy 500.0) and a failure (greedy 111.4)
at identical settings. Passing several seeds reports the median and the full
spread instead, which is what any comparison between arms should rest on.

Early stopping is OFF here by design. cartpole_dqn.train() defaults
stop_on_eval=True, which ends a run as soon as its greedy policy clears the
bar -- excellent for a single run, but it would let arms train for different
numbers of steps, and then a weaker greedy score might only mean that arm quit
sooner. Matched budgets are what make "any difference is attributable to the
network" true, so pass --stop-on-eval only if you want sample-efficiency
(steps-to-solve) rather than quality-at-a-fixed-budget. The summary prints the
mean step count per arm either way, so an unmatched comparison is visible.

Runtime note: the quantum arm uses the torch statevector backend (~8 ms per
gradient step). ``--quantum-backend qiskit`` is ~1900x slower and is only
practical on a smoke test, to confirm the two agree.
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

# arm name -> (agent kind, network construction kwargs)
ARMS = {
    "classical":       ("classical", {"hidden": (32, 32)}),
    "classical-small": ("classical", {"hidden": (10,)}),
    "quantum":         ("quantum",   {}),
    "quantum-noisy":   ("quantum",   {"backend": "torch-noisy"}),
}


def run_one(arm, seed, args):
    """Train one arm at one seed; returns run_data."""
    agent, kwargs = ARMS[arm]
    print(f"\n{'=' * 72}\n{arm.upper()}  seed={seed}\n{'=' * 72}")

    if agent == "quantum":
        cartpole_dqn.QUANTUM_KWARGS = {
            "n_layers": args.n_layers, "backend": args.quantum_backend,
            "seed": seed, **kwargs}
        if kwargs.get("backend") == "torch-noisy":
            cartpole_dqn.QUANTUM_KWARGS["shots"] = args.noisy_shots
    else:
        cartpole_dqn.CLASSICAL_KWARGS = dict(kwargs)

    # Name runs by arm, not by agent: both classical arms share agent
    # "classical" and would otherwise overwrite each other's output files.
    name = f"{arm}_s{seed}"

    tic = time.time()
    run = train(
        agent=agent,
        episodes=args.episodes,
        total_steps=args.total_steps,
        lr=args.lr_quantum if agent == "quantum" else args.lr_classical,
        eval_every_steps=args.eval_every_steps,
        eval_episodes=args.eval_episodes,
        n_step=args.n_step,
        stop_on_eval=args.stop_on_eval,
        seed=seed,
        out_dir=args.out_dir,
        name=name,
    )
    run.update(arm=arm, seed=seed, wall_sec=time.time() - tic,
               checkpoint=os.path.join(args.out_dir, f"{name}_policy.pt"))
    return run


def summarize(results):
    rows = []
    for arm, runs in results.items():
        greedy = [r["greedy_eval_mean"] for r in runs]
        rows.append({
            "arm": arm,
            "params": runs[0]["params"],
            "seeds": len(runs),
            "greedy_median": statistics.median(greedy),
            "greedy_mean": statistics.mean(greedy),
            "greedy_min": min(greedy),
            "greedy_max": max(greedy),
            "solved": sum(r["solved"] for r in runs),
            "steps": statistics.mean(sum(r["episode_rewards"]) for r in runs),
            "wall_sec": statistics.mean(r["wall_sec"] for r in runs),
        })
    return sorted(rows, key=lambda r: -r["greedy_median"])


def print_summary(rows, solve_reward=475.0):
    print(f"\n{'=' * 93}\nRESULTS   median over seeds; 'solved' = greedy mean >= "
          f"{solve_reward:.0f}\n{'=' * 93}")
    print(f"{'arm':<18}{'params':>8}{'median':>10}{'mean':>9}{'min':>8}{'max':>8}"
          f"{'solved':>9}{'steps':>9}{'wall':>10}")
    print("-" * 93)
    for r in rows:
        solved = f"{r['solved']}/{r['seeds']}"
        print(f"{r['arm']:<18}{r['params']:>8}{r['greedy_median']:>10.1f}"
              f"{r['greedy_mean']:>9.1f}{r['greedy_min']:>8.0f}{r['greedy_max']:>8.0f}"
              f"{solved:>9}{r['steps']:>9.0f}{r['wall_sec']:>9.0f}s")
    print("-" * 93)


def plot_curves(results, out_dir, window=50):
    """Median training curve per arm, with the inter-seed spread shaded."""
    fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    colors = {"classical": "tab:blue", "classical-small": "tab:cyan",
              "quantum": "tab:purple", "quantum-noisy": "tab:red"}

    for arm, runs in results.items():
        col = colors.get(arm)
        longest = max(len(r["episode_rewards"]) for r in runs)
        # Pad short runs with their final value, so the median does not lurch
        # downward when one seed stops earlier than the others.
        padded = np.array([r["episode_rewards"] + [r["episode_rewards"][-1]] *
                           (longest - len(r["episode_rewards"])) for r in runs])
        med = np.median(padded, axis=0)
        lo = np.percentile(padded, 25, axis=0)
        hi = np.percentile(padded, 75, axis=0)
        if len(med) >= window:
            k = np.ones(window) / window
            med, lo, hi = (np.convolve(v, k, mode="valid") for v in (med, lo, hi))
            xs = np.arange(window - 1, longest)
        else:
            xs = np.arange(longest)
        ax_t.plot(xs, med, color=col, label=f"{arm} ({runs[0]['params']} params)")
        ax_t.fill_between(xs, lo, hi, color=col, alpha=0.15)

        # Greedy evaluation is the metric 'solved' is actually judged on, and
        # unlike training reward it is not depressed by exploration noise.
        if runs[0].get("eval_history"):
            n = min(len(r["eval_history"]) for r in runs)
            steps = [s for s, _ in runs[0]["eval_history"][:n]]
            scores = np.median([[v for _, v in r["eval_history"][:n]] for r in runs], axis=0)
            ax_e.plot(steps, scores, color=col, marker="o", ms=3, label=arm)

    for ax, title, xlabel in (
            (ax_t, f"Training reward (median, {window}-episode mean)", "Episode"),
            (ax_e, "Greedy evaluation during training", "Environment step")):
        ax.axhline(475, color="green", ls="--", alpha=0.5, label="solved (475)")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Reward")
        ax.legend()

    plt.suptitle("CartPole-v1: classical DQN vs DRL+VQC")
    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_curves.png")
    plt.savefig(path)
    plt.close(fig)
    print(f"Saved training plots to {path}")


def make_gifs(results, out_dir):
    """Animate the best seed of each arm."""
    try:
        from watch import animate, load_policy_net, rollout
    except ImportError as exc:
        print(f"(skipping GIFs: {exc})")
        return
    for arm, runs in results.items():
        best = max(runs, key=lambda r: r["greedy_eval_mean"])
        if not os.path.exists(best["checkpoint"]):
            print(f"(skipping {arm} GIF: {best['checkpoint']} not found)")
            continue
        net = load_policy_net(best["checkpoint"])
        states = rollout(net, seed=0)
        path = os.path.join(out_dir, f"{arm}_solved.gif")
        animate(states, path, stride=2)
        print(f"Saved {path}  ({len(states) - 1} steps, seed {best['seed']}, "
              f"greedy {best['greedy_eval_mean']:.0f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0],
                   help="One seed by default. Pass several (e.g. --seeds 0 1 2 3 4) "
                        "to get medians and the inter-seed spread.")
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--lr-classical", type=float, default=2.3e-3)
    p.add_argument("--lr-quantum", type=float, default=1e-3)
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--eval-every-steps", type=int, default=2500)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--n-step", type=int, default=3,
                   help="n-step return window, passed to every arm alike.")
    p.add_argument("--stop-on-eval", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Stop each arm once its greedy policy clears the bar. "
                        "Off by default: it unmatches the step budgets the "
                        "cross-arm comparison rests on.")
    p.add_argument("--quantum-backend", default="torch")
    p.add_argument("--noisy-shots", type=int, default=1024,
                   help="Finite-sampling noise for the quantum-noisy arm.")
    p.add_argument("--out-dir", default="results/benchmark")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        # Step budget must be the binding limit and must clear learning_starts
        # (1000), or no gradient step runs and the check proves nothing.
        args.total_steps, args.seeds = 6000, [0]
        args.eval_every_steps = 2000
        print("SMOKE MODE: 6k steps, 1 seed — checks wiring, not performance.\n")

    os.makedirs(args.out_dir, exist_ok=True)
    results = {a: [run_one(a, s, args) for s in args.seeds] for a in args.arms}

    rows = summarize(results)
    print_summary(rows)
    plot_curves(results, args.out_dir)
    make_gifs(results, args.out_dir)

    path = os.path.join(args.out_dir, "benchmark.json")
    with open(path, "w") as f:
        json.dump({
            "config": vars(args),
            "summary": rows,
            "runs": {a: [{k: v for k, v in r.items() if k != "episode_rewards"}
                         for r in rs] for a, rs in results.items()},
        }, f, indent=2)
    print(f"Saved summary to {path}")


if __name__ == "__main__":
    main()
