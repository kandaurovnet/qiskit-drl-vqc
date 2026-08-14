#!/usr/bin/env python3
"""Q9: how much does finite-shot sampling cost a trained VQC agent?

Training runs on an exact statevector simulator, but a real QPU can only
estimate each expectation value from a finite number of shots (standard error
~1/sqrt(shots)). This measures what that noise does to a *trained* policy,
without retraining anything.

Two complementary measurements, because either alone is misleading:

1. Average episode reward. The metric the challenge actually asks for, but it
   confounds "the noise flipped a decision" with "the flip did not matter".
2. Action agreement against the exact model, on the states the exact policy
   really visits, bucketed by Q-value margin. A flip on a state where the two
   Q-values differ by 0.3% is not the same failure as a flip on a state where
   the policy was confident.

    python eval_shots.py results_q3seed/quantum_s2_policy.pt
    python eval_shots.py <ckpt> --episodes 20 --shots 1024 256 64
"""

import argparse
import json
import time

import gymnasium as gym
import numpy as np
import torch

from cartpole_dqn import normalize_obs
from vqc_checkpoint import load_vqc

MAX_STEPS = 500  # CartPole-v1 time limit


def rollout(model, episodes, seed, collect=False):
    """Greedy episodes. Returns (rewards, visited states) — states only if collect."""
    env = gym.make("CartPole-v1")
    rewards, states = [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total = 0.0
        for _ in range(MAX_STEPS):
            s = torch.tensor(normalize_obs(obs))
            if collect:
                states.append(s)
            with torch.no_grad():
                action = int(model(s.unsqueeze(0)).argmax(1).item())
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            if terminated or truncated:
                break
        rewards.append(total)
    env.close()
    return rewards, (torch.stack(states) if states else None)


def q_values(model, states, chunk=64):
    """Q-values for a batch of states, chunked to keep the estimator call sane."""
    out = []
    with torch.no_grad():
        for i in range(0, len(states), chunk):
            out.append(model(states[i:i + chunk]))
    return torch.cat(out)


def margin_buckets(q_exact, agree, edges=(0.01, 0.05, 0.20)):
    """Agreement rate split by how decisive the exact policy was.

    Margin is |Q0 - Q1| relative to the larger |Q|, so it is comparable across
    states whose absolute Q-values differ.
    """
    m = (q_exact[:, 0] - q_exact[:, 1]).abs() / q_exact.abs().max(1).values.clamp(min=1e-9)
    rows, lo = [], 0.0
    for hi in (*edges, float("inf")):
        sel = (m >= lo) & (m < hi)
        n = int(sel.sum())
        rows.append((lo, hi, n, float(agree[sel].float().mean()) if n else float("nan")))
        lo = hi
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--shots", nargs="+", type=int, default=[1024, 256, 64])
    p.add_argument("--seed", type=int, default=10_000, help="env seed base, as in training eval")
    p.add_argument("--max-states", type=int, default=1500)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    # The exact Qiskit model is the reference: same executor as the shot runs, so
    # any difference is sampling noise rather than a change of backend.
    print(f"exact reference ({args.episodes} episodes)...", flush=True)
    exact = load_vqc(args.checkpoint, backend="qiskit", shots=None)
    t0 = time.time()
    rew_exact, states = rollout(exact, args.episodes, args.seed, collect=True)
    if len(states) > args.max_states:  # evenly thinned, so all episodes stay represented
        states = states[torch.linspace(0, len(states) - 1, args.max_states).long()]
    q_exact = q_values(exact, states)
    act_exact = q_exact.argmax(1)
    print(f"  reward {np.mean(rew_exact):6.1f}   {len(states)} states   {time.time()-t0:.0f}s")

    rows = [{"shots": None, "reward_mean": float(np.mean(rew_exact)),
             "reward_min": min(rew_exact), "reward_max": max(rew_exact),
             "agreement": 1.0, "q_err_max": 0.0, "q_err_mean": 0.0}]

    for n in args.shots:
        print(f"shots={n}...", flush=True)
        t0 = time.time()
        m = load_vqc(args.checkpoint, backend="qiskit", shots=n)
        rew, _ = rollout(m, args.episodes, args.seed)
        q = q_values(m, states)
        agree = q.argmax(1) == act_exact
        err = (q - q_exact).abs()
        rows.append({
            "shots": n,
            "reward_mean": float(np.mean(rew)), "reward_min": min(rew), "reward_max": max(rew),
            "agreement": float(agree.float().mean()),
            "q_err_max": float(err.max()), "q_err_mean": float(err.mean()),
            "margin_buckets": margin_buckets(q_exact, agree),
        })
        print(f"  reward {np.mean(rew):6.1f}   agreement {agree.float().mean():.1%}   "
              f"{time.time()-t0:.0f}s")

    print(f"\n{'shots':>7}{'reward':>10}{'range':>14}{'agreement':>12}"
          f"{'|dQ| max':>11}{'|dQ| mean':>11}")
    print("-" * 65)
    for r in rows:
        rng = f"{r['reward_min']:.0f}-{r['reward_max']:.0f}"
        print(f"{str(r['shots'] or 'exact'):>7}{r['reward_mean']:>10.1f}{rng:>14}"
              f"{r['agreement']:>11.1%}{r['q_err_max']:>11.2f}{r['q_err_mean']:>11.2f}")

    print(f"\nAction agreement by Q-margin (fraction of |Q0-Q1| over max|Q|):")
    print(f"{'margin':>16}{'states':>8}" + "".join(f"{s:>10}" for s in args.shots))
    bucket_rows = [r["margin_buckets"] for r in rows[1:]]
    for i, (lo, hi, n, _) in enumerate(bucket_rows[0]):
        label = f"{lo:.0%}-{hi:.0%}" if hi != float("inf") else f">{lo:.0%}"
        cells = "".join(f"{b[i][3]:>9.1%}" if b[i][2] else f"{'-':>10}" for b in bucket_rows)
        print(f"{label:>16}{n:>8}{cells}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "episodes": args.episodes,
                       "results": rows}, f, indent=2)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
