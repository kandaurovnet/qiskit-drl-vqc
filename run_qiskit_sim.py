#!/usr/bin/env python3
"""Run a trained VQC agent on Qiskit simulators — the rehearsal for hardware.

Training uses the torch statevector executor because it is ~1900x faster, but
that path exists only in simulation. This script runs the *same trained weights*
through Qiskit instead, across the three regimes that sit between training and a
real QPU:

    exact    StatevectorEstimator, no sampling      -- what training assumed
    shots    StatevectorEstimator, 1/sqrt(N) noise  -- what any QPU forces on you
    noisy    a fake IBM device: gate errors, readout error, real coupling map,
             and the SWAPs its connectivity requires        -- closest to hardware

Everything here is local: no IBM account, no queue, no QPU quota. Anything that
fails here will fail on hardware too, so this is where to find it.

    python run_qiskit_sim.py results_q3seed/quantum_s2_policy.pt
    python run_qiskit_sim.py <ckpt> --mode noisy --device FakeManilaV2 --episodes 3
    python run_qiskit_sim.py <ckpt> --mode shots --shots 1024 --episodes 10

Runtime note: `noisy` transpiles to the device ISA and estimates every
expectation value from samples, so it is far slower than `exact`. Start with 1-2
episodes; a 500-step episode is 500 sequential estimator calls.
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


def build_backend(mode, device, shots):
    """Return the (backend, shots) pair that ``load_vqc`` needs for this mode."""
    if mode == "exact":
        return "qiskit", None
    if mode == "shots":
        return "qiskit", shots
    # A fake device carries the calibrated noise model and coupling map of a real
    # machine, so vqc.py takes its hardware path: transpile to ISA, then
    # apply_layout the observables onto the physical qubits they now act on.
    from qiskit_ibm_runtime import fake_provider
    try:
        return getattr(fake_provider, device)(), shots
    except AttributeError:
        raise SystemExit(f"unknown device {device!r}; try FakeMarrakesh, FakeTorino, "
                         f"FakeSherbrooke (modern) or FakeManilaV2 (2021-era, 5 qubits)")


def rollout(model, episodes, seed):
    """Greedy episodes. Returns per-episode rewards."""
    env = gym.make("CartPole-v1")
    rewards = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total = 0.0
        for _ in range(MAX_STEPS):
            state = torch.tensor(normalize_obs(obs))
            with torch.no_grad():
                action = int(model(state.unsqueeze(0)).argmax(1).item())
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            if terminated or truncated:
                break
        rewards.append(total)
    env.close()
    return rewards


def circuit_stats(model):
    """Depth and two-qubit gate count — the hardware cost the plan still lists as TBD."""
    qc = model.circuit
    ops = qc.count_ops()
    two_q = sum(n for g, n in ops.items() if g in ("cz", "cx", "ecr", "cy"))
    return {"qubits": qc.num_qubits, "depth": qc.depth(), "two_qubit_gates": two_q,
            "ops": dict(ops)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--mode", choices=["exact", "shots", "noisy"], default="exact")
    # Defaults to the fake twin of ibm_backend_inference.DEFAULT_BACKEND, so a
    # noisy run here rehearses the exact device the hardware path targets.
    p.add_argument("--device", default="FakeMarrakesh", help="fake backend for --mode noisy")
    p.add_argument("--shots", type=int, default=1024,
                   help="ignored by --mode exact; a backend with no shots defaults to 1024")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=10_000, help="env seed base, as in training eval")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    backend, shots = build_backend(args.mode, args.device, args.shots)
    label = args.mode if args.mode != "noisy" else f"{args.mode}:{args.device}"
    print(f"loading {args.checkpoint} on {label} (shots={shots})...", flush=True)

    t0 = time.time()
    model = load_vqc(args.checkpoint, backend=backend, shots=shots)
    stats = circuit_stats(model)
    print(f"  circuit: {stats['qubits']} qubits, depth {stats['depth']}, "
          f"{stats['two_qubit_gates']} two-qubit gates")
    if args.mode == "noisy":
        # Transpilation inflates both: the ansatz is rewritten into the device's
        # native gates and routed across its coupling map with SWAPs.
        print(f"  (transpiled for {args.device}; depth here is post-ISA)")

    rewards = rollout(model, args.episodes, args.seed)
    elapsed = time.time() - t0

    print(f"\n{label}: mean reward {np.mean(rewards):.1f} "
          f"(min {min(rewards):.0f}, max {max(rewards):.0f}) "
          f"over {args.episodes} episodes, {elapsed:.0f}s")
    print(f"  {'SOLVED' if np.mean(rewards) >= 475 else 'not solved'} (bar 475)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "mode": args.mode,
                       "device": args.device if args.mode == "noisy" else None,
                       "shots": shots, "episodes": args.episodes,
                       "rewards": rewards, "reward_mean": float(np.mean(rewards)),
                       "elapsed_sec": elapsed, "circuit": stats}, f, indent=2)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
