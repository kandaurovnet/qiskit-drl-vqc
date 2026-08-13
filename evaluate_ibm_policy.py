#!/usr/bin/env python3
"""Create a local preview for trained-policy IBM hardware evaluation.

This script deliberately performs no IBM connection and submits no Runtime
job.  It loads the trained torch-statevector VQC, follows the same normalized
greedy evaluation path as ``cartpole_dqn.evaluate()``, and selects one
high-margin state for each CartPole action.  The resulting JSON is the input
manifest for the later two-state IBM Runtime evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from qiskit.primitives import StatevectorEstimator

from cartpole_dqn import normalize_obs
from vqc import N_LAYERS, VQCQNetwork, build_observables, build_qnetwork_circuit


DEFAULT_CHECKPOINT = Path("results/quantum_policy.pt")
DEFAULT_OUTPUT = Path("results/ibm_policy_preview.json")
DEFAULT_SEED_START = 10_000
# Torch executes the circuit in float32, while Qiskit statevector simulation
# uses float64.  The trained output scale is approximately 100, so sub-micro
# expectation differences can become roughly 1e-5 to 1e-4 in Q-value units.
EXACT_VALIDATION_ATOL = 1e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Load and validate the trained VQC state dictionary."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    expected_keys = {"theta", "input_scale", "output_scale", "_obs"}
    missing = expected_keys - set(checkpoint)
    if missing:
        raise KeyError(f"Checkpoint is missing required entries: {sorted(missing)}")
    return checkpoint


def load_policy(checkpoint: dict[str, torch.Tensor], n_layers: int) -> VQCQNetwork:
    """Load the exact torch VQC architecture used by ``quantum_run``."""
    model = VQCQNetwork(n_layers=n_layers, backend="torch", seed=0)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    trainable_parameters = sum(p.numel() for p in model.parameters())
    if trainable_parameters != 46:
        raise RuntimeError(
            f"Expected the 46-parameter trained VQC, got {trainable_parameters}."
        )
    return model


def collect_greedy_states(
    model: VQCQNetwork,
    *,
    seed_start: int,
    max_episodes: int,
) -> tuple[list[dict], list[dict]]:
    """Collect states from the standard greedy CartPole evaluation path."""
    env = gym.make("CartPole-v1")
    candidates: list[dict] = []
    episodes: list[dict] = []

    try:
        for episode_index in range(max_episodes):
            seed = seed_start + episode_index
            observation, _ = env.reset(seed=seed)
            episode_reward = 0.0

            for step in range(500):
                raw = np.asarray(observation, dtype=np.float32)
                normalized = normalize_obs(raw)
                state = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    q_values = model(state)[0].cpu().numpy().astype(float)

                if q_values.shape != (2,) or not np.isfinite(q_values).all():
                    raise RuntimeError(f"Invalid Q-values at seed={seed}, step={step}: {q_values}")

                action = int(np.argmax(q_values))
                margin = float(abs(q_values[0] - q_values[1]))
                candidates.append(
                    {
                        "evaluation_seed": seed,
                        "episode_index": episode_index,
                        "step": step,
                        "raw_observation": [float(value) for value in raw],
                        "normalized_observation": [float(value) for value in normalized],
                        "exact_q_values": [float(value) for value in q_values],
                        "exact_action": action,
                        "exact_margin": margin,
                    }
                )

                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
                if terminated or truncated:
                    break

            episodes.append(
                {
                    "evaluation_seed": seed,
                    "reward": episode_reward,
                    "steps": step + 1,
                }
            )

            if {item["exact_action"] for item in candidates} == {0, 1}:
                break
    finally:
        env.close()

    return candidates, episodes


def select_high_margin_actions(candidates: list[dict]) -> list[dict]:
    """Select one high-confidence state for action 0 and one for action 1."""
    selected = []
    for action in (0, 1):
        matching = [item for item in candidates if item["exact_action"] == action]
        if not matching:
            raise RuntimeError(f"No action-{action} state was collected.")
        selected.append(max(matching, key=lambda item: item["exact_margin"]))

    if {item["exact_action"] for item in selected} != {0, 1}:
        raise AssertionError("Preview must contain exactly one state for each action.")
    if any(item["exact_margin"] <= 0 for item in selected):
        raise AssertionError("Selected states must have a positive Q-value margin.")
    return selected


def validate_selected_states_with_qiskit(
    selected_states: list[dict],
    checkpoint: dict[str, torch.Tensor],
    *,
    n_layers: int,
    atol: float = EXACT_VALIDATION_ATOL,
) -> dict:
    """Verify Torch Q-values with the exact Qiskit statevector estimator.

    Values are assigned through Parameter objects rather than relying on the
    circuit's displayed parameter order.  This is the same explicit mapping
    required by the later transpiled IBM Runtime circuit.
    """
    circuit, input_parameters, weight_parameters = build_qnetwork_circuit(
        n_layers=n_layers
    )
    observables = build_observables()
    estimator = StatevectorEstimator(seed=0)

    theta = checkpoint["theta"].detach().cpu().numpy().astype(float)
    input_scale = checkpoint["input_scale"].detach().cpu().numpy().astype(float)
    output_scale = checkpoint["output_scale"].detach().cpu().numpy().astype(float)

    expected_theta = len(weight_parameters)
    expected_inputs = len(input_parameters)
    if theta.shape != (expected_theta,):
        raise RuntimeError(f"Expected {expected_theta} theta values, got {theta.shape}.")
    if input_scale.shape != (n_layers, 4):
        raise RuntimeError(
            f"Expected input_scale shape {(n_layers, 4)}, got {input_scale.shape}."
        )
    if output_scale.shape != (2,):
        raise RuntimeError(f"Expected two output scales, got {output_scale.shape}.")

    max_abs_difference = 0.0
    for state in selected_states:
        normalized = np.asarray(state["normalized_observation"], dtype=float)
        input_angles = (input_scale * normalized[None, :]).reshape(-1)
        if input_angles.shape != (expected_inputs,):
            raise RuntimeError(
                f"Expected {expected_inputs} input angles, got {input_angles.shape}."
            )

        values_by_parameter = {
            **dict(zip(input_parameters, input_angles, strict=True)),
            **dict(zip(weight_parameters, theta, strict=True)),
        }
        parameter_values = [
            float(values_by_parameter[parameter]) for parameter in circuit.parameters
        ]
        if len(parameter_values) != expected_inputs + expected_theta:
            raise RuntimeError("Unexpected number of bound circuit parameters.")

        result = estimator.run(
            [(circuit, observables, [parameter_values])]
        ).result()[0]
        expectations = np.asarray(result.data.evs, dtype=float).reshape(-1)
        if expectations.shape != (2,) or not np.isfinite(expectations).all():
            raise RuntimeError(f"Invalid Qiskit expectations: {expectations}")

        qiskit_q_values = expectations * output_scale
        torch_q_values = np.asarray(state["exact_q_values"], dtype=float)
        absolute_difference = np.abs(qiskit_q_values - torch_q_values)
        max_abs_difference = max(max_abs_difference, float(absolute_difference.max()))

        np.testing.assert_allclose(
            qiskit_q_values,
            torch_q_values,
            rtol=0.0,
            atol=atol,
            err_msg=(
                f"Torch/Qiskit mismatch for seed={state['evaluation_seed']} "
                f"step={state['step']}"
            ),
        )
        qiskit_action = int(np.argmax(qiskit_q_values))
        if qiskit_action != state["exact_action"]:
            raise AssertionError("Torch and Qiskit selected different actions.")

        state.update(
            {
                "qiskit_exact_expectations": [float(value) for value in expectations],
                "qiskit_exact_q_values": [float(value) for value in qiskit_q_values],
                "qiskit_exact_action": qiskit_action,
                "torch_qiskit_abs_difference": [
                    float(value) for value in absolute_difference
                ],
            }
        )

    return {
        "status": "passed",
        "backend": "qiskit.primitives.StatevectorEstimator",
        "absolute_tolerance": atol,
        "max_absolute_q_value_difference": max_abs_difference,
        "input_parameters": expected_inputs,
        "weight_parameters": expected_theta,
        "output_scale_parameters": len(output_scale),
    }


def create_preview(
    checkpoint_path: Path,
    output_path: Path,
    *,
    seed_start: int = DEFAULT_SEED_START,
    max_episodes: int = 100,
    n_layers: int = N_LAYERS,
) -> dict:
    """Build and save a two-state local preview without contacting IBM."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path)
    model = load_policy(checkpoint, n_layers)
    candidates, episodes = collect_greedy_states(
        model,
        seed_start=seed_start,
        max_episodes=max_episodes,
    )
    selected = select_high_margin_actions(candidates)
    exact_validation = validate_selected_states_with_qiskit(
        selected,
        checkpoint,
        n_layers=n_layers,
    )

    preview = {
        "status": "preview",
        "qpu_job_submitted": False,
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "model": {
            "class": "VQCQNetwork",
            "n_layers": n_layers,
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "training_backend": "torch",
        },
        "state_source": {
            "environment": "CartPole-v1",
            "policy": "greedy",
            "normalization": "cartpole_dqn.normalize_obs",
            "evaluation_seed_start": seed_start,
            "episodes_used": len(episodes),
            "candidate_states": len(candidates),
            "episodes": episodes,
        },
        "selection": {
            "strategy": "maximum exact Q-value margin per action",
            "number_of_states": len(selected),
        },
        "torch_qiskit_exact_validation": exact_validation,
        "selected_states": selected,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(preview, indent=2), encoding="utf-8")
    return preview


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a local two-state preview for IBM VQC evaluation."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--max-episodes", type=int, default=100)
    args = parser.parse_args()

    preview = create_preview(
        args.checkpoint,
        args.output,
        seed_start=args.seed_start,
        max_episodes=args.max_episodes,
    )

    source = preview["state_source"]
    print(f"Checkpoint: {preview['checkpoint']}")
    print(f"Checkpoint SHA-256: {preview['checkpoint_sha256']}")
    print(f"Candidates collected: {source['candidate_states']}")
    print(f"Evaluation episodes used: {source['episodes_used']}")
    for index, state in enumerate(preview["selected_states"]):
        q_left, q_right = state["exact_q_values"]
        print(
            f"State {index}: seed={state['evaluation_seed']} step={state['step']} "
            f"Q=({q_left:.6f}, {q_right:.6f}) "
            f"action={state['exact_action']} margin={state['exact_margin']:.6f}"
        )
        qk_left, qk_right = state["qiskit_exact_q_values"]
        print(
            f"         Qiskit exact=({qk_left:.6f}, {qk_right:.6f}) "
            f"action={state['qiskit_exact_action']}"
        )
    validation = preview["torch_qiskit_exact_validation"]
    print(
        "Torch/Qiskit exact validation: "
        f"{validation['status']} "
        f"(max |delta Q|={validation['max_absolute_q_value_difference']:.3e})"
    )
    print(f"Preview saved to: {args.output}")
    print("No IBM connection was made and no QPU job was submitted.")


if __name__ == "__main__":
    main()
