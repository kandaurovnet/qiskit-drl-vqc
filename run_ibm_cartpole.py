#!/usr/bin/env python3
"""Run a short greedy CartPole episode with VQC inference on IBM hardware.

This is evaluation only.  The VQC is trained with the Torch statevector path,
then its frozen checkpoint is executed once per environment step through IBM
Runtime EstimatorV2.  Progress and each job ID are saved immediately so an
interrupted run can be resumed without submitting the same step twice.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

from cartpole_dqn import normalize_obs
from ibm_backend_inference import ACCOUNT_NAME, DEFAULT_BACKEND, prepare_vqc_for_backend
from vqc_checkpoint import infer_shape, load_vqc


DEFAULT_CHECKPOINT = Path("results/quantum_policy.pt")
DEFAULT_OUTPUT = Path("results/ibm_cartpole_run.json")


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _artifact_paths(output_path: Path) -> tuple[Path, Path]:
    """Return frozen-policy and evaluation-plot artifact names."""
    name = output_path.name
    prefix = name[:-len("_run.json")] if name.endswith("_run.json") else output_path.stem
    return (
        output_path.with_name(f"{prefix}_policy.pt"),
        output_path.with_name(f"{prefix}_evaluation.png"),
    )


def save_evaluation_artifacts(run: dict, checkpoint_path: Path, output_path: Path) -> None:
    """Create an evaluation artifact triplet without implying IBM training.

    The policy file is an exact copy of the frozen Torch-trained checkpoint;
    IBM hardware performs evaluation only and does not create new weights.
    """
    policy_path, plot_path = _artifact_paths(output_path)
    if checkpoint_path.resolve() != policy_path.resolve():
        shutil.copyfile(checkpoint_path, policy_path)

    steps = run["steps"]
    exact_model = load_vqc(checkpoint_path, backend="torch")
    normalized_batch = torch.tensor(
        [step["normalized_observation"] for step in steps], dtype=torch.float32
    )
    with torch.no_grad():
        exact_batch = exact_model(normalized_batch).cpu().numpy().astype(float)
    agreements = []
    for step, exact_q in zip(steps, exact_batch, strict=True):
        hardware_q = np.asarray(step["q_values"], dtype=float)
        exact_action = int(np.argmax(exact_q))
        agreement = int(step["action"]) == exact_action
        agreements.append(agreement)
        step["torch_exact_q_values"] = [float(v) for v in exact_q]
        step["torch_exact_action"] = exact_action
        step["hardware_exact_action_agreement"] = agreement
        step["absolute_q_value_error"] = [
            float(v) for v in np.abs(hardware_q - exact_q)
        ]

    xs = np.arange(1, len(steps) + 1)
    rewards = np.asarray([step["reward"] for step in steps], dtype=float)
    q_values = np.asarray([step["q_values"] for step in steps], dtype=float)
    actions = np.asarray([step["action"] for step in steps], dtype=int)

    fig, (ax_reward, ax_q) = plt.subplots(2, 1, figsize=(9, 7), dpi=150, sharex=True)
    ax_reward.step(xs, np.cumsum(rewards), where="post", color="tab:green", linewidth=2)
    ax_reward.set_ylabel("Cumulative reward")
    ax_reward.set_title("CartPole-v1 IBM hardware evaluation (frozen policy; no training)")
    ax_reward.grid(alpha=0.25)

    ax_q.plot(xs, q_values[:, 0], marker="o", label="Q(action 0)", color="tab:blue")
    ax_q.plot(xs, q_values[:, 1], marker="o", label="Q(action 1)", color="tab:orange")
    for x, action in zip(xs, actions, strict=True):
        ax_q.annotate(f"a={action}", (x, q_values[x - 1, action]),
                      xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
    ax_q.set_xlabel("IBM CartPole environment step")
    ax_q.set_ylabel("Hardware Q-value")
    ax_q.grid(alpha=0.25)
    ax_q.legend()
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    run["policy_artifact"] = policy_path.as_posix()
    run["policy_artifact_role"] = "exact copy of frozen Torch-trained VQC checkpoint"
    # Remove names produced by the earlier misleading training-plot convention.
    run.pop("training_plot", None)
    run.pop("training_plot_role", None)
    run["evaluation_plot"] = plot_path.as_posix()
    run["evaluation_plot_role"] = "IBM evaluation trajectory; no hardware training"
    run["torch_exact_reference"] = True
    run["hardware_exact_action_agreement_count"] = int(sum(agreements))
    run["hardware_exact_action_agreement_rate"] = float(np.mean(agreements))
    run["max_absolute_q_value_error"] = float(
        max(max(step["absolute_q_value_error"]) for step in steps)
    )
    _save(output_path, run)


def load_checkpoint(path: Path, prepared) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"theta", "input_scale", "output_scale"}
    missing = required - set(checkpoint)
    if missing:
        raise KeyError(f"Checkpoint is missing: {sorted(missing)}")

    theta = checkpoint["theta"].detach().cpu().numpy().astype(float)
    input_scale = checkpoint["input_scale"].detach().cpu().numpy().astype(float)
    output_scale = checkpoint["output_scale"].detach().cpu().numpy().astype(float)
    if theta.shape != (len(prepared.weight_parameters),):
        raise RuntimeError(f"Expected {len(prepared.weight_parameters)} weights, got {theta.shape}.")
    expected_input_shape = (prepared.n_layers, prepared.n_qubits)
    if input_scale.shape != expected_input_shape:
        raise RuntimeError(
            f"Expected input_scale shape {expected_input_shape}, got {input_scale.shape}."
        )
    if output_scale.shape != (2,):
        raise RuntimeError(f"Expected output_scale shape (2,), got {output_scale.shape}.")
    return theta, input_scale, output_scale


def bind_state(prepared, normalized_state, theta, input_scale) -> list[float]:
    normalized = np.asarray(normalized_state, dtype=float)
    if normalized.shape != (prepared.n_qubits,) or not np.isfinite(normalized).all():
        raise RuntimeError(f"Invalid normalized CartPole state: {normalized}")
    input_angles = (input_scale * normalized[None, :]).reshape(-1)
    values = {
        **dict(zip(prepared.input_parameters, input_angles, strict=True)),
        **dict(zip(prepared.weight_parameters, theta, strict=True)),
    }
    row = [float(values[parameter]) for parameter in prepared.circuit.parameters]
    expected_values = len(prepared.input_parameters) + len(prepared.weight_parameters)
    if len(row) != expected_values or not np.isfinite(row).all():
        raise RuntimeError(f"Expected {expected_values} finite VQC circuit values.")
    return row


def _new_run(*, backend, checkpoint, seed, max_steps, precision, prepared) -> dict:
    return {
        "status": "running",
        "agent": "ibm",
        "environment": "CartPole-v1",
        "evaluation_only": True,
        "policy_source": "trained quantum VQC checkpoint",
        "checkpoint": checkpoint.as_posix(),
        "inference_backend": backend,
        "execution": "qiskit_ibm_runtime.EstimatorV2",
        "params": (
            len(prepared.input_parameters) + len(prepared.weight_parameters) + 2
        ),
        "n_qubits": prepared.n_qubits,
        "n_layers": prepared.n_layers,
        "seed": seed,
        "episodes_requested": 1,
        "max_steps": max_steps,
        "target_precision": precision,
        "original_depth": prepared.original_depth,
        "isa_depth": prepared.transpiled_depth,
        "two_qubit_gates": prepared.two_qubit_gates,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": 0.0,
        "steps": [],
        "pending_step": None,
    }


def _restore_environment(run: dict):
    env = gym.make("CartPole-v1")
    observation, _ = env.reset(seed=int(run["seed"]))
    terminated = truncated = False
    for index, saved in enumerate(run["steps"]):
        expected = np.asarray(saved["observation"], dtype=float)
        if not np.allclose(observation, expected, rtol=0.0, atol=1e-6):
            env.close()
            raise RuntimeError(f"Cannot reproduce saved observation at step {saved['step']}.")
        observation, _, terminated, truncated, _ = env.step(int(saved["action"]))
        if (terminated or truncated) and index != len(run["steps"]) - 1:
            env.close()
            raise RuntimeError("Saved run contains steps after an environment terminal state.")
    return env, observation, terminated, truncated


def evaluate_ibm_cartpole(
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    output_path: Path = DEFAULT_OUTPUT,
    backend_name: str = DEFAULT_BACKEND,
    account_name: str = ACCOUNT_NAME,
    seed: int = 10_000,
    max_steps: int = 10,
    target_precision: float = 0.1,
    resume: bool = False,
) -> dict:
    """Execute one capped CartPole episode, submitting at most one job per step."""
    if max_steps < 1 or max_steps > 500:
        raise ValueError("max_steps must be between 1 and 500.")
    if target_precision <= 0:
        raise ValueError("target_precision must be positive.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if output_path.exists() and not resume:
        raise FileExistsError(
            f"{output_path} already exists. Use --ibm-resume to continue it, "
            "or choose a different --ibm-output."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    n_layers, n_qubits = infer_shape(checkpoint)
    prepared = prepare_vqc_for_backend(
        backend_name,
        account_name=account_name,
        n_qubits=n_qubits,
        n_layers=n_layers,
    )
    theta, input_scale, output_scale = load_checkpoint(checkpoint_path, prepared)

    if resume:
        if not output_path.is_file():
            raise FileNotFoundError(f"Cannot resume missing run: {output_path}")
        run = json.loads(output_path.read_text(encoding="utf-8"))
        if run.get("status") == "completed":
            return run
        expected = (backend_name, seed, max_steps, target_precision)
        actual = (
            run.get("inference_backend"), run.get("seed"), run.get("max_steps"),
            run.get("target_precision"),
        )
        if actual != expected:
            raise RuntimeError(f"Resume settings differ: saved={actual}, requested={expected}")
    else:
        run = _new_run(
            backend=prepared.backend.name,
            checkpoint=checkpoint_path,
            seed=seed,
            max_steps=max_steps,
            precision=target_precision,
            prepared=prepared,
        )
        _save(output_path, run)

    env, observation, terminated, truncated = _restore_environment(run)
    estimator = EstimatorV2(mode=prepared.backend)
    tic = time.time()
    try:
        while len(run["steps"]) < max_steps and not (terminated or truncated):
            step_index = len(run["steps"])
            normalized = normalize_obs(np.asarray(observation, dtype=np.float32))
            parameter_values = bind_state(
                prepared, normalized, theta, input_scale
            )

            pending = run.get("pending_step")
            if pending is not None:
                if int(pending["step"]) != step_index:
                    raise RuntimeError("Saved pending job does not match the current step.")
                if not np.allclose(
                    observation,
                    np.asarray(pending["observation"], dtype=float),
                    rtol=0.0,
                    atol=1e-6,
                ):
                    raise RuntimeError("Pending job observation cannot be reproduced.")
                service = QiskitRuntimeService(name=account_name)
                job = service.job(pending["job_id"])
                print(f"Resuming step {step_index}, job {job.job_id()}", flush=True)
            else:
                job = estimator.run(
                    [(prepared.circuit, list(prepared.observables), [parameter_values])],
                    precision=target_precision,
                )
                run["pending_step"] = {
                    "step": step_index,
                    "job_id": job.job_id(),
                    "observation": [float(v) for v in observation],
                    "normalized_observation": [float(v) for v in normalized],
                    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                _save(output_path, run)
                print(f"Step {step_index}: submitted job {job.job_id()}", flush=True)

            pub_result = job.result()[0]
            expectations = np.asarray(pub_result.data.evs, dtype=float).reshape(-1)
            if expectations.shape != (2,) or not np.isfinite(expectations).all():
                raise RuntimeError(f"Expected two expectation values, got {expectations}.")
            q_values = expectations * output_scale
            action = int(np.argmax(q_values))
            next_observation, reward, terminated, truncated, _ = env.step(action)

            run["steps"].append(
                {
                    "step": step_index,
                    "job_id": job.job_id(),
                    "observation": [float(v) for v in observation],
                    "normalized_observation": [float(v) for v in normalized],
                    "expectation_values": [float(v) for v in expectations],
                    "q_values": [float(v) for v in q_values],
                    "action": action,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            run["pending_step"] = None
            run["elapsed_sec"] += time.time() - tic
            tic = time.time()
            _save(output_path, run)
            print(
                f"Step {step_index}: Q=({q_values[0]:.6f}, {q_values[1]:.6f}), "
                f"action={action}, reward={reward}",
                flush=True,
            )
            observation = next_observation
            if terminated or truncated:
                break
    finally:
        env.close()

    total_reward = float(sum(step["reward"] for step in run["steps"]))
    run.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "episodes_run": 1,
            "episode_rewards": [total_reward],
            "greedy_eval_rewards": [total_reward],
            "greedy_eval_mean": total_reward,
            "solved": bool(total_reward >= 475 and not (len(run["steps"]) < 475)),
            "terminated": bool(terminated),
            "environment_truncated": bool(truncated),
            "truncated_by_step_cap": bool(
                len(run["steps"]) >= max_steps and not terminated and not truncated
            ),
            "job_ids": [step["job_id"] for step in run["steps"]],
            "wall_sec": float(run["elapsed_sec"]),
        }
    )
    save_evaluation_artifacts(run, checkpoint_path, output_path)
    return run
