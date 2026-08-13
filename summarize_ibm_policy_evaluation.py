#!/usr/bin/env python3
"""Summarize a completed IBM trained-policy evaluation for reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path("results/ibm_policy_evaluation_d9ulamt35hes73fk3400.json")
DEFAULT_JSON = Path("results/ibm_policy_hardware_summary.json")
DEFAULT_MARKDOWN = Path("results/ibm_policy_hardware_summary.md")


def summarize(result: dict) -> dict:
    if result.get("status") != "completed":
        raise RuntimeError("Only a completed IBM policy evaluation can be summarized.")
    states = result.get("runtime_results", [])
    if not states:
        raise RuntimeError("The evaluation has no Runtime state results.")

    errors = np.asarray([state["absolute_q_value_error"] for state in states], dtype=float)
    hardware_margins = np.asarray(
        [abs(state["runtime_q_values"][0] - state["runtime_q_values"][1]) for state in states]
    )
    exact_margins = np.asarray(
        [abs(state["exact_qiskit_q_values"][0] - state["exact_qiskit_q_values"][1]) for state in states]
    )
    agreements = np.asarray([state["action_agreement"] for state in states], dtype=bool)

    return {
        "status": "preliminary_hardware_evaluation",
        "source_result": f"results/ibm_policy_evaluation_{result['job_id']}.json",
        "backend": result["backend"],
        "job_id": result["job_id"],
        "target_precision": result["target_precision"],
        "states_evaluated": len(states),
        "action_agreement_count": int(agreements.sum()),
        "action_agreement_rate": float(agreements.mean()),
        "mean_absolute_q_value_error": float(errors.mean()),
        "max_absolute_q_value_error": float(errors.max()),
        "minimum_hardware_action_margin": float(hardware_margins.min()),
        "mean_hardware_action_margin": float(hardware_margins.mean()),
        "mean_exact_action_margin": float(exact_margins.mean()),
        "isa_depth": result["isa_depth"],
        "two_qubit_gates": result["two_qubit_gates"],
        "trainable_parameters": 46,
        "interpretation": (
            "Both selected actions agreed with exact simulation, but this is a "
            "two-state preliminary result. The minimum hardware action margin is "
            "small, so broader repeated-state testing is required before claiming "
            "hardware robustness."
        ),
    }


def markdown(summary: dict, result: dict) -> str:
    rows = []
    for state in result["runtime_results"]:
        exact = state["exact_qiskit_q_values"]
        hardware = state["runtime_q_values"]
        margin = abs(hardware[0] - hardware[1])
        rows.append(
            f"| {state['state_index']} | ({exact[0]:.3f}, {exact[1]:.3f}) "
            f"| ({hardware[0]:.3f}, {hardware[1]:.3f}) "
            f"| {state['exact_action']} | {state['runtime_action']} "
            f"| {'yes' if state['action_agreement'] else 'no'} | {margin:.3f} |"
        )

    return "\n".join(
        [
            "# Preliminary IBM hardware policy evaluation",
            "",
            f"- Backend: `{summary['backend']}`",
            f"- Runtime job: `{summary['job_id']}`",
            f"- Target precision: `{summary['target_precision']}`",
            f"- Trained VQC parameters: `{summary['trainable_parameters']}`",
            f"- ISA depth / two-qubit gates: `{summary['isa_depth']}` / `{summary['two_qubit_gates']}`",
            "",
            "| State | Exact Q-values | Hardware Q-values | Exact action | Hardware action | Agreement | Hardware margin |",
            "|---:|---:|---:|---:|---:|:---:|---:|",
            *rows,
            "",
            f"Action agreement was **{summary['action_agreement_count']}/{summary['states_evaluated']} "
            f"({summary['action_agreement_rate']:.0%})**. Mean absolute Q-value error was "
            f"**{summary['mean_absolute_q_value_error']:.3f}** and maximum absolute error was "
            f"**{summary['max_absolute_q_value_error']:.3f}**.",
            "",
            "This is a preliminary smoke-scale policy evaluation, not evidence of full-policy "
            "hardware robustness. In particular, the minimum hardware action margin was only "
            f"**{summary['minimum_hardware_action_margin']:.3f}**. A larger fixed state set and "
            "repeated runs are needed for a robustness claim.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    result = json.loads(args.input.read_text(encoding="utf-8"))
    summary = summarize(result)
    args.json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.markdown_output.write_text(markdown(summary, result), encoding="utf-8")
    print(f"Action agreement: {summary['action_agreement_count']}/{summary['states_evaluated']}")
    print(f"Mean absolute Q-value error: {summary['mean_absolute_q_value_error']:.6f}")
    print(f"Minimum hardware action margin: {summary['minimum_hardware_action_margin']:.6f}")
    print(f"JSON summary saved to: {args.json_output}")
    print(f"Markdown summary saved to: {args.markdown_output}")


if __name__ == "__main__":
    main()
