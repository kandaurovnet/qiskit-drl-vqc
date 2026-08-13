# IBM artifacts

This directory keeps IBM integration evidence separate from the comparable
CartPole run outputs in `results/`.

## Classification

| Directory | Importance | Contents |
|---|---|---|
| `core/` | Primary evidence | Completed IBM hardware evaluation and report-ready summaries |
| `validation/` | Required engineering support | Smoke test, exact-policy preview, and submission manifest used to validate the IBM path |
| `archive/` | Historical only | Incomplete or superseded jobs that must not be used for performance comparison |

The trained policy remains at `results/quantum_policy.pt` because it is the
model shared by the Quantum Torch evaluation and IBM inference.

Future end-to-end IBM CartPole runs should use the normal comparable result
location and naming convention, for example `results/ibm_cartpole_run.json`.

## Current artifacts

- `core/ibm_policy_evaluation_d9ulamt35hes73fk3400.json`: completed two-state
  hardware evaluation on `ibm_marrakesh`.
- `core/ibm_policy_hardware_summary.{json,md}`: compact metrics and report text.
- `validation/ibm_smoke_test_d9ujrck98n5s7392v2q0.json`: Runtime connectivity
  and result-retrieval smoke test.
- `validation/ibm_policy_preview.json`: trained Torch VQC states validated with
  Qiskit exact simulation.
- `validation/ibm_policy_submission_preview.json`: checkpoint, transpilation,
  and parameter-binding manifest created before QPU submission.
- `archive/ibm_policy_evaluation_d9uks0l35hes73fk2fv0.json`: partial
  element-wise broadcasting result; retained for traceability only.
