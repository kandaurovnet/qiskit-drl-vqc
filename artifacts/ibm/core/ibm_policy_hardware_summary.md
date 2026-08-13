# Preliminary IBM hardware policy evaluation

- Backend: `ibm_marrakesh`
- Runtime job: `d9ulamt35hes73fk3400`
- Target precision: `0.1`
- Trained VQC parameters: `46`
- ISA depth / two-qubit gates: `93` / `34`

| State | Exact Q-values | Hardware Q-values | Exact action | Hardware action | Agreement | Hardware margin |
|---:|---:|---:|---:|---:|:---:|---:|
| 0 | (101.719, 89.470) | (92.346, 92.321) | 0 | 0 | yes | 0.025 |
| 1 | (87.361, 101.617) | (75.556, 90.123) | 1 | 1 | yes | 14.567 |

Action agreement was **2/2 (100%)**. Mean absolute Q-value error was **8.881** and maximum absolute error was **11.805**.

This is a preliminary smoke-scale policy evaluation, not evidence of full-policy hardware robustness. In particular, the minimum hardware action margin was only **0.025**. A larger fixed state set and repeated runs are needed for a robustness claim.
