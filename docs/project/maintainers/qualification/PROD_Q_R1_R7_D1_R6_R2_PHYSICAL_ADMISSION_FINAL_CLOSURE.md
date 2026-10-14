# PROD-Q-R1-R7-D1-R6-R2 — Physical Admission Final Closure (evidence)

**Status:** READY FOR AUDIT (independent SHA review required)

## Corrective summary

- Removed portable `ActivationDomainCoordinatorOperationAuthority`; coordinator-synchronized depth scope on adapter for unit admission open.
- Removed public `admit_unit_execution` identifier-only path; sanctioned physical accounting uses `reserve_unit_execution` after coordinator-opened traffic admission.
- Added `test_prod_q_r1_r7_d1_r6_r2_physical_admission_final_closure.py` for bypass, SQLite ordering, busy retry, and lifecycle negatives.

## Verification commands

```text
uv run pytest -p no:xdist tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r2_physical_admission_final_closure.py \
  tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r1_activation_domain_coordinator_corrective.py \
  tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_activation_domain_coordinator.py \
  tests/unit/agent_distribution/test_prod_q_r1_r7_d1_venv_bundle_colocated_deployment_adapter.py \
  tests/unit/applications/test_prod_q_r1_r7_d1_r2_serving_admission_authority.py \
  tests/unit/applications/test_prod_q_r1_r7_d1_r4_serving_admission_authority_closure.py \
  tests/unit/applications/test_prod_q_r1_r7_d1b_serving_runtime_handoff.py \
  tests/unit/agent_distribution/test_sqlite_agent_distribution_bounded_store.py

uv run pyright intergrax/agent_distribution/activation_domain_coordinator.py \
  intergrax/agent_distribution/venv_bundle_colocated_runtime_deployment_adapter.py \
  intergrax/applications/_shared/serving_runtime_handoff.py
```
