# PROD-Q-R1-R7-D1-R6-R8 — Canonical store transaction safety closure

**Status:** READY FOR AUDIT (corrective child; parent D1-R6 remains BLOCKED pending independent SHA audit).

## Corrective findings

| ID | Resolution | Evidence |
|----|------------|----------|
| R8-B01 | `mutation_transaction` acquires lock inside `try`; `_connect` failure cannot skip `finally` release | `sqlite_stores.py`, R8 `test_mutation_connect_failure_releases_lock` |
| R8-B02 | Best-effort rollback/close with primary + cleanup chaining | `_sqlite_connection_cleanup_errors`, `_raise_sqlite_store_mutation_failure`, R8 fault matrix |
| R8-B03 | `_ActivationDomainSynchronization` typed ownership (no `getattr` probing) | `sqlite_stores.py`, R8 nested/concurrent tests |
| R8-B04 | S3-B uses `FakeInMemoryRuntimeDeploymentAdapter.reserve_unit_execution` + ordering events | R7 `test_s3_b_*` (updated) |
| R8-B05 | D1/D1b stale/stop + fresh adapter gate closed | R7 restart tests + R8 `test_d1b_reconstructed_*` |

## Commands (session)

```text
uv run pytest -p no:xdist tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r8_transaction_safety_closure.py tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r*.py tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r7_*.py tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_activation_domain_coordinator.py tests/unit/agent_distribution/test_agent_distribution_rollback.py tests/unit/applications/test_prod_q_r1_r7_d1b_serving_runtime_handoff.py tests/unit/agent_distribution/test_prod_q_r1_r7_d1_venv_bundle_colocated_deployment_adapter.py tests/unit/agent_distribution/test_prod_q_r1_r7_d0_*.py
uv run pyright intergrax/agent_distribution/sqlite_stores.py intergrax/agent_distribution/deployment.py intergrax/agent_distribution/activation_domain_coordinator.py
```
