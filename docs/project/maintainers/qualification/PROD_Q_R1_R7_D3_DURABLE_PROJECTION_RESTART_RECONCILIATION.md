# PROD-Q-R1-R7-D3 — Durable projection inputs & post-restart reconciliation

| Field | Value |
|---|---|
| **Status** | **READY FOR AUDIT** (implementation evidence — does not close PROD-Q-R1-R7, PROD-Q, or FRZ-PRD global PASS) |
| **Parent** | PROD-Q-R1-R7 |
| **Successor** | PROD-Q-R1-R7-D4 |
| **START_HEAD** | `b631c18cf270bde29264d5e83d0ebb631173f0dc` |
| **DEC anchors** | DEC-04 (durable AD + descriptor authority), DEC-08 (physical attestation before serving reliance) |

## Design summary

- **Durable projection inputs:** `DescriptorBackedRegistryProjectionInputStore` implements `RegistryProjectionInputStore` by reconstructing `RegistryProjectionInputBundle` from `RuntimeRegistryProjectionDescriptor` (SQLite) plus `RegistryProjectionAuthorityResolver` (revision, roster, lock, materialization). Session `register()` retains live resolver instances; `get()` rebuilds factory resolver via `assemble_production_registry_projection_input_bundle` — no callable pickle.
- **Production wiring:** `wire_production_lifecycle_services` and `wire_durable_reference_production_lifecycle_services` use the descriptor-backed store.
- **Restart reconciliation:** `ActivationService.reconcile_traffic_serving_physical_unit_after_process_restart` attests traffic-serving deployment instances (READY/SERVING). `reconcile_production_serving_environment_on_process_open` composes attestation, projection input reconstruction, rehydration, and coordinator `prepare_projection`. `open_production_process_lifecycle(..., serving_reconciliation_targets=...)` fails closed on reconciliation errors before committing the process slot.

## Evidence commands

```text
uv run pytest -p no:xdist tests/unit/applications/test_prod_q_r1_r7_d3_production_restart_reconciliation.py tests/unit/applications/test_prod_q_r1_r7_d2_production_process_lifecycle.py tests/unit/applications/test_registry_projection_descriptor_corruption.py tests/integration/agent_distribution/test_enterprise_agent_lifecycle_durable_e2e.py tests/integration/agent_distribution/test_enterprise_projection_rehydration_e2e.py
uv run pyright intergrax/applications/_shared/registry_projection.py intergrax/applications/_shared/production_lifecycle.py intergrax/applications/_shared/production_process_lifecycle.py intergrax/applications/_shared/production_process_serving_reconciliation.py intergrax/agent_distribution/activation.py
```

## D2 alignment note

Operator disposition: D2 remains **ACCEPTED / READY FOR AUDIT** within A1; this stage does not reopen D2 implementation.
