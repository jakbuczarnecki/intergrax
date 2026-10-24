# TOOL-LIFE-X L3-R7 — Certified owner-loss issuer evidence

| Field | Value |
| --- | --- |
| START_HEAD | `d3ef64897e0ab3ba32720d5505f42b6bcd2d59c6` |
| Scope | Sanctioned worker-execution issuer, composition binding, digest validation, recovery apply, reconciliation admission |
| Shared durable backend | **Not proven** — InMemory reference only (L3 OPEN) |
| Parent TOOL-LIFE-X | **Not CLOSED** |

## Authority graph

- **Commit RUNNING→UNKNOWN:** `ExternalOperationStateStore.apply_certified_owner_loss` (W4-C linearization).
- **Stage evidence:** `WorkerExecutionCertifiedOwnerLossCertifier` → store `stage_certified_owner_loss_evidence`.
- **Loss determination:** `WorkerInstanceRepository` revision/lifecycle vs durable `worker_instance_id` + `worker_ownership_epoch` on `ExternalOperationState`.
- **Issuer identity:** `tool-life-x.worker-execution-ownership.v1` via `CompositionCertifiedOwnerLossIssuerBinding`.

## Change inventory

| Path | Change |
| --- | --- |
| `intergrax/contracts/certified_owner_loss_evidence.py` | Digest helper, canonical issuer id, direct-invocation error |
| `intergrax/contracts/external_operation_cancellation.py` | Worker binding fields on durable state |
| `intergrax/runtime/external_operations/external_operation_state_store.py` | Worker bind CAS, digest checks, worker fields |
| `intergrax/runtime/external_operations/worker_execution_certified_owner_loss.py` | Issuer + certifier + composition configure |
| `intergrax/runtime/external_operations/certified_owner_loss_composition.py` | Composition registry |
| `intergrax/runtime/external_operations/external_operation_ownership.py` | Recovery apply via `evidence_id` |
| `intergrax/runtime/external_operations/recovery_external_operation_gate.py` | UNKNOWN reconciliation admission |
| `tests/unit/runtime/external_operations/test_tool_life_x_l3_r7_certified_owner_loss_issuer.py` | L3-R7 adversarial/positive matrix (scoped) |

## L1 / L2 validation

| Gate | Command | Result |
| --- | --- | --- |
| L1 | `uv run pytest tests/unit/runtime/external_operations/test_tool_life_x_l3_r6_tenant_safe_external_operation_foundation.py tests/unit/runtime/external_operations/test_tool_life_x_l3_r7_certified_owner_loss_issuer.py -q` | **33 passed** |
| L2 | above + W4-C + Batch C recovery tests (single invocation) | **60 passed** |

Logs: `.tmp/session/TOOL-LIFE-X-L3-R7/l1-pytest.log`, `l2-pytest.log`.

FINAL_COMMIT: set at publication.

## TL1–TL15 (parent-scoped)

| TL | Classification | Notes |
| --- | --- | --- |
| TL1–TL6 | ALREADY PROVEN | L3-R6 foundation @ START_HEAD |
| TL7 | VALIDATION REQUIRED → scoped PASS | Composition + issuer wired |
| TL8 | VALIDATION REQUIRED → scoped PASS | Positive path via certifier + apply |
| TL9 | ALREADY PROVEN | L2 W4-C/Batch C regressions |
| TL10–TL11 | OPEN | Issuer liveness E2E / shared store |
| TL12 | VALIDATION REQUIRED | Requires operator production composition call |
| TL13–TL15 | OPEN / PENDING | DIST-SCALE-X / audit |

## FRZ / ECAP (scoped)

| ID | Result |
| --- | --- |
| FRZ-BND-03 | PASS — single store linearization |
| FRZ-TEN | PASS — tenant-scoped CAS preserved |
| FRZ-REC | PASS — fail-closed default; certified path only |
| FRZ-REL-03/04/05 | PASS — L1/L2 CAS + issuer negatives |
| FRZ-SEC / FRZ-GOV | PASS — no trust-by-string; direct issuer blocked |

Global FRZ / parent CLOSED **not** promoted.

## Remaining blockers

- Production composition root must call `configure_worker_execution_certified_owner_loss`.
- Shared-store physical qualification (L3).
- Full §11.2 adversarial matrix expansion in dedicated follow-up tests if auditor requires row-per-case files.
