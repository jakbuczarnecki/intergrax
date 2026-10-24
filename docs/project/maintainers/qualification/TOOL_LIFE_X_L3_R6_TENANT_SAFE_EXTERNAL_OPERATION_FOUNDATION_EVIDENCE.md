# TOOL-LIFE-X L3-R6 — Tenant-safe external operation foundation evidence

| Field | Value |
| --- | --- |
| START_HEAD (remediation) | `ce57be26c63d7595272d47a5c94cc8f076ab69fe` |
| Scope | Foundation + R6-F01/F02 corrective batch — tenant-scoped store, certified-loss CAS authority, adversarial matrix |
| Positive RUNNING→UNKNOWN in production | **Not claimed** — `recover_running_operation_after_certified_owner_loss` remains fail-closed; apply path proven only under test issuer monkeypatch |
| Issuer liveness E2E | **OPEN (L3)** |
| Shared durable backend | **Not proven** — InMemory reference only |

## R6-F01 — root cause and correction

**Root cause:** `apply_certified_owner_loss` compared `evidence.epoch.store_revision` to the apply CAS `expected_revision`. After staging, durable revision advances (`N → N+1`) while the ownership epoch remains bound to pre-stage revision `N`, so a valid certificate was rejected or would require rewriting revisions.

**Correction:**

- Stage CAS at revision `N` requires `evidence.epoch.store_revision == N` and owner/tenant/operation alignment.
- Successful stage leaves physical `RUNNING` at revision `N+1` with staged slot intact.
- Apply CAS at revision `N+1` requires `evidence.epoch.store_revision + 1 == expected_revision` (ownership epoch `N`, apply at `N+1`).
- Successful apply transitions to `UNKNOWN` at revision `N+2`.

## R6-F02 — root cause and correction

**Root cause:** `ExternalOperationStateUpdate` exposed certification fields (`staged_certified_owner_loss`, `consumed_certified_loss_evidence_ids`, `ownership_epoch_fence`) to generic `compare_and_set`, allowing bypass of issuer staging and one-time consumption.

**Correction:** Generic `compare_and_set` raises `CertifiedOwnerLossAuthorityBypassError` when certification authority fields are present. Only `stage_certified_owner_loss_evidence` and `apply_certified_owner_loss` use internal `_compare_and_set_locked`. Legitimate provider/cancellation `UNKNOWN` via `mark_physical_terminal` / ownership helpers unchanged.

## Change inventory

| Path | Change |
| --- | --- |
| `intergrax/contracts/certified_owner_loss_evidence.py` | `CertifiedOwnerLossAuthorityBypassError` |
| `intergrax/runtime/external_operations/external_operation_state_store.py` | CAS split, stage/apply revision model, staging validations, occupied slot |
| `tests/unit/runtime/external_operations/test_tool_life_x_l3_r6_tenant_safe_external_operation_foundation.py` | Adversarial matrix (stage/apply, bypass, races, tenant) |

## L1 / L2 validation

| Gate | Command | Result |
| --- | --- | --- |
| L1 | `uv run pytest tests/unit/runtime/external_operations/test_tool_life_x_l3_r6_tenant_safe_external_operation_foundation.py -q` | **28 passed** |
| L2 | `uv run pytest tests/unit/runtime/external_operations/test_tool_life_x_l3_r6_tenant_safe_external_operation_foundation.py tests/unit/runtime/architecture/test_enterprise_scale_resilience_w4_c_distributed_cancellation_qualification.py tests/unit/runtime/nexus/tools/test_tool_life_x_batch_c_stalled_abandoned_recovery.py -q` | **55 passed** |

Session logs: `.tmp/session/l3-r6/l1-pytest.log`, `.tmp/session/l3-r6/l2-pytest.log` (gitignored).

FINAL_COMMIT: set at publication (see operator report).

## Tenant isolation audit

- All store paths require non-empty `tenant_id`; legacy empty tenant rejected (`ExternalOperationLegacyTenantBindingError`).
- Cross-tenant load returns `None`; cross-tenant CAS raises `ExternalOperationNotFoundError`.
- Certification epoch tenant/operation must match durable record at stage and apply.

## FRZ / ECAP (scoped)

| ID | Result |
| --- | --- |
| FRZ-BND-03 | PASS — single durable store; no shadow SM |
| FRZ-TEN | PASS — tenant-scoped keys and CAS |
| FRZ-REC | PASS — Path B deny; no uncertified RUNNING→UNKNOWN via generic CAS |
| FRZ-REL-03/04/05 | PASS — CAS, fencing, adversarial races in L2 |
| FRZ-SEC / FRZ-GOV | PASS — issuer allowlist + sanctioned entry points only |

Global FRZ / R7-Q / PROD-Q **not** promoted.

## TL1–TL15 closure matrix (L3-R6 foundation)

| TL | Owner | Contract / evidence | Classification | Missing | Test | FRZ | Dep |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TL1 | Runtime external ops | `ExternalOperationState` + tenant store | CLOSED foundation | shared backend | L2 | TEN | W4-C |
| TL2 | Contracts | `certified_owner_loss_evidence.py` | CLOSED types | issuer impl | L1 | BND-03 | TL1 |
| TL3 | Store CAS | `compare_and_set` vs cert paths | CLOSED R6-F02 | — | L1 adversarial | REC | TL2 |
| TL4 | Stage | `stage_certified_owner_loss_evidence` | CLOSED R6-F01 | — | L1 | REL | TL3 |
| TL5 | Apply | `apply_certified_owner_loss` | CLOSED R6-F01 | prod recovery API | L1 | REC | TL4 |
| TL6 | Ownership epoch | `ExternalOperationOwnershipEpochV1.store_revision` | CLOSED binding N/N+1 | — | L1 | REL | TL4–5 |
| TL7 | Composition binding | empty default allowlist | CLOSED fail-closed | ratified issuer | L1 negatives | SEC | TL2 |
| TL8 | Path B recovery | `recover_running_operation_after_certified_owner_loss` | CLOSED deny | positive wiring | L1 | REC | parent L3 |
| TL9 | W4-C adjacency | cancellation / terminal | CLOSED preserved | — | L2 W4-C | BND-03 | W4-C |
| TL10 | Issuer liveness | — | **OPEN** | E2E issuer | — | — | TL7 |
| TL11 | Shared durability | — | **OPEN** | Redis/SQLite store | — | — | TL1 |
| TL12 | Positive recovery prod | — | **OPEN** | operator composition | — | REC | TL10 |
| TL13 | DIST-SCALE-X | — | **OPEN** | not authorized | — | — | roadmap |
| TL14 | Independent audit | GitHub SHA | **PENDING** | auditor | — | — | commit |
| TL15 | Parent TOOL-LIFE-X | batch gates | **BLOCKED** | TL10–13 | — | — | TL14 |

## Remaining blockers (parent)

- Sanctioned issuer implementation and composition wiring (no test-only issuer in production).
- Shared-store qualification for external operation state.
- Positive owner-loss recovery enablement in production execution paths.
- TL10–TL13 distributed / issuer-liveness evidence.

## Post-step discovery

Roadmap §3.0.5 and TL7/TL8 ADR §20 unchanged by this batch; remediation stays within L3-R6 foundation authorization.
