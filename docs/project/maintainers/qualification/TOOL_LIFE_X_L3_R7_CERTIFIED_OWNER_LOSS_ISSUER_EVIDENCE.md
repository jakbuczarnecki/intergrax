# TOOL-LIFE-X L3-R7 — Certified owner-loss issuer evidence (R7 remediation)

| Field | Value |
| --- | --- |
| START_HEAD | `0c67624f9e7b5da53e978cf7b9b72fa76837ca3c` |
| Scope | R7-F01–F05 corrective batch — authoritative revocation port, issuer/store trust boundary |
| Shared durable backend | **Not proven** — InMemory reference only (L3 OPEN) |
| Parent TOOL-LIFE-X | **Not CLOSED** |

## R7 root causes and dispositions

| ID | Root cause | Disposition |
| --- | --- | --- |
| R7-F01 | `WorkerInstance.revision` treated as ownership epoch | **FIXED** — certification requires `WorkerExecutionOwnershipRevocationPort` proof; revision bumps irrelevant |
| R7-F02 | Lifecycle labels (e.g. RECOVERING, STOPPED) implied revocation | **FIXED** — lifecycle not consulted for loss |
| R7-F03 | Missing worker synthesized STOPPED/rev zero | **FIXED** — missing worker alone denies; revocation record may suffice without worker row |
| R7-F04 | Store `stage_*` accepted worker issuer evidence without revocation verify | **FIXED** — `verify_worker_execution_owner_loss_staging` on store CAS#1 for canonical issuer |
| R7-F05 | Cross-store assumed via revision/lifecycle | **MITIGATED** — linearization: revocation ledger + external-op store staging; **no** single TX across backends (L3 OPEN for partition) |

## Authority graph

- **Revocation authority:** `WorkerExecutionOwnershipRevocationPort` (composition-bound).
- **Commit RUNNING→UNKNOWN:** `ExternalOperationStateStore.apply_certified_owner_loss` (W4-C CAS#2).
- **Stage evidence:** `WorkerExecutionCertifiedOwnerLossCertifier` → store `stage_certified_owner_loss_evidence` (CAS#1 + store-native revocation verify).
- **Issuer identity:** `tool-life-x.worker-execution-ownership.v1` via `CompositionCertifiedOwnerLossIssuerBinding`.
- **Digest:** integrity only; staging requires matching authoritative `liveness_proof_ref`.

## Change inventory

| Path | Change |
| --- | --- |
| `intergrax/contracts/worker_execution_ownership_revocation.py` | Revocation contract + proof ref |
| `intergrax/runtime/external_operations/worker_execution_ownership_revocation.py` | In-memory revocation store |
| `intergrax/runtime/external_operations/certified_owner_loss_composition.py` | Revocation port + store staging guard |
| `intergrax/runtime/external_operations/worker_execution_certified_owner_loss.py` | Certifier uses revocation only |
| `intergrax/runtime/external_operations/external_operation_state_store.py` | Worker-issuer staging verify |
| `tests/unit/runtime/external_operations/test_tool_life_x_l3_r7_certified_owner_loss_issuer.py` | R7 adversarial matrix (scoped) |

## L1 / L2 validation

| Gate | Command | Result |
| --- | --- | --- |
| L1 | `uv run pytest tests/unit/runtime/external_operations/test_tool_life_x_l3_r6_tenant_safe_external_operation_foundation.py tests/unit/runtime/external_operations/test_tool_life_x_l3_r7_certified_owner_loss_issuer.py -q` | **41 passed** |
| L2 | above + W4-C + Batch C (single invocation) | **68 passed** |

Logs: `.tmp/session/TOOL-LIFE-X-L3-R7/l2-pytest.log`.

FINAL_COMMIT: set at publication.

## TL1–TL15 (parent-scoped)

| TL | Classification | Notes |
| --- | --- | --- |
| TL1–TL6 | ALREADY PROVEN | L3-R6 foundation |
| TL7 | VALIDATION REQUIRED → scoped PASS | Composition + revocation + issuer |
| TL8 | VALIDATION REQUIRED → scoped PASS | Positive path only with authoritative revocation |
| TL9 | ALREADY PROVEN | L2 W4-C/Batch C regressions |
| TL10–TL11 | OPEN | Shared-store / issuer liveness E2E |
| TL12 | VALIDATION REQUIRED | Production must wire revocation port + composition |
| TL13–TL15 | OPEN | DIST-SCALE-X / parent audit |

## FRZ / ECAP (scoped)

| ID | Result |
| --- | --- |
| FRZ-BND-03 | PASS — single store linearization for UNKNOWN |
| FRZ-TEN | PASS — tenant-scoped CAS + revocation scoped by tenant/op |
| FRZ-REC | PASS — fail-closed without revocation |
| FRZ-REL-03/04/05 | PASS — L2 CAS + negatives |
| FRZ-SEC / FRZ-GOV | PASS — store blocks direct forged staging |

Global FRZ / parent CLOSED **not** promoted.

## Remaining blockers

- Durable production `WorkerExecutionOwnershipRevocationPort` (shared with worker shutdown fence).
- Cross-store partition qualification (L3).
- Production composition: `configure_worker_execution_certified_owner_loss(revocation_port=…)`.
