# TOOL-LIFE-X L3-R8 — Durable revocation & shared-store evidence

| Field | Value |
| --- | --- |
| START_HEAD | `4c4c14472d780db7358de6c109c0a56528758efa` |
| Scope | SQLite `DURABLE_SINGLE_HOST` revocation ledger + external-operation store; R8 adversarial matrix |
| Multi-host shared durability | **Not proven** — no Redis/PostgreSQL (BLOCKED without operator infra) |
| Cross-store distributed TX | **Not claimed** — revocation + external-op colocated in one SQLite file; no partition L3 |
| Parent TOOL-LIFE-X | **Not CLOSED** |

## Authority graph

- **Execution ownership binding:** `ExternalOperationStateStore.bind_worker_execution_ownership` (RUNNING record).
- **Revocation authority:** `WorkerExecutionOwnershipRevocationPort` — durable rows in SQLite revocation table.
- **Certification:** `WorkerExecutionCertifiedOwnerLossCertifier` → `stage_certified_owner_loss_evidence` (CAS#1 + revocation verify) → `apply_certified_owner_loss` (CAS#2).
- **Live worker fence:** revocation proof required before staging; `WorkerInstance.revision`/lifecycle not authoritative.

## Change inventory

| Path | Change |
| --- | --- |
| `intergrax/runtime/external_operations/external_operation_state_store.py` | Shared `apply_external_operation_state_update` + certification semantics mixin |
| `intergrax/runtime/external_operations/sqlite_tool_life_durable_stores.py` | SQLite stores + `SqliteToolLifeDurableStoresBundle` |
| `tests/unit/runtime/external_operations/test_tool_life_x_l3_r8_sqlite_durable_revocation_shared_store.py` | R8 qualification |

## L1 / L2 / L3

| Gate | Command | Result |
| --- | --- | --- |
| L1 | R8 file only | **12 passed** |
| L2 | R8 + R6 + R7 + W4-C + Batch C (single invocation) | **79 passed** |
| L3 (scoped) | Separate SQLite connection / store reopen on same WAL file | **PASS** in R8 tests |
| L3 (BLOCKED) | Docker PostgreSQL, multi-host Redis, network partition E2E | **BLOCKED** — not authorized |

Log: `.tmp/session/TOOL-LIFE-X-L3-R8/l2-pytest.log`.

FINAL_COMMIT: set at publication.

## TL1–TL15 (parent §3.0.5 exit criteria)

| TL | Semantic owner | Classification | Notes |
| --- | --- | --- | --- |
| TL1 | Runtime external ops | ALREADY PROVEN | Bounded wait / policy |
| TL2 | Deadline hierarchy | VALIDATION REQUIRED | Partial waves |
| TL3 | Cancellation propagation | VALIDATION REQUIRED | MCP/HTTP gaps |
| TL4 | Retry budget | ALREADY PROVEN | Batch C |
| TL5 | Unknown outcome | ALREADY PROVEN | W4-C |
| TL6 | Side-effect retry safety | VALIDATION REQUIRED | Idempotency audit open |
| TL7 | Stalled/abandoned | ALREADY PROVEN | Batch C |
| TL8 | Worker crash recovery | VALIDATION REQUIRED | Scoped PASS — durable revoke + CAS UNKNOWN |
| TL9 | Late results | ALREADY PROVEN | W4-C |
| TL10 | SQL/DB limits | OPEN | Real infra BLOCKED |
| TL11 | MCP lifecycle | VALIDATION REQUIRED | L3-R1/R2 evidence |
| TL12 | HTTP/provider lifecycle | VALIDATION REQUIRED | Uncertainty preserved |
| TL13 | Real infrastructure failures | BLOCKED | Docker/DB not authorized |
| TL14 | No lifecycle bypass | VALIDATION REQUIRED | Ongoing |
| TL15 | Unresolved blockers = 0 | BLOCKED | Parent not CLOSED |

## FRZ / ECAP (scoped)

| ID | Result |
| --- | --- |
| FRZ-BND-03 | PASS — sanctioned store CAS linearization |
| FRZ-TEN | PASS — tenant-scoped keys |
| FRZ-REC | PASS — fail-closed without revocation / issuer |
| FRZ-REL-03/04/05 | PASS — CAS + fencing in L2 |
| FRZ-SEC / FRZ-GOV | PASS — composition-only revocation |

Global FRZ PASS / parent CLOSED **not** promoted.

## Remaining blockers

- Production composition wiring of `SqliteToolLifeDurableStoresBundle` (or approved multi-host backend).
- `SHARED_MULTI_HOST` qualification (Redis/PostgreSQL) — OPERATOR DECISION REQUIRED.
- Cross-store partition when revocation and external-op use separate physical backends.
- TL10–TL13 real infrastructure (Docker) — BLOCKED.
- MCP physical cancellation, full HTTP cross-provider certification.
