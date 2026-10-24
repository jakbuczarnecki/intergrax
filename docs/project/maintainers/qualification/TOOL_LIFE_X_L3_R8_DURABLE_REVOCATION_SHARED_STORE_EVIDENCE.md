# TOOL-LIFE-X L3-R8 — Durable revocation & shared-store evidence

| Field | Value |
| --- | --- |
| START_HEAD | `3034a880db8dc67770f3725000ddf241dc784b8d` |
| Scope | SQLite `DURABLE_SINGLE_HOST` revocation ledger + external-operation store; R8 adversarial matrix |
| Multi-host shared durability | **Not proven** — no Redis/PostgreSQL (BLOCKED without operator infra) |
| Cross-store distributed TX | **Not claimed** — revocation + external-op colocated in one SQLite file; no partition L3 |
| Parent TOOL-LIFE-X | **Not CLOSED** |

## Authority graph

- **Execution ownership binding:** `ExternalOperationStateStore.bind_worker_execution_ownership` (RUNNING record).
- **Revocation authority (store):** `WorkerExecutionOwnershipRevocationPort` — durable rows only after `assert_authoritative_worker_execution_revocation_allowed` + verified `WorkerExecutionOwnershipRevocationAuthorizationV1` inside `BEGIN IMMEDIATE`.
- **Revocation initiator (governance):** `WorkerExecutionOwnershipRevocationSupervisionPort` → `WorkerExecutionOwnershipRevocationAuthorizer` → `WorkerExecutionOwnershipRevocationRecorder` (composition-registered, typed).
- **Fencing:** optimistic read via `bound_worker_execution_epoch_is_revoked`; authoritative fence in SQLite `BEGIN IMMEDIATE` CAS via `cas_update_blocked_by_bound_worker_revocation`.
- **Certification:** `WorkerExecutionCertifiedOwnerLossCertifier` → `stage_certified_owner_loss_evidence` (CAS#1 + revocation verify) → `apply_certified_owner_loss` (CAS#2).
- **Live worker fence:** revocation proof required before staging; `WorkerInstance.revision`/lifecycle not authoritative.
- **Production default:** no supervision port wired → positive revocation **fail-closed** (`WorkerExecutionOwnershipRevocationSupervisionNotConfiguredError`).

## Change inventory

| Path | Change |
| --- | --- |
| `intergrax/contracts/worker_execution_ownership_revocation.py` | Typed authorization + supervision port contracts |
| `intergrax/runtime/external_operations/worker_execution_ownership_revocation.py` | Authorizer, recorder, colocated port protocol, authorization verify |
| `intergrax/runtime/external_operations/certified_owner_loss_composition.py` | Typed supervision/authorizer/recorder composition slots |
| `intergrax/runtime/external_operations/worker_execution_certified_owner_loss.py` | Explicit store resolution; supervision wiring |
| `intergrax/runtime/external_operations/sqlite_tool_life_durable_stores.py` | Authorization enforced in revoke TX; colocated store property |
| `tests/unit/runtime/external_operations/sqlite_tool_life_l3_r8_cross_process_proof.py` | Deterministic 2-process harness (qualification area) |
| `tests/unit/runtime/external_operations/tool_life_x_qualification_revocation_supervision.py` | L3 qualification supervision double |
| `tests/unit/runtime/external_operations/test_tool_life_x_l3_r8_sqlite_durable_revocation_shared_store.py` | R8 + F07/F08 adversarial matrix |

## L1 / L2 / L3

| Gate | Command | Result |
| --- | --- | --- |
| L1 | R7 + R8 files | **33 passed** |
| L2 | R6 + R7 + R8 + Batch C (single invocation) | **73 passed** |
| L3 (scoped) | WAL reopen + threaded revoke + **2-process** barrier races (recorder + persisted state) | **PASS** in R8 tests |
| L3 (BLOCKED) | Docker PostgreSQL, multi-host Redis, network partition E2E | **BLOCKED** — not authorized |

Log: `.tmp/session/TOOL-LIFE-X-L3-R8/l1-pytest.log`, `l2-pytest.log`.

## R8-F01–F08 disposition (corrective batch)

| Finding | Disposition |
| --- | --- |
| R8-F01 Unauthorized revocation writer | **RESOLVED** — revoke requires authoritative RUNNING worker binding (tenant/op/worker/epoch); idempotent row reuse |
| R8-F02 Revocation must fence | **RESOLVED** — revocation proof gates staging; CAS/terminal reject revoked bound epoch before apply |
| R8-F03 SQLite transaction boundary | **RESOLVED** — revoke: single `BEGIN IMMEDIATE` read state + insert; stage CAS: in-TX revocation proof re-read |
| R8-F04 Durability/concurrency | **PARTIAL PASS** — WAL cross-connection + threaded concurrent revoke in L3 scope; multi-host **BLOCKED** |
| R8-F05 Pre-TX-only revocation check | **RESOLVED** — in-TX revocation row re-read on every SQLite CAS |
| R8-F06 Terminal CAS vs revoke race | **RESOLVED** — linearization at SQLite `BEGIN IMMEDIATE` (terminal-first or revoke-first adversarial) |
| R8-F07 Revocation caller authority | **PARTIAL PASS** — typed supervision → authorizer → recorder → authorization verified at store TX; production supervision owner **not wired** (fail-closed default) |
| R8-F08 Cross-process proof | **PASS (scoped)** — deterministic barrier ordering, exit codes, persisted revocation rows |

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
| FRZ-REC | PASS — fail-closed without revocation / issuer / supervision |
| FRZ-REL-03/04/05 | PASS — CAS + fencing in L2 |
| FRZ-SEC / FRZ-GOV | PASS — composition-only revocation path |

Global FRZ PASS / parent CLOSED **not** promoted.

## Remaining blockers

- Production `WorkerExecutionOwnershipRevocationSupervisionPort` implementation (process supervisor / orchestrator — ADR §4.6).
- Production composition wiring of `SqliteToolLifeDurableStoresBundle` (or approved multi-host backend).
- `SHARED_MULTI_HOST` qualification (Redis/PostgreSQL) — OPERATOR DECISION REQUIRED.
- Cross-store partition when revocation and external-op use separate physical backends.
- TL10–TL13 real infrastructure (Docker) — BLOCKED.
- MCP physical cancellation, full HTTP cross-provider certification.
