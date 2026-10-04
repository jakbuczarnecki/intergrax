# STATE-X-R3 — Auxiliary Durable State Consistency Closure

**Parent:** STATE-X — Persistence, State & Recovery Certification

**START_HEAD (R3-R2 pre-audit):** `b04327dd5dac4d47059a995e9aea61266a360c2e`

**Status:** R3-R2 READY FOR AUDIT (pending independent GitHub SHA audit)

---

## Family classification (current-head)

| Family | R3 disposition |
|---|---|
| SX-F09 Execution Budget Durable State | **ACCEPTED via R3-R1** |
| SX-F10 Idempotency State | **R3-R2 under qualification** |
| SX-F11 Compensation Queue State | **R3-R2 under qualification** |
| SX-F12 Human Decision / HITL Persistence | NOT ENTERED |
| SX-F13 Scheduler Durable State | NOT ENTERED |
| SX-F14 Agent Checkpoint State | NOT ENTERED |

Parent R3 closure is **not** claimed.

---

## R3-R1 — Durable Budget CAS & Stale-Writer Closure

**Production delta:** `intergrax/runtime/execution/budget/persistence.py` only.

### Root cause (pre-fix)

`DurableExecutionBudgetLedger._persist_current_state` exported one snapshot before a `while True` CAS loop. After conflict, `_reload_from_storage()` refreshed in-memory state but retried the **same stale snapshot**, allowing a loser to overwrite the winner.

### Current-head semantics

| Case | Behavior |
|---|---|
| Mutation CAS success | Durable snapshot updated; `_last_known_raw` advanced |
| Mutation CAS conflict | Reload canonical durable state; raise `StaleRunBudgetSnapshotWriteError` |
| Initial create CAS loss | Load winner once; open ledger from durable truth (no recursion) |
| Redelivery settlement CAS loss | Load winner once; if already settled for requested `AttemptId`, use it; else stale error |

No automatic merge of conflicting budget snapshots. No unbounded persistence retry.

### Ownership (SX-F09)

| Role | Owner |
|---|---|
| Semantic budget consumption / reservation | `ExecutionBudgetLedger` |
| Durable per-Run snapshot | `RunBudgetPersistence` |
| Composition / restore | `DurableRunBudgetLedgerFactory` |

Configured `RunBudget` ≠ persisted `RunBudgetLedgerSnapshot` (FRZ-STA-06 preserved).

### Evidence matrix

Mechanical proof: `tests/qualification/state_x/test_state_x_r3_auxiliary_durable_state.py` (R3-R1-Q01..Q15).

| ID | Topic |
|---|---|
| R3-R1-Q01 | Ownership symbols |
| R3-R1-Q02 | No `while True` in `_persist_current_state` |
| R3-R1-Q03 | No recursive `create_ledger` |
| R3-R1-Q04 | Single-writer CAS success (KV + Document) |
| R3-R1-Q05 | Exact stale writer adversarial (KV + DocumentStore) |
| R3-R1-Q06 | Stale `grant_child_budget` does not escape |
| R3-R1-Q07 | Stale consume cannot overwrite winner |
| R3-R1-Q08 | Initial create race |
| R3-R1-Q09 | Redelivery settlement failed CAS + winner / stale branches |
| R3-R1-Q10 | Tenant A/B isolation (KV + Document) |
| R3-R1-Q11 | Different runs independent |
| R3-R1-Q12 | Corrupt durable bytes → `RunBudgetPersistenceError` |
| R3-R1-Q13 | Redelivery preserves consumption |
| R3-R1-Q15 | No authority mint in persistence module |

Provider conformance for canonical `RunBudgetPersistence` implementations is proven by parametrized **Q04**, **Q05**, and **Q10** (not function-existence checks).

---

## R3-R1-R1 — provider + redelivery race proof closure

**START_HEAD:** `47ec47831118d9f3b127d78b61587fb9202c4409`

**Production delta:** none — `intergrax/runtime/execution/budget/persistence.py` unchanged at qualification HEAD.

| Gap | Closure |
|---|---|
| GAP-1 (Q09 did not force redelivery CAS conflict) | `_RedeliveryRacePersistence` test wrapper; Q09 executes failed `compare_and_swap_snapshot` on redelivery branch, post-conflict `load_snapshot`, same-attempt winner accepted with full budget state, different-attempt winner → `StaleRunBudgetSnapshotWriteError` |
| GAP-2 (stale writer KV-only) | Q05 parametrized for `KvRunBudgetPersistence` and `DocumentStoreRunBudgetPersistence` via `DurableExecutionBudgetLedger` |

**Status:** R3-R1-R1 READY FOR AUDIT · R3-R1 READY FOR AUDIT (not CLOSED — independent GitHub SHA audit required).

### FRZ evidence (no PASS promotion)

Supporting: FRZ-STA-01, FRZ-STA-03, FRZ-STA-04, FRZ-STA-05, FRZ-STA-06, FRZ-REC-01, FRZ-REC-04, FRZ-REC-06, FRZ-REC-09, FRZ-REC-10, FRZ-TEN-04, FRZ-TEN-08.

---

## R3-R2 — Idempotency / Compensation External-Effect Recovery Closure

**START_HEAD:** `b04327dd5dac4d47059a995e9aea61266a360c2e`

### Production delta

| Path | Change |
|---|---|
| `intergrax/applications/_shared/compensation_side_effect_wiring.py` | Mandatory `authority: ParentExecutionAuthority`; removed `unrestricted_root()` fallback |
| `intergrax/runtime/tools/sqlite_idempotency_store.py` | `_row_to_claim` uses caller `tenant_id`/`key` (fixes active-claim path `IndexError` found by R3-R2-Q05/Q07) |

### SX-F10 closed-world inventory

Contract owner: `IdempotencyStore`. Providers: `InMemoryIdempotencyStore` (process-local), `SQLiteIdempotencyStore` (durable single-host), `RedisIdempotencyStore` (shared multi-host).

### SX-F11 ownership

Contract owner: `CompensationQueueStore`. Providers: `InMemoryCompensationQueueStore`, `SQLiteCompensationQueueStore`. Queue claim is processing lease only; physical effect via `CompensationSideEffectExecutionPort` → `ExecutionRuntime`.

### Evidence

Mechanical proof: `tests/qualification/state_x/test_state_x_r3_auxiliary_durable_state.py` (imports R3-R2-Q01..Q29 from `_r3_r2_qualification_tests.py`).

Redis executable evidence: local `redis:7-alpine` on `localhost:6379` for Q25–Q29; full multi-host Docker proof per `tests/system/tools_side_effect_safety/README.md` (operator/CI).

### FRZ evidence (no PASS promotion)

Primary: FRZ-STA-01..05, FRZ-REC-01/04/05/06/07/09/10, FRZ-TEN-04/08. Supporting: FRZ-CTR/TYP/GOV/EXE families as cited in R3-R2 task.

**Status:** R3-R2 READY FOR AUDIT (not CLOSED).

---

## Roadmap note

Next child after R3-R2 audit: **STATE-X-R3-R3**. STATE-X-R4 and TRACE-X remain NOT ENTERED.
