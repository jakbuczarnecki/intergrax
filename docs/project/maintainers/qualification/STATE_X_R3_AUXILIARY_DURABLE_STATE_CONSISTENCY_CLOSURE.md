# STATE-X-R3 — Auxiliary Durable State Consistency Closure

**Parent:** STATE-X — Persistence, State & Recovery Certification

**START_HEAD (R3-R2 pre-audit):** `b04327dd5dac4d47059a995e9aea61266a360c2e`

**Status:** R3-R2-R1 READY FOR AUDIT · R3-R2 BLOCKED PENDING INDEPENDENT R3-R2-R1 AUDIT

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

**Status:** R3-R2 BLOCKED PENDING INDEPENDENT R3-R2-R1 AUDIT (independent audit found Q23 / SX-F11 SQLite parity gaps).

---

## R3-R2-R1 — Compensation Production-Path & Durable Provider Qualification Closure

**START_HEAD:** `7f392e442f46ed3aff1c44d4b97d66aaeb703598`

### Root cause (independent R3-R2 audit)

- **BLOCKER A:** Q23 proved dedupe via test-owned `_IdempotentBoundInvoker` instead of canonical `RuntimeToolInvoker` + `IdempotencyPreEffectCoordinator`.
- **BLOCKER B:** SX-F11 qualification lacked provider-parity mechanical evidence (SQLite vs InMemory) for several queue invariants and canonical compensation effect path.

### Changed files (this closure)

| Path | Role |
|---|---|
| `tests/qualification/state_x/_r3_r2_support.py` | Typed queue/idempotency factories; canonical compensation stack builder (`build_declarative_invoker_from_tool_wiring` → `CatalogDeclarativeToolInvoker` → `build_compensation_side_effect_execution`); lab policy/governance test harness |
| `tests/qualification/state_x/_r3_r2_qualification_tests.py` | Q23 canonical RETRYABLE replay; Q30–Q32 crash/parity; parametrized SX-F11 matrix |
| `tests/qualification/state_x/inventory.py` | R3-R2 allowlist bookkeeping for `_r3_r2_qualification_tests.py` |
| `docs/project/maintainers/qualification/STATE_X_R3_AUXILIARY_DURABLE_STATE_CONSISTENCY_CLOSURE.md` | This section |

**Production delta:** NONE (qualification/test-only).

### Canonical production-path graph (after)

```text
CompensationQueueStore (claim)
→ compensation_side_effect_input_from_job
→ CompensationSideEffectExecutionPort (build_compensation_side_effect_execution)
→ ExecutionRuntime
→ BoundCompensationToolInvokeSession
→ CatalogDeclarativeToolInvoker.invoke
→ invoke_catalog_tool_request
→ RuntimeToolInvoker
→ IdempotencyPreEffectCoordinator
→ IdempotencyStore
→ physical tool handler
```

### SX-F11 provider matrix (mechanical)

| Invariant | InMemory | SQLite |
|---|:---:|:---:|
| enqueue dedupe | Q15 | Q15 |
| atomic claim | Q16 | Q16 |
| stale completion/failure reject | Q17 | Q17 |
| tenant isolation | Q18 | Q18 |
| stable idempotency key after RETRYABLE reclaim | Q22 | Q22 |
| expired RUNNING → UNCERTAIN | Q30 | Q30 |
| UNCERTAIN not claimable | Q30 / Q32 | Q30 / Q32 |
| current owner completes | Q31 | Q31 |
| production-path physical dedupe (RETRYABLE replay) | Q23 | Q23 |
| crash window + no duplicate effect (canonical path) | Q32 | Q32 |

### Crash vs RETRYABLE semantics (separate proofs)

- **Crash window (Q32):** effect completes + idempotency COMPLETED; queue `complete_claim` skipped; lease expiry → `UNCERTAIN` → no reclaim; `drain_pending_compensation_jobs` does not re-invoke; handler calls == 1.
- **Explicit RETRYABLE (Q23):** `fail_claim(..., retryable=True)` reclaim; same compensation idempotency key; canonical replay; handler calls == 1.

### Tenant Isolation Audit

- **tenant scope applicable:** YES
- **canonical tenant identity:** `tenant_id`
- **semantic storage identity:** `(tenant_id, idempotency_key)` for `CompensationQueueStore` (InMemory dict key; SQLite `PRIMARY KEY (tenant_id, idempotency_key)`)
- **tenant owner:** `CompensationQueueStore` / `IdempotencyStore` (per-tenant keys)
- **propagation path:** `CompensationJob.tenant_id` → queue persistence key → `CompensationClaim.tenant_id` → `complete_claim` / `fail_claim` mutation scope; idempotency: `CompensationSideEffectInput` → catalog invoke → idempotency store partition
- **state isolation:** Q12 (idempotency), Q18 (compensation queue same-key cross-tenant lifecycle)
- **provider/config isolation:** parametrized InMemory + SQLite factories (`tmp_path`-scoped SQLite DBs)
- **evidence/trace isolation:** declarative policy evaluation uses `state.tenant_id` on catalog dispatch state
- **async/recovery continuity:** Q22 stable key on RETRYABLE; Q30/Q32 UNCERTAIN fail-closed
- **cross-tenant path:** same compensation idempotency key under tenant A and tenant B coexists; tenant-scoped lookup, claim, completion, and failure/retry do not mutate the other tenant’s job (Q18)
- **fail-closed behavior:** UNCERTAIN not claimable; no automatic RUNNING→RETRYABLE
- **adversarial evidence:** Q18 — `tenant A + key X` and `tenant B + key X` (bit-identical key, runtime-asserted); exercised for InMemory and SQLite: coexistence, lookup isolation, claim isolation, `complete_claim` isolation, `fail_claim`/RETRYABLE reclaim and fence isolation
- **result:** PASS — qualification evidence; pending independent exact-SHA audit

### Test evidence

- `tests/qualification/state_x/test_state_x_r3_auxiliary_durable_state.py` (imports R3-R2 Q01–Q32 + Redis Q25–Q29)
- Supporting: `tests/unit/agents/persistence/test_pcm_compensation_coordination.py`, `tests/unit/applications/shared/test_reliability_idempotency_declarative_invoker_wiring.py`, `tests/unit/runtime/execution/test_compensation_side_effect_admission.py`

### FRZ evidence (no PASS promotion)

| FRZ-ID | Evidence |
|---|---|
| FRZ-STA-03 | Q16 atomic claim; Q23/Q32 transactional boundaries via canonical invoker + stores |
| FRZ-STA-04 | Q12; Q18 same-key `(tenant_id, idempotency_key)` queue isolation (InMemory + SQLite) |
| FRZ-STA-05 | Q17 stale fence rejection |
| FRZ-REC-01 | Q30, Q32 crash/recovery |
| FRZ-REC-04 | Q23 replay |
| FRZ-REC-06 | Q32 partial queue completion fail-closed |
| FRZ-REC-07 | Q30/Q32 UNCERTAIN modeling |
| FRZ-TEN-04 | Q18 same-key cross-tenant queue mutation blocked (InMemory + SQLite) |
| FRZ-TEN-08 | Q22, Q23 tenant/key continuity |

Parent FRZ-STA/REC/TEN criteria: revalidated via preserved Q01–Q29 (+ Redis when available).

**Status:** STATE-X-R3-R2-R1-R1 = READY FOR AUDIT · STATE-X-R3-R2-R1 = BLOCKED PENDING INDEPENDENT CHILD AUDIT · STATE-X-R3-R2 = BLOCKED · STATE-X-R3-R3 = NOT ENTERED.

---

## Roadmap note

Next child after R3-R2 audit: **STATE-X-R3-R3**. STATE-X-R4 and TRACE-X remain NOT ENTERED.
