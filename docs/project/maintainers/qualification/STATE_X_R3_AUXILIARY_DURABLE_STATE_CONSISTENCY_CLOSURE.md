# STATE-X-R3 — Auxiliary Durable State Consistency Closure

**Parent:** STATE-X — Persistence, State & Recovery Certification

**Accepted implementation/qualification baseline (R3-R2 chain):** `d5979a531e41f9a45a3a00b0b150765ee34bb0f8`

**Status (current):**

| Task | Result |
|---|---|
| STATE-X-R3-R2-R1-R1 | **CLOSED** — independently accepted @ `d5979a531e41f9a45a3a00b0b150765ee34bb0f8` |
| STATE-X-R3-R2-R1 | **CLOSED** — reconciled through accepted child |
| STATE-X-R3-R2 | **CLOSED** — reconciled |
| STATE-X-R3-R3 | **CLOSED** — independently accepted @ `e05b5b8eb3a60e8f46b5502083440e2e863e7204` (SX-F12) |
| STATE-X-R3-R4-R1 | **CLOSED** — independently accepted @ `0413a3f68f19faeee404b73986748e6fd583dec3` |
| STATE-X-R3-R4 | **CLOSED** — reconciled through accepted R3-R4-R1 (SX-F13) |
| STATE-X-R3-R5 | **BLOCKED** — pending independent R3-R5-R1 audit (SX-F14) |
| STATE-X-R3-R5-R1 | **READY FOR AUDIT** (Cursor) — checkpoint provider provenance |
| STATE-X (parent) | **CURRENT / MANDATORY** — not closed |

**Evidence provenance (R3-R2 chain):**

| Milestone | Exact SHA | Role |
|---|---|---|
| R3-R2 implementation + original qualification | `7f392e442f46ed3aff1c44d4b97d66aaeb703598` | accepted implementation contribution |
| R3-R2-R1 production-path / provider qualification remediation | `5b0e937c1bb4cbc309bf12b94402552deaab5d82` | canonical path + SQLite parity closure |
| R3-R2-R1-R1 same-key cross-tenant isolation | `d5979a531e41f9a45a3a00b0b150765ee34bb0f8` | final independently accepted child; reconciled R3-R2 chain baseline |

---

## Family classification (current-head)

| Family | R3 disposition |
|---|---|
| SX-F09 Execution Budget Durable State | **ACCEPTED via R3-R1** |
| SX-F10 Idempotency State | **ACCEPTED via R3-R2** |
| SX-F11 Compensation Queue State | **ACCEPTED via R3-R2** |
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

**Direct (scoped SX-F10/SX-F11 only):** FRZ-STA-01, FRZ-STA-02, FRZ-STA-03, FRZ-STA-04, FRZ-STA-05, FRZ-REC-01, FRZ-REC-04, FRZ-REC-06, FRZ-REC-07, FRZ-TEN-04, FRZ-TEN-08.

**Supporting only:** FRZ-REC-09 (tenant/key continuity on retry/replay — no restore qualification); FRZ-REC-10 (UNCERTAIN/stale-fence fail-closed — no corrupt-bytes or restore-truth qualification).

**No R3-R2 attribution:** FRZ-REC-05 (fork semantics — not mechanically proven in accepted R3-R2 chain; historical reconciliation once incorrectly mapped Q23/Q32 to fork).

**Other families:** FRZ-CTR/TYP/GOV/EXE as cited in R3-R2 task (supporting).

**Status:** **CLOSED** — independently accepted through remediation chain ending at `d5979a531e41f9a45a3a00b0b150765ee34bb0f8` (historical: independent R3-R2 audit found Q23 / SX-F11 SQLite parity gaps → **R3-R2-R1** / **R3-R2-R1-R1**).

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
- **result:** PASS — independently accepted @ `d5979a531e41f9a45a3a00b0b150765ee34bb0f8` (same-key cross-tenant matrix finalized in **R3-R2-R1-R1**)

### Test evidence

- `tests/qualification/state_x/test_state_x_r3_auxiliary_durable_state.py` (imports R3-R2 Q01–Q32 + Redis Q25–Q29)
- Supporting: `tests/unit/agents/persistence/test_pcm_compensation_coordination.py`, `tests/unit/applications/shared/test_reliability_idempotency_declarative_invoker_wiring.py`, `tests/unit/runtime/execution/test_compensation_side_effect_admission.py`

### FRZ evidence (no PASS promotion)

| FRZ-ID | Classification | Evidence |
|---|---|---|
| FRZ-STA-01 | direct (scoped) | Q01 ownership symbols; `IdempotencyStore` + `CompensationQueueStore` contract owners |
| FRZ-STA-02 | direct (scoped) | SX-F10/SX-F11 closed-world inventory — providers only, no duplicate semantic owners |
| FRZ-STA-03 | direct (scoped) | Q16 atomic claim; idempotency coordinator/store claim lifecycle; SQLite transactional behavior (`R1-SQLITE-ENV-01` remains OPEN) |
| FRZ-STA-04 | direct (scoped) | Q12; Q18 same-key `(tenant_id, idempotency_key)` queue isolation (InMemory + SQLite) |
| FRZ-STA-05 | direct (scoped) | Q17 stale fence rejection |
| FRZ-REC-01 | direct (scoped) | Q30, Q32 crash/recovery (UNCERTAIN fail-closed) |
| FRZ-REC-04 | direct (scoped) | Q23 canonical RETRYABLE replay — second physical effect blocked |
| FRZ-REC-06 | direct (scoped) | Q32 partial queue completion fail-closed; no duplicate re-execution |
| FRZ-REC-07 | direct (scoped) | Q30/Q32 UNCERTAIN modeling; not automatically reclaimable |
| FRZ-REC-09 | supporting only | Q22/Q23 tenant/key continuity on retry/replay — **no** restore qualification |
| FRZ-REC-10 | supporting only | Q17/Q30/Q32 fail-closed — **no** corrupt-bytes / restore-truth qualification |
| FRZ-TEN-04 | direct (scoped) | Q18 same-key cross-tenant queue mutation blocked (InMemory + SQLite) |
| FRZ-TEN-08 | direct (scoped) | Q22, Q23, Q18 retry/fence isolation — **no** checkpoint/restore proof |

**FRZ-REC-05:** no R3-R2 contribution (fork semantics).

Parent FRZ-STA/REC/TEN global criteria remain **OPEN**; revalidated via preserved Q01–Q29 (+ Redis when available).

**Status:** **CLOSED** — child **STATE-X-R3-R2-R1-R1** independently accepted at `d5979a531e41f9a45a3a00b0b150765ee34bb0f8`; qualification remediation accepted @ `5b0e937c1bb4cbc309bf12b94402552deaab5d82`.

---

## R3-R2-R1-R1 — Same-Key Cross-Tenant Compensation Queue Isolation Closure

**START_HEAD:** `5b0e937c1bb4cbc309bf12b94402552deaab5d82`

**Production delta:** NONE (qualification/test-only).

### Root cause (independent R3-R2-R1 audit)

- **BLOCKER:** Q18 tenant isolation did not mechanically prove bit-identical compensation idempotency key coexistence and full lifecycle isolation (lookup / claim / complete / fail / RETRYABLE reclaim / fence) across tenant A and tenant B on both InMemory and SQLite providers.

### Closure

| Path | Role |
|---|---|
| `tests/qualification/state_x/_r3_r2_qualification_tests.py` | Q18 expanded same-key cross-tenant adversarial matrix (InMemory + SQLite) |
| `docs/project/maintainers/qualification/STATE_X_R3_AUXILIARY_DURABLE_STATE_CONSISTENCY_CLOSURE.md` | This section |

**Exact accepted SHA:** `d5979a531e41f9a45a3a00b0b150765ee34bb0f8`

**Status:** **CLOSED / INDEPENDENTLY ACCEPTED**

---

## Roadmap note

Next mandatory child after R3-R3 remediation: **STATE-X-R3-R3-A1** (if authority blocker persists). **STATE-X** remains **CURRENT**. **STATE-X-R4** and **TRACE-X** remain **NOT ENTERED**.

---

## R3-R3 — SX-F12 Human Decision / HITL Persistence

**START_HEAD:** `91385b07ae1566f9f9ecff0d5705f1b141675da8`

### Contract owner and providers

| Role | Symbol / path |
|---|---|
| Semantic contract | `HumanDecisionPersistence` — `intergrax/runtime/human/persistence_contract.py` |
| InMemory provider | `InMemoryHumanDecisionPersistence` (non-durable; test/qualification) |
| SQLite provider | `SQLiteHumanDecisionStore` — `intergrax/runtime/human/store.py` |
| Composition owner | `create_sqlite_human_decision_store` / `open_human_decision_store_at` — `intergrax/runtime/persistence/sqlite_composition.py` |
| Record type | `HumanDecisionRecord` — `intergrax/runtime/human/models.py` |

`record()` is **insert-only** per global `decision_id`; duplicates raise `HumanDecisionPersistenceConflictError`.

### Caller inventory (production)

| Consumer | Classification |
|---|---|
| `runtime/nexus/orchestration/human_response.py` | WRITE EVIDENCE |
| `runtime/nexus/nexus_loop.py` | READ/WRITE EVIDENCE (injected store) |
| `tools/providers/hitl/service.py` | TOOLING/QUERY |
| `tools/registry/runtime_bindings.py` | structural `HumanDecisionStoreBinding` |
| `runtime/persistence/sqlite_composition.py` | COMPOSITION |
| `runtime/codecraft/ownership.py` (`resolve_codecraft_exec_authorization`) | **AUTHORITY CONSUMER — BLOCKER** |

`DecisionHumanReviewPort` remains a separate Decision/Governance domain; not collapsed into `HumanDecisionPersistence`.

### Evidence vs authority

- Human decision persistence = **durable evidence** only.
- **IN-SCOPE BLOCKER:** `resolve_codecraft_exec_authorization` maps scoped `HumanResponseVerdict.APPROVE` records to `CodeCraftExecAuthorization(authorized=True)` without canonical Governance re-evaluation (Case A). Mechanical proof: **R3-R3-Q17**.
- Proposed child: **STATE-X-R3-R3-A1** — Human Decision Evidence → Execution Authority Boundary Remediation.

### Provider parity matrix (mechanical)

| Invariant | InMemory | SQLite |
|---|:---:|:---:|
| contract instance | Q02 | Q02 |
| record/read round trip | Q03–Q04 | Q03–Q04 |
| duplicate ID fail-closed | Q03 | Q03 |
| duplicate cannot overwrite truth | Q04 | Q04 |
| wrong-tenant get blocked | Q05 | Q05 |
| task-list tenant isolation | Q06 | Q06 |
| escalation tenant isolation | Q07 | Q07 |
| queue summary tenant isolation | Q08 | Q08 |
| approver tenant consistency | Q09 | Q09 |
| deterministic ordering | Q10 | Q10 |
| no authority mint (persistence layer) | Q15 | Q15 |
| restart durability | N/A | Q11 |
| corrupt approver data fail-closed | N/A | Q12–Q14 |
| global `decision_id` collision | Q20 | Q20 |

### Tenant isolation

Proven via Q05–Q08, Q09 (write boundary), Q14 (read boundary), Q20 (global ID + tenant read scope).

### Typing

- `_approver_from_resolution` → `HumanApproverEvidence` (`human_response.py`).
- HITL intake approver narrowing (`intake_runner.py`); semantic `type: ignore` removed on approver boundary.

### FRZ mapping (scoped — no global PASS)

| FRZ | Classification | Evidence |
|---|---|---|
| FRZ-STA-01 | direct (scoped SX-F12) | Q01, Q19 single contract owner |
| FRZ-STA-02 | direct (scoped) | Q19 no duplicate production truth store |
| FRZ-STA-03 | direct (scoped) | atomic `record()` insert; conflict leaves prior row |
| FRZ-STA-04 | direct (scoped) | Q05–Q08 |
| FRZ-STA-05 | direct (scoped) | Q03–Q04, Q20 |
| FRZ-REC-01 | direct (scoped SQLite evidence reload) | Q11 |
| FRZ-REC-06 | supporting | Q12–Q14 partial/corrupt row fail-closed |
| FRZ-REC-10 | supporting | Q12–Q14 corrupt provenance |
| FRZ-TEN-04 | direct (scoped) | Q05–Q08, Q20 |
| FRZ-TEN-07 | supporting | Q09 approver tenant on record |
| FRZ-TEN-08 | supporting (SQLite human-decision scope) | Q11 |

**Not claimed:** FRZ-GOV-*, FRZ-EXE-* promotion; FRZ-REC-02/03/05/08/09; global TENANT-X/STATE-X closure.

### Findings

| ID | Classification |
|---|---|
| CodeCraft `resolve_codecraft_exec_authorization` uses persisted APPROVE as execution permission | **IN-SCOPE BLOCKER** → R3-R3-A1 |

**Status:** **BLOCKED PENDING INDEPENDENT A1 AUDIT** — persistence parity accepted @ `37b3e5e6d6e57601ca0289b94589d4d2d970d250`; A1 remediation landed in follow-up commit (see R3-R3-A1).

---

## R3-R3-A1 — Human Decision Evidence / Execution Authority Boundary

**START_HEAD:** `37b3e5e6d6e57601ca0289b94589d4d2d970d250`

### Root cause

`resolve_codecraft_exec_authorization` treated scoped `HumanDecisionRecord(APPROVE)` as `CodeCraftExecAuthorization(authorized=True)` on standalone CodeCraft paths (orchestrator, `codecraft.run`, wiring-bound without upstream flag).

### Before / after

**Before:** `HumanDecisionPersistence` → `APPROVE` → local resolver → `authorized=True` → physical execution.

**After:** persisted human decisions = evidence/query/correlation only; supervised standalone paths **fail closed** (`hitl_pending`); wiring-bound supervised path **fail closed** (A1-R1 removed hardcoded upstream bool).

### CodeCraft execution entrypoints

| Path | Classification |
|---|---|
| `runtime/codecraft/orchestrator.py` | STANDALONE FAIL-CLOSED when supervised/HITL profile |
| `tools/providers/codecraft/service.py` | STANDALONE FAIL-CLOSED |
| `runtime/codecraft/wiring_bound_capability_execution.py` | FAIL-CLOSED when supervised/HITL profile (A1-R1); autonomous delegates to catalog invoker |

### Persistence validation

Single helper `validate_human_decision_for_persistence` in `runtime/human/persistence_validation.py`; invoked by InMemory + SQLite before mutation.

### Ordering

Intake APPROVE path: `resolve_human_response_and_apply_canonical` before `persist_human_decision` (A1-Q16 AST + A1-Q04..Q06 runtime).

### Mechanical tests

`tests/qualification/state_x/_r3_r3_a1_qualification_tests.py` (A1-Q01..Q20 subset implemented; R3-R3 matrix preserved via `_r3_r3_qualification_tests.py`).

### Findings

| ID | Classification |
|---|---|
| (prior) CodeCraft evidence-as-authority | **remediated in A1** |

**Status:** **READY FOR AUDIT** (Cursor) — **STATE-X-R3-R3** remains **BLOCKED PENDING INDEPENDENT A1 AUDIT**; not CLOSED.

---

## R3-R3-A1-R1 — CodeCraft Authority Shortcut Removal

**START_HEAD:** `02ffbd9fc74676775d075de712d96e5a3d9699d7`

### Root cause

`WiringCodeCraftBoundCapabilityExecution` passed `upstream_canonical_hitl_satisfied=True` into `resolve_codecraft_exec_authorization`, minting local `authorized=True` without typed canonical HITL proof.

### Remediation

- Removed `upstream_canonical_hitl_satisfied` from production (`ownership.py`, all call-sites).
- Supervised / `require_hitl_before_exec` profiles **fail closed** on all CodeCraft entrypoints including wiring-bound execution (no catalog `invoke`, no `code.exec`).
- Autonomous / non-HITL wiring-bound paths still delegate physical execution to `ExecutionBoundCatalogToolInvoker` (canonical ToolRuntime / governance / MSE boundary unchanged).
- `HumanDecisionPersistence` remains **evidence only**; no execution allow path from persisted verdicts.
- Full canonical supervised CodeCraft HITL integration for execution-bound paths **not** implemented (architecture capability gap; see `UCA_6C_CANONICAL_HITL_REENTRY_PROOF.md`).

### Mechanical tests

`tests/qualification/state_x/_r3_r3_a1_r1_qualification_tests.py` (R1-Q01..Q20).

**Status:** **READY FOR AUDIT** (Cursor) — **STATE-X-R3-R3-A1** / **STATE-X-R3-R3** remain **BLOCKED PENDING INDEPENDENT R1 AUDIT**; not CLOSED.

---

## R3-R4 — SX-F13 Scheduler Durable State

**START_HEAD:** `e05b5b8eb3a60e8f46b5502083440e2e863e7204`

### Scope

SX-F13 only — `ScheduledResumePersistence`, `SchedulerLedger`, `LongRunningScheduler` WHEN-only orchestration, SQLite reference provider colocation without semantic merge.

### Root cause (pre-remediation)

`ScheduledResume.resume_metadata` (`human_approved`, `verdict`, `response_text`) was interpreted in `build_scheduled_resume_task()` to mint `TaskHumanInput` / synthetic scheduler approver — violating scheduler = WHEN-only.

### Remediation

- `validate_scheduled_resume_metadata()` in `scheduled_resume_metadata.py`; model validator on `ScheduledResume`; persistence schedule/load fail-closed.
- Delayed resume path no longer creates Human/Governance input from metadata; non-authoritative transport keys only (e.g. correlation).
- Canonical human **timeout** path unchanged (`build_timeout_resume_task` + `HumanTimeoutCoordinator` policy).
- Deterministic due ordering: `ORDER BY run_at_utc ASC, schedule_id ASC`.
- Duplicate `schedule_id` → `ScheduledResumeScheduleConflictError`.

### Before / after

**Before:** resume_metadata → human_approved/verdict → TaskHumanInput → synthetic approver → resume.

**After:** ScheduledResume → validated transport metadata → claim → checkpoint lookup → HostTaskExecutionPort → canonical admission.

### Tests

`tests/qualification/state_x/_r3_r4_qualification_tests.py` (R3-R4-Q01..Q35 + static gates). Historical SCHED-01 + `test_pcm_scheduler_integrity.py` replayed on current HEAD.

### FRZ (scoped, no global PASS)

Direct/supporting contributions documented in qualification matrix; global FRZ rows remain OPEN.

**Status:** **READY FOR AUDIT** (Cursor) — not CLOSED.

---

## R3-R4-R1 — Provider Validation Parity

**START_HEAD:** `71562caa919bf7d237d0b492d049f5dce8f903f9`

### Root cause

`ScheduledResumePersistence.schedule()` did not mechanically require full domain revalidation at the persistence acceptance boundary. SQLite revalidated via `model_validate` + `validate_scheduled_resume_metadata`; `MemoryScheduleStore` stored the passed object directly. Post-construction `model_copy(update={...})` can bypass Pydantic revalidation, so equivalent invalid semantic state could be accepted by one provider and rejected by another.

### Canonical validation owner

`validate_scheduled_resume_for_persistence()` in `intergrax/runtime/long_running/scheduled_resume.py` — single persistence acceptance rule: JSON round-trip `ScheduledResume.model_validate`, metadata authority-negative check, return canonical instance before any provider mutation.

### Providers

| Implementation | Role |
|---|---|
| `SQLiteTaskCheckpointStore.schedule` | Production reference provider |
| `MemoryScheduleStore` | R3-R4 qualification / replaceability provider |
| `_MemoryScheduleStore` (SCHED-01) | Historical qualification test provider |

All `schedule()` implementations invoke the canonical helper.

### Adversarial bypass

`valid = ScheduledResume(...)` → `invalid = valid.model_copy(update={"resume_metadata": {"human_approved": True}})` → both SQLite and Memory providers reject with `ScheduledResumeMetadataValidationError` (parameterized parity for `verdict`, `run_id`, `tenant_id`, `authorization_token`).

### No-mutation proof

Failed `schedule(invalid)` leaves SQLite row count 0 / empty `list_due`; Memory `_rows` unchanged.

### Provider matrix (R1)

SQLite and Memory/custom: valid accept, forbidden metadata reject, invalid no mutation, correlation metadata accept, duplicate `schedule_id` conflict — semantically aligned.

### R3-R4 regression

Claim/fence/UNCERTAIN/tenant/WHEN-only paths replayed via full `r3_r4` qualification suite on FINAL HEAD; R1 adds scoped regression hooks (Q15–Q17).

### FRZ (scoped, no global PASS)

Strengthens FRZ-STA-01/02/03/05, FRZ-REC-06/10 toward SX-F13; tenant rows as regression evidence only.

### Tests

`tests/qualification/state_x/_r3_r4_r1_qualification_tests.py` (R1-Q01..Q20). Inventory: `STATE_X_R3_R4_R1_PRE_AUDIT_HEAD`, `STATE_X_R3_R4_R1_ALLOWLIST_PATHS`.

**Status:** **CLOSED** @ `0413a3f68f19faeee404b73986748e6fd583dec3` (independent audit bookkeeping).

---

## R3-R5 — SX-F14 Agent Checkpoint State

**START_HEAD:** `0413a3f68f19faeee404b73986748e6fd583dec3`

### Scope

SX-F14 only — `AgentCheckpointStore`, `AgentRunCheckpoint`, ACP resume wiring, embedded `SideEffectRecord` lineage. Not global backup/restore, fork semantics, or TaskCheckpointPersistence merge.

### Root causes (pre-remediation)

1. Providers accepted post-construction `model_copy` mutations without canonical persistence revalidation.
2. `agent_id` could change within the same `(run_id, tenant_id)` stream on update.
3. `resolve_session_persistence` did not verify checkpoint `agent_id` against current effective agent.
4. `should_resume_acp_checkpoint` treated generic `human_response` metadata as an independent resume trigger.
5. Embedded side-effect records were not checked for run/step lineage before persistence.
6. SQLite load silently reconciled payload/column revision mismatch.

### Remediation

- `validate_agent_checkpoint_for_persistence()` in `checkpoint_store.py` — single acceptance rule before provider mutation (both InMemory and SQLite).
- `CheckpointAgentIdentityConflictError`, `CheckpointSideEffectLineageError`, `CheckpointDurableCorruptionError`, `CheckpointStreamIdentityConflictError`.
- Resume seam: `resolve_session_persistence(..., agent_id=)` validates run/tenant/agent continuity (FRZ-STA-08).
- Removed `human_response` branch from `should_resume_acp_checkpoint`.
- Durable revision disagreement fails closed on load.

### Before / after

**Before:** human_response → resume; checkpoint agent accepted without current-agent match; per-provider acceptance; silent revision reconcile.

**After:** ACP resume intent + checkpoint existence → current run/tenant/agent → canonical validator → state restore only.

### Provider matrix

| Invariant | InMemory | SQLite |
|---|---:|---:|
| contract | PASS | PASS |
| valid create | PASS | PASS |
| revision CAS | PASS | PASS |
| stale writer reject | PASS | PASS |
| step regression reject | PASS | PASS |
| agent identity immutable | PASS | PASS |
| invalid model reject | PASS | PASS |
| no mutation on invalid | PASS | PASS |
| tenant isolation | PASS | PASS |
| side-effect lineage | PASS | PASS |
| restart durability | N/A | PASS |
| corrupt durable bytes | N/A | PASS |

### Resume identity matrix

| Identity | Canonical source | Checkpoint role |
|---|---|---|
| tenant_id | current request/runtime | must match |
| run_id | current execution context | must match |
| agent_id | current effective agent | must match |
| revision | persistence provider | restored metadata |
| step_index | checkpoint state | resume position |
| execution authority | Governance/Execution | none |

### Tests

`tests/qualification/state_x/_r3_r5_qualification_tests.py` (R3-R5-Q01..Q35). Replayed: `test_pcm_checkpoint_cas_integrity.py`, `test_checkpoint_wiring.py`, GR10 ACP resume qualification.

### FRZ (scoped, no global PASS)

Primary evidence: FRZ-STA-01/02/03/04/05/08, FRZ-REC-01/03/04/06/10 toward SX-F14.

**Status:** **BLOCKED** — pending independent R3-R5 audit; superseded for provider provenance by **STATE-X-R3-R5-R1**.

---

## R3-R5-R1 — Checkpoint Provider Provenance & Composition

**START_HEAD:** `70ddff7e9d087b79d148be78c22344586af0744f`

### Root cause

`resolve_session_persistence()` resolved `AgentCheckpointStore` from `AgentRunRequest.metadata[AcpMetadataKey.CHECKPOINT_STORE]`, allowing a second composition path beside sanctioned host/runtime wiring (`HarnessHostRuntime`, `GraphExecutor`, `resolve_host_agent_checkpoint_store`).

### Trusted boundary

| Source | Role |
|---|---|
| `ACPSessionHostContext.agent_checkpoint_store` | sanctioned provider carrier |
| `resolve_session_persistence(..., checkpoint_store=)` | ACP session consumer |
| Public `metadata[CHECKPOINT_STORE]` | **no provider authority** (ignored) |
| `AcpMetadataKey.RESUME_FROM_CHECKPOINT` | lifecycle intent only |

### Remediation

- Extended `ACPSessionHostContext` with typed `agent_checkpoint_store`.
- `run_acp_session` passes `host.agent_checkpoint_store` into `resolve_session_persistence`.
- `merge_host_checkpoint_store` / `inject_acp_checkpoint_metadata` / task enricher propagate store via host context, not `CHECKPOINT_STORE` metadata.
- `build_acp_session_host_from_harness` injects `runtime.agent_checkpoint_store`.
- `wire_acp_run_request` classified test/lab helper (host context only).

### Before / after

**Before:** Host composition and public request metadata both supplied `AgentCheckpointStore`.

**After:** Host/application composition → `ACPSessionHostContext` → ACP session; request metadata cannot select durable provider.

### Proofs

- Adversarial: host store A vs request store B → restore uses A; B `get_latest` calls = 0.
- Missing host + request store → persistence disabled, no fallback.
- Ingress: `resolve_host_agent_checkpoint_store` + `make_acp_checkpoint_task_enricher` + `inject_acp_checkpoint_metadata` (Tier-3 shared wiring).
- Pluginability: InMemory/SQLite via host composition only; ACP session persistence imports contract only.

### Tests

`tests/qualification/state_x/_r3_r5_r1_qualification_tests.py` (R1-Q01..Q30). Inventory: `STATE_X_R3_R5_R1_PRE_AUDIT_HEAD`, `STATE_X_R3_R5_R1_ALLOWLIST_PATHS`.

### FRZ (scoped)

FRZ-STA-01/02/03/08, FRZ-REC-03/04/06, FRZ-TEN-04/08 — provider composition owner only; no global PASS.

**Status:** **READY FOR AUDIT** (Cursor) — **STATE-X-R3-R5** remains **BLOCKED** pending independent R1 audit.
