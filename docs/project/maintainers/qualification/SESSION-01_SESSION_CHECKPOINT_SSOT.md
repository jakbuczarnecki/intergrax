# SESSION-01 — Session/Checkpoint SSOT (qualification)

**SESSION-01-C1R audit (2026-09-17)**

| Field | SHA / result |
| ----- | ------------ |
| START_HEAD | `85e2079e1075a5185868de8356accb40d7f20e65` |
| START_ORIGIN_DEVELOPMENT | `60ba65bb86e5a74a821f90c8d6a2881e5ea2c83e` |
| TESTED_HEAD | `b10000a87` (TASK SHA for qualification rerun) |
| FINAL_HEAD | `b10000a87` (post C1R commit; verify `origin/development` after push) |
| Qualification command | `uv run pytest tests/qualification/session_01/ -q` |
| Collection | `uv run pytest tests/qualification/session_01/ --collect-only -q` → **26 tests, 0 errors** |
| SESSION batch | **26 passed** (includes mapped Q1..Q20 node batch + J3 cross-worker) |
| J3 root cause | Class-level `_HitlAgent.runs` summed worker A pause + worker B resume in one process; not production identity leak |
| J3 fix | Reset counter at worker boundary after pause (`test_worker_checkpoint_resume_via_queue_payload`) |
| Bounded regressions | EE continuation + human projection + GR-1 Q20 + MP4R7 + TR authority closure + J3 → **201 passed** |
| FULL REPO COLLECTION | **6 errors** (pytest_plugins conftest placement; integration optional/live) — downstream-owned |
| Working tree (post-commit) | **clean** at TASK SHA |
| SESSION-01 final | **CLOSED / ENTERPRISE QUALIFIED** (C1R at `b10000a87`; independent GitHub audit required) |

**TASK SHA (audit baseline):** `7b633520a81542ef729aa7f82a74dc100af10c35`  
**Qualification delta:** harness convergence + HITL grant ordering fix (see git log after commit).

## SSOT semantics (one owner per fact)

| Semantic fact | Canonical owner | Durable store | Projection consumers |
| ------------- | --------------- | ------------- | -------------------- |
| Execution identity (task/run/attempt/execution) | Execution Engine / `ActiveExecutionIdentity` | Execution lineage stores (domain) | Nexus, Task trace, Governance grants |
| Execution lifecycle (terminal, progress) | Execution Engine | Execution-owned persistence | Task state, diagnostics |
| Continuation lifecycle (pause/WAITING/resume) | `ExecutionContinuationPort` | `ExecutionContinuationStateStore` | `HumanPauseCoordinator`, Task governance fields |
| Pause request | `ExecutionContinuationPort.request_pause` | Continuation snapshot | Task pause_record (projection) |
| WAITING / human gate | Continuation lifecycle driver + port | Continuation snapshot | Human request projection |
| Resume authorization | `apply_resolution` → `RESUME_AUTHORIZED` | Continuation CAS | HITL resolution projection |
| Canonical resume | `ExecutionContinuationPort.resume` | Continuation CAS | `project_continuation` |
| Governed approval grant | `GovernedContinuationGrantCoordinator` | Task metadata / grant record (governance) | Side-effect execution gates |
| Human approval evidence | `HumanDecisionPersistence` (contract) | Provider (`InMemory` / SQLite impl) | Audit, intake replay |
| Task checkpoint snapshot | `TaskCheckpointPersistence` | SQLite default (`long_running`) | Long-running restore |
| Agent local checkpoint | `AgentCheckpointStore` | In-memory / SQLite (`agents/persistence`) | Agent resume only |
| Sandbox session | `SandboxSession` / hosted sandbox | Provider backend | UAEP route metadata |
| Memory session scope | Memory control `session_id` | Memory providers | Memory reads/writes |

## As-built inventory

| Mechanism | Contract | Store/provider | Owner | Canonical? | Durable? | Recoverable? | Purpose |
| --------- | -------- | -------------- | ----- | ---------: | -------: | -----------: | ------- |
| Execution continuation | `ExecutionContinuationPort` | Internal to service | Execution | yes | via store | yes (durable store) | Pause/resume authority |
| Continuation snapshots | `ExecutionContinuationStateStore` | In-memory default; `ExecutionContinuationDurableBacking` + `ReconstructedDurableExecutionContinuationStateStore` | Execution | yes | `is_durable` | restart recompose (`reconnect_execution_engine_continuation_dependencies`) | CAS episodes per four-ID |
| Lifecycle driver | `ExecutionContinuationLifecycleDriver` | n/a (uses store) | Execution | yes | n/a | n/a | PAUSED→WAITING transitions |
| Composition | `wire_execution_engine_continuation_dependencies` | Injected store | Nexus / Execution runtime | wiring | inherits store | yes | Production-like bundle |
| Internal HITL orchestration | `InternalOrchestrationContinuation` | Uses port store | Nexus (internal) | adapter | inherits | inherits | `establish_canonical_hitl_pause` |
| Task/Human pause UI | `HumanPauseCoordinator` | Task runtime governance | Human projection | no | task metadata | rebuild from continuation | Projection only |
| Governed continuation grant | `GovernedContinuationGrantCoordinator` | Task governance grant | Governance | grant auth | task-bound | re-validate on consume | Scoped approval |
| Task checkpoint | `TaskCheckpointPersistence` | `SQLiteTaskCheckpointStore` | Long-running / task | recovery snapshot | yes | resume_token path | Work snapshot, not lifecycle |
| Agent checkpoint | `AgentCheckpointStore` | SQLite / in-memory | Agent domain | agent-local | optional | agent resume | No ExecutionId minting |
| Human decisions | `HumanDecisionPersistence` | `InMemoryHumanDecisionPersistence` / SQLite | Human runtime | evidence | optional durable | replay | Not continuation authority |

## Definitions

- **Session (platform):** overloaded term — sandbox session, memory `session_id`, provider session; **no** single platform `SessionId` replacing `ExecutionId`.
- **Checkpoint:** versioned recovery snapshot (`TaskCheckpoint`, agent checkpoint); **not** pause/resume authority.
- **Continuation:** legal resumable execution episode keyed by `continuation_id` + four-ID identity; authority = `ExecutionContinuationPort`.

## Canonical continuation authority

`ExecutionContinuationPort` at `intergrax/contracts/execution_continuation.py`, implemented by `ExecutionContinuationService` + `ExecutionContinuationStateStore` (`intergrax/runtime/execution/continuation/`).

Production Nexus: `NexusLoop` → `wire_execution_engine_continuation_dependencies` → `InternalOrchestrationContinuation` → intake/graph runners (`require_internal_hitl_continuation`).

## Q20 closure (GV-01 carry-forward)

Test: `test_nexus_intake_governed_approval_without_nexus_ae_forwarding`  
Root cause: intake harness omitted `hitl_continuation`; pause used legacy `apply_pause` only.  
Fix: production-like harness helpers (`establish_canonical_governed_pause_for_hitl_test`, wired runner) + grant creation **before** `canonical_resume_after_authorization` in `NexusIntakeRunner`.

## SESSION-Q mapping (existing gates)

| ID | Evidence |
|----|----------|
| Q1 | `test_gr5_r2_canonical_pause_resume.py` |
| Q2 | `test_gr5_r3_continuation_projection.py` |
| Q3–Q4 | `test_gr5_r5_restart_exact_identity.py` |
| Q5–Q9 | `test_gr5_r2_r3_canonical_continuation_identity_enforcement.py`, continuation service tests |
| Q10–Q11 | projection + restart tests |
| Q12–Q13 | human persistence + governed grant tests |
| Q14 | Q20 test (GR-1) |
| Q15–Q16 | agent/task checkpoint unit tests (scope boundaries) |
| Q17 | custom store via `ExecutionContinuationStateStore` in continuation tests |
| Q18 | `TaskCheckpointPersistence` / `AgentCheckpointStore` contract tests |
| Q19 | CAS / stale revision continuation tests |
| Q20 | architecture gates (no global session registry) |

## Downstream follow-ups

| Item | Class | Owner |
|------|-------|-------|
| Dedicated `tests/qualification/session_01/` aggregator | **DONE** (`60ba65bb`, C1R J3 harness) | Harness |
| Production durable continuation default in all worker profiles | QUALIFY | Execution hosting |
| Legacy `apply_pause` callers outside tests | CONVERGE | Human/Nexus |
| Unrelated WIP on `development` worktree | BLOCKER | Operator |

## Durability model

- Default lab/test: `InMemoryExecutionContinuationStateStore` (`is_durable=False`).
- Durable path: `ExecutionContinuationDurableBacking` + export/restore (`export_durable_continuation_state`, `reconstructed_durable_execution_continuation_state_store`).
- No vendor types in contracts (`execution_continuation_state_store.py` is provider-neutral).

## Recovery flow (canonical)

```text
process recompose → load ExecutionContinuationStateStore
→ resolve_current_episode_for_identity
→ validate identity + revision
→ resume via ExecutionContinuationPort
→ project Task/Human
→ Execution Engine continues (not direct Nexus bypass)
```
