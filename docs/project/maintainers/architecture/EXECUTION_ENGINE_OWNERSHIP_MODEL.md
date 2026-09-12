# Execution Engine — Ownership & Boundary Certification Model (EE-A1)

**Classification:** `MAINTAINER_CERTIFICATION`  
**Status:** `CERTIFIED` (audit EE-A1 on `development`)  
**Audience:** Maintainers, enterprise qualification, architecture gates  

**Semantic authority:** [`UNIFIED_EXECUTION_ARCHITECTURE.md`](../../architecture/UNIFIED_EXECUTION_ARCHITECTURE.md) (UEA) and [`UNIFIED_EXECUTION_RUNTIME.md`](../../architecture/UNIFIED_EXECUTION_RUNTIME.md) (UER). On conflict, UEA wins.

**Related hub:** [`EXECUTION_ENGINE.md`](EXECUTION_ENGINE.md) (navigation only).  
**Bypass inventory (frozen):** [`../qualification/PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md`](../qualification/PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md)  
**Enterprise verification:** [`../qualification/EXECUTION_ENGINE_ENTERPRISE_VERIFICATION.md`](../qualification/EXECUTION_ENGINE_ENTERPRISE_VERIFICATION.md)

---

## 1. Execution Engine purpose

The **Execution Engine** is the platform area where **agentic work becomes durable, governed execution**:

- One **canonical execution path** for production side effects (agents, tools, child runs, compensation, recovery re-entry).
- **Single lifecycle owner** for root Run / Attempt / Execution admission and close.
- **Partitioned but non-duplicated** ownership for scheduling (orchestration vs long-running resume), identity mint, retry/resume, evidence, and governance evaluation.

The Engine is not a single class; it is the **contracted composition** of `ExecutionRuntime`, `ExecutionBoundary`, `StrategyExecutionRouter`, Nexus orchestration backends, `ChildExecutionRunner`, `RuntimeToolInvoker`, and frozen recovery/evidence planes.

---

## 2. Canonical execution flow

Production intent converges as follows (application layers supply **work intent** only; they do not own lifecycle):

```text
Application Intent (Task / host request)
        |
        v
Execution Admission (capacity, background identity admission, host port)
        |
        v
HostTaskExecutionPort / Execution facade
        |
        v
ExecutionRuntime  (sole root lifecycle owner)
        |
        v
ExecutionBoundary + active identity propagation
        |
        v
StrategyExecutionRouter
        |
        +-- AGENT --> AgentEnginePort --> Nexus (orchestration backend)
        |
        +-- ORCHESTRATION --> OrchestrationExecutor --> GraphExecutor / Nexus
        |
        v
ChildExecutionPort / ChildExecutionRunner  (child segments)
        |
        v
RuntimeToolInvoker --> Tool gateway --> Provider / integration adapter
```

**Background / queue (EP-06, EP-07):** broker worker → `execute_logical_task` → handler admitted via `BackgroundExecutionIdentity` → same runtime stack when handler performs execution work.

**Human continuation (EP-18):** governed grant clears HITL gate; resume re-enters canonical execution via `LongRunningCoordinator` / checkpoint contracts — no parallel lifecycle owner.

---

## 3. Ownership model

| Responsibility | Canonical owner | Duplicate owner (forbidden in production) |
| --- | --- | --- |
| Execution lifecycle (root open/close) | `ExecutionRuntime` (`runtime/execution/runtime.py`) | None |
| Identity creation (Run/Attempt/Execution mint) | `identity_authority.py` via `ExecutionRuntime` / `ChildExecutionRunner` / retry mint helpers | Governance, Nexus, tier-3 apps |
| Identity propagation (`bind_active_execution_identity`) | `ExecutionBoundary` (`boundary.py`) | None |
| Attempt lifecycle transitions (graph/Nexus) | `AttemptLifecycleService` | None |
| Retry **decision** + durable transition | `ExecutionAttemptRetryService` | Ad-hoc loops in agents/apps |
| Retry **execution** (re-admission) | Returns through `ExecutionRuntime` / graph runner | None |
| Orchestration scheduling (graph steps) | Nexus `GraphExecutor` / `NexusLoop` under active execution | None |
| Long-running **resume** scheduling | `LongRunningCoordinator` + `LongRunningScheduler` | Agent-local timers, worker schedulers |
| Recovery (checkpoint resume) | NPSC-5E plane (`LongRunningCoordinator`, checkpoint admission) | None |
| Partial fan-out recovery | `FanOutPartialRecoveryService` | Duplicate partial-retry loops |
| Evidence observe/persist/reconstruct | `RuntimeEventPersistence` + evidence contracts | Evidence must not schedule execution |
| Lineage | Execution lineage persistence + coordination contracts | None |
| Authority (side-effect authorization) | Attempt authority + `RuntimeToolInvoker` chain | Governance (evaluation only) |
| Governance | `agent_runtime_governance` / policy ports | Must not `execute()` or mint execution IDs |
| Terminal state / cancellation | `ExecutionRuntime` + W4 cancellation plane | None |
| Strategy selection | `StrategyExecutionRouter` | Nexus, applications |

### Identity authority (EE-A1)

| ID | Owner | Notes |
| --- | --- | --- |
| `RunId` | `identity_authority.mint_root_execution_identity` / background transport mint | Root and admitted background transport |
| `AttemptId` | `identity_authority` + `AttemptLifecycleService` for transitions | Retry uses `mint_retry_attempt_id()` |
| `ExecutionId` | `identity_authority` (root, child, retry segments) | Child via `mint_child_execution_id()` |
| `EventId` (runtime journal) | `mint_event_id()` at emission; **durability ownership** `RuntimeEventPersistence` | Store enforces claim integrity |

**Residual (non-violation, documented debt):** HTTP/MCP intake may pre-mint TaskId/RunId in `task_run_bridge` before durable host execution; lifecycle still opens in `ExecutionRuntime` at host entry (see enterprise verification §10).

### Entry classification (production)

| Element | Through Execution Engine? | Owner |
| --- | --- | --- |
| Agent execution | **YES** | `ExecutionRuntime` → Nexus / AgentEngine |
| Tool execution | **YES** | `RuntimeToolInvoker` inside active execution |
| Provider execution | **YES** (via tool/integration adapters) | Tools + integration wiring |
| Background jobs | **YES** (admitted handlers) | Background execution bootstrap |
| Scheduled tasks | **YES** (long-running ledger / queue) | `LongRunningScheduler`, queue worker |
| Recovery execution | **YES** (re-entry only) | NPSC-5E services |
| Human continuation | **YES** | Governed continuation + coordinator |

---

## 4. Forbidden bypasses

Production **must not**:

```text
Application / Agent --> Provider.call()           (skip RuntimeToolInvoker)
Worker --> tool.invoke() without admitted port    (compensation pattern was BY-02; closed)
Factory --> ChildExecutionRunner() local mint     (BY-01; closed)
Integration --> external API for agentic work     (without tool path + authority)
Governance --> execute() / schedule() / retry()   (evaluation only)
Evidence --> trigger execution or change authority
```

**Inventory verdict (U5 / P0):** `BYPASS = 0` for supported production entrypoints (22 EP rows; 19 CANONICAL, 3 LEGACY NON-PRODUCTION).

### Closed bypasses (historical)

| ID | Severity | Status |
| --- | --- | --- |
| BY-01 | P1 | Closed U4 — delegated subtask via `ChildExecutionPort` |
| BY-02 | P0 | Closed U2 — compensation via `CompensationSideEffectExecutionPort` |

### Non-production legacy paths (not certification failures)

| ID | Location | Risk |
| --- | --- | --- |
| EP-17 | `WorkStageCapabilityLoop` (tests / catalog only) | P3 — not imported from tier-3 apps |
| EP-20 | `intergrax/experiments/workflow.py` — direct `UnifiedTaskRunner` | P3 — lab only |
| EP-21 | `eval/nexus_eval_runner.py` | P3 — eval only |

### Architectural coupling risks (not execution bypasses)

| Risk ID | Location | Severity | Note |
| --- | --- | --- | --- |
| AC-01 | Nexus tool planning OpenAI schema export | LOW | Wire format; execution still via `RuntimeToolInvoker` |
| AC-02 | `external_operations/provider_cancellation.py` provider branches | LOW | Cancellation adapter registry, not execution entry |
| AC-03 | Intake `mint_intake_execution_identity` | MEDIUM (debt) | Pre-host mint; track convergence to runtime-only mint |

---

## 5. Extension rules

1. **New production entrypoint** → add EP row to P0 inventory + architecture gate before merge.
2. **New child execution surface** → only via `ChildExecutionRunner` import allowlist (frozen gate).
3. **New tool side effect** → register in catalog; invoke only through `RuntimeToolInvoker` with governance + authority chain.
4. **No new lifecycle owner** — extend `ExecutionRuntime` hooks/admission, do not fork runners in `applications/` or `agents/`.

---

## 6. Plugin rules

Target layering:

```text
Execution Engine
       |
       v
Capability / tool contract
       |
       v
Plugin adapter (integrations/providers, llm_adapters)
       |
       v
Vendor provider
```

- Runtime **must not** branch on vendor for **execution routing** (provider switches belong in adapter registration and tool wiring).
- Integrations **SQL `execute()`** and store drivers are **infrastructure**, not agentic execution bypasses, when reached only from tool handlers inside admitted execution.

---

## 7. Recovery rules

| Concern | Canonical owner |
| --- | --- |
| Retry decision + attempt transition | `ExecutionAttemptRetryService` |
| Resume / long-running progress | `LongRunningCoordinator`, `LongRunningScheduler` |
| Partial fan-out recovery | `FanOutPartialRecoveryService` |

Forbidden: duplicate retry/resume loops in tier-3 apps, agents, or workers that re-invoke Nexus without admission.

---

## 8. Evidence rules

Evidence plane **may:** observe, persist, reconstruct, export (NPSC-5F).

Evidence plane **must not:** control execution, change authority, or trigger schedule/retry/resume.

Durable runtime journal ownership: **`RuntimeEventPersistence`** (with integrity claims on `EventId`).

---

## 9. Scheduler audit (EE-A1)

| Scheduler | Role | Execution bypass? |
| --- | --- | --- |
| `LongRunningScheduler` | Durable long-running task ledger / resume | **No** — calls host execution |
| Nexus `GraphExecutor` | Orchestration step scheduling inside run | **No** — under active execution |
| `CapacityScheduler` | Capacity / admission governance | **No** — not agent task scheduler |
| `VendorKnowledgeSyncScheduler` | Vendor KB sync jobs | **No** — domain sync, not UAEP execution |
| `AdaptationScheduler` | Adaptive runtime tuning | **No** — not execution lifecycle |

**Verdict:** one **orchestration** scheduler (Nexus graph) and one **long-running resume** scheduler; no separate agent/worker/recovery execution schedulers that bypass `ExecutionRuntime`.

---

## 10. Architecture gates (automated proof)

| Gate | Path |
| --- | --- |
| P0 bypass inventory | `tests/unit/runtime/architecture/test_platform_execution_unification_p0_bypass_inventory.py` |
| Identity single authority | `tests/unit/runtime/architecture/test_execution_identity_single_authority_gate.py` |
| Canonical engine conformance | `tests/unit/runtime/architecture/test_npsc3c_d_canonical_execution_engine_conformance_gate.py` |
| Governance boundary | `tests/unit/runtime/architecture/test_npsc4_agent_runtime_governance_gate.py` |
| EE-A1 certification doc | `tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py` |

---

## 11. EE-A1 certification verdict

| Criterion | Result |
| --- | --- |
| No hidden production execution bypass | **PASS** (`BYPASS = 0`) |
| Single lifecycle owner | **PASS** (`ExecutionRuntime`) |
| Single orchestration + long-running scheduler model | **PASS** |
| Single execution identity mint authority (scoped AST) | **PASS** |
| Provider-neutral execution routing | **PASS** (coupling risks AC-* documented) |
| Architecture documentation | **PASS** (this document) |
