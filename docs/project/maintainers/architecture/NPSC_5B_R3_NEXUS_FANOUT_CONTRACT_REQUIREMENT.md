# NPSC-5B/R3 — Nexus Fan-Out Contract Requirement

**Status:** `R4 CANONICAL ADAPTER IMPLEMENTED` — NPSC-5B final qualification may proceed

**Series:** NPSC-5B — Bounded Multi-Agent Fan-Out / Fan-In

**Branch:** `development`

**Related:**

- [`NPSC_5B_CROSS_SYSTEM_FANOUT_OWNERSHIP_RECONCILIATION.md`](NPSC_5B_CROSS_SYSTEM_FANOUT_OWNERSHIP_RECONCILIATION.md) (R1 `FROZEN`, R2 `RETIRED`, R4 `PASS`)
- [`NPSC_5_MULTI_AGENT_PRODUCTION_ARCHITECTURE.md`](NPSC_5_MULTI_AGENT_PRODUCTION_ARCHITECTURE.md)

---

## 1. R3 gate outcome

```text
STATUS: R4 CANONICAL ADAPTER IMPLEMENTED
R2 correction retired; R3 shared contract delivered; R4 adapter wired
```

R3 contract-closure analysis (F1–F10) identified gaps in the public Execution/Nexus surface. R3 delivered the minimal shared contracts; R4 wired NPSC fan-out as a consumer-only adapter.

**Canonical production path:**

```text
FanOutRequest
→ BoundedMultiAgentFanOutService
→ FanOutOrchestrationPort
→ OrchestrationTopologySubmissionPort
→ Nexus
→ child Execution
→ MultiAgentCoordinationService
→ FanOutResult
```

---

## 2. Why R2 is invalid (correction required)

R2 removed the Agent Distribution `asyncio.Semaphore` scheduler but introduced a fan-out-specific mini-runtime in `intergrax/runtime/execution/multi_agent_fanout_orchestration.py` that locally constructs:

| R2 bypass | Why invalid |
| --------- | ----------- |
| `AgentRegistry()` | Bypasses canonical active revision / registry projection |
| `AgentEngine(registry)` | Second agent engine, not composition-root wired |
| `GraphExecutor(...)` | Local Nexus scheduler instance, not canonical `NexusLoop` host |
| `_FanOutCoordinationSlotAgent` / `_FanOutMergeAgent` | Synthetic scheduling agents |
| `PrefixStubLLMAdapter` | Technical stub LLM for topology nodes modeled as agents |
| `mint_task_id()` + `tenant_id="npsc-fanout"` | Synthetic identity bypass |
| `dict` + `asyncio.Lock` outcome side-channel | Process-local orchestration state masquerading as contract |
| `graph_node_id` + `fanout-slot-` prefix protocol | Magic metadata identity mapping |

R2 **appears** to route through `ExecutionWorkPort` + `ORCHESTRATION`, but the orchestration backend is `FanOutTopologyOrchestrator` — not the canonical `NexusLoop` / `OrchestrationExecutor` host wired at composition root (`build_host_task_execution`).

---

## 3. What exists today (partial, insufficient)

| Surface | Owner | Sufficient for NPSC fan-out? |
| ------- | ----- | ---------------------------- |
| `ExecutionWorkPort` + `ChildExecutionWorkPort` | Execution | **Partial** — child submission seam exists |
| `StrategyExecutionRouter` + `ExecutionCapability.ORCHESTRATION` | Execution | **Partial** — strategy routing exists |
| `OrchestrationExecutor` → `NexusOrchestrationPort.handle_task` | Execution → Nexus | **No** — bound to pre-materialized `Task`; no dynamic topology submission |
| `NexusLoop` (canonical host) | Nexus | **Partial** — owns registry, engine, budget, lifecycle; entry is classify → plan → graph, not typed topology injection |
| `GraphExecutor` (on `NexusLoop`) | Nexus | **Partial** — bounded scheduling + `ChildExecutionRunner` per node; not a public submission API; agent-centric node model |
| `FanOutRequest` / `FanOutResult` | Agent Distribution | **Yes** — semantic contracts are correct and retained |
| `MultiAgentCoordinationService` | Agent Distribution | **Yes** — per-slot specialist resolution owner |

---

## 4. Minimal reusable contract required

Owner: **Execution / Nexus** (shared surface). NPSC may consume; NPSC must not own.

### 4.1 Typed dynamic topology submission port

A public, typed API on the canonical orchestration host allowing consumers to submit:

```text
orchestration plan / topology
+ per-slot typed payloads
+ bounded concurrency policy (max_parallel_nodes equivalent)
```

**without** constructing:

```text
new GraphExecutor()
new AgentRegistry()
new AgentEngine()
```

Semantically equivalent to (names illustrative, not prescriptive):

```text
OrchestrationTopologySubmissionPort.submit(
    topology: OrchestrationPlan,
    scheduling_policy: OrchestrationSchedulingPolicy,
    execution_context: ActiveOrchestrationContext,  # inherits RunId/AttemptId/tenant/principal/budget
) -> OrchestrationExecutionHandle
```

**Current gap:** `OrchestrationExecutor.execute(task)` requires a bound `Task` and delegates to `NexusLoop.handle_task` — full intake/planning pipeline. No port accepts a caller-supplied dynamic topology with slot payloads.

### 4.2 Typed orchestration slot identity

Explicit, immutable mapping:

```text
consumer slot ID (e.g. FanOutItemId)
↔ orchestration node / slot ID
```

Must **not** rely on:

```text
metadata["graph_node_id"]
"fanout-slot-" string prefix parsing
```

**Current gap:** `ExecutionNode.node_id` is an untyped `str` with no typed slot-payload binding contract.

### 4.3 Typed per-slot outcome contract

Canonical orchestration must return or emit outcomes semantically equivalent to:

```text
slot_id
status (success | failure)
result | typed failure (code, message, optional partial)
```

Deterministic fan-in order follows **submission order**, not completion order.

Must **not** use:

```text
shared mutable dict
process-local closure capture
asyncio.Lock as result channel
AgentExecutionResult-only projection without slot binding
```

**Current gap:** Nexus returns `TaskResult` / per-node `AgentExecutionResult`. No typed `OrchestrationNodeOutcome` with explicit slot identity.

### 4.4 Child-work node executor (non-agent topology node)

Specialist slots must route:

```text
orchestration node
→ real child Execution (ChildExecutionRunner)
→ MultiAgentCoordinationService
→ DelegatedSubtaskService
→ specialist
```

**without** modeling the node as a synthetic `Agent` requiring stub LLM.

**Current gap:** `GraphExecutor` resolves nodes through `AgentRegistry` → `AgentEngine`. No public non-agent node executor contract for coordination-delegation work units.

### 4.5 Canonical host injection at composition root

Fan-out composition must receive the **existing** canonical orchestration host (`NexusLoop` / wired `OrchestrationExecutor` / host `GraphExecutor`) from composition root — not mint a parallel instance.

**Current gap:** `build_fan_out_orchestration_work_port(coordination)` constructs `FanOutTopologyOrchestrator` with local registry/engine/executor.

### 4.6 Partial failure semantics

Orchestration must support:

```text
slot A: success
slot B: failure
slot C: success
→ aggregate result preserves all three typed outcomes
```

without fail-fast sibling cancellation (unless explicitly configured by orchestration policy).

**Current gap:** No typed aggregate outcome contract; R2 relies on per-slot try/except inside synthetic agent + side-channel dict.

---

## 5. F1–F10 summary

| ID | Question | Answer |
| -- | -------- | ------ |
| F1 | Canonical Nexus host exists? | **YES** — `NexusLoop` via `build_host_task_execution` / `OrchestrationExecutor` |
| F2 | Typed dynamic topology submission API? | **NO** — `OrchestrationExecutor` is task-bound only |
| F3 | Child Execution per slot on canonical path? | **PARTIAL** — canonical `GraphExecutor` uses `ChildExecutionRunner`; R2 bypasses canonical host |
| F4 | Specialist resolution without concrete agent ID? | **NO on R2 path** — synthetic proxy agents; canonical path requires registry lookup by agent_id |
| F5 | Typed slot identity? | **NO** — R2 uses `fanout-slot-` metadata protocol |
| F6 | Typed per-slot outcome? | **NO** — R2 uses mutable side-channel |
| F7 | `max_concurrency` via canonical scheduler? | **NO on R2 path** — local `GraphExecutor(max_parallel_nodes=...)` |
| F8 | Identity / tenant propagation? | **NO on R2 path** — synthetic tenant/user/task |
| F9 | Budget concurrency safety? | **YES on canonical `ChildExecutionRunner` path** (`InMemoryExecutionBudgetLedger` lock); **NOT exercised by R2 slot path** |
| F10 | Partial failure without fail-fast? | **PARTIAL in R2** (per-slot catch) but via invalid architecture; **NO on canonical contract** |

---

## 6. Decision system impact

```text
DECISION SYSTEM CHANGE REQUIRED: NO
```

---

## 7. Expected consumers

- NPSC-5B multi-agent fan-out / fan-in
- Generic orchestration consumers (future workflow engines)
- Decision-produced orchestration plans (NPSC-5C and beyond)
- Any bounded parallel topology with typed per-unit outcomes

---

## 8. NPSC blocked until

1. Execution / Nexus delivers minimal contracts in §4.
2. Composition root wires `FanOutOrchestrationPort` to canonical host — no local `GraphExecutor` / `AgentRegistry` / `AgentEngine`.
3. R3 architecture gates pass on the full fan-out composition surface (not only `bounded_multi_agent_fanout.py`).
4. R2 correction target `multi_agent_fanout_orchestration.py` is removed or reduced to thin typed adapter only.

---

## 9. Recommended owner session

**Execution / Nexus** — shared contract design and implementation.

NPSC session resumes after contract delivery for adapter wiring and gate qualification.

---

## 10. Recommended next task (Execution / Nexus)

Define and implement minimal public contracts:

1. `OrchestrationTopologySubmissionPort` (or extend `NexusOrchestrationPort` with typed topology submission)
2. `OrchestrationSlotId` + typed slot payload binding
3. `OrchestrationSlotOutcome` + `OrchestrationAggregateResult`
4. Non-agent child-work node executor seam for coordination-delegation slots
5. Composition-root factory accepting canonical `NexusLoop` / orchestration host

Deliver architecture proof gates before NPSC re-engages.
