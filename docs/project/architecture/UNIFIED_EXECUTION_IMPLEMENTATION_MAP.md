# Unified Execution Implementation Map

**Status:** Canonical implementation-mapping artifact (UE-DOC-0.9R1)
**Classification:** `SUPPORTING_MODEL / SATELLITE` — subordinate to [`UNIFIED_EXECUTION_ARCHITECTURE.md`](UNIFIED_EXECUTION_ARCHITECTURE.md) (`META_ARCHITECTURE`); **not** a new DOMAIN, **not** an architecture authority, **not** paired with a separate implementation plan  
**Owner:** Intergrax Platform Architecture (migration coordination)  
**Audience:** Cursor UE-1+ implementers, domain maintainers, audit/review sessions  
**Evidence baseline (UE-DOC-0.9R1):**
- Original architecture mapping pin: `development` @ `0398101abb5374a060013400656eb0838b744bc2`
- Concurrent runtime change during UE-DOC-0.9 mapping: `49e765377935469566f7d265e1129f735b7f2ae9` (DIAG-5D — stable `ProblemId` / lifecycle reconciliation)
- UE-DOC-0.9 mapping commit: `4ce7dfee540b8cce770aa9a447096dcfa83c4b75`
- UE-DOC-0.9R1 revalidation baseline: `development` @ `862c0dc57928c3af34329a836482df89fec786f4` (includes DIAG-5D-R1 — occurrence-derived `first_seen_at` / `last_seen_at`)

---

## Purpose

This document is the **bridge** between frozen Unified Execution Architecture (UEA) and future runtime implementation. It maps, for every critical execution-path responsibility:

```text
TARGET ARCHITECTURE
    ↓
CURRENT COMPONENT (with code evidence)
    ↓
GAP
    ↓
REQUIRED TRANSFORMATION
    ↓
DEPENDENCIES
    ↓
IMPLEMENTATION ORDER
```

**Authority chain:** UEA (`UEA-INV-001`..`021`) → domain architecture hubs → **this map** → UE-1+ implementation slices. This map **must not** silently change target semantics in domain hubs.

**Disposition vocabulary** (required on every mapped component):

| Disposition | Meaning |
|-------------|---------|
| **KEEP** | Target responsibility already correct |
| **KEEP_AND_REWIRE** | Component stays; caller/entry/ownership changes |
| **TRANSFORM** | Concept remains; contract/identity/lifecycle must change |
| **SPLIT_RESPONSIBILITY** | Owns multiple target responsibilities that must move to canonical owners |
| **DEPRECATE** | Compatibility surface may remain temporarily; not target architecture |
| **REMOVE_AFTER_MIGRATION** | Target has no role for this path/abstraction |
| **CURRENT_EVIDENCE_ONLY** | Migration fact; not a future architecture component |

---

## 1. Public execution entry

### TARGET

`execution.execute(request=..., output_type=...)` — developer describes **what**; no public `execute_agent`, `execute_with_nexus`, `mode="react"|"agent"|"nexus"`.

### CURRENT competing surfaces

| Surface | Current role | Callers | Retained internally? | Target relationship | Disposition | Migration risk |
|---------|--------------|---------|----------------------|---------------------|-------------|----------------|
| `UnifiedTaskRunner.run_task` / `run_runtime_request` | De facto public harness entry; always routes to `NexusLoop` | `applications/*/serving/*`, `harness_task_routes`, `task_control`, MCP servers, LKW host | Yes (compat shim during migration) | Internal adapter behind `execution.execute` for orchestration-capable workloads only | **KEEP_AND_REWIRE** | High — universal Nexus coupling |
| `NexusLoop.handle_task` | Full classify→plan→graph→agent pipeline | `UnifiedTaskRunner` | Yes (orchestration strategy internals) | Orchestration strategy executor only | **KEEP_AND_REWIRE** | High — mistaken as universal entry |
| `AgentEngine.execute` / `_execute_agent_impl` | Direct agent execution (UAEP/ACP) | `GraphExecutor`, tests, some tooling | Yes (agentic strategy) | Agentic strategy executor behind boundary | **KEEP_AND_REWIRE** | Medium — bypass if not behind boundary |
| `HarnessKernel.execute_step` | ACP session step boundary | ACP agents via `on_next_step` bridge | Yes (ACP path) | ACP-specific; not universal entry | **KEEP** | Low if scoped to ACP |
| Application HTTP `/v1/*/run` routes | Product-shaped task submission | Tier-3 apps | Yes | Project `Task`/`TaskResult` → neutral execution request | **TRANSFORM** | Medium — product result leakage (SHM-FIX-C) |
| `WorkerRuntime` + `execute_logical_task` | Background task dispatch | Queue workers | Yes | Re-enter boundary with transport envelope | **KEEP_AND_REWIRE** | High — identity drift on redelivery |
| `ExecutionMode` (`STRICT`/`BALANCED`) on harness env | Host posture, not strategy | `harness/app.py`, wiring | Yes (governance posture) | **Not** strategy resolution signal | **KEEP** | Low — do not conflate with strategy |

**Evidence:** `intergrax/runtime/task/unified_task_runner.py`, `intergrax/runtime/nexus/nexus_loop.py:382`, `intergrax/agents/agent_engine.py:220`, `intergrax/runtime/kernel/step_kernel.py`, `intergrax/background_tasks/worker_runtime.py`, `intergrax/applications/_shared/harness_task_routes.py`, `tests/integration/applications/test_unified_execution_entry_j1.py`.

**Gap:** No neutral `execution.execute`; all major paths either Nexus-first or agent-direct.

**Transformation:** Introduce internal execution coordinator module; adapt existing runners as strategy backends; preserve old callers via compatibility facades until UE-10.

**Dependencies:** Wave 0 (`ExecutionId` contracts) → Wave 1 (boundary skeleton) → Wave 2/3 (strategy proofs).

---

## 2. Canonical identity

### TARGET

`TaskId` → `RunId` → `AttemptId` → `ExecutionId` → `EventId`.

### CURRENT mapping

| Concern | Current owner | Evidence | Gap | Transformation | Disposition |
|---------|---------------|----------|-----|----------------|-------------|
| **TaskId minting** | `Task.task_id` default_factory | `intergrax/runtime/task/task.py:57`, `mint_task_id()` in `execution_identity.py` | None at Task layer | Propagate into execution envelope | **KEEP** |
| **RunId minting** | `UnifiedTaskRunner` / checkpoint resume / explicit request | `unified_task_runner.py:67`, `execution_identity_from_checkpoint` | Run minted at task entry, not execution boundary | Boundary owns Run admission for new work | **KEEP_AND_REWIRE** |
| **AttemptId minting** | `NexusLoop.handle_task`, `bootstrap_background_execution`, `transition_active_execution_identity` | `nexus_loop.py:393`, `background_execution/bootstrap.py:84`, `execution_identity.py:94` | **New AttemptId on every worker boundary** (debt); `transition_retry` used without taxonomy guard | Classify retry types; preserve Attempt on transport redelivery | **TRANSFORM** |
| **ExecutionId** | **Missing** | UEA §25; no `ExecutionId` type in `execution_identity.py` | Canonical gap #1 | Add typed `ExecutionId`, `parent_execution_id`, minting authority at execution lifecycle layer | **TRANSFORM** (new contract) |
| **EventId** | `RuntimeEvent.event_id` | `runtime_event.py:90` | Events lack `execution_id` field | Add optional-then-required `execution_id` on `RuntimeEvent` | **TRANSFORM** |
| **Active identity propagation** | `ContextVar` via `bind_active_execution_identity` | `execution_identity.py:18-98` | Run+Attempt only; no Execution | Extend binding or parallel execution-scoped context | **TRANSFORM** |
| **RuntimeEvent identity** | Task/Run/Attempt + `EventId` + optional `parent_event_id`; optional `node_id` | `runtime_event.py:89-110` | No `ExecutionId`; `node_id` ≠ execution; `parent_event_id` is event causality only | Add `execution_id` where target contract requires full spine (`TaskId`/`RunId`/`AttemptId`/`ExecutionId`/`EventId`); keep `parent_event_id` for event causality; do **not** mandate `parent_execution_id` on every event — canonical execution lineage stays on Execution Tree | **TRANSFORM** |
| **Background bootstrap** | `BackgroundExecutionIdentity` | `background_execution/bootstrap.py:29-85` | Mints new Attempt every resolve | Same-work redelivery must preserve AttemptId | **TRANSFORM** |
| **Checkpoint identity** | `RuntimeCheckpoint.run_id/attempt_id` | `runtime_checkpoint.py:61-75` | No Execution Tree fields | Extend checkpoint with tree snapshot | **TRANSFORM** |
| **DIAG refs** | `RuntimeExecutionRef` | `observability/causal_evidence.py:56-64` | Stops at Attempt | Add `execution_id` when contract exists | **TRANSFORM** |

### Authority vs propagation vs persistence

| Layer | Current | Target owner |
|-------|---------|--------------|
| **Minting authority** | Task (TaskId), Runner/Nexus (Run/Attempt), Event factory (EventId) | Execution lifecycle layer mints ExecutionId; OBS/DIAG never mint |
| **Propagation** | `ActiveExecutionIdentity` ContextVar; task metadata; graph node context | Execution boundary binds active Execution; child inheritance |
| **Persistence** | Event store, checkpoint, background identity persistence, causal evidence | Same stores + execution tree projection |

**Explicit ExecutionId absence locations:** `execution_identity.py`, `RuntimeEvent`, `RuntimeExecutionRef`, `RuntimeExecutionContext`, `RuntimeCheckpoint`, `BackgroundExecutionIdentity`, all event emitters in `GraphExecutor`, `UAEPExecutor`, `NexusLoop` orchestration runners.

---

## 3. Execution record / envelope

### CURRENT carriers

| Structure | Role | Agent-specific? | Future universal envelope? | Disposition |
|-----------|------|-----------------|---------------------------|-------------|
| `Task` / `TaskResult` | Harness work unit + product-facing outcome | Partially — carries orchestration metadata, agent ids | **No** — remains task/orchestration intake artifact | **KEEP_AND_REWIRE** |
| `RuntimeExecutionContext` | UAEP per-step context | **Yes** — `agent_id`, contract, tool gateway | **No** — agent strategy context only | **KEEP** (agent scope) |
| `RuntimeRequest` | Nexus/agent request payload | Yes | Internal request DTO; not universal envelope | **KEEP** (internal) |
| `AgentExecutionResult` | Graph node outcome | **Yes** | **No** — must not become universal result | **DEPRECATE** as universal |
| `BackgroundExecutionIdentity` | Worker canonical ids | No | Partial — needs ExecutionId + envelope | **TRANSFORM** |
| `RuntimeState` | Nexus engine mutable state | Yes — planner/tool loop | Strategy-private | **KEEP** (Nexus/agentic internals) |

**Gap:** No neutral typed execution request/context/result triple.

**Transformation:** Introduce internal contracts (names not frozen): execution request (what + output_type + capabilities), execution context (identity + authority + budget allowance refs), execution result (status + typed output + strategy-private annex). Bridge from `Task` during migration.

**Anti-patterns to reject:** `dict[str, Any]` envelopes, `getattr`/`setattr` discovery, promoting `RuntimeExecutionContext` to universal.

---

## 4. Execution boundary

### TARGET

Canonical boundary coordinates: identity/lifecycle, authority/governance, budget, evidence/events, cancellation, checkpoint/reliability hooks, strategy executor dispatch — **without absorbing subsystem implementations**.

### CURRENT partial boundaries

| Component | Fragments provided | Missing | Disposition |
|-----------|-------------------|---------|-------------|
| `NexusLoop.handle_task` | Identity bind, policy engine, event bus, interrupt handler, budget config, checkpoint bridge | Not strategy-neutral; owns orchestration lifecycle | **SPLIT_RESPONSIBILITY** |
| `GraphExecutor.execute` | Node execution, delegation authority, events, retry, cancellation coordinator | Agent-centric; direct `AgentEngine`; no child Execution admission | **SPLIT_RESPONSIBILITY** |
| `UAEPExecutor.execute` | Step loop, governance emit, checkpoint cursor, events | Agent-only; not universal | **KEEP** (agentic internals) |
| `WorkerRuntime` + `admit_background_execution_handler` | Causal admission, identity bootstrap | No execution boundary re-entry; Attempt drift | **KEEP_AND_REWIRE** |
| `HarnessKernel.execute_step` | ACP step governance boundary | ACP-only | **KEEP** |

**God-object risk:** Do **not** expand `NexusLoop` or a new `ExecutionRuntime` class to own PolicyEngine, BudgetEnforcer, EventBus, CheckpointStore implementations. Boundary = **coordination interfaces** + mandatory guarantee hooks.

**Coordination vs ownership:**

| Concern | Boundary coordinates | Subsystem owns |
|---------|---------------------|----------------|
| Governance | Admission + evaluation points | `PolicyEngine`, `RuntimePolicyEngine`, governance service |
| Budget | Allowance reservation/consumption gates | `RunBudget`, `BudgetEnforcer` |
| Observability | Event emission contract | `RuntimeEventBus`, persistence |
| Checkpoint | Recovery handoff | `RuntimeCheckpoint`, UAEP cursors |
| Strategy | Resolver → executor dispatch | Inference/Agentic/Orchestration executors |

---

## 5. Strategy resolution

### TARGET

Deterministic `StrategyResolver`: reads explicit requirements/capabilities; does not invent topology; no magic LLM routing; does not own execution.

### CURRENT routing signals

| Signal | Source | Reusable? | Notes |
|--------|--------|-----------|-------|
| Task capability / `agent_id` | `Task.context` | Partial | Agent selection ≠ strategy |
| `NexusTaskClassifier` + planner | `NexusLoop` intake/planning | Orchestration hints only | Currently drives graph creation |
| `use_nexus_loop` app settings | Tier-3 settings | **Legacy coupling** | e.g. `ResearchBackendSettings` — not target |
| `ExecutionMode` | Harness env | Governance posture only | Not strategy |
| `ToolInvocationPlan.use_tools` | Tool runtime | Agentic/inference capability | Valid capability signal |
| Graph plan existence | Nexus planning | Orchestration signal | Must not auto-invent topology |

**Current components:** No `StrategyResolver` type. `UnifiedTaskRunner` implicitly selects orchestration (Nexus). `AgentEngine` invoked directly from `GraphExecutor`.

**Transformation:** New resolver module reading typed execution request capabilities → `inference` | `agentic` | `orchestration`. Rewire `UnifiedTaskRunner` to boundary + resolver. **Disposition:** new resolver **TRANSFORM**; `UnifiedTaskRunner` **KEEP_AND_REWIRE**; classifier/planner **KEEP** as orchestration internals.

---

## 6. Inference executor

### TARGET

Boundary → model invocation → canonical result; **no Nexus, no AgentEngine**.

### CURRENT paths

| Path | Viable? | Evidence |
|------|---------|----------|
| LLM adapters `generate` / `generate_structured` / `stream_messages` | **Yes** — core invocation | `intergrax/llm_adapters/contracts/llm_adapter.py` |
| `RoutingEvaluatingAdapter` | Partial — routing wrapper | `applications/_shared/routing_evaluating_adapter.py` |
| Nexus planner/model calls | **No** — coupled to orchestration | Nexus planning runners |
| UAEP | **No** — agent contract | `uaep.py` |

**Smallest proof path:** New `InferenceExecutor` behind boundary calling adapter registry with governance/budget/event hooks; reuse adapter streaming ABI.

**Gap:** No standalone inference executor; no boundary-gated direct model path.

**Disposition:** New inference executor **TRANSFORM**; LLM adapters **KEEP**.

---

## 7. Agent executor

### TARGET

Boundary → AgentExecutor → AgentEngine → UAEP.

### CURRENT stack

| Layer | File | Role | Disposition |
|-------|------|------|-------------|
| `GraphExecutor` | `graph_executor.py` | Outer facade calling `AgentEngine` per node | **SPLIT_RESPONSIBILITY** — orchestration node runner vs agentic executor |
| `AgentEngine` | `agent_engine.py` | Routes UAEP/ACP/legacy | **KEEP** |
| `UAEPExecutor` | `uaep.py` | Step loop, governance, checkpoint | **KEEP** |
| `AgentRuntime` | `agents/authoring/step_loop.py` | Authoring helper | **KEEP** (Tier-2) |
| `HarnessKernel` | `step_kernel.py` | ACP `on_next_step` boundary | **KEEP** (ACP only) |
| Nexus→agent bridge | `graph_executor.py` `_execute_node` path | Direct engine invoke | **KEEP_AND_REWIRE** → child Execution |

**Duplicate session loops:** UAEP step loop (`uaep.py`) vs ACP `HarnessKernel`/`on_next_step` vs `run_bounded_tool_loop` — **not duplicates of same responsibility** but overlapping iteration surfaces. Target: UAEP owns agent progression; ToolInvocationPattern owns tool mechanics; ACP remains parallel session protocol.

---

## 8. Iterative tool loop

### TARGET

One `ExecutionId`; LLM → ToolRuntime → ToolResult → CE → LLM.

### CURRENT paths

| Path | Mechanism | Evidence |
|------|-----------|----------|
| Bounded ReAct | `run_bounded_tool_loop` + `ToolInvocationPattern` | `tool_loop.py:361`, `tool_invocation_pattern.py` |
| UAEP tool steps | Agent step tools via `RuntimeExecutionContext` gateway | `uaep.py`, `runtime_execution_context.py` |
| Nexus `ToolRuntime` | Capability invocation / plan context | `tool_runtime.py` |
| Planner integration | `ToolPlannerProtocol` inside tool loop | `tool_loop.py` |

**Overlap:** `run_bounded_tool_loop` is shared mechanics; UAEP may call tools via step contract separately. Risk: two lifecycle owners if both mint independent iteration identity.

**Transformation:** Bind tool loop to active ExecutionId; UAEP invokes pattern for tool iterations; eliminate parallel generic loop engines (**no** `ToolLoopRuntime`).

**Disposition:** `ToolInvocationPattern` **KEEP**; `run_bounded_tool_loop` **KEEP**; UAEP **KEEP**; wiring **KEEP_AND_REWIRE**.

---

## 9. Tool selection / planning / invocation

| Layer | Component | Disposition | Transformation |
|-------|-----------|-------------|----------------|
| Selection (standard) | Tool registry + allowlists | **KEEP** | Wire through boundary authority |
| Selection (semantic) | Semantic selector modules in `intergrax/tools/` | **KEEP** | Narrow scope only — verify no lifecycle ownership |
| Selection (hierarchical) | Planner-driven narrowing | **KEEP** | |
| Planning | `ToolPlannerProtocol`, declarative planner | **KEEP** | |
| Execution | `ToolRuntime` | **KEEP** | |
| Invocation | `RuntimeToolInvoker` | **KEEP** | |
| Pattern | `ToolInvocationPattern`, `resolve_invocation_pattern` | **KEEP** | |
| Parallel | `run_bounded_tool_loop` thread pool path | **KEEP** | Budget gates per parallel batch |

**Evidence:** `tool_runtime.py`, `invoker.py`, `tool_invocation_pattern.py`, `tool_loop.py`.

---

## 10. Context feedback

### TARGET

ToolResult → typed context fragment → Context Engineering → bounded next model input.

### CURRENT

| Component | Role | Disposition |
|-----------|------|-------------|
| `ContextCompiler` | Token budget + candidate classification | **KEEP** |
| `ContextEngine` (protocol) | CE orchestration | **KEEP** |
| `fragment_bridge` / CE tags | Tool output → candidates | **KEEP_AND_REWIRE** |
| `ContextManager` (Nexus) | Graph-level context assembly | **KEEP** (orchestration) |
| `AgentStepContext` | ACP step context | **KEEP** (ACP) |
| Direct prompt concat | Legacy heuristics in compiler | **DEPRECATE** paths using string heuristics over CE fragments |

**Note:** `ToolOutputContextProvider` is a **target CE role**; closest current implementation is `fragment_bridge` + `ContextCompiler.classify_candidates` (`context_compiler.py:53-76`).

**Gap:** Tool results not consistently routed through CE fragments on all paths.

**CE must not own Execution lifecycle.**

---

## 11. Memory

| Interaction | Current | Target | Disposition |
|-------------|---------|--------|-------------|
| Iteration/session state | `RuntimeState`, UAEP cursors, task metadata | Execution-scoped ephemeral | **KEEP** / **TRANSFORM** checkpoint |
| CE fragments | Context assembly | Temporary until model call | **KEEP** |
| Durable writes | `MemoryView` on `RuntimeExecutionContext`, `MEMORY_WRITE` events | Explicit policy-gated writes only | **KEEP_AND_REWIRE** |
| Recall | Memory providers via CE | On-demand recall | **KEEP** |
| Hidden scratchpad | Tool output auto-persist risk on some paths | **Reject** — tool feedback ≠ durable memory | Audit + rewire |

**Evidence:** `runtime_execution_context.py:54-65`, `MEMORY.md` UE-DOC-0.8 alignment.

---

## 12. Nexus

### TARGET

Nexus owns **what executes next**; requests child Executions through canonical boundary.

### CURRENT violations

| Location | Violation | Migration |
|----------|-----------|-----------|
| `GraphExecutor` → `AgentEngine` direct | Bypasses Execution boundary; `AgentExecutionResult` at node boundary | Replace with child Execution request |
| `NexusLoop.handle_task` | Owns full Run lifecycle + identity | Retain orchestration phases; delegate identity/tree to boundary |
| `plan_to_execution_graph` | Topology OK | **KEEP** — definition plane |
| Node scheduling | `GraphExecutor.execute` batches | **KEEP_AND_REWIRE** — schedule child Executions |
| Fan-out/merge | Parallel batches + `MergeStrategy` | **KEEP** — admission per branch Execution |
| Failure handling | `RetryEngine` at node level | Map to retry taxonomy (§17) |

**Disposition:** `NexusLoop` **SPLIT_RESPONSIBILITY**; `GraphExecutor` **TRANSFORM**; `AgentRouter` **KEEP** (agent pick within node, not strategy).

---

## 13. Orchestration definition

| Current | Target | Disposition |
|---------|--------|-------------|
| `NexusPlan`, `plan_to_execution_graph`, `ExecutionGraph` / `ExecutionNode` | `OrchestrationDefinition` semantics | **KEEP** — static topology |
| `node_id` on graph nodes | `NodeId` (definition) | **KEEP** |
| Runtime state on graph (`ExecutionNodeStatus`, outputs) | Execution Tree (runtime) | **SPLIT** — move runtime tree to Execution layer |
| Agent assumptions in nodes | Node may reference agent definition | **KEEP** with Node≠Execution separation |

**No second graph engine.**

---

## 14. Execution tree

### CURRENT hierarchy concepts

| Concept | Classification | Evidence |
|---------|----------------|----------|
| `ExecutionGraph` / node runs | **Topology** (definition) | `execution_graph.py` |
| `AgentExecutionResult` per node | **Local step identity** (agent run) | `agent_execution_result.py` |
| `graph_node_id` in metadata/checkpoint | Bridge field | `runtime_checkpoint.py` |
| `HANDOFF_*` / delegation metadata | **Future Execution relation** | `graph_executor.py`, delegation contracts |
| `Task.run_id` | Run scope | `task.py` |
| Broker `TaskRequest` ids | **Transport** — not RunId | `bootstrap.py` docstring |

**Transformation:** Single canonical tree from `ExecutionId` + `parent_execution_id`. No competing trees in DIAG or Nexus private ledgers.

---

## 15. Governance / authority

| Entry point | Current | Target | Disposition |
|-------------|---------|--------|-------------|
| `PolicyEngine` / `RuntimePolicyEngine` | Nexus + interrupt handler | Execution admission + inner points | **KEEP** |
| `EffectiveDelegationAuthority` | Graph node metadata | Child ≤ parent on Execution tree | **TRANSFORM** |
| `GovernanceResolution` in UAEP | Per-step | Per agentic Execution steps | **KEEP** |
| Tool authorization | `RuntimeToolInvoker` + policy | Tool consume through boundary | **KEEP_AND_REWIRE** |
| HITL | `HumanApprovalHookCoordinator`, `NexusHitlRunner` | Governance-owned pause/resume | **KEEP_AND_REWIRE** |

**StrategyResolver and Nexus must not become policy owners.**

---

## 16. Budget

| Component | Current | Gap | Disposition |
|-----------|---------|-----|-------------|
| `RunBudget` | Per-run limits dataclass | No hierarchical execution allowances | **TRANSFORM** |
| `BudgetEnforcer` / `budget_ticks` | Mid-run enforcement on `RuntimeState` | No child reservation/release | **TRANSFORM** |
| LLM cost checks | Adapter tracking / tenant scope | Not execution-tree scoped | **KEEP_AND_REWIRE** |
| Nexus `run_budget` config | Passed into NexusLoop | Parallel fan-out overcommit risk | **TRANSFORM** |
| Tool pre-invoke | `enforce_tool_call_budget` | Same | **KEEP** |

**No separate executor-specific ledgers.**

---

## 17. Retry / HITL / cancellation

| Mechanism | Location | Retry class | Issue | Disposition |
|-----------|----------|-------------|-------|-------------|
| Provider/local LLM retry | Adapters/failover | A local | OK if no Attempt mint | **KEEP** |
| `RetryEngine` (graph) | `retry_engine.py` | B/C mixed | May blur Attempt boundaries | **TRANSFORM** |
| `transition_active_execution_identity` | `execution_identity.py:94` | C if whole-Run | Used without taxonomy guard | **TRANSFORM** |
| `bootstrap_background_execution` | `bootstrap.py:84` | E redelivery | **Mints new Attempt every delivery** — debt | **TRANSFORM** |
| HITL pause/resume | `NexusHitlRunner`, UAEP checkpoint | D | Identity preserved at Attempt | **KEEP_AND_REWIRE** |
| `CancellationCoordinator` | `graph_executor.py` imports | Tree cancel | Not execution-tree aware yet | **TRANSFORM** |
| `ExecutionInterruptHandler` | `interrupts/handler.py` | D / policy | | **KEEP** |

---

## 18. Checkpoint

| Field/area | Stays | Extend | Disposition |
|------------|-------|--------|-------------|
| `run_id`, `attempt_id` on `RuntimeCheckpoint` | Yes | Authoritative via `TaskCheckpoint.runtime` | **KEEP** |
| UAEP cursors | Yes | | **KEEP** |
| Graph snapshot / node_states | Yes | Map to Execution Tree snapshot | **TRANSFORM** |
| Execution Tree | Missing | Add root + per-execution state | **TRANSFORM** |
| Budget reservations | Missing | Add reconciliation fields | **TRANSFORM** |
| HITL pending | Yes | | **KEEP** |

**Checkpoint is not identity owner** — carries identity, does not mint ExecutionId.

---

## 19. Observability

| Area | Current | Target | Disposition |
|------|---------|--------|-------------|
| `RuntimeEvent` factory paths | GraphExecutor, UAEP, Nexus runners, middleware | All carry `execution_id` where contract requires full identity spine | **TRANSFORM** |
| `RuntimeEventBus` | Central emit | Unchanged role | **KEEP** |
| Event persistence | `RuntimeEventPersistence` | | **KEEP** |
| Mandatory vs telemetry | `event_category`, `ops_hint` | Preserve distinction | **KEEP** |

**RuntimeEvent identity vs lineage (frozen):** `RuntimeEvent` must structurally carry `TaskId`, `RunId`, `AttemptId`, `ExecutionId`, and `EventId` where the target contract requires all five. `parent_event_id` expresses **event causality** when applicable. Canonical execution lineage (`parent_execution_id`) is owned by the **Execution Tree** / execution lifecycle layer — it may be projected into evidence or event contracts only when explicitly justified; it must **not** become a second canonical lineage authority on every `RuntimeEvent`.

**Do not propose per-token RuntimeEvent streaming.**

---

## 20. DIAG

Post **DIAG-5D** / **DIAG-5D-R1** (revalidated at UE-DOC-0.9R1 baseline).

### Ownership (frozen)

| Layer | Role |
|-------|------|
| Execution Runtime | Owns canonical `ExecutionId` + Execution Tree (`parent_execution_id` on Execution records) |
| Observability | Records/persists canonical execution evidence |
| DIAG reconstruction | Interprets those facts; **never** mints `ExecutionId` |
| `ProblemGroupingEngine` | Groups diagnostic assessments into ephemeral grouping hypotheses |
| `ProblemLifecycleEngine` | Reconciles validated hypotheses into stable derived `Problem` records |
| `ProblemPersistence` | Durable store for derived Problems — not execution truth |

**Forbidden substitutions:** `ProblemId`, grouping signature, reconciliation key, and `ProblemGroupingSubjectRef` must **not** become substitutes for `ExecutionId` or canonical execution lineage.

### Component mapping

| Structure | Current (HEAD) | Unified-execution need | Disposition |
|-----------|----------------|------------------------|-------------|
| `RuntimeExecutionRef` | `task_id`, `run_id`, `attempt_id`, `tenant_id` | `execution_id` when contract exists | **TRANSFORM** |
| `ProblemGroupingSubjectRef` | `tenant_id`, `task_id`, `run_id` | Execution-aware diagnostic subject identity/provenance (e.g. `ExecutionId` when the diagnostic subject is one Execution) | **TRANSFORM** |
| `ProblemGroupingEngine` | Consumes typed `DiagnosticAssessment`-derived subject data; proposes ephemeral candidates; does not own canonical runtime execution identity | Consume richer Execution-aware typed refs/provenance when `ExecutionId` is canonical | **KEEP_AND_REWIRE** |
| `ProblemLifecycleEngine` | Reconciles validated grouping hypotheses into stable derived `Problem` identity; does not re-run reconstruction; does not own RuntimeEvent/Execution truth | Preserve derived Problem lifecycle role; subject ref shape flows from `ProblemGroupingSubjectRef` transform | **KEEP** |
| `ProblemPersistence` / deterministic reconciliation | Derived Problem durability; occurrence-derived timestamps (DIAG-5D-R1) | No Execution identity ownership | **KEEP** |
| `PlatformCausalEvidence` | Links transport → execution refs | Wire through `execution_id` on refs | **KEEP_AND_REWIRE** |

**`ProblemId`:** derived diagnostic identity (stable recurrence bucket) — **not** `ExecutionId`, **not** root cause, **not** canonical execution lineage.

**DIAG consumes migrated identity; never mints ExecutionId.**

**Evidence:** `intergrax/runtime/observability/causal_evidence.py`, `intergrax/runtime/diagnostics/problem_grouping.py`, `intergrax/runtime/diagnostics/problem_lifecycle.py`, `intergrax/runtime/diagnostics/problem_persistence.py`, `intergrax/runtime/diagnostics/deterministic_problem_reconciliation.py`.

---

## 21. Background / distributed

| Component | Current | Debt | Transformation |
|-----------|---------|------|----------------|
| `BackgroundExecutionIdentityPersistence` | Stable Task/Run per transport ref | | **KEEP** |
| `resolve_background_execution` | New Attempt every call | **UE-DOC-0.7 debt** | Preserve Attempt on same-work redelivery |
| `WorkerRuntime` | Dispatches handlers | No boundary re-entry | Re-enter boundary with envelope |
| `BrokerWorkerBase` | Transport adapter | | **KEEP** |
| `admit_background_execution_handler` | Causal admission | | **KEEP_AND_REWIRE** |
| Celery dispatcher | Provider-specific | Transport only | **KEEP** |

**Target:** same-work redelivery preserves Task/Run/Attempt/**ExecutionId**.

---

## 22. Streaming

| Component | Current | Gap | Disposition |
|-----------|---------|-----|-------------|
| `LLMStreamEvent` | Adapter ABI `PARTIAL`/`FINAL` | Tool-call completion gate incomplete on some paths | **KEEP** (adapters) |
| `stream_messages` / `stream_with_tools` | Provider implementations | Governance before user release | **KEEP_AND_REWIRE** |
| User delivery | Tier-3 / harness routes | Output policy application | **TRANSFORM** |
| Budget on stream | Partial via adapter tracking | Reservation/reconciliation on FINAL | **TRANSFORM** |
| Cancellation/backpressure | Provider-dependent | Uniform boundary hooks | **KEEP_AND_REWIRE** |

**No `StreamingRuntime`.**

---

## 23. Structured output

| Path | Current | Target | Disposition |
|------|---------|--------|-------------|
| `generate_structured` | LLM adapters | `output_type` validation at boundary | **KEEP** |
| Manual JSON parse | Legacy agent paths | Remove | **DEPRECATE** |
| `AgentExecutionResult.structured_data` | Agent-shaped | Strategy-private annex | **DEPRECATE** as universal |
| `TaskResult` metadata | Product projections | Application layer | **KEEP** (Tier-3) |

---

## 24. Result model

| Type | Belongs to | Disposition |
|------|------------|-------------|
| Neutral execution result (new) | Execution boundary return | **TRANSFORM** (new) |
| `AgentExecutionResult` | Agentic strategy / graph node | **KEEP** — not universal |
| `TaskResult` | Task/harness orchestration outcome | **KEEP** — application projection source |
| `ToolInvocationResult` | Tool pattern | **KEEP** |
| LLM adapter results | Inference strategy private | **KEEP** |

**Strong typed contracts; no universal result dict.**

---

## 25. Component disposition table

| Component | Current role | Current caller(s) | Target owner | Disposition | Required transformation | Dependency | Removal condition | Risk |
|-----------|--------------|-------------------|--------------|-------------|-------------------------|------------|-------------------|------|
| `UnifiedTaskRunner` | Nexus-only task entry | Apps, harness, workers | Execution boundary | KEEP_AND_REWIRE | Delegate to boundary + resolver | ExecutionId, boundary | All callers use `execute()` | High |
| `NexusLoop` | Orchestration lifecycle | UnifiedTaskRunner | Orchestration strategy | SPLIT_RESPONSIBILITY | Shed identity/lifecycle ownership | Boundary | Nexus-only internal | High |
| `GraphExecutor` | Agent graph runner | Nexus graph runner | Orchestration executor | TRANSFORM | Child Execution admission | ExecutionId, boundary | Direct AgentEngine removed | High |
| `AgentEngine` | Agent dispatch | GraphExecutor, tests | Agentic executor | KEEP_AND_REWIRE | Only via boundary | Boundary | None | Medium |
| `UAEPExecutor` | UAEP step loop | AgentEngine | Agentic internals | KEEP | ExecutionId on events | ExecutionId | None | Medium |
| `AgentRouter` | Agent selection | GraphExecutor | Orchestration (node) | KEEP | Node-level only | None | None | Low |
| `run_bounded_tool_loop` | Tool iteration mechanics | UAEP, Nexus tools | ToolInvocationPattern | KEEP | Bind ExecutionId | ExecutionId | None | Medium |
| `ToolRuntime` / `RuntimeToolInvoker` | Tool exec | Tool loop, UAEP | Tools domain | KEEP | Authority via boundary | Governance | None | Low |
| `RuntimeExecutionContext` | UAEP step ctx | UAEP | Agent strategy | KEEP | Add execution_id field | ExecutionId | None | Low |
| `RuntimeEvent` | Audit spine | All runtime | Observability | TRANSFORM | Add `execution_id`; full spine where required; `parent_event_id` = event causality; lineage via Execution Tree — not mandatory `parent_execution_id` on every event | ExecutionId contract | None | Medium |
| `ActiveExecutionIdentity` | Run/Attempt CV | Nexus, GraphExecutor | Execution lifecycle | TRANSFORM | Execution scope | ExecutionId | None | Medium |
| `RunBudget` / `BudgetEnforcer` | Run limits | Nexus, RuntimeState | Budget domain | TRANSFORM | Hierarchical reservations | Execution tree | None | High |
| `RuntimeCheckpoint` | Recovery | Nexus, UAEP, long-running | Checkpoint owner | TRANSFORM | Tree-aware state | ExecutionId | None | High |
| `bootstrap_background_execution` | Worker identity | WorkerRuntime | Background + boundary | TRANSFORM | Attempt preservation | Execution envelope | Redelivery tests pass | High |
| `RuntimeExecutionRef` | DIAG transport ref | Causal evidence, DIAG | Observability | TRANSFORM | Add `execution_id` after canonical identity contract | ExecutionId | None | Medium |
| `ProblemGroupingSubjectRef` | Diagnostic subject key | `ProblemGroupingEngine`, `ProblemLifecycleEngine` | DIAG (derived) | TRANSFORM | Execution-aware subject identity/provenance | ExecutionId, `ProblemGroupingEngine` | None | Medium |
| `ProblemGroupingEngine` | Ephemeral grouping hypotheses | DIAG reconstruction pipeline | DIAG | KEEP_AND_REWIRE | Consume Execution-aware typed refs; no Execution identity ownership | ExecutionId on refs | None | Medium |
| `ProblemLifecycleEngine` | Stable derived `Problem` reconciliation | Post-grouping pipeline | DIAG | KEEP | Preserve lifecycle role; no RuntimeEvent/Execution truth ownership | `ProblemGroupingEngine` | None | Low |
| `ProblemPersistence` | Derived Problem durability | `ProblemLifecycleEngine` | DIAG | KEEP | No Execution identity ownership | DIAG-5D contract | None | Low |
| `Task` / `TaskResult` | Harness DTO | Apps | Task intake (not Execution) | KEEP_AND_REWIRE | Map to execution request | Boundary | Optional later | Medium |
| `AgentExecutionResult` | Node result | GraphExecutor | Agentic private | DEPRECATE | Stop universal promotion | Neutral result | Consumers migrated | Medium |
| `HarnessKernel` | ACP steps | ACP agents | ACP domain | KEEP | None | None | None | Low |
| `PolicyEngine` | Policy eval | Nexus, interrupts | Governance | KEEP | Admission points | Boundary | None | Low |
| `ContextCompiler` | CE budget | Nexus context | Context Engineering | KEEP | Tool fragment path | CE bridge | None | Low |
| LLM adapters | Model I/O | UAEP, planners, inference | LLM Adapters | KEEP | Behind inference executor | Boundary | None | Low |

---

## 26. Transformation waves

Validated against repository evidence at HEAD pin.

| Wave | Scope | Key deliverables | Prerequisites |
|------|-------|------------------|---------------|
| **0 — Contract foundation** | Types only | `ExecutionId`, `parent_execution_id`, neutral request/result protocols; `RuntimeEvent` field plan | None |
| **1 — Boundary skeleton** | Coordination | Execution coordinator; guarantee interfaces; compat shims | Wave 0 |
| **2 — Direct inference** | Proof A | Inference executor; adapter path without Nexus | Wave 1 |
| **3 — Agentic executor** | Proof B | AgentEngine/UAEP behind boundary; loop ownership | Wave 1 |
| **4 — Tools/context/stream** | Proof B detail | ToolResult→CE; streaming FINAL gate; tool-call completion | Waves 2–3 |
| **5 — Nexus child Execution** | Proof C | GraphExecutor requests child Executions; remove direct engine | Waves 0–1, 3 |
| **6 — Governance/budget hierarchy** | Proofs D/E | Child authority; reservations; parallel admission | Waves 0–1, 5 |
| **7 — Checkpoint/reliability** | Proofs F–H | Tree-aware recovery; retry taxonomy | Waves 0, 5–6 |
| **8 — Distributed execution** | Proof I | Envelope/reference; redelivery identity; worker re-entry | Waves 0–1, 7 |
| **9 — OBS/DIAG closure** | Proof J | ExecutionId on events/refs; reconstruction | Waves 0, 8 |
| **10 — Legacy removal** | Cleanup | Remove bypass paths, universal wrappers | Waves 2–9 |
| **11 — Architectural proof** | Scenarios A–J | Integration proof suite per UEA §28 | Wave 10 |

---

## 27. Migration dependency DAG

```text
ExecutionId (+ parent_execution_id)
   ↓
Execution Boundary (coordinator)
   ├→ StrategyResolver
   ├→ InferenceExecutor → LLM Adapters
   ├→ AgenticExecutor → AgentEngine → UAEP
   │     ├→ ToolInvocationPattern / run_bounded_tool_loop
   │     ├→ ToolRuntime / RuntimeToolInvoker
   │     └→ ContextCompiler / CE fragment bridge
   └→ OrchestrationExecutor → NexusLoop → GraphExecutor
           ↓
        child Execution (re-enters Boundary)

ExecutionId
   ├→ RuntimeEvent (+ emitters)
   ├→ RuntimeCheckpoint
   ├→ BackgroundExecutionIdentity / envelope
   └→ RuntimeExecutionRef / DIAG

Boundary
   ├→ Governance (PolicyEngine admission)
   ├→ Budget (RunBudget reservations)
   ├→ Observability (event contract)
   ├→ Cancellation / InterruptHandler
   └→ Checkpoint handoff

Independent tracks (can follow early):
   - Streaming output policy (after inference path exists)
   - DIAG algorithm updates (after RuntimeEvent carries ExecutionId)
   - Provider stream parity (non-blocking)
```

---

## 28. Staging independence

| Change | Must precede | Can follow | Independent |
|--------|--------------|------------|-------------|
| `ExecutionId` type | Event/checkpoint/background migration | Consumer adoption | Yes — structural add first |
| `RuntimeEvent.execution_id` optional | Mandatory enforcement | DIAG grouping | Partially independent |
| Nexus child Execution | Boundary skeleton | DIAG rewrite | No |
| Background Attempt fix | Execution envelope | OBS full migration | After Wave 0–1 |
| Streaming governance | Inference/agentic path | Provider parity | Yes |
| Budget hierarchy | Execution tree | Tool loop | No |
| StrategyResolver | Boundary | Nexus rewrite | No |

---

## 29. No legacy-as-target (explicit rejections)

| Rejected pattern | Correct mapping |
|------------------|-----------------|
| Everything through Nexus because `UnifiedTaskRunner` already does | Resolver selects strategy; Nexus orchestration-only |
| `RuntimeExecutionContext` as universal envelope | Agent strategy context only |
| `GraphExecutor` as canonical AI engine | Orchestration node scheduler → child Executions |
| `AgentExecutionResult` as universal result | Strategy-private; neutral execution result |
| New Attempt per background delivery | Same-work redelivery preserves Attempt |
| `ToolInvocationPattern` as agent lifecycle engine | Mechanics only; UAEP owns progression |
| Observability minting missing ExecutionId | Execution lifecycle mints; OBS records |

---

## 30. No big-bang rewrite

Prefer: reuse, rewire, split ownership, typed bridges, gated removal. **Forbidden:** `NewExecutionFrameworkV2`, `StreamingRuntime`, `ToolLoopRuntime`, second event spine, second checkpoint system, second budget ledger.

---

## 31. Implementation slice candidates

High-level roadmap preserved; subdivided where evidence requires safe staging.

### UE-1 — Developer entry point

| Slice | Outcome | Scope | Prerequisites | Files/areas | Exclusions | Acceptance | Rollback |
|-------|---------|-------|---------------|-------------|------------|------------|----------|
| **UE-1A** | `ExecutionId` contracts | Types, validators, minting rules | None | `intergrax/contracts/` | Behavior cutover | Unit tests for identity | N/A |
| **UE-1B** | Boundary skeleton | Coordinator interfaces, compat shim | UE-1A | new `intergrax/runtime/execution/` | Strategy rewrite | Old callers still work | Feature flag |
| **UE-1C** | `execution.execute` facade | Public internal API | UE-1B | boundary module | Nexus removal | Callable from tests | Shim to UnifiedTaskRunner |

### UE-2 — Execution intent

| Slice | Outcome | Scope | Prerequisites | Exclusions | Acceptance |
|-------|---------|-------|---------------|------------|------------|
| **UE-2A** | Neutral execution request DTO | capabilities, output_type, payload | UE-1A | Task removal | Typed request tests |
| **UE-2B** | Task→request adapter | Bridge from harness Task | UE-2A | App route changes | Existing apps pass |

### UE-3 — Strategy resolution

| Slice | Outcome | Scope | Prerequisites | Exclusions | Acceptance |
|-------|---------|-------|---------------|------------|------------|
| **UE-3A** | `StrategyResolver` | Deterministic capability read | UE-2A | LLM routing | Unit matrix |
| **UE-3B** | Runner rewire | UnifiedTaskRunner → boundary | UE-1C, UE-3A | Delete Nexus | Orchestration still works |

### UE-4 — Shared boundary

| Slice | Outcome | Scope | Prerequisites | Acceptance |
|-------|---------|-------|---------------|------------|
| **UE-4A** | Admission hooks | governance/budget/event/cancel interfaces | UE-1B | Hook invocation tests |
| **UE-4B** | Active execution binding | ExecutionId in context | UE-1A | Propagation tests |

### UE-5 — Structured output

| Slice | Outcome | Scope | Prerequisites | Acceptance |
|-------|---------|-------|---------------|------------|
| **UE-5A** | Inference structured path | `output_type` validation | UE-2 inference | Schema conformance |
| **UE-5B** | Neutral result envelope | Typed output carrier | UE-5A | No AgentExecutionResult leak |

### UE-6 — Tools + autonomous execution

| Slice | Outcome | Scope | Prerequisites | Acceptance |
|-------|---------|-------|---------------|------------|
| **UE-6A** | Agentic behind boundary | AgentEngine path | UE-4 | UAEP tests green |
| **UE-6B** | Tool loop ExecutionId | `run_bounded_tool_loop` bind | UE-4B, UE-6A | Scenario B |
| **UE-6C** | CE tool feedback | fragment bridge | UE-6B | CE integration tests |

### UE-7 — Nexus boundary

| Slice | Outcome | Scope | Prerequisites | Acceptance |
|-------|---------|-------|---------------|------------|
| **UE-7A** | Child Execution API | GraphExecutor admission | UE-4, UE-1A | Scenario C unit |
| **UE-7B** | Remove direct AgentEngine child | GraphExecutor rewire | UE-7A | No direct engine in graph path |

### UE-8 — Enforcement

| Slice | Outcome | Scope | Prerequisites | Acceptance |
|-------|---------|-------|---------------|------------|
| **UE-8A** | Authority inheritance | delegation on Execution tree | UE-7 | Child ≤ parent tests |
| **UE-8B** | Budget reservations | hierarchical RunBudget | UE-7 | Parallel fan-out tests |

### UE-9 — Migration

| Slice | Outcome | Scope | Prerequisites | Acceptance |
|-------|---------|-------|---------------|------------|
| **UE-9A** | Background identity fix | Attempt on redelivery | UE-1A, envelope | Scenario I |
| **UE-9B** | RuntimeEvent migration | execution_id field | UE-1A | Event schema tests |
| **UE-9C** | Checkpoint tree | RuntimeCheckpoint extend | UE-7 | Resume tests |

### UE-10 — Architectural proof

| Slice | Outcome | Scope | Prerequisites | Acceptance |
|-------|---------|-------|---------------|------------|
| **UE-10** | Scenario suite A–J | Integration proofs per UEA §28 | UE-1..9 | Proof harness green |

---

## 32. Risk register

| Risk | Trigger | Mitigation | Safety gate |
|------|---------|------------|-------------|
| God-object Execution Boundary | Centralizing Policy/Budget/OBS impl | Coordination interfaces only | Architecture review per slice |
| Dual lifecycle during transition | Parallel old/new callers | Compat shims + feature flags | Caller inventory tests |
| Mixed old/new identity | Partial ExecutionId rollout | Optional field → required phased | Schema guard tests |
| Nexus direct-executor bypass | GraphExecutor unchanged | UE-7B enforcement | Static import audit |
| Budget double counting | Parallel fan-out | Hierarchical reservations | UE-8B load tests |
| Retry identity drift | `transition_retry` misuse | Taxonomy guards | REL unit matrix |
| Worker redelivery drift | `mint_attempt_id` in bootstrap | UE-9A | Redelivery integration |
| Event schema partial migration | Optional execution_id forever | Deprecation deadline | OBS conformance |
| Checkpoint incompatible state | Tree fields added ad hoc | Versioned checkpoint schema | Resume round-trip |
| Tool-loop duplicate lifecycle | UAEP + pattern both own loop | Clear ownership doc + code | UE-6B review |
| Streaming pre-governance disclosure | User stream before policy | Output release policy | UE-5/LLM gate tests |
| Application result breaking changes | TaskResult shape change | Projection layer in Tier-3 | App contract tests |
| Context path divergence | Direct concat survives | CE fragment enforcement | CE lint/tests |
| Authority expansion on child work | Nexus scheduling | Effective ≤ parent invariant | GOV tests |

---

## 33. Removal register

| Path | Why obsolete | Migrate first | Removal gate |
|------|--------------|---------------|--------------|
| `UnifiedTaskRunner` → Nexus hardwire | Strategy resolver required | `execution.execute` + UE-3B | Zero direct imports |
| `GraphExecutor` → `AgentEngine` direct | UEA-INV-021 | UE-7B child Execution | Graph path audit clean |
| `AgentExecutionResult` as harness return | Neutral result | UE-5B + app projections | No Tier-3 dependency |
| `transition_active_execution_identity` unguarded | Retry taxonomy | UE-7 + REL alignment | Call-site guards |
| New Attempt per background delivery | UEA-INV-011/§11 | UE-9A | Redelivery proof I |
| `use_nexus_loop` product settings as strategy | Not capability-based | UE-3A | Settings removed |
| Universal promotion of product metadata in core | SHM-FIX-C | App adapters | Audit clean |

---

## 34. Current implementation evidence index

Primary symbols verified at UE-DOC-0.9R1 baseline (`862c0dc57928c3af34329a836482df89fec786f4`):

| Domain | Symbols / paths |
|--------|-------------------|
| Identity | `intergrax/contracts/execution_identity.py` |
| Task entry | `intergrax/runtime/task/unified_task_runner.py`, `intergrax/runtime/task/task.py` |
| Nexus | `intergrax/runtime/nexus/nexus_loop.py`, `intergrax/runtime/nexus/execution/graph_executor.py`, `intergrax/runtime/nexus/agent_router.py` |
| Agent/UAEP | `intergrax/agents/agent_engine.py`, `intergrax/agents/uaep.py`, `intergrax/runtime/kernel/step_kernel.py` |
| Tools | `intergrax/runtime/nexus/tools/tool_loop.py`, `tool_runtime.py`, `invoker.py`, `tool_invocation_pattern.py` |
| Context | `intergrax/runtime/nexus/context/context_compiler.py`, `intergrax/context/orchestrator.py` |
| Budget | `intergrax/runtime/nexus/budget/budget_models.py`, `budget_ticks.py` |
| Governance | `intergrax/runtime/policy/policy_engine.py`, `runtime_policy_engine.py` |
| Events/OBS | `intergrax/runtime/events/runtime_event.py`, `event_bus.py` |
| DIAG | `intergrax/runtime/observability/causal_evidence.py`, `intergrax/runtime/diagnostics/problem_grouping.py`, `intergrax/runtime/diagnostics/problem_lifecycle.py`, `intergrax/runtime/diagnostics/problem_persistence.py`, `intergrax/runtime/diagnostics/deterministic_problem_reconciliation.py` |
| Checkpoint | `intergrax/runtime/long_running/runtime_checkpoint.py` |
| Background | `intergrax/runtime/background_execution/bootstrap.py`, `intergrax/background_tasks/worker_runtime.py`, `intergrax/queueing/providers/broker_worker_base.py` |
| Streaming | `intergrax/llm_adapters/contracts/stream_event.py`, adapter `stream_messages`/`stream_with_tools` |
| Results | `intergrax/contracts/agent_execution_result.py`, `intergrax/runtime/task/task.py` (`TaskResult`) |

---

## Acceptance self-check (UE-DOC-0.9R1)

| Question | Answer |
|----------|--------|
| Implementer can find current code per major target responsibility? | **YES** — §34 + per-section evidence |
| Disposition (keep/rewire/transform/split/remove) clear? | **YES** — §25 table + per-section |
| Prerequisites before change clear? | **YES** — waves §26, DAG §27, slice prereqs §31 |
| UE-1 slices derivable without new architecture session? | **YES** — §31 |
| One Execution Boundary, not new framework? | **YES** — §4, §29 |
| Nexus orchestration-only preserved? | **YES** — §12 |
| AgentEngine/UAEP preserved? | **YES** — §7 |
| ToolRuntime/selection/pattern preserved? | **YES** — §8–9 |
| LLM adapter streaming ABI preserved? | **YES** — §22 |
| Current violations mapped, not normalized? | **YES** — §12, §17, §29 |
| Topology vs tree vs transport distinguished? | **YES** — §13–14 |
| ExecutionId propagation path identified? | **YES** — §2, §19–21 |
| Redelivery/new-Attempt mapped as debt? | **YES** — §17, §21, §33 |
| Streaming governance gap before user release? | **YES** — §22 |
| Removals gated after consumer migration? | **YES** — §33 |
| Evidence baseline reports concurrent DIAG-5D during UE-DOC-0.9? | **YES** — header baseline |
| DIAG mapping reflects DIAG-5D / DIAG-5D-R1 (not stale grouping/lifecycle TRANSFORM)? | **YES** — §20, §25 |
| `ProblemLifecycleEngine` preserved as derived Problem owner, not runtime identity? | **YES** — §20 |
| `ProblemId` distinct from `ExecutionId`? | **YES** — §20 |
| DIAG remains derived interpretation only? | **YES** — §20 |
| `RuntimeEvent` requires `execution_id` at target? | **YES** — §2, §19 |
| Every `RuntimeEvent` requires `parent_execution_id`? | **NO** — §2, §19 |
| Canonical execution lineage owned by Execution Tree? | **YES** — §14, §19 |
| `parent_event_id` is event causality only? | **YES** — §2, §19 |

---

**Subordinate links:** [`UNIFIED_EXECUTION_ARCHITECTURE.md`](UNIFIED_EXECUTION_ARCHITECTURE.md) · [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) · [`../maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md`](../maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md)
