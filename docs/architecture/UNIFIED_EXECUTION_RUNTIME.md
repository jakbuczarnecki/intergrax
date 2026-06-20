# Unified Execution Runtime

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/UNIFIED_EXECUTION_RUNTIME.md`](../plan/UNIFIED_EXECUTION_RUNTIME.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 4–5, 8, 23–24  
**Audit instruction:** [`audit/UNIFIED_EXECUTION_RUNTIME.md`](../audit/UNIFIED_EXECUTION_RUNTIME.md)  
**Last updated:** 2026-06-19 — SEC-PLANES-EVOL follow-on register (enterprise hardening backlog)  
---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (UNIFIED_EXECUTION_RUNTIME canon).

- **Implement / audit default:** UAEP + PolicyEngine + RuntimeEvent spine (§42.1–§42.15). Extended: [`arch/UNIFIED_EXECUTION_RUNTIME_runtime_extended.md`](arch/UNIFIED_EXECUTION_RUNTIME_runtime_extended.md).
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/UNIFIED_EXECUTION_RUNTIME.md`](../plan/UNIFIED_EXECUTION_RUNTIME.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/UNIFIED_EXECUTION_RUNTIME.md`](../guides/audit_slices/UNIFIED_EXECUTION_RUNTIME.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---


## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`arch/UNIFIED_EXECUTION_RUNTIME_runtime_extended.md`](arch/UNIFIED_EXECUTION_RUNTIME_runtime_extended.md) | runtime extended |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

## 42.1 Runtime Event Model

Every meaningful runtime transition MUST emit a `RuntimeEvent`.

Events are the **primary audit and orchestration signal**. Hooks, observability, policy, and recovery subscribe to events — they MUST NOT rely on hidden callbacks inside agents.

**Event spine canon:** [`OBSERVABILITY.md`](OBSERVABILITY.md#observability-event-spine) — signal-plane boundaries, [event ownership rules](OBSERVABILITY.md#event-ownership-rules), [required correlation fields](OBSERVABILITY.md#required-correlation-fields), [Cursor review checklist](OBSERVABILITY.md#cursor-review-checklist).

### 42.1.1 RuntimeEvent Contract

```text
RuntimeEvent:
    event_id: str              # UUID, globally unique
    task_id: str               # Nexus task identifier
    run_id: str                # execution run (may span retries)
    node_id: str | null        # ExecutionGraph node, if applicable
    agent_id: str | null       # agent responsible for this event
    step_id: str | null        # AgentStep identifier, if applicable
    event_type: RuntimeEventType
    phase: ExecutionPhase      # see §42.31
    severity: EventSeverity    # DEBUG | INFO | WARNING | ERROR | CRITICAL
    payload: dict              # structured, schema-versioned
    timestamp: datetime        # UTC, ISO-8601
    correlation_id: str        # ties related events across agents/tools
    parent_event_id: str | null # causal chain
    schema_version: str         # e.g. "runtime_event.v1"
```

### 42.1.2 RuntimeEventType (minimum set)

```text
TASK_CREATED
TASK_CLASSIFIED
PLAN_CREATED | PLAN_UPDATED | PLAN_FAILED
SKILL_RESOLVED | SKILL_IMPORT_FAILED
AGENT_SELECTED
CONTEXT_BUILT | CONTEXT_ASSEMBLED | CONTEXT_TRIMMED | INGESTION_FAILED
STEP_STARTED | STEP_COMPLETED | STEP_FAILED
TOOL_REQUESTED | TOOL_COMPLETED | TOOL_DENIED | TOOL_FAILED
VALIDATION_STARTED | VALIDATION_PASSED | VALIDATION_FAILED
DECISION_EMITTED
INTERRUPT_REQUESTED | INTERRUPT_HANDLED | INTERRUPT_ESCALATED
HUMAN_APPROVAL_REQUESTED | HUMAN_APPROVAL_RECEIVED | HUMAN_APPROVAL_TIMEOUT
PAUSE_REQUESTED | PAUSED | RESUMED
RETRY_SCHEDULED | RETRY_STARTED
CANCELLATION_REQUESTED | CANCELLED
MEMORY_READ | MEMORY_WRITE
HANDOFF_INITIATED | HANDOFF_COMPLETED
TRACE_PERSISTED
TASK_COMPLETED | TASK_FAILED
```

### 42.1.3 Example Payload — STEP_COMPLETED

```json
{
  "event_id": "evt_8f3a2b1c-...",
  "task_id": "task_legal_review_001",
  "run_id": "run_20260527_001",
  "node_id": "node_legal_review",
  "agent_id": "legal",
  "step_id": "step_clause_analysis",
  "event_type": "STEP_COMPLETED",
  "phase": "STEP_EXECUTION",
  "severity": "INFO",
  "payload": {
    "step_name": "clause_analysis",
    "step_index": 3,
    "duration_ms": 4200,
    "artifacts": ["artifact_clause_flags.json"],
    "decision": "CONTINUE"
  },
  "timestamp": "2026-05-27T14:32:01.123Z",
  "correlation_id": "corr_task_legal_review_001",
  "parent_event_id": "evt_step_started_...",
  "schema_version": "runtime_event.v1"
}
```

### 42.1.4 Rules

- Every `AgentStep` MUST emit `STEP_STARTED` and `STEP_COMPLETED` or `STEP_FAILED`.
- Every `ToolRuntime.invoke` MUST emit `TOOL_*` events.
- Every `AgentDecision` MUST emit `DECISION_EMITTED`.
- Events MUST be persisted to trace storage (§42.24).
- Events MUST NOT contain secrets; redact at emission time.

### 42.1.5 Runtime event catalog (ops filters)

**Phase DX-5.7.** Canonical mapping lives in code: `intergrax.runtime.events.phase_coverage` (`EVENT_PHASE_COVERAGE`, `EVENT_OPS_FILTER_HINTS`). Gate: `test_all_runtime_event_types_have_execution_phase` and `test_all_runtime_event_types_have_ops_filter_hint`.

| `RuntimeEventType` | `ExecutionPhase` | Ops filter hint | Typical subscriber |
|--------------------|------------------|-----------------|-------------------|
| `TASK_CREATED` | `INTAKE` | `trace:intake` | TraceStore, metrics |
| `TASK_CLASSIFIED` | `CLASSIFICATION` | `trace:classification` | TraceStore |
| `PLAN_CREATED` | `PLANNING` | `ops:planning` | TraceStore, planner metrics |
| `PLAN_UPDATED` | `PLANNING` | `ops:planning` | TraceStore |
| `PLAN_FAILED` | `PLANNING` | `ops:alert` | Alerting, TraceStore |
| `AGENT_SELECTED` | `AGENT_SELECTION` | `trace:selection` | TraceStore |
| `SKILL_RESOLVED` | `AGENT_SELECTION` | `trace:skills` | TraceStore |
| `SKILL_IMPORT_FAILED` | `AGENT_SELECTION` | `ops:alert` | Alerting |
| `CONTEXT_BUILT` | `CONTEXT_BUILDING` | `trace:context` | TraceStore |
| `CONTEXT_ASSEMBLED` | `CONTEXT_BUILDING` | `trace:context` | TraceStore |
| `CONTEXT_TRIMMED` | `CONTEXT_BUILDING` | `trace:context` | TraceStore |
| `INGESTION_FAILED` | `CONTEXT_BUILDING` | `ops:alert` | Alerting |
| `MEMORY_READ` | `CONTEXT_BUILDING` | `ops:memory` | TraceStore |
| `MEMORY_WRITE` | `CONTEXT_BUILDING` | `ops:memory` | TraceStore |
| `STEP_STARTED` | `STEP_EXECUTION` | `trace:step` | TraceStore, UAEP |
| `STEP_COMPLETED` | `STEP_EXECUTION` | `trace:step` | TraceStore |
| `STEP_FAILED` | `STEP_EXECUTION` | `ops:alert` | Alerting, recovery |
| `TOOL_REQUESTED` | `STEP_EXECUTION` | `ops:tool_audit` | ToolRuntime audit |
| `TOOL_COMPLETED` | `STEP_EXECUTION` | `ops:tool_audit` | ToolRuntime audit |
| `TOOL_DENIED` | `STEP_EXECUTION` | `ops:alert` | PolicyEngine, alerting |
| `TOOL_FAILED` | `STEP_EXECUTION` | `ops:alert` | Alerting, recovery |
| `TASK_PROGRESS` | `STEP_EXECUTION` | `ops:progress` | Long-running UI, scheduler |
| `HANDOFF_INITIATED` | `STEP_EXECUTION` | `ops:handoff` | Graph executor |
| `HANDOFF_COMPLETED` | `STEP_EXECUTION` | `ops:handoff` | Graph executor |
| `VALIDATION_STARTED` | `VALIDATION` | `trace:validation` | TraceStore |
| `VALIDATION_PASSED` | `VALIDATION` | `trace:validation` | TraceStore |
| `VALIDATION_FAILED` | `VALIDATION` | `ops:alert` | Alerting |
| `DECISION_EMITTED` | `FINALIZATION` | `trace:decision` | TraceStore, hooks |
| `INTERRUPT_REQUESTED` | `INTERRUPT_HANDLING` | `ops:hitl` | HITL queue |
| `INTERRUPT_HANDLED` | `INTERRUPT_HANDLING` | `ops:hitl` | HITL queue |
| `INTERRUPT_ESCALATED` | `INTERRUPT_HANDLING` | `ops:alert` | Alerting |
| `RUNTIME_HANDLER_FAILED` | `INTERRUPT_HANDLING` | `ops:alert` | Alerting |
| `HUMAN_APPROVAL_REQUESTED` | `HUMAN_APPROVAL` | `ops:hitl` | HITL / PagerDuty |
| `HUMAN_APPROVAL_RECEIVED` | `HUMAN_APPROVAL` | `ops:hitl` | HITL |
| `HUMAN_APPROVAL_TIMEOUT` | `HUMAN_APPROVAL` | `ops:alert` | Alerting |
| `PAUSE_REQUESTED` | `HUMAN_APPROVAL` | `ops:hitl` | Scheduler |
| `PAUSED` | `HUMAN_APPROVAL` | `ops:hitl` | Scheduler |
| `RESUMED` | `HUMAN_APPROVAL` | `ops:hitl` | Scheduler |
| `RETRY_SCHEDULED` | `RETRY_HANDLING` | `ops:retry` | RetryEngine metrics |
| `RETRY_STARTED` | `RETRY_HANDLING` | `ops:retry` | RetryEngine metrics |
| `CANCELLATION_REQUESTED` | `COMPLETION` | `ops:completion` | TraceStore |
| `CANCELLED` | `COMPLETION` | `ops:completion` | TraceStore |
| `TASK_COMPLETED` | `COMPLETION` | `ops:completion` | SLO dashboards |
| `TASK_FAILED` | `COMPLETION` | `ops:alert` | Alerting, SLO burn |
| `TRACE_PERSISTED` | `TRACE_PERSISTENCE` | `trace:persistence` | TraceStore |

**Filter token legend:** `trace:*` — default observability scrape; `ops:alert` — page-worthy failures; `ops:hitl` — human-in-the-loop queues; `ops:tool_audit` — tool policy audits; `ops:completion` — terminal task outcomes; `ops:retry` — retry scheduler; `ops:planning` — planner failures/updates; `ops:handoff` — graph delegation; `ops:memory` — memory store access; `ops:progress` — checkpointed long runs.

### 42.1.6 Layered event identity (OBS-EVOL-9)

**Canon:** [`architecture/OBSERVABILITY.md`](../architecture/OBSERVABILITY.md) §4.4 · **ADR:** [`ADR-OBS-003`](../adr/entries/2026-06-17/ADR-OBS-003.md)

`RuntimeEvent` carries:

```text
event_type      # spine — platform lifecycle (~50 at publication)
event_kind      # semantic namespaced id (domain extensions)
event_category  # derived ops grouping
```

- **Tier-2/3** extend via `emit_domain_signal(kind, payload)` → spine `DOMAIN_SIGNAL`.
- **Platform** adds spine types only via ADR + `EventCatalog` entry.
- **Pre-release:** consolidate adaptive/capacity/hook enums to `platform.*` kinds (OBS-EVOL-9.7).

### 42.1.7 Event catalog governance

| Rule | Enforcement |
|------|-------------|
| New spine `RuntimeEventType` | ADR + `EventCatalogEntry` + emission gate |
| New domain signal | `event_kind` registry + extension `payload_schema_id` |
| Debug detail | `DiagnosticPayload` (Plane B) — not spine unless operator-facing |
| Bus subscription | Prefer `event_category` / `kind_prefix` over enum lists |

**Code (target):** `intergrax/runtime/events/event_catalog.py`, `signals.py`

---

## 42.2 Event Bus Architecture

The **Runtime Event Bus** is the Tier-1 pub/sub backbone for all runtime signals.

```text
Producer (NexusLoop, AgentEngine, ToolRuntime, ValidationEngine)
    → RuntimeEventBus.publish(RuntimeEvent)
        → subscribers: HookRegistry, TraceStore, PolicyEngine, Metrics, RecoveryCoordinator
```

### 42.2.1 Event Bus Contract

```text
interface RuntimeEventBus:
    publish(event: RuntimeEvent) -> None
    subscribe(event_types: list[RuntimeEventType], handler: EventHandler) -> SubscriptionId
    unsubscribe(subscription_id: SubscriptionId) -> None
```

### 42.2.2 Delivery Semantics

- **Synchronous dispatch** for hooks and policy (same execution thread/task context).
- **Async fan-out** permitted for metrics and external sinks only — MUST NOT block step execution.
- Handlers MUST be idempotent where possible.
- Handler failure MUST emit `RUNTIME_HANDLER_FAILED` and follow escalation policy (§42.38).

### 42.2.3 Anti-Pattern

Agents MUST NOT publish directly to external queues, webhooks, or Slack. They emit decisions and events **through** the runtime bus only.

---

## 42.3 Hook System

Hooks are **registered, ordered, inspectable interceptors** invoked by the runtime at defined points.

Hooks are NOT agent code. Hooks are Tier-1 runtime extensions (§42.22).

### 42.3.1 HookPoint Enum

```text
BEFORE_TASK_INTAKE
AFTER_TASK_INTAKE
BEFORE_CLASSIFICATION | AFTER_CLASSIFICATION
BEFORE_PLANNING | AFTER_PLANNING
BEFORE_AGENT_SELECTION | AFTER_AGENT_SELECTION
BEFORE_CONTEXT_BUILD | AFTER_CONTEXT_BUILD
BEFORE_STEP | AFTER_STEP
BEFORE_TOOL_CALL | AFTER_TOOL_CALL
BEFORE_VALIDATION | AFTER_VALIDATION
BEFORE_DECISION | AFTER_DECISION
BEFORE_INTERRUPT | AFTER_INTERRUPT
BEFORE_HUMAN_APPROVAL | AFTER_HUMAN_APPROVAL
BEFORE_RETRY | AFTER_RETRY
BEFORE_HANDOFF | AFTER_HANDOFF
BEFORE_FINALIZATION | AFTER_FINALIZATION
BEFORE_TRACE_PERSIST | AFTER_TRACE_PERSIST
```

### 42.3.2 Hook Handler Contract

```text
HookContext:
    task_id, run_id, node_id, agent_id, step_id
    phase: ExecutionPhase
    mutable_runtime_state: RuntimeStateView   # read-mostly; mutation via approved APIs only
    event: RuntimeEvent | null

HookResult:
    action: ALLOW | BLOCK | MODIFY | ESCALATE
    modified_payload: dict | null
    reason: str | null
```

### 42.3.3 Example — Cost Guard Hook

```text
@hook(BEFORE_TOOL_CALL)
def enforce_cost_ceiling(ctx: HookContext) -> HookResult:
    if ctx.runtime_state.accumulated_cost_usd > ctx.runtime_state.cost_ceiling:
        return HookResult(action=BLOCK, reason="cost_ceiling_exceeded")
    return HookResult(action=ALLOW)
```

### 42.3.4 Rules

- Hooks run in **priority order** (integer priority, lower first).
- Hooks MUST NOT call adapters directly; they influence policy and decisions only.
- Hooks MUST be registered in `HookRegistry` at application startup (Tier-3) or Nexus bootstrap.

**Authoring reference:** full `HookPoint` list and orchestration hook placement — [`guides/AGENT_CREATION_GUIDE.md` Appendix I §I.2](guides/AGENT_CREATION_GUIDE.md#i2-orchestration-control-plane-map) · governance hooks Appendix H.

---

## 42.4 Standard Agent Lifecycle

Every agent execution follows the **same lifecycle**, enforced by `AgentEngine` and `NexusLoop`.

```text
REGISTERED          # in AgentRegistry
    → SELECTED      # Nexus chose agent for task/node
    → CONTEXT_BUILDING
    → READY
    → RUNNING       # one or more AgentSteps
    → DECIDING      # AgentDecision emitted
    → VALIDATING
    → [PAUSED | INTERRUPTED | RETRYING | HANDOFF]
    → COMPLETED | FAILED | CANCELLED
```

### 42.4.1 State Transition Rules

- Only Nexus / AgentEngine MAY transition global agent lifecycle states.
- Agents MUST NOT set lifecycle state directly.
- Agents signal intent via `AgentDecision` only.
- Every transition MUST emit a `RuntimeEvent`.

### 42.4.2 Lifecycle vs Task Lifecycle

- **Task lifecycle** (§23): global user-facing task states.
- **Agent lifecycle** (this section): per-agent execution within a task.
- One task may contain multiple agent lifecycles (sequential, parallel, handoff).

---

## 42.5 Unified Agent Execution Protocol

The **Unified Agent Execution Protocol (UAEP)** is the mandatory sequence for all agent invocations.

```text
protocol UnifiedAgentExecution:

    1. Nexus selects agent (capability match + policy)
    2. AgentEngine.prepare_execution(agent, RuntimeExecutionContext)
    3. Middleware: BEFORE_CONTEXT_BUILD hooks
    4. agent.build_context(request) → context
    5. Middleware: AFTER_CONTEXT_BUILD hooks
    6. FOR each AgentStep in agent.get_steps(context) OR runtime-controlled step plan:
           a. Middleware: BEFORE_STEP
           b. AgentEngine.execute_step(agent, step, context)
           c. emit STEP_* events
           d. collect AgentDecision from step
           e. Middleware: AFTER_STEP
           f. IF decision != CONTINUE: break loop (Nexus handles)
    7. agent.validate(output, context) → ValidationResult
    8. Middleware: BEFORE_VALIDATION / AFTER_VALIDATION
    9. AgentEngine.build_execution_result(...) → AgentExecutionResult
   10. Return to Nexus with AgentDecision + result
```

### 42.5.1 Rules

- No agent MAY bypass steps 3–8.
- `execute()` on `Agent` interface (§13) MUST delegate to UAEP via `AgentEngine`.
- Direct `Agent.run()` from agent code is **forbidden** outside AgentEngine (§42.41).

---

## 42.6 Agent Step Lifecycle

Each internal agent step follows a micro-lifecycle:

```text
STEP_PLANNED
    → STEP_STARTED
    → [TOOL_REQUESTED → TOOL_COMPLETED]*   # via ToolRuntime only
    → STEP_DECIDING
    → STEP_COMPLETED | STEP_FAILED | STEP_SKIPPED
```

### 42.6.1 AgentStep Contract

```text
AgentStep:
    step_id: str
    step_name: str
    step_index: int
    input_schema: JSONSchema
    output_schema: JSONSchema
    allowed_tools: list[str]          # subset of agent contract
    max_duration_ms: int
    max_retries: int                  # runtime-managed (§42.34)
    idempotent: bool
    trace_label: str
```

### 42.6.2 Step Execution Pseudocode

```text
async def execute_step(agent, step, context):
    emit(STEP_STARTED)
    middleware.run(BEFORE_STEP)
    try:
        output = await agent.run_step(step, context, tool_gateway=ToolRuntime)
        decision = agent.decide_after_step(step, output, context)
        emit(DECISION_EMITTED, decision=decision)
        middleware.run(AFTER_STEP)
        emit(STEP_COMPLETED)
        return output, decision
    except Exception as e:
        emit(STEP_FAILED, error=str(e))
        return None, AgentDecision(type=FAIL, reason=str(e))
```

---

## 42.7 Agent Decision Model

Agents express control flow intent through **`AgentDecision`** — never through side effects or direct runtime manipulation.

### 42.7.1 AgentDecision Contract

```text
AgentDecisionType:
    CONTINUE          # proceed to next step
    COMPLETE          # agent finished successfully
    RETRY             # request runtime-managed retry (§42.34)
    REQUEST_HUMAN     # pause for human input/approval
    INTERRUPT         # structured interrupt (§42.8)
    ESCALATE          # elevate to supervisor/policy/human
    MODIFY_PLAN       # request Nexus replanning
    FAIL              # terminal failure for this agent/node
    CANCEL            # request task cancellation

AgentDecision:
    type: AgentDecisionType
    reason: str
    severity: EventSeverity
    payload: dict                    # structured context for Nexus
    interrupt: ExecutionInterrupt | null
    suggested_plan_delta: PlanDelta | null
    human_request: HumanRequest | null
    retry_hint: RetryHint | null
    confidence: float | null
```

### 42.7.2 Example — LegalAgent Critical Clause

```text
# LegalAgent detects a severe contract issue during step "clause_analysis"

return AgentDecision(
    type=INTERRUPT,
    reason="critical_liability_clause_detected",
    severity=CRITICAL,
    payload={
        "clause_id": "§14.2",
        "issue": "unlimited_liability",
        "evidence_artifact": "artifact_clause_flags.json"
    },
    interrupt=ExecutionInterrupt(
        interrupt_type=POLICY_REVIEW_REQUIRED,
        source_agent_id="legal",
        source_step_id="step_clause_analysis",
        recommended_action=REQUEST_HUMAN,
        blocking=True,
        metadata={"risk_level": "critical"}
    )
)
```

### 42.7.3 Rules

- Agent MUST NOT call `pause()`, `sleep()` waiting for human, or stop the event loop.
- Agent MUST NOT send Slack messages directly for approval.
- Nexus interprets `AgentDecision` via **PolicyEngine** (§42.11).
- `DECISION_EMITTED` event MUST precede any Nexus action on the decision.

---

## 42.8 Execution Interrupt Model

**Interrupts** are formal, structured requests to change global execution flow.

### 42.8.1 ExecutionInterrupt Contract

```text
ExecutionInterrupt:
    interrupt_id: str
    interrupt_type: InterruptType
    source_agent_id: str
    source_step_id: str | null
    task_id: str
    run_id: str
    blocking: bool                    # if true, no further steps until handled
    recommended_action: AgentDecisionType
    metadata: dict
    created_at: datetime

InterruptType:
    POLICY_REVIEW_REQUIRED
    SAFETY_VIOLATION
    COST_CEILING_BREACH
    VALIDATION_CRITICAL_FAILURE
    EXTERNAL_DEPENDENCY_FAILURE
    HUMAN_JUDGMENT_REQUIRED
    PLAN_OBSOLESCENCE
    AGENT_HANDOFF_REQUIRED
    RUNTIME_RECOVERY_REQUIRED
```

### 42.8.2 Interrupt Handling Flow

```text
Agent emits AgentDecision(INTERRUPT, interrupt=...)
    → EventBus: INTERRUPT_REQUESTED
    → Middleware: BEFORE_INTERRUPT hooks
    → PolicyEngine.evaluate_interrupt(interrupt) → PolicyDecision
    → Nexus InterruptHandler:
          REQUEST_HUMAN → pause + approval flow (§42.10)
          MODIFY_PLAN   → replan (§42.31 PLANNING phase)
          ESCALATE      → escalation flow (§42.38)
          FAIL          → mark node failed, propagate per graph policy
    → Middleware: AFTER_INTERRUPT hooks
    → EventBus: INTERRUPT_HANDLED | INTERRUPT_ESCALATED
```

### 42.8.3 Rules

- Interrupts are **idempotent** by `interrupt_id`.
- Duplicate interrupts MUST dedupe within the same `run_id`.
- Non-blocking interrupts MAY allow parallel nodes to continue (graph policy).

---

## 42.9 Pause / Resume Model

Pause is a **runtime state**, not an agent implementation detail.

### 42.9.1 Pause Triggers

- `AgentDecision.REQUEST_HUMAN`
- `PolicyDecision.require_human`
- External operator pause (API/CLI)
- Cost / safety hook BLOCK with pause semantics
- `ExecutionInterrupt.blocking == true`

### 42.9.2 Pause Contract

```text
PauseRecord:
    pause_id: str
    task_id: str
    run_id: str
    reason: str
    paused_at: datetime
    paused_phase: ExecutionPhase
    checkpoint: RuntimeCheckpoint    # serializable execution snapshot
    resume_token: str
    expires_at: datetime | null
```

### 42.9.3 Resume Flow

```text
resume(task_id, resume_token, operator_input?)
    → validate token + checkpoint integrity
    → emit RESUMED
    → restore RuntimeExecutionContext from checkpoint
    → continue UAEP from paused phase/step
```

### 42.9.4 Rules

- Checkpoints MUST include: plan snapshot, graph node states, context refs, pending decisions.
- Agents MUST NOT hold exclusive locks on external systems across pause; use idempotent re-entry.
- Long pauses MUST support expiry and escalation (§42.38).

---

## 42.10 Human In The Loop Runtime Flow

Human approval is a **first-class runtime phase**, not ad-hoc agent logic.

```text
AgentDecision.REQUEST_HUMAN | Interrupt → HUMAN_JUDGMENT_REQUIRED
    → emit HUMAN_APPROVAL_REQUESTED
    → Middleware: BEFORE_HUMAN_APPROVAL
    → PauseRecord created; task → waiting_for_human (§23)
    → Notification via Tier-0 adapter (Slack/Teams/UI) — triggered by Nexus, NOT agent
    → Human responds: APPROVE | REJECT | MODIFY | DELEGATE
    → emit HUMAN_APPROVAL_RECEIVED
    → Middleware: AFTER_HUMAN_APPROVAL
    → PolicyEngine maps response → CONTINUE | MODIFY_PLAN | FAIL | ESCALATE
    → Resume or replan
```

### 42.10.1 HumanRequest Contract

```text
HumanRequest:
    request_id: str
    prompt: str
    options: list[HumanOption]      # APPROVE, REJECT, EDIT, ...
    context_artifacts: list[str]
    urgency: LOW | NORMAL | HIGH | CRITICAL
    timeout_seconds: int | null
    default_on_timeout: AgentDecisionType | null
```

### 42.10.2 Autonomy level (user steering)

Distinct from host `ExecutionMode` (STRICT | BALANCED | EXPLORATORY) and agent `AgentExecutionMode` (SYNC | ASYNC).

```text
AutonomyLevel:
    MANUAL       # user approves meaningful actions
    ASK          # agent proposes; policy gates risky steps
    AUTONOMOUS   # execute within policy envelope
```

| Field | Location | Semantics |
|-------|----------|-----------|
| `TaskExecutionOptions.autonomy_level` | Task envelope | User/session slider value |
| Effective level | `PolicyEngine` | `min(user, tenant ceiling, execution_mode ceiling, agent risk)` |

**Mid-run changes:** operator or client MAY set autonomy before the next UAEP step; downgrade takes effect immediately for new tool calls; upgrade MUST NOT bypass unresolved HITL items.

**Events:** `AUTONOMY_LEVEL_SET`, `AUTONOMY_LEVEL_CHANGED` on `ops:governance` channel.

**Full model:** [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md) §35.

**Implementation coverage (2026-06-09):** runtime enforcement **Done** (REL-ADV). **HTTP mid-run setter** is lab-only (`harness_task_routes`); see [`ORCHESTRATION.md`](ORCHESTRATION.md) §59.4 · [`TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md) §23.7.

---

## 42.11 Policy Engine

The **PolicyEngine** interprets decisions, interrupts, and hook results against configurable rules.

### 42.11.1 PolicyDecision Contract

```text
PolicyDecision:
    action: ALLOW | DENY | MODIFY | ESCALATE | REQUIRE_HUMAN
    reason: str
    modified_decision: AgentDecision | null
    enforcement_level: ADVISORY | MANDATORY
    policy_rule_id: str
    audit_payload: dict
```

### 42.11.2 Policy Inputs

- `AgentContract.risk_level`
- Application Tier-3 config (industry rules)
- Tool access policy
- Cost ceilings
- Human approval requirements
- Regulatory / legal governance profiles (e.g. legal_application strict mode)

### 42.11.3 Example

```text
Policy: "critical legal interrupt MUST require human before COMPLETE"

evaluate(AgentDecision(COMPLETE), context):
    if context.has_unresolved_critical_interrupt:
        return PolicyDecision(
            action=REQUIRE_HUMAN,
            reason="unresolved_critical_interrupt",
            enforcement_level=MANDATORY
        )
```

### 42.11.4 RuntimePolicyBundle (Operator View)

Multiple policy mechanisms exist today (`PolicyEngine`, `ToolAccessPolicy`, `BudgetPolicy`, plan-loop policy, org/legal fragments). For operators and Tier-3 wiring, treat them as one composed object per application run:

```text
RuntimePolicyBundle:
    tool_access: ToolAccessPolicy | ToolScopePolicy
    memory_write: MemoryWritePolicy defaults
    budget: BudgetPolicy | null
    hitl: HumanApprovalPolicy | null
    plan_loop: PlanLoopPolicy | null
    domain_fragments: dict[str, object]   # e.g. legal_failure_policy_id
```

**Composition rules:**

- Tier-3 application factory builds the bundle once at startup → `ApplicationBuildContext.policy_bundle` → `RuntimeConfig.policy_bundle` / `RuntimeContext.policy_bundle` via `applications/_shared/runtime_config_bridge.py` (also maps `RuntimePolicyBundle.tool_access` → `RuntimeConfig.tool_scope_policy` when the bundle carries a `ToolScopePolicy` implementation).
- Nexus and UAEP read from the bundle — agents MUST NOT construct parallel policy objects.
- Skill `policy_fragment_id` (§7.1.8) merges into `domain_fragments` or tool policy — never bypasses `ToolRuntime`.

Implementation: [`plan/UNIFIED_EXECUTION_RUNTIME.md) R-Policy (Done).

### 42.11.5 How to read policy for a run (operator)

For a single task/run, policy is **composed once** at Tier-3 startup and read downstream — do not hunt per-agent ad-hoc rules.

| Step | What to inspect | Where |
|------|-----------------|--------|
| 1 | Application bundle | `ApplicationBuildContext.policy_bundle` → `RuntimePolicyBundle` (tool access, budget, HITL, plan-loop, `domain_fragments`) |
| 2 | Agent + skills | `AgentContract.skill_ids` → resolved `allowed_tools` + `policy_fragment_ids` (`SKILL_RESOLVED` event) |
| 3 | Nexus execution | `ToolAccessPolicy` / `ToolRuntime` enforce allow-list per step (`resolve_allowed_tools_from_config` reads bundle + agent contract); `BudgetPolicy` on token/cost ceilings |
| 4 | Legal / org overlays | `domain_fragments` keys (e.g. `legal.contract_review.policy`) — org settings may clamp flags before runtime |
| 5 | Human gates | `PolicyDecision.action == REQUIRE_HUMAN` → HITL pause; resume via checkpoint / approval metadata |

**Trace checklist:** `RuntimeEvent` stream (`PLAN_CREATED`, `SKILL_RESOLVED`, `CONTEXT_ASSEMBLED`, tool events) + Nexus trace DB for planner/tool steps. Planner hard failures emit `PLAN_FAILED` (parse / PlanSource).

**Authoring reference:** Tier-3 control-plane map (profiles, bundles, observability mandatory vs optional, verification commands) — [`guides/AGENT_CREATION_GUIDE.md` Appendix H](guides/AGENT_CREATION_GUIDE.md#appendix-h--governance-policy--observability-control-plane). Harness operator context: [`guides/HARNESS_ENVIRONMENT.md`](guides/HARNESS_ENVIRONMENT.md#harness-control-plane-authoring).

### 42.11.6 Guardrail catalog (operator index)

**Terminology:** In Intergrax, **guardrails** are **not** a separate Tier-0/Tier-1 package. They are the **enforcement surface** of Policy & Governance — typed checks at UAEP hook points that produce `PolicyDecision`, `ValidationResult`, security inspection results, or provider safety signals (`refusal`, `content_filter`). Canonical Harness AI term: [`PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md) §5.3.1.

**Ideal model:** prompt, output, tools, cost, execution time — [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.3. **Third-party guardrail engines** (NeMo Guardrails, Guardrails AI, LLM Guard, OpenGuardrails, …) are wired as **Integration Library** backends — see [`INTEGRATIONS.md`](INTEGRATIONS.md) §47.

#### Guardrail types → hook points → owners

| Guardrail type | UAEP / Nexus hook | Primary owner | Enforcement artifact | Fail mode |
|----------------|-------------------|---------------|----------------------|-----------|
| **Tool allow-list / scope** | `before_tool_call` | `ToolRuntime` + `ToolAccessPolicy` | `PolicyDecision` DENY | Fail-closed |
| **Tool argument / injection** | `before_tool_call` | `tool_security.py` + middleware | Block / modify `ToolRequest` | Fail-closed on CRITICAL |
| **Budget / token / quota** | pre-LLM, pre-tool | `BudgetPolicy`, `cost_quota.py` | DENY / degrade | Configurable |
| **Prompt injection (input)** | pre-LLM, intake | `prompt_security.py` + optional `llm_guardrail` integration | `PromptInspectionResult` → DENY | Profile-driven |
| **Context / memory write** | pre-store | `MemoryWritePolicy` | DENY | Fail-closed |
| **Plan / step policy** | pre-step, plan loop | `PlanLoopPolicy`, `plan_validator` | MODIFY / DENY | MANDATORY on strict hosts |
| **Structural output (L0)** | post-step, post-node | `NexusValidationEngine`, CVL `L0Gateway` | `ValidationResult` | Retry / FAIL |
| **Semantic output (L1)** | post-step, completion | CVL `eval.judge`, `CriticOrchestrator` | Score + policy consequence | Opt-in by `CriticProfile` |
| **Human gate (L2)** | interrupt, completion | `PolicyEngine` + `HitlRunner` | `REQUIRE_HUMAN` | Pause until resume |
| **Provider safety** | post-LLM | `LLM_ADAPTERS` envelope | `refusal`, `finish_reason=content_filter` | Surface to PolicyEngine |
| **Retrieval poisoning** | pre-RAG inject | `retrieval_security.py` | Quarantine / deny chunk | Tenant-scoped |
| **Tenant isolation** | all paths | `tenant_security.py` | DENY cross-tenant | Fail-closed |
| **Execution time / retry** | step, graph node | `RetryEngine`, `ExecutionGuard` | RETRY / FAIL | Budget-capped |
| **Cost optimization cap** | adaptive loop | `OptimizationGuardrail` | Cap recommendation ratio | Advisory |

#### Composition flow (single run)

```text
Tier-3 startup
    → RuntimePolicyBundle + ApplicationSecurityProfile
    → optional IntegrationProfile.llm_guardrail slug

INTAKE / pre-run
    → PolicyEngine (autonomy ceiling, domain_fragments)

Per UAEP step:
    before_step → ContextManager (budget overlays)
    pre-LLM     → prompt_security + optional llm_guardrail.scan_input()
    post-LLM    → refusal/content_filter + optional llm_guardrail.scan_output()
    before_tool_call → ToolAccessPolicy + tool_security
    after_tool_call  → trace + policy hooks
    post-step   → NexusValidationEngine (L0) → CVL L1 if enabled

Terminal:
    PolicyEngine (unresolved interrupts) → REQUIRE_HUMAN or FAIL
    CVL final verification if require_critic_on_completion
```

#### What is explicitly not a guardrail layer

| Anti-pattern | Why |
|--------------|-----|
| Agent-local `if` checks without trace | Untestable, bypasses PolicyEngine |
| Direct vendor guardrail SDK in Tier-2 | Violates tier boundaries — use Integration → middleware |
| Duplicate policy objects per agent | Use `RuntimePolicyBundle` only (§42.11.4) |
| CVL rubrics in Nexus for domain logic | Tier-2 owns rubric content; Nexus orchestrates only |

**Implementation plan:** [`plan/UNIFIED_EXECUTION_RUNTIME.md`](../plan/UNIFIED_EXECUTION_RUNTIME.md) Phase **GR-DOC** (documentation) + [`plan/INTEGRATIONS.md`](../plan/INTEGRATIONS.md) Phase **M.12** (vendor adapters).

---

## 42.12 ToolRuntime Enforcement Rules

All Tier-0 tool and adapter access MUST go through **`ToolRuntime`** (§22, `intergrax/runtime/nexus/tools/tool_runtime.py`).

### 42.12.1 ToolRequest / ToolResponse Contracts

```text
ToolRequest:
    request_id: str
    tool_name: str
    agent_id: str
    step_id: str
    input: dict
    risk_level: RiskLevel
    timeout_ms: int
    idempotency_key: str | null

ToolResponse:
    request_id: str
    status: SUCCESS | DENIED | TIMEOUT | FAILED
    output: dict | null
    error: str | null
    duration_ms: int
    trace_ref: str
```

### 42.12.2 Enforcement Rules

1. **No direct adapter imports** in `agents/` (§42.41).
2. `ToolAccessPolicy` MUST filter against `AgentContract.allowed_tools`.
3. Every invoke MUST emit `TOOL_REQUESTED` and terminal `TOOL_*` event.
4. Denied tools return `ToolResponse(status=DENIED)` — agents MUST handle gracefully via `AgentDecision`.
5. Sandbox-required tools MUST route through `SandboxAdapter` policy.
6. Retries for tools are **runtime-managed** (§42.34), not agent loops.

---

## 42.13 Shared Execution Contracts

Canonical contract bundle — all MUST be implemented or delegated by `AgentEngine`:

| Contract | Owner | Purpose |
|----------|-------|---------|
| `AgentContract` | Tier-2 agent | Capability declaration (§12) |
| `RuntimeExecutionContext` | Tier-1 | Unified per-run context (§42.13.1) |
| `AgentStep` | Tier-2 / runtime | Step boundary (§42.6) |
| `AgentDecision` | Tier-2 emit, Tier-1 interpret | Control flow (§42.7) |
| `ExecutionInterrupt` | Tier-2 emit, Tier-1 handle | Structured interrupts (§42.8) |
| `AgentExecutionResult` | Tier-1 assemble | Output to Nexus (§14) |
| `ValidationResult` | Tier-2 + Tier-1 | Validation (§42.16) |
| `RuntimeEvent` | Tier-1 emit | Observability (§42.1) |
| `ToolRequest/ToolResponse` | Tier-1 | Tool gateway (§42.12) |
| `PolicyDecision` | Tier-1 | Governance (§42.11) |

### 42.13.1 RuntimeExecutionContext Contract

```text
RuntimeExecutionContext:
    task_id: str
    run_id: str
    node_id: str | null
    agent_id: str
    correlation_id: str
    phase: ExecutionPhase
    contract: AgentContract
    request: RuntimeRequest
    context: RuntimeContext          # agent-built domain context
    state: RuntimeStateView          # read-only runtime state for agent
    tool_gateway: ToolGateway        # ToolRuntime facade ONLY
    event_emitter: EventEmitter      # emit agent-scoped events (wrapped → EventBus)
    memory_view: MemoryView          # policy-scoped memory (§42.35)
    trace: TraceWriter
    metadata: dict
```

Agents receive `RuntimeExecutionContext` — never raw global singletons.

---

## 42.14 Cross-Agent Communication Contracts

Agents MUST NOT call each other directly.

Cross-agent work flows through **Nexus orchestration**:

```text
Agent A completes → AgentExecutionResult + AgentDecision
    → Nexus updates ExecutionGraph / shared context
    → Nexus selects Agent B
    → AgentEngine runs Agent B with enriched RuntimeExecutionContext
```

### 42.14.1 Shared Context Contract

```text
SharedTaskContext:
    task_id: str
    artifacts: dict[str, ArtifactRef]
    structured_outputs: dict[str, dict]   # keyed by agent_id or node_id
    memory_namespace: str
    version: int                           # optimistic concurrency
```

Writes to `SharedTaskContext` MUST go through `ContextManager` (Tier-1), not agent-private globals.

### 42.14.2 Context Assembly Options

Per-node agent context is bounded by typed intake options on the task:

```text
TaskContextAssemblyOptions:
    summary_tier: FULL | SUMMARY_ONLY | STRUCTURED_ONLY | MINIMAL
    max_prior_chars: int
    max_prior_entries: int
    include_shared_handoffs: bool
    include_shared_artifacts: bool
```

Canonical placement: `TaskExecutionOptions.context` (§23 typed task contract).

`ContextManager.build_agent_context()` resolves policy from `task.options.context`, assembles `AgentContextBundle` with provenance, and applies summary-tier rules before agent execution.

Legacy flat metadata keys remain supported via `task_metadata_bridge` for JSON/API serialization only.

Handoff payloads in shared context use keys prefixed with `handoff:` (see §42.15).

### 42.14.3 Graph Delegation (Subagent Equivalent)

Harness literature describes **subagents** as autonomous units with their own rules, model, and memory. Intergrax implements the **same outcome** through Tier-1 orchestration — not nested harness instances.

| Harness subagent | Intergrax delegation |
|------------------|----------------------|
| Spawn child with own context | `ExecutionGraph` node with `DelegationSpec` |
| Isolated memory | `MemoryView` namespace `task_id/delegation/{node_id}/` |
| Bounded parent context | `TaskContextAssemblyOptions` override on child node |
| Traceability | `parent_run_id`, `parent_node_id` on child metadata |

**Forbidden:** Tier-2 agent spawning another agent by direct import or private API. **Required:** Nexus schedules child node after parent decision or plan edge.

**Declarative `DELEGATES_TO` (implemented):** Tier-3 `ApplicationGraphSpec` may declare `DELEGATES_TO` as authoring sugar; `graph_spec_to_plan.py` **expands** it to a **child `PlanStep` / `ExecutionNode`** with `DelegationSpec` on the **child** node ([ADR-FLOW-001](adr/entries/2026-06-07/ADR-FLOW-001.md) Option C). `SubtaskContract` supplies objective, scopes, and budget envelope on the child delegation path (FLOW-14/15).

Implementation: R-Delegate (**Done**) for contracts and memory namespace; graph expansion (**Done**, Phase FLOW) in [`plan/UNIFIED_EXECUTION_RUNTIME.md) · operational narrative [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §13.

```text
DelegationSpec:
    child_agent_id: str
    isolated_memory_namespace: str
    context_assembly: TaskContextAssemblyOptions | null
    inherit_tool_policy: bool              # default true — intersect with child contract
```

---

## 42.15 Agent Handoff Contracts

**Handoff** is a Nexus-mediated transfer of responsibility between agents.

```text
AgentHandoff:
    handoff_id: str
    from_agent_id: str
    to_agent_id: str | null             # null → Nexus selects by capability
    to_capability: str | null
    payload: dict
    reason: str
    artifacts: list[str]
    required_validation: list[str]      # validator agent ids or rules
```

### 42.15.1 Handoff Flow

```text
AgentDecision(MODIFY_PLAN) or explicit handoff step
    → emit HANDOFF_INITIATED
    → Nexus validates handoff policy
    → update graph / insert new node
    → AgentEngine runs target agent
    → emit HANDOFF_COMPLETED
```

---
