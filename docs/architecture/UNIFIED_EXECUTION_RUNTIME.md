# Unified Execution Runtime

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/UNIFIED_EXECUTION_RUNTIME.md`](../plan/UNIFIED_EXECUTION_RUNTIME.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 4–5, 8, 23–24  
---

## 42.1 Runtime Event Model

Every meaningful runtime transition MUST emit a `RuntimeEvent`.

Events are the **primary audit and orchestration signal**. Hooks, observability, policy, and recovery subscribe to events — they MUST NOT rely on hidden callbacks inside agents.

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
- Direct `RuntimeEngine.run()` from agent code is **forbidden** outside AgentEngine (§42.41).

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

**Declarative `DELEGATES_TO` (implemented):** Tier-3 `ApplicationGraphSpec` may declare `DELEGATES_TO` as authoring sugar; `graph_spec_to_plan.py` **expands** it to a **child `PlanStep` / `ExecutionNode`** with `DelegationSpec` on the **child** node ([ADR-FLOW-001](adr/ADR-FLOW-001.md) Option C). `SubtaskContract` supplies objective, scopes, and budget envelope on the child delegation path (FLOW-14/15).

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

## 42.16 Validation Contract Model

Validation is **multi-stage** and enforced by runtime gates.

### 42.16.1 ValidationContract

```text
ValidationContract:
    validation_id: str
    scope: STEP | AGENT | NODE | TASK
    rules: list[ValidationRule]
    on_failure: RETRY | INTERRUPT | FAIL | REQUEST_HUMAN

ValidationRule:
    rule_id: str
    description: str
    severity: WARNING | ERROR | CRITICAL
    evaluator: str                      # registered validator id or agent id

ValidationResult:
    valid: bool
    errors: list[ValidationError]
    warnings: list[ValidationWarning]
    stage: str
    validator_id: str
```

### 42.16.2 Validation Stages (ordered)

1. **Step-local** — agent `validate_step()` (optional)
2. **Agent-local** — agent `validate()` (§13, required)
3. **Runtime** — `NexusValidationEngine`
4. **Dedicated ValidatorAgent** — graph node (§42.30)
5. **Human** — when policy requires

Failure at CRITICAL severity MUST NOT silently downgrade to WARNING.

---

## 42.17 Runtime State Machine

Global runtime state machine for a **single task run**:

```text
                    ┌─────────────┐
                    │   INTAKE    │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
               ┌───│CLASSIFICATION│───┐
               │   └──────┬──────┘   │
               │          ▼          │
               │   ┌─────────────┐ │
               │   │  PLANNING   │◄┘ MODIFY_PLAN
               │   └──────┬──────┘
               │          ▼
               │   ┌─────────────┐
               │   │CTX + SELECT │ (CONTEXT_BUILDING + AGENT_SELECTION)
               │   └──────┬──────┘
               │          ▼
         ┌─────┴─────────────────────────────┐
         │         STEP_EXECUTION            │◄──┐ RETRY
         └─────┬─────────────────────────────┘   │
               │          │                      │
               ▼          ▼                      │
        ┌──────────┐ ┌───────────┐               │
        │VALIDATION│ │ INTERRUPT │───────────────┤
        └────┬─────┘ └─────┬─────┘               │
             │             │ PAUSE / HUMAN         │
             ▼             ▼                      │
        ┌─────────────────────────┐              │
        │  RETRY / REPLAN / ESCALATE             │
        └────────────┬────────────┘              │
                     ▼                           │
              ┌─────────────┐                    │
              │FINALIZATION │                    │
              └──────┬──────┘                    │
                     ▼                           │
              ┌─────────────┐                    │
              │ TRACE + DONE│                    │
              └─────────────┘                    │
                     │                           │
              COMPLETED | FAILED | CANCELLED     │
```

Only **NexusLoop** / **TaskLifecycle** MAY drive these transitions.

---

## 42.18 Runtime Step Contracts

Runtime-level steps (distinct from AgentSteps) are internal Nexus operations:

```text
RuntimeStep:
    INTAKE_NORMALIZE
    CLASSIFY_TASK
    BUILD_PLAN
    RESOLVE_AGENTS
    BUILD_EXECUTION_GRAPH
    EXECUTE_NODE
    COMPOSE_PARTIAL_RESULTS
    VALIDATE_GLOBAL
    APPLY_POLICY
    FINALIZE_RESPONSE
    PERSIST_TRACE
```

Each runtime step MUST:

- emit phase-aligned `RuntimeEvent`
- run applicable middleware hooks
- record duration and outcome in trace
- be idempotent where retry applies

---

## 42.19 AgentEngine Responsibilities

**`AgentEngine` is the single canonical executor for all Tier-2 agents.**

Location: `intergrax/agents/agent_engine.py` (evolving toward full §42 compliance).

### 42.19.1 AgentEngine MUST

- Resolve agent from `AgentRegistry`
- Build `RuntimeExecutionContext`
- Run UAEP (§42.5) including middleware pipeline
- Invoke agent steps through runtime-controlled loop (§42.33)
- Route all tool calls through `ToolRuntime`
- Collect `AgentDecision` after each step
- Invoke validation stages
- Emit `RuntimeEvent` stream for agent execution
- Assemble `AgentExecutionResult`
- Return control to Nexus — never own global task loop

### 42.19.2 AgentEngine MUST NOT

- Embed domain logic for Legal, Research, UX, etc.
- Select agents globally (Nexus responsibility)
- Bypass PolicyEngine or HookRegistry
- Allow agents to mutate unchecked global state

### 42.19.3 Target Interface

```text
class AgentEngine:
    async def execute(
        self,
        agent: Agent,
        request: RuntimeRequest,
        nexus_context: NexusExecutionContext,
    ) -> AgentExecutionBundle:
        """
        Returns: AgentExecutionResult + final AgentDecision + event stream ref
        """

    async def execute_step(
        self,
        agent: Agent,
        step: AgentStep,
        ctx: RuntimeExecutionContext,
    ) -> StepExecutionResult:
        ...
```

Agents implement **`run_step` / domain pipeline** — NOT **`execute` lifecycle**.

---

## 42.20 Runtime Middleware Pipeline

Middleware composes hooks into an **ordered execution pipeline** around every runtime operation.

```text
Request
  → [before_* hooks in priority order]
  → core operation (step, tool, validation, interrupt, human)
  → [after_* hooks in reverse priority order]
  → result
```

### 42.20.1 Standard Middleware Stages

| Stage | HookPoint |
|-------|-----------|
| Before/after step | `BEFORE_STEP`, `AFTER_STEP` |
| Before/after tool | `BEFORE_TOOL_CALL`, `AFTER_TOOL_CALL` |
| Before/after validation | `BEFORE_VALIDATION`, `AFTER_VALIDATION` |
| Before/after interrupt | `BEFORE_INTERRUPT`, `AFTER_INTERRUPT` |
| Before/after human | `BEFORE_HUMAN_APPROVAL`, `AFTER_HUMAN_APPROVAL` |

### 42.20.2 Middleware Stack Example

```text
middleware_stack = [
    TraceMiddleware(priority=10),
    CostAccountingMiddleware(priority=20),
    PolicyEnforcementMiddleware(priority=30),
    SafetyRedactionMiddleware(priority=40),
    CustomAppMiddleware(priority=100),   # Tier-3 registered
]
```

### 42.20.3 Rules

- Middleware MUST be stateless or use scoped context only.
- Middleware MAY return BLOCK — core operation MUST NOT run.
- Agent code MUST NOT register middleware; Tier-3 applications register at bootstrap.

---

## 42.21 Runtime Extensibility Rules

Extensions are allowed only through **approved extension points**:

1. `HookRegistry` — hooks (§42.3)
2. `ToolRegistry` — new tools (Tier-0 + registration)
3. `AgentRegistry` — new agents (Tier-2)
4. `PolicyEngine` rules — Tier-3 config
5. `ValidationEngine` rules — registered validators
6. Middleware plugins — Tier-3 bootstrap

### Forbidden Extension Points

- Subclassing `NexusLoop` per agent
- Monkey-patching `AgentEngine`
- Agent-specific event bus instances
- Private fork of `ToolRuntime`

---

## 42.22 Runtime Plugin / Hook Architecture

```text
RuntimePlugin:
    plugin_id: str
    version: str
    compatible_runtime: semver range
    register(bus: RuntimeEventBus, hooks: HookRegistry, policy: PolicyEngine) -> None
    on_shutdown() -> None
```

Tier-3 applications MAY load plugins at startup.

Plugins MUST declare compatible runtime schema versions (§42.29).

Plugins MUST NOT import agent domain modules.

---

## 42.23 Structured Runtime Event Payloads

All payloads MUST be JSON-serializable and schema-versioned.

### 42.23.1 Payload Schemas (minimum)

```text
decision.v1:
    decision_type, reason, severity, interrupt_id?

tool.v1:
    tool_name, status, duration_ms, redacted_input_summary

validation.v1:
    valid, error_count, warning_count, stage, rule_ids_failed

interrupt.v1:
    interrupt_type, blocking, recommended_action, metadata

human.v1:
    request_id, option_selected, operator_id?, comment?

handoff.v1:
    from_agent, to_agent, capability, artifact_ids
```

Unknown payload fields MUST be preserved (forward compatibility).

---

## 42.24 Runtime Observability Protocol

Observability is **event-first** (§42.1), trace-second, metrics-third.

### 42.24.1 Trace Requirements

Every run MUST produce a **TraceRecord** containing:

- ordered `RuntimeEvent` list (or reference)
- execution graph snapshot
- agent decisions with timestamps
- tool calls (redacted)
- validation outcomes
- cost aggregation
- final status + reason

### 42.24.2 Correlation

- `correlation_id` = task-level
- `run_id` = attempt-level (retries create new run_id or retry branch id per policy)
- `parent_event_id` = causal chain

### 42.24.3 Inspectability Guarantee

An operator MUST reconstruct **why** the runtime stopped using trace + events alone — without reading agent source code.

**Unified run journal (OBS-DEPTH.1):** ``build_unified_run_journal()`` in
``intergrax/runtime/events/unified_run_journal.py`` merges persisted
``RuntimeEvent`` rows with trace-bridged events (``trace_bridge``) into one
chronological timeline. Debug ``GET /debug/tasks/{run_id}/trace?include_runtime=true``
and the debug CLI use this journal when runtime event persistence is configured.

---

## 42.25 Runtime Safety Enforcement

Safety controls are **mandatory defaults**, not optional agent behavior.

| Control | Enforcement layer |
|---------|-------------------|
| Tool allowlist | ToolAccessPolicy + contract |
| Sandbox for code/browser | ToolRuntime routing |
| PII redaction in traces | TraceMiddleware |
| Cost ceilings | PolicyEngine + hooks |
| Human gate for CRITICAL | PolicyEngine |
| Secret exclusion from events | Event emitter |

Violations emit `SAFETY_VIOLATION` interrupt and follow escalation (§42.38).

---

## 42.26 Runtime Cancellation Semantics

```text
cancel(task_id, reason, initiated_by)
    → emit CANCELLATION_REQUESTED
    → propagate to active nodes (graph policy)
    → cancel in-flight tool calls (best-effort)
    → agent steps receive CancelledError at next checkpoint
    → emit CANCELLED
    → finalize partial trace
```

### Rules

- Cancellation is cooperative at step boundaries — steps MUST checkpoint frequently.
- Parallel nodes: cancellation propagates to all descendants unless isolated branch policy says otherwise.
- Cancelled tasks MUST NOT emit COMPLETE decisions.

---

## 42.27 Agent Capability Versioning

```text
CapabilityDescriptor:
    capability: str              # e.g. "legal.contract_review"
    version: semver              # e.g. "2.1.0"
    agent_id: str
    contract_version: str
    deprecated: bool
    superseded_by: str | null
```

Nexus routes by `(capability, version range)` from Tier-3 config.

Breaking capability changes MUST bump major version.

---

## 42.28 Contract Versioning

All runtime contracts carry `schema_version` or semver:

- `runtime_event.v1`
- `agent_contract.v1`
- `agent_decision.v1`
- `validation_result.v1`

Breaking changes require new major version; runtime MUST support N and N-1 during migration windows (§42.29).

---

## 42.29 Runtime Compatibility Guarantees

```text
RuntimeVersion:
    runtime: semver              # intergrax runtime package
    contract_bundle: str         # e.g. "uaep-1.0"
    supported_event_schema: list[str]
    supported_agent_contract: list[str]
```

**Guarantees:**

- Tier-2 agents declare `required_runtime >= X`
- Tier-3 applications pin runtime version in config
- Nexus rejects agents with incompatible contract versions at registration time
- Event consumers MUST ignore unknown fields

**Code (2026-05-27):** `intergrax/runtime/schema/registry.py` exposes `RUNTIME_SCHEMA_REGISTRY`, `current_runtime_version()`, and `validate_schema_version()`. `intergrax/runtime/events/phase_coverage.py` maps every `RuntimeEventType` to an `ExecutionPhase`. Persistence enforces both via `ValidatingRuntimeEventPersistence` (wrapped by `resolve_runtime_event_persistence()`).

---

## 42.30 Runtime Scheduling Model

Nexus schedules work through **ExecutionGraph** (§24) with explicit modes:

| Mode | Description |
|------|-------------|
| **Sequential** | Node B starts after Node A completes successfully |
| **Parallel** | Independent nodes in same batch |
| **Speculative** | Provisional branch; commit or discard on validation |
| **Validator** | ValidatorAgent node gates downstream edges |
| **Retry branch** | Subgraph re-execution on RETRY decision |
| **Cancellation propagate** | Parent cancel → child cancel |

### 42.30.1 Scheduling Pseudocode

```text
for batch in graph.topological_batches():
    if batch.mode == PARALLEL:
        results = await gather([execute_node(n) for n in batch.nodes])
    else:
        for node in batch.nodes:
            result = await execute_node(node)
            decision = result.decision
            if decision.type in (INTERRUPT, REQUEST_HUMAN, FAIL, CANCEL):
                handle_global_decision(decision)
                break
    merge_results(batch)
    validate_batch_if_required()
```

---

## 42.31 Runtime Execution Phases

Canonical **`ExecutionPhase`** enum — aligns events, hooks, traces, and state machine:

```text
INTAKE
CLASSIFICATION
PLANNING
CONTEXT_BUILDING
AGENT_SELECTION
STEP_EXECUTION
VALIDATION
INTERRUPT_HANDLING
RETRY_HANDLING
HUMAN_APPROVAL
FINALIZATION
TRACE_PERSISTENCE
COMPLETION
```

Every `RuntimeEvent.phase` MUST use this enum.

Phase transitions MUST be logged.

---

## 42.32 Agent Local Loop Standardization

Agents MAY implement multi-step domain logic, but local loops MUST follow the **standard shape**:

```text
class DomainAgent(Agent):
    def get_steps(self, context) -> list[AgentStep]:
        """Declarative step list OR runtime-generated from pipeline template."""

    async def run_step(self, step, ctx: RuntimeExecutionContext) -> StepOutput:
        """Domain logic ONLY. No adapter calls — use ctx.tool_gateway."""

    def decide_after_step(self, step, output, ctx) -> AgentDecision:
        """Return CONTINUE | INTERRUPT | ... — no side effects."""
```

### Rules

- `max_steps` from contract enforced by AgentEngine (hard stop → FAIL decision).
- No `while True` without step counter and runtime checkpoint.
- Local loop iteration = one `AgentStep` per iteration — **not** hidden inner loops.

---

## 42.33 Runtime-Controlled Local Loops

The **runtime** owns the loop construct; the agent owns **step bodies**.

```text
# CORRECT — runtime loop
steps = agent.get_steps(ctx)
for step in steps:
    if ctx.should_cancel(): break
    output, decision = await engine.execute_step(agent, step, ctx)
    if decision.type != CONTINUE:
        return decision

# FORBIDDEN — agent-owned loop calling adapters (§42.41)
async def execute(...):
    while not done:
        await postgres.query(...)   # FORBIDDEN
```

Pipeline classes (e.g. LegalPipeline) MUST decompose into `AgentStep` sequences invokable by AgentEngine.

---

## 42.34 Runtime-Managed Retries

Retries are **never** implemented as agent-internal `for attempt in range(n)` against adapters.

```text
RetryHint:
    retryable: bool
    reason: str
    backoff_ms: int | null
    max_attempts: int | null        # capped by contract + policy

RetryEngine (Tier-1):
    on AgentDecision(RETRY) or ValidationResult retryable:
        emit RETRY_SCHEDULED
        apply backoff
        emit RETRY_STARTED
        re-enter STEP_EXECUTION or subgraph (§42.30)
        increment run attempt counter
```

Agent emits **intent** (`RETRY`); runtime executes retry policy (§31).

---

## 42.35 Runtime-Controlled Memory Access

```text
MemoryView:
    read(namespace: str, key: str) -> MemoryRecord | null
    write(namespace: str, key: str, value: dict, policy: MemoryWritePolicy) -> void
    list(namespace: str, prefix: str) -> list[MemoryRecord]
```

### Rules

- Agents MUST NOT write to Redis/PostgreSQL memory adapters directly.
- Namespaces scoped by `task_id` + policy from Tier-3 config.
- Every read/write emits `MEMORY_READ` / `MEMORY_WRITE` events.
- Cross-agent shared memory uses `SharedTaskContext` via ContextManager (§42.14).

---

## 42.36 Runtime-Controlled Tool Access

See §42.12. Summary:

- `ctx.tool_gateway.invoke(ToolRequest)` — only path
- Policy + contract enforced on every call
- Tool results attached to trace automatically
- Agent code receives `ToolResponse`, not raw adapter clients

---

## 42.37 Runtime Governance Model

**Governance** = contracts + policy + hooks + validation + observability working together.

```text
Governance layers:
    1. AgentContract (static declaration)
    2. ToolAccessPolicy (per-invocation)
    3. PolicyEngine (decision/interrupt)
    4. ValidationEngine (multi-stage)
    5. HookRegistry (cross-cutting rules)
    6. Tier-3 application config (industry rules)
```

No single layer is sufficient alone.

Governance failures MUST default to **fail-closed** for CRITICAL risk agents (legal, financial, safety).

---

## 42.38 Runtime Escalation Flow

```text
ESCALATE decision or policy mandate
    → emit INTERRUPT_ESCALATED
    → EscalationRouter:
          SUPERVISOR_AGENT (future)
          HUMAN_OPERATOR
          APPLICATION_ADMIN
          FAIL_TASK
    → record escalation chain in trace
    → pause or continue per policy
```

Escalation MUST NOT be silently swallowed.

---

## 42.39 Runtime Critical Event Handling

`severity == CRITICAL` events trigger:

1. Immediate PolicyEngine evaluation
2. Optional automatic pause (`blocking interrupt`)
3. Mandatory trace persistence before continuing
4. Human notification for configured Tier-3 profiles

Critical events include: safety violations, unlimited liability detection, cost runaway, validation CRITICAL failures.

---

## 42.40 Runtime Recovery Model

```text
RecoveryCoordinator:
    on RUNTIME_RECOVERY_REQUIRED interrupt or node failure:
        1. load checkpoint (§42.9)
        2. classify: transient | permanent | partial
        3. transient → RETRY_HANDLING phase
        4. partial → replan excluding completed nodes
        5. permanent → FAIL with full trace
        emit recovery events at each sub-step
```

Recovery MUST be deterministic given same checkpoint + inputs (reproducibility).

---

## 42.41 Forbidden Runtime Patterns

The following are **explicitly forbidden** in Tier-2 agents and discouraged everywhere:

| Pattern | Why forbidden |
|---------|---------------|
| **Direct adapter access** | Bypasses ToolRuntime policy and trace |
| **Private runtime loops** | Uncontrolled execution, untraceable |
| **Hidden side effects** | Slack/email/DB writes outside contract |
| **Direct global state mutation** | Breaks reproducibility |
| **Uncontrolled background tasks** | `asyncio.create_task` without runtime registration |
| **Runtime bypassing** | Calling `RuntimeEngine` outside AgentEngine |
| **Unmanaged async execution** | Fire-and-forget coroutines in agents |
| **Untraceable execution paths** | Logic without STEP/TOOL events |
| **Custom retry loops in agents** | Duplicates RetryEngine, causes cost runaway |
| **Agent-to-agent direct calls** | Bypasses Nexus governance |
| **Custom EventBus instances** | Fragments observability |
| **Human prompts inside agent** | Must use REQUEST_HUMAN decision |
| **Duplicate Tier-0 mechanisms** | Second LLM layer, logger, tool registry, RAG stack, DB client (§5.2) |
| **§42 scaffold as parallel platform** | Must wire into existing trace/tools/LLM — not replace them |
| **New universal Tier-0 without human approval** | Violates §5.2.4 platform governance |

Violation in code review MUST block merge.

Reference also: §43 Anti-Patterns (architectural), §43.8 (redundancy), §42.33 (loop ownership).

---

## 42.42 Runtime Middleware Pipeline (Canonical Reference)

Full hook catalog for implementers:

```text
before_step(agent, step, ctx)           → allow | block | modify context
after_step(agent, step, output, decision, ctx)

before_tool_call(request, ctx)          → allow | deny | modify request
after_tool_call(request, response, ctx)

before_validation(target, ctx)          → allow | skip | augment rules
after_validation(result, ctx)           → fail-closed override

before_interrupt(interrupt, ctx)        → escalate | modify | allow
after_interrupt(outcome, ctx)

before_human_approval(request, ctx)
after_human_approval(response, ctx)
```

Implementations: `intergrax/runtime/middleware/` (target module layout).

All middleware MUST register with priority and emit diagnostic events on BLOCK/DENY.

---

## 42.43 Multi-Agent Collaboration Flow (Reference)

End-to-end example: **PM → UX → Legal → Validator → Human → Finalization**

```text
Task: "Design and validate new checkout flow for SaaS product"

1. INTAKE → CLASSIFICATION → PLANNING
   Plan nodes: [pm_spec, ux_flow, legal_review, compliance_validate, human_signoff, finalize]

2. AGENT_SELECTION + STEP_EXECUTION: PMAgent
   → SharedTaskContext.artifacts["product_spec.md"]
   → AgentDecision(COMPLETE)

3. STEP_EXECUTION: UXAgent (sequential after pm_spec)
   → reads spec via MemoryView / SharedTaskContext
   → artifacts["ux_wireframe.json"]
   → AgentDecision(COMPLETE)

4. STEP_EXECUTION: LegalAgent
   → detects CRITICAL clause issue
   → AgentDecision(INTERRUPT, interrupt=POLICY_REVIEW_REQUIRED)
   → INTERRUPT_HANDLING → PolicyEngine → REQUEST_HUMAN

5. HUMAN_APPROVAL: operator approves exception with comment
   → RESUMED → LegalAgent step re-run or CONTINUE per policy
   → AgentDecision(COMPLETE)

6. STEP_EXECUTION: ValidatorAgent (validator scheduling mode)
   → ValidationResult(valid=true)
   → AgentDecision(COMPLETE)

7. FINALIZATION: Nexus FinalResponseComposer
   → TRACE_PERSISTENCE → COMPLETION
```

All cross-agent data via `SharedTaskContext` / artifacts — never direct calls.

**Authoring reference:** orchestration control plane (Nexus runners, `ExecutionGraph`, `DelegationSpec`, hooks, customization surfaces) — [`guides/AGENT_CREATION_GUIDE.md` Appendix I](guides/AGENT_CREATION_GUIDE.md#appendix-i--orchestration-control-plane).

**End-to-end flow reference (diagrams, edge cases, plan traceability):** [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md).

---

## 42.44 AgentEngine As Universal Executor (Summary)

```text
┌──────────────────────────────────────────┐
│            NexusLoop (Tier-1)            │
│  plan · schedule · policy · graph · HITL │
└────────────────────┬─────────────────────┘
                     │ execute_node(agent_id)
┌────────────────────▼─────────────────────┐
│           AgentEngine (Tier-1)           │
│  UAEP · middleware · steps · validation  │
│  ToolRuntime gateway · events · decisions│
└────────────────────┬─────────────────────┘
                     │ run_step(domain only)
┌────────────────────▼─────────────────────┐
│         Domain Agent (Tier-2)            │
│  pipeline · prompts · domain validation  │
│  NO runtime · NO adapters · NO globals   │
└──────────────────────────────────────────┘
```

**This is the canonical execution stack for Intergrax.**

Every new agent MUST integrate through this stack.

No exceptions without architecture decision record.

---
---

---

## 42.44 Identity, Trust, and Tenancy

Every execution MUST carry identity, scope, and data boundaries (AUDIT_MAP §4).

### 42.44.1 Identity kinds

| Kind | Examples | Propagation |
|------|----------|-------------|
| User | Human operator, API user | `tenant_id`, roles → tool policy |
| Service | Tier-3 host, worker | `service_identities` on `IdentityProfile` |
| Agent | `agent_id` on contract | Scoped tool allow-list |

### 42.44.2 Tenancy rules

- `tenant_id` REQUIRED on trace events and policy evaluation for multi-tenant hosts.
- Subagents MUST NOT inherit unrestricted parent permissions — delegation contracts cap scope.
- Secrets ONLY via integration secrets backends — never in agent code or manifests.

### 42.44.3 Code map

| Module | Role |
|--------|------|
| `fastapi_core/auth/` | API key extraction, request context |
| `applications/_shared/identity_wiring.py` | Profile → host auth |
| `runtime/architecture/tenant_security.py` | Tenant isolation verification |
| `integrations/providers/.../identity_*` | Auth0, Keycloak, WorkOS hosts |
| `tools/providers/identity/` | `identity.*` tools |

Tier-3 declares posture in `IdentityProfile`; Tier-1 enforces on execution path. **Plan:** [`plan/UNIFIED_EXECUTION_RUNTIME.md`](../plan/UNIFIED_EXECUTION_RUNTIME.md) V-REM-SEC, SEC.

---

## 42.45 Security and Data Governance

Agent-native threats MUST have explicit defenses (AUDIT_MAP §23):

| Threat | Defense module |
|--------|----------------|
| Prompt injection | `prompt_security.py` |
| Tool injection | `tool_security.py` + middleware |
| Retrieval poisoning | `retrieval_security.py`, `retrieval_security_wiring.py` |
| Tenant isolation | `tenant_security.py` |
| Audit trail | Policy + trace on governance-critical actions |

`ApplicationSecurityProfile` (Tier-3) toggles defenses per host. Wiring MUST reach `ToolRuntime` and RAG retrieval path — not documentation-only.

**Authoring:** [`guides/AGENT_CREATION_GUIDE.md` Appendix S](../guides/AGENT_CREATION_GUIDE.md).

---

## 42.46 Cost and Resource Governance

Cost control MUST be enforceable at runtime (AUDIT_MAP §24):

- budget envelopes by tenant / application / agent / model / tool,
- token and tool quotas,
- forecast and anomaly signals,
- optimization recommendations under policy constraints.

| Module | Role |
|--------|------|
| `cost_budget.py` | Budget envelopes |
| `cost_quota.py` | Quotas |
| `cost_forecast.py` | Forecasting |
| `cost_optimization.py` | Optimization loops |

`RuntimePolicyBundle.budget` merges into Nexus and UAEP. Observability emits cost signals (see [`OBSERVABILITY.md`](OBSERVABILITY.md)).

**Plan:** [`plan/UNIFIED_EXECUTION_RUNTIME.md`](../plan/UNIFIED_EXECUTION_RUNTIME.md) Phase COST.

---
