# UNIFIED_EXECUTION_RUNTIME — §42.16+ runtime depth

**Parent hub:** [`UNIFIED_EXECUTION_RUNTIME.md`](../UNIFIED_EXECUTION_RUNTIME.md)

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
4. `PolicyEngine` rules — Tier-3 config + `intergrax.policy_rules` EP (S3)
5. `ValidationEngine` rules — registered validators
6. Middleware plugins — Tier-3 bootstrap (`RuntimePlugin`)
7. **Security defense plugins** — `intergrax.security_defenses` EP → S2 middleware (Phase SEC-PLANES; §42.45.3)
8. **Integration catalog** — vendor security backends (`llm_guardrail`, `secrets_store`, `identity_provider`, `security_scanner`) — S1/S2 via Tier-3 profile slugs

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
    7. Optional llm_guardrail integration (vendor scanners — §42.11.6, INTEGRATIONS §47)
```

No single layer is sufficient alone. **Guardrail catalog** (types, hooks, fail modes): §42.11.6.

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
| **Runtime bypassing** | Calling `AgentEngine` outside AgentEngine |
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

Agent-native threats MUST have explicit defenses (AUDIT_MAP §23). Security is a **runtime property of the Harness Agent OS** — not a separate tier, domain pair, or parallel execution engine.

**Canonical index:** Security & Trust Planes (§42.45.3) · guardrail hook map (§42.11.6) · ideal model §3.2–3.3 · **Plan:** [Phase SEC-PLANES](../plan/UNIFIED_EXECUTION_RUNTIME.md#phase-sec-planes--security--trust-planes-active) (Done) · [Phase SEC-PLANES-EVOL](../plan/UNIFIED_EXECUTION_RUNTIME.md#phase-sec-planes-evol--enterprise-hardening-active) (Active).

### 42.45.1 Agent-native threat map

| Threat | Plane | Defense module |
|--------|-------|----------------|
| Prompt injection | S2 | `prompt_security.py` + optional `llm_guardrail` |
| Tool injection | S2 | `tool_security.py` + `ToolInjectionDefenseMiddleware` |
| Retrieval poisoning | S2 | `retrieval_security.py`, `retrieval_security_wiring.py` |
| Cross-tenant leak | S1+S2 | `tenant_security.py`, `IdentityProfile` |
| Missing auth / scope | S1 | `IdentityProfile` + identity integrations |
| Secrets in agent code | S1 | `secrets_store` integration only (SYS-INV-17) |
| Budget / quota overrun | S3 | `BudgetPolicy`, `cost_quota.py` |
| High-risk without human | S3 | `PolicyEngine` → HITL |
| RESTRICTED data without encryption | S1+S3 | `DataClassification` + `EncryptionEnforcementMiddleware` (ENC-*) |
| Override without audit | S1 | `critical_action_signing.py`, immutable audit trail |
| Audit trail gap | S1+S3 | Policy + trace on governance-critical actions |

`ApplicationSecurityProfile` (Tier-3) toggles S2 defenses per host. Wiring MUST reach `ToolRuntime` and RAG retrieval path — not documentation-only.

### 42.45.2 Architectural decision — no separate Security tier

| Decision | Rationale |
|----------|-----------|
| **No** standalone `SecurityEngine` or 23rd domain pair | Violates **SYS-INV-10** (one canonical path per concern); duplicates UAEP + PolicyEngine |
| **No** guardrails as a separate Tier-0/Tier-1 package | Guardrails are the **enforcement surface** of Policy & Governance (§42.11.6) |
| **Yes** Security & Trust Planes as a **logical index** inside UAEP | Same pattern as modality planes ([`MODALITY.md`](MODALITY.md)) — documentation + provider catalog, not a new runtime loop |
| **Yes** modular providers and plugins | Through approved extension points (§42.21) — integrations, `policy_rules`, `intergrax.security_defenses` EP |

Governance failures MUST default to **fail-closed** on strict / CRITICAL-risk hosts (`SecurityEnvelope.strict()`).

### 42.45.3 Security and Trust Planes (canonical)

Three planes compose security on the **same UAEP hook timeline** (§42.11.6). Planes differ by **question** and **artifact**, not by separate pipelines.

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  S1 — Identity & Trust Plane                                             │
│  Who acts? Which tenant? Where do secrets live? Is the action signed?    │
├──────────────────────────────────────────────────────────────────────────┤
│  S2 — Runtime Defense Plane                                              │
│  Is this payload / tool / chunk safe? (inspection at UAEP hooks)          │
├──────────────────────────────────────────────────────────────────────────┤
│  S3 — Governance & Compliance Plane                                      │
│  May execution continue? Limits? HITL? Data classification? Org rules?   │
└──────────────────────────────────────────────────────────────────────────┘
                              ▲
                              │ vendor backends (Integration catalog)
                              └── llm_guardrail · secrets_store · identity ·
                                  security_scanner · sandbox_host · …
```

| Plane | Question | Primary artifact | Typical fail mode |
|-------|----------|------------------|-------------------|
| **S1 Trust** | Identity, tenancy, secrets, signing | Auth context, tenant scope, `CriticalActionSignature` | Fail-closed on strict hosts |
| **S2 Defense** | Agent-native threat inspection | `PromptInspectionResult`, blocked/modified `ToolRequest`, quarantined chunk | Fail-closed on CRITICAL |
| **S3 Governance** | Policy, budgets, compliance, HITL | `PolicyDecision`, `ValidationResult` | Profile-driven |

**Discipline:** S2 **inspects** (is it safe?); S3 **decides** (is it allowed?). Both emit trace. Agent-local `if` checks without trace are forbidden (§42.11.6 anti-patterns).

**Ideal model mapping:** S1 ↔ IDEAL §3.2 Identity & Trust · S2+S3 ↔ IDEAL §3.3 Policy & Governance · AUDIT_MAP §4, §5, §23.

### 42.45.4 Composition root — `SecurityEnvelope`

Tier-3 hosts declare the full trust boundary in one typed bundle (`intergrax/applications/contracts/environment_profile/bundles.py`):

```text
SecurityEnvelope
  ├── identity: IdentityProfile                    → S1
  ├── application_security: ApplicationSecurityProfile → S2 toggles
  ├── guardrails: GuardrailProfile                 → S2 vendor (llm_guardrail)
  ├── policy_rules: PolicyRulesProfile | None      → S3 custom rules
  ├── compliance: ComplianceProfile              → S3
  └── organizational_policy: OrganizationalPolicyEnvelope | None → S3 org overlays
```

**Shipped presets:** `SecurityEnvelope.lab()`, `SecurityEnvelope.strict()` (S1+S2 defense bundles), `SecurityEnvelope.production()` (S1+S2+S3 + encryption bridge) · integration preset `harness_defense_stack()`.

**Wiring entry points (Tier-3):** `wire_application_security()`, `wire_application_guardrail()`, `wire_policy_bundle()`, `build_harness_host_runtime()` — assembly validated by `security_assembly_resolver` + CI `check_harness_security_wiring.py`.

### 42.45.5 Provider and extension catalog

Modularity is delivered through **four extension surfaces** — not a monolithic engine:

| Surface | Entry point / mechanism | Plane | Shipped examples | Author extension |
|---------|-------------------------|-------|------------------|------------------|
| **Integration catalog** | `IntegrationPlugin` + profile slug | S1, S2 | Vault, Auth0, LLM Guard, Trivy, Semgrep | New integration EP |
| **Policy rules** | `intergrax.policy_rules` | S3 | `harness_lab.yaml` | Custom `PolicyRuleHandlerPlugin` |
| **Native defenses** | Profile toggles → middleware | S2 | `PromptDefenseMiddleware`, `ToolInjectionDefenseMiddleware` | Enable via `ApplicationSecurityProfile` |
| **Runtime plugins** | `RuntimePlugin` at Tier-3 bootstrap | S2, S3 | Metrics, persistence hooks | `register(bus, hooks, policy)` |
| **Security defense plugins** | `intergrax.security_defenses` | S2 | `harness.strict_injection` | `SecurityDefensePlugin` on declared `HookPoint`s |

**Authoring:** [`guides/AGENT_CREATION_GUIDE.md` Appendix H](../guides/AGENT_CREATION_GUIDE.md#appendix-h--governance-policy--observability-control-plane) · [Appendix S](../guides/AGENT_CREATION_GUIDE.md#appendix-s--security-control-plane-closeout) · [`guides/EXTENSION_AUTHOR_GUIDE.md`](../guides/EXTENSION_AUTHOR_GUIDE.md) §10 (policy rules) · §12 (security defenses).

Tier-2 agents **MUST NOT** implement parallel security — they consume a host already configured via `SecurityEnvelope`.

### 42.45.6 Execution timeline (single run)

```text
STARTUP (Tier-3)
  SecurityEnvelope → RuntimePolicyBundle + middleware list + integration slugs
  security_runtime_bridge → RuntimeConfig.security_profile + llm_guardrail slug

INTAKE
  S1: tenant verify, identity scope
  S3: PolicyEngine (autonomy ceiling, domain_fragments)

PER UAEP STEP (see §42.11.6 for full guardrail table)
  before_step     → S3: plan/step policy, context budget
  pre-LLM         → S2: prompt_security + optional llm_guardrail.scan_input()
  post-LLM        → S2: provider safety + optional scan_output()
  before_tool     → S2+S3: tool_security + ToolAccessPolicy + injection middleware
  after_tool      → trace + policy hooks
  pre-RAG inject  → S2: retrieval_security (tenant-scoped)
  pre-memory      → S3: MemoryWritePolicy
  post-step       → S3: ValidationEngine (L0) → CVL L1 if enabled

TERMINAL
  S3: unresolved interrupts → HITL or FAIL
  S1: audit trail (immutable when enabled)
  S1: critical_action_signing for override / promotion / security config change
```

### 42.45.7 Forbidden patterns

| Pattern | Why forbidden |
|---------|---------------|
| Standalone `SecurityEngine` beside UAEP | Duplicate path; SYS-INV-10 |
| Vendor guardrail SDK in Tier-2 agents | SYS-INV-17 — use Integration → middleware |
| Defense in agent code without trace | Untestable; bypasses PolicyEngine |
| Parallel policy object per agent | Single `RuntimePolicyBundle` per host |
| Harness-native blockchain / receipt product | Out of scope M.6; Tier-3 adapter pattern when product requires portable attestation |
| Tier-0 encryption SDK in agents | Use `secrets_store` integration + ENC bridge (§42.45.9) |

### 42.45.8 Maturity — Done vs planned

| Capability | Status | Plan ID |
|------------|--------|---------|
| S2 native defenses + Tier-3 wiring (SEC-1–3) | **Done** | Phase SEC |
| S2 vendor guardrails (M.12) | **Done** | M.12 / GR-INT |
| S3 policy_rules EP (DX-5.8) | **Done** | GOV-DOC.3 |
| S1 identity / secrets integrations | **Done** | M.6, H-INT-10 |
| S1 critical action signing | **Done** | AUDIT-IDEAL-4.1 |
| Security & Trust Planes canon (this section) | **Done** | SEC-PLANES-DOC.1 |
| `intergrax.security_defenses` EP | **Done** | SEC-EXT-* |
| Shipped defense bundles + `bootstrap_security_providers()` | **Done** | SEC-BUNDLE-* |
| Encryption enforcement bridge (`RESTRICTED` → secrets_store) | **Done** | ENC-* |
| Author map Appendix H / EXTENSION §12 sync | **Done** | SEC-PLANES-DOC.2–3 |

**Follow-on (enterprise hardening):** [Phase SEC-PLANES-EVOL](../plan/UNIFIED_EXECUTION_RUNTIME.md#phase-sec-planes-evol--enterprise-hardening-closed) — **Done** (2026-06-19).

| Capability | Status | Plan ID |
|------------|--------|---------|
| `bootstrap_security_providers()` in `catalog_bootstrap` | **Done** | SEC-EVOL-1 |
| Lab EP fixture + discovery gate | **Done** | SEC-EVOL-2 |
| Security spine signals (`platform.security.*`) | **Done** | SEC-EVOL-3 |
| Encrypt-via-adapter for RESTRICTED payloads | **Done** | SEC-EVOL-4 |
| Defense plugin inspection budget / timeout | **Done** | SEC-EVOL-5 |
| Enterprise maturity author checklist (§42.45.10) | **Done** | SEC-EVOL-DOC-1 |

### 42.45.9 Encryption posture matrix

| Layer | Mechanism | Owner |
|-------|-----------|-------|
| **Transit** | TLS on HTTP/MCP hosts | Tier-3 deployment / reverse proxy |
| **Secrets at rest** | `IntegrationProfile.secrets_store` slug (Vault, Doppler, …) | S1 integration catalog |
| **RESTRICTED payload gate** | `EncryptionEnforcementMiddleware` + `require_secrets_store_for_encryption` | S2/S3 profile on strict hosts — **deny** when backend missing |
| **RESTRICTED payload transform** | Encrypt via `secrets_store` integration adapter before persist/tool return | S1 integration — **Done** SEC-EVOL-4 / SEC-ENT-1 |
| **Field-level KMS** | Not in harness — use integration adapter in Tier-3 product | Out of SEC-PLANES scope |

No duplicate KMS SDK in Tier-0 — agents consume resolved secrets via platform integrations only.

### 42.45.10 Enterprise hardening — maturity model and backlog

Phase SEC-PLANES (2026-06-19) delivers a **harness-grade** Security & Trust Planes foundation. The items below close gaps identified in the post-implementation enterprise audit — they do **not** introduce a new tier or `SecurityEngine`.

| Maturity area | SEC-PLANES baseline | SEC-PLANES-EVOL target |
|---------------|---------------------|------------------------|
| **Bootstrap** | `bootstrap_security_providers()` callable; shipped bundles at import time | Auto-invoke from `catalog_bootstrap` so EP discovery is default on host startup |
| **Author / author DX** | Protocol + EP group documented | Lab fixture package in repo + CI discovery gate for third-party authors |
| **Observability** | Middleware blocks propagate via hook denial | Typed spine-adjacent domain signals: `platform.security.defense_blocked`, `platform.security.encryption_denied` |
| **Encryption runtime** | Fail-closed **deny** when `RESTRICTED` lacks `secrets_store` | Optional **encrypt-via-adapter** path when backend is configured (persist/tool output) |
| **Resilience** | Plugins run synchronously on hook path | Per-plugin inspection budget / timeout guard to limit DoS on hot paths |
| **Multi-tenant** | `TenantSecurityMiddleware` on native path | Author checklist: defense plugins MUST respect tenant scope from `HookContext` |

**Canonical plan register:** [Phase SEC-PLANES-EVOL](../plan/UNIFIED_EXECUTION_RUNTIME.md#phase-sec-planes-evol--enterprise-hardening-closed) · [Phase SEC-ENT](../plan/UNIFIED_EXECUTION_RUNTIME.md#phase-sec-ent--enterprise-production-closed).

**Explicitly out of scope:** harness-native blockchain; SOC2/ISO certification evidence; Tier-0 KMS SDK; duplicate PolicyEngine.

### 42.45.11 Enterprise production readiness

Phase SEC-ENT (2026-06-19) closes harness-scope **enterprise production** gaps identified after SEC-PLANES-EVOL.

| Capability | Mechanism | Status |
|------------|-----------|--------|
| Live secrets-store encryptor | `resolve_restricted_payload_encryptor(env)` → `SecretsStorePayloadEncryptor` | **Done** SEC-ENT-1 |
| Harness envelope fallback | `HarnessEnvelopeEncryptor` when adapter unavailable | **Done** SEC-EVOL-4 |
| Typed spine payloads | `SecurityDefenseBlockedPayloadV1`, `SecurityEncryptionDeniedPayloadV1` | **Done** SEC-ENT-2 |
| Tenant-scope defense guard | `PluginSecurityDefenseMiddleware` blocks cross-tenant before inspect | **Done** SEC-ENT-4 |
| Ops counters | `wire_security_spine_subscriber()` on host wiring | **Done** SEC-ENT-5 |
| CI spine audit | `check_harness_security_spine_signals.py` | **Done** SEC-ENT-3 |

**SIEM / ops subscribe path:** subscribe to `RuntimeEventBus` with `kind_prefix="platform.security."` or consume persisted `DOMAIN_SIGNAL` events with `ops_hint=ops:alert`. Counters available via `SecuritySpineCounters` for in-process dashboards.

**Remaining product-tier work (out of harness):** field-level KMS in Tier-3 products, SOC2 evidence packs, vendor SIEM dashboard templates.

**Authoring (wire-time closeout):** [`guides/AGENT_CREATION_GUIDE.md` Appendix S](../guides/AGENT_CREATION_GUIDE.md).

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
