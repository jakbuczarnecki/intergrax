# AGENT_CONTRACTS_AND_ASSEMBLY — extended depth

**Parent hub:** [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](../AGENT_CONTRACTS_AND_ASSEMBLY.md)

# 22. Tier and Terminology Canon

## 22.1 Four tiers — operational definitions

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ Tier-3  APPLICATION     Product host: intake, profiles, roster, deploy │
│ Tier-2  AGENT           Domain worker: contract, pattern, on_next_step   │
│ Tier-1  NEXUS           Agent OS: graph, policy, lifecycle, trace      │
│ Tier-0  PLATFORM          Catalogs: LLM, tools, skills, RAG, memory    │
└─────────────────────────────────────────────────────────────────────────┘
```

| Term | Tier | One-sentence definition |
|------|------|-------------------------|
| **Harness (practical)** | 0+1+3 | Nexus + platform catalogs + application wiring — the governed execution environment |
| **Nexus** | 1 | Agent Operating System: **`NexusLoop`** — one `Task` lifecycle, multi-agent graphs, governance |
| **Nexus planning executor** | 1 | `planning/StepExecutor` — runs **ExecutionPlan** steps (orchestration plane); **not** agent cognitive steps §38 |
| **Harness step kernel** | 0+1 | **`HarnessKernel.execute_step`** — deterministic one agent-runtime cycle (policy, trace, gateways) §38 |
| **Agent** | 2 | Python class: `AgentContract` + **`on_next_step`** domain logic + optional cognitive pattern |
| **Agent session loop** | 2 | **`Agent.run()`** — agent decision loop until terminal; **not** NexusLoop |
| **Application** | 3 | Deployable shell: normalizes user input → `Task` → returns product output |
| **Product** | — | Business offering built from Tier-3 app + selected Tier-2 agents |

## 22.2 Runnable agent instance

A **single run** materializes:

```text
ApplicationEnvironmentProfile (Tier-3)
    + AgentRegistry entry (Tier-2 class + contract)
    + Resolved LLMProfile, ToolProfile, MemoryProfile, PolicyRules
    + Task (capability, metadata, tenant_id)
        → UnifiedTaskRunner → NexusLoop → AgentEngine → acp_run
            → agent.run(AgentRunRequest) → on_next_step loop (§29 · §38)
```

The agent **class** is registered at bootstrap; it is **invoked per graph node**, not a long-lived OS process. Internal UAEP shim details: §13.3 — not author vocabulary.

## 22.3 Responsibility matrix (detailed)

| Concern | Tier-0 | Tier-1 Nexus | Tier-2 Agent (ACP) | Tier-3 Application |
|---------|--------|--------------|--------------------|--------------------|
| User intake / chat API | adapters | — | — | **owner** |
| `Task` construction | — | consumes | — | **owner** |
| Capability routing | — | **owner** (`AgentRouter`) | declares capabilities | roster + hints |
| Multi-agent topology | — | **owner** (`GraphExecutor`) | — | `ApplicationGraphSpec` |
| Agent session loop (`acp_run`) | — | **owner** (`AgentEngine` + kernel) | `on_next_step` content | — |
| Tool invocation policy | `ToolRegistry` | `ToolRuntime` | via `step_ctx.invoke_tool` | `ToolProfile` |
| LLM calls | `LLMAdapter` | tenant scope, budgets | inside `on_next_step` / pattern | `LLMProfile` |
| Memory read/write | stores | `MemoryView` policy | via `step_ctx.memory_view` | `MemoryProfile` |
| Prompt assets | `YamlPromptRegistry` | injection | prompt ids in agent | `PromptProfile` |
| HITL / interrupt | — | **owner** | `StepOutcome.pause_hitl` | flags on profile |
| Trace / metrics | backends | event bus, hooks | emits via runtime | `ObservabilityProfile` |
| Cognitive pattern (ReAct, etc.) | — | — | **owner** (ACP library) | — |
| Domain business rules | — | — | **owner** | — |

**Rule:** If a row says Nexus **owner**, Tier-2 agents MUST NOT reimplement it privately.

---

# 23. Three Cognition Planes

Intergrax deliberately separates three planning scopes. ACP operates primarily on **Plane 2**; agents MUST understand all three.

```mermaid
flowchart TB
    subgraph P1["Plane 1 — Nexus task cognition"]
        TC[TaskClassifier]
        TP[TaskPlanner / Graph seed]
        NP[NexusPlan]
        TC --> TP --> NP
    end

    subgraph P2["Plane 2 — ACP agent cognition"]
        ONS[on_next_step]
        SO[StepOutcome]
        ONS --> SO
    end

    subgraph P3["Plane 3 — Tool cognition"]
        CTP[CatalogToolPlanner]
        TPD[ToolPlanDecision]
        TR[ToolRuntime]
        CTP --> TPD --> TR
    end

    NP --> GE[GraphExecutor]
    GE --> ONS
    ONS --> CTP
    ONS --> TR
```

| Plane | Question | Primary types | ACP role |
|-------|----------|---------------|----------|
| **1 — Nexus** | Which agents, what order, parallelism? | `NexusPlan`, `PlanStep` | Agent emits replan/handoff via `StepOutcome` + contract |
| **2 — ACP** | What does this agent do in one node? | `StepOutcome`, `AcpSessionState`, `AgentRunTrace` | **Primary author surface** (`on_next_step` §29) |
| **3 — Tool** | Which tools this LLM iteration? | `ToolPlanDecision` | `ReActAgent` triggers via `step_ctx.invoke_tool` or tool loop service |

**Internal bridge:** `UAEPExecutor` / `get_steps` may still execute under `HarnessKernel` — authors do not implement them (§13.3 · ACP-CLOSE-LEG-4).

**Anti-pattern ACP-AP-01:** Implementing multi-agent sequential workflows entirely inside one agent's `on_next_step` private graph without Nexus — bypasses merge policy, parallel caps, and per-node trace.

**Anti-pattern ACP-AP-02:** Nexus micromanaging tool-level ReAct loops inside `GraphExecutor` — belongs to Plane 3 or `ReActAgent` inside Plane 2.

**Canon:** [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) §5.

---

# 24. Agent Class Hierarchy

## 24.1 Target hierarchy (post-ACP)

```text
Agent (ABC)                          intergrax/agents/agent_contract.py
├── HarnessReferenceAgent (ABC)      intergrax/agents/harness_reference_agent.py
│   ├── IntergraxAgent (ABC)         intergrax/agents/authoring/base.py
│   │   └── @step linear agents
│   └── CognitiveAgent (ABC)         intergrax/agents/authoring/patterns/base.py  [ACP-1]
│       ├── ReflexAgent              patterns/reflex.py                         [ACP-2]
│       ├── ReActAgent               patterns/react.py                          [ACP-3]
│       ├── PlanExecuteAgent         patterns/plan_execute.py                   [ACP-4]
│       ├── DecompositionAgent       patterns/decomposition.py                [ACP-5]
│       └── ReflectionAgent          patterns/reflection.py                     [ACP-6]
└── (legacy non-UAEP Agent)          deprecated — AgentEngine fallback        [ACP-LEG]
```

## 24.2 Class responsibilities

| Class | Author implements | Framework provides |
|-------|-------------------|-------------------|
| `Agent` | contract, routing, validation | registry contract |
| `HarnessReferenceAgent` | `get_steps`, `run_step` | UAEP type enforcement |
| `IntergraxAgent` | `@step` methods, `build_context` | step discovery, default `decide_after_step` chain |
| `CognitiveAgent` | `perceive`, `reason`, `act`, `evaluate` | loop wiring, budget, metadata schema |
| `*PatternAgent` | domain hooks + prompts | pattern-specific state machine |

## 24.3 CognitiveAgent protocol (normative spec)

```text
class CognitiveAgent(HarnessReferenceAgent):

    # --- metadata ---
    cognitive_pattern: ClassVar[str]   # reflex | react | plan_execute | decomposition | reflection
    pattern_version: ClassVar[str]     # e.g. acp.v1

    # --- domain hooks (subclass MUST implement) ---
    async def perceive(self, ctx: RuntimeExecutionContext) -> Observation
    async def reason(self, ctx: RuntimeExecutionContext, observation: Observation) -> ReasoningResult
    async def act(self, ctx: RuntimeExecutionContext, reasoning: ReasoningResult) -> StepOutput
    def evaluate(
        self,
        ctx: RuntimeExecutionContext,
        output: StepOutput,
    ) -> AgentEvaluation                          # continue | complete | fail | replan | human

    # --- framework wired (subclass MUST NOT override without super) ---
    def get_steps(self, context: RuntimeContext) -> list[AgentStep]
    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput
    def decide_after_step(...) -> AgentDecision
```

### 24.3.1 Mapping operator mental model → UAEP

| Operator concept | ACP implementation |
|------------------|-------------------|
| `should_generate_next_step(state)` | `evaluate()` returns `CONTINUE` or pattern loop continues inside `run_step` |
| `is_final_answer(state)` | `evaluate()` returns `COMPLETE` → `decide_after_step` → `AgentDecisionType.COMPLETE` |
| `should_replan(state)` | `evaluate()` → `MODIFY_PLAN` with `suggested_plan_delta` |
| `should_request_human(state)` | `evaluate()` → `REQUEST_HUMAN` with `human_request` payload |
| Incremental state | `ctx.metadata["acp.state.v1"]` (see §25) |

**Decision helpers (ACP-7):** `intergrax/agents/authoring/decisions.py` — primary `finish()`, `continue_with()`, `pause_for_human()`, `request_replan()`, `delegate_handoff()` → `StepOutcome` factories §32.0.4; legacy UAEP `complete()` / `continue_to()` / `delegate_to()` deprecated; `to_step_outcome()` bridges `AgentDecision` for UAEP shim only.

## 24.4 Single UAEP step vs internal micro-loop

Patterns differ in **where the loop lives**:

| Pattern | UAEP `get_steps` | Loop location | Typical `max_steps` (contract) |
|---------|------------------|---------------|-------------------------------|
| **Reflex** | 1 step | none | 1 |
| **ReAct** | 1 step | inside `run_step` (reason→act iterations) | 1 UAEP step; `max_react_iterations` in pattern |
| **Plan-execute** | 2+ steps OR 1 step with internal phases | `get_steps` chain or phased `act` | 2–10 |
| **Decomposition** | 1 step | inside `run_step` (question queue) | 1 UAEP step; `max_sub_questions` |
| **Reflection** | 1–2 steps | act → critic → revise inside `run_step` | 1–2 |

**Invariant:** Even with internal micro-loops, the agent MUST respect `ctx.should_cancel()`, budget hooks, and emit trace labels per iteration (`trace.write`).

---

# 25. Runtime Execution Context / State Model

## 25.1 RuntimeExecutionContext fields

Canonical type: `intergrax/contracts/runtime_execution_context.py`

| Field | Purpose |
|-------|---------|
| `task_id`, `run_id`, `node_id` | Correlation for trace and graph |
| `agent_id` | Contract id |
| `contract` | Resolved `AgentContract` for step |
| `metadata` | **Incremental run state** (checkpoints, ACP state, governance) |
| `tool_gateway` | Bound `ToolRuntime` facade |
| `memory_view` | Policy-scoped memory read/write |
| `trace` | `TraceWriter` for structured diagnostics |
| `request` | `RuntimeRequest` carrier (metadata bridge) |
| `domain_context` | Optional typed domain object (agent-local) |

## 25.2 ACP state envelope (`acp.state.v1`)

Stored in `ctx.metadata["acp.state.v1"]` — JSON-serializable dict for checkpoint resume.

```json
{
  "schema_version": "acp.state.v1",
  "pattern": "decomposition",
  "pattern_version": "1.0.0",
  "iteration": 3,
  "phase": "reason",
  "observation_digest": "…",
  "reasoning_trace": [
    {"step": 1, "question": "…", "answer": "…", "tools_used": ["rag.retrieve"]}
  ],
  "pending_sub_questions": ["…"],
  "final_answer_candidate": null,
  "budget": {
    "react_iterations_used": 2,
    "react_iterations_max": 8,
    "llm_calls": 5,
    "tokens_in": 900,
    "tokens_out": 300,
    "tokens_total": 1200,
    "cost_usd": 0.04
  }
}
```

**Rules:**

- Agents MUST NOT store secrets in `acp.state.v1`.
- Checkpoint resume: `UAEPAgentWithResume` + `RUNTIME_CHECKPOINT_KEY` in metadata ([`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.8).
- Nexus-owned checkpoint cursor: `UAEP_STEP_CURSOR_KEY` — do not conflate with ACP inner iteration.
- **Author code (normative — §32.0):** MUST NOT read or write `acp.state.v1` via ad-hoc `dict` keys. Use **`AcpSessionState`** (platform envelope) plus optional **agent-specific Pydantic subclass** with `extra=forbid`. Harness serializes to/from JSON at checkpoint boundaries only.
- **Budget counters (including tokens):** harness-owned — authors **read** via `load_session_state().budget`; MUST NOT increment counters in `state_delta` (ACP-AP-13).

## 25.3 Configuration injection path

```mermaid
sequenceDiagram
    participant App as Tier-3 Host
    participant Prof as ApplicationEnvironmentProfile
    participant UTR as UnifiedTaskRunner
    participant AE as AgentEngine
    participant Agent as Tier-2 Agent

    App->>Prof: wire_application_environment()
    App->>UTR: run_task(Task)
    UTR->>AE: execute node
    AE->>Agent: build_context(RuntimeRequest)
    Note over Agent: reads request.metadata profile slices
    Agent->>Agent: get_steps / run_step
```

| Config slice | Set by | Consumed in agent |
|--------------|--------|-------------------|
| `LLMProfile` | Tier-3 | `build_context` → `RuntimeContext.config` |
| `ToolProfile` / skills | Tier-3 | contract `allowed_tools` + gateway policy |
| `OrchestrationProfile` | Tier-3 | Nexus only — agent reads via metadata if needed |
| `cognitive_pattern` | Tier-2 class | `AgentContract` extension field (ACP-0) |
| `max_steps`, `risk_level` | Tier-2 contract | enforced by UAEP + policy |

**Anti-pattern ACP-AP-03:** Hardcoding `tenant_id`, API keys, or model names in agent source — use profile injection.

## 25.4 Invocation-time token usage (agent vs environment)

**Goal:** At every `on_next_step`, the agent can see **how many LLM tokens this agent run has consumed** and **how many the whole application environment has consumed** — to drive adaptive decisions (e.g. downgrade `model_hint` to a cheaper model, early-complete, or request HITL before budget exhaustion).

### 25.4.1 Scopes

| Scope | Meaning | Typical source | Author access |
|-------|---------|----------------|---------------|
| **Agent** | Cumulative tokens for **this** `agent.run()` session (all steps, including tool-loop LLM rounds) | `HarnessKernel` after each `llm_router.complete` / declarative `llm` action | `state.budget.tokens_*` on `AcpSessionState` |
| **Environment** | Cumulative tokens for the **host task / Nexus graph** (all agents + orchestration LLM if any) | Task-level aggregator → `ApplicationRunSummary.total_llm_tokens` + in-flight step | `step_ctx.invocation_usage.environment` |

**Single-agent direct `run()`:** when no multi-agent graph is active, `environment.tokens_total >= agent.tokens_total` (environment MAY include non-agent harness overhead; agent MUST NOT assume equality).

**Multi-agent Nexus graph:** environment rollup is **strictly greater or equal** to any single agent's counters — agents use environment scope for **global** budget policy and agent scope for **local** phase decisions.

### 25.4.2 Contracts

```text
AcpTokenUsage:                              # intergrax/contracts/acp_state.py
    tokens_in: int
    tokens_out: int
    tokens_total: int                        # harness-maintained: in + out (or adapter-reported total)
    llm_calls: int
    cost_usd: float

AcpBudgetState:                             # nested under acp.state.v1.budget
    steps_used, tool_calls, llm_calls
    tokens_in, tokens_out, tokens_total, cost_usd
    react_iterations_used, react_iterations_max

AcpInvocationUsageView:                     # read-only on AgentStepContext
    agent: AcpTokenUsage                     # mirror of budget token fields at step boundary
    environment: AcpTokenUsage               # task/application rollup
```

**Harness update rule (normative):** after each LLM call recorded on `AgentStepRecord.llm_calls`, the kernel MUST:

1. Increment `acp.state.v1.budget` token fields (**agent scope**).
2. Refresh `step_ctx.invocation_usage` before the **next** `on_next_step` (**both scopes**).
3. Persist environment rollup under task metadata key **`acp.usage.v1`** for checkpoint/trace correlation (authors do not write this key).

**Cross-domain:** token metering originates in [`LLM_ADAPTERS.md`](LLM_ADAPTERS.md) (`LLMUsageTracker`, `LLMMetricsCollector`); budget envelopes in [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) (`cost_budget`, `BudgetEnvelope`). ACP exposes **author-readable rollups** only — Tier-2 MUST NOT import vendor metrics SDKs.

**Plan:** ACP-TOK-1 (**Done** — `acp_token_metering_bridge.py`, `test_acp_token_usage_metering.py`).

## 25.5 Token budget limits, enforcement, and application reactions

**Goal:** Tier-3 applications assign **optional** per-agent and environment token limits. When a limit is set, the **engine** enforces it (hard stop before the next LLM call). When no limit is set, usage rollups still flow to agent state so developers implement **soft** reactions (model downgrade, early complete). **How** the environment reacts to threshold breach or hard exceed — abort, HITL, notify user, custom hook — is **fully configurable** on the application host.

### 25.5.1 Limit assignment (application → agent)

Limits are declared in Tier-3; merged into `EffectiveAgentRunEnvironment` at `run()` (§30). **None** means *no hard cap* — not zero.

| Source | Field | Scope | Typical use |
|--------|-------|-------|-------------|
| **Application environment** | `ApplicationEnvironmentProfile.cost_profile.max_total_tokens` | Task / graph (environment) | Global cap for one Nexus run |
| **Application roster** | `AgentBinding.budget_slice.max_total_tokens` | Single agent node | Per-role cap (legal vs research) |
| **Per-run request** | `AgentRunRequest.execution_options.max_total_tokens` | Single `agent.run()` | One-off tighten for intake |
| **Per-run override** | `environment_overrides` budget patch (policy-bound) | Agent or environment | Operator `configure_run` |

**Merge order (most specific wins for agent scope; environment uses min of active caps):**

```text
platform default (none)
  → cost_profile (environment)
  → AgentBinding.budget_slice (agent)
  → execution_options / environment_overrides (request)
  → policy deny on widen (STRICT hosts)
```

```text
AgentBudgetSlice:                           # intergrax/contracts/agent_budget.py
    max_total_tokens: int | null
    max_llm_calls: int | null
    enforcement: hard | advisory            # default hard when limit set
    warn_threshold_ratio: float | null        # overrides env default for this agent
```

**`advisory` enforcement:** limit is visible in state (`tokens_limit`, `tokens_remaining`) but harness does **not** block — author must react in `on_next_step`. **`hard` enforcement:** harness blocks the next LLM invocation and applies reaction policy (below).

### 25.5.2 Engine enforcement vs author soft control

| Posture | Limit assigned? | Engine behavior | Author behavior |
|---------|-----------------|-----------------|-----------------|
| **Hard enforced** | `max_total_tokens` non-null + `enforcement=hard` | Pre-LLM check in `HarnessKernel`; exceed → reaction policy; no further tokens for that scope | Read `tokens_remaining`; optional proactive downgrade **before** hard stop |
| **Advisory only** | limit set + `enforcement=advisory` | Meters usage; emits warn events at threshold; does not block | Implement downgrade / complete / `pause_hitl` in `on_next_step` |
| **No limit** | all null | Meters usage only (§25.4) | Optional soft strategy from raw `tokens_total` |

**Invariant:** metering (§25.4) is **always on** when LLM adapters report tokens — independent of whether a limit exists.

**Author-visible limit fields (read-only, harness-maintained):**

```text
AcpBudgetState / AcpTokenUsage:
    tokens_limit: int | null
    tokens_remaining: int | null              # limit - tokens_total when limit set

AcpInvocationUsageView:
    agent.*     — per-agent scope limits + usage
    environment.* — environment scope limits + usage
```

### 25.5.3 Environment reaction policies (Tier-3 configurable)

Applications configure **what happens** when a threshold is crossed or a hard limit is hit. Declared on `CostProfile.budget_reaction` ([`TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md) §43 · **Done**; Nexus `RunBudget` env cap remains **Partial** COST-1).

```text
BudgetReactionProfile:
    on_agent_limit_exceeded: abort | hitl | degrade_model | notify_only | custom_hook
    on_environment_limit_exceeded: abort | hitl | pause_graph | notify_only | custom_hook
    notify_channels: list[in_app | webhook | slack | email | trace_only]
    warn_threshold_ratio: float = 0.80        # emit BUDGET_THRESHOLD before hard exceed
    custom_hook_id: str | null                # host-registered callback id
    user_message_template: str | null         # surfaced to end user when notify_channels includes in_app
```

| Reaction | Engine effect | User / operator surface |
|----------|---------------|---------------------------|
| **abort** | `StepOutcome.fail` / run `status=failed`, `terminal_reason=budget_exceeded`, `AgentRunError(BUDGET_EXCEEDED)` | Error payload + trace; optional `user_message_template` |
| **hitl** | `StepOutcome.pause_hitl` / Nexus HITL runner; resume after approval | HITL ticket + governance snapshot §29 |
| **degrade_model** | Force `StepLLMRouter` to cheapest allowed model for subsequent steps | Trace warning; agent may observe lower `model_id` — **target:** unify with `BudgetExceededDegradeRule` ([`LLM_ADAPTERS.md`](LLM_ADAPTERS.md) M-LLM-X.9.6 · [ADR-LLM-003](../adr/entries/2026-06-19/ADR-LLM-003.md)) |
| **notify_only** | Run continues (advisory exceed) or soft-stop per binding; notifications fired | Webhook/Slack/email via integration slugs |
| **custom_hook** | Host invokes registered `BudgetReactionHook` with structured payload | Application-defined (dashboard, billing, paging) |
| **pause_graph** | Nexus pauses graph execution (environment exceed only) | ApplicationRunSummary + task status |

**Notification wiring:** channels reference **integration slugs** on the host (`notification_channel`, webhook tools) — not vendor SDKs in Tier-2. See `notify_tool_wiring` pattern on Tier-3 hosts.

**Events (normative):** harness emits `RuntimeEvent` payloads:

- `BUDGET_THRESHOLD` — `tokens_total / tokens_limit >= warn_threshold_ratio`
- `BUDGET_EXCEEDED` — hard limit crossed (includes scope: `agent` | `environment`, limit source, counters)

Subscribers: observability spine, application hooks, FastAPI SSE/WebSocket bridges — configured per host.

### 25.5.4 Application configuration example

```python
# applications/my_product/manifest.py
AgentBinding.mount(
    LegalAgent,
    factory=build_legal_agent,
    budget_slice=AgentBudgetSlice(
        max_total_tokens=32_000,
        enforcement=BudgetLimitEnforcement.HARD,
        warn_threshold_ratio=0.75,
    ),
)

# host environment_profile.py
CostProfile(
    budget_enforcement_enabled=True,
    max_total_tokens=120_000,              # environment cap
    budget_reaction=BudgetReactionProfile(
        on_agent_limit_exceeded=BudgetExceededReaction.HITL,
        on_environment_limit_exceeded=BudgetExceededReaction.ABORT,
        notify_channels=[BudgetNotifyChannel.SLACK, BudgetNotifyChannel.IN_APP],
        warn_threshold_ratio=0.80,
        user_message_template="Token budget reached for this session.",
    ),
)
```

### 25.5.5 Cross-domain alignment

| Layer | Role |
|-------|------|
| [`TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md) | `CostProfile`, `AgentBinding.budget_slice`, host hook registration |
| [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) | `RunBudget`, `BudgetPolicy`, `BudgetEnforcer` (Nexus pipeline — environment scope) |
| [`LLM_ADAPTERS.md`](LLM_ADAPTERS.md) | Token metering source |
| ACP §25.4 / §32.6 / §33.4 | Author read surface + adaptive downgrade |

**Implementation status:** per-agent `AgentBinding.budget_slice` + ACP kernel pre-LLM enforcement + reaction hooks = **Done** (ACP-TOK-2 · ACP-TOK-3 · ACP-TOK-CI). Environment `CostProfile` + host `budget_reaction` wiring = **Done** (Tier-3 §43 · APP-PROD-7). Nexus `RunBudget` graph-level env cap = **Partial** (COST-1).

**Plan:** ACP-TOK-1 · ACP-TOK-2 · ACP-TOK-3 · ACP-TOK-CI — **Done**.

---

# 26. Cognitive Pattern Catalog

## 26.1 Pattern selection guide

| Pattern | When to use | User-visible behavior | Risk |
|---------|-------------|----------------------|------|
| **Reflex** | Single LLM call or deterministic transform | Immediate answer | Low |
| **ReAct** | Tool-heavy tasks, dynamic tool choice | Think → act → observe loop | Medium |
| **Plan-execute** | Known phase sequence (gather→analyze→report) | Distinct phases | Medium |
| **Decomposition** | Open-ended research, Cursor-style task breakdown | Sub-questions until confidence | Medium–High |
| **Reflection** | High-stakes outputs needing verification | Draft → critic → revise | High |

```mermaid
flowchart TD
    Start([New agent hypothesis]) --> Q1{Single shot sufficient?}
    Q1 -->|yes| R[ReflexAgent]
    Q1 -->|no| Q2{Needs dynamic tools?}
    Q2 -->|yes| RT[ReActAgent]
    Q2 -->|no| Q3{Fixed phase pipeline?}
    Q3 -->|yes| PE[PlanExecuteAgent or IntergraxAgent @step]
    Q3 -->|no| Q4{Open-ended exploration?}
    Q4 -->|yes| D[DecompositionAgent]
    Q4 -->|no| Q5{Quality gate required?}
    Q5 -->|yes| RF[ReflectionAgent]
    Q5 -->|no| RT
```

## 26.2 ReflexAgent

**Intent:** One perception → one action → complete.

```text
get_steps: [AgentStep(id="reflex_main")]
run_step:
    obs = await perceive(ctx)
    reasoning = await reason(ctx, obs)      # may be trivial passthrough
    output = await act(ctx, reasoning)
    return output
decide_after_step: COMPLETE
```

**Use cases:** echo probes, classifiers, single-shot summarization, deterministic ETL.

**Limits:** `max_react_iterations = 0`; no tool loop unless `act` calls one tool explicitly.

## 26.3 ReActAgent

**Intent:** Reason about next action, invoke tools, observe results, repeat until stop condition.

```text
run_step (single UAEP step):
    state = load_acp_state(ctx)
  WHILE iterations < max_react_iterations AND NOT should_stop(state):
        obs = await perceive(ctx)              # includes tool results from prior iter
        reasoning = await reason(ctx, obs)     # LLM: thought + planned tool calls
        output = await act(ctx, reasoning)     # ctx.invoke_tool(...) per ToolRequest
        eval = evaluate(ctx, output)
        if eval.terminal: break
        persist_acp_state(ctx, state)
    return final StepOutput
decide_after_step: COMPLETE | FAIL | REQUEST_HUMAN per evaluate
```

**Integration:** Plane 3 `CatalogToolPlanner` may assist tool selection; `ReActAgent` MAY call `ctx.invoke_tool` directly with schemas from `contract.allowed_tools`.

**Cross-plan:** [`plan/TOOLS.md`](../plan/TOOLS.md) **TOOL-ENG-6** (bounded ReAct tool loop) — `ReActAgent` MUST use shared budget keys in `acp.state.v1.budget`.

**Stop conditions:**

- LLM returns no tool calls and `evaluate` marks answer sufficient
- `max_react_iterations` exhausted → `FAIL` or `REQUEST_HUMAN`
- Policy denial on tool → `INTERRUPT` or `FAIL` per severity
- `ctx.should_cancel()` → cooperative exit

## 26.4 PlanExecuteAgent

**Intent:** Explicit plan phases — either multiple UAEP steps or labeled phases inside one step.

**Mode A — multi UAEP step (preferred for trace clarity):**

```text
get_steps: [plan, execute_phase_1, execute_phase_2, ..., synthesize]
decide_after_step: CONTINUE chain until last → COMPLETE
```

**Mode B — internal phase machine** (long plans with dynamic branch):

```text
get_steps: [AgentStep(id="plan_execute_main")]
run_step: switch state.phase: plan | execute | synthesize
```

**Use cases:** legal review pipelines, research gather→synthesize, multi-document workflows.

**Nexus interaction:** Global replan → `MODIFY_PLAN` when execute phase discovers new agents needed (e.g. escalate to specialist node).

## 26.5 DecompositionAgent

**Intent:** Iteratively decompose task into sub-questions (Cursor-style), answer each with tools/knowledge, converge to final answer.

```text
run_step:
    state = init with root_question from request
    WHILE NOT converged(state) AND sub_questions < max:
        q = next_open_question(state)
        obs = await perceive(ctx)           # context for q
        reasoning = await reason(ctx, obs)  # answer q + spawn child questions
        output = await act(ctx, reasoning)  # tools, memory writes
        merge_into_state(state, output)
        eval = evaluate(ctx, output)        # converged? need more tools?
    return synthesize_final_answer(state)
```

**State keys:** `pending_sub_questions`, `answered`, `reasoning_trace`, `confidence`.

**Convergence criteria (subclass):**

- `evaluate` confidence ≥ threshold
- no open questions
- budget exhausted → `REQUEST_HUMAN` or best-effort `COMPLETE` with warning

## 26.6 ReflectionAgent

**Intent:** ReAct + critic verification loop ([`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md)).

```text
run_step:
    draft = await act_after_reasoning(...)     # ReAct or PlanExecute inner
    verdict = await critic_verify(draft, ctx)  # CVL hooks / CriticProfile
    if verdict.pass: return draft
    if verdict.revise: revise and loop (max_reflection_rounds)
    if verdict.escalate: REQUEST_HUMAN / INTERRUPT
```

**Integration:** `critic_gateway.verify_reflection_draft` — reads `CriticGraphHooks` from `step_ctx.metadata` (`AcpRunContextKey.CRITIC_HOOKS`). Tier-3 attaches hooks via `build_acp_session_host_from_harness` / `ACPSessionHostContext.critic_graph_hooks`. ReflectionAgent MUST NOT import critic orchestrator SDKs directly.

**Use cases:** legal/clinical/financial outputs, contract generation, compliance summaries.

## 26.7 Pattern conformance metadata

`AgentContract` extension (ACP-0):

```text
cognitive_pattern: reflex | react | plan_execute | decomposition | reflection | custom
pattern_config: dict   # max_iterations, confidence_threshold, etc.
```

CI script `check_agent_pattern_conformance.py` (ACP-13) validates pattern class matches contract field.

---

# 27. End-to-End Execution Flows

## 27.1 Flow A — Single agent chat (S1)

```mermaid
sequenceDiagram
    participant User
    participant App as Tier-3 App
    participant NL as NexusLoop
    participant AE as AgentEngine
    participant ACP as acp_run
    participant Ag as DecompositionAgent

    User->>App: chat message
    App->>App: build Task(capability)
    App->>NL: handle_task
    NL->>NL: classify → plan (1 node)
    NL->>AE: execute node
    AE->>ACP: run(AgentRunRequest)
    loop on_next_step iterations (§29)
        ACP->>Ag: on_next_step → StepOutcome
        Ag->>Ag: perceive → reason → act → evaluate (pattern)
        Ag->>Ag: step_ctx.invoke_tool / memory_view
    end
    ACP->>ACP: AgentRunResult + trace
    NL->>NL: finalize TaskResult
    App->>User: reply
```

## 27.2 Flow B — Multi-agent sequential (S3)

```mermaid
sequenceDiagram
    participant NL as NexusLoop
    participant G as GraphExecutor
    participant A as ResearchAgent
    participant B as SynthesizerAgent

    NL->>G: NexusPlan node A
    G->>A: agent.run (Plane 2 / §29)
    A-->>G: terminal StepOutcome + artifacts in SharedTaskContext
    G->>B: agent.run with shared memory
    B-->>G: COMPLETE
    G->>NL: merge → TaskResult
```

**Rule:** Agents A and B each use own ACP pattern; **topology** is Plane 1 only.

## 27.3 Flow C — Agent requests human (S7)

```text
on_next_step → StepOutcome.pause_hitl(...)
HarnessKernel → INTERRUPT / HITL queue
Task → WAITING_FOR_HUMAN
(resume token) → same agent.run path with human_approved metadata
```

Agent MUST NOT block the event loop waiting for operator input.

## 27.4 Flow D — MODIFY_PLAN (cross-plane)

```text
DecompositionAgent.evaluate → insufficient capability
AgentDecision(MODIFY_PLAN, suggested_plan_delta)
Nexus PolicyEngine → allow/deny
NexusPlanningRunner → replan → new graph nodes
```

Use when decomposition discovers need for **another registered agent**, not internal sub-step.

## 27.5 Registration and bootstrap

```text
ApplicationManifest
    → wire_application_environment(profile)
    → build_application_registry() → AgentRegistry.register(MyAgent())
    → build_nexus_loop_from_environment()
    → UnifiedTaskRunner

Developer code path:
    python -m intergrax.scaffold new-agent analyst --capability research.deep --pattern decomposition
    → implement perceive/reason/act/evaluate in agents/analyst/
    → register in applications/*/host/wiring.py
```

---

# 28. ACP Code Map, Maturity, and Gaps

## 28.1 Code map

| Component | Status | Path |
|-----------|--------|------|
| `Agent` ABC | **Done** | `intergrax/agents/agent_contract.py` |
| `UAEPAgent` protocol | **Done** | `intergrax/agents/uaep_protocol.py` |
| `UAEPExecutor` | **Done** | `intergrax/agents/uaep.py` |
| `AgentEngine` | **Done** | `intergrax/agents/agent_engine.py` |
| `IntergraxAgent` + `@step` | **Done** | `intergrax/agents/authoring` |
| `HarnessReferenceAgent` | **Done** | `intergrax/agents/harness_reference_agent.py` |
| `CognitiveAgent` base | **Done** ACP-1 | `intergrax/agents/authoring/patterns/base.py` |
| Pattern classes | **Done** ACP-2–6 | `intergrax/agents/authoring/patterns/*.py` |
| Reference pattern probes | **Done** ACP-9 | `intergrax/agents/authoring/patterns/reference.py` |
| Legacy pipeline bridge | **Removed** (LEG-3 · LEG-5 Done) | ADR-FLOW-005 — ACP-only execution |
| `AgentRunRequest` / `Result` | **Done** ACP-DX-1 | `intergrax/contracts/agent_run.py` |
| `merge_environment` | **Done** ACP-DX-2 | `intergrax/agents/run_environment.py` |
| Scaffold `--pattern` | **Done** ACP-8 | `intergrax/scaffold/new_agent.py` |

## 28.2 Maturity scorecard (ACP)

| Capability | Before ACP | After ACP (2026-06-11) | Target |
|------------|------------|------------------------|--------|
| UAEP-first authoring | L3 | L3 (bridge internal) | L3 internal-only |
| Pattern library | L0 (ad hoc) | **L3** | L3 |
| Mental model clarity | L1–L2 | **L3** (§29 single entry · PAT-3) | L3 |
| Legacy path removal | L2 (dual path) | **L3** (ACP-CLOSE-LEG-5 — pipeline stack deleted) | L3 |
| ReAct + tool loop unity | L1 | **L3** (TOOL-ENG-6 · PAT-1 Done) | L3 |
| Decomposition agent DX | L0 | **L3** | L3 |
| Reflection + CVL wiring | L2 | **L3** (ACP-CLOSE-PAT-2 Done) | L3 |

## 28.3 Gap register (ACP)

**Audit sync (2026-06-13 · ACP-LC 2026-06-17):** **37 Closed** · **0 Open** · ACP-FINISH complete; Full Harness LC closeout — no open P0/P1 in domain scope.

**Audit revalidation (2026-06-19, ACP-MAINT-DOC-01):** Fleet **17/17** migrated · `check_agent_acp_close_ci.py` green (skill resolution in umbrella · production readiness mean 100%) · AS-3 `boundary_demo` migrated off author-time `allowed_tools`. Deferred cross-domain: COST-1 graph `RunBudget` cap · FAUDIT-REG.1.

| ID | Gap | Priority | Plan row | Status |
|----|-----|----------|----------|--------|
| GAP-ACP-01 | No `CognitiveAgent` base | P0 | ACP-1 | **Closed** |
| GAP-ACP-02 | No pattern classes | P0 | ACP-2–6 | **Closed** |
| GAP-ACP-03 | Dual UAEP / AgentEngine path | P0 | ACP-CLOSE-LEG-1..3 | **Closed** |
| GAP-ACP-04 | ReAct at tool layer only | P1 | ACP-CLOSE-PAT-1 · TOOL-ENG-6 | **Closed** |
| GAP-ACP-05 | `build_context` duplicates profile | P1 | ACP-CFG | **Closed** |
| GAP-ACP-06 | No scaffold `--pattern` | P1 | ACP-8 | **Closed** |
| GAP-ACP-07 | Terminology docs scattered | P1 | ACP-CLOSE-PAT-3 | **Closed** |
| GAP-ACP-08 | `acp.state.v1` / `AcpSessionState` not in contracts | **P0** | ACP-0 + ACP-DX-6 | **Closed** |
| GAP-ACP-35 | No `StepOutcome` factories | **P0** | ACP-DX-6 | **Closed** |
| GAP-ACP-09 | No typed `AgentRunRequest`/`Result` | P0 | ACP-DX-1 | **Closed** |
| GAP-ACP-10 | No `merge_environment` / per-agent binding | P0 | ACP-DX-2 | **Closed** |
| GAP-ACP-11 | Author docs still expose UAEP first | P1 | ACP-DOC.4 | **Closed** (Appendix AC); PAT-3 for residual |
| GAP-ACP-12 | No typed `on_next_step` / `StepOutcome` | P0 | ACP-STEP-1 | **Closed** |
| GAP-ACP-13 | No `AgentRunTrace` on `AgentRunResult` | P0 | ACP-OBS-1 | **Closed** |
| GAP-ACP-14 | No `ApplicationRunSummary` orchestration journal | P1 | ACP-OBS-2 | **Closed** |
| GAP-ACP-15 | No per-step LLM router on step context | P1 | ACP-LLM-1 | **Closed** |
| GAP-ACP-16 | Shared state visibility not typed (`SharedContextView`) | P2 | ACP-STATE-1 | **Closed** |
| GAP-ACP-17 | §31–§36 canon not in implementation | P0 | ACP-DOC.5 | **Closed** |
| GAP-ACP-18 | No hard AgentRunError / TerminalReason enums | P0 | ACP-CON-1 | **Closed** |
| GAP-ACP-19 | state_delta merge semantics not in contracts | P0 | ACP-CON-2 | **Closed** |
| GAP-ACP-20 | Side-effect mode unspecified in code | P1 | ACP-CON-3 | **Closed** |
| GAP-ACP-21 | Capability routing by class name in some paths | P1 | ACP-CON-6 | **Closed** |
| GAP-ACP-22 | Security guards not CI-enforced for agent gateways | P1 | ACP-CON-7 | **Closed** |
| GAP-ACP-23 | No organizational policy envelope on agent merge | P1 | ACP-ORG-1..3 | **Closed** |
| GAP-ACP-24 | No compliance metrics on policy verdicts in trace | P2 | ACP-ORG-4 | **Closed** |
| GAP-ACP-25 | No checkpoint/resume/replay beyond sketch | P0 | ACP-PROD-1 · ACP-CLOSE-PROD-1..2 | **Closed** |
| GAP-ACP-26 | No side-effect idempotency / dedupe model | P0 | ACP-PROD-2 · ACP-CLOSE-PROD-6 | **Closed** |
| GAP-ACP-27 | No tool transaction / compensation contract | P0 | ACP-PROD-3 · ACP-CLOSE-PROD-5 | **Closed** |
| GAP-ACP-28 | No formal agent threat model section | P1 | ACP-PROD-7 | **Closed** |
| GAP-ACP-29 | No data governance / privacy contract for trace/memory | P1 | ACP-PROD-8 | **Closed** |
| GAP-ACP-30 | No schema migration policy for run/trace contracts | P1 | ACP-PROD-11 | **Closed** |
| GAP-ACP-31 | SharedContextView concurrency rules unspecified | P1 | ACP-PROD-5 | **Closed** |
| GAP-ACP-32 | Artifact contract missing (loose string list) | P1 | ACP-PROD-6 | **Closed** |
| GAP-ACP-33 | Release gates / CI matrix not normative for agents | P1 | ACP-PROD-9..10 | **Closed** |
| GAP-ACP-34 | `RequestIdentity` + memory_scope not in contracts | P0 | ACP-DX-1 + ACP-DX-2 §30.9 | **Closed** |
| GAP-ACP-36 | No agent + environment token rollups in invocation state | P1 | ACP-TOK-1 §25.4 | **Closed** |
| GAP-ACP-37 | No per-agent limits + configurable exceed reactions from application | P1 | ACP-TOK-2 · ACP-TOK-3 §25.5 | **Closed** |

## 28.4 Anti-patterns (ACP)

| ID | Anti-pattern | Correct approach |
|----|--------------|------------------|
| ACP-AP-01 | Multi-agent workflow inside one `run_step` | `ApplicationGraphSpec` + Nexus graph |
| ACP-AP-02 | Nexus schedules individual tool iterations | `ReActAgent` or `CatalogToolPlanner` |
| ACP-AP-03 | Secrets/model in agent source | Tier-3 profile injection |
| ACP-AP-04 | Direct vendor SDK in Tier-2 | `ctx.invoke_tool` + Tier-0 adapters |
| ACP-AP-05 | Custom event bus from agent | `ctx.emit_event` / runtime bus only |
| ACP-AP-06 | New agent without UAEP | Scaffold + `HarnessReferenceAgent` minimum |
| ACP-AP-07 | Fat agent base with GraphExecutor | ADR-AGENT-001 rejected option |
| ACP-AP-08 | Super-agent hides multi-agent graph in opaque state | Use Nexus graph + `SharedContextView` §34; UC-3 only for single cognitive process |
| ACP-AP-09 | Ad-hoc `terminal_reason` strings | Use `TerminalReason` enum §37.5 |
| ACP-AP-10 | Mixed immediate + declarative side effects in one step | Pick one mode per step §32.8 |
| ACP-AP-11 | Raw `dict` state access (`state["plan_cursor"]`) in `on_next_step` | Typed `AcpSessionState` / agent subclass §32.0 |
| ACP-AP-12 | In-place mutation of `step_ctx.state` | Return `StepOutcome.continue_with(state_delta=…)` only §32.0.2 |
| ACP-AP-13 | Agent-maintained token/cost counters in subclass state | Read `budget` + `invocation_usage` §25.4; harness owns increments |
| ACP-AP-14 | Hardcoded token limits or exceed handling in agent | `AgentBinding.budget_slice` + `BudgetReactionProfile` §25.5 |
| ACP-AP-13 | Implicit continue — empty outcome or missing `next_action` | Explicit `StepOutcome.continue_with()` or terminal factory §32.0.3 |
| ACP-AP-14 | God-method `on_next_step` (> ~40 lines without delegation) | Phase helpers `_step_plan`, `_step_execute` §32.0.4 |
| ACP-AP-15 | Free-text `terminal_reason` or ad-hoc error strings | `TerminalReason` + `AgentRunError` enums §37.4–§37.5 |

## 28.5 Related documents

| Document | Relationship |
|----------|--------------|
| [`adr/entries/2026-06-11/ADR-AGENT-002.md`](../adr/entries/2026-06-11/ADR-AGENT-002.md) | Author `run()` facade decision |
| [`adr/entries/2026-06-11/ADR-AGENT-003.md`](../adr/entries/2026-06-11/ADR-AGENT-003.md) | Step loop + dual observability |
| [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.4–§42.7 | UAEP lifecycle, decisions |
| [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) | End-to-end narrative S1–S7 |
| [`TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md) §22–§23 · §22.6 | Application shell + profile injection §30 · hierarchical bundles (ADR-APP-003) |
| [`MEMORY.md`](MEMORY.md) · [`RAG.md`](RAG.md) · [`TOOLS.md`](TOOLS.md) | Per-agent resource planes §30 |
| [`guides/AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) | **Appendix AC** — author `run()` + patterns |
| [`plan/TOOLS.md`](../plan/TOOLS.md) TOOL-ENG-6 | Tool loop for ReActAgent |
| [`plan/CRITIC_VERIFICATION.md`](../plan/CRITIC_VERIFICATION.md) | ReflectionAgent critic hooks |

**Implementation:** Phase **ACP** **Done** (2026-06-11). **ACP-CLOSE** + **ACP-FINISH** **Done** (2026-06-13) — mutating platform gates green (`check_agent_acp_close_ci.py`). ADR-AGENT-001/002/003 accepted.

---

# 29. Author-Facing `run()` Facade

**ADR:** [ADR-AGENT-002](../adr/entries/2026-06-11/ADR-AGENT-002.md) · [ADR-AGENT-003](../adr/entries/2026-06-11/ADR-AGENT-003.md)  
**Goal:** One obvious session API for Tier-2 authors; **`on_next_step`** for domain iterations; Nexus + UAEP remain implementation details.

## 29.0 Author terminology — single canonical entry (ACP-CLOSE-PAT-3)

**Normative vocabulary** for Tier-2 session/run/step terms lives **in §29 through §29.6**. [`AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) §1 and Appendix AC **link here** — they MUST NOT redefine terms. Internal runtime names (`UAEPExecutor`, `get_steps`, `AgentEngine`) are for platform engineers only (§13 · §38).

| Term | Where defined |
|------|----------------|
| Session entry `run(AgentRunRequest)` | §29.1–§29.3 |
| Request/result field matrix | §29.2 · §29.2.1 |
| `on_next_step` / `StepOutcome` | §32.0 · §32.5 (execution); §29.4 loop |
| Rejected mental-model alternatives | §29.6 |
| Tier / plane vocabulary | §22–§23 |

## 29.1 Design principle — one agent, two entries, one engine

```text
┌──────────────────────────────────────────────────────────────────┐
│  AUTHOR                                                           │
│  class MyAgent(DecompositionAgent):                               │
│      async def reason(self, ctx, obs): ...  # domain only         │
│  result = await agent.run(AgentRunRequest(...))                   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│  FRAMEWORK (intergrax/agents/)                                    │
│  run()  = agent decision loop §38                                 │
│    loop: AgentRuntime.advance_step()                              │
│            → on_next_step()        # agent decides                │
│            → HarnessKernel.execute_step()  # harness executes     │
│        → AgentRunTrace §31 on result                              │
└────────────────────────────┬─────────────────────────────────────┘
              ┌──────────────┴──────────────┐
              │                             │
┌─────────────▼────────────┐   ┌────────────▼─────────────────────┐
│ Direct run (lab, pytest)  │   │ Task → NexusLoop → graph node     │
│ agent.run(request)        │   │ → same run/UAEP for that agent    │
└──────────────────────────┘   └──────────────────────────────────┘
```

## 29.2 `AgentRunRequest` contract (normative target)

**Shipped** (`intergrax/contracts/agent_run.py` — **ACP-DX-1 Done**). Nexus bridge maps legacy `RuntimeRequest` when needed.

```text
AgentRunRequest:
    schema_version: str = "agent_run.v1"
    input: str | dict                    # user/domain payload
    identity: RequestIdentity            # §30.9 — tenant + authenticated principal
    session_id: str | null
    correlation_id: str | null
    agent_id: str | null                 # usually from registry binding
    metadata: dict                       # host + user external parameters
    state: dict | null                   # prior acp.state.v1 or opaque resume blob
    environment_overrides: AgentEnvironmentOverrides | null   # §30.3
    execution_options: dict | null       # budgets, autonomy hints (policy-bound)

RequestIdentity:
    tenant_id: str                        # mandatory isolation boundary
    user_id: str | null                   # authenticated end-user; see §30.9 memory_scope
    principal_type: user | service | org_system   # who acts in this run
    auth_subject: str | null              # stable subject from IdentityProfile / token (sub)

AgentRunResult:
    schema_version: str = "agent_run.v1"
    status: succeeded | failed | paused | cancelled
    output: str | dict
    state: dict                          # updated incremental state (acp.state.v1)
    artifacts: list[ArtifactRef]         # §40.6 — typed refs
    structured_data: dict
    confidence: float | null
    errors: list[str]
    warnings: list[str]
    trace_id: str
    run_id: str
    trace: AgentRunTrace                 # §31 — full agent execution journal
    used_tools: list[str]                # summary rollup from trace
    cost: dict | null
    duration_ms: int
    terminal_reason: str | null          # e.g. goal_met, budget_exceeded, hitl_pause
    governance: dict | null              # HITL / interrupt resolution when paused
```

**Rules:**

- `metadata` carries **external parameters** from application/intake (Slack thread, job id, locale, feature flags) — agents read via `ctx` / hooks, not global env vars.
- **`identity` MUST be set by Tier-3 intake** from authenticated context (`IdentityProfile`) — agents MUST NOT invent `user_id` or `tenant_id`.
- When `memory_scope=user` (default §30.9), `user_id` MUST be present or harness returns `VALIDATION_FAILED`.
- `state` is **authoritative for resume** within one agent run series; Nexus checkpoint holds task-level cursor separately.
- Secrets MUST NOT appear in `state` or `metadata` without redaction at intake.
- All result fields MUST be populated per §37.1 — no ad-hoc extra top-level keys on `AgentRunResult`.
- `errors` entries MUST use **`AgentRunError`** with controlled `code` §37.4.
- `terminal_reason` MUST be from controlled vocabulary §37.5 when `status` is terminal or paused.

### 29.2.1 Field semantics (hard contract — ACP-DX-1)

| Field | Type | Semantics |
|-------|------|-----------|
| `identity` | `RequestIdentity` | **`tenant_id` + optional `user_id`** — from authenticated intake §30.9; propagated to memory namespace |
| `identity.tenant_id` | `str` | Hard boundary — all memory/RAG/trace labels |
| `identity.user_id` | `str /| null` | End-user when `principal_type=user`; required for default user-scoped memory |
| `identity.principal_type` | enum | `user` (interactive), `service` (daemon), `org_system` (org-wide background agent) |
| `input` | `str /| dict` | Domain payload after application normalization; immutable for session |
| `metadata` | `dict[str, JSONValue]` | External params; read-only for agent; host-owned schema per product |
| `state` | `dict /| null` | Wire/checkpoint blob of `acp.state.v1` on resume; authors use `AcpSessionState` §32.0 — not raw dict in Tier-2 |
| `environment_overrides` | `AgentEnvironmentOverrides /| null` | Per-run narrow of tools/memory/RAG/LLM slices §30; policy-bound |
| `execution_options` | `AgentExecutionOptions /| null` | See below |
| `trace` | `AgentRunTrace` | Authoritative Plane B journal §31 |
| `terminal_reason` | `TerminalReason /| null` | Required when `status ∈ {succeeded, failed, paused, cancelled}` |
| `governance` | `GovernanceSnapshot /| null` | HITL ticket id, pause cause, approver, resume token when paused |
| `cost` | `AgentRunCost` | Rollup: `{tokens_in, tokens_out, llm_usd, tool_units, total_usd}` |
| `duration_ms` | `int` | Wall clock session duration |
| `warnings` / `errors` | `list[AgentRunError]` | Structured; `errors` non-empty ⇒ `status=failed` unless recovered |

```text
AgentExecutionOptions:
    max_steps: int | null
    max_total_tokens: int | null                      # §25.5 per-run agent cap (policy-bound)
    max_cost_usd: float | null
    max_wall_ms: int | null
    autonomy_level: strict | balanced | exploratory   # maps to policy profile
    side_effect_mode: immediate | declarative         # §32.8; default immediate
    checkpoint_every_step: bool = true                 # §37.2
```

```text
AgentRunError:
    code: AgentRunErrorCode              # §37.4
    message: str
    step_index: int | null
    retriable: bool
    details: dict | null                 # no secrets
```

## 29.3 Two entry postures (explicit)

| Posture | Caller | When |
|---------|--------|------|
| **Direct `run`** | Test, notebook, simple 1-agent host | Fast iteration; no graph |
| **`Task` → Nexus** | Production host, multi-agent, HITL | Same agent class; graph + governance |

**Invariant:** Changing posture MUST NOT require rewriting domain hooks — only wiring in Tier-3.

## 29.4 What `run()` does internally (author MUST NOT duplicate)

```text
async def run(request: AgentRunRequest) -> AgentRunResult:
    1. validate request + contract
    2. merged = merge_environment(host_profile, agent_binding, request)   # §30
    3. runtime_request = to_runtime_request(request, merged)
    4. hooks: on_run_start(merged) optional subclass
    5. trace = AgentRunTrace(run_id=...)
    6. loop until terminal:
         step_ctx = build_step_context(merged, state, trace)
         outcome = await AgentRuntime.advance_step(self, step_ctx)
           # internally: on_next_step → HarnessKernel.execute_step
         trace.append_step(outcome.record)
         if outcome.is_terminal: break
         state = outcome.state_delta
    7. hooks: on_run_end(result) optional subclass
    8. return AgentRunResult(..., trace=trace, terminal_reason=outcome.reason)
```

Implementation note: **`AgentRuntime.advance_step`** is the stable name; **`execute_next_step`** remains a deprecated alias until ACP-STEP-2. Kernel maps to **`UAEPExecutor`** step path today.

## 29.5 Subclass extension points (flexibility)

Authors MAY override **only** these for customization without forking harness:

| Hook | Purpose | Default |
|------|---------|---------|
| **`on_next_step`** | One domain iteration — primary cognitive hook | pattern base / `@step` driver |
| `perceive` / `reason` / `act` / `evaluate` | Cognitive pattern decomposition | Pattern base → may call `on_next_step` |
| `@step` methods | Linear pipelines | `IntergraxAgent` sequential `on_next_step` |
| `configure_run(merged_env) -> dict` | Per-run tweaks (prompt ids, thresholds) | no-op |
| `merge_environment(profile, request)` | Agent-specific overlay on host profile | contract defaults |
| `on_run_start` / `on_run_end` | Telemetry side effects (no I/O bypass) | no-op |
| `validate_output(result)` | Domain validation beyond base | contract rules |

Authors MUST NOT override `run()`, **`AgentRuntime.advance_step`**, or **`HarnessKernel.execute_step`** to skip policy/trace unless in gated test doubles.

## 29.6 Mapping author mental model ↔ rejected alternatives

| User concept | Intergrax mapping |
|--------------|-------------------|
| „`run` jak Nexus” | `Agent.run()` — harness inside base |
| „pipeline agenta” | Many `on_next_step` inside one `run()` |
| "run after every step" | **`AgentRuntime.advance_step`** inside `run()` — not many external `run()` calls |
| „Nexus wykonuje plan agenta” | **No** — agent planuje w `on_next_step`; kernel wykonuje jeden cykl §38 |
| "Nexus removed" | **No** — Nexus orchestrates `Task`; `run` executes one agent node |
| „konfiguracja w klasie” | **Defaults on contract** + **runtime merge** from environment §30 |
| "full trace in run" | `AgentRunResult.trace` §31 |
| "application logs orchestration" | `ApplicationRunSummary` §31 — separate plane |

---

# 30. Per-Agent Environment and Resource Binding

**Goal:** Each agent can have **its own** memory namespaces, tool allowlists, skills, RAG/knowledge backends, and LLM posture — while the **application/environment** injects external parameters per deployment and per request.

## 30.1 Three configuration layers (merge order)

```text
Layer 1 — Platform catalog (Tier-0)
    ToolRegistry, SkillRegistry, IntegrationRegistry, RAG engines, memory stores

Layer 2 — Application environment (Tier-3)
    ApplicationEnvironmentProfile: LLMProfile, ToolProfile, MemoryProfile,
    IntegrationProfile, PromptProfile, OrchestrationProfile,
    PolicyRulesProfile, GuardrailProfile, ExecutionMode,
    OrganizationalPolicyEnvelope (optional) §39, ...

Layer 3 — Agent binding (Tier-3 roster + Tier-2 contract)
    AgentContract + AgentBinding: agent_id, skill_ids, extra_tools,
    cognitive_pattern, memory_namespace, rag_collection_id, risk,
    org_role_id (optional) §39, ...

MERGE (lowest priority → wins last):
    platform defaults
    → application profile
    → organizational policy envelope §39          # org-wide rules (simulated company)
    → agent contract/binding (+ org role slice)   # virtual employee posture
    → request.environment_overrides
    → subclass configure_run()  (domain tuning only; cannot widen tools or override org rules in STRICT)
```

```mermaid
flowchart LR
    subgraph T3["Tier-3 Host"]
        PROF["ApplicationEnvironmentProfile"]
        BIND["AgentBinding / manifest roster"]
        TASK["Task or AgentRunRequest"]
    end

    subgraph MERGE["merge_environment()"]
        M["Effective AgentRunEnvironment"]
    end

    subgraph T2["Tier-2 Agent"]
        AG["MyAgent.run()"]
        HOOKS["reason / act / ..."]
    end

    subgraph T0["Tier-0 via gateways"]
        TOOLS["ToolRuntime"]
        MEM["MemoryView"]
        RAG["RAG / retrieval tools"]
        LLM["LLMAdapter"]
    end

    PROF --> M
    BIND --> M
    TASK --> M
    M --> AG --> HOOKS
    AG --> TOOLS & MEM & RAG & LLM
```

## 30.2 `AgentEnvironmentOverrides` (per-run, from application)

```text
AgentEnvironmentOverrides:
    tool_allowlist_extra: list[str] | null      # intersection only in STRICT
    tool_denylist: list[str] | null
    skill_ids_override: list[str] | null
    memory_namespace: str | null                # explicit namespace override
    memory_scope: user | org | task | custom | null   # override contract scope §30.9
    rag_collection: str | null                   # vector store / knowledge scope
    llm_profile_slug: str | null                # must exist in host LLMProfile
    prompt_catalog_overlay: str | null
    metadata_patch: dict | null                # merged into request.metadata
```

**Application responsibilities:**

- Map HTTP/Slack/queue payload → `metadata` + optional `environment_overrides`.
- Never pass raw credentials — pass **integration slugs** resolved by Tier-3 wiring.
- Multi-agent apps set **per-node** overrides on `Task` metadata when graph nodes need different RAG scope.

## 30.3 Per-agent resource binding on contract

Extend `AgentContract` / `AgentBinding` (see ACP-0b, ACP-DX-2):

```text
AgentContract (per agent defaults):
    allowed_tools / skill_ids / extra_tools
    cognitive_pattern, pattern_config
    memory_scope: user | org | task | custom     # default user for interactive agents §30.9
    memory_namespace_template: str | null     # used when scope=custom; placeholders §30.9
    default_rag_collection: str | null
    required_integration_slugs: list[str]     # e.g. postgres, qdrant, slack
    modality_requirements: list[str] | null

AgentBinding (manifest roster entry):
    agent_id, factory, mount policy
    org_role_id: str | null                    # §39 virtual employee role
    budget_slice: AgentBudgetSlice | null     # §25.5 per-agent token/LLM limits
    memory_scope_override: user | org | task | custom | null   # §30.9
    tool_profile_slice: ToolProfile | null    # optional narrowing per agent
    memory_profile_slice: MemoryProfile | null
    integration_profile_slice: IntegrationProfile | null
    environment_preset: str | null           # named preset from host
```

**Examples:**

| Agent | Own memory | Own tools | Own knowledge base |
|-------|------------|-----------|-------------------|
| Legal | scope **user** + matter: `legal/{tenant}/{user}/{matter_id}` | `rag.retrieve`, `doc.parse` | collection `legal_clauses` |
| Research | scope **user**: `research/{tenant}/{user}` | `websearch.query`, `rag.retrieve` | collection `web_cache` |
| Org batch analyst | scope **org**: `org/{tenant}/analytics` — no `user_id` segment | internal tools | org knowledge base |
| Echo lab | scope **task** | none | none |

Implementation: at `run()` merge → `RuntimeExecutionContext.memory_view` scoped to namespace; `tool_gateway` filtered to effective allowlist; RAG via **tool** `rag.retrieve` with collection in tool args or metadata — not direct Qdrant client in agent.

## 30.4 `EffectiveAgentRunEnvironment` (runtime materialized)

Single object built once per `run()` and passed through `RuntimeExecutionContext.domain_context` or metadata key `agent_run_env.v1`:

```text
EffectiveAgentRunEnvironment:
    agent_id, tenant_id, user_id, run_id      # user_id null only when memory_scope≠user
    memory_scope: user | org | task | custom   # resolved effective scope §30.9
    resolved_memory_namespace: str             # materialized from template + identity
    llm: resolved LLM adapter + model params
    tools: effective allowlist + ToolRuntime gateway
    skills: resolved skill manifests
    memory: MemoryView + namespace + retention policy
    rag: collection ids, RetrievalService bridge config
    prompts: catalog path + agent prompt ids + org SOP overlays §39
    policy: RuntimePolicyBundle slice for this agent risk tier
    organizational: OrganizationalPolicyContext | null   # §39 — merged envelope + role
    observability: trace labels prefix "{agent_id}." + org compliance labels §39.5
```

Subclass hooks receive `merged: EffectiveAgentRunEnvironment` (ACP-DX-3). Authors read **`merged.organizational`** for active playbooks and channel rules — never hardcode org policy in agent source.

## 30.5 Flexibility patterns for derived classes

### Pattern A — Environment-driven, zero hardcoding

```text
# Subclass only implements reasoning; all backends from host:
class AnalystAgent(DecompositionAgent):
    contract_id = "analyst"
    capabilities = ("research.deep",)
    # memory_namespace_template on contract; host wires Qdrant slug
```

### Pattern B — Agent defaults + request overrides

```text
def merge_environment(self, base, request):
    ns = base.memory_namespace
    if matter_id := request.metadata.get("matter_id"):
        ns = f"{ns}/{matter_id}"
    return base.model_copy(update={"memory_namespace": ns})
```

### Pattern C — Factory injection (Tier-3)

```text
# manifest AgentBinding factory receives LabHarnessContext:
def build_analyst(ctx: LabHarnessContext) -> AnalystAgent:
    return AnalystAgent(harness=ctx, tool_profile=ctx.tool_profile)
```

Factory MUST NOT import `applications.*` from `agents` package.

### Pattern D — Multi-database / multi-knowledge

Agent uses **multiple tools** bound to different integration slugs (`postgres.legal`, `qdrant.research`) — all via `ctx.invoke_tool`; contract declares `required_integration_slugs`; host `IntegrationProfile` maps slugs to backends.

## 30.6 STRICT vs BALANCED enforcement

| Mode | Tool widening from `configure_run` | Extra tools from request |
|------|-----------------------------------|--------------------------|
| **STRICT** | Denied | Intersection with contract only | **Organizational rules mandatory** — agent cannot override §39 |
| **BALANCED** | Allowed if in host ToolProfile | Policy engine decides | Org rules enforced; limited agent-local exceptions via policy |
| **EXPLORATORY** | Lab only | Widest within host profile | Org envelope optional (lab may omit) |

## 30.7 Anti-patterns (environment)

| ID | Anti-pattern | Correct |
|----|--------------|---------|
| ENV-AP-01 | `os.environ` / `.env` read in agent hooks | Profile + `request.metadata` |
| ENV-AP-02 | Direct `QdrantClient` / `psycopg` in Tier-2 | `rag.retrieve` / integration tools |
| ENV-AP-03 | Global singleton memory for all agents | Per-agent namespace §30.3 |
| ENV-AP-04 | Application passes secrets in metadata | Secret store + integration slug |
| ENV-AP-05 | Each agent duplicates `build_context` RuntimeConfig | `merge_environment` + harness injection ACP-CFG |
| ENV-AP-06 | Org rules encoded in agent `if` statements | `OrganizationalPolicyEnvelope` + policy rules §39 |
| ENV-AP-07 | Hardcoded token limits in agent source | `AgentBinding.budget_slice` + `CostProfile` §25.5 |

## 30.8 Application token budgets and reaction wiring

**Responsibility split:**

| Tier | Owns |
|------|------|
| **Tier-3 application** | `CostProfile`, `AgentBinding.budget_slice`, `BudgetReactionProfile`, notification integration slugs, custom hook registration on host runtime |
| **Tier-1 harness** | Merge limits, meter usage (§25.4), enforce hard caps, emit `BUDGET_*` events, apply reaction policy |
| **Tier-2 agent** | Read usage/limits; optional soft strategy; MUST NOT enforce platform caps in agent code |

**Host wiring checklist:**

1. Set `cost_profile.max_total_tokens` when the whole task/graph needs a ceiling.
2. Set `AgentBinding.budget_slice` per roster entry when roles differ (e.g. cheap triage vs expensive analysis).
3. Configure `cost_profile.budget_reaction` for exceed/threshold behavior and user notification.
4. Register `BudgetReactionHook` on `HarnessHostRuntime` when `custom_hook` is used.
5. Leave limits **unset** for lab agents that should only use advisory metering — authors implement reactions in code.

**Cross-plan:** [`plan/TIER3_APPLICATION_ENVIRONMENT.md`](../plan/TIER3_APPLICATION_ENVIRONMENT.md) COST-1 extension · ACP-TOK-2.
| ENV-AP-07 | Compliance checked only post-hoc in app code | `PolicyVerdictRecord` on every step §39.5 |

## 30.8 Code map (target)

| Component | Status | Path |
|-----------|--------|------|
| `Agent.run` delegate | **Done** | `intergrax/agents/agent_contract.py` |
| `AgentRunRequest` / `Result` | **Done** ACP-DX-1 | `intergrax/contracts/agent_run.py` |
| `AgentEnvironmentOverrides` | **Done** ACP-DX-1 | `intergrax/contracts/agent_run.py` |
| `merge_environment` | **Done** ACP-DX-2 | `intergrax/agents/run_environment.py` |
| `EffectiveAgentRunEnvironment` | **Done** ACP-DX-2 | `intergrax/agents/run_environment.py` |
| `on_next_step` / `StepOutcome` | **Done** ACP-STEP-1 | `intergrax/agents/authoring/step_loop.py` |
| `AgentRuntime.advance_step` | **Done** ACP-STEP-2 | `intergrax/agents/authoring/step_loop.py` |
| `HarnessKernel.execute_step` | **Done** ACP-STEP-2b | `intergrax/runtime/kernel/step_kernel.py` |
| `execute_next_step` (alias) | Deprecated | same as `advance_step` |
| `AgentRunTrace` | **Done** ACP-OBS-1 | `intergrax/contracts/agent_run_trace.py` |
| `StepLLMRouter` | **Done** ACP-LLM-1 | `intergrax/agents/authoring/llm_router.py` |
| `SharedContextView` | **Done** ACP-STATE-1 | `intergrax/contracts/shared_context.py` |
| `OrganizationalPolicyEnvelope` | **Done** ACP-ORG-1 | `intergrax/applications/contracts/org_policy.py` |
| `OrganizationalPolicyContext` | **Done** ACP-ORG-2 | `intergrax/agents/run_environment.py` |
| Per-agent binding on manifest | **Done** ACP-DX-5 | `intergrax/applications/contracts` |
| Reference merge in lab | **Done** ACP-CFG | `intergrax/agents/reference_harness.py` |

**Cross-domain:** [`MEMORY.md`](MEMORY.md) §5 user LTM + org profile · [`TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md) `IdentityProfile` · [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) identity.

---

## 30.9 Identity, tenant/user, and memory scope

**Goal:** Every authenticated caller is bound to **`tenant_id`** and, by default, **`user_id`** for memory read/write. **Org-wide agents** (background, not acting on behalf of a single user) MAY use **`memory_scope=org`** when contract, binding, or environment explicitly allows — without per-user partitioning.

**Cross-domain:** [`MEMORY.md`](MEMORY.md) — User LTM (`tenant_id` + `user_id`), Org profile (`org_id`), Task KV.

### 30.9.1 Request identity (normative)

Tier-3 intake maps authenticated session → `RequestIdentity`:

```text
Interactive chat / HITL / user-triggered Task:
    principal_type = user
    tenant_id      = from auth / tenant resolver
    user_id        = from IdentityProfile / JWT sub (REQUIRED)
    auth_subject   = stable provider subject

Background org job / scheduler / virtual employee (org-wide):
    principal_type = org_system | service
    tenant_id      = org tenant
    user_id        = null
    memory_scope   = org (from contract/binding — REQUIRED)

Service-to-service (no end-user):
    principal_type = service
    user_id        = null unless impersonation flag in governance metadata
```

**Rules:**

- Harness MUST reject `memory_scope=user` without `user_id` → `VALIDATION_FAILED`.
- Agents MUST NOT read/write memory outside `resolved_memory_namespace` on `memory_view`.
- Cross-user reads within same tenant are **forbidden** unless `memory_scope=org` and policy allows.

### 30.9.2 Memory scope modes

| `memory_scope` | Namespace pattern (default template) | When to use |
|----------------|----------------------------------------|-------------|
| **`user`** (default) | `{agent_id}/{tenant_id}/{user_id}` or contract template | Interactive agents, per-user LTM/STM |
| **`org`** | `org/{tenant_id}/{agent_id}` or `org/{tenant_id}/shared` | Org batch jobs, virtual employees acting for company, shared playbooks |
| **`task`** | `task/{tenant_id}/{task_id}/{agent_id}` | Ephemeral task KV; optional `user_id` in metadata for audit only |
| **`custom`** | `memory_namespace_template` with placeholders | Legal matter, case id, team workspace |

**Template placeholders:** `{tenant_id}`, `{user_id}`, `{agent_id}`, `{org_id}`, `{session_id}`, `{task_id}`, plus keys from `request.metadata` (e.g. `{matter_id}`).

**Merge resolution:**

```text
effective_scope =
    request.environment_overrides.memory_scope
    ?? AgentBinding.memory_scope_override
    ?? AgentContract.memory_scope
    ?? host MemoryProfile.default_scope
    ?? user

resolved_memory_namespace = render(template, identity, metadata)
```

### 30.9.3 Write and read semantics

| Operation | user scope | org scope |
|-----------|------------|-----------|
| **Read** | Only keys under user's namespace | Org namespace; no user sub-partition |
| **Write** | Persist with `user_id` in scope key | Persist at org level; trace records `principal_type=org_system` |
| **Resume** | Prior state must match same `user_id` | Prior state matched by `tenant_id` + org namespace |
| **STRICT** | Deny if `user_id` mismatch on resume | Deny cross-tenant always |

Session/STM (chat history): tied to `session_id` **and** `user_id` when interactive — see [`MEMORY.md`](MEMORY.md) Session store.

### 30.9.4 Examples

```text
# Support agent — per authenticated customer
memory_scope: user
template: "support/{tenant_id}/{user_id}"
→ User A never sees User B's thread memory

# Nightly compliance scanner — org agent, no end-user
memory_scope: org
principal_type: org_system
user_id: null
template: "org/{tenant_id}/compliance"
→ Reads org-wide findings store; not user-partitioned

# Legal analyst with matter override
memory_scope: custom
template: "legal/{tenant_id}/{user_id}/{matter_id}"
metadata.matter_id from intake
```

### 30.9.5 Anti-patterns

| ID | Anti-pattern | Correct |
|----|--------------|---------|
| ID-AP-01 | Global memory key without tenant/user | `resolved_memory_namespace` §30.9 |
| ID-AP-02 | Agent picks `user_id` from untrusted metadata | Tier-3 sets `RequestIdentity` from auth only |
| ID-AP-03 | Org agent with `memory_scope=user` and null user | `memory_scope=org` on contract |
| ID-AP-04 | Shared org memory readable by wrong tenant | `tenant_id` on every store operation |

**Plan:** **ACP-DX-1** includes `RequestIdentity`; **ACP-DX-2** resolves scope in `merge_environment`; test: user isolation + org agent without user_id.

---

# 31. Dual Observability: Application and Agent Planes

**ADR:** [ADR-AGENT-003](../adr/entries/2026-06-11/ADR-AGENT-003.md)  
**Observability spine:** [`OBSERVABILITY.md`](OBSERVABILITY.md) §1.2  
**Goal:** Application logs **orchestration**; agent `run()` returns **execution journal** — complementary, not duplicated.

## 31.1 Two planes (normative)

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ PLANE A — Application orchestration (Tier-3 + Nexus)                     │
│ ApplicationRunSummary / Task trace                                       │
│  • which agents selected, graph edges, handoffs                          │
│  • request intake metadata, session/tenant                               │
│  • per-node AgentRunResult.status + terminal_reason (rollup)             │
│  • orchestration errors, HITL gates, task-level pause/resume             │
│  • NOT: internal tool args, per-step LLM prompts inside one agent        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    each graph node  │  agent.run(request)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PLANE B — Agent execution (Tier-2 session)                               │
│ AgentRunTrace on AgentRunResult.trace                                    │
│  • step_index, step_id, cognitive_phase (optional)                       │
│  • decisions, state deltas (redacted acp.state.v1 snapshot)            │
│  • tool invocations: tool_id, latency, status, policy verdict            │
│  • RAG: collection, query hash, hit count, citation ids                  │
│  • LLM: model_id, adapter, tokens, latency (no raw secrets)            │
│  • errors/warnings per step, budget counters                             │
└─────────────────────────────────────────────────────────────────────────┘
```

## 31.2 `AgentRunTrace` contract (target — ACP-OBS-1)

**Shipped:** `intergrax/contracts/agent_run_trace.py` (**ACP-OBS-1 Done**).

```text
AgentRunTrace:
    schema_version: str = "agent_run_trace.v1"
    run_id: str
    agent_id: str
    correlation_id: str | null
    started_at: datetime
    ended_at: datetime | null
    steps: list[AgentStepRecord]

AgentStepRecord:
    step_index: int
    step_id: str
    started_at / ended_at
    status: succeeded | failed | skipped | paused
    decision: str | null                    # human-readable decision label
    state_snapshot: dict | null             # redacted incremental state
    tool_calls: list[ToolCallRecord]
    rag_calls: list[RagCallRecord]
    llm_calls: list[LlmCallRecord]
    events: list[str]                       # RuntimeEvent ids or compact refs
    policy_verdicts: list[PolicyVerdictRecord]   # §39.5 org + platform rules
    error: str | null
```

**Rules:**

- Harness **`HarnessKernel.execute_step`** MUST append one `AgentStepRecord` per iteration (via `AgentRuntime.advance_step`).
- Authors MUST NOT write directly to external sinks — use `step_ctx.emit_diagnostic` (policy-bound).
- Plane A consumes **`AgentRunResult`** summary fields + optional trace pointer; Plane B is authoritative for step detail.
- Aligns with [`OBSERVABILITY.md`](OBSERVABILITY.md): same `trace_id` links planes when Nexus invokes `run`.

## 31.3 `ApplicationRunSummary` (target — ACP-OBS-2)

Tier-3 host or Nexus `Task` completion emits orchestration journal:

```text
ApplicationRunSummary:
    task_id: str
    application_id: str
    graph_spec_id: str | null
    agent_invocations: list[AgentInvocationSummary]
    terminal_status: succeeded | failed | paused | cancelled
    terminal_reason: str
    duration_ms: int

AgentInvocationSummary:
    agent_id: str
    run_id: str
    node_id: str | null
    input_summary: str                      # redacted
    output_summary: str
    status: str
    terminal_reason: str | null
    trace_id: str                           # join to Plane B
```

**Rules:**

- Multi-agent prod flows MUST use **`Task → NexusLoop`** so Plane A is automatic.
- Direct `agent.run()` in lab/notebook: Plane B only; Plane A optional via host wrapper.

## 31.4 Developer experience

| Need | API |
|------|-----|
| Debug one agent quickly | `result = await agent.run(...); result.trace.steps` |
| Eval / regression on steps | Parse `AgentStepRecord.tool_calls` / `llm_calls` |
| Prod ops dashboard | Plane A `ApplicationRunSummary` + trace_id drill-down |
| HITL resume | Plane A task pause; Plane B `status=paused` on last step |

---

# 32. Agent Step Loop (`on_next_step`)

**ADR:** [ADR-AGENT-003](../adr/entries/2026-06-11/ADR-AGENT-003.md)  
**Execution stack:** §38 · **UAEP map:** `AgentRuntime.advance_step` + `HarnessKernel.execute_step` (ACP-STEP-2).

## 32.0 Author readability and typed contracts (normative)

**Foundation:** Agent authoring DX treats **readability at code-review time** as a **first-class requirement**, equal to correctness and policy safety. A reviewer MUST understand what happened in a step — success, continue, pause, policy block, validation failure — **from the author's `on_next_step` (or `@step`) source alone**, without running the application or reading harness internals.

**Hard rule — typed contracts only:** The **author-facing** step loop API accepts and returns **only strongly typed Pydantic models and enums** (`extra=forbid`). The harness MAY serialize to JSON at persistence/checkpoint boundaries; authors MUST NOT depend on untyped `dict`, `Any`, or stringly-typed control flags in domain code.

| Surface | Typed contract | Author `dict` access |
|---------|----------------|----------------------|
| Run I/O | `AgentRunRequest`, `AgentRunResult` | **Forbidden** on public fields |
| Step context | `AgentStepContext` | **Forbidden** for `state` — use `AcpSessionState` |
| Step decision | `StepOutcome` + factories | **Forbidden** — no bare dict return |
| Errors / terminal | `AgentRunError`, `AgentRunErrorCode`, `TerminalReason`, `StepNextAction` | **Forbidden** — no free-text reasons |
| State delta | `StateDelta` (typed merge patch) | Built from model `model_dump` — not hand-rolled keys |
| Environment | `EffectiveAgentRunEnvironment` | Read-only view — not `metadata` scraping |

**Rejected author surfaces (implementation MUST NOT expose):**

- `step_ctx.state: dict` without typed accessor
- `return {"is_terminal": True, ...}` or mutating `step_ctx.state[...] = ...`
- `terminal_reason: str` or `errors: list[str]` on production paths
- Implicit continue (missing outcome / default empty delta)

Legacy UAEP (`run_step`, `decide_after_step`) bridges to the same typed loop internally (ACP-STEP-3) — authors migrating SHOULD move to `on_next_step` + typed state.

### 32.0.1 Three operations every author performs every step

Every `on_next_step` iteration is **exactly three operations**. Authors MUST make each visible in source:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. READ   — load current session state (typed)                          │
│ 2. UPDATE — declare state changes (typed delta only)                    │
│ 3. DECIDE — tell harness: continue | complete | fail | pause | replan   │
└─────────────────────────────────────────────────────────────────────────┘
```

| Operation | Author API | Harness applies |
|-----------|------------|-----------------|
| **READ** | `state = self.load_session_state(step_ctx)` → `AcpSessionState` or agent subclass | `step_ctx.state` is deserialized snapshot of `acp.state.v1` |
| **UPDATE** | `state_delta = self.session_state_delta(partial_model)` or `StepOutcome.continue_with(state_delta=…)` | JSON merge patch §37.2; `_version` bump |
| **DECIDE** | **One** `StepOutcome` factory as final `return` | Loop, trace, HITL, Nexus handoff per `next_action` / `is_terminal` |

**Invariant:** The **last statement** of `on_next_step` (or each `@step` shim) MUST be `return StepOutcome.<factory>(...)`. Reviewers use that line as the **contract with the environment**.

### 32.0.2 READ — typed session state

```text
AcpSessionState:                          # platform envelope — ACP-0
    schema_version: Literal["acp.state.v1"]
    _version: int                           # harness-owned; authors read, do not set manually
    pattern: CognitivePattern | null
    phase: str | null                       # author-defined phase id (enum in subclass preferred)
    iteration: int = 0
    budget: AcpBudgetState | null              # harness-owned counters incl. tokens_in/out/total §25.4
    # … pattern-specific fields in agent subclass only

Agent-specific (recommended):
    class ResearchAgentState(AcpSessionState):
        plan_steps: list[PlanStep]
        plan_cursor: int = 0
        root_question: str | null = null
        model_config = ConfigDict(extra="forbid")
```

**Rules:**

- Authors define **one Pydantic state model per agent** (subclass of `AcpSessionState`) with `extra=forbid`.
- **READ** via `Agent.load_session_state(step_ctx) -> AcpSessionState` (framework helper — ACP-DX-6) or `ResearchAgentState.model_validate(step_ctx.state_snapshot)`.
- Authors MUST NOT use `state.get("plan_cursor")` or similar in production agent code — CI: `check_agent_typed_state.py` (ACP-DX-6).
- Optional `domain_context` on internal bridge types remains agent-local typed object — not a substitute for session state.

### 32.0.3 UPDATE — state_delta only, never in-place

Authors MUST NOT mutate `step_ctx.state`, `ctx.metadata`, or loaded Pydantic models in place and pass them back.

**Correct:**

```python
async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
    state = ResearchAgentState.model_validate(step_ctx.state_snapshot)
    next_cursor = state.plan_cursor + 1
    return StepOutcome.continue_with(
        state_delta={"plan_cursor": next_cursor, "phase": "execute"},
    )
```

**Forbidden:**

```python
step_ctx.state["plan_cursor"] += 1          # ACP-AP-12
state.plan_cursor += 1; return StepOutcome()  # in-place + implicit continue — ACP-AP-12/13
```

Delta keys MUST correspond to fields on the agent state model. Harness validates unknown keys against registered state schema when `AgentContract.state_schema` is set (ACP-0).

### 32.0.4 DECIDE — StepOutcome factories (control flow vocabulary)

Authors express **all** control flow through named factories on `StepOutcome` (ACP-DX-6). Each factory sets `is_terminal`, `next_action`, and `terminal_reason` consistently — authors MUST NOT set conflicting combinations manually.

```text
StepOutcome.continue_with(state_delta, *, diagnostics=None)
    → is_terminal=false, next_action=continue
    Meaning: "step succeeded; apply delta; run another iteration"

StepOutcome.complete(output, *, terminal_reason=goal_met, state_delta=None, …)
    → is_terminal=true, next_action implicit terminal
    Meaning: "goal met; return output to environment"

StepOutcome.fail(errors, *, terminal_reason=policy_denied|validation_failed|error, …)
    → is_terminal=true or next_action=fail
    Meaning: "unrecoverable or policy block; environment receives structured errors"

StepOutcome.pause_hitl(reason, *, governance_snapshot=None, state_delta=None)
    → is_terminal=false, next_action=pause_hitl, terminal_reason=human_required
    Meaning: "pause session; Nexus HITL runner resumes later"

StepOutcome.replan(state_delta, *, diagnostics=None)
    → is_terminal=true, terminal_reason=replanned, next_action=replan
    Meaning: "end this agent run; Nexus may schedule new run with updated plan"
```

**Reviewer checklist per step:** read the final `return StepOutcome.*` — it MUST answer:

1. **Continue?** → `continue_with` or non-terminal `pause_hitl`
2. **Done with answer?** → `complete` + `terminal_reason`
3. **Blocked / error?** → `fail` + `AgentRunError` list + `terminal_reason`
4. **Need human?** → `pause_hitl`
5. **Need external replan?** → `replan`

### 32.0.5 Code structure — readable `on_next_step`

| Rule | Limit / pattern |
|------|-----------------|
| `on_next_step` body | **≤ ~40 lines** of control flow; delegate domain work to `_step_<phase>` or pattern hooks |
| Phase routing | `match state.phase:` or early guard returns — visible at top of method |
| Preconditions | First lines: validation → `return StepOutcome.fail(...)` |
| Side effects | One mode per step §32.8 — gateways inside helpers, not scattered |
| `@step` linear agents | Framework maps each method to one `StepOutcome` — same READ/UPDATE/DECIDE rules |

Scaffold (`new-agent`) MUST emit: typed state subclass stub, `on_next_step` skeleton with phase `match`, and `return StepOutcome.*` examples (ACP-8).

### 32.0.6 What the environment learns (without reading harness)

| Author return | Environment / Nexus sees |
|---------------|--------------------------|
| `StepOutcome.complete(...)` | `AgentRunResult.status=succeeded`, `output`, `terminal_reason=goal_met` (or explicit) |
| `StepOutcome.fail(..., terminal_reason=policy_denied)` | `status=failed`, `errors[]` with `POLICY_DENIED`, trace step record |
| `StepOutcome.pause_hitl(...)` | `status=paused`, `governance` snapshot, Plane A task pause |
| `StepOutcome.continue_with(...)` | Next iteration; updated `acp.state.v1` in checkpoint |
| `StepOutcome.replan(...)` | Session ends `terminal_reason=replanned`; graph may MODIFY_PLAN |

Domain narrative belongs in `diagnostics` (typed `StepDiagnostics` model) — optional, redacted in prod traces.

### 32.0.7 Implementation modules (target)

| Module | Responsibility |
|--------|----------------|
| `intergrax/contracts/acp_state.py` | `AcpSessionState`, `AcpBudgetState`, `AcpTokenUsage`, `AcpInvocationUsageView` (ACP-0 · ACP-TOK-1) |
| `intergrax/contracts/agent_run.py` | `StepOutcome`, enums, `AgentStepContext` (ACP-DX-1, ACP-STEP-1) |
| `intergrax/agents/authoring/step_outcome.py` | Factories + validation (ACP-DX-6) |
| `intergrax/agents/authoring/state_access.py` | `load_session_state`, `session_state_delta` (ACP-DX-6) |
| `scripts/maintenance/check_agent_typed_state.py` | CI: forbid raw dict state in `agents` (ACP-DX-6) |

**Plan rows:** ACP-0, ACP-DX-1, ACP-DX-6, ACP-STEP-1, ACP-CON-1.

---

## 32.1 Session vs step (invariants)

| Level | API | Count per user request |
|-------|-----|------------------------|
| **Session** | `agent.run(AgentRunRequest)` | **1** per graph node (or 1 in direct mode) |
| **Step** | `on_next_step(AgentStepContext)` | **0..N** until terminal |

**Rejected:** application calling `agent.run()` repeatedly for each internal reasoning iteration.

## 32.2 `AgentStepContext` (target — ACP-STEP-1)

```text
AgentStepContext:
    run_id: str
    step_index: int
    input: str | dict                       # original run input + accumulated context
    state_snapshot: dict                      # internal serialization of acp.state.v1 — authors use load_session_state() §32.0
    merged_environment: EffectiveAgentRunEnvironment   # §30
    memory_view: AgentMemoryView              # namespace-scoped §30.3
    tool_gateway: ToolGateway                 # policy-bound invoke
    rag_gateway: RagGateway | null
    llm_router: StepLLMRouter                 # §33
    invocation_usage: AcpInvocationUsageView | null   # §25.4 — read-only agent + environment tokens
    shared_context: SharedContextView | null  # §34 — multi-agent only
    metadata: dict                            # request.metadata passthrough
    trace_sink: StepTraceSink                 # harness-only append helpers
```

Authors receive **views and gateways** — not raw `RuntimeExecutionContext` in public API (advanced tests may use internal types).

**Author state access (normative — §32.0):** use `Agent.load_session_state(step_ctx) -> AcpSessionState` (or agent subclass). Do **not** treat `state_snapshot` as the authoring API — it exists for harness checkpoint serialization only.

## 32.3 `StepOutcome` (target — ACP-STEP-1)

```text
StepOutcome:
    is_terminal: bool
    terminal_reason: TerminalReason | null    # required when is_terminal §37.5
    output: str | dict | null                 # final when terminal
    state_delta: StateDelta                   # §37.2 — merge patch into acp.state.v1
    next_action: continue | pause_hitl | fail | replan
    artifacts: list[ArtifactRef]              # §40.6
    confidence: float | null
    errors: list[AgentRunError]              # step-level structured errors §37.4
    diagnostics: dict | null
    requested_actions: list[StepActionRequest] | null   # declarative mode only §32.8
```

Remove ambiguous `requested_tools` hint — use **`requested_actions`** in declarative mode or **`tool_gateway.invoke`** in immediate mode.

**Harness behavior after `on_next_step` returns:**

1. Validate and apply `state_delta` per §37.2 (merge patch, version bump).
2. Execute side effects per **`side_effect_mode`** §32.8 (immediate already done in step, or run `requested_actions`).
3. Emit `RuntimeEvent`s → `AgentStepRecord` (include error codes §37.4).
4. Optional checkpoint when `checkpoint_every_step` (default true).
5. If `is_terminal`: finalize `AgentRunResult` with `terminal_reason`.
6. If `pause_hitl`: stop loop; `status=paused`; `terminal_reason=human_required`.
7. Else: increment `step_index`; enforce budgets; continue until terminal or guard §32.6.

## 32.4 `AgentRuntime.advance_step` (framework — not overridable)

One **agent iteration** — **glue only** between domain hook and harness kernel. Alias: `execute_next_step` (deprecated).

**Invariant:** `advance_step` MUST NOT contain policy engine calls, trace append, budget accounting, or state-merge logic — those belong to **`HarnessKernel.execute_step`** (§38.1 L1 · §38.3).

```text
async def AgentRuntime.advance_step(agent, step_ctx) -> StepOutcome:
    1. outcome = await agent.on_next_step(step_ctx)           # L2 — AGENT DECIDES
    2. await HarnessKernel.execute_step(outcome, step_ctx)    # L1 — HARNESS EXECUTES (policy, trace, state, budgets)
    3. return outcome
```

## 32.4b `HarnessKernel.execute_step` (deterministic primitive)

**Not** NexusLoop. **Not** agent planning. Single **harness cycle** — central deterministic primitive for one agent step (§38):

```text
async def HarnessKernel.execute_step(outcome, step_ctx) -> StepExecutionRecord:
    input:  acp.state.v1 + EffectiveAgentRunEnvironment + StepOutcome intent
    do:     policy pre-check (tools, budget, autonomy, org overlays §39 when wired)
            validate + apply state_delta (§37.2)
            run declarative requested_actions if mode=declarative (§32.8)
            policy post-check on outcome + side effects
            enforce step/session budgets (§32.6)
            record tool/RAG/LLM/memory events (immediate mode: via gateways during on_next_step)
            emit RuntimeEvents; append AgentStepRecord to run trace (Plane B)
            optional checkpoint when checkpoint_every_step
    output: StepExecutionRecord + updated state snapshot + decision metadata
```

**Target module:** `intergrax/runtime/kernel/step_kernel.py` (ACP-STEP-2b).  
**Disambiguation:** `intergrax/runtime/nexus/planning/step_executor.py` runs **ExecutionPlan** steps — different plane (§38).

## 32.5 Cognitive patterns and `@step`

| Author style | How steps are produced |
|--------------|------------------------|
| **`on_next_step` override** | Full control — super-agent, custom loops |
| **`CognitiveAgent` pattern** | Base implements `on_next_step` calling perceive→reason→act→evaluate |
| **`@step` linear** | Framework maps each `@step` to sequential `on_next_step` calls |
| **Legacy UAEP** | `run_step`/`decide_after_step` bridged to same loop (ACP-STEP-3) |

## 32.6 Budgets and termination

| Guard | Source |
|-------|--------|
| `max_steps` | contract + `execution_options` |
| token/cost budget | policy + `StepLLMRouter` + §25.4 usage rollups |
| time budget | harness timer on `run()` |
| HITL | `StepOutcome.next_action=pause_hitl` → Nexus HITL runner |

**Author visibility (normative):** before choosing `model_hint`, tool depth, or early termination, agents SHOULD read:

- **Agent scope:** `load_session_state(step_ctx).budget.tokens_total` (and `cost_usd`)
- **Environment scope:** `step_ctx.invocation_usage.environment.tokens_total` when present

When `max_total_tokens` is assigned with **hard** enforcement (§25.5), `HarnessKernel` blocks before the next LLM call and applies `BudgetReactionProfile`. When **no** limit is assigned, only metering applies — authors use `tokens_total`, `tokens_remaining`, and `warn_threshold_ratio` for soft strategy (e.g. switch to `local.fast` at 80% of an advisory cap).

## 32.7 Super-agent vs multi-agent graph (risk guard)

| Pattern | When OK | When anti-pattern |
|---------|---------|-------------------|
| **Super-agent (UC-3)** | One coherent cognitive process; sub-tasks are phases of same agent contract | Agent replaces graph: hidden planner+critic+executor roles that should be separate capabilities |
| **Multi-agent graph (UC-2)** | Distinct capabilities, handoffs via `SharedContextView` | — |

**Rule:** if another agent contract would be a better fit for a sub-task, add a Nexus graph node — do not embed a private agent roster in `acp.state.v1`. See ACP-AP-08.

## 32.8 Side-effect execution modes (normative)

Authors MUST use **one mode per step** — never mix for the same tool/RAG/LLM call.

| Mode | Author API | Harness timing |
|------|------------|------------------|
| **`immediate`** (default) | Call `tool_gateway.invoke`, `rag_gateway.retrieve`, `llm_router.complete` **inside** `on_next_step` | Trace records calls as they occur; policy enforced at invoke |
| **`declarative`** | Return `StepOutcome.requested_actions: list[StepActionRequest]` | Harness executes actions **after** `on_next_step` returns, before next step |

```text
StepActionRequest:
    kind: tool | rag | llm
    tool_id: str | null
    args: dict
    model_hint: str | null              # llm kind only
    idempotency_key: str                 # REQUIRED when kind is mutating tool §40.2
    side_effect_id: str | null           # harness-assigned if omitted
```

**Rules:**

- Default for new agents and scaffold: **`immediate`** until author opts into `execution_options.side_effect_mode=declarative`.
- Declarative mode: author MUST NOT also invoke the same gateway for the same logical action in the same step (ACP-AP-10).
- Cognitive pattern bases document which mode they use (`ReActAgent`: immediate in `act`; optional declarative for batch tool plans).

---

# 33. Per-Step LLM Routing

**Goal:** Author picks **model per step** (local vs frontier) within host allowlist; harness enforces policy.

## 33.1 `StepLLMRouter` (target — ACP-LLM-1)

```text
StepLLMRouter:
    async def complete(prompt_bundle, *, model_hint: str | null) -> LlmStepResult
    def list_allowed_models() -> list[str]
    @property effective_model: str              # after policy resolution
```

**Rules:**

- `model_hint` from author MUST be in merged `LLMProfile.allowed_models` unless BALANCED/EXPLORATORY policy widens.
- STRICT production hosts: unknown hint → policy deny or default model + warning in trace.
- All LLM calls recorded in `AgentStepRecord.llm_calls`.
- No direct `openai` / vendor imports in Tier-2 — router uses Tier-0 adapters via Nexus policy.

## 33.2 Typical author pattern

```text
async def on_next_step(self, ctx):
    if ctx.step_index == 0:
        ctx.llm_router.set_hint("local.fast")      # classify / extract
    else:
        ctx.llm_router.set_hint("frontier.reasoning")
    ...
```

## 33.3 Environment merge interaction §30

Merge order resolves default model; per-step hint overrides for **that step only**:

```text
host LLMProfile → AgentBinding.llm_slice → configure_run → StepLLMRouter.set_hint
```

## 33.4 Token-aware model selection (adaptive downgrade)

Authors MAY change `model_hint` per step based on §25.4 usage — without importing cost SDKs or reading Nexus internals.

```python
async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
    state = self.load_session_state(step_ctx)
    budget = state.budget
    env = (
        step_ctx.invocation_usage.environment
        if step_ctx.invocation_usage is not None
        else None
    )
    agent_tokens = budget.tokens_total if budget is not None else 0
    env_tokens = env.tokens_total if env is not None else agent_tokens

    # Example: downgrade when environment burn is high but agent still has work
    model_hint = "frontier.reasoning"
    if env_tokens > 40_000 or (budget is not None and budget.cost_usd > 0.50):
        model_hint = "local.fast"

    result = await step_ctx.llm_router.complete(prompt, model_hint=model_hint)
    ...
```

**Rules:**

- Downgrade MUST stay within `LLMProfile.allowed_models` — router resolves hints; policy denies unknown models on STRICT hosts.
- Agents MUST NOT maintain parallel token counters in agent state subclasses — use harness rollups only (ACP-AP-13).
- `AgentRunResult.cost` remains the **final** agent-run rollup; `invocation_usage` is the **in-flight** decision surface.

---

# 34. Shared State and Cross-Agent Visibility

**Goal:** Multi-agent graphs share **explicit** facts without agents reading Nexus internals.

## 34.1 Visibility matrix

| Data | Agent private | Shared (graph) | Nexus / application |
|------|---------------|----------------|---------------------|
| `acp.state.v1` | **Yes** — per run | No | Checkpoint blob only |
| Tool results in step | Yes until published | Optional via `shared_context.publish` | Audit rollup |
| User intake metadata | Read-only via ctx | Read-only | **Owner** |
| Agent selection / routing | No | No | **Owner** |
| Secrets | No direct access | No | Integration profile |
| Prior agent output | Via `shared_context` handoff | **Yes** | Summarized in Plane A |

## 34.2 `SharedContextView` (target — ACP-STATE-1)

Full concurrency rules: §40.5.

```text
SharedContextView:
    get(key, default) -> (value, version)
    publish(key, value, *, visibility: node | subgraph | task, expected_version: int | null) -> PublishResult
    compare_and_swap(key, expected_version, new_value) -> bool
    keys() -> list[str]
```

**Rules:**

- Available when `run()` invoked from Nexus graph node with task shared store.
- Direct lab `run()`: `shared_context=None` — agent MUST tolerate absence.
- Agents MUST NOT import `intergrax/runtime/nexus` for graph state.

## 34.4 Handoff pattern (multi-agent)

```text
Agent A run → publishes structured_data to shared_context
Graph edge → Agent B run request.metadata["handoff_from"] = A.run_id
Agent B on_next_step → reads shared_context.get("finding.summary")
```

---

# 35. Use-Case Catalog (Agent + Environment)

Canonical scenarios — all supported by **same** agent class + environment merge §30.

| ID | Scenario | Entry | Agent pattern | Environment |
|----|----------|-------|---------------|-------------|
| **UC-1** | Simple chat (1 agent) | Direct `agent.run()` | `on_next_step` or `@step` | Host profile + optional metadata |
| **UC-2** | Multi-agent pipeline | `Task → Nexus` graph | One class per role; `shared_context` handoffs | `ApplicationGraphSpec` + per-node `AgentBinding` |
| **UC-3** | Super-agent (plan in one class) | Direct or Nexus single node | `on_next_step` with internal plan queue in `acp.state.v1` | Wide tool/RAG binding on one agent |
| **UC-4** | Notebook / pytest iteration | Direct `run()` | Any | `LabHarnessContext` merge |
| **UC-5** | HITL approval mid-run | `Task` + HITL runner | `StepOutcome.pause_hitl` | Policy profile STRICT |
| **UC-6** | Per-step local vs frontier LLM | Direct or Nexus | `StepLLMRouter` hints §33 | `LLMProfile` with multiple allowed models |
| **UC-7** | Per-agent memory + RAG isolation | Any | Namespace in `memory_view` / `rag_gateway` | `AgentBinding` slices §30 |
| **UC-8** | Resume after checkpoint | Nexus checkpoint | Same agent; `request.state` blob | Task store + agent state |
| **UC-9** | Legal / research prod host | `Task` | Contract-declared capabilities | Tier-3 manifest roster |
| **UC-10** | Eval harness on traces | Batch direct `run()` | Any | Fixture profiles; assert on `result.trace` |
| **UC-11** | Simulated organization / virtual workforce | `Task` + org profile | Agents as **org roles**; envelope constrains all | `OrganizationalPolicyEnvelope` + role `AgentBinding` §39 |

**Flexibility rule:** UC-2 and UC-3 are **not** mutually exclusive — choose graph vs super-agent per product scale, not per framework fork. **UC-11** stacks on UC-1/2/9 — same agent classes, different org envelopes per deployment.

---

# 36. Final Architecture: Agent + Environment Cooperation

**Synthesis** of §13, §29–§35 and ADR-AGENT-001..003.

## 36.1 Responsibility split (final)

| Layer | Delivers to author | Delivers to ops |
|-------|-------------------|-----------------|
| **Tier-3 Application** | Profiles, roster, `AgentBinding`, external params in `metadata` | `ApplicationRunSummary`, graph, HITL |
| **Tier-2 Agent class** | Override `on_next_step`; optional `configure_run` | `AgentRunResult` + `AgentRunTrace` |
| **Tier-1 Nexus** | Transparent when using `Task` | Task orchestration, checkpoints |
| **Tier-0 Harness** | Tools, memory, RAG, LLM adapters via gateways | Policy, observability spine |

## 36.2 Author workflow (target DX)

```text
1. Scaffold agent (--pattern optional) — emits typed state subclass + StepOutcome skeleton §32.0
2. Declare contract: capabilities, tools, memory, RAG, cognitive_pattern, state_schema
3. Implement on_next_step: READ (typed state) → domain work → UPDATE (state_delta) → DECIDE (StepOutcome factory)
4. Wire AgentBinding in application manifest
5. Test: await agent.run(AgentRunRequest(...)) in pytest — assert terminal_reason + trace steps
6. Prod: same agent class on Nexus graph node — zero rewrite
```

## 36.3 Speed + flexibility guarantees

| Guarantee | Mechanism |
|-----------|-----------|
| Fast local iteration | Direct `run()` without Nexus |
| No config in agent source | `merge_environment` §30 |
| Per-agent resources | Binding slices + gateways |
| Full session observability | `AgentRunTrace` §31 |
| Prod multi-agent | Nexus unchanged (ADR-AGENT-001) |
| Per-step model/tool changes | `on_next_step` + routers §32–§33 |
| Subclass freedom | Any hierarchy under `IntergraxAgent` / `CognitiveAgent` |
| **Org policy without agent forks** | `OrganizationalPolicyEnvelope` on host §39 |
| **Virtual employees** | `AgentBinding.org_role_id` + shared envelope |
| **Compliance measurable** | `PolicyVerdictRecord` + eval suites §39.5 |

## 36.4 Implementation alignment (2026-06-13 audit)

| Component | Status | Remaining |
|-----------|--------|-----------|
| Session entry | **Done** — `AgentRunRequest`/`Result` via `acp_run.py` | — |
| Step loop | **Done** — `on_next_step` → `advance_step` → `HarnessKernel` | — |
| Trace on result | **Done** — `AgentRunTrace` on `AgentRunResult` | — |
| App orchestration log | **Done** — `ApplicationRunSummary` | — |
| Per-step LLM | **Done** — `StepLLMRouter` | — |
| Environment merge | **Done** — `merge_environment` + binding slices | — |
| Production reliability | **Done** — ACP-PROD-1..12 + **ACP-CLOSE-PROD-1..8**; §40.12 reference green; mutating checkpoint/idempotency **100%** | Per-roster `production_mode` promotion (§40.15 thresholds) |
| Legacy paths | **Done** — LEG-1..5; UAEP author surface removed | — |
| ReAct + tools | **Done** — `tool_loop_step` + `react_budget` | CI-17 ACP-AP-02 gate **Done** |
| Token usage in invocation state | **Done** — ACP-TOK-1 | — |
| Per-agent limits + exceed reactions | **Done** — ACP-TOK-2 · ACP-TOK-3 · ACP-TOK-CI | Nexus `RunBudget` env cap **Partial** (COST-1) |

## 36.5 Related ADRs and plan

| Artifact | Role |
|----------|------|
| [ADR-AGENT-001](../adr/entries/2026-06-11/ADR-AGENT-001.md) | ACP patterns; Nexus stays |
| [ADR-AGENT-002](../adr/entries/2026-06-11/ADR-AGENT-002.md) | `run()` facade |
| [ADR-AGENT-003](../adr/entries/2026-06-11/ADR-AGENT-003.md) | Step loop + dual observability |
| [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) | ACP-DX, ACP-STEP, ACP-OBS, ACP-LLM, ACP-STATE, ACP-CON |

---

# 37. Pre-Implementation Operational Contracts

**Purpose:** Close audit gaps **before** code lands (ACP-CON-*). Normative for `intergrax/contracts/agent_run.py` and step loop implementation.

## 37.1 Hard session contract (summary)

Full field matrix: §29.2.1. Implementation MUST use **Pydantic models** with `extra=forbid` on `AgentRunRequest`, `AgentRunResult`, `AgentStepContext`, `StepOutcome`, `AgentRunError`, `AcpSessionState`, and agent-specific state subclasses.

**Typed-only author surface (normative — §32.0):**

- Harness MUST reject (validation error / CI failure) author code paths that return untyped dicts from `on_next_step`, mutate session state in place, or emit free-text `terminal_reason` / `errors: list[str]` on `AgentRunResult`.
- `AgentRunRequest.state` and `AgentRunResult.state` are **JSON transport** for checkpoint/resume — authors interact via **`AcpSessionState`** helpers, not raw dict keys in Tier-2 agents.
- Round-trip tests required (ACP-DX-1). State factory + merge tests (ACP-DX-6, ACP-CON-2).

## 37.2 `state_delta` semantics

`acp.state.v1` is the agent-private incremental state blob inside `AgentRunRequest.state` / result `state`.

| Rule | Semantics |
|------|-----------|
| **Merge model** | `state_delta` is a **JSON Merge Patch** (RFC 7396): shallow merge into current `acp.state.v1` |
| **Delete** | Key present with JSON `null` in delta ⇒ remove key from state |
| **Replace subtree** | Replace entire sub-object by supplying new object at key (no deep merge below first level unless `state_patch_depth=deep` in options — default **shallow**) |
| **No full replace via delta** | `state_delta` MUST NOT replace entire state root in one step unless `is_terminal` and explicit migration hook |
| **Version** | Harness maintains `acp.state.v1._version: int`, incremented after each successful apply |
| **Checkpoint** | When `checkpoint_every_step=true` (default), persist `{state, step_index, run_id}` after each step for resume |
| **Resume conflict** | If incoming `request.state._version` < checkpoint version ⇒ `VALIDATION_FAILED` unless `force_resume` governance flag |
| **Full persistence spec** | Checkpoint transaction boundaries, replay, crash recovery — **§40.1** |
| **Author read** | `load_session_state(step_ctx)` → typed `AcpSessionState` §32.0 — not ad-hoc dict |
| **Author write** | `StepOutcome.*(state_delta=…)` only — merge patch keys from typed `model_dump` subset |

```text
StateDelta = dict[str, JSONValue]   # wire format for merge engine — authors build via session_state_delta() §32.0
```

## 37.3 Side-effect boundary

See §32.8. Implementation enforces mutual exclusion per step via runtime check (ACP-CON-3).

## 37.4 `AgentRunErrorCode` (controlled taxonomy)

All failures in trace, result `errors`, and step records MUST use these codes (extensible only via ADR):

| Code | Meaning | Typical `retriable` |
|------|---------|---------------------|
| `POLICY_DENIED` | Policy engine blocked tool/LLM/RAG/memory | false |
| `TOOL_FAILED` | Tool gateway error | true |
| `LLM_FAILED` | LLM adapter error | true |
| `RAG_FAILED` | Retrieval error | true |
| `BUDGET_EXCEEDED` | Cost/token/step/time budget hit | false |
| `MAX_STEPS_EXCEEDED` | Step loop guard | false |
| `VALIDATION_FAILED` | Output/state validation | false |
| `HITL_REQUIRED` | Human approval needed | false |
| `CANCELLED` | User/task cancellation | false |
| `INTERNAL_ERROR` | Unexpected harness bug | false |

## 37.5 `TerminalReason` (controlled vocabulary)

Used on `AgentRunResult.terminal_reason`, `StepOutcome.terminal_reason`, and Plane A rollup:

| Value | When |
|-------|------|
| `goal_met` | Success — domain goal satisfied |
| `best_effort` | Terminal success with degraded quality (warnings) |
| `budget_exceeded` | Cost/token budget |
| `max_steps_exceeded` | Step limit |
| `human_required` | HITL pause |
| `policy_denied` | Terminal policy block |
| `validation_failed` | Domain or contract validation failed |
| `cancelled` | Operator/user cancel |
| `error` | Unrecoverable error (`INTERNAL_ERROR` or exhausted retries) |
| `replanned` | Agent chose replan — session ends; Nexus may start new run |
| `delegated` | Agent requests delegation to another capability (Nexus graph edge) |

Free-text reasons MUST NOT appear in production paths — map to enum + put detail in `diagnostics` / `AgentRunError.message`.

## 37.6 Capability-based routing (enforcement)

```text
Task.required_capability  →  AgentRegistry.query(capabilities contains token)
                         →  AgentBinding in manifest selects implementation class
                         →  NOT: import agents.foo.BarAgent in NexusLoop
```

Acceptance: integration test routes by `research.web_search` with two implementations registered — correct agent selected without class name in task payload (ACP-CON-6).

## 37.7 Security model (memory / RAG / tools)

| Guard | STRICT mode behavior | Verification |
|-------|----------------------|--------------|
| Tool widening | Deny tools not in merged allowlist §30.6 | Policy unit tests |
| Memory namespace | Agent reads/writes only bound namespace | `memory_view` scope tests |
| RAG collection | Collection must be in binding | gateway pre-check |
| Secrets | Never in state/metadata/trace | redaction at intake + lint |
| Vendor SDK in Tier-2 | Forbidden | `check_agents_vendor_imports.py` |
| External sinks | Only via gateways / `emit_diagnostic` | static check ACP-CON-7 |
| STRICT tool invoke | `configure_run` cannot widen tools | §30.6 |

## 37.8 Maturity note (external audit alignment)

| Dimension | Canon | Code (2026-06-13) |
|-----------|-------|-------------------|
| Mental model clarity | 9/10 | **10/10** — typed loop shipped; §32.0 CI green |
| Agent flexibility | 9/10 | **9.5/10** — patterns + scaffold `--pattern` |
| Observability spec | 9/10 | **9.5/10** — dual planes on result |
| Production readiness | 9/10 target | **9.5/10** — mutating **platform gate Done** (ACP-CLOSE-PROD-* + §40.12 + CI-1/3); per-agent `production_mode` via §40.15 scoreboard |
| DX / readability | 9/10 (§32.0) | **9.5/10** — factories + typed-state CI |
| Typed author surface | Required §32.0 | **Done** — UAEP internal bridge only (ACP-CLOSE-LEG **Done**) |

**Audit gate (2026-06-13 — post ACP-FINISH):** conceptual architecture **10/10**; platform implementation **9.5/10**; **mutating agents production-ready (platform)** — §40.12 reference checklist green (ACP-CLOSE-PROD-7); scoreboard mutating checkpoint/idempotency **100%** (ACP-CLOSE-PROD-8); compensation queue + cross-run idempotency **Done** (ACP-CLOSE-PROD-5/6); ACP-CLOSE CI-1/2/3 + ACP-TOK-CI wired in regression gate workflow (`check_agent_acp_close_ci.py` green).

**Recommended decision (accepted):** keep Nexus as Agent OS; implement `run()` + `on_next_step()` + typed contracts — do **not** merge Nexus into agent class (ADR-AGENT-001..003). **`NexusLoop` MUST NOT become the agent plan brain** — see §38.

---

# 38. Execution Responsibility Stack: NexusLoop vs Step Kernel

**Purpose:** Remove ambiguity between **application orchestration** (NexusLoop) and **deterministic agent step execution** (HarnessKernel). Prevents `nexus.run()` from sounding like the agent's reasoning engine.

## 38.1 Four layers (normative)

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ L4  Application + NexusLoop.handle_task()                               │
│     • intake → Task • agent graph • capability routing • HITL • checkpoints │
│     • ApplicationRunSummary (Plane A)                                   │
│     DOES NOT: plan inside one agent's reasoning loop                    │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ graph node invokes once per agent role
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ L3  Agent.run()  — agent decision loop                                  │
│     • merge environment §30 • loop until terminal                       │
│     • owns: "do I need a plan?", "is plan stale?", "next move?", "done?" │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ each iteration
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ L2  Agent.on_next_step()  — author domain hook                          │
│     • READ typed state • UPDATE state_delta • DECIDE StepOutcome §32.0  │
│     DOES NOT: bypass policy, mutate state in-place, return untyped dict │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ StepOutcome
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ L1  HarnessKernel.execute_step()  — deterministic runtime primitive     │
│     input: state + effective config + StepOutcome                       │
│     • policy pre/post • gateways • trace • budgets • state merge §37.2  │
│     output: StepExecutionRecord + events                                │
│     DOES NOT: choose next agent • replan domain • analyze full agent plan│
└─────────────────────────────────────────────────────────────────────────┘
         ▲
         │ AgentRuntime.advance_step() = L2 call + L1 call (glue only — no harness logic)
```

## 38.2 Canonical names and aliases

| Canonical | Layer | Role | Avoid confusing with |
|-----------|-------|------|---------------------|
| **`NexusLoop.handle_task`** | L4 | Multi-agent Task OS | — |
| **`Agent.run`** | L3 | Agent session decision loop | `nexus.run()`, repeated `run()` per micro-step |
| **`Agent.on_next_step`** | L2 | Domain decision hook | `run_step` author override |
| **`AgentRuntime.advance_step`** | L3 glue | One iteration orchestration | Nexus planning |
| **`HarnessKernel.execute_step`** | L1 | Deterministic harness cycle | **`planning/StepExecutor`** (ExecutionPlan) |
| `execute_next_step` | — | **Deprecated alias** of `advance_step` | — |
| `UAEPExecutor` / `run_step` | — | **Legacy implementation** of L1+L2 bridge | — |

**Rejected public names:** `nexus.run()`, `NexusRuntime.run()` as author-facing agent session API.

## 38.3 Decision ownership matrix

| Question | Owner |
|----------|-------|
| Which agents run in this Task? | **NexusLoop** + registry capability routing §37.6 |
| Do I need an internal plan? | **`on_next_step`** / cognitive pattern |
| Is my plan still valid? | **`on_next_step`** |
| Execute next cognitive iteration? | **`Agent.run`** loop (via `advance_step`) |
| Change LLM model this step? | **`on_next_step`** + `StepLLMRouter` §33 |
| Invoke tool / RAG / skill? | **`on_next_step`** (immediate) or **`StepOutcome.requested_actions`** (declarative) |
| Is output final? | **`on_next_step`** → `StepOutcome.is_terminal` |
| Critic / replan / HITL? | **`on_next_step`** → `next_action` / `TerminalReason` |
| Enforce policy on I/O? | **`HarnessKernel.execute_step`** |
| Record trace events? | **`HarnessKernel.execute_step`** |
| Merge state safely? | **`HarnessKernel.execute_step`** §37.2 |

## 38.4 `planning/StepExecutor` disambiguation

| Component | Path | Executes |
|-----------|------|----------|
| **HarnessKernel** | `runtime/kernel/step_kernel.py` *(target)* | One **agent runtime** cycle (ACP cognitive step) |
| **Planning StepExecutor** | `runtime/nexus/planning/step_executor.py` | **ExecutionPlan** steps (orchestration / tool-plan plane) |

Documentation and code reviews MUST NOT conflate these two "step executors".

## 38.5 End-to-end flow (correct mental model)

```text
Application normalizes input → Task
NexusLoop selects graph → node "analyst"
  → Agent.run(request)                         # once per node
       loop:
         on_next_step: "decompose question"    # agent decides
         HarnessKernel.execute_step: trace+policy+gateways
         on_next_step: "call rag.retrieve"      # agent decides
         HarnessKernel.execute_step: ...
         on_next_step: is_terminal, goal_met
       → AgentRunResult + AgentRunTrace
NexusLoop merges node output → next graph edge or Task complete
```

**Not:** NexusLoop or `nexus.run()` decomposes the question internally. **Agent** decomposes; **kernel** executes safely; **NexusLoop** orchestrates agents.

## 38.6 Implementation plan rows

| ID | Deliverable |
|----|-------------|
| ACP-STEP-2 | `AgentRuntime.advance_step` |
| ACP-STEP-2b | `HarnessKernel.execute_step` |
| ACP-STEP-3 | UAEP `run_step` bridge → advance_step + kernel |
| ACP-DOC.7 | This section §38 |

---

# 39. Organizational Policy Envelope & Virtual Workforce

**Goal:** Tier-3 **environment** can simulate an **organization** with its own procedures, regulations, and channel rules — constraining **virtual employee agents** without forking agent code. Rules must be **easy to configure**, **enforced at harness boundaries**, and **measured** in trace and ops dashboards.

**Cross-domain:** [`TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md) §22 · [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.11 · [`OBSERVABILITY.md`](OBSERVABILITY.md) §1.2

## 39.1 Concept — organization as environment, agents as roles

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  OrganizationalPolicyEnvelope (Tier-3 — one per simulated org / tenant) │
│  • code of conduct • channel policy • SOP/playbooks • scenario bindings │
│  • PolicyRulesProfile + GuardrailProfile + PromptProfile overlays       │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ applies to ALL agents in this host
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  AgentBinding.org_role_id  — virtual employee posture                   │
│  "customer_service_rep" | "legal_analyst" | "sales_assistant"         │
│  narrows: tools, prompts, RAG collections, escalation paths           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Agent (Tier-2) — domain worker                                         │
│  on_next_step: job logic ONLY — org rules come from merged env §30     │
└─────────────────────────────────────────────────────────────────────────┘
```

**Examples of org-imposed rules (data, not agent code):**

| Rule type | Configuration surface | Enforcement point |
|-----------|----------------------|-------------------|
| Never insult customers | `GuardrailProfile` + org `inline_rules` | pre/post LLM, output scan |
| Always follow scenario X | `PromptProfile` + RAG playbook collection | prompt overlay + critic eval |
| Never call — email only | `ToolProfile` deny `phone.*`, allow `email.send` | tool gateway pre-invoke |
| Always log case id in reply | `PolicyRulesProfile` + prompt overlay | validation post-step |
| Escalate above €10k | org rule → HITL trigger | policy engine → `pause_hitl` |

## 39.2 `OrganizationalPolicyEnvelope` (Tier-3 contract — ACP-ORG-1)

`intergrax/applications/contracts/org_policy.py`. Attached to `ApplicationEnvironmentProfile.organizational_policy`.

```text
OrganizationalPolicyEnvelope:
    schema_version: str = "org_policy_envelope.v1"
    organization_id: str
    display_name: str
    execution_mode: strict | balanced | exploratory     # org default; may inherit host ExecutionMode

    # Declarative rules (machine-evaluated)
    policy_rules: PolicyRulesProfile                    # rules_path → host/policy/rules/*.yaml
    guardrails: GuardrailProfile                        # tone, PII, respect, vendor scanners

    # Procedures & scenarios (human-authored, machine-injected)
    sop_catalog_path: Path | null                       # prompt catalog / playbook ids
    scenario_bindings: list[ScenarioBinding]            # intent → required playbook
    rag_playbook_collection: str | null                 # regulated knowledge base

    # Channel & action constraints
    channel_policy: ChannelPolicy
    tool_policy_overlay: ToolPolicyOverlay | null       # deny/allow patterns on top of ToolProfile
    communication_rules: CommunicationRules

    # Measurement
    compliance_profile_id: str | null                   # eval suite + dashboard template
    observability_labels: dict[str, str]               # e.g. org=acme, sector=finance

ScenarioBinding:
    scenario_id: str
    trigger: str | list[str]           # capability, metadata key, or classifier label
    required_playbook_id: str
    mandatory: bool

ChannelPolicy:
    allowed_channels: list[str]         # e.g. email, chat, ticket
    denied_channels: list[str]         # e.g. phone, sms
    default_channel: str | null

CommunicationRules:
    required_disclosures: list[str]     # prompt overlay ids
    forbidden_topics: list[str]         # policy rule refs
    tone: str | null                    # e.g. formal, empathetic
    locale_default: str | null
```

**Flexibility:** swap envelope per deployment — same `CustomerServiceAgent` class runs under **strict bank** or **exploratory lab** envelope via host profile only.

## 39.3 `OrganizationalPolicyContext` (runtime — ACP-ORG-2)

Materialized in `merge_environment()` → `EffectiveAgentRunEnvironment.organizational`:

```text
OrganizationalPolicyContext:
    organization_id: str
    org_role_id: str | null
    active_scenario_id: str | null
    active_playbook_ids: list[str]
    channel_policy: ChannelPolicy              # resolved effective
    effective_tool_denies: list[str]            # merged org + role + STRICT
    prompt_overlay_ids: list[str]               # SOP layers injected this run
    policy_bundle_slice: RuntimePolicyBundle    # org + role fragments §42.11.4
```

Authors MAY read `step_ctx.merged_environment.organizational` to **select playbook-consistent behavior** — MUST NOT reimplement policy checks that harness already enforces.

## 39.4 Enforcement stack (where org rules apply)

Org rules MUST NOT live only in documentation — they bind at **harness hook points**:

```text
Intake (Tier-3)
  → normalize metadata, attach org_id, scenario hints

merge_environment (ACP-DX-2)
  → merge envelope + role → OrganizationalPolicyContext

Agent.on_next_step (Tier-2)
  → domain intent only; optional playbook-aware reasoning

HarnessKernel.execute_step (Tier-0/1) — §38
  1. policy pre-check  — tool/channel/scenario allowlist
  2. prompt compose    — org SOP overlays + communication_rules
  3. gateway invoke    — tool deny (e.g. phone.*) → POLICY_DENIED
  4. guardrail scan    — input/output respect rules
  5. policy post-check — scenario completion, required disclosures
  6. PolicyVerdictRecord → AgentStepRecord §39.5
```

| Enforcement mode | Agent can override org rule? |
|------------------|------------------------------|
| **STRICT** | **No** — `configure_run` and `environment_overrides` cannot widen denied tools/channels |
| **BALANCED** | Only where `RuntimePolicyBundle` explicitly allows exception + logs verdict |
| **EXPLORATORY** | Lab — envelope optional |

**Agent remains decision owner** (what to do next); **organization remains constraint owner** (what is allowed).

## 39.5 Measurement & compliance observability (ACP-ORG-4)

Every evaluated rule produces a **`PolicyVerdictRecord`** on the step trace:

```text
PolicyVerdictRecord:
    rule_id: str
    rule_source: org_envelope | org_role | platform | guardrail
    phase: pre_step | pre_tool | post_llm | post_step
    verdict: allow | deny | warn
    code: AgentRunErrorCode | null       # POLICY_DENIED when deny
    message: str                          # redacted
    scenario_id: str | null
    playbook_id: str | null
```

**Rollups:**

| Plane | Metrics |
|-------|---------|
| **AgentRunTrace** (Plane B) | `policy_verdicts[]` per step; denial counts by `rule_id` |
| **AgentRunResult** | `compliance_summary: {deny_count, warn_count, rules_triggered[]}` |
| **ApplicationRunSummary** (Plane A) | Org-level compliance score per Task; agent role breakdown |
| **Eval / CI** | Golden scenarios per `compliance_profile_id` — assert zero `POLICY_DENIED` on happy path |

**Ops dashboards (target):** policy denial rate by org, by role, by rule_id; scenario adherence; channel violation attempts (e.g. blocked `phone.dial`).

See [`OBSERVABILITY.md`](OBSERVABILITY.md) — extend spine with `policy.verdict` event type (ACP-ORG-4).

## 39.6 Authoring workflow — virtual workforce

```text
1. Define OrganizationalPolicyEnvelope on ApplicationEnvironmentProfile
2. Add policy YAML under host/policy/rules/ (code of conduct, channel rules)
3. Register playbooks in PromptProfile / RAG collection
4. Map agents in manifest:
     AgentBinding(agent_id="cs_agent", org_role_id="customer_service_rep")
5. Agent implements on_next_step — reads merged.organizational for active playbook
6. Test: pytest with strict envelope fixture — assert trace policy_verdicts
7. Prod: same agent class, different envelope per customer org (multi-tenant)
```

**Same agent, three orgs:**

| Deployment | Envelope change | Agent code |
|------------|-----------------|------------|
| Bank STRICT | deny phone, formal tone, finance SOP | unchanged |
| Retail BALANCED | allow chat, promotional playbook | unchanged |
| Internal lab | no envelope | unchanged |

## 39.7 Anti-patterns (organizational)

| ID | Anti-pattern | Correct |
|----|--------------|---------|
| ORG-AP-01 | `if org == "acme": don't call` in agent | envelope `ChannelPolicy` |
| ORG-AP-02 | Org rules only in system prompt prose | `PolicyRulesProfile` + measurable verdicts |
| ORG-AP-03 | Per-agent duplicate compliance logic | shared envelope + role slice |
| ORG-AP-04 | Compliance audit by reading chat logs manually | `PolicyVerdictRecord` + eval suites |
| ORG-AP-05 | Agent bypasses denied tool via raw HTTP in hook | gateways only §37.7 |

## 39.8 Related documents & plan

| Document | Relationship |
|----------|--------------|
| [`TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md) §39 | **Canonical home** — org envelope, virtual workforce, APP-CON |
| [`TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md) §22 · §22.6 | `PolicyRulesProfile`, `GuardrailProfile`, profile wiring · bundle grouping (ADR-APP-003) |
| [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.11 | `RuntimePolicyBundle`, guardrails |
| [`OBSERVABILITY.md`](OBSERVABILITY.md) | Trace spine + compliance metrics |
| [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md) | Scenario/playbook validation via critic |

| Plan ID | Deliverable |
|---------|-------------|
| ACP-ORG-1 | `OrganizationalPolicyEnvelope` Pydantic model on profile |
| ACP-ORG-2 | `merge_environment` → `OrganizationalPolicyContext` |
| ACP-ORG-3 | HarnessKernel policy phases + channel/tool overlays |
| ACP-ORG-4 | `PolicyVerdictRecord` + compliance_summary on result |
| ACP-ORG-5 | Reference host fixture + eval golden scenarios |
| ACP-DOC.8 | This section §39 |

---
