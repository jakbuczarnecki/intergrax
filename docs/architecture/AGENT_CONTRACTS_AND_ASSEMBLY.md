# Agent Contracts, Registry, and Capability Model

**Status:** Canonical architecture (domain pair 1:1) · **Production coding gate:** §40 + ACP-PROD-* + **ACP-CLOSE-PROD-*** **Done** (mutating agents platform-ready)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 17–20, 31 (+ ACP cognitive patterns §21)  
**Audit instruction:** [`audit/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../audit/AGENT_CONTRACTS_AND_ASSEMBLY.md)  
**ADR:** [`adr/entries/2026-06-11/ADR-AGENT-001.md`](../adr/entries/2026-06-11/ADR-AGENT-001.md) · [`adr/entries/2026-06-11/ADR-AGENT-002.md`](../adr/entries/2026-06-11/ADR-AGENT-002.md) · [`adr/entries/2026-06-11/ADR-AGENT-003.md`](../adr/entries/2026-06-11/ADR-AGENT-003.md) — ACP · `run()` · `on_next_step` · dual observability  

> **Practical minimal authoring path:** [`guides/AGENT_AUTHOR_MINIMAL_PATH.md`](../guides/AGENT_AUTHOR_MINIMAL_PATH.md)

**Observability spine:** [`OBSERVABILITY.md`](OBSERVABILITY.md#observability-event-spine) — agents extend Plane B via `DiagnosticPayload`; execution truth lives on `RuntimeEvent` (Plane A). See §31 and [event ownership rules](OBSERVABILITY.md#event-ownership-rules).

**Retry / recovery:** agents emit recovery **intent** only — runtime owns retry policy, layers and stop reasons — [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md#attempt-ledger) · [`SYSTEM_INVARIANTS.md`](../guides/SYSTEM_INVARIANTS.md) §8.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (AGENT_CONTRACTS_AND_ASSEMBLY canon).

- **Implement / audit default:** §12–§21 (contract, registry, capability, ACP). Extended §22–§39 + checklist §45: [`satellites/AGENT_CONTRACTS_AND_ASSEMBLY_extended_depth.md`](satellites/AGENT_CONTRACTS_AND_ASSEMBLY_extended_depth.md). §40+: [`satellites/AGENT_CONTRACTS_AND_ASSEMBLY_production_gates.md`](satellites/AGENT_CONTRACTS_AND_ASSEMBLY_production_gates.md).
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../guides/audit_slices/AGENT_CONTRACTS_AND_ASSEMBLY.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---
## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`satellites/AGENT_CONTRACTS_AND_ASSEMBLY_extended_depth.md`](satellites/AGENT_CONTRACTS_AND_ASSEMBLY_extended_depth.md) | extended depth |
| [`satellites/AGENT_CONTRACTS_AND_ASSEMBLY_production_gates.md`](satellites/AGENT_CONTRACTS_AND_ASSEMBLY_production_gates.md) | production gates |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


## Table of contents

| § | Topic |
|---|--------|
| [§12](#12-agent-contract) | Agent contract |
| [§13](#13-agent-interface-run-facade-step-loop-and-uaep) | Agent interface: `run()`, step loop, UAEP |
| [§14](#14-agent-execution-result) | Agent execution result |
| [§15](#15-agent-registry) | Agent registry |
| [§16](#16-capability-model) | Capability model |
| [§17](#17-prompt-registry-architecture) | Prompt registry |
| [§18](#18-registry-architecture) | Registry architecture |
| [§19](#19-capability-graph-architecture) | Capability graph |
| [§20](#20-agent-lifecycle-governance) | Agent lifecycle governance |
| [§21](#21-agent-cognitive-architecture-acp) | **Agent Cognitive Architecture (ACP)** |
| [§22](#22-tier-and-terminology-canon) | Tier and terminology canon |
| [§23](#23-three-cognition-planes) | Three cognition planes |
| [§24](#24-agent-class-hierarchy) | Agent class hierarchy |
| [§25](#25-runtime-execution-context-state-model) | Runtime execution context / state |
| [§25.4](#254-invocation-time-token-usage-agent-vs-environment) | Invocation-time token usage (agent vs environment) |
| [§25.5](#255-token-budget-limits-enforcement-and-application-reactions) | Token budget limits, enforcement, application reactions |
| [§26](#26-cognitive-pattern-catalog) | Cognitive pattern catalog |
| [§27](#27-end-to-end-execution-flows) | End-to-end execution flows |
| [§28](#28-acp-code-map-maturity-and-gaps) | ACP code map, maturity, gaps |
| [§29](#29-author-facing-run-facade) | **Author-facing `run()` facade** |
| [§30](#30-per-agent-environment-and-resource-binding) | **Per-agent environment & resources** |
| [§30.9](#309-identity-tenantuser-and-memory-scope) | **Identity, tenant/user, memory scope** |
| [§31](#31-dual-observability-application-and-agent-planes) | **Dual observability planes** |
| [§32](#32-agent-step-loop-on_next_step) | **Agent step loop (`on_next_step`)** |
| [§32.0](#320-author-readability-and-typed-contracts-normative) | **Author readability & typed contracts (normative)** |
| [§33](#33-per-step-llm-routing) | **Per-step LLM routing** |
| [§34](#34-shared-state-and-cross-agent-visibility) | **Shared state & cross-agent visibility** |
| [§35](#35-use-case-catalog-agent--environment) | **Use-case catalog** |
| [§36](#36-final-architecture-agent--environment-cooperation) | **Final architecture synthesis** |
| [§37](#37-pre-implementation-operational-contracts) | **Pre-implementation operational contracts** |
| [§38](#38-execution-responsibility-stack-nexusloop-vs-step-kernel) | **Execution stack: NexusLoop vs step kernel** |
| [§39](#39-organizational-policy-envelope-virtual-workforce) | **Organizational policy envelope & virtual workforce** |
| [§40](#40-production-reliability-safety-persistence-and-release-gates) | **Production reliability, safety, persistence, release gates** |
| [§45](#45-checklist-for-new-agent-implementation) | New agent checklist |

---

# 12. Agent Contract

Every agent MUST implement a clear contract.

The contract should be easy for humans and LLMs to understand.

Minimum required fields:

```text
AgentContract:
    id
    name
    description
    version
    capabilities
    input_schema
    output_schema
    allowed_tools
    required_adapters
    execution_mode
    max_steps
    max_duration
    max_cost
    risk_level
    validation_rules
    failure_modes
```

---


---

# 13. Agent Interface: `run()` Facade, Step Loop, and UAEP

**ADR:** [ADR-AGENT-002](../adr/entries/2026-06-11/ADR-AGENT-002.md) · [ADR-AGENT-003](../adr/entries/2026-06-11/ADR-AGENT-003.md)

## 13.1 Primary author API — session `run()` (ADR-AGENT-002)

**Authors SHOULD use one session entry point:**

```text
result = await my_agent.run(request: AgentRunRequest) -> AgentRunResult
```

`Agent.run()` on the base class orchestrates harness services; subclasses implement **domain logic only** via **`on_next_step`** (§32), ACP hooks, or `@step` methods. See §29 for full contract and §31 for **`AgentRunTrace`** on the result.

**Internal engine:** `run()` → merge environment §30 → **step loop** (`execute_next_step` → `on_next_step`) → UAEP/policy/trace. Authors MUST NOT reimplement this stack.

## 13.2 Primary author API — step `on_next_step()` (ADR-AGENT-003)

**Authors SHOULD override one step hook for non-linear / cognitive agents:**

```text
async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome
```

**Readability rule (normative — §32.0):** every author MUST be able to answer, from code alone and without running the app: (1) what state was read, (2) what state changed, (3) whether the session continues, pauses, or terminates — via **typed** `AcpSessionState` + `StepOutcome` factories only. Untyped `dict` on the author surface is **not supported**.

| Responsibility | Owner |
|----------------|-------|
| Domain reasoning, tool intent, RAG query, LLM model choice (within profile) | **Author** (`on_next_step`) |
| Whether plan is needed / still valid / terminal / replan / HITL | **Author** (`on_next_step`) |
| Policy, trace events, tool gateway, memory view, budgets, state merge | **Harness kernel** (`HarnessKernel.execute_step`) §38 |
| One agent iteration glue (domain hook + kernel) | **Agent runtime** (`AgentRuntime.advance_step`) §38 |
| Multi-agent graph, task routing, application governance | **NexusLoop** (Tier-1) §38 |

One **`run()`** invocation executes **many** iterations of **`advance_step`** until `StepOutcome.is_terminal`. See §32 · §38.

## 13.3 Framework surface (Tier-2 vs author visibility)

| Surface | Who implements | Author visibility |
|---------|----------------|-------------------|
| `run(AgentRunRequest)` | Base `IntergraxAgent` / `CognitiveAgent` | **Public — agent decision loop** |
| `on_next_step(AgentStepContext)` | Subclass | **Public — domain step** |
| `AgentRuntime.advance_step` | Framework only | **Internal — one iteration; alias `execute_next_step`** |
| `HarnessKernel.execute_step` | Tier-0/1 kernel | **Internal — deterministic harness cycle** |
| `perceive` / `reason` / `act` / `evaluate` | Subclass / pattern base | **Public — may delegate to `on_next_step`** |
| `@step` methods | Subclass | **Public — linear pipelines** |
| `get_contract`, `can_handle` | Subclass | **Public — registration** |
| `configure_run` / `merge_environment` | Subclass optional hooks | **Public — per-agent tuning** §30 |
| `NexusLoop.handle_task` | Tier-1 | **Internal — application orchestration only** |
| `get_steps`, `run_step`, `decide_after_step` | Framework (UAEP bridge) | **Internal — legacy map to §32/§38** |
| `build_context` | Subclass or harness injection | **Advanced** §30 |

## 13.4 UAEP (internal protocol — implementation of step loop)

Every production agent MUST satisfy **`UAEPAgent`** via the framework base class. Nexus graph nodes invoke the same path whether the caller used `agent.run()` or `Task → NexusLoop`.

```text
UAEPAgent (framework-wired, not author boilerplate):
    get_steps(context) -> list[AgentStep]
    async run_step(step, ctx: RuntimeExecutionContext) -> StepOutput
    decide_after_step(step, output, ctx) -> AgentDecision   # optional; prefer on_next_step
```

**Mapping (normative — §38):**

```text
Agent.run()                    = agent decision loop (many iterations)
Agent.on_next_step()           = agent decides: plan? tool? model? terminal? HITL?
AgentRuntime.advance_step()    = one iteration glue (calls on_next_step + kernel)
HarnessKernel.execute_step()   = deterministic harness cycle (NO domain planning)
NexusLoop.handle_task()        = application / multi-agent orchestration ONLY

Legacy UAEP names (implementation today):
  run_step / UAEPExecutor  →  advance_step + execute_step bridge (ACP-STEP-3)
  planning/StepExecutor    →  ExecutionPlan runner — NOT agent cognitive step
```

**Rules:**

- Agents MUST NOT implement private OS lifecycles (HTTP servers, global schedulers, direct vendor SDK calls).
- Agents MUST NOT bypass `Agent.run()` / `on_next_step` with Tier-1 execution shortcuts (ACP-CLOSE-LEG-5).
- Control flow via **`StepOutcome`** / **`AgentDecision`** (§42.7).
- Optional: `UAEPAgentWithResume.resume_step` for checkpointed long steps.

## 13.5 Removed legacy paths (ACP-CLOSE-LEG-5)

| Path | Status |
|------|--------|
| `RuntimeEngine` / `RuntimePipeline` / `runtime_steps/` | **Removed** — [ADR-FLOW-005](../adr/entries/2026-06-12/ADR-FLOW-005.md) |
| `agents/*/steps/pipeline.py` / `uaep_pipeline_bridge.py` | **Removed** — use `on_next_step` + cognitive patterns |
| `execute()` pseudocode | Replaced by `run()` + `on_next_step` |
| Override `execute_next_step` / `advance_step` | **Forbidden** — bypasses policy/trace |
| `nexus.run()` as agent session API | **Forbidden** — use `Agent.run()`; NexusLoop for Task only §38 |

**Author loop control (normative):** every domain iteration is decided in **`on_next_step`** (or a cognitive pattern delegating to it). The harness runs **`HarnessKernel.execute_step`** — policy, trace, gateways, budgets — without choosing tools, models, or termination.

**Execution entry (two paths, same author hook):**

| Path | When | Outer loop | Cognition |
|------|------|------------|-----------|
| **ACP session** | `metadata["acp.session.v1"]` (Tier-3 harness task enricher sets by default) | `run_acp_session` → `AgentRuntime.advance_step` (multi-iteration) | `on_next_step` each iteration |
| **UAEP bridge (Nexus default)** | `CognitiveAgent` / fleet agents | `UAEPExecutor` over `get_steps()` (typically **one** cognitive step) | `run_step` → `acp_uaep_shim` → **`on_next_step`**; ReAct/plan-execute loop **inside** `on_next_step` |

`UaepPipelineStubAgent` in `testing_support/` is **test-only**. Product agents MUST NOT author custom `get_steps`/`run_step` beyond `CognitiveAgent` defaults — implement domain logic in `on_next_step` / pattern hooks.

```text
# ACP session (opt-in via acp.session.v1)
Agent.run(AgentRunRequest)
  └─ run_acp_session: for max_iterations
        ├─ agent.on_next_step(step_ctx) → StepOutcome   ← AUTHOR
        └─ HarnessKernel.execute_step(outcome)          ← HARNESS

# Nexus production default (fleet CognitiveAgent)
AgentEngine → UAEPExecutor → run_step → on_next_step → HarnessKernel (via uaep_step_bridge)
```

No Tier-1 code path may inject fixed step order (retired `RuntimePipeline` / `runtime_steps/`). Tool loops (ReAct) run **inside** `on_next_step` via `run_bounded_tool_loop` + `ctx.invoke_tool`, not via Nexus graph scheduling (ADR-TOOL-002).

## 13.6 Authoring facades

| Facade | Module | Use when |
|--------|--------|----------|
| `IntergraxAgent` | `intergrax/agents/authoring/base.py` | `@step` linear agents; inherits `run()` + default `on_next_step` |
| `CognitiveAgent` + patterns §26 | `intergrax/agents/authoring/patterns/` | ReAct, decomposition, reflection — patterns implement `on_next_step` |
| `HarnessReferenceAgent` | `harness_reference_agent.py` | Low-level UAEP ABC (framework/tests) |

**Guide:** [`guides/AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) Appendix AC · **Plan:** Phase **ACP** + **ACP-DX** + **ACP-STEP** rows.

---


---

# 14. Agent Execution Result

Every agent should return a structured result.

Recommended structure:

```text
AgentExecutionResult:
    agent_id
    run_id
    status
    summary
    artifacts
    structured_data
    evidence
    confidence
    warnings
    errors
    used_tools
    cost
    duration
    next_recommendations
```

The result must be inspectable by Nexus and by humans.

---


---

# 15. Agent Registry

Nexus discovers agents through the Agent Registry.

The registry stores:

- agent id
- name
- description
- version
- capabilities
- required adapters
- allowed tools
- execution modes
- cost profile
- risk profile
- status

Nexus MUST use the registry for agent selection.

Agents MUST NOT be hardcoded into Nexus logic unless explicitly needed for a minimal prototype.

Even in prototypes, hardcoded agents should be treated as temporary.

---


---

# 16. Capability Model

A capability describes what an agent can do.

Examples:

```text
capability: vendor.discovery
capability: vendor.scoring
capability: legal.contract_review
capability: research.web_search
capability: problem_radar.source_monitoring
capability: problem_radar.clustering
capability: onboarding.daily_guidance
```

Nexus should route tasks to capabilities, not only to specific class names.

This allows agents to be replaced later.

**Routing invariant (normative — §37.6):** production Nexus agent selection MUST resolve **`required_capability`** (or task capability token) → registry entries by **`capabilities[]`** match — **not** by Python class name, module path, or hardcoded agent id. Class name MAY appear only in Tier-3 manifest **`AgentBinding`** for wiring a chosen capability to a concrete implementation. CI: `check_capability_routing.py` (ACP-CON-6).

---


---

# 17. Prompt Registry Architecture

Prompt artifacts are **governed platform assets**, not ad-hoc strings in agents.

## 17.1 Requirements

- ownership and versioning on every prompt id (`PromptMeta`),
- composable layers: system / task / policy / context,
- deterministic policy injection overlays,
- regression suites on golden prompt catalogs,
- Tier-3 `PromptProfile` selects YAML catalog path per host.

## 17.2 Code map

| Module | Role |
|--------|------|
| `intergrax/prompts/registry/` | YamlPromptRegistry, governance validation |
| `intergrax/runtime/architecture/prompt_registry_governance.py` | Ownership / risk tier gates |
| `intergrax/runtime/architecture/prompt_composition.py` | Layer composition |
| `intergrax/runtime/architecture/prompt_policy_overlay.py` | Policy overlays |
| `intergrax/runtime/architecture/prompt_regression_suite.py` | Golden regression |
| `intergrax/applications/_shared/prompt_wiring.py` | Environment → Nexus prompt registry |

**Authoring:** [`guides/AGENT_CREATION_GUIDE.md` Appendix M](../guides/AGENT_CREATION_GUIDE.md) · **Plan:** [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) Phase PE.

**Context assembly boundary:** Prompt Registry supplies governed **fragments** only — not a full LLM window. Production context **MUST** flow through `ContextCompiler` / `ContextEngine` or an explicitly approved equivalent. See [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) §12 Context Path Unification.

---

# 18. Registry Architecture

Registries are versioned, snapshot-capable catalogs — not mutable globals.

## 18.1 Registry types

| Registry | Tier | Consumed by |
|----------|------|-------------|
| Agent registry | 1 | Nexus agent selection |
| Tool registry | 0 | `ToolRuntime` |
| Skill registry | 0 | Skill resolver |
| Integration registry | 0 | Provider hosts |
| Prompt registry | 0/1 | Nexus steps, eval |
| Evaluation registry | 1 | EvalRunner, release gates |

## 18.2 Assembly pattern

Tier-3 `wire_application_environment()` materializes registries from `ApplicationEnvironmentProfile` tool/skill/integration/prompt profiles → `RuntimeConfig` via `runtime_config_bridge.py` and domain `*_assembly_resolver.py` modules.

Snapshots and conformance CI validate registry shape before release (`scripts/check_agents_lifecycle_metadata.py`, harness registry guards). **Durable cross-host snapshots:** `applications/_shared/registry_snapshot_store.py` (AUDIT-IDEAL-19.1) + `check_registry_snapshot_diff.py`.

**Plan:** [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) Phase REG.

---

# 19. Capability Graph Architecture

> **Tier-3 consumption:** environment-scoped graph view, blast-radius deploy gates, and ops health dimensions — [`TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md) §50.1 · §51. **Do not duplicate** graph taxonomy in Tier-3; extend via `EnvironmentCapabilityGraphView` only.

Registries and capability layers MUST be represented as a typed dependency graph:

```text
Integration -> Tool -> Skill -> Policy -> Agent -> Application -> Product
```

## 19.1 Minimum requirements

- typed node and edge taxonomy,
- dependency lineage and provenance,
- blast-radius impact analysis for version/policy/runtime changes,
- compatibility validation on graph edges before release.

## 19.2 Code map

| Module | Role |
|--------|------|
| `runtime/architecture/capability_graph.py` | Core graph model |
| `capability_graph_lineage.py` | Lineage / provenance |
| `capability_graph_compatibility.py` | Edge compatibility |
| `capability_graph_applications.py` | Application slice |
| `scripts/phase_v_capability_graph_guard.py` | CI guard + blast-radius impact (AUDIT-IDEAL-20.1) |
| `scripts/check_capability_graph_strict_deploy.py` | STRICT deploy gate (APP-OPS-1) |

Nexus routes to **capabilities** (§16), not hardcoded class names. Graph edges MUST reflect manifest roster per application — not global cross-product shortcuts.

**Plan:** [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) Phase CG.

---

# 20. Agent Lifecycle Governance

Beyond contract shape (§12) and registry metadata (§15):

| Stage | Requirement |
|-------|-------------|
| Certification | quality + policy + security gates before production |
| Promotion | dev → staging → production with evidence |
| Deprecation | migration windows, runtime filters for retired agents |
| Retirement | rollback/archive semantics |
| Ownership | explicit owner + escalation path |

**Code:** `runtime/architecture/agent_lifecycle_governance.py`, `agent_certification.py`, `agent_promotion.py`, `production_ownership.py`. **On-call gate (AUDIT-IDEAL-31.1):** `check_agents_lifecycle_metadata.py`, `check_on_call_ownership_model.py`, `on_call_contact` on `AgentContract`.

Runtime MUST reject or reroute retired/deprecated agents in production mode (V-REM-ALG.*). **Plan:** Phase AS + V-REM in [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md).

---

# 21. Agent Cognitive Architecture (ACP)

**Status:** Canonical architecture — **platform delivered** (Phase ACP + ACP-CLOSE + ACP-FINISH **Done**); AUDIT-IDEAL §12–§20 **Done** (2026-06-13)  
**ADR:** [ADR-AGENT-001](../adr/entries/2026-06-11/ADR-AGENT-001.md)  
**Plan:** [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) — ACP · ACP-CLOSE · ACP-FINISH · AUDIT-IDEAL **Done**  
**Cross-domain:** [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) (planes 1–3) · [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) (narrative) · [`TOOLS.md`](TOOLS.md) TOOL-ENG-6 (tool loop) · [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md#verification-safety-boundaries) (reflection / verification safety) · [`CODE_CRAFT.md`](CODE_CRAFT.md#codecraft-safety-boundary) (ephemeral codegen — no agent-local craft loops)

## 21.1 Purpose

Define the **Agent Cognitive Architecture (ACP)** — how Tier-2 agents are authored, how they interact with Tier-1 Nexus, and how **cognitive patterns** (reflex, ReAct, plan-execute, decomposition, reflection) are implemented **without** collapsing the Harness into agent classes.

ACP answers:

> **How does a developer build a production-grade agent quickly, with the right reasoning pattern, while staying inside UAEP and platform governance?**

ACP does **not** replace Nexus, redefine tiers, or introduce a second execution engine.

## 21.2 Design invariants

| ID | Invariant |
|----|-----------|
| **ACP-INV-01** | Nexus remains Agent OS — global orchestration, policy, HITL, multi-agent graph |
| **ACP-INV-02** | All agent runs use UAEP step loop (legacy UAEP authoring retired — ACP-CLOSE-LEG-5) |
| **ACP-INV-03** | Cognitive patterns are **Tier-2 libraries** — no imports from `applications/` |
| **ACP-INV-04** | Side effects only through `RuntimeExecutionContext.tool_gateway` → `ToolRuntime` |
| **ACP-INV-05** | Control flow via `AgentDecision` — never `sleep()` for HITL, never direct Slack/webhooks |
| **ACP-INV-06** | Configuration: **contract + pattern in agent**; **governance profile in Tier-3** |
| **ACP-INV-07** | Three cognition planes MUST NOT be collapsed into one opaque agent loop (§23) |
| **ACP-INV-08** | Every step emits `STEP_*` and `DECISION_EMITTED` events (§42.1) |
| **ACP-INV-09** | **`run()` is the author entry**; UAEP is internal; Nexus is application entry for `Task` (ADR-AGENT-002) |
| **ACP-INV-10** | Per-agent resources (memory, tools, RAG) bound via contract + environment — not hardcoded SDK clients |
| **ACP-INV-11** | **`NexusLoop` orchestrates tasks/graphs**; **`HarnessKernel.execute_step`** runs one deterministic agent runtime cycle — kernel MUST NOT plan agent reasoning (§38) |
| **ACP-INV-12** | **Organizational policy** is Tier-3 environment data — agents consume `OrganizationalPolicyContext`; harness enforces; agents MUST NOT embed org-specific regulations in source (§39) |
| **ACP-INV-13** | **Authenticated runs default to user-scoped memory** (`tenant_id` + `user_id`) — org-wide memory only when contract/binding/env declare `memory_scope=org` §30.9 |

## 21.3 Rejected architecture (audit outcome)

The following proposals were audited and **rejected** — documented here to prevent regression:

```text
REJECTED: Move NexusLoop responsibilities into IntergraxAgent base class
REJECTED: Agent owns PolicyEngine, GraphExecutor, or AgentRegistry
REJECTED: All RuntimeConfig / ApplicationEnvironmentProfile fields hardcoded in agent source
REJECTED: Private while-True agent loops without UAEP step boundaries
REJECTED: nexus.run() / NexusLoop as agent plan brain — planning lives in Agent.on_next_step (§38)
REJECTED: Nexus "run" naming for agent session step executor — blurs Agent OS vs step kernel
```

**Rationale:** Harness-first strategy ([`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §0.2) — the runtime is the durable product; agents are replaceable workers.

---
