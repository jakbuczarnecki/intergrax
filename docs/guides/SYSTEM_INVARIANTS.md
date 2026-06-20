# Intergrax — System Invariants

**Status:** Canonical index (2026-06-20)  
**Audience:** Architects, reviewers, implementation agents, external auditors  
**Audit ID:** P2-ARCH-01  
**Related:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md) · [`MATURITY_TAXONOMY.md`](MATURITY_TAXONOMY.md) · [`AGENT_AUTHOR_MINIMAL_PATH.md`](AGENT_AUTHOR_MINIMAL_PATH.md) · [`TIER3_PRODUCT_HYPOTHESIS_CONTRACT.md`](TIER3_PRODUCT_HYPOTHESIS_CONTRACT.md) · [`INTERGRAX_DEVELOPMENT_STRATEGY.md`](INTERGRAX_DEVELOPMENT_STRATEGY.md) · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md)

---

## 1. Purpose

Intergrax spreads **non-negotiable architectural rules** across 22 domain pairs, ADRs, and CI gates. This document is the **single cross-layer authority** — normative MUST / MUST NOT / SHOULD rules plus a terse `SYS-INV-*` index with CI pointers.

**This file is not a second canon.** When semantics change, update the domain architecture first, then adjust the cross-layer rule and §5 row here. Do not copy long tables from ACP / APP / ORCH into this guide.

---

## 2. How to use

| Situation | Action |
|-----------|--------|
| **Onboarding** | Read [Cross-Layer System Invariants](#cross-layer-system-invariants) + §4 (execution stack) before opening a domain pair |
| **Code review** | Check diff against cross-layer rules and §5 index; escalate violations before merge |
| **Domain implementation** | Read your domain pair; use cross-layer rules as guardrails |
| **External audit** | Cross-layer rules + §5 + §7 (CI mapping) + §8 (rejected patterns) + [MATURITY_TAXONOMY.md](MATURITY_TAXONOMY.md) four-axis statements |
| **LLM session start** | Hub → this file (cross-layer section) → one domain pair (per-iteration rule) |

**Decision hierarchy** (when rules appear to conflict): [`INTERGRAX_DEVELOPMENT_STRATEGY.md`](INTERGRAX_DEVELOPMENT_STRATEGY.md) §Decision hierarchy — strategy → ideal architecture → domain architecture → domain plan.

---

## 3. Meta-invariant

| ID | Rule | Canon |
|----|------|-------|
| **SYS-INV-00** | **The Harness is the product; agents are replaceable.** Platform runtime outlives any single agent implementation. | Hub · [`PLATFORM_FOUNDATION.md`](../architecture/PLATFORM_FOUNDATION.md) §2 · [ADR-AGENT-001](../adr/entries/2026-06-11/ADR-AGENT-001.md) |

---

# Cross-Layer System Invariants

Normative rules that **MUST** hold across Tier-0..3, Nexus, agents, tools, context, LLM, observability, policy, and documentation. Domain-specific detail lives in architecture pairs (§6); CI mapping in §5 and §7.

## 1. Tier boundary invariants

- Tier-0 `intergrax/` provides universal, domain-agnostic primitives.
- Tier-0 **MUST NOT** import from `agents/` or `applications/`.
- Tier-1 `intergrax/runtime/` owns Nexus, AgentEngine, UAEP, policy, runtime events, orchestration execution.
- Tier-2 `agents/` owns domain-specific reasoning and domain step decisions.
- Tier-3 `applications/` owns deployable hosts, manifests, profiles, rosters, intake surfaces and product composition.
- Business/product logic **MUST NOT** be implemented in Tier-0 or Tier-1 unless it is truly domain-agnostic.
- Applications compose the harness; they **MUST NOT** become agents.
- Agents run inside the harness; they **MUST NOT** become private runtimes.

**Canon:** [`PLATFORM_FOUNDATION.md`](../architecture/PLATFORM_FOUNDATION.md) §5 · [`TIER3_APPLICATION_ENVIRONMENT.md`](../architecture/TIER3_APPLICATION_ENVIRONMENT.md) §28

## 2. Nexus invariants

- Nexus orchestrates tasks, graphs, routing, retries, HITL and finalization.
- Nexus **MUST NOT** become a domain reasoning agent.
- Nexus **MUST NOT** encode business conclusions, domain rubrics or product-specific decisions.
- Nexus **MAY** select agents and execution topology through typed plans.
- Nexus **MUST** delegate domain reasoning to Tier-2 agents.
- Nexus **MUST** delegate verification to CVL / validation mechanisms.
- Nexus **MUST** delegate side effects to ToolRuntime.
- All application surfaces **MUST** converge on `UnifiedTaskRunner.run_task()` and `NexusLoop.handle_task()`.

**Canon:** [`ORCHESTRATION.md`](../architecture/ORCHESTRATION.md) §56 · [`NEXUS_EXECUTION_FLOW.md`](../architecture/NEXUS_EXECUTION_FLOW.md)

## 3. Agent invariants

- Agent authors implement domain behavior through `Agent.run()` / `on_next_step()` / approved step APIs.
- Agents decide the next domain move inside their bounded local loop.
- Agents **MUST NOT** implement private orchestration runtimes.
- Agents **MUST NOT** create HTTP servers, schedulers, queues, global loops or private OS lifecycles.
- Agents **MUST NOT** call vendor SDKs directly.
- Agents **MUST NOT** bypass ToolRuntime, PolicyEngine, ContextCompiler, MemoryView or RuntimeEventBus.
- Agents **MUST NOT** directly write to external systems except through approved tools.
- Agents **MUST** return typed outputs suitable for validation and evaluation, not only raw text.

**Canon:** [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §21 · [`UNIFIED_EXECUTION_RUNTIME.md`](../architecture/UNIFIED_EXECUTION_RUNTIME.md) §42

## 4. Tool and integration invariants

- ToolRuntime is the only side-effect gateway for agent-invokable actions.
- Agent-invokable side effects **MUST** pass through ToolRuntime.
- Agents and graph nodes **MUST NOT** call tool handlers or integrations directly.
- Tools are agent-facing semantic operations.
- Integrations are backend/vendor-facing adapters and **MUST NOT** be agent-facing.
- External intake adapters **MUST** hand tasks to Tier-3 intake / `UnifiedTaskRunner.run_task()`, not directly to agents.
- Skills are declarative composition packs, not runtime loops.
- Skills **MUST NOT** contain execution control flow.
- Tool access **MUST** be resolved from AgentContract, SkillResolver, ToolProfile and RuntimePolicyBundle.
- Production applications **MUST NOT** enable all tools by default.

**Canon:** [`TOOLS.md`](../architecture/TOOLS.md) · [`INTEGRATIONS.md`](../architecture/INTEGRATIONS.md) · [`SKILLS.md`](../architecture/SKILLS.md)

## 5. Context, memory and RAG invariants

- Memory is persisted state.
- Context is what the model sees in a specific step.
- Knowledge/RAG is document or corpus retrieval.
- Trace is immutable audit evidence.
- These concepts **MUST NOT** be conflated.
- ContextCompiler / ContextEngine is the **canonical production path** for LLM-facing context.
- Any alternative production context path **MUST** be explicitly approved and documented ([`CONTEXT_ENGINEERING.md`](../architecture/CONTEXT_ENGINEERING.md) §12 Context Path Unification).
- Lab/test shortcuts **MUST NOT** be promoted to production by accident.
- Agents **MUST NOT** hand-assemble production prompts from unbounded history.
- Agents **MUST NOT** query vector stores directly.
- RAG retrieval **MUST** go through the approved RAG service / catalog tools.
- Knowledge indexes, long-term memory indexes and episodic/session indexes **MUST** remain logically separated.
- Vector indexes are retrieval indexes, not primary stores.

**Canon:** [`CONTEXT_ENGINEERING.md`](../architecture/CONTEXT_ENGINEERING.md) · [`MEMORY.md`](../architecture/MEMORY.md) · [`RAG.md`](../architecture/RAG.md)

## 6. LLM invariants

- LLM calls **MUST** go through LLMAdapter / approved routing abstractions.
- Agents **MUST NOT** import provider SDKs directly.
- LLM responses **MUST** use typed envelopes, not bare strings.
- Structured outputs **MUST** be validated before being consumed by executors.
- Planner, critic and producer profiles **SHOULD** be separable when risk or policy requires judge separation.

**Canon:** [`LLM_ADAPTERS.md`](../architecture/LLM_ADAPTERS.md) · [`REASONING_AND_COGNITION.md`](../architecture/REASONING_AND_COGNITION.md)

## 7. Observability invariants

- `RuntimeEvent` is the canonical runtime event/audit envelope for meaningful execution transitions.
- Logs, metrics, diagnostic traces and external sinks **MUST NOT** become competing sources of execution truth.
- New runtime components **SHOULD** emit through RuntimeEventBus / approved observability spine.
- New components **MUST** preserve correlation identifiers where available.
- Agents **MUST NOT** create private trace stores or private logging pipelines.
- Every meaningful execution transition **MUST** be traceable.
- Event payloads **MUST NOT** contain secrets; redaction **MUST** happen before persistence or external export where required.
- Domain-specific events **SHOULD** use namespaced `event_kind` / payload schemas instead of expanding platform lifecycle enums unnecessarily.

**Canon:** [`OBSERVABILITY.md`](../architecture/OBSERVABILITY.md) — [Observability Event Spine](../architecture/OBSERVABILITY.md#observability-event-spine) · [`UNIFIED_EXECUTION_RUNTIME.md`](../architecture/UNIFIED_EXECUTION_RUNTIME.md) §42

## 8. Reliability, policy and HITL invariants

- Agents emit intent; runtime and policy execute retries, escalation, HITL and failure handling.
- Agents **MUST NOT** implement unbounded retry loops against tools, LLMs or integrations.
- Human approval is managed by Nexus / HITL mechanisms, not ad-hoc agent messages.
- High-risk side effects require policy approval and trace evidence.
- LLM-as-judge alone **MUST NOT** authorize irreversible high-risk side effects.
- Idempotency **MUST** be used for side-effectful tools where applicable.

**Canon:** [`RELIABILITY_FAILURE_AND_HITL.md`](../architecture/RELIABILITY_FAILURE_AND_HITL.md) · [`CRITIC_VERIFICATION.md`](../architecture/CRITIC_VERIFICATION.md)

## 9. Adaptive and scaling invariants

- Adaptive Harness Intelligence may observe, propose and evaluate changes.
- Production auto-apply of adaptive changes requires explicit product/governance decision.
- AHI **MUST NOT** silently mutate prompts, routing, policies or profiles in production.
- Elastic Capacity Plane scales infrastructure capacity only.
- ECP **MUST NOT** decide agent topology or domain execution strategy.

**Canon:** [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](../architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) · [`ELASTIC_CAPACITY_AND_SCALING.md`](../architecture/ELASTIC_CAPACITY_AND_SCALING.md)

## 10. Documentation authority invariants

- `docs/intergrax_runtime_architecture.md` is the hub, not the detailed owner of every subsystem.
- Each `docs/architecture/*.md` file is the source of truth for its own subsystem.
- Each architecture document **SHOULD** have a corresponding implementation plan in `docs/plan/*.md`.
- When documents conflict, the more specific domain architecture owns subsystem-specific rules.
- Any cross-layer rule **MUST** be reflected or referenced from this file (`SYSTEM_INVARIANTS.md`).

**Canon:** [`INTERGRAX_DEVELOPMENT_STRATEGY.md`](INTERGRAX_DEVELOPMENT_STRATEGY.md) · hub [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)

---

## 4. Execution stack (L1–L4)

Responsibility split — **who decides vs who executes vs who orchestrates**:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ L4  Application + NexusLoop.handle_task()                               │
│     Environment: profiles, AgentBinding, RequestIdentity, org envelope  │
│     Orchestration: Task graph, capability routing, HITL, Plane A log      │
│     DOES NOT: plan inside one agent's cognitive loop                      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ graph node → one Agent.run() per role
┌───────────────────────────────▼─────────────────────────────────────────┐
│ L3  Agent.run() — session decision loop (many steps, one user-facing run) │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ each iteration
┌───────────────────────────────▼─────────────────────────────────────────┐
│ L2  Agent.on_next_step() — author domain hook                             │
│     READ typed state · UPDATE state_delta · DECIDE StepOutcome            │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ StepOutcome
┌───────────────────────────────▼─────────────────────────────────────────┐
│ L1  HarnessKernel.execute_step() — deterministic harness primitive        │
│     policy · gateways · trace · budgets · state merge · checkpoint hook   │
│     DOES NOT: domain replan · choose next graph agent                     │
└─────────────────────────────────────────────────────────────────────────┘
```

**Three cognition planes** (do not collapse):

| Plane | Owner | Question | Canon |
|-------|-------|----------|-------|
| **1 — Nexus task** | Tier-1 | Which agents, in what order? | [`ORCHESTRATION.md`](../architecture/ORCHESTRATION.md) §56 · [`REASONING_AND_COGNITION.md`](../architecture/REASONING_AND_COGNITION.md) §5 |
| **2 — UAEP step** | Tier-1 + Tier-2 | What does this agent do in one graph node? | [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §32 · [`UNIFIED_EXECUTION_RUNTIME.md`](../architecture/UNIFIED_EXECUTION_RUNTIME.md) §42 |
| **3 — Tool loop** | Tier-1 | Which tools does the LLM invoke this iteration? | [`TOOLS.md`](../architecture/TOOLS.md) · [ADR-TOOL-002](../adr/entries/2026-06-11/ADR-TOOL-002.md) |

---

## 5. System invariants (`SYS-INV-*` index)

Cross-layer normative rules above; this table maps **IDs · CI gates · domain canon** for reviews and audits.

Format: **ID · Layer · Rule · Canon · CI (if enforced)**

### 5.1 Tier boundaries and imports

| ID | Layer | Rule | Canon | CI |
|----|-------|------|-------|-----|
| **SYS-INV-01** | Tier-0 | `intergrax/` **never** imports `agents/` or `applications/`. | [`PLATFORM_FOUNDATION.md`](../architecture/PLATFORM_FOUNDATION.md) §5.1 · §5.3 | `check_intergrax_no_applications_imports.py` |
| **SYS-INV-02** | Tier-2 | `agents/` **never** imports `applications/`. | [`PLATFORM_FOUNDATION.md`](../architecture/PLATFORM_FOUNDATION.md) §5.3 | `check_agents_no_tier3_imports.py` |
| **SYS-INV-03** | Tier-0 | Platform catalogs **never** orchestrate, route agents, or host HTTP product surfaces. | [`PLATFORM_FOUNDATION.md`](../architecture/PLATFORM_FOUNDATION.md) §5.1.2 | review |
| **SYS-INV-04** | Tier-1 | Nexus **never** implements agent business workflows or domain prompts. | [`PLATFORM_FOUNDATION.md`](../architecture/PLATFORM_FOUNDATION.md) §5.2 | review |
| **SYS-INV-05** | Tier-2 | Agents **never** own global orchestration, HTTP host wiring, or `AgentRegistry` lifecycle. | ACP §21.2 **ACP-INV-01** · §21.3 Rejected | `check_acp_ci_conformance_matrix.py` |
| **SYS-INV-06** | Tier-3 | Applications **compose** profiles and surfaces — **never** implement `on_next_step` cognition. | APP §28.1 **APP-INV-03** · §28.2 Rejected | `check_agent_registry_bypass.py` |

### 5.2 Execution path (one pipeline)

| ID | Layer | Rule | Canon | CI |
|----|-------|------|-------|-----|
| **SYS-INV-07** | All | Every unit of work → **`Task` → `UnifiedTaskRunner` → `NexusLoop.handle_task()`** — no surface-specific Nexus forks. | ORCH §56.2 · APP **APP-INV-02** | `check_product_intake_parity.py` |
| **SYS-INV-08** | Entry | Application entry: **`run_task()`** / HTTP `/run`. Agent entry: **`Agent.run()`**. Never swap or use `NexusLoop` as agent session API. | ACP **ACP-INV-09** · APP **APP-INV-09** | `check_agent_acp_close_ci.py` |
| **SYS-INV-09** | Tier-2 | Agents **never call agents** — collaboration only via `ExecutionGraph` + `SharedTaskContext`. | ORCH §56.2.2 · UAEP §42 | review |
| **SYS-INV-10** | Tier-0/1 | **One canonical path** per universal concern (tools, RAG, LLM, memory, trace) — no parallel Tier-0 mechanisms. | [`PLATFORM_FOUNDATION.md`](../architecture/PLATFORM_FOUNDATION.md) §5.4 | `check_architecture_debt_burn_down.py` |

### 5.3 Responsibility split (L1–L4)

| ID | Layer | Rule | Canon | CI |
|----|-------|------|-------|-----|
| **SYS-INV-11** | Tier-1 | **Nexus orchestrates** tasks and graphs — **never cognates** (never replaces `on_next_step`). | Hub · ACP **ACP-INV-11** · [ADR-AGENT-001](../adr/entries/2026-06-11/ADR-AGENT-001.md) | review |
| **SYS-INV-12** | Tier-1 | **`HarnessKernel.execute_step`** runs one harness cycle — **never replans** domain reasoning or selects the next graph agent. | ACP §38 · **ACP-INV-11** | unit tests (`HarnessKernel`) |
| **SYS-INV-13** | Tier-1 | **`AgentRuntime.advance_step`** is glue only — policy, trace, budget, and state merge live in the kernel. | ACP §32.1 · §38.3 | review |
| **SYS-INV-14** | Tier-3 | Application **hooks are boundaries**, not step loops — no `on_next_orchestration_step()`. | APP §28.2 Rejected · **APP-INV-07** | review |
| **SYS-INV-15** | Tier-2 | Authors **never override** `run()`, `advance_step`, or `execute_step` in production agents. | ACP §29.4 · §32.0 | `check_agent_acp_close_ci.py` |

### 5.4 Platform gateways (single door)

| ID | Layer | Rule | Canon | CI |
|----|-------|------|-------|-----|
| **SYS-INV-16** | Tier-2 | Tools **never bypass** `ToolRuntime` / `RuntimeToolInvoker` / `tool_gateway`. | ACP **ACP-INV-04** · [`TOOLS.md`](../architecture/TOOLS.md) §42.12 | `check_agent_registry_bypass.py` · `check_tool_invocation_patterns.py` |
| **SYS-INV-17** | Tier-2 | Integrations **never** via vendor SDK in agent code — catalog + Tier-3 wiring only. | [`PLATFORM_FOUNDATION.md`](../architecture/PLATFORM_FOUNDATION.md) §5.2 · [`INTEGRATIONS.md`](../architecture/INTEGRATIONS.md) | `check_agents_vendor_imports.py` · `check_integration_vendor_imports.py` |
| **SYS-INV-18** | Tier-2 | RAG **never** via direct `vectorstore.query` in agents — `ToolRuntime` / retrieval tools. | [`RAG.md`](../architecture/RAG.md) §Design | `check_agent_registry_bypass.py` |
| **SYS-INV-19** | Tier-2 | Memory **never** via direct adapter writes — `memory_view` + resolved namespace only. | ACP §30.9 · [`MEMORY.md`](../architecture/MEMORY.md) | review |
| **SYS-INV-20** | Tier-2 | LLM **never** via raw provider SDK — injected `LLMAdapter` or `StepLLMRouter` port. | [`LLM_ADAPTERS.md`](../architecture/LLM_ADAPTERS.md) §Design | `check_agents_llm_adapter_response.py` |
| **SYS-INV-21** | Tier-2 | Skills **never** replace `ToolRuntime` or appear as fake `ToolContract` entries. | [`SKILLS.md`](../architecture/SKILLS.md) · PLATFORM §7.1.8 | review |

### 5.5 Cognition planes and graphs

| ID | Layer | Rule | Canon | CI |
|----|-------|------|-------|-----|
| **SYS-INV-22** | Tier-1 | **Three cognition planes stay separate** — Nexus must not micromanage tool loops; agents must not rewrite global topology without delegation contracts. | RCL §5 · ACP **ACP-INV-07** | `check_reasoning_gates.py` |
| **SYS-INV-23** | Tier-1 | **`GraphExecutor` never** implements tool-level ReAct loops — tool iterations belong inside agent steps. | [`TOOLS.md`](../architecture/TOOLS.md) · [ADR-TOOL-002](../adr/entries/2026-06-11/ADR-TOOL-002.md) | `check_agent_acp_ap02_tool_loop_boundary.py` |
| **SYS-INV-24** | Tier-1 | Capability routing resolves **`required_capability` → `capabilities[]`** — not Python class name or module path. | ACP §15 · §37.6 | `check_capability_routing.py` |

### 5.6 Governance, typing, and side effects

| ID | Layer | Rule | Canon | CI |
|----|-------|------|-------|-----|
| **SYS-INV-25** | Tier-3 | Organizational policy is **Tier-3 data** — agents consume merged context; harness enforces at hooks. | ACP **ACP-INV-12** · APP **APP-INV-08** · TIER3 §39 | review |
| **SYS-INV-26** | Tier-2 | Control flow via **`StepOutcome` factories** — never `sleep()` for HITL, never direct Slack/webhooks. | ACP **ACP-INV-05** · UAEP §42.10 | `check_agent_typed_state.py` |
| **SYS-INV-27** | Tier-2 | Author-facing APIs are **typed Pydantic only** — no untyped `dict` control flags in domain code. | ACP §32.0 | `check_agent_typed_state.py` |
| **SYS-INV-28** | Tier-2 | Side effects and diagnostics go **through the runtime bus** — agents do not publish to external queues directly. | UAEP §42.1 · §42.24 | review |
| **SYS-INV-29** | Tier-3 | **`ApplicationEnvironmentProfile`** is the composition root — no ad-hoc `getattr` wiring in hosts. | APP **APP-INV-06** · **APP-INV-10** | `check_harness_no_getattr.py` |
| **SYS-INV-30** | Tier-1 | **`AdaptationEngine.propose()` never** runs inside `NexusLoop` hot path. | [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](../architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) | review |

---

## 6. Related invariant registers (domain detail)

Use these for **full tables**, rationale, and rejected-architecture history. §5 above is the cross-domain subset.

| Register | Location | Count |
|----------|----------|-------|
| **ACP design invariants** | [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §21.2 | ACP-INV-01 … 13 |
| **APP design invariants** | [`TIER3_APPLICATION_ENVIRONMENT.md`](../architecture/TIER3_APPLICATION_ENVIRONMENT.md) §28.1 | APP-INV-01 … 10 |
| **Orchestration platform invariants** | [`ORCHESTRATION.md`](../architecture/ORCHESTRATION.md) §56.2 | 7 numbered rules |
| **Hub strategic summaries** | [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md) Agent + Application sections | ADR-AGENT-001..003 · APP-CON §28.1 |
| **Tier import matrix** | [`PLATFORM_FOUNDATION.md`](../architecture/PLATFORM_FOUNDATION.md) §5.3 | FAUDIT-TIER |
| **Governance naming** | [`GOVERNANCE_CONSISTENCY_AUDIT.md`](GOVERNANCE_CONSISTENCY_AUDIT.md) | overlap risks |

**ID mapping:** `SYS-INV-*` rows **summarize** domain IDs (e.g. SYS-INV-11 ↔ ACP-INV-11). Prefer domain IDs in plan rows and ADRs; use `SYS-INV-*` in reviews and audit checklists.

---

## 7. CI enforcement map (selected)

Not every invariant has a dedicated gate — some are architectural review obligations. Primary scripts:

| Concern | Script |
|---------|--------|
| Tier-0 → applications import | `scripts/check_intergrax_no_applications_imports.py` |
| Tier-2 → applications import | `scripts/check_agents_no_tier3_imports.py` |
| Agent direct integration/tool bypass | `scripts/check_agent_registry_bypass.py` |
| Vendor SDK in Tier-2 | `scripts/check_agents_vendor_imports.py` |
| Capability routing | `scripts/check_capability_routing.py` |
| Typed step outcomes / state | `scripts/check_agent_typed_state.py` |
| ACP conformance matrix | `scripts/check_acp_ci_conformance_matrix.py` |
| Tool loop vs graph boundary | `scripts/check_agent_acp_ap02_tool_loop_boundary.py` |
| Profile wiring without getattr | `scripts/check_harness_no_getattr.py` |
| LLM adapter response typing | `scripts/check_agents_llm_adapter_response.py` |
| Reasoning plane gates | `scripts/check_reasoning_gates.py` |

**Regression bundle** (after harness changes): see [`AGENT_INSTRUCTIONS.md`](AGENT_INSTRUCTIONS.md) Verification section.

---

## 8. Rejected patterns (quick anti-regression)

Do not reintroduce — full rationale in domain §21.3 / §28.2:

```text
REJECTED: NexusLoop responsibilities inside IntergraxAgent base class
REJECTED: Agent owns PolicyEngine, GraphExecutor, or AgentRegistry
REJECTED: Application.on_next_orchestration_step() mirroring Agent.on_next_step
REJECTED: Tier-3 multi-agent pipelines in factory.py while-loops
REJECTED: Private while-True agent loops without UAEP step boundaries
REJECTED: nexus.run() / NexusLoop as agent plan brain
REJECTED: CapabilityRegistry as parallel catalog (use AgentRegistry + CapabilityGraph)
REJECTED: Second execution engine or 23rd domain pair for "application contracts"
REJECTED: Monolithic implementation plan files under plan/phases/
```

---

## 9. Maintenance

| Event | Action |
|-------|--------|
| New cross-layer rule | Add or update [Cross-Layer System Invariants](#cross-layer-system-invariants) subsection; add §5 row if CI-mapped |
| New domain invariant (ACP-INV-*, APP-INV-*) | Add or update §5 row; link domain section — do not duplicate prose |
| New CI gate for an existing rule | Update §7 and the row's CI column |
| Semantic change | Domain architecture + ADR first; then this index |
| External audit (P2-ARCH-*) | Point auditors here; drill into §6 registers for evidence |

**Plan row:** [`PLATFORM_FOUNDATION.md`](../plan/PLATFORM_FOUNDATION.md) **P2-ARCH-01** (Done).

**ADR:** no ADR needed — index-only guide; canon unchanged.

---

## 10. Reading order

1. This file — [Cross-Layer System Invariants](#cross-layer-system-invariants) + §4 execution stack  
2. [`MATURITY_TAXONOMY.md`](MATURITY_TAXONOMY.md) — four-axis maturity vocabulary  
3. Hub [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
4. **One** domain pair for your task ([`AGENTS.md`](../../AGENTS.md) task routing)  
5. Author guides when building: [`AGENT_AUTHOR_MINIMAL_PATH.md`](AGENT_AUTHOR_MINIMAL_PATH.md) · [`TIER3_PRODUCT_HYPOTHESIS_CONTRACT.md`](TIER3_PRODUCT_HYPOTHESIS_CONTRACT.md) · [`AGENT_CREATION_GUIDE.md`](AGENT_CREATION_GUIDE.md) · [`APPLICATION_CREATION_GUIDE.md`](APPLICATION_CREATION_GUIDE.md)
6. Deep layer closeout (full domain): [`LAYER_COMPLETION_MODE.md`](LAYER_COMPLETION_MODE.md)
