# PLATFORM_FOUNDATION — production gates (§40+)

**Parent hub:** [`PLATFORM_FOUNDATION.md`](../PLATFORM_FOUNDATION.md)

# 43. Anti-Patterns

The following patterns are forbidden or strongly discouraged.

## 43.1 Fat Agent Anti-Pattern

Do not create agents that contain:

- routing
- global orchestration
- global memory
- scheduler
- UI logic
- platform state

---

## 43.2 Fat Nexus Anti-Pattern

Do not put domain-specific workflows directly inside Nexus.

Nexus should orchestrate, not become the agent.

---

## 43.3 UI-Driven Architecture Anti-Pattern

Do not design the runtime around a frontend screen.

The runtime must work from API, Slack, Teams, CLI or chat.

---

## 43.4 Prompt-Only Architecture Anti-Pattern

Do not treat prompts as the architecture.

Prompts are part of agents and reasoning, but the runtime must have real execution structures.

---

## 43.5 Unobservable Execution Anti-Pattern

Do not execute important steps without traces.

If it cannot be inspected, it cannot be trusted.

---

## 43.6 Product Too Early Anti-Pattern

Do not build billing, marketplace, advanced UI or enterprise features before validating the runtime.

---

## 43.7 Agent-Application Conflation Anti-Pattern

Do not implement agent capabilities inside `applications`.

Do not put FastAPI hosts, env settings, or HTTP serving inside `agents`.

See §7.4 for the canonical repository split.

See §42.41 for forbidden runtime patterns (direct adapters, private loops, runtime bypass).

---

## 43.8 Platform Redundancy Anti-Pattern

Do NOT introduce a second implementation of a universal platform concern when Tier-0 already provides one.

Examples of forbidden redundancy:

- new LLM adapter layer alongside `intergrax/llm_adapters`
- agent-local logging instead of `intergrax/logging.py`
- custom vector store client instead of `intergrax/rag`
- duplicate tool execution path instead of `ToolRuntime` → `ToolRegistry`
- new trace store instead of extending `RunTraceWriter` / existing trace pipeline
- direct SDK calls (OpenAI, Anthropic, Redis, boto3) in agents when Tier-0 adapter exists

If existing Tier-0 is genuinely insufficient for a **cross-cutting** need, follow §5.2.4 — propose to human, do not implement autonomously.

---


---

# 44. Decision Records

## 44.1 Decision: Nexus Has Global Loop

Decision:

Nexus owns the global reasoning and execution loop.

Reason:

Global coordination must be centralized to avoid chaotic autonomous agents.

---

## 44.2 Decision: Agents May Have Local Loops

Decision:

Agents may contain bounded local execution loops.

Reason:

Complex domain tasks require local multi-step execution.

Constraint:

Agent loops must be bounded by contracts, limits and validation rules.

---

## 44.3 Decision: Slack And Teams Are Adapters

Decision:

Slack and Teams are Tier-0 adapters / interaction surfaces.

Reason:

Communication tools should not own orchestration.

---

## 44.4 Decision: Intergrax Is A Laboratory First

Decision:

Intergrax is currently an internal experimentation runtime, not a full SaaS product.

Reason:

The current strategic goal is rapid hypothesis validation.

---

## 44.5 Decision: Agents Are Capabilities

Decision:

Agents are capability modules, not independent products.

Reason:

This allows rapid creation, replacement and composition.

---

## 44.6 Decision: Agents And Applications Are Separate Repository Roots

Decision:

Agent capability code lives under `agents/<name>`.

Execution environments (Tier-3: host, serving, env config) live under `applications/<name>`.

The framework package `intergrax` MUST remain free of product-specific agent implementations.

Reason:

The same agent can be reused across notebooks, tests, and multiple Tier-3 applications without duplicating domain logic.

---

## 44.7 Decision: Four-Tier Platform Model

Decision:

Intergrax is structured as Tier-0 (Platform), Tier-1 (Nexus Agent OS), Tier-2 (Agents), Tier-3 (Applications).

Reason:

Clear separation enables: reusable infrastructure, stable orchestration, swappable agents, and multiple isolated product environments from the same runtime — analogous to OS → apps → IDE products.

---

## 44.8 Decision: Unified Event-Driven Execution Runtime

Decision:

All agent execution MUST conform to §42 Unified Execution Runtime Specification: shared `AgentEngine`, `RuntimeEvent` stream, `AgentDecision` model, middleware pipeline, and ToolRuntime-only adapter access.

Reason:

Intergrax must be a governed Agent OS / Harness environment — not a loose collection of agent implementations. Unified execution enables reproducibility, inspectability, safe orchestration, and Cursor-like / Antigravity-like experimentation at scale.

Constraint:

Agents provide domain logic only. Runtime owns lifecycle, hooks, interrupts, retries, and governance.

---

## 44.9 Decision: Reuse Tier-0 — No Redundant Universal Mechanisms

Decision:

All implementation MUST reuse existing Tier-0 platform modules. New universal Tier-0 capabilities require explicit human approval before implementation.

Reason:

Intergrax already has canonical LLM, logging, tracing, tools, RAG, memory, and adapter layers. Duplicate mechanisms fragment observability, increase cost, and break the Agent OS model.

Constraint:

§42 orchestration wiring integrates with — does not replace — existing Tier-0. See §5.2, §8.8, §39.8.

---

## 44.10 Decision: Integration Library As Single Tier-0 Catalog

Decision:

All reusable **external infrastructure** adapters (databases, caches, queues, chat, search, issue trackers, observability backends, cloud infrastructure facades) MUST be registered in **`intergrax/integrations`** under category contracts (§7.1).

**Excluded:** LLM providers remain in **`intergrax/llm_adapters`** only — not in the Integration Library (§7.1.2).

Reason:

Multiple agent teams need the same building blocks (Redis cache, Jira tasks, Slack HITL). A catalog with universal contracts lets one platform team maintain adapters while product teams compose them in Tier-3 applications — without copying SDK code into `agents`.

Constraint:

- New **providers** follow Phase M checklist in the implementation plan.
- New **categories** require §5.2.4 human approval.
- Legacy modules (`queueing`, `distributed`, etc.) are wrapped, not duplicated.

---


---

# 48. Naming Guidance

Recommended naming:

- NexusRuntime
- AgentContract
- AgentRegistry
- ToolRegistry
- AdapterRegistry
- ExecutionContext
- TaskContext
- TaskState
- ExecutionGraph
- ExecutionNode
- TraceEvent
- AgentExecutionResult
- ValidationResult
- ShadowWorkspace
- SandboxSession

Names should be explicit and boring.

Avoid clever names that make the architecture harder for LLMs and humans to understand.

---


---

# 49. LLM Readability Rules For This Project

Because Cursor AI and LLM agents will read this project, code and documentation should follow these rules:

- use explicit names
- avoid hidden magic
- avoid ambiguous abstractions
- prefer small files with clear responsibility
- document public contracts
- include examples
- include state transitions
- include schemas
- include error cases
- avoid overly clever metaprogramming
- keep architecture boundaries visible

LLMs perform better when responsibilities are explicit.

---


---

# 50. Future Evolution

Intergrax may later evolve into:

- enterprise AI operating system
- organization-wide agent platform
- agent marketplace
- visual workflow builder
- autonomous business process runtime
- multi-tenant SaaS
- internal company AI worker ecosystem

But these are future possibilities.

**Near-term platform priority (post Phase R MVP):**

> Prove **production harness** on the certified Agent OS: reference business agents, skill catalog depth, and stable provider paths — while keeping the laboratory fast path for new hypotheses.

**Long-term evolution** (§50) — marketplace, multi-tenant SaaS, visual workflow builder — remains out of scope until harness proof (Phase S) is met.

---


---

# 51. Final Canonical Statement

Intergrax is a **four-tier AI platform**: Platform (Tier-0) → Nexus Agent OS (Tier-1) → Agents (Tier-2) → Applications (Tier-3).

Intergrax is a **unified, event-driven Agent OS and Harness AI runtime** governed by §42 Unified Execution Runtime Specification.

Intergrax serves **both** as an internal **agent experimentation laboratory** and as a **Harness AI environment** for production agent work. New capabilities SHOULD start in the lab workflow (§2); capabilities that ship to users MUST consume Integration → Tool → **Skill** → Agent (§5.3, §7.1.8) on the shared Nexus harness — not private runtimes or duplicated instruction packs.

Tier-1 Nexus is the global orchestration runtime (Agent OS).

Tier-2 agents are bounded capability modules with shared contracts and **runtime-controlled** local loops — executed exclusively through `AgentEngine`.

Tier-0 adapters are reusable integrations consumed **only** through `ToolRuntime` policy.

Tier-3 applications are configured environments that compose Nexus + agents for specific industries or use cases.

Every agent MUST emit `RuntimeEvent`s, return `AgentDecision`s, and pass through the middleware pipeline. No private runtimes. No direct adapter access. No execution bypass.

**Platform reuse is mandatory:** one canonical mechanism per universal concern (LLM, logging, tools, RAG, trace, memory). Do not duplicate Tier-0. New universal components require human approval (§5.2.4).

The architecture must optimize for rapid hypothesis validation, observability, modularity, enforceable execution governance, and clean separation of responsibilities.

The system should make it possible to quickly implement a new agent, run it through Nexus, observe results, evaluate business value and decide whether the capability deserves further investment.

This is the core architectural direction of Intergrax.

---


---

# 53. Harness Architecture Hardening Index

Post-U hardening topics are **owned by domain pairs** (architecture + plan), not this file.

| Topic | Architecture | Plan |
|-------|--------------|------|
| Capability graph | [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) §19 | [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) |
| Agent lifecycle governance | [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) §20 | same |
| Prompt registry | [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) §17 | same |
| Registry snapshots / assembly | [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) §18 | same |
| Context quality | [`MEMORY.md`](MEMORY.md) §19 | [`plan/MEMORY.md`](../plan/MEMORY.md) |
| Knowledge graph / hybrid retrieval | [`MEMORY.md`](MEMORY.md) §20 · [`INTEGRATIONS.md`](INTEGRATIONS.md) | [`plan/INTEGRATIONS.md`](../plan/INTEGRATIONS.md) |
| Evaluation operations | [`EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) §42 | [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) |
| Architecture metrics / debt | [`EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) §43 | same |
| Security / tenant isolation | [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.45–§42.46 | [`plan/UNIFIED_EXECUTION_RUNTIME.md`](../plan/UNIFIED_EXECUTION_RUNTIME.md) |
| Cost governance | [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.47 | same |
| Identity / trust / tenancy | [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.44 | same |
| Multi-agent coordination patterns | [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) §27 | [`plan/NEXUS_EXECUTION_FLOW.md`](../plan/NEXUS_EXECUTION_FLOW.md) |
| Modality plane | [`MODALITY.md`](MODALITY.md) | [`plan/MODALITY.md`](../plan/MODALITY.md) |

**Harness-first lock (normative):**

```text
Harness -> Runtime -> Agents -> Applications -> Products
```

Business-agent work (K.1/K.2, product apps) remains deferred until explicit product reprioritization — see [`plan/PLATFORM_FOUNDATION.md`](../plan/PLATFORM_FOUNDATION.md) §6.3.

**Phase V implementation baseline** (code modules, not duplicated here): `intergrax/runtime/architecture` — capability graph, lifecycle, eval, security, cost, context/prompt quality, graph RAG helpers. Traceability: [`plan/PLATFORM_FOUNDATION.md`](../plan/PLATFORM_FOUNDATION.md) Phase V / FAUDIT-32.

---
