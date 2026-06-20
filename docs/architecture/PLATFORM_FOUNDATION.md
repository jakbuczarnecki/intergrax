# Platform Foundation

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/PLATFORM_FOUNDATION.md`](../plan/PLATFORM_FOUNDATION.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 1–2, 32  
**Audit instruction:** [`audit/PLATFORM_FOUNDATION.md`](../audit/PLATFORM_FOUNDATION.md)  
---

# 1.1 Documentation boundary (platform vs product)

**In scope for this document and for [`plan/PLATFORM_FOUNDATION.md):**

- **Intergrax Harness AI / Agent OS** — Tier-0 platform, Tier-1 Nexus runtime, reference Tier-2/Tier-3 wiring patterns, and the **infrastructure** to run agent environments.

**Out of scope (each artifact owns its own canon):**

| Artifact | Own documentation (architecture · roadmap · implementation) |
|----------|----------------------------------------------------------------|
| **Tier-3 business environment** (`applications/<product>/`) | `ARCHITECTURE.md`, `IMPLEMENTATION_PLAN.md`, product `README.md` — e.g. [`applications/local_workspace_application/`](../applications/local_workspace_application/) |
| **Tier-2 business agent** (`agents/<name>/`) | `ARCHITECTURE.md`, agent `README.md`, local notebooks/tests — e.g. `agents/local_indexer/` |

Platform docs describe **how to compose** agents and application hosts on the Harness. They do **not** replace product-specific architecture or deployment plans for a given business environment or business agent.

Navigation: [README.md — Documentation index](../README.md#documentation-index) · [`INTERGRAX_DEVELOPMENT_STRATEGY.md`](guides/INTERGRAX_DEVELOPMENT_STRATEGY.md) §Documentation boundary · plan [§4.0a](../plan/PLATFORM_FOUNDATION.md#40a-implementation-scope-split-infrastructure-vs-business)

---


---

# 2. Executive Summary

Intergrax is an AI Operating System / Agent Runtime / **Harness AI environment**.

**Strategic goal (priority 1):** build a **production-grade Harness AI** and Agent OS — orchestration, tools, skills, context, policy, trace, and composable agents at a standard comparable to modern agent platforms (Cursor, Claude Code, Codex-class harnesses, Viktor, enterprise agent runtimes). See [`INTERGRAX_DEVELOPMENT_STRATEGY.md`](guides/INTERGRAX_DEVELOPMENT_STRATEGY.md).

**Operating model — two modes on one codebase:**

| Mode | Goal |
|------|------|
| **Laboratory** | Rapid experimentation — create, run, observe, validate or discard agent hypotheses quickly |
| **Production harness** | Certified runtime + reference business agents + stable integration paths + operational observability |

Intergrax is **not** a finished multi-tenant SaaS product today (§4). The laboratory mode remains the **fast path for new ideas**; production harness is the **strategic destination** for agents that graduate from experiments (Phase S in the implementation plan).

The ideal workflow is:

```text
new idea
    -> define agent capability
    -> implement agent contract
    -> register agent in Nexus
    -> connect integrations, tools, and skills (Skill Library MVP)
    -> run experiment
    -> observe traces, cost, quality and failures
    -> validate or reject hypothesis
```

Intergrax should make it easy to test ideas such as:

- Problem Radar Agent
- Customer Vendor / Partner Discovery Agent
- Legal Agent
- UX Agent
- PM Agent
- Research Agent
- Onboarding Agent
- Sales Analysis Agent
- Business Process Agent

The core asset is not any single agent.

The core asset is the runtime that allows agents to be created and tested quickly.

The platform is organized in **four tiers** (see §5.1): Platform → Nexus (Agent OS) → Agents → Applications.

Execution is governed by **§42 Unified Execution Runtime Specification** — event-driven orchestration, shared `AgentEngine`, hooks, `AgentDecision`, interrupts, and ToolRuntime enforcement.

Platform work MUST **extend and wire** existing Tier-0 modules — not introduce parallel universal mechanisms (§5.2).

**Capability stack (Tier-0 + Tier-2):** Integration → Tool → **Skill** → Agent (§7.1.6–§7.1.8). Skills are composable packs; tools remain atomic LLM operations.

**Model & modality stack (Tier-0):** three **modality planes** — generative LLM (§5.2.2), media ingest/RAG (§7.1.2), dedicated inference — vision CV, speech, classical ML (§7.1.9). Catalog index: [`architecture/MODALITY.md`](architecture/MODALITY.md).

---


---

# 3. What Intergrax Is

Intergrax IS:

- a **four-tier AI platform** (Platform → Nexus → Agents → Applications — §5.1)
- an AI Operating System (Nexus / Tier-1)
- an Agent Runtime
- a Harness AI Environment
- an Orchestration Runtime
- an Agent Experimentation Laboratory
- a Capability Execution Platform
- a runtime for testing business and technical agent hypotheses
- a system for integrating agentic work with real organizational tools
- a **Skill Library** for reusable capability packs (tools + prompts + policy) above the Tool Library (§7.1.8; **MVP Done**, Phase R)

Intergrax is designed to answer this question:

> Can we rapidly create, run and evaluate new AI agents without rebuilding infrastructure every time?

---


---

# 4. What Intergrax Is Not

Intergrax is NOT:

- a chatbot
- a simple LLM wrapper
- a prompt collection
- a single agent
- a group chat between agents
- a frontend-heavy SaaS product
- a workflow builder at this stage
- a marketplace at this stage
- a clone of NotebookLM
- a direct competitor to Cursor AI
- a direct competitor to Viktor
- a product-first startup at this stage

Intergrax should learn from Cursor AI, Viktor, NotebookLM and modern agent runtimes. The **laboratory** optimizes for controlled experimentation; the **harness** optimizes for governed, repeatable production agent work — same runtime, different maturity gates (implementation plan Phase L → Q/Q+/R → S).

---


---

# 5. Core Architectural Thesis

Modern AI systems are moving away from isolated chatbots and toward runtime environments for intelligent work.

The strongest systems are not only models.

The strongest systems are environments that provide:

- orchestration
- execution lifecycle
- tool access
- memory boundaries
- task state
- integrations
- observability
- retries
- validation
- sandboxing
- human-in-the-loop controls
- agent registration
- capability composition

Intergrax follows this direction.

The main thesis is:

> The future value is not in building one agent. The value is in building the runtime that allows many agents to be built, tested and orchestrated quickly.

---

# 5.1 Four-Tier Platform Model

Intergrax is organized as a **four-tier platform**. This is the canonical mental model for the entire system.

Think of it like a classical software stack:

```text
Tier-3  Applications     →  ready-made environments (like an IDE product built on an OS)
Tier-2  Agents           →  specialized programs (like user applications)
Tier-1  Nexus Runtime    →  agent operating system (like Linux / Windows for apps)
Tier-0  Platform         →  universal infrastructure (like drivers, libc, network stack)
```

## Tier-0 — Platform (Universal Components)

**Role:** reusable, domain-agnostic technical building blocks.

Tier-0 provides capabilities that **any** runtime or agent may use. It does not know about business tasks, agent roles, or product configuration.

**Includes:**

- LLM providers and tokenization (`intergrax/llm_adapters/`, `intergrax/tokenizers/`)
- Memory and conversation history primitives (`intergrax/memory/`, session storage)
- RAG: embeddings, vector stores, document loaders (`intergrax/rag/`)
- Tool integrations and invokers (`intergrax/tools/`, websearch, multimedia)
- Infrastructure adapters: PostgreSQL, Redis, queues, Kafka, file storage — **catalogued in** `intergrax/integrations/` (§7.1)
- Network and ingestion: HTTP clients, web fetch, file parsing
- Observability primitives: logging, error models, trace persistence backends
- Shared utilities: time, config helpers, idempotency stores

**Rules:**

- Tier-0 MUST NOT contain orchestration logic.
- Tier-0 MUST NOT contain agent business logic.
- Tier-0 MUST NOT decide which agent runs.
- Tier-0 exposes **stable interfaces** (adapters, managers, registries).
- Tier-0 is the **single source of truth** for each universal platform capability (§5.2).
- Higher tiers MUST consume Tier-0 through existing entry points — not reimplement them.

**Repository:** primarily `intergrax/` subpackages **outside** `runtime/nexus/` orchestration (e.g. `adapters/`, `rag/`, `tools/`, `queueing/`, `llm_adapters/`, `memory/`, `logging.py`).

---

## 5.2 Platform Reuse And No-Redundancy Principle

This is a **mandatory** architectural constraint for all implementation agents (humans, Cursor AI, autonomous coders).

### 5.2.1 Default Rule: Reuse, Do Not Reinvent

When implementing Tier-1 (Nexus), Tier-2 (agents), or Tier-3 (applications):

```text
DO:     compose, configure, orchestrate, and wire EXISTING Tier-0 modules
DO NOT: implement new universal mechanisms that duplicate Tier-0 responsibilities
```

Implementation work is primarily **integration and orchestration** — not rebuilding platform infrastructure.

### 5.2.2 Canonical Single Mechanisms (Examples)

The platform MUST maintain **one canonical path** per universal concern. All tiers MUST use it.

| Concern | Canonical Tier-0 mechanism | Forbidden |
|---------|---------------------------|-----------|
| LLM calls | `intergrax/llm_adapters/` (`LLMAdapter`, `LLMAdapterResponse` envelope, registry, `LLMProfile`, metrics, resilience, Nexus `llm_tenant_scope`; replay via `trace_replay_bridge`) | Direct vendor SDKs in agents; duplicate LLM stacks; bare `str` adapter returns |
| Logging | `intergrax/logging.py` and established log patterns | `print()`, ad-hoc loggers, duplicate logging frameworks |
| Tracing (pipeline) | Nexus `trace_event()` / `RunTraceWriter` | Parallel untracked diagnostic streams |
| Tools | `intergrax/tools/` (`ToolRegistry`, `ToolExecutor`, Tool Library §7.1.6) | Agent-local tool registries; boolean `use_rag` / `use_websearch` plan flags (deprecated §22.2) |
| RAG | `intergrax/rag/` (`RagProfile`, `RetrievalService`, `IngestPipeline`) | Duplicate embedding/retrieval stacks; dense-only `vectorstore.query` bypass in agents/Nexus |
| Ephemeral Code Craft | `intergrax/codecraft/` + `runtime/codecraft/` (`CodeCraftProfile`, `CodeCraftOrchestrator`) | Agent-local generate→exec loops; parallel sandbox runtimes; ephemeral tools in global ToolRegistry |
| Web search | `intergrax/websearch/` | Custom HTTP search clients in agents |
| Memory / session | `intergrax/memory/`, Nexus session storage | Direct Redis/PostgreSQL access from agents |
| Queues | `intergrax/queueing/` | Ad-hoc background job systems |
| Tokenization | `intergrax/tokenizers/` | Inline tiktoken/token counting duplicates |
| File / storage adapters | Tier-0 adapters | Agent-local S3/filesystem clients bypassing adapters |
| External integrations (DB, cache, chat, search, …) | `intergrax/integrations/` catalog + category contracts | Direct vendor SDK imports in `agents/`; LLM slugs in Integration Library |
| Vendor SDK bridges | `intergrax/llm_adapters/providers/*/_sdk_bridge.py`, `integrations/providers/*/` | `getattr` reflection in `runtime/` or `agents/` (Q+-I.1 quarantine — bridges only) |
| Errors / classification | `intergrax/runtime/nexus/errors/` | Siloed error models per agent |

This table is illustrative, not exhaustive. The rule is general:

> **If a universal capability already exists in Tier-0, use it. Do not add a second one.**

### 5.2.3 What §42 Contracts Are (And Are Not)

§42 Unified Execution Runtime defines **orchestration contracts and governance wiring** — how Nexus and agents coordinate execution.

§42 does **NOT** authorize creating duplicate Tier-0 implementations. For example:

- `RuntimeEvent` MUST integrate with — not replace — existing trace/logging infrastructure.
- `ToolRuntime` MUST delegate to existing `ToolRegistry` / tool steps — not a parallel tool system.
- `AgentEngine` MUST use existing `AgentEngine` / pipeline — not a second execution engine.

When §42 scaffold modules are wired (Phase P4+), they MUST **wrap and unify** existing mechanisms, not fork them.

### 5.2.4 Human Approval Required For New Universal Mechanisms

If implementation requires a **new universal mechanism** — a component that:

- would live in Tier-0 (or behave like Tier-0),
- would be reused by multiple agents, Nexus, or applications,
- would introduce a **new class** of platform capability not covered by existing modules,

then the implementation agent MUST **STOP and request human decision** before writing code.

Required escalation format:

```text
PROPOSAL: New universal Tier-0 mechanism
Name: <proposed component>
Problem: <why existing Tier-0 is insufficient>
Alternatives considered: <extend existing X / configure Y / orchestrate in Tier-1>
Redundancy risk: <what would be duplicated>
Recommendation: <extend existing | new module — needs approval>
```

The human operator decides whether to:

1. **Extend** an existing Tier-0 module (preferred),
2. **Configure** Tier-1 orchestration only (preferred when sufficient),
3. **Approve** a new Tier-0 module (exception — requires explicit consent).

Autonomous agents MUST NOT silently add new universal integrations, adapters, registries, or infrastructure layers.

### 5.2.5 Allowed Without Escalation

These do **NOT** require human approval (normal development):

- New **Tier-2 agent** domain logic (prompts, steps, models) using existing Tier-0 via Tier-1 policy.
- New **Tier-3 application** wiring (host, config, agent roster).
- Tier-1 **orchestration** glue that calls existing Tier-0 APIs (NexusLoop, hooks wiring, policy rules).
- Tier-2 **agent-local** models, prompts, validation rules scoped to one capability.
- Bug fixes and conformance improvements in existing Tier-0 modules.

### 5.2.6 Redundancy Is An Architectural Defect

Duplicate mechanisms cause:

- inconsistent observability and cost tracking,
- divergent failure handling,
- higher maintenance and LLM-agent confusion,
- violation of the Agent OS / harness model.

Redundant implementations MUST be rejected in code review — even if they "work" in isolation.

Reference: §8.8, §39.8, §42.41, §43.8.

---

## Tier-1 — Nexus Runtime (Agent Operating System)

**Role:** the **Agent OS** — orchestrates agents the way an operating system orchestrates applications.

Nexus uses Tier-0 components to create a **controlled execution environment** for agents: lifecycle, routing, policy, shared services, and observability.

**Includes:**

- Global Nexus loop (`NexusLoop`, task intake, classification, planning); implementation split under `runtime/nexus/orchestration/` (`intake_runner`, `planning_runner`, `graph_runner`, `hitl_runner`, `task_events`, `lifecycle_bridge`, …) — loop file orchestrates only
- Agent registry and capability routing (`AgentRegistry`, `AgentRouter`)
- Task lifecycle and state machine (`Task`, `TaskLifecycle`)
- Execution graph and multi-agent coordination (`ExecutionGraph`, `GraphExecutor`)
- Context management and memory coordination policy
- Tool runtime gateway and adapter access policy
- Validation, retry, failure handling, human-in-the-loop gates
- Contracts: `AgentContract`, `AgentExecutionResult`, `ValidationResult`
- Trace system integration at task level
- `AgentEngine` bridge (Nexus → agent local loop)

**Analogy:** Nexus is to agents what an OS kernel + scheduler is to applications. Agents **run inside** Nexus; they do not replace it.

**Rules:**

- Tier-1 MUST remain **domain-agnostic** (no Legal logic, no UX logic inside Nexus).
- Tier-1 MUST NOT implement concrete agent business workflows.
- Tier-1 owns **global** orchestration; agents own **local** bounded execution.

**Repository:** `intergrax/runtime/`, `intergrax/contracts/`, `intergrax/agents/` (framework ABC only — **not** concrete agents).

---

## Tier-2 — Agents (Specialized Capability Modules)

**Role:** fully functional, domain-specialized modules that perform **concrete business or technical work**.

Each agent is a bounded capability: researcher, UX designer, PM, tester, marketer, legal reviewer, vendor discovery, etc.

**Includes per agent (`agents/<name>/`):**

- Agent class implementing `Agent` + `AgentContract`
- Declared capabilities (e.g. `research.web_search`, `legal.contract_review`)
- Local processing loop: pipeline, steps, prompts, domain models
- Agent-local governance and validation rules
- Agent-local tracing helpers
- Optional local tool bridge (via Tier-1 `ToolRuntime`, not raw Tier-0 bypass)

**Rules:**

- Agents MUST implement shared contracts (`get_contract()`, `can_handle()`, `build_context()`, `validate()`).
- Agents MAY have **bounded local loops** (multi-step domain execution).
- Agents MUST NOT own global orchestration, global routing, or HTTP host wiring.
- Agents consume Tier-0 **through** Tier-1 policies (not uncontrolled direct access in production).
- Agents MUST be runnable via Nexus without starting an HTTP server.

**Repository:** `agents/` at repository root (`agents/legal/`, `agents/research/`, `agents/echo/`, …).

---

## Tier-3 — Applications (Ready-Made Environments)

**Role:** **isolated, configured environments** that compose Nexus + a selected set of agents + rules + integrations for a specific context.

An application is not an agent. It is the **product shell** — the “Cursor AI for X” pattern: a ready environment for a defined industry, company type, or use case.

**Includes per application (`applications/<name>/`):**

- Host entrypoint (`main.py`, `factory.py`, `settings.py`, `wiring.py`)
- HTTP/CLI serving layer (routes, auth, tenant config)
- **Self-contained operational configuration** — own `.env` and `.env.example` (application-prefixed variables; see §7.4.8)
- Environment profiles (dev/staging/prod), SKU rules, feature flags
- Agent registry wiring: which agents are registered, with which IDs and policies
- `IntegrationProfile` composition — which Tier-0 backends this environment uses
- Orchestration config: default capabilities, routing hints, multi-agent topologies
- **Deployment package** — `docker/` (Dockerfile, optional `docker-compose.yml`) sufficient to build an image and push to production (see §7.4.8)

**Self-sufficiency rule:** A Tier-3 application is a **runnable, deployable environment** on its own. A developer MUST be able to start the host and build a container using **only** files under `applications/<name>/` plus the monorepo Python dependencies (`pyproject.toml` / `uv` at repository root). Application-specific secrets and toggles MUST NOT live only in the repository-root `.env.example`.

**Example environments:**

- `legal_application` — legal review for law firms (Legal agent + compliance rules)
- `research_application` — research → summarize pipeline for analysts
- `intergrax_assistant_application` — harness-native conversational lab (hub agent + swappable LLM + optional specialist delegation) — see §7.4.11
- `local_workspace_application` — Local Knowledge Workspace (LKW)
- `dispute_sim_application` — Dispute Simulation Workspace (DSW)
- Future: `agency_application`, `saas_pm_application`, `ecommerce_ux_application`

**Rules:**

- Applications MUST NOT contain agent domain logic (pipeline steps, prompts).
- Applications compose Tier-2 agents; they do not reimplement them.
- Multiple applications MAY reuse the same Tier-2 agent with different config.
- Applications are the only layer that binds **product-specific** env vars and deployment.

**Repository:** `applications/` at repository root.

---

## Tier Mapping Summary

| Tier | Name | Role | Analogy | Repository |
|------|------|------|---------|------------|
| **0** | Platform | Universal components & adapters | Drivers, network stack, libc | `intergrax/` (non-orchestration packages) |
| **1** | Nexus Runtime | Agent OS — orchestration & policy | Operating system | `intergrax/runtime/`, `intergrax/contracts/`, `intergrax/agents/` (ABC) |
| **2** | Agents | Specialized capability modules | Applications / programs | `agents/<name>/` |
| **3** | Applications | Configured environments | IDE product, industry workspace | `applications/<name>/` |

## Dependency Direction (Strict)

```text
Tier-3 Applications  →  Tier-2 Agents  →  Tier-1 Nexus  →  Tier-0 Platform
```

- Higher tiers import lower tiers only.
- Tier-0 MUST NOT import agents or applications.
- Tier-1 MUST NOT import concrete agents or applications.
- Tier-2 MUST NOT import applications.

**Cross-domain invariant index:** [`guides/SYSTEM_INVARIANTS.md`](../guides/SYSTEM_INVARIANTS.md) (P2-ARCH-01) — terse “never violate” rules with links to this section and domain pairs.

**Enforcement (FAUDIT-TIER, 2026-06-06):** Tier-3 application manifest metadata for harness capability-graph seeding lives in `intergrax/applications/reference/harness_manifest_catalog.py` (static reference data, not `from applications.*` under `intergrax/`). CI: `scripts/check_intergrax_no_applications_imports.py` and `scripts/check_agents_no_tier3_imports.py`.

## Relationship To “Layer 1 / 2 / 3” Naming

Earlier sections and diagrams may refer to **Layer 1 / 2 / 3**. Mapping:

| Legacy layer name | Canonical tier |
|-------------------|----------------|
| Layer 1 — Components / Adapters | **Tier-0** Platform |
| Layer 2 — Nexus Runtime | **Tier-1** Nexus (Agent OS) |
| Layer 3 — Agents | **Tier-2** Agents |
| *(not in old 3-layer model)* | **Tier-3** Applications |

**Always prefer Tier-0..3 terminology** in new code and documentation.

## Code Labels (`DeploymentTier`)

The package `intergrax.agent_kit.tiers` exposes `DeploymentTier` enum labels aligned with this model:

- `PLATFORM` (0) → Tier-0
- `FRAMEWORK` (1) → Tier-1
- `AGENT` (2) → Tier-2
- `APPLICATION` (3) → Tier-3

(`PRODUCT` is a deprecated alias for `AGENT` in legacy metadata.)

---

## 5.3 Harness AI Alignment (Conceptual Model)

Intergrax is a **Harness AI environment** (Agent OS). Industry harness literature uses vocabulary that maps to Intergrax as follows.

### 5.3.0 Terminology (Harness vs Application vs Agent)

| Term | Tier | Role |
|------|------|------|
| **Platform / Tier-0** | 0 | Catalogs: integrations, tools, skills, LLM adapters, modality inference |
| **Nexus / Runtime** | 1 | Orchestration loop, policy engine, trace, context, graph execution |
| **Agent** | 2 | Autonomous business logic: UAEP steps, `AgentContract`, prompts (`agents/`) |
| **Application** | 3 | Deployable **environment**: manifest, `ApplicationEnvironmentProfile`, host wiring (`applications/`) |
| **Harness (practical)** | 1+3+0 | Nexus + application wiring + platform catalogs — not a single Python package |
| **Product** | — | Business offering composed of Tier-3 app + selected Tier-2 agents |

IDEAL chain: `Harness → Runtime → Agents → Applications → Products`. Intergrax **Application** = Tier-3 host (IDEAL “environment”), not the Tier-2 agent module.

### 5.3.1 Core mapping

| Harness AI term | Intergrax implementation |
|-----------------|---------------------------|
| **Scaffold** | `python -m intergrax.scaffold` (`new-agent`, `new-application`, `new-stack`, `new-skill`) |
| **Harness** | Tier-1 **Nexus** + Tier-0 platform + Tier-3 **Application** wiring (policy, tools, integrations, trace) |
| **LLM** | Tier-0 `intergrax/llm_adapters/` — invoked per step/plan; not embedded inside Tier-2 agent class |
| **Agent** | Tier-2 module (`agents/<name>/`) implementing `Agent` + `AgentContract` + UAEP |
| **Runnable agent instance** | Harness + selected agent + `LLMProfile` + resolved `skill_ids` / `allowed_tools` + `RuntimePolicyBundle` for one run |
| **Tool** | Tier-0 atomic `ToolContract` — LLM/MCP/FastAPI invocable (§7.1.6) |
| **Skill** | Tier-0 composable **`SkillManifest`** — tools + prompts + policy fragment (§7.1.8) |
| **Context engineering** | Tier-1 `ContextManager` + `TaskContextAssemblyOptions` + `MemoryView` + `ContextBudgetPolicy` (§28.1) |
| **Subagent** | **Graph delegation** — Nexus `ExecutionGraph` child node, not nested OS (§42.14.3) |
| **Policy** | `PolicyEngine`, `ToolAccessPolicy`, budgets, HITL, org profiles — composed as `RuntimePolicyBundle` (§42.11.4) |
| **Guardrails** | Cross-cutting enforcement vector of Policy & Governance — not a separate tier. Hook-time checks (prompt, tool, output, cost, time) mapped in UAEP §42.11.6; optional vendor engines via Integration category `llm_guardrail` ([`INTEGRATIONS.md`](INTEGRATIONS.md) §47) |
| **Modality / ML** | Planes B+C via **tools** + optional **`ModalityProfile`** (§7.1.9); generative vision/audio via **`LLMProfile`** (Plane A); never vendor SDKs in agents |

### 5.3.2 Agent composition (not harness + LLM only)

```text
Harness (Nexus + app wiring)
    → runs Tier-2 Agent
        → composes SkillManifest(s)  →  resolves tool_ids, prompts, policy fragments
        → AgentEngine / UAEP steps
        → ToolRuntime.invoke(tool_id)  →  Integration adapters
        → LLM adapters (per step / planner)
        → Modality tools (vision.detect, speech.*, ml.predict)  →  Plane C registry / speech_provider
```

Agents MUST NOT call integrations directly. Agents MUST NOT import CV/ML SDKs (`ultralytics`, `torch`, `onnxruntime`, …) when a catalog tool or adapter exists (§7.1.9). Skills MUST NOT replace `ToolRuntime` or appear as fake `ToolContract` entries.

### 5.3.3 Architectural decision: Skill layer (ADR)

| Option | Description | Verdict |
|--------|-------------|---------|
| **1 — Skills = tools** | Encode instructions + multi-tool workflows as oversized tools | **Rejected** — breaks atomic LLM function schema, MCP export, risk/idempotency per operation, and external tool ecosystems |
| **2 — Skill Library** | Fourth layer: Integration → Tool → **Skill** → Agent | **Adopted** — **MVP Done**; importers for external formats (e.g. Cursor `SKILL.md`) after manifest validation |

Implementation tracker: [`plan/PLATFORM_FOUNDATION.md) Appendix E · catalog [`architecture/SKILLS.md`](architecture/SKILLS.md).

---


---

# 6. High Level Architecture

Intergrax consists of **four platform tiers** (see §5.1). The diagram below shows Tier-0 through Tier-3.

```text
+--------------------------------------------------------------+
|                      TIER-3 — APPLICATIONS                   |
|              Ready-made configured environments              |
|--------------------------------------------------------------|
| legal_application          research_application              |
| agency_workspace           saas_pm_environment   (future)  |
|  • host + serving + env config + agent registry wiring       |
|  • industry rules, roles, interaction topology               |
+--------------------------------------------------------------+

+--------------------------------------------------------------+
|                        TIER-2 — AGENTS                       |
|           Specialized capability modules (domain)            |
|--------------------------------------------------------------|
| LegalAgent    ResearchAgent    UXAgent       PMAgent         |
| TesterAgent   MarketerAgent    VendorDiscoveryAgent (future K) ... |
|  • contracts, pipelines, steps, local loops                  |
|  • business logic; runs inside Nexus                         |
+--------------------------------------------------------------+

+--------------------------------------------------------------+
|                   TIER-1 — NEXUS RUNTIME                     |
|                    Agent Operating System                    |
|--------------------------------------------------------------|
| NexusLoop          AgentRegistry       TaskLifecycle         |
| ExecutionGraph     AgentRouter         ContextManager        |
| ValidationEngine   RetryEngine         ToolRuntime           |
| AgentEngine        Trace coordination  Human approval        |
|  • global orchestration; domain-agnostic                     |
+--------------------------------------------------------------+

+--------------------------------------------------------------+
|                    TIER-0 — PLATFORM                         |
|              Universal components & adapters                 |
|--------------------------------------------------------------|
| LLM Providers    Memory / History    RAG / Vector Store      |
| PostgreSQL       Redis               Queue / Kafka           |
| Web Search       File Storage        Logging / Errors        |
| Slack / Teams    Browser             Sandbox executor        |
|  • no orchestration; no agent business logic                 |
+--------------------------------------------------------------+
```

**Execution flow:**

```text
User / API (Tier-3)
    → Nexus intake (Tier-1)
    → select & run agents (Tier-2)
    → agents call platform services (Tier-0) under Nexus policy
    → Nexus validates, traces, composes response
    → Application returns result (Tier-3)
```

---


---

# 7. Layer Responsibility Summary

> **Canonical naming:** Tier-0..3 (§5.1). Subsections below retain legacy “Layer N” labels where noted.

## 7.0 Tier Overview

| Tier | Section | Package / folder |
|------|---------|------------------|
| Tier-0 Platform | §7.1 | `intergrax/` + **`integrations/`** + **`tools/`** + **`skills/`** catalogs; rag, memory, queueing, … |
| Tier-1 Nexus | §7.2 | `intergrax/runtime/`, `intergrax/contracts/` |
| Tier-2 Agents | §7.3 | `agents/<name>/` |
| Tier-3 Applications | §7.4 | `applications/<name>/` |

---

## 7.1 Tier-0: Platform (legacy: Layer 1 — Components / Adapters)

Layer 1 (Tier-0) contains reusable technical integrations.

Examples:

- database adapters
- cache adapters
- message queue adapters
- Slack adapter
- Teams adapter
- email adapter
- browser automation adapter
- file system adapter
- vector store adapter
- LLM provider adapter
- sandbox adapter
- logging adapter

Layer 1 (Tier-0) MUST NOT contain orchestration logic.

Layer 1 (Tier-0) MUST NOT contain business-specific agent logic.

Layer 1 (Tier-0) exposes capabilities to Nexus and agents through stable interfaces.

### 7.1.1 Integration Library — Canonical Catalog

Tier-0 external integrations MUST live in a **single, discoverable catalog** under:

```text
intergrax/integrations/
├── contracts/          # category-level Protocol / ABC (backend-agnostic)
├── registry/           # IntegrationRegistry, factory, capability lookup
├── _shared/            # config schema, health probes, retry helpers
└── providers/
    └── <category>/       # IntegrationCategory (relational_store, message_bus, …)
        └── <slug>/       # vendor package (postgresql, s3, slack, …)
            ├── __init__.py
            ├── adapter.py          # implements category contract(s) (optional for thin P2 shells)
            ├── bundle.py           # create_* factory — composition root for Tier-3
            ├── opens.py            # ONLY module that imports vendor SDK (when full package)
            ├── config.py           # pydantic settings + env keys
            ├── config.example.yaml # copy-paste for Tier-3 applications (optional)
            ├── USAGE.md            # English: env vars, IntegrationProfile, factory example
            └── tests/              # under tests/unit/integrations/providers/<category>/ in repo root
```

**Layout map:** `intergrax/integrations/providers/layout.py` — slug → category folder.

**Documentation:** all **167** registered providers ship `providers/<category>/<slug>/USAGE.md` (English). Regenerate via `scripts/generate_integration_usage_docs.py`. Catalog index: [`docs/architecture/INTEGRATIONS.md`](architecture/INTEGRATIONS.md).

**Rules:**

- One **provider folder per integration** under **`providers/<category>/<slug>/`** — category matches `IntegrationCategory` (see `providers/layout.py`).
- Providers implement **category contracts** from `integrations/contracts/` — not ad-hoc SDK wrappers.
- Agents and Nexus MUST NOT import vendor SDKs directly when a catalog provider exists (§5.2).
- Existing Tier-0 modules (`queueing/`, `distributed/`, `websearch/`, `rag/`, `runtime/notifications/`, `runtime/interactions/`) remain valid; new work and refactors **register through** `IntegrationRegistry` and gradually wrap legacy providers (evolve, not rewrite).
- **`intergrax/llm_adapters/` is out of scope** for the Integration Library — LLM providers use `LLMAdapterRegistry` (§5.2.2), not `IntegrationRegistry`. Cloud facades (`aws`, `azure`, `gcp`) MUST NOT wrap or re-export Bedrock, Azure OpenAI, Vertex, or other LLM adapters.
- Production access from agents goes through **`ToolRuntime`** (tools) or **Tier-3 wiring** (stores, queues, notifications) — never raw clients in `agents/`.

**Separation of concerns:**

| Layer | Owns |
|-------|------|
| `integrations/contracts/` | What “a PostgreSQL adapter” or “a notification channel” MUST expose |
| `integrations/providers/<category>/<slug>/` | How a specific vendor satisfies the contract |
| `integrations/registry/` | Discovery, env-based factory, health aggregation |
| Tier-3 `applications/<name>/` | Which integrations are enabled for a product environment |
| Tier-2 `agents/<name>/` | Domain logic; declares **capability needs**, not vendor wiring |

### 7.1.2 Integration Categories And Abstract Contracts

Each category defines a **small, stable contract** (Protocol or ABC). Providers may implement one or more categories.

| Category | Contract module (planned) | Purpose | Example providers |
|----------|---------------------------|---------|-------------------|
| **relational_store** | `contracts/relational_store.py` | SQL CRUD, migrations hook, tenant-scoped connections | sqlite, postgresql, mysql, oracle, mssql, databricks |
| **document_store** | `contracts/document_store.py` | Document / wide-column CRUD | mongodb, cassandra, dynamodb |
| **key_value_cache** | `contracts/key_value_cache.py` | Cache, distributed locks, idempotency | redis, memcached |
| **message_bus** | `contracts/message_bus.py` | Async tasks, pub/sub, consumer groups | kafka, rabbitmq, celery, sqs, service_bus, pubsub, temporal, nats |
| **object_storage** | `contracts/object_storage.py` | Blob read/write, presigned URLs | s3, azure_blob, gcs, minio, filesystem |
| **vector_store** | `contracts/vector_store.py` | Embedding index (delegates to `rag/` impl) | qdrant, pinecone, chroma, weaviate, milvus, inmemory |
| **search_provider** | `contracts/search_provider.py` | Web / enterprise search | google_cse, bing, brave, serpapi, tavily, exa |
| **notification_channel** | `contracts/notification_channel.py` | Outbound alerts (HITL, escalation) | slack, teams, email_smtp, webhook, discord, twilio |
| **secrets_store** | `contracts/secrets_store.py` | Tenant credentials, API keys | vault |
| **graph_store** | `contracts/graph_store.py` | Agent memory graphs, dependencies | neo4j |
| **interaction_surface** | `contracts/interaction_surface.py` | Inbound events → canonical Task | slack, teams, lab_json |
| **collaboration_suite** | `contracts/collaboration_suite.py` | Mail, calendar, directory (MS365, Google) | ms365_graph, google_workspace |
| **issue_tracker** | `contracts/issue_tracker.py` | Issues, sprints, comments | jira, azure_devops, github, linear |
| **wiki_knowledge** | `contracts/wiki_knowledge.py` | Pages, spaces, search | confluence, notion, sharepoint |
| **observability_backend** | `contracts/observability_backend.py` | Metrics, logs, traces, error tracking | prometheus, elasticsearch, otel, langfuse, datadog, clickhouse, sentry, langsmith, helicone, posthog, braintrust, signoz, honeycomb, arize, phoenix, wandb, opensearch |
| **browser_automation** | `contracts/browser_automation.py` | Headless fetch / interact | playwright, firecrawl, selenium |
| **cloud_platform** | `contracts/cloud_platform.py` | Unified auth, region, credential chain; factory for native **infrastructure** services (storage, queues, secrets — not LLM) | aws, azure, gcp |

Category contracts MUST be **backend-agnostic**: same method names and DTOs whether the backend is SQLite or Oracle.

**Out of scope for `intergrax/integrations/` (separate Tier-0 modules):**

| Concern | Canonical module | Notes |
|---------|------------------|-------|
| **LLM providers** | `intergrax/llm_adapters/` (`LLMAdapter`, `LLMAdapterRegistry`, `LLMProfile`, metrics) | 19 slugs — [architecture/LLM_ADAPTERS.md](architecture/LLM_ADAPTERS.md) §5.2.2 |
| **Tokenization** | `intergrax/tokenizers/` | Not an external integration slug |
| **RAG pipeline** | `intergrax/rag/` | Vector stores + document parsers use **catalog bridges**; orchestration stays in `rag/` |
| **Ephemeral Code Craft** | `intergrax/codecraft/` + `runtime/codecraft/` **Done** (ECC-0…ECC-6) | Composes `runtime/sandbox/` + `codecraft.*` tools; not a second sandbox |
| **Model & modality inference** | `intergrax/model_inference/` (planned), tools, optional integration hosts | Vision CV (YOLO, ONNX, …), classical ML, speech APIs — §7.1.9; **not** LLM slugs in Integration Library |

#### RAG stack (Tier-0)

**Canonical domain pair:** [`architecture/RAG.md`](RAG.md) ↔ [`plan/RAG.md`](../plan/RAG.md) — `RetrievalService`, `IngestPipeline`, `RagProfile`, M-RAG register, golden harness, integration boundaries.

Summary: one retrieval path (`rag.retrieve` + Nexus `rag.retrieve` (catalog)); vector stores and parsers via Integration Library catalog bridges; Knowledge vs user memory boundary in [`architecture/MEMORY.md`](MEMORY.md).

Do **not** add an `llm_provider` category or LLM slugs to the Integration Catalog backlog.

#### Ephemeral Code Craft (ECC)

**Canonical domain pair:** [`architecture/CODE_CRAFT.md`](CODE_CRAFT.md) ↔ [`plan/CODE_CRAFT.md`](../plan/CODE_CRAFT.md) — harness-orchestrated dynamic codegen loop; `codecraft.*` catalog tools; `CodeCraftProfile`; execution substrate `runtime/sandbox/`. ADR: [`adr/entries/2026-06-10/ADR-CODECRAFT-001.md`](../adr/entries/2026-06-10/ADR-CODECRAFT-001.md).

### 7.1.3 Integration Catalog (Initial Backlog)

Status legend: **Exists** = implemented elsewhere in Tier-0 today; **Catalog** = target `integrations/providers/<slug>/`; **Planned** = not started.

#### P0 — Foundation (lab + first production apps)

| Slug | Category | Status | Rationale |
|------|----------|--------|-----------|
| `sqlite` | relational_store | **Done** | `providers/sqlite/` — **single entry** `create_sqlite_integration()` (trace, events, checkpoints, HITL, task memory, experiments, idempotency, session, org) |
| `postgresql` | relational_store | Beta | Production relational store (`RelationalStore` via psycopg3); multi-tenant `tenant_schema` |
| `redis` | key_value_cache | **Done** | `providers/redis/` — **single entry** `create_redis_integration()` wraps KV, idempotency, rate limit, semaphore, rerank cache |
| `kafka` | message_bus | **Done** (+ adopcja) | `providers/kafka/` — runtime transport delegates here |
| `celery` | message_bus | **Done** | `providers/celery/` — `create_celery_integration()` + `create_celery_worker_app()` |
| `google_cse` | search_provider | **Done** | `providers/google_cse/` — `create_google_cse_integration()` |
| `bing` | search_provider | **Done** | `providers/bing/` — `create_bing_integration()` |
| `slack` | notification_channel + interaction_surface | **Done** (+ adopcja) | `providers/slack/` — runtime wiring delegates here |
| `teams` | notification_channel + interaction_surface | **Done** (+ adopcja) | `providers/teams/` — runtime wiring delegates here |
| `webhook` | notification_channel | **Done** (+ adopcja) | `providers/webhook/` — generic HTTP outbound |
| `log` | notification_channel | **Done** (+ adopcja) | `providers/log/` — process log; lab profile default |
| `lab_json` | interaction_surface | **Done** (+ adopcja) | `providers/lab_json/` — laboratory JSON intake |
| `rabbitmq` | message_bus | **Done** (+ adopcja) | `providers/rabbitmq/` — runtime transport delegates here |

#### P1 — Common enterprise stack

| Slug | Category | Status | Rationale |
|------|----------|--------|-----------|
| `mysql` | relational_store | Beta | Production relational store (`RelationalStore` via pymysql); optional `tenant_database` |
| `rabbitmq` | message_bus | **Done** (+ adopcja) | `providers/rabbitmq/` — runtime transport delegates here |
| `prometheus` | observability_backend | Beta | PromQL instant/range queries via HTTP API v1 |
| `jira` | issue_tracker | Beta | Task ingestion via REST v3 (`get_issue`, `add_comment`, `search_issues`) |
| `confluence` | wiki_knowledge | Beta | RAG / runbooks via REST (`get_page`, `search_pages`) |
| `ms365_graph` | collaboration_suite | Beta | Mail, calendar, directory via Microsoft Graph (client credentials) |
| `email_smtp` | notification_channel | Planned | HITL / reports without chat vendor lock-in |
| `s3` | object_storage | **Beta** | Artifacts, large uploads, shadow/sandbox exports |
| `filesystem` | object_storage | Partial | Local / dev; shadow workspace roots |
| `aws` | cloud_platform | Beta | IAM/STS auth; defaults for S3, SQS, DynamoDB, ElastiCache |
| `azure` | cloud_platform | Beta | Managed identity / service principal; defaults for Blob, Service Bus, Azure SQL |
| `gcp` | cloud_platform | Beta | ADC / service account; defaults for GCS, Pub/Sub, Cloud SQL |

#### P1.1 Cloud platforms — service mapping

Platform adapters are **facades**: one credential model + region/tenant config, then delegate to **infrastructure** category providers. They do **not** configure LLM — use `LLMAdapterRegistry` separately in Tier-3. Tier-3 may set `cloud_platform: aws` and inherit defaults for object storage and queues without wiring each slug separately.

| Platform slug | Auth model | Native services (category → slug) |
|---------------|------------|-----------------------------------|
| **`aws`** | Access key, IAM role, SSO profile, STS assume-role | object_storage → `s3`; message_bus → `sqs`; document_store → `dynamodb`; key_value_cache → `elasticache` (redis); secrets → platform helper |
| **`azure`** | Managed identity, service principal, connection string | object_storage → `azure_blob`; message_bus → `service_bus`; relational_store → `azure_sql`; secrets → Key Vault helper |
| **`gcp`** | Service account JSON, workload identity, ADC | object_storage → `gcs`; message_bus → `pubsub`; relational_store → `cloud_sql`; secrets → Secret Manager helper |

**Rule:** service-level slugs (`s3`, `azure_blob`, `gcs`, `sqs`) remain in the catalog for **explicit** or **multi-cloud** setups. When an app declares a single cloud, `IntegrationRegistry` MAY resolve category defaults from the platform facade (e.g. `object_storage` → S3 when `cloud_platform: aws`).

#### P2 — Extended / on-demand

| Slug | Category | Status | Rationale |
|------|----------|--------|-----------|
| `oracle` | relational_store | Beta | Enterprise clients on Oracle |
| `mssql` | relational_store | Beta | Microsoft SQL deployments |
| `azure_sql` | relational_store | Beta | Azure SQL via pyodbc |
| `cloud_sql` | relational_store | Beta | GCP Cloud SQL via pg8000 |
| `cassandra` | document_store | Beta | High-volume log / event retention (partition-scoped CQL) |
| `dynamodb` | document_store | Beta | AWS document/KV (via `aws` facade) |
| `memcached` | key_value_cache | Beta | Simple cache tier |
| `elasticache` | key_value_cache | Beta | Managed Redis on AWS (via `aws` facade) |
| `sqs` | message_bus | Beta | AWS-native queues (also via `aws` facade) |
| `service_bus` | message_bus | Beta | Azure-native queues (via `azure` facade) |
| `pubsub` | message_bus | Beta | GCP-native messaging (via `gcp` facade) |
| `azure_blob` | object_storage | Beta | Azure artifact storage (also via `azure` facade) |
| `gcs` | object_storage | Beta | GCP artifact storage (also via `gcp` facade) |
| **`elasticsearch`** | observability_backend | **Beta** | Log search / aggregations (`_search` + Lucene `query_string`); complements `prometheus` |
| **`databricks`** | relational_store | **Beta** | SQL Warehouse / Unity Catalog; lakehouse analytics via `RelationalStore` |
| **`mongodb`** | document_store | **Beta** | Flexible JSON documents; partition-scoped CRUD via PyMongo |
| **`pinecone`** | vector_store | **Beta** | Catalog bridge to `rag/`; `IntegrationProfile.vector_store` |
| **`qdrant`** | vector_store | **Beta** | Catalog bridge to `rag/`; self-hosted / cloud vectors |
| **`chroma`** | vector_store | **Beta** | Catalog bridge to `rag/`; embedded or HTTP Chroma |
| **`s3`** | object_storage | **Beta** | AWS S3 put/get/delete/presigned_url via catalog |
| `otel` | observability_backend | Beta | Unified traces/metrics export |
| `playwright` | browser_automation | Beta | Dynamic web research beyond HTTP fetch |
| `azure_devops` | issue_tracker | Beta | Microsoft ALM |
| `github` | issue_tracker | Beta | Dev-centric task sources |
| `linear` | issue_tracker | Beta | Linear issues API |
| `google_workspace` | collaboration_suite | Beta | Gmail / Calendar for Google tenants |
| `notion` / `sharepoint` | wiki_knowledge | Beta | Internal docs beyond Confluence |
| `email_smtp` | notification_channel | Beta | Outbound mail without chat vendors |
| `brave` / `serpapi` | search_provider | Beta | Alternative web research APIs |

P2/P3 batch implementations (2026-05-30) centralize shared logic in `intergrax/integrations/_shared/p2/` (`configs.py`, `clients.py`, `factories.py`); `providers/<slug>/` packages are thin registration shells except `azure_blob` and `s3` (full packages).

**Vector-store note:** `pinecone`, `qdrant`, and `chroma` implementations live in `intergrax/rag/vectorstore/`. Integration Library adds thin catalog bridges (`providers/<slug>/`) so Tier-3 can set `IntegrationProfile.vector_store`. RAG bootstrap (`create_default_vectorstore_manager()`) resolves stores via the catalog — see Phase M.6 P2 in the implementation plan.

New integrations require **human approval** when they introduce a new **category** (§5.2.4). New **providers** within an existing category follow the provider checklist in the implementation plan (Phase M).

### 7.1.4 IntegrationRegistry And Tier-3 Composition

Applications (Tier-3) **compose** integrations at startup — agents stay vendor-agnostic.

```text
applications/legal_application/settings.py
    → declares IntegrationProfile (enabled slugs + env)
    → IntegrationRegistry.resolve(IntegrationCategory.RELATIONAL_STORE) → PostgreSQL adapter
    → wire_nexus_observability(trace_store=…, event_store=…)
    → create_notification_adapter("slack")
    → create_interaction_adapter("teams")
```

`IntegrationProfile` uses catalog manifests, plugin classes, or validated slug strings plus typed categories (`IntegrationCategory`) in application code. YAML/env may use strings; Tier-2 agents must not import provider packages.

```python
from intergrax.integrations import (
    IntegrationCategory,
    IntegrationProfile,
    register_default_integrations,
)

register_default_integrations()
profile = IntegrationProfile(
    key_value_cache="redis",
    relational_store="sqlite",
    options={"sqlite": {"data_dir": "build/lab"}},
)
cache = profile.resolve(IntegrationCategory.KEY_VALUE_CACHE)
db = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
```

Declarative YAML (Tier-3 / deployment only):

```yaml
integrations:
  cloud_platform: aws              # optional — sets defaults for aws-native services
  relational_store: sqlite
  key_value_cache: redis
  message_bus: celery
  notification_channel: slack
  search_provider: google_cse
```

**Forbidden:** hard-coding `import redis` or `psycopg2` inside `agents/<name>/`. **Required:** `register_default_integrations()` in Tier-3 factory, then `resolve()`, `create_redis_integration()`, `create_sqlite_integration()`, or other factories from `integrations/providers/<slug>/`; agents use ToolRuntime for tools.

### 7.1.5 Provider Maintenance Model

| Activity | Owner | Gate |
|----------|-------|------|
| New category contract | Platform team | Architecture review + §5.2.4 if new universal |
| New provider in existing category | Integration team | Contract conformance tests + README |
| Security / credential rotation | Tier-3 application | Env + secret store; never in agent code |
| Deprecation | Platform team | Registry marks `deprecated`; 1 release warning |
| Live vendor tests | CI optional job | `pytest -m integration_live` — secrets in CI only |

Each provider README MUST document: auth model, required env vars, rate limits, idempotency behavior, and a **smoke command** runnable from lab.

### 7.1.5.1 Tier-0 Plugin Catalogs (Phase P-Ext)

All three Tier-0 catalogs (integrations, tools, skills) share one **plugin-native** registration model. Shipped first-party providers and third-party pip packages use the same APIs.

| Layer | Protocol | Register API | Entry point group |
|-------|----------|--------------|-------------------|
| Integration | `IntegrationPlugin` | `register_integration_plugin()` | `intergrax.integrations` |
| Tool | `ToolPlugin` | `register_tool_plugin()` | `intergrax.tools` |
| Skill | `SkillPlugin` | `register_skill_plugin()` | `intergrax.skills` |

**Tier-3 bootstrap (single call):**

```python
from intergrax.core.catalog_bootstrap import bootstrap_catalogs

bootstrap_catalogs(
    register_shipped=True,
    integration_preset="full",  # or "core" for lab cold-start (~12 integration slugs)
    tool_bundle_ids=None,     # or ("rag", "websearch") for lazy tool catalog
    skill_bundle_ids=None,    # or ("harness",) for lazy skill catalog
    discover_entry_points=True,
    integration_plugins=(MyIntegrationPlugin,),
)
```

`build_application_tool_wiring` / `build_application_skill_wiring` call `bootstrap_catalogs(register_shipped=True)` idempotently.

**Rules:**

- No central enum of all integration slugs — identity is `IntegrationManifest.slug` per provider.
- External packages MUST NOT edit Intergrax core; register via entry points or explicit plugin classes at startup.
- **Runtime Nexus plugins** (`RuntimePlugin`, `plugin_bootstrap.py`) are a **separate** extensibility plane from Tier-0 catalog plugins.
- Tool execution observability (trace, scope policy, error mapping) lives in `RuntimeToolInvoker` — not in plugin registration.

Author guide: [`guides/EXTENSION_AUTHOR_GUIDE.md`](guides/EXTENSION_AUTHOR_GUIDE.md). Implementation tracker: [`plan/PLATFORM_FOUNDATION.md) Phase P-Ext + Appendix I.

### 7.1.6 Tool Library — Canonical Catalog

Tier-0 agent-facing capabilities MUST live in a **single, discoverable Tool Library** under `intergrax/tools/`, mirroring the Integration Library pattern (§7.1.1).

**Problem this solves:** Integrations answer *how to talk to a backend* (Jira REST, PostgreSQL, Bing API). LLM agents and MCP clients need *what to call* — semantically named operations with JSON schemas, descriptions, risk metadata, and trace-enforced execution. Agents MUST NOT call integration contracts directly.

**Four-layer capability model (canonical):**

```text
Tier-2  Agent                 →  contract, UAEP steps, skill_ids[], domain governance
Tier-0  Skill Library (R)     →  composable packs: tool_ids + prompts + policy fragment
Tier-0  Tool Library          →  atomic LLM/MCP operations (JSON schema)
Tier-0  Integration Library   →  vendor/backend Protocols (swappable at deploy)
```

| Layer | Package | Consumer | Example |
|-------|---------|----------|---------|
| **Integration** | `intergrax/integrations/` | Tool handlers, Tier-3 wiring, RAG bootstrap | `IssueTracker.search_issues(jql)` |
| **Tool** | `intergrax/tools/providers/<domain>/` | LLM tool-calling, MCP, UAEP `ToolRequest` | `jira.search_tasks(project, status, assignee)` |
| **Skill** | `intergrax/skills/providers/<domain>/` | Agent composition, importers | `legal.contract_review`, `research.literature_scan` |
| **Agent** | `agents/<name>/` | Nexus routing | `skill_ids=["legal.contract_review"]` or `["research.literature_scan"]` |

**Target layout:**

```text
intergrax/tools/
├── core/                   # ToolContract, execution models, ToolProvider protocol (exists)
├── registry/               # ToolCatalog, register_default_tools(), ToolProfile (Phase O — Done)
├── exporters/              # OpenAI / Anthropic / MCP schema export (Phase O)
├── _shared/                # schema helpers, LLM description lint, JQL/query builders
└── providers/
    └── <domain>/           # e.g. jira/, rag/, websearch/, sandbox/
        ├── contracts.py    # Input/Output Pydantic models per tool
        ├── handlers.py     # ToolHandler implementations
        ├── bundle.py       # register_*_tools(registry, ctx)
        ├── USAGE.md
        └── tests/
```

**Rules:**

- One **domain folder** per tool family (`jira/`, `websearch/`, `rag/`, …) — not one folder per vendor SDK.
- Tool handlers **compose** integration contracts — they MUST NOT reimplement vendor HTTP/SDK calls when a catalog integration exists.
- Vendor SDKs remain in `integrations/providers/<slug>/opens.py` only (§7.1.1).
- Agents and Nexus MUST NOT import `integrations/providers/*` for side effects; they invoke tools via **`ToolRuntime`** / `ToolRequest` (§22, §42.12).
- Tool handlers receive dependencies through **`ToolWiringContext`** (Tier-3 composition): resolved integrations, RAG managers, websearch executors — injected at startup, not looked up ad hoc inside handlers.
- **`ToolProvider.register_tools(registry, ctx)`** is the production registration contract (explicit wiring — no magic discovery).

**Separation of concerns:**

| Layer | Owns |
|-------|------|
| `tools/core/contracts.py` | What every tool MUST expose to runtime (schema, risk, side_effects) |
| `tools/providers/<domain>/` | LLM-facing semantics + business logic above integrations |
| `tools/registry/` | Catalog, `ToolProfile`, default registration |
| Tier-3 `applications/<name>/` | Which tools are enabled + integration instances passed into `ToolWiringContext` |
| Tier-2 `agents/<name>/` | `allowed_tools` allow-list — capability policy, not vendor wiring |
| Tier-1 `runtime/nexus/tools/` | Enforcement: `RuntimeToolInvoker`, `ToolAccessPolicy`, trace, idempotency |

**Tool vs integration — decision rule:**

| Question | Answer → layer |
|----------|----------------|
| Is the consumer an LLM choosing a function call? | **Tool** |
| Is it swapping Postgres for MySQL at deploy time? | **Integration** |
| Does it need `description` + JSON Schema for model tool selection? | **Tool** |
| Is it a stable backend Protocol with no LLM metadata? | **Integration** |

**Out of scope for Tool Library:**

| Concern | Canonical module | Notes |
|---------|------------------|-------|
| **LLM providers** | `intergrax/llm_adapters/` | Not tools — separate registry (§5.2.2) |
| **Agent business logic** | `agents/<name>/` | Domain steps; may *call* tools, not define platform catalog entries |
| **Orchestration / planning** | Tier-1 Nexus | Selects tools; does not implement tool handlers |
| **Cursor-style skill files** | `intergrax/skills/importers/` | Import via **`SkillImporter`** → validated `SkillManifest`; MUST NOT register as `ToolContract` |

**Dual export (agent + MCP):**

Every catalog tool MUST be exportable as:

1. OpenAI-compatible function schema (for `ToolsAgent` / native tool-calling LLMs),
2. MCP tool definition (for `applications/<app>/mcp/server.py`),
3. `ToolRequest.tool_name` value (for UAEP / `RuntimeToolGateway`).

Single source of truth: `ToolContract` in the catalog — not parallel schema definitions per surface.

**Catalog reference:** [`architecture/TOOLS.md`](architecture/TOOLS.md) — 11 first-party tools **Done** (Phase O.4, 2026-05-30) · Implementation: Phase O in [`plan/PLATFORM_FOUNDATION.md).

### 7.1.7 Unified Tool Model — Everything Is a Tool

**Target architecture (Phase O.5+):** All agent-invokable platform capabilities — including today’s pipeline flags `use_rag`, `use_websearch`, and registered function tools — converge on **one mechanism**: named tools in `ToolRegistry`, invoked through `ToolRuntime`.

**Current state (2026-06-19, PF-MAINT-LEG-02):** `ToolInvocationPlan` is **`tool_ids`-first** — legacy `use_rag` / `use_websearch` fields removed from the runtime bridge. Gateway payloads may still map deprecated boolean keys to catalog ids silently; `use_tools` remains for the bounded tool planner loop. Context steps (`run_rag_context`, `run_websearch_context`) dispatch when the corresponding catalog tool id is present in `tool_ids`.

**Target state:**

```text
Agent / planner
    → planned tool_ids: ["rag.retrieve", "websearch.query", "jira.search_tasks"]
    → ToolRuntime.invoke_request(ToolRequest)  # per tool, or batched plan
    → RuntimeToolInvoker → ToolHandler
    → integrations + domain logic
```

| Legacy flag / step | Target catalog tool_id | Underlying integration / module |
|--------------------|------------------------|----------------------------------|
| `use_rag` / `rag.retrieve` (catalog) | `rag.retrieve` | `intergrax/rag/` + `IntegrationProfile.vector_store` |
| `use_websearch` / `websearch.query` (catalog) | `websearch.query` | `SearchProvider` via `IntegrationProfile.search_provider` |
| `use_tools` / `run_bounded_tool_loop` / `ctx.invoke_tool` | *(explicit tool_ids)* | `ToolRegistry` entries |
| Sandbox execution | `sandbox.exec` | `intergrax/runtime/sandbox/` (already a tool_id) |
| Ephemeral Code Craft | `codecraft.*` **Done** (ECC-0…ECC-6) | `intergrax/codecraft/` + `runtime/codecraft/` — see [`CODE_CRAFT.md`](CODE_CRAFT.md) |

**Migration rules:**

1. **No new boolean capability flags** — new platform capabilities MUST ship as catalog tools with `ToolContract`.
2. **`ToolInvocationPlan`** uses `tool_ids: Sequence[str]` as canonical — **Done** at runtime bridge (PF-MAINT-LEG-02); gateway payload booleans map to ids only.
3. **`rag.retrieve` (catalog) / `websearch.query` (catalog)** become thin **compatibility shims** that delegate to `rag.retrieve` / `websearch.query` handlers until all callers migrate.
4. **`LegalToolPlan` / engine plan models** replace `use_rag` / `use_websearch` with `tools: list[str]` (or structured `PlannedToolCall`).
5. **Context injection tools:** `rag.retrieve` and `websearch.query` MAY declare `injects_context: true` so Nexus knows to merge results into LLM prompt context (replaces implicit step behavior) — see §22.1.

**Why unify:**

- One policy surface (`ToolAccessPolicy`, `allowed_tools`, trace, idempotency, risk).
- One schema export path for LLM tool selection and MCP.
- Agents reason about **tools**, not parallel abstractions (flags vs registry).
- Tier-3 enables tools via `ToolProfile` — same ergonomics as `IntegrationProfile`.

**Forbidden after Phase O.5:**

- Adding new `use_*` booleans to plan models for platform capabilities.
- Agent code branching on `use_rag` instead of invoking `rag.retrieve` or listing it in `allowed_tools`.
- Direct `rag.retrieve` (catalog) / `websearch.query` (catalog) invocation from Tier-2 agents (must use `ToolRequest`).

### 7.1.8 Skill Library — Composable Capability Packs

**Status:** Architecture **defined**; implementation **MVP Done** (Phase R, 2026-06-01).  
**Catalog:** [`architecture/SKILLS.md`](architecture/SKILLS.md) · **Harness AI terms:** §5.3 (this document).

**First-party catalog (2026-06-08):** **149** skills · **41** bundles — full table in [`architecture/SKILLS.md`](architecture/SKILLS.md#first-party-catalog-149-skills--41-bundles). Product-facing examples:

| skill_id | Bundle | Typical agent |
|----------|--------|---------------|
| `legal.contract_review` | `legal` | `LegalAgent` |
| `research.literature_scan` | `research` | `ResearchAgent` |
| `knowledge.openai_strict` | `knowledge` | OpenAI-hosted retrieval hosts |

**Runtime events:** `SKILL_RESOLVED` / `SKILL_IMPORT_FAILED` via `runtime/events/context_skill_recording.py`; registration and import service call `RuntimeEventBus.record()`.

#### Problem

- **Integrations** answer *how to connect* (Postgres, Bing, Jira).
- **Tools** answer *what single operation the LLM may invoke* (`rag.retrieve`, `websearch.query`).
- **Agents** today often duplicate the same bundle of tool allow-lists, prompt instructions, and policy snippets — that bundle is a **skill** in harness terminology, not a tool.

#### What a skill is

A **skill** is a **versioned, declarative capability pack** that groups:

| Field | Purpose |
|-------|---------|
| `skill_id`, `version` | Stable reference (`legal.contract_review@1.0.0`, `research.literature_scan@1.0.0`) |
| `description` | Human + planner readable purpose |
| `tool_ids` | Subset of catalog tools required for the goal (e.g. `rag.retrieve`, `websearch.query`) |
| `prompt_instruction_ids` | References into Prompt Registry (not raw prompt blobs in runtime) |
| `policy_fragment_id` | Optional link to tool/memory/HITL fragment |
| `risk_tier`, `tags` | Governance and discovery |

Skills are **not** invoked by the LLM as functions. The runtime **resolves** skills at **agent registration** (`AgentRegistry.register`) into:

- merged `allowed_tools` (skill `tool_ids` ∪ `extra_tools`; then intersected with `ToolProfile` / `ToolAccessPolicy` at runtime),
- `prompt_instruction_ids` and `policy_fragment_id` in `ResolvedSkillPack` (trace + capability graph; **automatic** `ContextManager` / `RuntimePolicyBundle` merge — **SK-BRIDGE.*** in [`plan/SKILLS.md`](plan/SKILLS.md), not yet shipped).

#### What a skill is not

| Anti-pattern | Why forbidden |
|--------------|---------------|
| Skill as `ToolContract` | Breaks atomic tool schema, MCP export, per-tool risk/retry/trace |
| Skill as full Tier-2 agent | Too coarse; prevents composition of multiple skills per agent |
| Unvalidated markdown dropped into prompt | No governance, no tool allow-list, not reproducible |
| Skill calling integrations directly | Must use tools (same rule as agents) |

#### Skill vs tool — decision rule

| Question | Answer → layer |
|----------|----------------|
| Does the LLM choose it in a tool-call turn? | **Tool** |
| Is it swapping Redis for Memcached at deploy? | **Integration** |
| Is it a reusable pack of tools + instructions for a business goal? | **Skill** |
| Is it domain orchestration with UAEP steps and contract? | **Agent** |

#### External skill compatibility

External ecosystems (e.g. Cursor `SKILL.md`, internal markdown packs) MAY be attached **only** through:

```text
External file  →  SkillImporter.validate()  →  SkillManifest  →  SkillRegistry
```

Rejected imports MUST emit `SKILL_IMPORT_FAILED` (trace) and MUST NOT partially attach tool access.

#### Target layout

```text
intergrax/skills/
├── core/           # SkillManifest, SkillProvider protocol
├── registry/       # SkillCatalog, SkillProfile, bootstrap
├── resolver.py     # SkillResolver → ResolvedSkillPack
├── importers/      # cursor_skill_md, …
└── providers/
    └── <domain>/   # manifests.py + plugin.py + USAGE.md
```

#### Agent composition

```text
AgentContract:
    skills: list[SkillManifest]    # catalog manifest refs; resolved at register
    extra_tools: list[ToolContract]  # optional tools beyond skill union
    allowed_tools: list[str]       # OUTPUT — set by AgentRegistry after resolution
```

Resolution order (canonical — see [`architecture/SKILLS.md`](architecture/SKILLS.md)):

1. Validate `contract.skills` against `SkillRegistry`; expand `requires_skills`.
2. Union skill `tool_ids` with `extra_tools[].tool_id` → replace `allowed_tools`.
3. At run time, intersect with `ToolProfile` and `ToolAccessPolicy`.
4. **Planned (SK-BRIDGE.*):** attach `prompt_instruction_ids` / policy fragments to context and policy bundle automatically.

#### Relationship to Tier-2 pipelines

UAEP `get_steps` / domain pipelines remain **agent-local orchestration**. A skill MAY include an optional **`step_template_id`** (future) for shared step sequences — it does NOT execute steps itself; `AgentEngine` does.

### 7.1.9 Model & Modality Plane (Vision, Audio, Classical ML)

**Status:** Architecture **defined** (2026-06-02); harness registry + modality tools + lab `ModalityProfile` wiring **Done** (Phase W-ML); remote Triton/HF live serving **incremental**.  
**Catalog index:** [`architecture/MODALITY.md`](architecture/MODALITY.md) · **Harness alignment:** §5.3 · **ADR:** extends §44.10 (LLM stays out of Integration Library).

#### Strategic intent

A scalable Harness AI MUST support **multimodal cognition** and **deterministic model inference** without becoming a monolithic MLOps platform. Intergrax extends the existing four-layer stack (Integration → Tool → Skill → Agent) with **three modality planes**, each with its own Tier-0 registry and the same governance hooks (policy, trace, budgets) as LLM and RAG.

#### ADR summary (modality plane)

| Decision | Verdict |
|----------|---------|
| LLM providers (incl. native multimodal APIs) in Integration Catalog | **Rejected** — §7.1.2, §44.10 |
| Skills wrapping entire CV/TTS pipelines as one fake tool | **Rejected** — §7.1.8 anti-patterns |
| Dedicated **Plane C** registry for YOLO/ONNX/sklearn + atomic tools | **Adopted** |
| Media ingest (Whisper, OCR, parsers) as **Plane B** via existing RAG/document_parser | **Adopted** (partially implemented) |
| `speech_provider` integration category for SaaS TTS/STT (ElevenLabs, …) | **Adopted** (planned) |
| Tier-2 agents importing `torch` / `ultralytics` directly | **Rejected** |

#### Three modality planes

```text
Plane A — Generative cognition     intergrax/llm_adapters/     (dialog, native vision/audio LLM APIs)
Plane B — Media → text (ingest)    document_parser + rag/      (indexing, transcripts, OCR text)
Plane C — Dedicated inference      model_inference/ (planned)  (YOLO, ONNX, sklearn, remote serving)
```

| Plane | Use when | Examples |
|-------|----------|----------|
| **A** | Reasoning, planning, conversational multimodal | Gemini vision, GPT-4o, Claude image input |
| **B** | Files/URLs → searchable text or embeddings | `whisper`, `docling`, `ImageSmartLoader`, `HFEmbeddingProvider` |
| **C** | Deterministic CV/ML, regulated boxes, low-latency detection | YOLO, ONNX Runtime, OpenVINO, Triton, `ml.predict` |

**Routing rule:** If the task requires **reproducible geometry** (bounding boxes, masks, calibrated scores) or **fixed latency SLA**, prefer **Plane C**. If the task requires **semantic description in dialog**, prefer **Plane A**. If the task is **knowledge indexing**, use **Plane B** only.

#### Plane A — Generative multimodal (LLM adapters)

- **Canonical module:** `intergrax/llm_adapters/` — unchanged separation from integrations.
- **Message model:** `intergrax/llm/messages.py` — `AttachmentRef` types (`image`, `audio`, `video`, …).
- **Target adapter contract extensions:**
  - `supports_vision()`, `supports_audio_input()`, `supports_audio_output()` (capability flags),
  - mapping `AttachmentRef` → vendor content parts in `generate_messages` / streaming paths.
- **Policy:** multimodal attachments subject to `ContextBudgetPolicy`, MIME allowlists, and tenant media quotas (extends V-COST).

See [architecture/LLM_ADAPTERS.md](architecture/LLM_ADAPTERS.md) — **Multimodal capabilities** section.

#### Plane B — Media ingest (existing RAG stack)

Plane B is **not** a separate top-level package; it composes:

| Component | Location | Notes |
|-----------|----------|-------|
| Document parsers | `integrations/providers/document_parser/<slug>/` | `whisper`, `yt_dlp`, `docling`, … |
| Smart loaders | `intergrax/multimedia/`, `rag/document_loaders/` | Image OCR/caption, audio pipelines |
| Embeddings | `intergrax/rag/embedding/` | `hf`, `openai`, `ollama` providers |
| Sparse encoders | `rag/vectorstore/sparse/` | `bm25_hash`, optional `splade` |

Ingest MUST remain invocable through **tools** (`rag.ingest_document`, retrieval stack) — not ad-hoc agent SDK calls.

#### Plane C — Vision inference engine (extensible)

**Target module:** `intergrax/model_inference/` (Tier-0, Phase W-ML).

Designed for **market-standard production CV** and portable classical ML — same extensibility pattern as `llm_adapters/` and `integrations/`:

| Subsystem | Role |
|-----------|------|
| `model_inference/contracts/` | `VisionInferenceAdapter`, `ModelInferenceAdapter`, DTOs (`DetectionResult`, …) |
| `model_inference/registry/` | `VisionInferenceRegistry`, `ModelInferenceRegistry`, profiles |
| `model_inference/providers/<slug>/` | Backend-specific bridges (thin; no business logic) |
| `model_inference/execution/` | `ModalityInferenceExecutor`, thread-pool offload for heavy vision slugs |
| `model_inference/workers/` | Optional distributed workers via Tier-3 `message_bus` (Celery/Kafka) |

**Vision backend families (non-exhaustive, pluggable by slug):**

| Slug family (planned) | Technology | Typical deployment |
|-----------------------|------------|--------------------|
| `yolo_ultralytics` | Ultralytics YOLO (v8+) | Local GPU worker or sidecar |
| `onnxruntime` | ONNX Runtime | Edge CPU/GPU, cross-platform |
| `openvino` | Intel OpenVINO | On-prem Intel |
| `tensorrt` | NVIDIA TensorRT | Low-latency GPU |
| `torchscript` | TorchScript `.pt` | Lab / air-gapped |
| `triton_grpc` / `torchserve` | Remote model server | K8s model mesh |
| `huggingface_inference` | HF Inference Endpoints | Managed API |
| `roboflow` / `sagemaker` / `vertex_prediction` | Cloud CV/ML hosts | Enterprise VPC |

**Contract requirements (all Plane C adapters):**

- Typed **input schema** (URI, bytes ref, MIME, optional ROI) and **output schema** (detections, masks, class scores).
- **`model_id` + `version`** on every invocation; trace exports slug, latency, device, batch size — not raw media by default.
- **Idempotency** where inference is side-effect free; **risk_tier** for tools exposing CV output to downstream LLM steps.
- **Resource policy:** max batch, max resolution, GPU memory class — enforced before invoke (PolicyEngine + `ModalityProfile`).

**YOLO / object-detection example flow:**

```text
Agent step  →  ToolRuntime.invoke("vision.detect")
           →  VisionInferenceRegistry.resolve(profile)
           →  yolo_ultralytics adapter  →  DetectionResult JSON
           →  trace + modality_metrics  →  optional LLM step (Plane A) for narrative only
```

#### Plane C — Classical ML (non-vision)

- **`ModelArtifact`** metadata: id, semver, input/output JSON schema, owner, license, risk_tier.
- **Tools:** `ml.predict`, `ml.batch_predict` (planned); optional `ml.explain` (high risk).
- **Artifacts** stored in `object_storage`; loading in worker or remote host — not in Nexus request thread for heavy models.

#### Speech and audio output (SaaS)

- **Category (planned):** `speech_provider` in Integration Library.
- **Examples:** `elevenlabs`, `azure_speech`, `deepgram`, `openai_tts` (slug placeholders until W-ML.2).
- **Tools:** `speech.synthesize`, `speech.transcribe` — output URIs via `object_storage` contract.
- **Ingest STT** may continue via `document_parser/whisper` (Plane B); SaaS STT is optional Plane C via integration.

#### Hugging Face — platform roles

| Role | Module | Rule |
|------|--------|------|
| Embeddings | `rag/embedding/providers/hf_embedding_provider.py` | RAG only |
| Hub artifacts | Governance (pin revision, license) | V-SEC / artifact policy |
| Hosted inference | `ml_inference_host` integration | Remote Plane C |
| Local transformers | Worker pool only | Never Nexus hot path |

Env: `INTERGRAX_DEFAULT_HF_EMBED_MODEL` (existing). Do not conflate Hub download with runtime agent imports.

#### Integration categories (planned — require §5.2.4 approval)

| Category | Contract (planned) | Distinct from |
|----------|-------------------|---------------|
| `speech_provider` | TTS/STT SaaS | `document_parser` (file ingest) |
| `vision_serving` | Remote CV gRPC/REST (Triton, TorchServe) | Plane A LLM vision |
| `ml_inference_host` | Managed endpoints (HF, SageMaker, Azure ML, Vertex) | `cloud_platform` facade |

Add rows to §7.1.2 table when each category is approved — do not overload `observability_backend` or `document_parser`.

#### Tool surface (planned)

| tool_id | Plane | Atomic operation |
|---------|-------|------------------|
| `vision.detect` | C | Object detection / boxes |
| `vision.segment` | C | Instance/semantic masks |
| `vision.ocr_regions` | C | Layout OCR blocks |
| `speech.synthesize` | C | TTS → storage URI |
| `speech.transcribe` | B/C | Audio → text |
| `ml.predict` | C | Structured inference |

Each tool is one LLM-callable function with MCP export, risk class, and retry policy — same as §7.1.6.

#### ModalityProfile (Tier-3 / agent assembly)

Optional profile composes with `LLMProfile` (ideal §17, implementation plan Phase W-ML):

| Field | Purpose |
|-------|---------|
| `allowed_planes` | Subset of `generative`, `ingest`, `vision_inference`, `classical_ml`, `speech` |
| `vision_model_ids` | Allowlist of registered CV models |
| `max_media_bytes` | Attachment and ingest size cap |
| `require_deterministic_cv` | Force Plane C for detection tasks in regulated domains |

Resolution merges with `RuntimePolicyBundle` and `ToolAccessPolicy` (intersection, not bypass).

#### Observability

| Plane | Metrics hook |
|-------|----------------|
| A | `llm_metrics` (existing) |
| B | `rag_metrics`, parser trace (existing) |
| C | `modality_metrics` on `tool_invocation_end` (per tool), aggregated on `TASK_COMPLETED` runtime event + `export_run_metrics` (`inference_ms`, `media_bytes`, `tts_characters`, …) |

Extend V-COST envelopes: `inference_ms`, `media_bytes`, `tts_characters`.

#### Explicit non-goals (harness boundary)

- Online training, hyperparameter search, feature stores as platform products.
- Registering LLM slugs as integrations.
- CV pipelines as monolithic skills without atomic tools.
- Duplicating `llm_adapters` for each ONNX file — use Plane C registry instead.

#### Implementation tracker

Phase **W-ML** in [`plan/PLATFORM_FOUNDATION.md). Existing Plane B assets: M.6 (`whisper`, `yt_dlp`), image/audio smart loaders, HF embeddings.

---

## 7.2 Tier-1: Nexus Runtime (legacy: Layer 2 — Nexus Runtime)

Nexus is the central runtime and orchestration layer — the **Agent OS** (Tier-1).

Nexus owns:

- global task understanding
- routing
- planning
- task decomposition
- agent selection
- execution graph
- state transitions
- lifecycle management
- retry strategy
- validation strategy
- context distribution
- tool access policy
- adapter access policy
- human approval flow
- observability
- final response construction

Nexus MUST remain domain-agnostic.

Nexus MUST NOT become a Legal Agent, Vendor Agent or Problem Radar Agent.

---

## 7.3 Tier-2: Agents (legacy: Layer 3 — Agents)

Agents are bounded capability modules (Tier-2).

Agents own domain-specific execution.

Examples:

- ProblemRadarAgent searches and clusters user pains from sources such as Reddit, Hacker News and other public sources.
- VendorDiscoveryAgent finds, classifies and evaluates companies for a client need.
- LegalAgent analyzes legal documents according to defined legal-review rules.
- OnboardingAgent supports new employees through a structured onboarding process.
- UXAgent, PMAgent, TesterAgent, MarketerAgent — future Tier-2 role agents sharing the same contracts.

Agents MUST implement stable contracts.

Agents MAY have their own local reasoning loop (bounded by contract limits).

Agents MUST NOT own global orchestration.

Agents run **inside** Tier-1 Nexus; they consume Tier-0 **through** Nexus policy.

---

## 7.4 Tier-3: Applications And Repository Layout

Intergrax separates **Tier-2 agents** from **Tier-3 applications** at the repository level.

This split is NOT a deployment detail.

This split is a core architectural boundary.

An **application** is a ready-made environment: Nexus (Tier-1) + configured agent roster (Tier-2) + rules + integrations — analogous to a specialized product like “Cursor for legal work” or “Cursor for agency PM”.

### 7.4.1 Four Repository Roots

```text
intergrax/              Tier-0 + Tier-1 — platform + Nexus Agent OS
agents/                 Tier-2 — specialized agents
applications/           Tier-3 — ready-made configured environments
```

| Root | Tier | Role |
|------|------|------|
| `intergrax/` (platform packages) | **0** | LLM, RAG, tools, memory, queues, logging, adapters |
| `intergrax/runtime/`, `intergrax/contracts/` | **1** | Nexus, registry, task lifecycle, orchestration |
| `intergrax/agents/` | **1** (contract only) | `Agent` ABC, `AgentEngine` — not concrete agents |
| `agents/<name>/` | **2** | LegalAgent, ResearchAgent, UXAgent, … |
| `applications/<name>/` | **3** | legal_application, research_application, … |

**Important distinction:**

- `intergrax/agents/` is the **framework contract** (`Agent`, `AgentEngine`) — it MUST NOT contain concrete agent implementations.
- `agents/` at repository root is where **concrete agents** live (`LegalAgent`, `EchoAgent`, future `ResearchAgent`).

### 7.4.2 What Belongs In Tier-2 (`agents/<name>/`)

An agent capability module MUST contain:

- agent class implementing `Agent` / `AgentContract`
- domain models and business rules for that capability
- pipeline, steps, prompts
- agent-local governance and validation
- agent-local tracing helpers
- agent unit and integration tests for capability behavior
- notebooks for capability experiments

An agent capability module MUST NOT contain:

- FastAPI host or `uvicorn` entrypoint
- HTTP route definitions (`/v1/...`)
- environment-specific settings (`.env`, SKU profiles, API keys wiring)
- product deployment manifests (Dockerfile, k8s) — deployment belongs in **Tier-3** `applications/<name>/docker/`, not under `agents/`
- global orchestration or cross-agent routing

Agents MUST be runnable through Nexus (`AgentEngine`, `NexusLoop`) **without** starting an HTTP server.

Tier-3 applications MUST own Docker/k8s manifests for **their** host — not Tier-2 agents.

### 7.4.3 What Belongs In Tier-3 (`applications/<name>/`)

An application is a **ready-made environment** (Tier-3) that composes Nexus + agents + configuration.

An application MUST contain:

- host package (`main.py`, `factory.py`, `settings.py`, `wiring.py`)
- serving layer (FastAPI routers, request/response mapping)
- **`.env.example`** — documented, application-prefixed variables (committed); **`.env`** — local overrides (gitignored)
- environment-level configuration (env vars, product profiles, tenant defaults)
- registration of agents into `AgentRegistry` (explicit `registry.register()` — no auto-discovery)
- `IntegrationProfile` wiring (or equivalent typed composition in `integration_wiring.py`)
- orchestration config: agent roles, default capabilities, interaction topology
- **`README.md`** — three-command quickstart: pytest, `uvicorn`, `docker/build-docker.sh` (or `.bat`)
- **`docker/`** — Dockerfile, `.dockerignore`, build scripts, optional compose (Phase N scaffold; see implementation plan)
- application integration tests under `<app>_tests/` (avoids clashing with repo `tests/` package)

An application SHOULD contain (when scaffolded via `new-application`):

- **`manifest.py`** — `ApplicationManifest`: declarative roster, integration profile hints, feature flags (Phase N)
- optional `integrations.yaml` — ops-friendly profile overlay (env still authoritative in code)

An application MUST NOT contain:

- pipeline steps or domain logic that belongs to a single agent
- duplicated agent implementation (import from `agents/<name>/` instead)
- Nexus runtime internals

Applications are **thin composition layers**.

They wire agents to adapters, config, and transport — they do not implement agent reasoning.

### 7.4.4 Dependency Direction

```text
applications/<app>/  →  agents/<agent>/  →  intergrax/
```

Rules:

- `intergrax/` MUST NOT import code from `agents/` or `applications/`.
- Harness reference manifests for capability-graph seeding live in `intergrax/applications/reference/` (Tier-0 catalog, no `applications/` imports).
- `agents/` MUST NOT import code from `applications/`.
- `applications/` MAY import from `agents/` and `intergrax/`.
- Multiple applications MAY register the same agent with different config.

This preserves the framework as a stable, product-agnostic core.

### 7.4.5 Reference Layout: Legal

Canonical split (reference implementation):

```text
agents/legal/
    legal_agent.py          # LegalAgent(Agent)
    domain/                 # legal domain models
    pipeline/               # LegalPipeline, execution loop
    steps/                  # extract, normalize, recommend, ...
    governance/             # agent-local policy ports
    runtime/                # agent-local tool bridge
    tracing/                # agent-local diagnostics
    tests/                  # capability tests

applications/legal_application/
    host/                   # FastAPI app, factory, settings
    serving/                # mount_legal_agent_routes, chat API
    legal_tests/            # host/serving integration tests
```

During migration, `applications/legal_agent/` was removed. Use `agents/legal/` and `applications/legal_application/` directly.

New code MUST NOT import `legal_agent` as a package path.

### 7.4.6 Creating A New Capability

Recommended workflow:

```text
1. python -m intergrax.scaffold new-agent <name> --capability <cap>.<action>
   → creates agents/<name>/

2. Implement domain logic in agents/<name>/
   → get_contract(), UAEP steps, pipeline

3. (Recommended for deployable POC) python -m intergrax.scaffold new-application <name>_application
      --profile lab|product --agents <name>
   → creates applications/<name>_application/ with host, .env.example, docker/, manifest (Phase N)

   OR use universal lab_application (edit wiring.py) for quick HTTP experiments

4. Register agent in AgentRegistry (smoke test, notebook, or application wiring/manifest)

5. Run: pytest → uvicorn → docker build → observe trace → evaluate
```

Not every agent requires a dedicated application.

Notebook-only or test-only experiments MAY use `agents/<name>/` without creating `applications/`.

Create a **dedicated** Tier-3 application when you need a stable host, **isolated env/Docker**, or a production push path for a specific agent concept.

Use **`lab_application`** when experimenting with many agents in one shared debug surface.

### 7.4.7 Anti-Pattern: Agent-Application Monolith

Do NOT place agent implementation, pipeline, host, serving, and env config in a single package under `applications/`.

This was the legacy `applications/legal_agent/` layout.

It couples capability code to deployment, makes reuse across environments harder, and violates Tier-2 / Tier-3 boundaries.

If agent logic and host live together, split them before adding a second agent or second deployment target.

### 7.4.8 Tier-3 Application Environment (Self-Contained Operational Package)

A Tier-3 application is an **isolated, configured execution environment** — not a runtime sandbox (see §7.4.9).

Each application under `applications/<app>/` MUST be operable as a **self-contained package**:

| Concern | Owned by application | Notes |
|---------|---------------------|-------|
| Configuration | `.env.example` (committed), `.env` (local, gitignored) | Variables use an **application prefix** (e.g. `LAB_`, `LEGAL_`, `MYAPP_`). Root `.env.example` documents Tier-0/platform only. |
| Python package | `host/`, `serving/`, `__init__.py` | Import path via `pyproject.toml` `pythonpath` (`applications`). No separate `pyproject.toml` per app required. |
| Agent roster | `host/wiring.py` and/or `manifest.py` | Explicit `AgentRegistry.register()`; contract id overrides in factory when needed. |
| Integrations | `integration_wiring.py` + `IntegrationProfile` | Selects sqlite/redis/slack/… — agents stay vendor-agnostic. |
| Run locally | `README.md` + `host/main.py` | `load_dotenv()` in `main.py` loads **application directory** `.env` when present. |
| Deploy | `docker/Dockerfile` + `build-docker.sh` / `build-docker.bat` | Image build from monorepo root (scripts wrap BuildKit or classic `docker build`); `CMD` → `uvicorn <package>.host.main:app`. |
| Verify | `<app>_tests/` | Host smoke + optional HTTP contract tests. |

**Canonical layout (target — Phase N scaffold):**

```text
applications/<app>/
    __init__.py
    manifest.py              # ApplicationManifest — roster, profile, features (Phase N)
    README.md
    BUILD_AND_DEPLOY.md   # local run, tests, Docker build/push runbook (scaffold)
    .env.example
    .env                     # gitignored
    host/
        main.py              # ASGI app, load_dotenv
        factory.py           # create_*_application()
        settings.py          # from_env(), prefixed env keys
        wiring.py            # build_*_registry() from manifest
        integration_wiring.py
    serving/
        fastapi_router.py
    mcp/
        server.py              # FastMCP tools; mounted on FastAPI via fastapi_mcp.couple_fastapi_with_mcp
    docker/
        Dockerfile
        .dockerignore
        build-docker.sh      # image build from repo root (Linux/macOS/Git Bash)
        build-docker.bat     # same on Windows (cmd)
        docker-compose.yml   # optional — ollama, redis, volumes
    <app>_tests/
        host/test_*_smoke.py
```

**Reference implementations today:**

- `applications/lab_application/` — universal experimentation (`IntegrationProfile.lab()`, debug API)
- `applications/legal_application/` — product profile (`fastapi_core`, auth, legal routes)
- `applications/research_application/` — product-style research host

**Goal:** Time from agent POC to **docker-pushable** lab host should match agent scaffold speed (implementation plan Phase N).

### 7.4.9 Terminology: Application Environment vs Runtime Sandbox

These terms MUST NOT be conflated in documentation, scaffold CLI, or env variable names.

| Term | Tier | Meaning |
|------|------|---------|
| **Application environment** | Tier-3 | Product/lab **host** under `applications/<name>/` — HTTP entry, env, Docker, agent roster, integrations. |
| **Runtime sandbox** | Tier-1 | **Task isolation** for tool/file execution — `sandbox.exec`, `SandboxSessionManager`, `metadata.sandbox` on `Task`. Optional per-task; wired through UAEP/Nexus. |

Enabling runtime sandbox on a task does **not** create a new application directory.

Scaffold command `new-application` creates a Tier-3 **application environment**.

Task flag / metadata `sandbox=True` enables Tier-1 **runtime sandbox** inside an existing host.

### 7.4.10 Application Composition Contract (Phase N)

Tier-2 agents declare **`AgentContract`** (capabilities, tools, risk).

Tier-3 applications declare an **`ApplicationManifest`** (Phase N.1 — `intergrax/applications/contracts/manifest.py`):

- `app_id`, `route_prefix`, environment defaults
- `agents[]` — roster entries via **`AgentBinding.mount(AgentClass, factory=...)`** (strongly typed); serialized scaffold uses ``deserialize(import_path=...)`` only
- `integration_profile` — typed `IntegrationProfile`
- `features` — scheduler, debug surface, interaction routes, default sandbox-on-task (boolean map)

The manifest is the **roster contract** (who is mounted). **Instance creation** is unified via ``build_application_registry()`` (Phase N.2.1 — ``intergrax/applications/_shared/wiring.py``):

| Priority | Source | Use when |
|----------|--------|----------|
| 1 | ``factory=`` on ``AgentBinding.mount()`` | **Preferred** — typed callable; mypy/IDE see class + factory |
| 2 | ``builders[type[Agent]]`` | Type-keyed map in `host/agent_builders.py`` (lab) |
| 3 | ``builder_key`` / ``factory_path`` strings | Scaffold-generated manifests only |
| 4 | zero-arg ``agent_type()`` | Simple agents with no Tier-3 config |

``ApplicationBuildContext`` carries ``manifest``, ``settings``, and ``integration_profile`` into every factory.

**Rules:**

- Manifest describes **wiring**, not domain logic.
- Secrets and heavy config stay in ``settings.from_env()`` — binding ``config`` is for lightweight options only.
- `host/wiring.py` calls ``build_application_registry(manifest, ctx, builders=...)``.
- Reference: ``AgentBinding.mount(EchoAgent)`` + ``LAB_AGENT_BUILDERS`` keyed by type; ``AgentBinding.mount(LegalAgent, factory=build_legal_agent_from_context)``.
- `python -m intergrax.scaffold new-application` generates manifest + builders skeleton (Phase N.3).

See [`plan/PLATFORM_FOUNDATION.md) Phase N for step-by-step delivery.

**Usage guides:** composition engine — [`intergrax/applications/USAGE.md`](../intergrax/applications/USAGE.md); application hosts — [`applications/USAGE.md`](../applications/USAGE.md).

### 7.4.11 Intergrax Assistant Application (IAA)

**Role:** Harness-native **conversational lab** — ChatGPT-shaped product shell for experimenting with the full Agent OS on a **swappable LLM adapter** (local Ollama default).

| Piece | Location |
|-------|----------|
| Tier-3 host | `applications/intergrax_assistant_application/` |
| Hub agent | `agents/intergrax_assistant/` — capability `platform.assist` |
| Architecture | [`applications/intergrax_assistant_application/ARCHITECTURE.md`](../applications/intergrax_assistant_application/ARCHITECTURE.md) |
| ADR | [`ADR-INTERGRAX_ASSISTANT-001`](../applications/intergrax_assistant_application/adr/ADR-INTERGRAX_ASSISTANT-001.md) |

**Topology (hub-and-spoke):**

```text
Client (HTTP / MCP)
    → intergrax_assistant_application (Tier-3)
        → NexusLoop: classify → plan (engine when `INTERGRAX_ASSISTANT_ENGINE_PLANNER=true`) → graph
            → intergrax_assistant (hub, default)
            → optional DelegationSpec → Legal / Research / … (env-mounted roster)
        → FinalResponseComposer → client
```

**Differentiators vs other hosts:**

| Host | Pattern |
|------|---------|
| `lab_application` | Multi-agent debug lab — no chat product contract |
| `legal_application` | Single-domain chat SKU |
| `local_workspace_application` | Fixed multi-agent file pipeline (LKW) |
| **IAA** | General chat hub + **LLM env swap** + optional platform delegation |

LLM resolution: `ApplicationEnvironmentProfile.llm_profile` from `INTERGRAX_LLM_PROVIDER` / `INTERGRAX_LLM_MODEL` (see [`architecture/LLM_ADAPTERS.md`](architecture/LLM_ADAPTERS.md)). Default port `8096`.

---


---

# 8. Core Design Principles

## 8.1 Runtime First

The runtime is more important than any single agent.

Agents are replaceable.

The runtime is the long-term asset.

Implementation rule:

> Do not optimize the architecture around one agent. Optimize around the ability to create many agents quickly.

---

## 8.2 Experimentation First

Intergrax is currently a laboratory.

It should optimize for:

- fast agent creation
- fast hypothesis testing
- clear observability
- low setup cost
- easy deletion of failed experiments
- simple integration with existing tools

It should NOT currently optimize for:

- enterprise UI complexity
- marketplace features
- billing
- advanced tenant management
- unnecessary abstractions
- premature distributed complexity

---

## 8.3 Nexus Owns Global Reasoning

Nexus owns the global reasoning loop.

**Canonical depth:** [`architecture/REASONING_AND_COGNITION.md`](architecture/REASONING_AND_COGNITION.md) — three cognition planes, planners, classifiers, `DecisionRecord`.

Nexus decides:

- what the user wants
- which agents are needed
- whether the task is simple or complex
- whether execution is sequential or parallel
- when to retry
- when to ask a human
- when to stop
- how to compose the final answer

---

## 8.4 Agents Own Local Execution

Agents own local domain execution.

Agents decide:

- how to perform their bounded task
- which local tools to use
- how to improve their local result
- how to validate their local output

Agents do not decide the global workflow unless explicitly delegated by Nexus for a bounded subtask.

---

## 8.5 Integrations Are Adapters

Slack, Teams, email, databases, browser automation and other external tools are adapters.

They are not agents.

They are not Nexus.

They are infrastructure capabilities exposed to the runtime.

---

## 8.6 UI Is Optional

Intergrax is not frontend-first.

The runtime must work without a heavy UI.

Slack, Teams, chat, CLI or a lightweight internal dashboard can be valid interaction surfaces.

UI must not define the architecture.

---

## 8.7 Observability Is Mandatory

Every meaningful step must be observable.

An agent experiment without traces is not useful.

The system must show:

- what was requested
- what Nexus understood
- what plan was created
- which agents were selected
- which tools were used
- what data was processed
- what failed
- what was retried
- what the result was
- why the system stopped

---

## 8.8 Reuse Tier-0 — Do Not Duplicate Platform Mechanisms

Intergrax is not a greenfield project. The Tier-0 platform already provides LLM adapters, logging, RAG, tools, memory, queues, tracing primitives, and infrastructure adapters.

**Mandatory rules:**

1. **Search Tier-0 first.** Before adding any integration or utility, locate the existing canonical module.
2. **Wire, don't rebuild.** Tier-1 and Tier-2 work composes existing capabilities through Nexus, `ToolRuntime`, and `AgentEngine`.
3. **One mechanism per concern.** Never introduce a second LLM layer, logging stack, tool registry, vector store client, or trace system.
4. **Extend in place.** If Tier-0 is insufficient, prefer extending the existing module over creating a parallel one.
5. **Human gate for new universals.** New Tier-0 capabilities require explicit human approval (§5.2.4).

Implementation agents (including Cursor AI) that propose new universal modules MUST escalate to the human operator — this is a form of **human-in-the-loop governance for platform evolution**.

Anti-pattern:

```text
# FORBIDDEN — duplicate LLM path in agent
from openai import AsyncOpenAI
client = AsyncOpenAI(...)
response = await client.chat.completions.create(...)

# REQUIRED — canonical path
from intergrax.llm_adapters...  # via AgentEngine / configured adapter
```

---


---

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

Do not implement agent capabilities inside `applications/`.

Do not put FastAPI hosts, env settings, or HTTP serving inside `agents/`.

See §7.4 for the canonical repository split.

See §42.41 for forbidden runtime patterns (direct adapters, private loops, runtime bypass).

---

## 43.8 Platform Redundancy Anti-Pattern

Do NOT introduce a second implementation of a universal platform concern when Tier-0 already provides one.

Examples of forbidden redundancy:

- new LLM adapter layer alongside `intergrax/llm_adapters/`
- agent-local logging instead of `intergrax/logging.py`
- custom vector store client instead of `intergrax/rag/`
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

Agent capability code lives under `agents/<name>/`.

Execution environments (Tier-3: host, serving, env config) live under `applications/<name>/`.

The framework package `intergrax/` MUST remain free of product-specific agent implementations.

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

All reusable **external infrastructure** adapters (databases, caches, queues, chat, search, issue trackers, observability backends, cloud infrastructure facades) MUST be registered in **`intergrax/integrations/`** under category contracts (§7.1).

**Excluded:** LLM providers remain in **`intergrax/llm_adapters/`** only — not in the Integration Library (§7.1.2).

Reason:

Multiple agent teams need the same building blocks (Redis cache, Jira tasks, Slack HITL). A catalog with universal contracts lets one platform team maintain adapters while product teams compose them in Tier-3 applications — without copying SDK code into `agents/`.

Constraint:

- New **providers** follow Phase M checklist in the implementation plan.
- New **categories** require §5.2.4 human approval.
- Legacy modules (`queueing/`, `distributed/`, etc.) are wrapped, not duplicated.

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

**Phase V implementation baseline** (code modules, not duplicated here): `intergrax/runtime/architecture/` — capability graph, lifecycle, eval, security, cost, context/prompt quality, graph RAG helpers. Traceability: [`plan/PLATFORM_FOUNDATION.md`](../plan/PLATFORM_FOUNDATION.md) Phase V / FAUDIT-32.

---
