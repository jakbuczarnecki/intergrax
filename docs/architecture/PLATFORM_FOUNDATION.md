# Platform Foundation

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/PLATFORM_FOUNDATION.md`](../plan/PLATFORM_FOUNDATION.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 1–2, 32  
**Audit instruction:** [`audit/PLATFORM_FOUNDATION.md`](../audit/PLATFORM_FOUNDATION.md)  
---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (PLATFORM_FOUNDATION canon).

- **Implement / audit default:** §1–§5 + §5.2 reuse + §5.3 terminology. Extended §7–§8: [`arch/PLATFORM_FOUNDATION_extended_depth.md`](arch/PLATFORM_FOUNDATION_extended_depth.md). §43+: [`arch/PLATFORM_FOUNDATION_production_gates.md`](arch/PLATFORM_FOUNDATION_production_gates.md).
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/PLATFORM_FOUNDATION.md`](../plan/PLATFORM_FOUNDATION.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/PLATFORM_FOUNDATION.md`](../guides/audit_slices/PLATFORM_FOUNDATION.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---


## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`arch/PLATFORM_FOUNDATION_extended_depth.md`](arch/PLATFORM_FOUNDATION_extended_depth.md) | extended depth |
| [`arch/PLATFORM_FOUNDATION_production_gates.md`](arch/PLATFORM_FOUNDATION_production_gates.md) | production gates |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

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

**Cross-layer invariants (canonical):** [`guides/SYSTEM_INVARIANTS.md`](../guides/SYSTEM_INVARIANTS.md#cross-layer-system-invariants) (P2-ARCH-01) — MUST/MUST NOT rules across all tiers; `SYS-INV-*` index links to this §5 and domain pairs.

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
