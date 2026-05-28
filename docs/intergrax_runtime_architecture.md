# INTERGRAX_RUNTIME_ARCHITECTURE.md

Status: Canonical architecture and implementation guide  
Audience: Humans, LLMs, Cursor AI agents, implementation agents, future maintainers  
Purpose: Define the Intergrax runtime architecture, implementation rules, agent model, orchestration model, adapter model, experimentation model and forbidden patterns.

---

# 1. Purpose Of This Document

This document is the primary architecture and implementation specification for the Intergrax runtime.

This document is intentionally written to be readable by:

- humans
- LLMs
- GPT-like models
- Claude-like models
- Gemini-like models
- Cursor AI agents
- autonomous coding agents
- future implementation agents

This document MUST be treated as the canonical source of truth for implementing Intergrax.

The platform is defined as **four tiers** (Tier-0 Platform → Tier-1 Nexus → Tier-2 Agents → Tier-3 Applications). See §5.1.

The **Unified Execution Runtime Specification** (§42) defines the canonical implementation contracts for AgentEngine, events, hooks, lifecycle, decisions, interrupts, policy, middleware, and governance. All agent implementations MUST conform to §42.

**Critical implementation rule:** Intergrax MUST **reuse existing Tier-0 platform mechanisms** — not duplicate them. New universal components require explicit human approval. See §5.2, §8.8, §39.8.

When an implementation decision is unclear, the implementation agent MUST prefer the principles, boundaries and contracts defined in this document.

This document is NOT a marketing document.

This document is NOT a product roadmap.

This document is an architectural and implementation guide.

---

# 2. Executive Summary

Intergrax is an AI Operating System / Agent Runtime / Harness AI environment.

The current goal is NOT to build a finished SaaS product.

The current goal is to build an internal agent experimentation laboratory where new agentic capabilities can be created, tested, observed, validated, improved or discarded quickly.

The ideal workflow is:

```text
new idea
    -> define agent capability
    -> implement agent contract
    -> register agent in Nexus
    -> connect required adapters/tools
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

Intergrax is designed to answer this question:

> Can we rapidly create, run and evaluate new AI agents without rebuilding infrastructure every time?

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

Intergrax should learn from Cursor AI, Viktor, NotebookLM and modern agent runtimes, but the current goal is to build a controlled internal experimentation environment.

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
- Infrastructure adapters: PostgreSQL, Redis, queues, Kafka, file storage
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
| LLM calls | `intergrax/llm_adapters/` (`LLMAdapter`, registry) | Direct OpenAI/Anthropic SDK calls; second LLM wrapper in agents |
| Logging | `intergrax/logging.py` and established log patterns | `print()`, ad-hoc loggers, duplicate logging frameworks |
| Tracing (pipeline) | Nexus `trace_event()` / `RunTraceWriter` | Parallel untracked diagnostic streams |
| Tools | `intergrax/tools/` (`ToolRegistry`, `ToolExecutor`) | Agent-local tool registries |
| RAG | `intergrax/rag/` | Duplicate embedding/retrieval stacks in agents |
| Web search | `intergrax/websearch/` | Custom HTTP search clients in agents |
| Memory / session | `intergrax/memory/`, Nexus session storage | Direct Redis/PostgreSQL access from agents |
| Queues | `intergrax/queueing/` | Ad-hoc background job systems |
| Tokenization | `intergrax/tokenizers/` | Inline tiktoken/token counting duplicates |
| File / storage adapters | Tier-0 adapters | Agent-local S3/filesystem clients bypassing adapters |
| Errors / classification | `intergrax/runtime/nexus/errors/` | Siloed error models per agent |

This table is illustrative, not exhaustive. The rule is general:

> **If a universal capability already exists in Tier-0, use it. Do not add a second one.**

### 5.2.3 What §42 Contracts Are (And Are Not)

§42 Unified Execution Runtime defines **orchestration contracts and governance wiring** — how Nexus and agents coordinate execution.

§42 does **NOT** authorize creating duplicate Tier-0 implementations. For example:

- `RuntimeEvent` MUST integrate with — not replace — existing trace/logging infrastructure.
- `ToolRuntime` MUST delegate to existing `ToolRegistry` / tool steps — not a parallel tool system.
- `AgentEngine` MUST use existing `RuntimeEngine` / pipeline — not a second execution engine.

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

- Global Nexus loop (`NexusLoop`, task intake, classification, planning)
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

- Host entrypoint (`main.py`, `factory.py`, `settings.py`)
- HTTP/CLI serving layer (routes, auth, tenant config)
- Environment configuration (.env profiles, SKU rules, feature flags)
- Agent registry wiring: which agents are registered, with which IDs and policies
- Orchestration config: default capabilities, routing hints, multi-agent topologies
- Deployment wiring (optional Docker/k8s)

**Example environments:**

- `legal_application` — legal review for law firms (Legal agent + compliance rules)
- `research_application` — research → summarize pipeline for analysts
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
| TesterAgent   MarketerAgent    VendorDiscoveryAgent  ...     |
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

# 7. Layer Responsibility Summary

> **Canonical naming:** Tier-0..3 (§5.1). Subsections below retain legacy “Layer N” labels where noted.

## 7.0 Tier Overview

| Tier | Section | Package / folder |
|------|---------|------------------|
| Tier-0 Platform | §7.1 | `intergrax/` adapters, rag, tools, memory, queueing, … |
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
- product deployment manifests (Dockerfile, k8s) unless explicitly shared infra
- global orchestration or cross-agent routing

Agents MUST be runnable through Nexus (`AgentEngine`, `NexusLoop`) **without** starting an HTTP server.

### 7.4.3 What Belongs In Tier-3 (`applications/<name>/`)

An application is a **ready-made environment** (Tier-3) that composes Nexus + agents + configuration.

An application MUST contain:

- host package (`main.py`, `factory.py`, `settings.py`, wiring)
- serving layer (FastAPI routers, request/response mapping)
- environment-level configuration (env vars, product profiles, tenant defaults)
- registration of agents into `AgentRegistry`
- orchestration config: agent roles, default capabilities, interaction topology
- industry- or company-specific rules (env-level, not agent domain code)

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
1. python -m intergrax.scaffold new-agent <name> --capabilities <cap>.<action>
   → creates agents/<name>/

2. Implement domain logic in agents/<name>/
   → get_contract(), build_context(), pipeline

3. (Optional) Create applications/<name>_application/
   → host + serving if HTTP/product entry is needed

4. Register agent in AgentRegistry (notebook, test, or application factory)

5. Run through NexusLoop → observe trace → evaluate
```

Not every agent requires an application.

Notebook-only or test-only experiments MAY use `agents/<name>/` without creating `applications/`.

Create an application only when a stable host, env config, or external API surface is required.

### 7.4.7 Anti-Pattern: Agent-Application Monolith

Do NOT place agent implementation, pipeline, host, serving, and env config in a single package under `applications/`.

This was the legacy `applications/legal_agent/` layout.

It couples capability code to deployment, makes reuse across environments harder, and violates Tier-2 / Tier-3 boundaries.

If agent logic and host live together, split them before adding a second agent or second deployment target.

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
from intergrax.llm_adapters...  # via RuntimeEngine / configured adapter
```

---

# 9. Dual Loop Architecture

Intergrax uses a dual-loop architecture.

There are two types of loops:

1. Global Nexus Loop
2. Local Agent Loop

This is a required architectural decision.

---

# 9.1 Global Nexus Loop

The Nexus loop is mandatory.

The Nexus loop controls global execution.

Responsibilities:

- receive user task
- classify task
- determine complexity
- create or update plan
- select agents
- prepare context
- execute agents
- evaluate results
- decide next step
- handle retries
- coordinate parallel work
- coordinate sequential work
- request human approval when required
- finalize output

Pseudo-flow:

```text
while task.status not in [completed, failed, cancelled]:

    current_state = load_task_state(task_id)

    reasoning_result = reason_about_current_state(current_state)

    next_action = determine_next_action(reasoning_result)

    if next_action.type == "execute_agent":
        result = execute_agent(next_action.agent, next_action.input)
        store_result(result)

    if next_action.type == "execute_parallel_agents":
        results = execute_agents_in_parallel(next_action.agents)
        store_results(results)

    if next_action.type == "ask_human":
        pause_and_request_human_input()

    if next_action.type == "retry":
        execute_retry_policy()

    validation_result = validate_current_state()

    update_task_state(validation_result)
```

---

# 9.2 Local Agent Loop

Agents MAY have local loops — but loops MUST be **runtime-controlled** (§42.32, §42.33).

Local loops are allowed when an agent requires multiple internal steps.

The agent loop MUST be bounded by:

- the input contract
- the output contract
- max steps
- max time
- max cost
- allowed tools
- validation rules

Pseudo-flow:

```text
while local_goal_not_completed and limits_not_exceeded:

    local_state = inspect_local_state()

    local_next_step = decide_local_next_step(local_state)

    local_result = execute_local_step(local_next_step)

    validate_local_result(local_result)

    update_local_state(local_result)

return agent_output_artifact
```

---

# 9.3 Why Both Loops Are Required

If only Nexus has a loop:

- Nexus becomes too large
- Nexus micromanages every domain
- domain-specific logic leaks into the runtime
- implementation becomes rigid

If only agents have loops:

- global orchestration becomes chaotic
- agents become mini-platforms
- state becomes fragmented
- retries become inconsistent
- final output becomes unpredictable

Correct decision:

> Nexus has the global loop. Agents may have bounded local loops.

---

# 10. Nexus Responsibilities

Nexus is responsible for the following areas.

## 10.1 Task Intake

Nexus receives tasks from:

- chat interface
- Slack
- Teams
- API
- CLI
- internal scheduler
- webhook
- event trigger

Task intake normalizes input into a standard Task object.

---

## 10.2 Task Classification

Nexus classifies the task.

Possible classifications:

- simple question
- single-agent task
- multi-agent task
- long-running workflow
- monitoring task
- scheduled task
- human-approval-required task
- unsafe task
- unsupported task

---

## 10.3 Planning

Nexus creates a plan when needed.

A plan may include:

- steps
- dependencies
- agent assignments
- required tools
- expected artifacts
- validation criteria
- human approval points
- risk level

---

## 10.4 Agent Selection

Nexus selects agents based on:

- task intent
- agent registry
- declared capabilities
- required tools
- previous performance
- cost
- availability
- risk level

---

## 10.5 Execution Graph

Nexus manages the execution graph.

The execution graph defines:

- nodes
- dependencies
- parallel branches
- sequential branches
- waiting states
- retry states
- failed states
- completed states

---

## 10.6 State Management

Nexus owns global task state.

Global state includes:

- task id
- run id
- user input
- normalized task
- current plan
- execution graph
- agent outputs
- tool outputs
- validation results
- human messages
- final result
- status

---

## 10.7 Context Management

Nexus decides what context is passed to each agent.

Agents MUST receive only the context needed for their bounded task.

Nexus prevents uncontrolled context growth.

---

## 10.8 Tool And Adapter Access Policy

Nexus defines which tools and adapters an agent may use.

Agents should not automatically receive access to every integration.

Tool access should be explicit.

---

## 10.9 Validation

Nexus validates whether the global task is complete.

Validation can include:

- schema validation
- rule validation
- secondary agent validation
- tests
- consistency checks
- human approval

---

## 10.10 Final Response

Nexus composes the final response to the user.

Agents produce artifacts.

Nexus decides how artifacts are presented.

---

# 11. Agent Responsibilities

Agents are specialized execution modules.

An agent is responsible for:

- understanding its local task
- using allowed tools
- executing domain-specific logic
- producing structured output
- validating local output
- reporting uncertainty
- reporting failures
- returning artifacts to Nexus

An agent is NOT responsible for:

- global orchestration
- global task lifecycle
- global retries
- user communication outside the contract
- creating unrelated agents
- bypassing Nexus
- owning cross-agent memory

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

# 13. Suggested Agent Interface

This is conceptual pseudocode, not a required programming language implementation.

```text
interface Agent:

    get_contract() -> AgentContract

    can_handle(task_context) -> CapabilityMatchResult

    execute(agent_input, execution_context) -> AgentExecutionResult

    validate(agent_output, execution_context) -> ValidationResult
```

Agent implementations should be simple.

The goal is to let developers focus on domain logic, not infrastructure.

All `execute()` implementations MUST delegate to `AgentEngine` and the Unified Agent Execution Protocol (§42.5). Agents MUST NOT implement private runtime lifecycles.

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

---

# 17. Adapter Architecture

Adapters are reusable integrations with external systems.

Examples:

- SlackAdapter
- TeamsAdapter
- EmailAdapter
- PostgreSqlAdapter
- RedisAdapter
- BrowserAdapter
- WebSearchAdapter
- FileSystemAdapter
- VectorStoreAdapter
- LlmProviderAdapter
- SandboxAdapter

Adapters MUST be treated like infrastructure components.

Adapters MUST NOT contain business workflow logic.

Adapters MUST NOT decide which agent to run.

Adapters expose operations.

Nexus or agents call those operations through explicit permissions.

---

# 18. Slack / Teams / Communication Integration Philosophy

Intergrax should support Slack and Teams as interaction surfaces.

This follows the Viktor-like idea where an AI worker can live inside organizational communication tools.

Slack and Teams should be implemented as adapters.

They may provide:

- task intake
- notifications
- approval requests
- progress updates
- final responses
- interactive buttons
- user context
- channel context

They should NOT own the runtime.

Correct model:

```text
Slack message
    -> SlackAdapter
    -> normalized Task
    -> Nexus Runtime
    -> Agent execution
    -> Nexus final result
    -> SlackAdapter sends response
```

Incorrect model:

```text
Slack bot contains orchestration logic
Slack bot directly manages agents
Slack bot stores global task state
```

---

# 19. UI / UX Testing Requirement

Even though Intergrax is not frontend-heavy, agents must be testable and observable.

The system should support minimal UI/UX surfaces for:

- viewing task list
- viewing task status
- viewing execution trace
- viewing agent outputs
- viewing tool calls
- viewing errors
- viewing artifacts
- approving or rejecting steps
- re-running tasks
- comparing outputs

This may be implemented as:

- lightweight dashboard
- CLI
- chat interface
- Slack/Teams messages
- internal debug panel

The UI is for observability and experimentation, not product polish.

---

# 20. Shadow Workspace Model

A Shadow Workspace is an isolated temporary workspace used to perform work without directly modifying the main environment.

Inspired by Cursor-like execution environments.

Shadow Workspaces may be used for:

- code experiments
- document analysis
- temporary data transformations
- simulated business workflows
- vendor research sessions
- legal document review sessions
- onboarding simulations

A Shadow Workspace should provide:

- isolation
- temporary storage
- reproducibility
- rollback safety
- inspectable artifacts
- cleanup

---

# 21. Sandbox Model

A sandbox is a controlled execution environment.

Use sandboxes for:

- code execution
- browser automation
- file manipulation
- risky tool use
- external data extraction
- generated script execution

Sandbox execution should be:

- isolated
- observable
- permission-controlled
- interruptible
- disposable
- reproducible when possible

---

# 22. Tool Runtime

Tools are callable operations exposed to Nexus and agents.

Examples:

- search web
- read file
- write file
- query database
- send Slack message
- create document
- call LLM
- run browser action
- execute script in sandbox

Tools must have:

- name
- description
- input schema
- output schema
- risk level
- permission requirement
- timeout
- retry policy

Tools should be registered in a Tool Registry.

All agent and Nexus tool invocation MUST route through `ToolRuntime` with policy enforcement (§42.12, §42.36). Direct adapter calls from agents are forbidden (§42.41).

---

# 23. Task Lifecycle

Every task should move through explicit states.

Recommended lifecycle:

```text
created
    -> classified
    -> planned
    -> waiting_for_resources
    -> running
    -> waiting_for_human
    -> validating
    -> completed
```

Failure states:

```text
failed
cancelled
expired
partially_completed
needs_more_information
```

Every transition should be logged.

---

# 24. Execution Graph

Complex tasks should be represented as execution graphs.

An execution graph contains:

- nodes
- dependencies
- execution status
- assigned agent
- input
- output
- validation result
- retry count

Example:

```text
Task: Find business partner for AI logistics project

Node 1: Analyze project description
Node 2: Define partner criteria
Node 3: Search companies
Node 4: Enrich company profiles
Node 5: Score companies
Node 6: Validate ranking
Node 7: Generate final recommendation
```

Some nodes may run sequentially.

Some nodes may run in parallel.

---

# 25. Sequential And Parallel Execution

Nexus decides whether execution is sequential or parallel.

Sequential execution is preferred when:

- later steps depend on previous outputs
- task risk is high
- context must be controlled
- quality is more important than speed

Parallel execution is allowed when:

- subtasks are independent
- agents work on separate data
- research can be split
- validation can run independently

Nexus must merge parallel results.

---

# 26. Long Running Tasks

Intergrax must support long-running tasks.

Examples:

- monitor Reddit for problem signals for 30 days
- onboard new employees for 2 weeks
- analyze monthly sales data
- audit vendors over multiple stages
- review a large document set

Long-running tasks require:

- persistent state
- resumability
- scheduled execution
- progress updates
- failure recovery
- human interruption
- partial results

---

# 27. Memory Model

Memory must be explicit and bounded.

Types of memory:

1. Task Memory
2. Agent Local Memory
3. User / Organization Memory
4. Long-Term Knowledge Memory
5. Execution Trace Memory

Rules:

- Nexus owns global task memory.
- Agents may own local temporary memory.
- Long-term memory must be explicitly written.
- Agents must not silently mutate global memory.
- Sensitive memory writes should require policy checks.

---

# 28. Context Management

Context is expensive and dangerous when uncontrolled.

Nexus must control what context each agent receives.

Rules:

- pass only relevant context
- avoid dumping entire history into agents
- use summaries when needed
- separate task context from user memory
- separate evidence from interpretation
- preserve provenance

---

# 29. Validation Model

Validation is mandatory.

Validation should not rely only on LLM confidence.

Possible validation types:

- schema validation
- rule-based validation
- data completeness validation
- source citation validation
- secondary model review
- separate validator agent
- human review
- executable tests
- consistency checks

Validation should be defined before or during planning.

For high-risk tasks, Nexus should create a validation contract before execution.

---

# 30. Failure Model

Failures are expected.

The system must treat failure as normal.

Failure types:

- agent failure
- tool failure
- adapter failure
- timeout
- invalid output
- missing data
- low confidence
- unsafe action
- human rejection
- incomplete result

Failure handling options:

- retry same step
- retry with different agent
- ask human
- degrade gracefully
- return partial result
- stop execution
- mark as failed

---

# 31. Retry Policy

Retries must be controlled.

Every retry should have:

- reason
- retry count
- changed strategy if possible
- stop condition

Do not retry endlessly.

Retries should be visible in traces.

---

# 32. Human In The Loop

Human approval may be required for:

- sending external messages
- modifying external systems
- deleting data
- financial actions
- legal conclusions
- risky automation
- uncertain results

Nexus manages human approval.

Agents may request approval via `AgentDecision.REQUEST_HUMAN`, but Nexus controls the approval flow (§42.10).

Agents MUST NOT implement ad-hoc human gates or send approval messages directly.

---

# 33. Observability And Tracing

Every execution should create a trace.

The canonical observability model is **event-first** (§42.1, §42.24): `RuntimeEvent` stream persisted to trace storage, with correlation via `task_id`, `run_id`, and `correlation_id`.

Trace should include:

- task id
- run id
- user input
- normalized task
- plan
- reasoning summaries
- agent selections
- agent inputs
- agent outputs
- tool calls
- adapter calls
- errors
- retries
- validation results
- final result
- timestamps
- cost information if available

Observability exists for:

- debugging
- evaluation
- hypothesis validation
- cost control
- safety
- future improvement

---

# 34. Evaluation Model

Since Intergrax is an experimentation laboratory, every agent should be evaluated.

Evaluation criteria may include:

- task success
- output quality
- factuality
- completeness
- cost
- latency
- usefulness
- repeatability
- user satisfaction
- failure frequency
- business value

Agents should not be considered successful only because they produced text.

---

# 35. Experimentation Workflow

The expected workflow for a new idea:

```text
1. Define hypothesis
2. Define agent capability
3. Define expected output
4. Define validation criteria
5. Implement minimal agent
6. Register agent
7. Run through Nexus
8. Observe execution trace
9. Compare outputs
10. Decide: keep, improve, pause or delete
```

Example hypothesis:

> ProblemRadarAgent can discover repeated user complaints from Reddit and Hacker News and cluster them into potential product ideas.

This should become an agent experiment, not a full product.

---

# 36. Example: Problem Radar Agent

Purpose:

Identify repeated user problems, complaints and pain signals from public sources.

Possible sources:

- Hacker News
- Reddit
- forums
- review sites
- product communities
- social platforms

Possible steps:

```text
1. Collect posts/comments
2. Filter noise
3. Extract pain statements
4. Group similar pains
5. Cluster by market/problem
6. Score frequency and intensity
7. Apply problem quality filters
8. Generate opportunity report
```

Expected output:

```text
ProblemRadarOutput:
    clusters
    representative_quotes
    source_links
    frequency_estimate
    intensity_score
    affected_user_group
    possible_product_ideas
    mom_test_risk_notes
    confidence
```

This agent should be implemented as a capability module and executed through Nexus.

---

# 37. Example: Vendor Discovery Agent

Purpose:

Find, categorize, evaluate and recommend companies for a given business need.

Possible use cases:

- find subcontractors
- find business partners
- find potential customers
- audit vendors
- compare companies

Possible steps:

```text
1. Understand customer need
2. Define search criteria
3. Search company sources
4. Enrich company profiles
5. Categorize companies
6. Score fit
7. Detect risks
8. Produce recommendation
```

Expected output:

```text
VendorDiscoveryOutput:
    companies
    categories
    fit_scores
    strengths
    risks
    evidence
    source_links
    recommendation
    confidence
```

---

# 38. Example: Organization Worker Agent

Purpose:

Act as a virtual worker inside an organization through Slack, Teams or other communication tools.

Possible tasks:

- prepare monthly sales analysis
- onboard employees
- coordinate document review
- summarize project updates
- monitor operational signals
- prepare vendor reports

Architecture:

```text
User message in Slack
    -> SlackAdapter
    -> Nexus
    -> task classification
    -> agent selection
    -> execution
    -> progress updates
    -> final response in Slack
```

The Slack integration is only an interaction surface.

The runtime remains in Nexus.

---

# 39. Implementation Rules For Cursor AI

When Cursor AI or an LLM coding agent implements Intergrax, it MUST follow these rules.

## 39.1 Always Preserve Layer Boundaries

Do not put orchestration logic into adapters.

Do not put business agent logic into Nexus.

Do not put platform lifecycle logic into agents.

---

## 39.2 Prefer Contracts Over Hardcoding

Use contracts, registries and schemas.

Avoid direct hardcoded branching such as:

```text
if task contains "vendor": run VendorAgent
```

Prefer capability matching.

---

## 39.3 Build Minimal Useful Runtime First

Initial implementation should focus on:

- AgentContract
- AgentRegistry
- Task object
- Nexus execution loop
- basic ToolRegistry
- basic TraceLogger
- simple adapter model
- one or two example agents

Do not build the entire platform prematurely.

---

## 39.4 Every New Agent Must Be Runnable Through Nexus

Agents should not be executed as standalone scripts except for isolated unit tests.

The normal path is:

```text
Task -> Nexus -> Agent -> Result -> Nexus
```

---

## 39.5 Every Agent Must Produce Structured Output

Agents must not return only raw text.

Raw text may exist as summary, but structured data is required for evaluation.

---

## 39.6 Every Execution Must Be Traceable

No hidden execution.

Every meaningful decision should produce a trace event or structured log.

---

## 39.7 Prefer Simple Internal UI

If a UI is needed, build a minimal debug/inspection surface.

Do not build a polished SaaS frontend at this stage.

---

## 39.8 Reuse Tier-0 — Never Duplicate Universal Mechanisms

Before writing code, Cursor AI and implementation agents MUST:

1. Identify whether the needed capability **already exists** in Tier-0 (§5.2.2).
2. Use the **canonical entry point** (LLM adapters, logging, tools, RAG, trace, memory, queues).
3. Implement **orchestration and domain logic only** in Tier-1 / Tier-2 / Tier-3.
4. **STOP and ask the human** if a new universal Tier-0 mechanism appears necessary (§5.2.4).

Cursor AI MUST NOT:

- add parallel LLM client wrappers,
- create agent-local logging or tracing systems,
- introduce duplicate tool registries or adapter facades,
- add new PostgreSQL/Redis/file clients in agents when Tier-0 adapters exist,
- implement §42 scaffold as standalone replacements for existing Nexus trace/tool/LLM paths.

When wiring §42 (events, hooks, UAEP), **integrate with** existing `RunTraceWriter`, `ToolRuntime`, `RuntimeEngine` — do not fork them.

---

# 40. Recommended Minimal First Implementation

The first implementation milestone should include:

```text
core/
    AgentContract
    AgentRegistry
    Task
    TaskState
    NexusRuntime
    ExecutionContext
    AgentExecutionResult
    ValidationResult
    TraceLogger

components/
    LlmProviderAdapter
    SlackAdapter interface placeholder
    TeamsAdapter interface placeholder
    StorageAdapter
    QueueAdapter placeholder

agents/
    EchoAgent
    ResearchAgent prototype
    ProblemRadarAgent prototype

applications/
    legal_application/          # host + serving + env config (composes agents/legal)
    <name>_application/         # future execution environments

runtime/
    NexusLoop
    TaskClassifier
    Planner
    AgentRouter
    ExecutionGraph
```

This is enough to validate the architecture.

Do not start with too many agents.

---

# 41. Minimal Runtime Flow

The first usable flow should be:

```text
1. User submits task
2. Nexus creates Task object
3. Nexus classifies task
4. Nexus creates simple plan
5. Nexus selects agent from registry
6. Nexus executes agent
7. Agent returns structured result
8. Nexus validates result
9. Nexus logs full trace
10. Nexus returns final response
```

This validates the entire skeleton.

---

# 42. Unified Execution Runtime Specification

This section is the **canonical implementation specification** for the Intergrax execution environment.

It extends §5.1 (Four-Tier Model), §5.2 (Platform Reuse), §9 (Dual Loop), §12–§14 (Contracts), §22–§33 (ToolRuntime, Lifecycle, Validation, Observability) without changing architectural direction.

**Important:** §42 specifies **how** Nexus orchestrates execution (events, decisions, hooks, lifecycle). It does **not** permit building duplicate Tier-0 infrastructure. All §42 implementation MUST reuse existing platform modules (§5.2, §8.8).

Intergrax MUST be a **single, unified, event-driven Agent OS** — not a collection of loosely coupled agent implementations.

Every agent:

- implements the same contracts (§12, §42.13)
- runs through the same lifecycle (§42.4, §42.6)
- uses the same `AgentEngine` executor (§42.19)
- emits the same `RuntimeEvent` stream (§42.1)
- returns the same `AgentDecision` objects (§42.7)
- passes through the same middleware pipeline (§42.20, §42.42)
- accesses Tier-0 only through `ToolRuntime` policy (§42.12)

Agents provide **domain logic**. The runtime owns **execution governance**.

```text
┌─────────────────────────────────────────────────────────────┐
│  Tier-3 Application (host, config, agent roster)            │
└───────────────────────────┬─────────────────────────────────┘
                            │ Task intake
┌───────────────────────────▼─────────────────────────────────┐
│  Tier-1 Nexus — Global loop, policy, scheduling, graph    │
│  EventBus → Hooks → Middleware → Validation → Interrupts    │
└───────────────────────────┬─────────────────────────────────┘
                            │ AgentEngine.execute_step()
┌───────────────────────────▼─────────────────────────────────┐
│  Tier-2 Agent — domain steps, local reasoning, decisions    │
│  NO private runtime · NO direct adapters · NO hidden loops  │
└───────────────────────────┬─────────────────────────────────┘
                            │ ToolRuntime.invoke() only
┌───────────────────────────▼─────────────────────────────────┐
│  Tier-0 Platform — LLM, DB, Redis, files, web, sandbox      │
└─────────────────────────────────────────────────────────────┘
```

**Implementation status note:** Some §42 contracts describe the **target canonical runtime**. Existing code (e.g. `AgentEngine`, `NexusLoop`, `ToolRuntime`) implements subsets. **Scaffold modules** (contracts + `runtime/events/`, `runtime/hooks/`, `runtime/middleware/`, `runtime/policy/`) were added 2026-05-27 — see [`INTERGRAX_IMPLEMENTATION_GAP_ANALYSIS.md`](INTERGRAX_IMPLEMENTATION_GAP_ANALYSIS.md) §15–§16. New code MUST converge toward §42. Gaps are tracked in implementation plans; agents MUST NOT introduce patterns forbidden by §42.41.

### §42 Table Of Contents

| Section | Topic |
|---------|-------|
| 42.1 | Runtime Event Model |
| 42.2 | Event Bus Architecture |
| 42.3 | Hook System |
| 42.4 | Standard Agent Lifecycle |
| 42.5 | Unified Agent Execution Protocol (UAEP) |
| 42.6 | Agent Step Lifecycle |
| 42.7 | Agent Decision Model |
| 42.8 | Execution Interrupt Model |
| 42.9 | Pause / Resume Model |
| 42.10 | Human In The Loop Runtime Flow |
| 42.11 | Policy Engine |
| 42.12 | ToolRuntime Enforcement Rules |
| 42.13 | Shared Execution Contracts |
| 42.14 | Cross-Agent Communication Contracts |
| 42.15 | Agent Handoff Contracts |
| 42.16 | Validation Contract Model |
| 42.17 | Runtime State Machine |
| 42.18 | Runtime Step Contracts |
| 42.19 | AgentEngine Responsibilities |
| 42.20 | Runtime Middleware Pipeline |
| 42.21–42.22 | Extensibility & Plugin Architecture |
| 42.23–42.24 | Event Payloads & Observability Protocol |
| 42.25–42.26 | Safety & Cancellation |
| 42.27–42.29 | Capability, Contract & Compatibility Versioning |
| 42.30–42.31 | Scheduling Model & Execution Phases |
| 42.32–42.36 | Local Loops, Retries, Memory & Tool Access |
| 42.37–42.40 | Governance, Escalation, Critical Events, Recovery |
| 42.41 | Forbidden Runtime Patterns |
| 42.42 | Middleware Pipeline (canonical hook catalog) |
| 42.43 | Multi-Agent Collaboration Flow (PM→UX→Legal→Validator→Human) |
| 42.44 | AgentEngine As Universal Executor (summary) |

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
AGENT_SELECTED
CONTEXT_BUILT
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

# 45. Checklist For New Agent Implementation

Before implementing a new agent, answer:

```text
1. What hypothesis does this agent test?
2. What capability does it provide?
3. What input does it require?
4. What structured output does it produce?
5. What tools/adapters does it need?
6. What is the validation rule?
7. What are failure modes?
8. What is the maximum acceptable cost/time?
9. How will success be evaluated?
10. How will Nexus route tasks to this agent?
11. Which AgentSteps does the agent declare (§42.6)?
12. Which AgentDecision types can the agent emit (§42.7)?
13. Does the agent conform to UAEP via AgentEngine (§42.5)?
14. Are all tool calls routed through ToolRuntime (§42.12)?
15. Are forbidden runtime patterns avoided (§42.41)?
```

If these questions cannot be answered, do not implement the agent yet.

---

# 46. Checklist For New Adapter Implementation

Before implementing a new adapter, answer:

```text
1. What external system does it connect to?
2. What operations does it expose?
3. What permissions are required?
4. Is it read-only or write-capable?
5. What are risk levels?
6. What errors can happen?
7. What timeout/retry policy is needed?
8. What data should be logged?
9. What data must be protected?
10. Which agents or runtime components may use it?
```

Adapters should be generic and reusable.

---

# 47. Checklist For Nexus Changes

Before changing Nexus, answer:

```text
1. Is this change domain-agnostic?
2. Does it belong in runtime rather than an agent?
3. Does it improve orchestration, lifecycle, validation or observability?
4. Does it preserve layer boundaries?
5. Does it make agents easier to implement?
6. Does it avoid hardcoded business logic?
7. Is the behavior traceable?
8. Can it support future agents?
9. Does it emit RuntimeEvents and respect UAEP (§42)?
10. Does it integrate with HookRegistry / middleware pipeline (§42.20)?
```

If the change is domain-specific, it probably belongs in an agent.

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

Current priority:

> Build a reliable minimal runtime for fast agent experimentation.

---

# 51. Final Canonical Statement

Intergrax is a **four-tier AI platform**: Platform (Tier-0) → Nexus Agent OS (Tier-1) → Agents (Tier-2) → Applications (Tier-3).

Intergrax is a **unified, event-driven Agent OS and Harness AI runtime** governed by §42 Unified Execution Runtime Specification.

The current purpose of Intergrax is to serve as an internal laboratory for rapid experimentation with agentic business functionality.

Tier-1 Nexus is the global orchestration runtime (Agent OS).

Tier-2 agents are bounded capability modules with shared contracts and **runtime-controlled** local loops — executed exclusively through `AgentEngine`.

Tier-0 adapters are reusable integrations consumed **only** through `ToolRuntime` policy.

Tier-3 applications are configured environments that compose Nexus + agents for specific industries or use cases.

Every agent MUST emit `RuntimeEvent`s, return `AgentDecision`s, and pass through the middleware pipeline. No private runtimes. No direct adapter access. No execution bypass.

**Platform reuse is mandatory:** one canonical mechanism per universal concern (LLM, logging, tools, RAG, trace, memory). Do not duplicate Tier-0. New universal components require human approval (§5.2.4).

The architecture must optimize for rapid hypothesis validation, observability, modularity, enforceable execution governance, and clean separation of responsibilities.

The system should make it possible to quickly implement a new agent, run it through Nexus, observe results, evaluate business value and decide whether the capability deserves further investment.

This is the core architectural direction of Intergrax.

