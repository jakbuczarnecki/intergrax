# Intergrax

[![Regression gate](https://github.com/jakbuczarnecki/intergrax-ai/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/jakbuczarnecki/intergrax-ai/actions/workflows/unit-tests.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Harness AI](https://img.shields.io/badge/Harness%20AI-Agent%20OS-6c5ce7.svg)](#harness-ai--the-core-idea)
[![Docs](https://img.shields.io/badge/docs-canonical-green.svg)](#documentation-index)
[![LLM context](https://img.shields.io/badge/llms.txt-available-orange.svg)](llms.txt)

**Agent OS and Harness AI runtime for building, orchestrating, experimenting with, and validating specialized AI agents.**

Intergrax is a **production-grade Harness AI platform** — not a single chatbot or domain agent, but the **runtime environment** that lets teams define agent capabilities as reusable modules, compose multi-agent graphs, enforce policy and observability consistently, and graduate ideas from a fast **laboratory** to a governed **production harness**.

**Strategic goal:** build a modern Agent Operating System aligned with practices used by leading agent platforms (Cursor, Claude Code, Codex-class harnesses, Viktor, Google ADK-style labs, enterprise agent runtimes). See [Development Strategy](docs/INTERGRAX_DEVELOPMENT_STRATEGY.md).

---

## Table of contents

- [What is Intergrax?](#what-is-intergrax)
- [Harness AI — the core idea](#harness-ai--the-core-idea)
- [Laboratory vs production harness](#laboratory-vs-production-harness)
- [Quick start](#quick-start)
- [Four-tier platform model](#four-tier-platform-model)
- [Capability stack](#capability-stack-integration--tool--skill--agent)
- [Nexus runtime and UAEP](#nexus-runtime-and-uaep)
- [Experimentation workflow](#experimentation-workflow)
- [Applications — isolated deployable environments](#applications--isolated-deployable-environments)
- [Platform building blocks (Tier-0)](#platform-building-blocks-tier-0)
- [Integration Library](#integration-library)
- [Tool Library](#tool-library)
- [Skill Library](#skill-library)
- [LLM adapters](#llm-adapters)
- [Modality and ML plane](#modality-and-ml-plane)
- [Memory, RAG, and context engineering](#memory-rag-and-context-engineering)
- [Governance, policy, and human-in-the-loop](#governance-policy-and-human-in-the-loop)
- [Orchestration and multi-agent graphs](#orchestration-and-multi-agent-graphs)
- [Observability and trace](#observability-and-trace)
- [Extensibility — plugin catalogs](#extensibility--plugin-catalogs)
- [Developer experience](#developer-experience)
- [Reference agents and applications](#reference-agents-and-applications)
- [Repository layout](#repository-layout)
- [Architecture maturity and audits](#architecture-maturity-and-audits)
- [Adaptive Harness Intelligence (L4)](#adaptive-harness-intelligence-l4)
- [Critic & Verification Layer (PEV Verify)](#critic--verification-layer-pev-verify)
- [Start here](#start-here)
- [Documentation index](#documentation-index)
- [Documentation update rules](#documentation-update-rules)
- [Status](#status)
- [Audience](#audience)
- [Local infrastructure](#local-infrastructure)
- [License](#license)
- [Contributing & community](#contributing--community)

---

## What is Intergrax?

Intergrax answers one question:

> **Can we rapidly create, run, and evaluate new AI agents without rebuilding infrastructure every time?**

It is designed for teams who treat **the Harness as the durable product** and agents as **replaceable execution units** — composable workers assembled from profiles, tools, skills, and policy, not one-off LLM scripts.

### What Intergrax is

| Role | Description |
|------|-------------|
| **Four-tier AI platform** | Platform → Nexus (Agent OS) → Agents → Applications |
| **Harness AI environment** | Governed runtime: policy, tools, skills, context, trace, composable agents |
| **Agent experimentation laboratory** | Fast hypothesis validation — idea to first traced run in **under one hour** |
| **Agent Operating System** | Nexus orchestrates lifecycle, graphs, HITL, retries, and ToolRuntime |
| **Capability execution platform** | Integration → Tool → Skill → Agent stack with full extensibility |
| **Integration bridge** | Connect agentic work to real organizational systems (Jira, Slack, PostgreSQL, …) |

### What Intergrax is not

Intergrax is **not** a chatbot, a prompt collection, a single agent, a workflow builder, a marketplace, or a finished multi-tenant SaaS product today. It **learns from** Cursor, Viktor, NotebookLM, and modern agent runtimes — the **laboratory** optimizes controlled experimentation; the **harness** optimizes governed, repeatable production agent work on the **same codebase**.

Canon: [architecture §3–§4](docs/intergrax_runtime_architecture.md) · [ideal Harness AI model](docs/IDEAL_HARNESS_AI_ARCHITECTURE.md)

### Documentation boundary

Repository **`docs/`** architecture canon and **[implementation plan](docs/INTERGRAX_IMPLEMENTATION_PLAN.md)** describe the **Harness AI / Agent OS platform** — infrastructure to run and govern agent environments.

They **do not** replace per-product or per-agent documentation:

| Layer | Platform docs (`docs/`) | Own docs (mandatory for domain work) |
|-------|-------------------------|--------------------------------------|
| Tier-3 business environment | Wiring patterns, `applications/USAGE.md` | `applications/<product>/ARCHITECTURE.md`, `IMPLEMENTATION_PLAN.md` |
| Tier-2 business agent | [Agent creation guide](docs/guides/AGENT_CREATION_GUIDE.md), scaffold workflow | `agents/<name>/ARCHITECTURE.md`, agent README, local plan |

Examples: [Intergrax Assistant](applications/intergrax_assistant_application/ARCHITECTURE.md) (harness chat lab), [Local Knowledge Workspace](applications/local_workspace_application/ARCHITECTURE.md), and [Dispute Simulation Workspace](applications/dispute_sim_application/ARCHITECTURE.md) are documented under their application folders, not in the platform implementation plan.

Details: [Strategy §Documentation boundary](docs/INTERGRAX_DEVELOPMENT_STRATEGY.md#documentation-boundary) · [Architecture hub](docs/intergrax_runtime_architecture.md#documentation-boundary-platform-vs-product) · [Plan §4.0a](docs/INTERGRAX_IMPLEMENTATION_PLAN.md#40a-implementation-scope-split-infrastructure-vs-business)

---

## Harness AI — the core idea

The main architectural thesis:

> **The future value is not in building one agent. The value is in building the runtime that allows many agents to be built, tested, and orchestrated quickly.**

Intergrax implements the industry Harness AI chain:

```text
Harness  →  Runtime (Nexus)  →  Agents  →  Applications  →  Products
```

| Term | Intergrax implementation |
|------|---------------------------|
| **Harness** | Tier-1 Nexus + Tier-0 catalogs + Tier-3 application wiring (policy, tools, integrations, trace) |
| **Scaffold** | `python -m intergrax.scaffold` — `new-agent`, `new-application`, `new-stack`, `new-skill` |
| **Runnable agent instance** | Harness + agent + `LLMProfile` + resolved `skill_ids` / `allowed_tools` + `RuntimePolicyBundle` |
| **Tool** | Atomic `ToolContract` — LLM/MCP invocable operation |
| **Skill** | Composable `SkillManifest` — tools + prompts + policy fragment (not an LLM function) |
| **Subagent** | Graph delegation via `ExecutionGraph` — not a nested OS |
| **Policy** | `PolicyEngine`, budgets, HITL, `RuntimePolicyBundle` |

**Agent composition flow:**

```text
Harness (Nexus + app wiring)
    → runs Tier-2 Agent
        → composes SkillManifest(s)  →  resolves tool_ids, prompts, policy
        → AgentEngine / UAEP steps
        → ToolRuntime.invoke(tool_id)  →  Integration adapters
        → LLM adapters (per step / planner)
        → Modality tools (vision, speech, ML)  →  Plane C registry
```

**Single vocabulary source for Harness terms:** [architecture/PLATFORM_FOUNDATION.md §5.3](docs/architecture/PLATFORM_FOUNDATION.md)

**Target reference model:** [IDEAL_HARNESS_AI_ARCHITECTURE.md](docs/IDEAL_HARNESS_AI_ARCHITECTURE.md) — policy-first, composable-by-default, trace-everything, human-governed autonomy, progressive extensibility.

**L4 differentiation (Done):** [Adaptive Harness Intelligence Architecture](docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) — governed closed-loop harness improvement (observe → propose → gate → apply → verify); lab observe enabled by default; canon [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md).

---

## Laboratory vs production harness

Intergrax deliberately supports **two modes on one codebase**:

| Mode | Purpose | Primary metric |
|------|---------|----------------|
| **Laboratory** | Fast hypothesis validation; discard failed ideas quickly | Time from idea → first traced run **< 1 hour** |
| **Production harness** | Reliable Agent OS for business agents at organizational scale | Reference agents + stable integration paths + ops SLOs |

**Laboratory is the adoption phase; production harness is the strategic destination.**

- New capabilities **start** in the lab workflow (`lab_application`, pytest, debug API).
- Capabilities that ship to users **graduate** through maturity gates (Phase L → Q/Q+/R → S → U → V).
- Business agents (Phase K) require **explicit product prioritization** — harness platform work is **Done**; default queue is [maintenance only](docs/INTERGRAX_IMPLEMENTATION_PLAN.md#61-harness-platform-maintenance-default--band-1).

Strategy: [INTERGRAX_DEVELOPMENT_STRATEGY.md](docs/INTERGRAX_DEVELOPMENT_STRATEGY.md) · Lab stack: [guides/HARNESS_ENVIRONMENT.md](docs/guides/HARNESS_ENVIRONMENT.md)

---

## Quick start

### Prerequisites

- Python 3.12, [`uv`](https://github.com/astral-sh/uv) package manager
- Repository clone with `agents/` and `applications/` on `pythonpath` (configured in `pyproject.toml`)

### Verify platform health

```bash
uv run pytest tests/acceptance/agent_os -m agent_os -q
uv run pytest -m gate -q
python scripts/check_harness_no_getattr.py
```

### Scaffold and run your first agent

```bash
# 1. Create agent skeleton (Tier-2)
python -m intergrax.scaffold new-agent my_agent --capability domain.action

# 2. Implement domain logic in agents/my_agent/ (steps, prompts, contract)

# 3. Run tests
uv run pytest agents/my_agent/tests/ -q

# 4. Run via shared lab (Tier-3)
uv run uvicorn lab_application.host.main:app --host 127.0.0.1 --port 8090
# POST /v1/lab/run  →  GET /debug/tasks/{id}/trace
```

**Full workflow:** [guides/AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md) — scaffold → register → run → inspect → evaluate → keep · improve · pause · delete.

**CLI shortcuts (Phase DX / AA):**

```bash
uv run intergrax run <module>:app
uv run intergrax doctor
uv run python -m intergrax.scaffold new-stack <name>   # agent + application bundle
```

---

## Four-tier platform model

Intergrax is organized as a **four-tier stack**. Higher tiers compose lower tiers; they do not reimplement platform infrastructure.

```text
Tier-3  Applications     →  deployable products (legal API, lab host, research service)
Tier-2  Agents           →  specialized capability modules (LegalAgent, ResearchAgent)
Tier-1  Nexus Runtime    →  Agent OS (NexusLoop, AgentEngine, UAEP, governance)
Tier-0  Platform         →  universal building blocks (integrations, tools, skills, LLM, RAG)
```

| Tier | Role | Repository path |
|------|------|-----------------|
| **Tier-0 — Platform** | Integrations, tools, skills, LLM adapters, RAG, memory, logging, trace backends | `intergrax/` (outside Nexus orchestration) |
| **Tier-1 — Nexus Runtime** | Task lifecycle, `NexusLoop`, `AgentEngine`, UAEP, execution graph, governance, event bus | `intergrax/runtime/` |
| **Tier-2 — Agents** | Domain logic: contracts, pipelines, prompts, agent-local governance | `agents/` |
| **Tier-3 — Applications** | Isolated deployable environments — compose agents into products | `applications/` |

### Dependency rules (mandatory)

```text
intergrax/       MUST NOT import from agents/ or applications/
agents/          MUST NOT import from applications/
applications/    MAY import from agents/ and intergrax/
```

Agents **must not** import vendor SDKs or integration slugs directly — they consume Tier-0 through Nexus policy and `ToolRuntime`. This separation keeps the **Harness stable** while agents and product surfaces evolve independently.

**Reuse rule:** Tier-1/2/3 work is **composition and wiring** of existing Tier-0 modules — not parallel universal mechanisms. See [architecture/PLATFORM_FOUNDATION.md §5.2](docs/architecture/PLATFORM_FOUNDATION.md).

---

## Capability stack (Integration → Tool → Skill → Agent)

Within Tier-0 and Tier-2, Intergrax uses a **four-layer capability model** (Harness AI alignment):

| Layer | What it is | Invoked by LLM? | Example |
|-------|------------|-----------------|---------|
| **Integration** | Swappable backend contract (no LLM schema) | No | PostgreSQL, Bing, Jira REST |
| **Tool** | Single atomic operation exposed to LLM / MCP | **Yes** | `rag.retrieve`, `jira.search_tasks` |
| **Skill** | Reusable pack: `tool_ids` + prompts + policy fragment | **No** | `legal.contract_review`, `harness.tool_smoke` |
| **Agent** | Domain module: UAEP steps, contract, `skill_ids[]` | — | `LegalAgent` in `agents/legal/` |

**Skills are not tools** — the model still calls tools; the runtime resolves skills into allow-lists and instructions before the run. External skill formats (e.g. Cursor `SKILL.md`) attach only after validation to `SkillManifest`.

```text
Integration  →  Tool  →  Skill  →  Agent  →  Nexus (Harness)  →  Application wiring
```

Catalogs: [architecture/INTEGRATIONS.md](docs/architecture/INTEGRATIONS.md) · [architecture/TOOLS.md](docs/architecture/TOOLS.md) · [architecture/SKILLS.md](docs/architecture/SKILLS.md) · Architecture hub [intergrax_runtime_architecture.md](docs/intergrax_runtime_architecture.md)

---

## Nexus runtime and UAEP

**Nexus** (Tier-1) is the Agent Operating System — it orchestrates agents the way an OS orchestrates applications. Agents **run inside** Nexus; they do not replace it.

### Core components

| Component | Role |
|-----------|------|
| **NexusLoop** | Task intake, classification, planning, lifecycle (`Task` → `RUNNING` → `COMPLETED` / `WAITING_FOR_HUMAN`) |
| **AgentRegistry** | Agent registration, capability routing, skill/tool resolution |
| **AgentEngine** | Bridge Nexus → agent local UAEP loop |
| **ExecutionGraph** | Multi-agent workflows (e.g. Research → Summary) |
| **ToolRuntime** | Unified tool gateway — `ToolRequest` / policy / trace / idempotency (§42.12) |
| **PolicyEngine** | Pre-run, pre-tool, post-tool governance hooks |
| **ContextManager** | Context assembly, budget trimming, memory views (§28.1) |
| **Adaptive Control Plane** | L4 closed-loop harness improvement — signal collection, proposals, governance, canary, verification (`intergrax/runtime/adaptive/`) |

Orchestration modules live under `intergrax/runtime/nexus/orchestration/` (`intake_runner`, `planning_runner`, `graph_runner`, `hitl_runner`, `task_events`, `lifecycle_bridge`, …).

### UAEP — Unified Agent Execution Protocol

Every Tier-2 agent executes through **UAEP**:

```text
get_steps  →  run_step  →  decide_after_step
```

Orchestrated by `AgentEngine` inside `NexusLoop`. The **Unified Execution Runtime Specification** (architecture §42) defines canonical contracts for events, hooks, lifecycle, decisions, interrupts, policy, and governance. All agent implementations **must** conform to §42.

**Registration rule:** new agents integrate through `AgentRegistry.register()` — never by editing `NexusLoop` or task lifecycle code.

Canon: [architecture/UNIFIED_EXECUTION_RUNTIME.md](docs/architecture/UNIFIED_EXECUTION_RUNTIME.md) · **End-to-end flow (diagrams, edge cases):** [architecture/NEXUS_EXECUTION_FLOW.md](docs/architecture/NEXUS_EXECUTION_FLOW.md) · Author guide: [guides/AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md) · Orchestration control plane: [Appendix I](docs/guides/AGENT_CREATION_GUIDE.md#appendix-i--orchestration-control-plane)

Legacy Supervisor / LangGraph orchestration is **deprecated** and **optional** (`[langgraph-legacy]` extra only). Production paths use the Nexus runtime model.

---

## Experimentation workflow

The intended Harness AI loop:

```text
new idea
  → define agent capability (AgentContract + capability id)
  → scaffold agent under agents/          (< 1 hour target)
  → implement domain logic (steps, prompts)
  → register in AgentRegistry
  → wire integrations, tools, skills (Tier-3 or lab)
  → run via NexusLoop (Task → AgentEngine → UAEP)
  → inspect traces, cost, quality, failures
  → record experiment decision
  → keep · improve · pause · delete
```

### Scaffold commands

| Scaffold | Creates | Command |
|----------|---------|---------|
| Agent | `agents/<name>/` — UAEP, tests, notebook | `python -m intergrax.scaffold new-agent <name> --capability <domain>.<action>` |
| Application | `applications/<app>/` — manifest, host, `.env.example`, docker | `python -m intergrax.scaffold new-application <name> --profile lab\|product` |
| Skill | `intergrax/skills/providers/<domain>/` — manifest, prompts | `python -m intergrax.scaffold new-skill <skill_id>` |
| Full stack | Agent + application bundle | `python -m intergrax.scaffold new-stack <name>` |

### Inspect and evaluate

```bash
# Debug CLI
python -m intergrax.debug

# Lab HTTP debug API
GET /debug/tasks/{id}/trace?include_runtime=true
GET /debug/tasks/{id}/metrics
GET /debug/tasks/{id}/events
```

Experiments, shadow workspaces, and sandbox isolation: [guides/AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md) Appendices A–B.

Regression gate:

```bash
uv run pytest -m gate -q
```

---

## Applications — isolated deployable environments

**Applications** turn reusable **agent capabilities** into **separate, deployable products**. An agent answers *what the system can do*. An application answers *how that capability is hosted, configured, and delivered* — as its own API, with its own env, Docker image, and integration profile.

Think of Tier-2 agents as **specialized modules** and Tier-3 applications as **ready-made shells** — like shipping the same engine in different vehicles: a lab van, a legal product API, or a research service.

### What an application is

| Property | Meaning |
|----------|---------|
| **Isolated environment** | Own `.env` / `.env.example`, settings, HTTP routes, optional `docker/` |
| **Composition only** | Wires agents + integrations; **no** domain logic (stays in `agents/<name>/`) |
| **Deployable unit** | `uvicorn` locally; `applications/<app>/docker/build-docker.sh` for image build |
| **Template-born** | Scaffold `new-application` — same ergonomics as `new-agent` |
| **Reusable agents** | Same `LegalAgent` can power different applications with different config |

### Concept → runtime

```text
agents/legal/              Tier-2  — LegalAgent, pipeline, prompts
agents/research/           Tier-2  — ResearchAgent, SummaryAgent
        │
        │  AgentBinding.mount(AgentClass, factory=...)
        ▼
applications/legal_application/   Tier-3  — manifest, host, .env, Docker
applications/lab_application/     Tier-3  — universal lab + debug API
        │
        ▼
NexusLoop + IntegrationProfile + FastAPI  →  HTTP / Docker in production
```

### Applications vs agents vs shared lab

| | **Agent** (`agents/`) | **Shared lab** (`lab_application`) | **Dedicated application** (`applications/<app>/`) |
|--|------------------------|-------------------------------------|---------------------------------------------------|
| Purpose | Reusable capability | Experiment many agents in one debug surface | One product / POC with own deployment |
| Domain logic | Yes | No | No |
| Own `.env` / Docker | No | Partial (lab defaults) | Yes — full isolation |
| Typical use | Build & test capability | Quick HTTP + `/debug/*` | Ship legal API, concept lab, customer host |

### Available environments and agents

**Full index:** [`agents/README.md`](agents/README.md) (Tier-2 roster) · [`applications/README.md`](applications/README.md) (Tier-3 hosts)

| Application | Port | Agents | Role |
|-------------|------|--------|------|
| [`lab_application/`](applications/lab_application/) | 8090 | Echo, SignoffProbe, Legal, Research, … | Universal lab + debug trace API |
| [`poc_template_application/`](applications/poc_template_application/) | 8095 | Echo | Canonical Tier-3 scaffold reference |
| [`legal_application/`](applications/legal_application/) | 8000 | LegalAgent | Contract review product API |
| [`research_application/`](applications/research_application/) | 8010 | ResearchAgent, SummaryAgent | Research → summarize pipeline |
| [`local_workspace_application/`](applications/local_workspace_application/) | 8020 | LocalIndexer, LocalSearch, LocalSynthesizer | **LKW** — local knowledge workspace |
| [`dispute_sim_application/`](applications/dispute_sim_application/) | 8025 | DisputeIntake, DisputeAnalyst, DisputeStrategist, DisputeScenario | **DSW** — dispute simulation workspace |
| [`intergrax_assistant_application/`](applications/intergrax_assistant_application/) | 8096 | IntergraxAssistant (+ optional Legal, Research, …) | **IAA** — harness chat lab, swappable LLM |

### Example applications in this repository

| Application | What it demonstrates |
|-------------|-------------------|
| [`poc_template_application/`](applications/poc_template_application/) | **Canonical Tier-3 shell** — H-APP + `build_harness_host_runtime` |
| [`lab_application/`](applications/lab_application/) | Universal lab — multiple agents, debug API, `IntegrationProfile.lab_stack()` |
| [`legal_application/`](applications/legal_application/) | Product host — scaffold `LegalAgent`, auth, FastAPI core |
| [`research_application/`](applications/research_application/) | Multi-agent HTTP host — research + summary agents |
| [`local_workspace_application/`](applications/local_workspace_application/) | **Local Knowledge Workspace (LKW)** — local file index, search, synthesis ([ARCHITECTURE.md](applications/local_workspace_application/ARCHITECTURE.md)) |
| [`dispute_sim_application/`](applications/dispute_sim_application/) | **Dispute Simulation Workspace (DSW)** — case intake, argument analysis, strategy, court simulation ([ARCHITECTURE.md](applications/dispute_sim_application/ARCHITECTURE.md)) |
| [`intergrax_assistant_application/`](applications/intergrax_assistant_application/) | **Intergrax Assistant (IAA)** — ChatGPT-shaped harness lab, local/cloud LLM swap, hub + specialist delegation ([ARCHITECTURE.md](applications/intergrax_assistant_application/ARCHITECTURE.md)) |

**Usage guides:**

- Composition engine: [`intergrax/applications/USAGE.md`](intergrax/applications/USAGE.md)
- Application folder layout: [`applications/USAGE.md`](applications/USAGE.md)
- Tier-3 environment (Appendix F): [guides/AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md#appendix-f--tier-3-application-environment)
- Architecture rules: [§7.4.8–§7.4.10](docs/intergrax_runtime_architecture.md)

---

## Platform building blocks (Tier-0)

Shared infrastructure used by all agents through **one canonical path** per concern:

| Concern | Module | Documentation |
|---------|--------|---------------|
| **LLM adapters** | `intergrax/llm_adapters/` — 19 providers | [architecture/LLM_ADAPTERS.md](docs/architecture/LLM_ADAPTERS.md) |
| **Integration Library** | `intergrax/integrations/` — 167 providers | [architecture/INTEGRATIONS.md](docs/architecture/INTEGRATIONS.md) |
| **Tool Library** | `intergrax/tools/` — atomic LLM/MCP operations | [architecture/TOOLS.md](docs/architecture/TOOLS.md) |
| **Skill Library** | `intergrax/skills/` — composable capability packs | [architecture/SKILLS.md](docs/architecture/SKILLS.md) |
| **RAG** | `intergrax/rag/` — embeddings, vector stores, ingest | Architecture [§7.1.2](docs/intergrax_runtime_architecture.md) |
| **Memory** | `intergrax/memory/` — STM/LTM, session, hooks | Phase MEM **Done** (48/48) |
| **Modality inference** | vision, speech, classical ML tools | [architecture/MODALITY.md](docs/architecture/MODALITY.md) |
| **FastAPI core** | `intergrax/fastapi_core/` — shared API primitives for Tier-3 hosts | Architecture [§7.4](docs/intergrax_runtime_architecture.md) |
| **MCP export** | Tool catalog → MCP clients | [architecture/TOOLS.md](docs/architecture/TOOLS.md) · [intergrax/tools/USAGE.md](intergrax/tools/USAGE.md) |

**Forbidden in agents:** direct vendor SDKs, duplicate tool registries, boolean `use_rag` / `use_websearch` flags (deprecated — use catalog `tool_ids`).

---

## Integration Library

External systems are **pluggable, environment-specific infrastructure** — not hard-coded dependencies inside agents.

| Property | Benefit |
|----------|---------|
| **Universal contracts** | Agents depend on category protocols (`RelationalStore`, `VectorStore`, `MessageBus`, …), not vendor SDKs |
| **Modular providers** | Each backend is `providers/<category>/<slug>/` — swap SQLite for PostgreSQL by config |
| **Portable profiles** | Same agent runs in lab (`IntegrationProfile.lab()`), customer VPC, or multi-cloud |
| **Safe boundaries** | Vendor SDKs live only in provider `opens.py` — never in `agents/` |

**167 providers** registered — relational stores, document DBs, Redis/Kafka/RabbitMQ/SQS, S3/Azure/GCS, Pinecone/Qdrant/Chroma, web search, Slack/Teams/SMTP, Jira/GitHub/Linear, Confluence/Notion, observability backends, Playwright, AWS/Azure/GCP facades, CI/CD (GitHub Actions, GitLab CI, Argo CD, …), security scanners, sandbox hosts, identity providers, speech, workflow orchestrators, CRM.

```python
from intergrax.integrations.registry.profile import IntegrationProfile

profile = IntegrationProfile(
    relational_store="postgresql",
    vector_store="qdrant",
    notification_channel="slack",
)
# Agents receive resolved contract instances — no boto3, no pymongo in agent code
```

**Full catalog:** [docs/architecture/INTEGRATIONS.md](docs/architecture/INTEGRATIONS.md) — per-slug `USAGE.md` at `providers/<category>/<slug>/USAGE.md`

Control plane: [AGENT_CREATION_GUIDE Appendix K](docs/guides/AGENT_CREATION_GUIDE.md#appendix-k--integration--rag-control-plane) · Audit layer 13: [INTEGRAX_HARNESS_AUDIT_MAP.md §13](docs/INTEGRAX_HARNESS_AUDIT_MAP.md)

---

## Tool Library

Agents invoke **tools** — named, schema-defined operations for LLM planners and MCP clients. Tools compose the Integration Library underneath.

| Property | Benefit |
|----------|---------|
| **LLM-first contracts** | `tool_id`, description, JSON Schema — native tool-calling + MCP |
| **Composable semantics** | `jira.search_tasks(project, status)` builds JQL internally |
| **Unified execution** | All invocations through `ToolRuntime` — policy, trace, idempotency |
| **Dual export** | Same entry → OpenAI function schema, MCP tool, UAEP `ToolRequest` |
| **Unified model** | RAG and web search are catalog tools (`rag.retrieve`, `websearch.query`) |

```python
from intergrax.tools.providers.jira.service import JIRA_SEARCH_TASKS_TOOL_ID
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID

AgentContract(
    id="pm",
    allowed_tools=[JIRA_SEARCH_TASKS_TOOL_ID, RAG_RETRIEVE_TOOL_ID],
)
```

**Full catalog:** [docs/architecture/TOOLS.md](docs/architecture/TOOLS.md) · **Wiring guide:** [intergrax/tools/USAGE.md](intergrax/tools/USAGE.md)

Control plane: [Appendix J](docs/guides/AGENT_CREATION_GUIDE.md#appendix-j--tools--skills-control-plane) · Audit layer 11: [AUDIT_MAP §11](docs/INTEGRAX_HARNESS_AUDIT_MAP.md)

---

## Skill Library

**Skills** sit between tools and agents — reusable packs for business or functional goals without being a single LLM function call.

| Property | Benefit |
|----------|---------|
| **`SkillManifest`** | Versioned `skill_id`, `tool_ids`, prompt refs, optional policy fragment |
| **Composition** | Agents declare `skill_ids` on `AgentContract`; runtime merges into `allowed_tools` + context |
| **External skills** | Importers (Cursor `SKILL.md`) → validated manifest |
| **No tool confusion** | Skills never register as `ToolContract` |

Platform harness skills in lab: `harness.tool_smoke`, `harness.context_demo`, `harness.trace_read`, `harness.policy_smoke`, `harness.stack_demo` — see [guides/HARNESS_ENVIRONMENT.md](docs/guides/HARNESS_ENVIRONMENT.md).

**Catalog:** [architecture/SKILLS.md](docs/architecture/SKILLS.md) · Architecture [§7.1.8](docs/intergrax_runtime_architecture.md) · Audit layer 12: [AUDIT_MAP §12](docs/INTEGRAX_HARNESS_AUDIT_MAP.md)

---

## LLM adapters

Language models are accessed through **`LLMAdapter`** — not vendor SDKs directly. Nineteen providers: OpenAI, Claude, Gemini, Ollama, Azure, Mistral, Bedrock, Groq, vLLM, Together, Fireworks, OpenRouter, DeepSeek, xAI, llama.cpp, Cohere, and more.

| Property | Benefit |
|----------|---------|
| **Typed response envelope** | `LLMAdapterResponse` — `content`, `tool_calls`, `usage`, `finish_reason`, `refusal` (Phase M-LLM-R **Done**) |
| **Unified contract** | `generate_messages`, `stream_messages`, `generate_with_tools`, `generate_structured` |
| **Lazy registry** | `LLMAdapterRegistry.create("openai")` loads only needed provider |
| **Tool-ready** | Native tool loops for `CatalogToolPlanner` |
| **Usage tracking** | Per-`run_id` token and latency stats; trace bridge from response envelope |
| **Outside integrations** | LLM providers are **not** Integration Library slugs |

```python
from intergrax.llm_adapters import LLMAdapterResponse
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

llm = LLMAdapterRegistry.create(LLMProvider.OPENAI, model="gpt-4o-mini")
response: LLMAdapterResponse = llm.generate_messages([...], run_id=task_id)
text = response.content
```

**Full catalog:** [architecture/LLM_ADAPTERS.md](docs/architecture/LLM_ADAPTERS.md) · Phase [M-LLM-R](docs/plan/phases/misc-phases.md) · Audit layer 6: [AUDIT_MAP §6](docs/INTEGRAX_HARNESS_AUDIT_MAP.md)

---

## Modality and ML plane

Harness AI at scale needs more than text LLMs — images, audio, CV detectors (YOLO, ONNX), embeddings, and batch classifiers. Intergrax organizes modality in **three planes**:

| Plane | Purpose | Module |
|-------|---------|--------|
| **A — Generative** | Dialog, reasoning, native multimodal LLM APIs | `intergrax/llm_adapters/` |
| **B — Ingest** | Files/streams → text or embeddings for knowledge | RAG ingest, document parsers, transcription |
| **C — Dedicated inference** | Deterministic CV/ML, TTS, served models | `vision.detect`, `ml.predict`, `speech.synthesize` tools |

**Routing discipline:**

- Regulated detection (boxes, masks, scores) → Plane C tools
- Semantic Q&A over an image in conversation → Plane A when policy allows
- Archive indexing → Plane B only

Agents invoke modalities through **tools** and profiles only — never vendor SDKs (`ultralytics`, `torch`, …) in `agents/`.

**Full index:** [architecture/MODALITY.md](docs/architecture/MODALITY.md) · Phase W-ML **Done** · Audit layer 29: [AUDIT_MAP §29](docs/INTEGRAX_HARNESS_AUDIT_MAP.md)

---

## Memory, RAG, and context engineering

| Layer | Components | Status |
|-------|------------|--------|
| **RAG** | `RetrievalService`, `RagProfile`, `IngestPipeline`, hybrid sparse/dense | Phase M-RAG **Done** |
| **Memory** | STM/LTM, session storage, hooks, task memory | Phase MEM **Done** (48/48) · depth → [MEM-DEPTH](docs/plan/phases/rag-context-memory.md) |
| **Context engineering** | `ContextManager`, `ContextBudgetPolicy`, `CONTEXT_ASSEMBLED` / `CONTEXT_TRIMMED` events | Phase CTX **Done** |
| **Prompt registry** | Versioned prompts, assembly, policy overlays | Phase PE **Done** |

**Architecture (deep dive):** [architecture/MEMORY.md](docs/architecture/MEMORY.md) — stores, lifecycle, context compiler, strategy matrix, flows · [MEMORY.md](docs/architecture/MEMORY.md)

Control planes: [Appendix G](docs/guides/AGENT_CREATION_GUIDE.md#appendix-g--memory--rag-naming-phase-q) · [Appendix L](docs/guides/AGENT_CREATION_GUIDE.md#appendix-l--context-engineering-control-plane) · [Appendix M](docs/guides/AGENT_CREATION_GUIDE.md#appendix-m--prompt-registry-control-plane)

Audit layers: [§14 RAG](docs/INTEGRAX_HARNESS_AUDIT_MAP.md) · [§15 Memory](docs/INTEGRAX_HARNESS_AUDIT_MAP.md) · [§16 Context](docs/INTEGRAX_HARNESS_AUDIT_MAP.md) · [§17 Prompt](docs/INTEGRAX_HARNESS_AUDIT_MAP.md)

---

## Governance, policy, and human-in-the-loop

Intergrax is **policy-first** — nothing executes without permission, constraint checks, and traceability.

| Capability | Implementation |
|------------|----------------|
| **Policy engine** | `PolicyEngine`, `RuntimePolicyBundle`, pre/post tool hooks |
| **Tool access policy** | `allowed_tools` enforcement via `ToolRuntime` |
| **Budgets** | Context, cost, execution time limits |
| **Human-in-the-loop** | `WAITING_FOR_HUMAN`, approval workflows, escalation |
| **Shadow workspace** | Isolated execution for risky side effects |
| **Sandbox** | `sandbox.exec` tool with session isolation |

```text
pre-run  →  pre-tool  →  tool invoke  →  post-tool  →  post-run
                ↑                              ↑
           PolicyEngine                   HITL gates
```

Control plane: [AGENT_CREATION_GUIDE Appendix H](docs/guides/AGENT_CREATION_GUIDE.md#appendix-h--governance-policy--observability-control-plane) · [Appendix A HITL](docs/guides/AGENT_CREATION_GUIDE.md#appendix-a--human-in-the-loop) · [Appendix B shadow/sandbox](docs/guides/AGENT_CREATION_GUIDE.md#appendix-b--shadow-workspace-and-sandbox)

Audit layers: [§5 Policy](docs/INTEGRAX_HARNESS_AUDIT_MAP.md) · [§23 Security](docs/INTEGRAX_HARNESS_AUDIT_MAP.md) · [§24 Cost](docs/INTEGRAX_HARNESS_AUDIT_MAP.md)

---

## Orchestration and multi-agent graphs

Nexus supports **single-agent** and **multi-agent** execution through `ExecutionGraph`:

```text
Task intake  →  classifier  →  planner  →  graph_runner
                                              │
                    ResearchAgent  ──edge──►  SummaryAgent
```

| Feature | Description |
|---------|-------------|
| **Planner** | Goal decomposition into plans and steps |
| **Graph spec** | `ApplicationGraphSpec` — nodes, edges, parallel cap |
| **Delegation** | Child nodes as subagents — not nested OS instances |
| **HITL runner** | Pause/resume at graph boundaries |
| **Retry layers** | Architecture §31.1 — classified failure handling |

Phase ORCH **Done** (2026-06-05). Control plane: [Appendix I](docs/guides/AGENT_CREATION_GUIDE.md#appendix-i--orchestration-control-plane) · [Appendix C graphs](docs/guides/AGENT_CREATION_GUIDE.md#appendix-c--multi-agent-graphs)

Audit layers: [§7 Cognition](docs/INTEGRAX_HARNESS_AUDIT_MAP.md) · [§9 Orchestration](docs/INTEGRAX_HARNESS_AUDIT_MAP.md) · [§10 Subagents](docs/INTEGRAX_HARNESS_AUDIT_MAP.md)

---

## Observability and trace

Every decision and invocation is **traceable** — core Harness AI principle. All tiers share one **Harness Observability Spine** (event bus + typed trace + unified journal) — no per-agent or per-app telemetry forks.

| Surface | Purpose |
|---------|---------|
| **Architecture** | [architecture/OBSERVABILITY.md](docs/architecture/OBSERVABILITY.md) — spine, signal planes, extension contracts, persistence, scaling |
| **Run traces** | SQLite-persisted `TraceEvent` timeline (`DiagnosticPayload`) per run |
| **Runtime events** | Canonical `RuntimeEvent` bus (`TOOL_*`, `LLM_CALL`, `AGENT_SELECTED`, …) |
| **Unified journal** | `build_unified_run_journal()` — one chronological timeline per run |
| **Debug CLI** | `python -m intergrax.debug` |
| **Debug HTTP API** | `intergrax.debug.app` — `/debug/tasks/{id}/trace`, `/metrics`, `/events` |
| **OTLP** | OpenTelemetry export via `IntegrationProfile.harness_environment()` |
| **Prometheus** | LLM/RAG metrics — [architecture/LLM_ADAPTERS.md](docs/architecture/LLM_ADAPTERS.md) |

Lab wiring: [guides/HARNESS_ENVIRONMENT.md](docs/guides/HARNESS_ENVIRONMENT.md) · Architecture: [architecture/OBSERVABILITY.md](docs/architecture/OBSERVABILITY.md) · [ADR-OBS-001](docs/adr/ADR-OBS-001.md) · Implementation: [Phase OBS-BUS](docs/plan/phases/observability-reliability.md) · Control plane: [Appendix Q](docs/guides/AGENT_CREATION_GUIDE.md#appendix-q--observability-control-plane-closeout) (wiring) · [Appendix R](docs/guides/AGENT_CREATION_GUIDE.md#appendix-r--reliability-control-plane-closeout) (reliability) · [Appendix S](docs/guides/AGENT_CREATION_GUIDE.md#appendix-s--security-control-plane-closeout) (security) · [Appendix T](docs/guides/AGENT_CREATION_GUIDE.md#appendix-t--cost-governance-control-plane-closeout) (cost) · [Appendix U](docs/guides/AGENT_CREATION_GUIDE.md#appendix-u--evaluation-control-plane-closeout) (evaluation)

Audit layer 21: [AUDIT_MAP §21](docs/INTEGRAX_HARNESS_AUDIT_MAP.md)

---

## Extensibility — plugin catalogs

Shipped providers and **pip-installable extensions** register through the same plugin protocols (Phase P-Ext **Done**):

| Layer | Protocol | Entry point | Register |
|-------|----------|-------------|----------|
| Integration | `IntegrationPlugin` | `intergrax.integrations` | `register_integration_plugin()` |
| Tool | `ToolPlugin` | `intergrax.tools` | `register_tool_plugin()` |
| Skill | `SkillPlugin` | `intergrax.skills` | `register_skill_plugin()` |

```python
from intergrax.core.catalog_bootstrap import bootstrap_catalogs

bootstrap_catalogs(
    register_shipped=True,
    discover_entry_points=True,
    integration_plugins=(MyIntegrationPlugin,),
    tool_plugins=(MyToolPlugin,),
    skill_plugins=(MySkillPlugin,),
)
```

Tools work **standalone** for LLM/MCP. Skills compose allow-lists — they are not invokable tools.

**Author guide:** [guides/EXTENSION_AUTHOR_GUIDE.md](docs/guides/EXTENSION_AUTHOR_GUIDE.md) · Architecture [§7.1.5.1](docs/intergrax_runtime_architecture.md) · Plan [Phase P-Ext](docs/INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Developer experience

Intergrax optimizes **time-to-first-traced-run** — opinionated scaffolds, typed wiring APIs, and zero `getattr` hacks in harness paths.

| Tool | Purpose |
|------|---------|
| `intergrax.scaffold` | `new-agent`, `new-application`, `new-skill`, `new-stack` |
| `intergrax run` | Launch application ASGI host |
| `intergrax doctor` | Platform health checks |
| `pytest -m gate` | Regression gate (996 tests) |
| `check_harness_no_getattr.py` | Zero grandfathered reflection in harness paths |
| `check_harness_observability_wiring.py` | Tier-3 observability assembly validation |
| `check_llm_adapter_typed_returns.py` | LLM adapter typed-return CI guard (M-LLM-R) |

**Decision hierarchy** for all work:

```text
1. Development Strategy (strategic goal)
2. Architecture canon (living spec)
3. Implementation plan (status map)
```

Standard work cycle: ANALYSIS → ARCHITECTURE ASSESSMENT → PLAN ASSESSMENT → DOCUMENTATION → IMPLEMENTATION → VERIFICATION (gate) → CONCLUSIONS

Audit layer 27: [AUDIT_MAP §27 Developer Experience](docs/INTEGRAX_HARNESS_AUDIT_MAP.md)

---

## Reference agents and applications

**Canonical roster:** [`agents/README.md`](agents/README.md) · **Application hosts:** [`applications/README.md`](applications/README.md)

### Tier-2 agents

| Category | Agents | Capabilities | Host |
|----------|--------|--------------|------|
| **Harness / lab** | Echo, SignoffProbe | `echo.basic`, harness probe | `lab_application` |
| **Research** | Research, Summary | `research.web_search`, `research.pipeline`, `research.summarize` | `research_application` |
| **Legal review** | Legal | `legal.review` | `legal_application` |
| **LKW product** | LocalIndexer, LocalSearch, LocalSynthesizer | `local.workspace.index`, `.search`, `.synthesize` | `local_workspace_application` |
| **DSW product** | DisputeIntake, DisputeAnalyst, DisputeStrategist, DisputeScenario | `dispute.intake`, `.analyze`, `.strategy`, `.scenario` | `dispute_sim_application` |
| **IAA harness chat** | IntergraxAssistant | `platform.assist` | `intergrax_assistant_application` |
| **Harness demo** | OrganizationWorker | `org.vendor_report` | `lab_application` (optional) |
| **Deferred** | ProblemRadar | `problem_radar.scan` | — (Phase K.1) |

Agents execute through **UAEP**, orchestrated by `AgentEngine` inside `NexusLoop`. Per-agent docs: `agents/<name>/README.md` + `ARCHITECTURE.md`.

### Control plane closeouts (platform Done)

All harness control planes are **closed** — authoring maps live in AGENT_CREATION_GUIDE appendices:

| Phase | Focus | Appendix |
|-------|-------|----------|
| AS | Agent assembly | [N](docs/guides/AGENT_CREATION_GUIDE.md#appendix-n--agent-assembly-control-plane) |
| REG | Registry architecture | [O](docs/guides/AGENT_CREATION_GUIDE.md#appendix-o--registry-architecture-control-plane) |
| CG | Capability graph | [P](docs/guides/AGENT_CREATION_GUIDE.md#appendix-p--capability-graph-control-plane) |
| OBS | Observability wiring | [Q](docs/guides/AGENT_CREATION_GUIDE.md#appendix-q--observability-control-plane-closeout) |
| REL | Reliability wiring | [R](docs/guides/AGENT_CREATION_GUIDE.md#appendix-r--reliability-control-plane-closeout) |
| SEC | Security wiring | [S](docs/guides/AGENT_CREATION_GUIDE.md#appendix-s--security-control-plane-closeout) |
| COST | Cost governance | [T](docs/guides/AGENT_CREATION_GUIDE.md#appendix-t--cost-governance-control-plane-closeout) |
| EVAL | Evaluation wiring | [U](docs/guides/AGENT_CREATION_GUIDE.md#appendix-u--evaluation-control-plane-closeout) |
| CRIT-V | Critic & Verification Layer | [CVL](docs/architecture/CRITIC_VERIFICATION.md) · [Phase CRIT-V](docs/plan/phases/evaluation-adaptive-critic.md) |
| ORCH | Orchestration | [I](docs/guides/AGENT_CREATION_GUIDE.md#appendix-i--orchestration-control-plane) |
| TS | Tools & skills | [J](docs/guides/AGENT_CREATION_GUIDE.md#appendix-j--tools--skills-control-plane) |
| INT | Integrations | [K](docs/guides/AGENT_CREATION_GUIDE.md#appendix-k--integration--rag-control-plane) |
| RAG | Retrieval | [K](docs/guides/AGENT_CREATION_GUIDE.md#appendix-k--integration--rag-control-plane) |
| CTX | Context engineering | [L](docs/guides/AGENT_CREATION_GUIDE.md#appendix-l--context-engineering-control-plane) |
| PE | Prompt registry | [M](docs/guides/AGENT_CREATION_GUIDE.md#appendix-m--prompt-registry-control-plane) |
| MEM | Memory platform | Plan [Phase MEM](docs/plan/phases/rag-context-memory.md) |
| W-ADAPT | Adaptive Harness Intelligence (L4 runtime) | [AHIA](docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) · [Phase W-ADAPT](docs/plan/phases/evaluation-adaptive-critic.md) |

---

## Repository layout

```text
intergrax/              # Tier-0 platform + Tier-1 Nexus
  integrations/         # Integration Library (167 providers)
  tools/                # Tool Library + MCP export
  skills/               # Skill Library + importers
  llm_adapters/         # 19 LLM providers
  rag/                  # Retrieval, ingest, embeddings
  memory/               # Session, STM/LTM
  runtime/              # Tier-1 Nexus + L4 Adaptive Control Plane
    nexus/              # NexusLoop, AgentEngine, UAEP, governance, orchestration
    adaptive/           # L4 ACP — signals, proposals, governance, canary, verification
  applications/         # Tier-3 composition engine (manifest, wiring API)
  scaffold/             # new-agent, new-application, new-skill, new-stack
agents/                 # Tier-2 specialized agents
applications/           # Tier-3 isolated deployable environments
docs/                   # Architecture canon, plan, guides, audits
infra/                  # Local Docker compose for integration backends
notebooks/              # Runnable examples and integration demos
tests/                  # Unit, integration, acceptance (gate: pytest -m gate)
scripts/                # Harness CI checks, evidence collectors
```

---

## Architecture maturity and audits

Intergrax maintains a **layered audit model** — the platform is audited one architectural layer at a time, never declared "complete" after a shallow whole-system review.

| Document | Role |
|----------|------|
| [IDEAL_HARNESS_AI_ARCHITECTURE.md](docs/IDEAL_HARNESS_AI_ARCHITECTURE.md) | Target Harness AI reference (9 logical layers, L0–L4 maturity) |
| [architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md](docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) | **L4 Adaptive Harness Intelligence (AHI)** — closed-loop runtime spec; Phase W-ADAPT **Done** (70/70) |
| [INTEGRAX_HARNESS_AUDIT_MAP.md](docs/INTEGRAX_HARNESS_AUDIT_MAP.md) | 32 auditable layers with DoD, evidence, risk scoring |
| [guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md](docs/guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) | Cursor/agent prompt template for focused audits |

**Maturity model (L0–L4):** evolution is evidence-driven, not declaration-driven. Phase V L3/L4 gate evidence: [IMPLEMENTATION_PLAN](docs/INTERGRAX_IMPLEMENTATION_PLAN.md).

**Audit phases:**

1. **Core Harness Integrity** — tiers, UAEP, runtime, registry
2. **Capability Platform** — tools, skills, integrations, RAG, memory
3. **Production Readiness** — security, cost, SLOs, operational excellence

---

## Adaptive Harness Intelligence (L4)

Intergrax is designed to evolve beyond static harness configuration toward an **Adaptive Harness Intelligence (AHI)** model — a Tier-1 **Adaptive Control Plane** that improves quality, cost, and operational efficiency over time through **evidence-driven, policy-governed closed loops**.

This is **not classical reinforcement learning** (neural policy training, unconstrained reward maximization). It is **bounded harness adaptation**: contextual bandits, statistical gates, evaluation registry feedback, and human-governed policy learning — aligned with IDEAL §25 and the L4 maturity model.

| Concept | Description |
|---------|-------------|
| **What it does** | Observes run outcomes → proposes versioned profile changes → validates through governance → applies via shadow/canary → verifies improvement → rolls back on failure |
| **What it discovers** | Operational patterns in traces (tool/agent/HITL sequences), routing inefficiencies, cost anomalies, eval regressions |
| **What stays human-governed** | Policy-learning mutations, promotion to production traffic, skill/workflow creation from mined patterns |
| **Current status** | **Done** — Wave **W-ADAPT-0–7** complete (70/70). **Lab** collects signals in L4-O observe mode by default (`LAB_ADAPTIVE_OBSERVE=true`). |

**Primary document:** [architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md](docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) — full business case, component specification, data contracts, flow diagrams, phased roadmap (W-ADAPT-0 through W-ADAPT-7), KPIs, and L4 runtime acceptance gates.

**Implementation plan:** [INTERGRAX_IMPLEMENTATION_PLAN.md — Phase W-ADAPT](docs/plan/phases/evaluation-adaptive-critic.md) — **70/70 Done**, Band 2y closed, ADR [`ADR-ADAPT-001`](docs/adr/ADR-ADAPT-001.md).

**Canon summary:** [architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md](docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md)

**Competitive angle:** Most harnesses stop at trace + manual tuning. Intergrax targets **auditable, rollback-ready, capability-graph-aware** continuous improvement of the runtime itself — while agents remain replaceable execution units.

---

## Critic & Verification Layer (PEV Verify)

Intergrax implements **Plan–Execute–Verify (PEV)** through the **Critic & Verification Layer (CVL)** — a tier-separated stack that judges whether partial and final agent outputs are actually correct.

| Layer | Type | Owner | Examples |
|-------|------|-------|----------|
| **L0** | Deterministic | Harness + Agent contract | Schema, `NexusValidationEngine`, executable tests |
| **L1** | Semantic (opt-in) | Harness primitives + Agent rubrics | `eval.judge`, `eval.trajectory`, ValidatorAgent |
| **L2** | Authoritative | Policy + Human | HITL sign-off, compliance review |

**Tier separation:** The Harness orchestrates *how* verification runs (hooks, retry, registry, release gates). Agents supply *what* is verified (domain rubrics, ValidatorAgents). Applications configure *when* and *how strictly* (`CriticProfile`, golden datasets).

LLM-as-judge is **opt-in** — not mandatory on every run. Structural L0 validation runs on every graph node by default.

| Document | Purpose |
|----------|---------|
| [architecture/CRITIC_VERIFICATION.md](docs/architecture/CRITIC_VERIFICATION.md) | Full CVL specification — competencies, components, flows |
| [INTERGRAX_IMPLEMENTATION_PLAN — Phase CRIT-V](docs/plan/phases/evaluation-adaptive-critic.md) | Implementation register (Band 2ak, **Active**) |
| [architecture/CRITIC_VERIFICATION.md](docs/architecture/CRITIC_VERIFICATION.md) | CVL canon |
| [ADR-CRITIC-001](docs/adr/ADR-CRITIC-001.md) | Architecture decision — tier-separated verify stack |

**Builds on:** Phase EVAL (evaluation registry wiring), Phase FLOW (graph hooks), existing `NexusValidationEngine`.

---

## Start here

Task-oriented navigation for platform docs in [`docs/`](docs/). Product and agent docs live under `applications/<product>/` and `agents/<name>/` — see [Documentation boundary](#documentation-boundary).

| I want to… | Read |
|------------|------|
| Understand strategic direction | [INTERGRAX_DEVELOPMENT_STRATEGY.md](docs/INTERGRAX_DEVELOPMENT_STRATEGY.md) |
| **Platform vs product/agent docs** | [Strategy §Documentation boundary](docs/INTERGRAX_DEVELOPMENT_STRATEGY.md#documentation-boundary) · [Architecture hub](docs/intergrax_runtime_architecture.md#documentation-boundary-platform-vs-product) · [Plan §4.0a](docs/INTERGRAX_IMPLEMENTATION_PLAN.md#40a-implementation-scope-split-infrastructure-vs-business) |
| Understand the platform | Strategy doc, then implementation plan §0, then architecture canon §1–§5 |
| See what to implement next (harness) | [Phase MEM-DEPTH](docs/plan/phases/rag-context-memory.md) (Band 2am) · [§6.1am](docs/INTERGRAX_IMPLEMENTATION_PLAN.md#61am-harness-implementation-queue--memory-intelligence-depth-active) · [§6.1](docs/INTERGRAX_IMPLEMENTATION_PLAN.md#61-harness-platform-maintenance-default--band-1) gate |
| **Memory architecture (stores, lifecycle, context compiler)** | [architecture/MEMORY.md](docs/architecture/MEMORY.md) · [Phase MEM](docs/plan/phases/rag-context-memory.md) · [Phase MEM-DEPTH](docs/plan/phases/rag-context-memory.md) |
| **Observability architecture (spine, bus, extension)** | [architecture/OBSERVABILITY.md](docs/architecture/OBSERVABILITY.md) · [ADR-OBS-001](docs/adr/ADR-OBS-001.md) · [Phase OBS-BUS](docs/plan/phases/observability-reliability.md) |
| **Observability wiring (control plane)** | [AGENT_CREATION_GUIDE Appendix Q](docs/guides/AGENT_CREATION_GUIDE.md#appendix-q--observability-control-plane-closeout) · [Phase OBS](docs/plan/phases/observability-reliability.md) |
| **Critic & Verification Layer (PEV verify)** | [architecture/CRITIC_VERIFICATION.md](docs/architecture/CRITIC_VERIFICATION.md) · [Phase CRIT-V](docs/plan/phases/evaluation-adaptive-critic.md) · [ADR-CRITIC-001](docs/adr/ADR-CRITIC-001.md) |
| **Full Nexus execution flow** | [architecture/NEXUS_EXECUTION_FLOW.md](docs/architecture/NEXUS_EXECUTION_FLOW.md) |
| Create a new agent | [guides/AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md) |
| Wire integrations / tools / skills | [architecture/INTEGRATIONS.md](docs/architecture/INTEGRATIONS.md) · [architecture/TOOLS.md](docs/architecture/TOOLS.md) · [architecture/SKILLS.md](docs/architecture/SKILLS.md) |
| **All Tier-2 agents** | [agents/README.md](agents/README.md) |
| **All Tier-3 application hosts** | [applications/README.md](applications/README.md) |
| L4 adaptive harness | [architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md](docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) · [Phase W-ADAPT](docs/plan/phases/evaluation-adaptive-critic.md) |
| Harness environment (lab, OTLP) | [guides/HARNESS_ENVIRONMENT.md](docs/guides/HARNESS_ENVIRONMENT.md) |
| Governance / policy / HITL | [AGENT_CREATION_GUIDE Appendix H](docs/guides/AGENT_CREATION_GUIDE.md#appendix-h--governance-policy--observability-control-plane) |
| Orchestration / graphs | [AGENT_CREATION_GUIDE Appendix I](docs/guides/AGENT_CREATION_GUIDE.md#appendix-i--orchestration-control-plane) · [architecture/NEXUS_EXECUTION_FLOW.md](docs/architecture/NEXUS_EXECUTION_FLOW.md) |
| Harness audit (32 layers) | [INTEGRAX_HARNESS_AUDIT_MAP.md](docs/INTEGRAX_HARNESS_AUDIT_MAP.md) |
| Business backlog only | [Plan §6.3a](docs/INTERGRAX_IMPLEMENTATION_PLAN.md#63a-business-backlog-register-consolidated) |

**AI context files (repository root):** [llms.txt](llms.txt) · [llms-full.txt](llms-full.txt) · [AGENTS.md](AGENTS.md) · [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Documentation index

All platform documentation lives in [`docs/`](docs/). **One source of truth per topic.** Navigation and update rules live in **this README** (GitHub landing page).

**Scope:** These documents cover the **Intergrax Harness / Agent OS platform** only. Each **business environment** (`applications/<product>/`) and **business agent** (`agents/<name>/`) has its own architecture and implementation plan — see [Documentation boundary](#documentation-boundary) above.

### Strategy and architecture

| Document | Read when you want to… |
|----------|------------------------|
| [INTERGRAX_DEVELOPMENT_STRATEGY.md](docs/INTERGRAX_DEVELOPMENT_STRATEGY.md) | Strategic goal, decision hierarchy, lab vs production, work cycle |
| [intergrax_runtime_architecture.md](docs/intergrax_runtime_architecture.md) | **Architecture hub** — full concept map, tier model, reading order, legacy § redirects |
| [architecture/](docs/architecture/README.md) | **Decomposed architecture canon** — UAEP, orchestration, governance, per-domain contracts |
| [architecture/MEMORY.md](docs/architecture/MEMORY.md) | Memory & context — stores, lifecycle, context compiler, strategy selection, flows |
| [architecture/OBSERVABILITY.md](docs/architecture/OBSERVABILITY.md) | Harness Observability Spine — signal planes, persistence, extension contracts |
| [architecture/NEXUS_EXECUTION_FLOW.md](docs/architecture/NEXUS_EXECUTION_FLOW.md) | Nexus execution flow — diagrams, edge cases, plan traceability |
| [IDEAL_HARNESS_AI_ARCHITECTURE.md](docs/IDEAL_HARNESS_AI_ARCHITECTURE.md) | Ideal Harness AI target — evaluate implementation alignment |
| [adr/README.md](docs/adr/README.md) | Harness ADR index — architecture decision records |
| [architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md](docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) | **Adaptive Harness Intelligence (L4)** — business case, ACP architecture, W-ADAPT implementation waves |
| [architecture/CRITIC_VERIFICATION.md](docs/architecture/CRITIC_VERIFICATION.md) | **Critic & Verification Layer (CVL)** — PEV verify stack, L0/L1/L2 critics, tier competencies, CRIT-V roadmap |
| [INTERGRAX_IMPLEMENTATION_PLAN.md](docs/INTERGRAX_IMPLEMENTATION_PLAN.md) | **Plan hub** — priority ladder, §6.1 queue, phase index |
| [plan/phases/](docs/plan/phases/) | Decomposed phase registers (ORCH, MEM, CRIT-V, …) |
| [plan/appendices/](docs/plan/appendices/) | Traceability appendices A–N |

### Authoring and workflow

| Document | Read when you want to… |
|----------|------------------------|
| [guides/AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md) | **Canonical agent workflow** — scaffold → register → run → inspect (Appendices A–U) |
| [guides/EXTENSION_AUTHOR_GUIDE.md](docs/guides/EXTENSION_AUTHOR_GUIDE.md) | Tier-0 plugin catalogs — integrations, tools, skills; entry points |
| [applications/USAGE.md](applications/USAGE.md) | Application layout — env, Docker, host, deploy triad |
| [intergrax/applications/USAGE.md](intergrax/applications/USAGE.md) | Composition engine — manifest, typed bindings, registry |
| [intergrax/tools/USAGE.md](intergrax/tools/USAGE.md) | Wire catalog tools in applications and agents |

### Capability catalogs

| Document | Read when you want to… |
|----------|------------------------|
| [architecture/INTEGRATIONS.md](docs/architecture/INTEGRATIONS.md) | **167 providers** — contracts, env vars, per-slug USAGE links |
| [architecture/TOOLS.md](docs/architecture/TOOLS.md) | Tool Library — **172** catalog tools · **42** bundles; atomic LLM/MCP operations |
| [architecture/SKILLS.md](docs/architecture/SKILLS.md) | Skill Library — manifests, importers, harness presets |
| [architecture/LLM_ADAPTERS.md](docs/architecture/LLM_ADAPTERS.md) | 19 LLM providers — `LLMAdapterResponse` envelope, streaming, tools, metrics |
| [architecture/MODALITY.md](docs/architecture/MODALITY.md) | Vision, audio, ML — three modality planes |

### Operations and environment

| Document | Read when you want to… |
|----------|------------------------|
| [guides/HARNESS_ENVIRONMENT.md](docs/guides/HARNESS_ENVIRONMENT.md) | Lab harness stack, OTLP, skill/tool presets, SLO catalog |
| [infra/README.md](infra/README.md) | Local Docker infra — compose profiles, `manage.sh` |
| [infra/PORTS.md](infra/PORTS.md) | Host port matrix for integration backends |

### Audits and quality

| Document | Read when you want to… |
|----------|------------------------|
| [INTEGRAX_HARNESS_AUDIT_MAP.md](docs/INTEGRAX_HARNESS_AUDIT_MAP.md) | Layer-by-layer audit map (32 layers) |
| [guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md](docs/guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) | Focused audit prompt for coding agents |

### Quick navigation paths

```text
Strategic direction     →  INTERGRAX_DEVELOPMENT_STRATEGY.md
Platform architecture   →  intergrax_runtime_architecture.md (hub) + docs/architecture/
UAEP / §42 contracts    →  architecture/UNIFIED_EXECUTION_RUNTIME.md
New agent (< 1 hour)    →  guides/AGENT_CREATION_GUIDE.md
Integrations (167)      →  architecture/INTEGRATIONS.md
Tools                   →  architecture/TOOLS.md + intergrax/tools/USAGE.md
Skills                  →  architecture/SKILLS.md
LLM providers           →  architecture/LLM_ADAPTERS.md (typed response envelope)
Modality / CV / ML      →  architecture/MODALITY.md
Lab environment         →  guides/HARNESS_ENVIRONMENT.md
Agents & environments   →  agents/README.md + applications/README.md
New application         →  applications/USAGE.md + poc_template_application/
Plugin extension        →  guides/EXTENSION_AUTHOR_GUIDE.md
Ideal harness target    →  IDEAL_HARNESS_AI_ARCHITECTURE.md
L4 adaptive harness     →  architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md · canon §54
Memory (deep)           →  architecture/MEMORY.md · Phase MEM · Phase MEM-DEPTH
Observability (deep)    →  architecture/OBSERVABILITY.md · ADR-OBS-001 · Phase OBS-BUS
Nexus execution flow    →  architecture/NEXUS_EXECUTION_FLOW.md
Phase status / gates    →  INTERGRAX_IMPLEMENTATION_PLAN.md + plan/phases/
Harness audit           →  INTEGRAX_HARNESS_AUDIT_MAP.md
Governance / HITL       →  guides/AGENT_CREATION_GUIDE.md Appendix H
Orchestration / graphs  →  guides/AGENT_CREATION_GUIDE.md Appendix I
Reliability / security  →  guides/AGENT_CREATION_GUIDE.md Appendices R–S
Cost / evaluation       →  guides/AGENT_CREATION_GUIDE.md Appendices T–U
```

---

## Documentation update rules

When changing platform documentation, update the **canonical file for that topic** — do not fork parallel guides.

1. **Strategy** → `docs/INTERGRAX_DEVELOPMENT_STRATEGY.md`
2. **Architecture hub** → `docs/intergrax_runtime_architecture.md` (concept map + links)
3. **Architecture domain** → `docs/architecture/<domain>.md` (or specialized docs: MEMORY, OBS, AHI, CVL)
4. **Memory deep dive** → `docs/architecture/MEMORY.md`
5. **Observability deep dive** → `docs/architecture/OBSERVABILITY.md`
6. **Status / phases / gaps** → `docs/INTERGRAX_IMPLEMENTATION_PLAN.md` + `docs/plan/phases/`
7. **Agent workflow** → `docs/guides/AGENT_CREATION_GUIDE.md`
8. **Integration or tool catalog** → `docs/architecture/INTEGRATIONS.md` or `docs/architecture/TOOLS.md`
9. **Skills** → `docs/architecture/SKILLS.md`
10. **Modality / ML** → `docs/architecture/MODALITY.md`
11. **Harness AI terms** → `docs/architecture/PLATFORM_FOUNDATION.md` §5.3 only
12. **Nexus execution flow** → `docs/architecture/NEXUS_EXECUTION_FLOW.md`
13. **Navigation / phase focus** → **this README** (`Start here`, `Status`)
14. After each harness PR: run gate + getattr audit; update gate count in plan footer

---

## Status

**Last updated:** 2026-06-08

Intergrax is under **active development** (private R&D). The **harness platform** maintenance queue runs on every PR ([§6.1](docs/INTERGRAX_IMPLEMENTATION_PLAN.md#61-harness-platform-maintenance-default--band-1)). **Active implementation:** [Phase CRIT-V — Critic & Verification Layer](docs/plan/phases/evaluation-adaptive-critic.md) (Band 2ak). **Planned:** [Phase OBS-BUS — Unified Observability Spine](docs/plan/phases/observability-reliability.md) (Band 2al). Business agents (Phase K) are **end of plan** until explicit product prioritization ([§6.3](docs/INTERGRAX_IMPLEMENTATION_PLAN.md#63-end-of-plan--deferred-product-work-only)).

| Phase | Focus | Status |
|-------|--------|--------|
| **L** | Agent OS certification | **Done** — Appendix A 20/20 |
| **M / M-LLM / M-RAG / N / O** | Integration Library, LLM, RAG, applications, tools | **Done** |
| **M.6 P4/P5/P6** | Integration catalog expansion + harness depth | **Done** — 33/34 + 32/32 |
| **M-LLM-R** | Typed `LLMAdapterResponse` completion envelope | **Done** (39/39) |
| **Q / Q+** | Harness quality + post-audit hardening | **Done** — Appendix C, D |
| **R** | Harness AI alignment — Skill Library, context, delegation, policy | **Done (MVP)** — Appendix E |
| **S** | Harness environment GA | **Done** — [guides/HARNESS_ENVIRONMENT.md](docs/guides/HARNESS_ENVIRONMENT.md) |
| **T / U** | Harness cleanliness + production hardening | **Done** — Appendix G |
| **V** | Harness architecture hardening — capability graph, lifecycle, metrics, prompt/eval/context/security/cost | **Done** (2026-06-05) |
| **W-ML** | Model & modality plane | **Done** — [architecture/MODALITY.md](docs/architecture/MODALITY.md) |
| **W-ADAPT** | Adaptive Harness Intelligence (L4 runtime) | **Done** (70/70) — [AHIA](docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) |
| **CRIT-V** | Critic & Verification Layer (PEV verify depth) | **Active** (16/24) — [CVL](docs/architecture/CRITIC_VERIFICATION.md) |
| **OBS-BUS** | Unified Observability Spine (full mechanism) | **Planned** (0/8) — [architecture/OBSERVABILITY.md](docs/architecture/OBSERVABILITY.md) |
| **P-Ext** | Tier-0 plugin catalogs | **Done** (61/61) |
| **AA** | Agents & applications conformance | **Platform Done** |
| **MEM** | Memory platform | **Done** (48/48) |
| **ORCH / TS / INT / RAG / CTX / PE / AS / REG / CG / OBS / REL / SEC / COST / EVAL** | Control plane closeouts | **Done** |
| **FAUDIT-32** | Full 32-layer architecture audit + remediation (Band 2ad) | **Done** (23/23 + follow-up) |
| **K** | Business agents | **End of plan** — deferred |

**Regression gate:** `uv run pytest -m gate -q` — **996 passed** (2026-06-07)

**Harness CI:** `python scripts/check_harness_no_getattr.py` · `python scripts/check_llm_adapter_typed_returns.py` · `uv run python scripts/check_harness_prompt_golden_catalog.py` · `uv run python scripts/check_agents_lifecycle_metadata.py`

Full tracker: [`docs/INTERGRAX_IMPLEMENTATION_PLAN.md`](docs/INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Audience

Intergrax is for teams and developers who want to:

- build **specialized agents** on a shared Harness rather than one-off LLM scripts,
- **experiment rapidly** in a controlled laboratory with full traceability,
- orchestrate **multi-agent ecosystems** with policy, HITL, and observability,
- **extend** the platform via pip-installable integration/tool/skill plugins,
- integrate AI capabilities into business systems via **Tier-3 application hosts** — isolated, template-based environments that compose agent specializations into deployable products,
- evaluate platform maturity against an **ideal Harness AI reference model** with evidence-driven L0–L4 gates.

---

## Local infrastructure

Optional Docker backends for integration development and lab presets:

```bash
# See infra/README.md for compose profiles
cd infra && ./manage.sh up redis qdrant postgresql
```

Port matrix: [infra/PORTS.md](infra/PORTS.md)

Stable lab stack slugs: `sqlite`, `postgresql`, `redis`, `qdrant`, `slack`, `sentry`, `otel`, `lab_json`, `log` — [guides/HARNESS_ENVIRONMENT.md](docs/guides/HARNESS_ENVIRONMENT.md)

---

## License

All rights reserved © Artur Czarnecki. See [LICENSE](LICENSE).

This repository is currently in private R&D stage. Commercial licensing and partnership opportunities are available upon request.

---

## Contributing & community

| Resource | Purpose |
|----------|---------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, work cycle, PR process |
| [AGENTS.md](AGENTS.md) | Instructions for AI coding agents (Cursor, Codex, Claude Code) |
| [llms.txt](llms.txt) | Concise project map for LLM crawlers and context tools |
| [llms-full.txt](llms-full.txt) | Extended LLM context map with full doc index |
| [SECURITY.md](SECURITY.md) | Security policy and vulnerability reporting |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Community standards |
| [CITATION.cff](CITATION.cff) | Citation metadata for research and publications |

**For AI agents:** start with [AGENTS.md](AGENTS.md) → [Start here](#start-here) → relevant canon in `docs/`.

---

**Maintainer:** Artur Czarnecki  
**Repository:** [Intergrax](https://github.com/jakbuczarnecki/intergrax-ai)  
**Contact:** jakbu.czarnecki.83@gmail.com
