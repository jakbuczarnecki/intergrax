# Intergrax

[![Regression gate](https://github.com/jakbuczarnecki/intergrax/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/jakbuczarnecki/intergrax/actions/workflows/unit-tests.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Harness AI](https://img.shields.io/badge/Harness%20AI-Agent%20OS-6c5ce7.svg)](#harness-ai--the-core-idea)
[![Docs](https://img.shields.io/badge/docs-canonical-green.svg)](#documentation-index)
[![LLM context](https://img.shields.io/badge/llms.txt-available-orange.svg)](llms.txt)

**Agent OS and Harness AI runtime** for building, orchestrating, experimenting with, and validating specialized AI agents.

---

## Overview

- **What:** Intergrax is a **Harness AI platform** — the durable runtime that runs many agents, not a single chatbot or domain bot.
- **What it provides:** Nexus Agent OS, Tier-0 catalogs (**185** integrations · **190** tools · **149** skills in **41** bundles), LLM, RAG, memory, policy, trace, multi-agent graphs, and Tier-3 application hosts.
- **Who it is for:** Teams building **governed multi-agent systems** — platform engineers, agent architects, Harness AI researchers, and product teams shipping agent-backed applications.
- **Why it is different:** **The Harness is the product; agents are replaceable.** You compose capabilities from Integration → Tool → Skill → Agent, enforce policy at `ToolRuntime`, and graduate ideas from a fast **laboratory** to a governed **production harness** on one codebase.
- **Problem it solves:** Stop rebuilding infrastructure for every new agent. Target: **idea → first traced Nexus run in under one hour.**

```text
                Intergrax

          ┌──────────────────┐
          │    Harness AI    │
          └────────┬─────────┘
                   │
              Nexus Runtime
                   │
         ┌─────────┴─────────┐
         │                   │
      Agents            Applications
         │                   │
         └─────────┬─────────┘
                   │
              AI Products
```

Strategic direction: [Development Strategy](docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md) · Ideal target: [IDEAL_HARNESS_AI_ARCHITECTURE.md](docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)

---

## Why another AI framework?

Most AI projects build individual agents.

**Intergrax builds the runtime that allows many governed agents and applications to coexist on one platform.**

---

## Audience

This repository is for you if you are:

| Role | Why Intergrax |
|------|----------------|
| **AI systems architect** | Four-tier Harness AI model, policy-first execution, evidence-driven maturity (L0–L4) |
| **Agent platform engineer** | Nexus Agent OS, UAEP, `ToolRuntime`, orchestration graphs, observability spine |
| **Multi-agent runtime developer** | Delegation, subagents, parallel graphs, HITL — without nested OS forks |
| **Harness AI researcher** | Lab workflow, trace inspection, evaluation hooks, adaptive harness (L4) |
| **Product team shipping agents** | Tier-3 application shells — isolated deployable hosts composing Tier-2 agents |

**Not the primary audience:** teams looking for a finished SaaS chatbot, a prompt library, or a no-code workflow builder.

---

## Quick start

**Goal:** clone → install → verify → run → inspect.

### Prerequisites

Python 3.12 · [`uv`](https://github.com/astral-sh/uv) · Git

### 1. Install

```bash
git clone https://github.com/jakbuczarnecki/intergrax.git
cd intergrax
uv sync --extra dev
```

### 2. Verify

```bash
uv run intergrax doctor
uv run pytest -m gate -q
```

### 3. Run the lab host

```bash
uv run uvicorn lab_application.host.main:app --host 127.0.0.1 --port 8090
```

### 4. Execute and inspect

```bash
# Submit a run (Echo agent via capability routing)
curl -s -X POST http://127.0.0.1:8090/v1/lab/run \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","capability":"echo.basic"}'

# Inspect trace (replace {task_id} from response)
curl -s "http://127.0.0.1:8090/debug/tasks/{task_id}/trace?include_runtime=true"
```

**Next steps:** scaffold your own agent, register it, rerun through the lab, inspect `/debug/tasks/{id}/metrics` and `/events`.

| Command | Purpose |
|---------|---------|
| `python -m intergrax.scaffold new-agent {name} --capability domain.action` | Create Tier-2 agent skeleton |
| `python -m intergrax.scaffold new-application {name} --profile lab` | Create Tier-3 host |
| `python -m intergrax.scaffold new-stack {name}` | Agent + application bundle |
| `uv run intergrax run {module}:app` | Launch any ASGI application host |
| `python -m intergrax.debug` | Debug CLI |

**Full workflow:** [Agent Creation Guide](docs/guides/AGENT_CREATION_GUIDE.md) · **Contributing setup:** [CONTRIBUTING.md](CONTRIBUTING.md)

---

## What you can do today

| Action | How | Learn more |
|--------|-----|------------|
| **Scaffold a new agent** | `python -m intergrax.scaffold new-agent …` | [Agent Creation Guide](docs/guides/AGENT_CREATION_GUIDE.md) |
| **Build a Tier-3 application** | `new-application` / `new-stack` | [applications/USAGE.md](applications/USAGE.md) |
| **Connect integrations** | `IntegrationProfile` + Tier-3 wiring | [architecture/INTEGRATIONS.md](docs/architecture/INTEGRATIONS.md) |
| **Attach tools and skills** | `ToolProfile`, `SkillProfile`, `skill_ids` on contract | [architecture/TOOLS.md](docs/architecture/TOOLS.md) · [architecture/SKILLS.md](docs/architecture/SKILLS.md) |
| **Run through Nexus** | Lab or product host → `NexusLoop` → `AgentEngine` → UAEP | [NEXUS_EXECUTION_FLOW.md](docs/architecture/NEXUS_EXECUTION_FLOW.md) |
| **Inspect traces** | `/debug/tasks/{id}/trace`, `intergrax.debug` | [HARNESS_ENVIRONMENT.md](docs/guides/HARNESS_ENVIRONMENT.md) |
| **Evaluate execution** | Evaluation profile, online registry, CVL hooks | [CRITIC_VERIFICATION.md](docs/architecture/CRITIC_VERIFICATION.md) |
| **Extend via plugins** | `ToolPlugin`, `IntegrationPlugin`, `SkillPlugin` EPs | [EXTENSION_AUTHOR_GUIDE.md](docs/guides/EXTENSION_AUTHOR_GUIDE.md) |

Reference hosts: [`applications/README.md`](applications/README.md) · Reference agents: [`agents/README.md`](agents/README.md)

---

## Harness AI — the core idea

> **The future value is not in building one agent. The value is in building the runtime that allows many agents to be built, tested, and orchestrated quickly.**

Intergrax implements the Harness AI chain:

```text
Harness  →  Runtime (Nexus)  →  Agents  →  Applications  →  Products
```

| Term | Intergrax implementation |
|------|---------------------------|
| **Harness** | Tier-1 Nexus + Tier-0 catalogs + Tier-3 wiring (policy, tools, integrations, trace) |
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
        → Modality tools (vision, speech, ML)
```

**Vocabulary canon:** [architecture/PLATFORM_FOUNDATION.md §5.3](docs/architecture/PLATFORM_FOUNDATION.md) · **Target model:** [IDEAL_HARNESS_AI_ARCHITECTURE.md](docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)

---

## Laboratory vs production harness

Two modes on **one codebase**:

| Mode | Purpose | Primary metric |
|------|---------|----------------|
| **Laboratory** | Fast hypothesis validation | Idea → first traced run in under **1 hour** |
| **Production harness** | Governed Agent OS at organizational scale | Stable integration paths + ops SLOs |

New capabilities start in the lab (`lab_application`, pytest, debug API). Capabilities that ship to users graduate through maturity gates. Business agents (Phase K) require **explicit product prioritization** — default harness queue is [gate maintenance](docs/plan/PLATFORM_FOUNDATION.md#61-harness-platform-maintenance-default--band-1).

Details: [INTERGRAX_DEVELOPMENT_STRATEGY.md](docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md) · [HARNESS_ENVIRONMENT.md](docs/guides/HARNESS_ENVIRONMENT.md)

---

## Four-tier platform model

```text
Tier-3  Applications     →  deployable products (legal API, lab host, research service)
Tier-2  Agents           →  specialized capability modules (LegalAgent, ResearchAgent)
Tier-1  Nexus Runtime    →  Agent OS (NexusLoop, AgentEngine, UAEP, governance)
Tier-0  Platform         →  universal building blocks (integrations, tools, skills, LLM, RAG)
```

| Tier | Role | Path |
|------|------|------|
| **Tier-0 — Platform** | Integrations, tools, skills, LLM, RAG, memory | `intergrax/` (outside Nexus orchestration) |
| **Tier-1 — Nexus** | Task lifecycle, graphs, governance, event bus | `intergrax/runtime/` |
| **Tier-2 — Agents** | Domain logic: contracts, pipelines, prompts | `agents/` |
| **Tier-3 — Applications** | Isolated deployable environments | `applications/` |

**Dependency rules:**

```text
intergrax/       MUST NOT import from agents/ or applications/
agents/          MUST NOT import from applications/
applications/    MAY import from agents/ and intergrax/
```

Agents consume Tier-0 through Nexus policy and **`ToolRuntime`** — never vendor SDKs directly. Tier-1/2/3 work is **composition and wiring**, not parallel platform mechanisms.

Canon: [architecture/PLATFORM_FOUNDATION.md §5.2](docs/architecture/PLATFORM_FOUNDATION.md) · Hub: [intergrax_runtime_architecture.md](docs/intergrax_runtime_architecture.md)

---

## Capability stack (Integration → Tool → Skill → Agent)

| Layer | What it is | Invoked by LLM? | Example |
|-------|------------|-----------------|---------|
| **Integration** | Swappable backend contract | No | PostgreSQL, Bing, Jira REST |
| **Tool** | Single atomic operation | **Yes** | `rag.retrieve`, `jira.search_tasks` |
| **Skill** | Reusable pack: `tool_ids` + prompts + policy | **No** | `legal.contract_review`, `harness.tool_smoke` |
| **Agent** | Domain module: UAEP steps, `skill_ids[]` | — | `LegalAgent` in `agents/legal/` |

```text
Integration  →  Tool  →  Skill  →  Agent  →  Nexus (Harness)  →  Application wiring
```

Skills are **not** tools — the runtime resolves skills into allow-lists and instructions before the run.

Catalogs: [INTEGRATIONS.md](docs/architecture/INTEGRATIONS.md) · [TOOLS.md](docs/architecture/TOOLS.md) · [SKILLS.md](docs/architecture/SKILLS.md)

---

## Nexus runtime and UAEP

**Nexus** (Tier-1) is the Agent Operating System. Agents **run inside** Nexus; they do not replace it.

| Component | Role |
|-----------|------|
| **NexusLoop** | Task intake, classification, planning, lifecycle |
| **AgentRegistry** | Registration, capability routing, skill/tool resolution |
| **AgentEngine** | Bridge Nexus → agent UAEP loop |
| **ExecutionGraph** | Multi-agent workflows, delegation, parallel cap |
| **ToolRuntime** | Unified tool gateway — policy, trace, idempotency (§42.12) |
| **PolicyEngine** | Pre/post tool governance, budgets, HITL |
| **ContextManager** | Context assembly, budget trimming, memory views |

### UAEP — Unified Agent Execution Protocol

```text
get_steps  →  run_step  →  decide_after_step
```

Orchestrated by `AgentEngine` inside `NexusLoop`. All agents conform to the **Unified Execution Runtime Specification** (§42): events, hooks, `AgentDecision`, interrupts, policy.

**Registration rule:** integrate through `AgentRegistry.register()` — never edit `NexusLoop` for one agent.

Deep dive: [UNIFIED_EXECUTION_RUNTIME.md](docs/architecture/UNIFIED_EXECUTION_RUNTIME.md) · **End-to-end flow:** [NEXUS_EXECUTION_FLOW.md](docs/architecture/NEXUS_EXECUTION_FLOW.md) · **Orchestration strategies:** [ORCHESTRATION.md §50–§54](docs/architecture/ORCHESTRATION.md) · **Tool engine pipeline:** [TOOLS.md#tool-execution-pipeline](docs/architecture/TOOLS.md#tool-execution-pipeline)

---

## Tier-0 catalog summary

Shipped first-party catalogs (verified via `register_default_integrations(preset='full')`, `register_default_tools()`, `register_default_skills()` — **2026-06-08**).

```text
Integration  →  vendor backend (Postgres, Bing, Jira, …)
Tool         →  atomic LLM/MCP operation (rag.retrieve, websearch.query, …)
Skill        →  composable pack (tool_ids + prompts + policy fragment)
```

| Layer | Catalog size | Module | Architecture | Plan | Usage / authoring |
|-------|--------------|--------|--------------|------|-------------------|
| **Integrations** | **185** slugs · **30** contract categories (116 STABLE · 69 BETA) | [`intergrax/integrations/`](intergrax/integrations/) | [INTEGRATIONS.md](docs/architecture/INTEGRATIONS.md) | [plan/INTEGRATIONS.md](docs/plan/INTEGRATIONS.md) | Per-provider [USAGE.md](docs/architecture/INTEGRATIONS.md#implemented-providers-185) under `intergrax/integrations/providers/` |
| **Tools** | **190** `tool_id`s · **48** bundles | [`intergrax/tools/`](intergrax/tools/) | [TOOLS.md](docs/architecture/TOOLS.md) | [plan/TOOLS.md](docs/plan/TOOLS.md) | [intergrax/tools/USAGE.md](intergrax/tools/USAGE.md) |
| **Skills** | **149** `skill_id`s · **41** bundles | [`intergrax/skills/`](intergrax/skills/) | [SKILLS.md](docs/architecture/SKILLS.md) | [plan/SKILLS.md](docs/plan/SKILLS.md) | Per-skill `USAGE.md` under `intergrax/skills/providers/{bundle}/{skill_id}/` |

**Control plane (profiles, wiring, resolver):** [AGENT_CREATION_GUIDE.md Appendix J](docs/guides/AGENT_CREATION_GUIDE.md#appendix-j--tools--skills-control-plane) · **Extension plugins:** [EXTENSION_AUTHOR_GUIDE.md](docs/guides/EXTENSION_AUTHOR_GUIDE.md)

**Skill bundles (41):** `harness`, `rag`, `workspace`, `memory`, `research`, `knowledge`, `legal`, `ops`, `dev`, `browser`, `collaboration`, `data`, `platform`, `sandbox`, `hitl`, `graph`, `storage`, `message_bus`, `cache`, `eval`, `modality`, `notify`, `cost`, `identity`, `health`, `context`, `agent`, `vector_store`, `crm`, `billing`, `metrics`, `catalog`, `cloud_platform`, `code`, `filesystem`, `http`, `interaction`, `jira`, `gitlab`, `ml`, `openai` — **149** skills — full index in [SKILLS.md § First-party catalog](docs/architecture/SKILLS.md#first-party-catalog-149-skills--41-bundles).

---

## Platform capabilities

Tier-0 building blocks — one canonical path per concern. Agents use these through Nexus; they do not reimplement them.

| Concern | Scale / module | Documentation |
|---------|----------------|---------------|
| **Integrations** | **185** providers · `intergrax/integrations/` | [architecture/INTEGRATIONS.md](docs/architecture/INTEGRATIONS.md) · [plan](docs/plan/INTEGRATIONS.md) |
| **Tools** | **190** catalog tools · **48** bundles · `intergrax/tools/` | [architecture/TOOLS.md](docs/architecture/TOOLS.md) · [plan](docs/plan/TOOLS.md) · [USAGE](intergrax/tools/USAGE.md) |
| **Skills** | **149** skills · **41** bundles · `intergrax/skills/` | [architecture/SKILLS.md](docs/architecture/SKILLS.md) · [plan](docs/plan/SKILLS.md) |
| **LLM adapters** | 19 providers · typed `LLMAdapterResponse` | [architecture/LLM_ADAPTERS.md](docs/architecture/LLM_ADAPTERS.md) |
| **RAG** | Retrieval, ingest, hybrid/graph/agentic, golden eval | [architecture/RAG.md](docs/architecture/RAG.md) · [plan](docs/plan/RAG.md) |
| **Memory** | STM/LTM, context compiler, Knowledge vs LTM boundary | [architecture/MEMORY.md](docs/architecture/MEMORY.md) · [plan](docs/plan/MEMORY.md) |
| **Modality / ML** | Vision, speech, classical ML via catalog tools | [architecture/MODALITY.md](docs/architecture/MODALITY.md) |
| **Governance & HITL** | Policy bundle, budgets, shadow workspace, sandbox | [UAEP §42.11](docs/architecture/UNIFIED_EXECUTION_RUNTIME.md) · [Appendix H](docs/guides/AGENT_CREATION_GUIDE.md#appendix-h--governance-policy--observability-control-plane) |
| **LLM guardrails** | Vendor scanners via Integration `llm_guardrail` (M.12) | [INTEGRATIONS §47](docs/architecture/INTEGRATIONS.md) · [UAEP §42.11.6](docs/architecture/UNIFIED_EXECUTION_RUNTIME.md) · [ADR-GR-001](docs/adr/ADR-GR-001.md) |
| **Observability** | Event bus, trace DB, unified journal, OTLP | [architecture/OBSERVABILITY.md](docs/architecture/OBSERVABILITY.md) |
| **Plugins** | pip-installable integration/tool/skill catalogs | [EXTENSION_AUTHOR_GUIDE.md](docs/guides/EXTENSION_AUTHOR_GUIDE.md) |

**Control-plane authoring maps:** [AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md) Appendices A–U · **32-layer audit:** [INTEGRAX_HARNESS_AUDIT_MAP.md](docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md)

---

## Applications

**Applications** turn agent capabilities into **isolated, deployable products** — own env, host, Docker, integration profile. Domain logic stays in `agents/`; applications **wire only**.

```text
agents/legal/  ──mount──►  applications/legal_application/  ──►  NexusLoop + FastAPI
agents/*       ──mount──►  applications/lab_application/      ──►  universal lab + /debug/*
```

| Application | Port | Role |
|-------------|------|------|
| [`lab_application/`](applications/lab_application/) | 8090 | Universal lab + debug trace API |
| [`poc_template_application/`](applications/poc_template_application/) | 8095 | Canonical Tier-3 scaffold reference |
| [`legal_application/`](applications/legal_application/) | 8000 | Contract review product API |
| [`research_application/`](applications/research_application/) | 8010 | Research → summarize pipeline |
| [`local_workspace_application/`](applications/local_workspace_application/) | 8020 | Local Knowledge Workspace (LKW) |
| [`dispute_sim_application/`](applications/dispute_sim_application/) | 8025 | Dispute Simulation Workspace (DSW) |
| [`intergrax_assistant_application/`](applications/intergrax_assistant_application/) | 8096 | Harness chat lab (IAA) |

Full index: [`applications/README.md`](applications/README.md) · Composition engine: [`intergrax/applications/USAGE.md`](intergrax/applications/USAGE.md) · Tier-3 guide: [Appendix F](docs/guides/AGENT_CREATION_GUIDE.md#appendix-f--tier-3-application-environment)

---

## Experimentation workflow

```text
new idea  →  scaffold agent  →  implement domain logic  →  register
  →  wire tools/skills/integrations  →  run via NexusLoop  →  inspect trace
  →  keep · improve · pause · delete
```

Regression gate: `uv run pytest -m gate -q`

---

## Repository layout

```text
intergrax/              # Tier-0 platform + Tier-1 Nexus
  integrations/         # Integration Library
  tools/                # Tool Library + MCP export
  skills/               # Skill Library
  llm_adapters/         # LLM providers
  rag/ · memory/        # Retrieval and memory
  runtime/nexus/        # NexusLoop, AgentEngine, UAEP, orchestration
  runtime/adaptive/     # L4 Adaptive Control Plane
  applications/         # Tier-3 composition engine
  scaffold/             # new-agent, new-application, new-stack
agents/                 # Tier-2 specialized agents
applications/           # Tier-3 deployable hosts
docs/                   # Architecture canon (20 domain pairs) + guides
infra/                  # Local Docker compose for backends
tests/ · scripts/       # Gate tests and harness CI checks
```

---

## Documentation index

<a id="start-here"></a>

### Start here

| I want to… | Read |
|------------|------|
| Understand strategic direction | [INTERGRAX_DEVELOPMENT_STRATEGY.md](docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md) |
| Understand the platform | [intergrax_runtime_architecture.md](docs/intergrax_runtime_architecture.md) → pick a domain pair |
| See implementation status | [plan/PLATFORM_FOUNDATION.md](docs/plan/PLATFORM_FOUNDATION.md) |
| Create a new agent | [AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md) |
| Full Nexus execution flow | [NEXUS_EXECUTION_FLOW.md](docs/architecture/NEXUS_EXECUTION_FLOW.md) |
| See catalog sizes (integrations / tools / skills) | [Tier-0 catalog summary](#tier-0-catalog-summary) |
| Wire integrations / tools / skills | [INTEGRATIONS.md](docs/architecture/INTEGRATIONS.md) · [TOOLS.md](docs/architecture/TOOLS.md) · [SKILLS.md](docs/architecture/SKILLS.md) · [Appendix J](docs/guides/AGENT_CREATION_GUIDE.md#appendix-j--tools--skills-control-plane) |
| RAG engine / retrieval | [RAG.md](docs/architecture/RAG.md) · [plan/RAG.md](docs/plan/RAG.md) · [Appendix K §K.5](docs/guides/AGENT_CREATION_GUIDE.md#appendix-k--integration--rag-control-plane) |
| All agents / applications | [agents/README.md](agents/README.md) · [applications/README.md](applications/README.md) |
| Harness audit (32 layers) | [INTEGRAX_HARNESS_AUDIT_MAP.md](docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md) |
| Business backlog only | [plan/PLATFORM_FOUNDATION.md §6.3a](docs/plan/PLATFORM_FOUNDATION.md#63a-business-backlog-register-consolidated) |

**AI context:** [llms.txt](llms.txt) · [llms-full.txt](llms-full.txt) · [AGENTS.md](AGENTS.md) · [CONTRIBUTING.md](CONTRIBUTING.md)

**One source of truth per topic.** Platform docs live in [`docs/`](docs/); product and agent docs live under `applications/{product}/` and `agents/{name}/`.

### Canonical map

| Area | Links |
|------|-------|
| **Strategy & hub** | [Strategy](docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md) · [Architecture hub](docs/intergrax_runtime_architecture.md) · [Ideal model](docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) |
| **Domain canon (19 pairs)** | `docs/architecture/{DOMAIN}.md` ↔ `docs/plan/{DOMAIN}.md` — indexed in [hub](docs/intergrax_runtime_architecture.md) |
| **Execution** | [UAEP / §42](docs/architecture/UNIFIED_EXECUTION_RUNTIME.md) · [Nexus flow](docs/architecture/NEXUS_EXECUTION_FLOW.md) · [Orchestration](docs/architecture/ORCHESTRATION.md) |
| **Authoring** | [Agent guide](docs/guides/AGENT_CREATION_GUIDE.md) · [Extension guide](docs/guides/EXTENSION_AUTHOR_GUIDE.md) · [applications/USAGE.md](applications/USAGE.md) |
| **Operations** | [HARNESS_ENVIRONMENT.md](docs/guides/HARNESS_ENVIRONMENT.md) · [infra/README.md](infra/README.md) |
| **ADRs** | [docs/adr/README.md](docs/adr/README.md) |

**Documentation boundary:** platform `docs/` describe the Harness / Agent OS. Each business environment and agent maintains its own `ARCHITECTURE.md` and local plan — see [Strategy §Documentation boundary](docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md#documentation-boundary).

**Update rules:** canonical file per topic — strategy → hub → domain pair → guides. Details in [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md).

---

## Project snapshot

**Last updated:** 2026-06-08 · **Stage:** active private R&D

| Dimension | Status |
|-----------|--------|
| **Platform maturity** | Harness platform **complete** — Tier-0 catalogs, Nexus Agent OS, control-plane closeouts, L4 adaptive runtime (W-ADAPT Done) |
| **Active development** | Default queue: [§6.1 gate maintenance](docs/plan/PLATFORM_FOUNDATION.md#61-harness-platform-maintenance-default--band-1) · depth bands: [MEM-DEPTH](docs/plan/MEMORY.md), [CRIT-V](docs/plan/CRITIC_VERIFICATION.md), [OBS-BUS](docs/plan/OBSERVABILITY.md) |
| **Business agents** | Phase K — **end of plan** until explicit product prioritization ([§6.3](docs/plan/PLATFORM_FOUNDATION.md#63-end-of-plan--deferred-product-work-only)) |
| **Regression gate** | `uv run pytest -m gate -q` — CI green ([workflow badge](#intergrax)) |

**Also in the platform:**

| Capability | Doc |
|------------|-----|
| **Adaptive Harness Intelligence (L4)** | [architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md](docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) |
| **Critic & Verification (PEV)** | [architecture/CRITIC_VERIFICATION.md](docs/architecture/CRITIC_VERIFICATION.md) |
| **Reasoning & cognition** | [architecture/REASONING_AND_COGNITION.md](docs/architecture/REASONING_AND_COGNITION.md) |
| **Elastic capacity** | [architecture/ELASTIC_CAPACITY_AND_SCALING.md](docs/architecture/ELASTIC_CAPACITY_AND_SCALING.md) |

Full phase tracker: [plan/PLATFORM_FOUNDATION.md](docs/plan/PLATFORM_FOUNDATION.md) · [intergrax_runtime_architecture.md](docs/intergrax_runtime_architecture.md)

---

## Local infrastructure

Optional Docker backends for integration development:

```bash
cd infra && ./manage.sh up redis qdrant postgresql
```

[infra/README.md](infra/README.md) · [infra/PORTS.md](infra/PORTS.md) · Lab stack: [HARNESS_ENVIRONMENT.md](docs/guides/HARNESS_ENVIRONMENT.md)

---

## License

All rights reserved © Artur Czarnecki. See [LICENSE](LICENSE).

This repository is currently in private R&D stage. Commercial licensing and partnership opportunities are available upon request.

---

## Contributing & community

| Resource | Purpose |
|----------|---------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, work cycle, PR process |
| [AGENTS.md](AGENTS.md) | Instructions for AI coding agents |
| [SECURITY.md](SECURITY.md) | Security policy |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Community standards |
| [CITATION.cff](CITATION.cff) | Citation metadata |

**Maintainer:** Artur Czarnecki · **Repository:** [Intergrax](https://github.com/jakbuczarnecki/intergrax) · **Contact:** jakbu.czarnecki.83@gmail.com
