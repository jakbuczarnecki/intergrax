# Intergrax

**Agent OS and Harness AI runtime for building, orchestrating, and validating specialized AI agents.**

---

## What is Intergrax?

Intergrax is an **AI Operating System** — a controlled **Harness AI** environment for designing agentic capabilities, running them through a shared runtime, and deciding which ideas to keep, improve, or discard.

The core asset is not a single chatbot or domain agent. It is the **runtime** that lets you:

- define agent capabilities as reusable modules,
- register and compose agents in a multi-agent graph,
- enforce tools, governance, and observability consistently,
- run experiments with full traceability (CLI and HTTP debug surfaces).

Intergrax is developed as an **internal agent experimentation laboratory** on the path toward a broader agent platform. See [Documentation](#documentation) below for all canonical docs.

---

## Documentation

All platform documentation lives in [`docs/`](docs/). Canonical docs — one source of truth per topic:

| Document | Read when you want to… |
|----------|------------------------|
| [docs/README.md](docs/README.md) | Navigate docs, see current phase focus, update rules |
| [intergrax_runtime_architecture.md](docs/intergrax_runtime_architecture.md) | Full architecture canon (tiers, Nexus, UAEP §42) |
| [INTERGRAX_IMPLEMENTATION_PLAN.md](docs/INTERGRAX_IMPLEMENTATION_PLAN.md) | Phase status, gaps, priority, business-agent checklist (Appendix A) |
| [AGENT_CREATION_GUIDE.md](docs/AGENT_CREATION_GUIDE.md) | Create an agent: scaffold → register → run → inspect |
| [INTEGRATIONS.md](docs/INTEGRATIONS.md) | **Integration Library** — catalog of all **72** wired providers; each at `providers/<category>/<slug>/USAGE.md` |
| [TOOLS.md](docs/TOOLS.md) | **Tool Library** — LLM-facing agent tools (RAG, web search, Jira, sandbox, …) |
| [LLM_ADAPTERS.md](docs/LLM_ADAPTERS.md) | **LLM adapters** — OpenAI, Claude, Gemini, Ollama, Azure, Mistral, Bedrock |
| [intergrax/tools/USAGE.md](intergrax/tools/USAGE.md) | Wire catalog tools in applications and agents (quick start) |
| [intergrax/applications/USAGE.md](intergrax/applications/USAGE.md) | Tier-3 composition engine: manifest, typed bindings, registry |
| [applications/USAGE.md](applications/USAGE.md) | Application layout: env, Docker, host, run |

**Quick paths:** new agent → [AGENT_CREATION_GUIDE](docs/AGENT_CREATION_GUIDE.md) · **integrations catalog** → [INTEGRATIONS](docs/INTEGRATIONS.md) · **tools catalog** → [TOOLS](docs/TOOLS.md) · **LLM providers** → [LLM_ADAPTERS](docs/LLM_ADAPTERS.md) · **new application** → [applications/USAGE](applications/USAGE.md) · current phase → [IMPLEMENTATION_PLAN](docs/INTERGRAX_IMPLEMENTATION_PLAN.md) §1–§4 · deep architecture → [runtime_architecture](docs/intergrax_runtime_architecture.md) §1–§5

---

## Four-tier model

Intergrax is organized as a **four-tier stack**. Higher tiers compose lower tiers; they do not reimplement platform infrastructure.

| Tier | Role | Repository path |
|------|------|-----------------|
| **Tier-0 — Platform** | Universal, domain-agnostic building blocks: LLM adapters, RAG, tools, memory, logging, trace persistence | `intergrax/` (outside Nexus orchestration) |
| **Tier-1 — Nexus Runtime** | Agent OS: task lifecycle, `NexusLoop`, `AgentEngine`, UAEP, execution graph, governance, event bus | `intergrax/runtime/` |
| **Tier-2 — Agents** | Specialized capability modules: contracts, domain logic, pipelines, agent-local governance | `agents/` |
| **Tier-3 — Applications** | Isolated, deployable environments — compose agents into separate products (see below) | `applications/` |

**Dependency rules:**

- `intergrax/` must not import from `agents/` or `applications/`.
- `agents/` must not import from `applications/`.
- `applications/` may import from `agents/` and `intergrax/`.

This separation keeps the **Harness** (runtime) stable while agents and product surfaces evolve independently.

---

## Applications — isolated environments you can ship

**Applications** are how Intergrax turns reusable **agent capabilities** into **separate, deployable products**.  
An agent answers *what the system can do* (legal review, research, echo tests). An application answers *how that capability is hosted, configured, and delivered* — as its own API, with its own env, Docker image, and integration profile.

Think of Tier-2 agents as **specialized modules** and Tier-3 applications as **ready-made shells** — like shipping the same engine in different vehicles: a lab van, a legal product API, or a research service. Each vehicle is an **isolated environment**; they share Nexus and agents, not each other's secrets or deployment config.

### What an application is

| Property | Meaning |
|----------|---------|
| **Isolated environment** | Own `.env` / `.env.example`, settings, HTTP routes, optional `docker/` — not mixed with the repo root or other apps |
| **Composition only** | Wires agents + integrations; **no** domain logic (that stays in `agents/<name>/`) |
| **Deployable unit** | Built to `uvicorn` locally; `applications/<app>/docker/build-docker.sh` (or `.bat`) for image build / push |
| **Template-born** | Created from a standard layout (scaffold `new-application` — Phase N); same ergonomics as `new-agent` for agents |
| **Reusable agents** | The same `LegalAgent` or `EchoAgent` can power **different** applications with different config, roster, and integrations |

### How it works (concept → runtime)

```text
agents/legal/              Tier-2  — LegalAgent, pipeline, prompts (reusable capability)
agents/research/           Tier-2  — ResearchAgent, SummaryAgent
        │
        │  AgentBinding.mount(AgentClass, factory=...)
        ▼
applications/legal_application/   Tier-3  — manifest, host, .env, Docker
applications/my_concept_lab/      Tier-3  — another product using Echo + your new agent
        │
        ▼
NexusLoop + IntegrationProfile + FastAPI  →  HTTP / Docker on production
```

1. **Define agents** under `agents/` (capabilities, UAEP steps, `AgentContract`).
2. **Declare a roster** in `applications/<app>/manifest.py` — which agents are active, with typed `AgentBinding.mount(AgentClass, factory=...)`.
3. **Build instances** via application factories (`host/agent_factories.py`, `host/agent_builders.py`) and `build_application_registry()`.
4. **Expose HTTP** in `host/factory.py` + `serving/`; configure Tier-0 backends through `IntegrationProfile` in `integration_wiring.py`.
5. **Run or ship** with app-local env and Docker — without copying agent code into the application tree.

The composition engine lives in **`intergrax/applications/`** (manifest contract, wiring API). Concrete hosts live in **`applications/<app_name>/`**.

### Applications vs agents vs shared lab

| | **Agent** (`agents/`) | **Shared lab** (`lab_application`) | **Dedicated application** (`applications/<app>/`) |
|--|------------------------|-------------------------------------|---------------------------------------------------|
| Purpose | Reusable capability | Experiment many agents in one debug surface | One product / POC with its own deployment |
| Domain logic | Yes | No | No |
| Own `.env` / Docker | No | Partial (lab defaults) | Yes — full isolation |
| Typical use | Build & test capability | Quick HTTP + `/debug/*` | Ship legal API, concept lab, customer-facing host |

You can validate an agent with pytest or the shared lab first, then **promote** it into a dedicated application when you need a stable host or production path.

### Scaffold and templates

| Scaffold | Creates | Command |
|----------|---------|---------|
| Agent template | `agents/<name>/` — UAEP, tests, notebook | `python -m intergrax.scaffold new-agent <name> --capability <domain>.<action>` |
| Application template | `applications/<app>/` — manifest, host, `.env.example`, docker (Phase N) | `AGENT_CREATION_GUIDE.md` Step 4E; `new-application --profile lab\|product` |

Both scaffolds follow the same idea: **opinionated folder layout + defaults** so you start from a working structure, not an empty repo.

### Example applications in this repository

| Application | What it demonstrates |
|-------------|-------------------|
| [`lab_application/`](applications/lab_application/) | Universal lab — multiple agents, debug API, `IntegrationProfile.lab()` |
| [`legal_application/`](applications/legal_application/) | Product host — configured `LegalAgent`, auth, FastAPI core, typed factory |
| [`research_application/`](applications/research_application/) | Multi-agent HTTP host — research pipeline wiring |

**Usage guides (define, invoke, deploy):**

- Composition engine: [`intergrax/applications/USAGE.md`](intergrax/applications/USAGE.md)
- Application folder layout: [`applications/USAGE.md`](applications/USAGE.md)
- Architecture rules: [`docs/intergrax_runtime_architecture.md`](docs/intergrax_runtime_architecture.md) §7.4.8–§7.4.10

---

## Harness AI workflow

The intended experimentation loop:

```text
new idea
  → define agent capability (AgentContract)
  → implement Tier-2 agent under agents/
  → register in AgentRegistry
  → run via NexusLoop (Task → AgentEngine → UAEP)
  → inspect traces, cost, quality, failures
  → keep · improve · pause · delete
```

Scaffold a new agent:

```bash
python -m intergrax.scaffold new-agent <name> --capability <domain>.<action>
```

When the agent needs its **own deployable host**, add a Tier-3 application (manifest + wiring) — see [Applications — isolated environments](#applications--isolated-environments-you-can-ship) and [`applications/USAGE.md`](applications/USAGE.md).

Run the regression gate:

```bash
uv run pytest -m gate -q
```

---

## Reference agents

| Agent | Capability | Path |
|-------|------------|------|
| Echo | Minimal UAEP reference | `agents/echo/` |
| Research | Web research pipeline | `agents/research/` |
| Research Summary | Summarization stage in multi-agent flow | `agents/research/` |
| Legal | Contract analysis and legal review | `agents/legal/` |

Agents execute through **UAEP** (Unified Agent Execution Protocol): `get_steps` → `run_step` → `decide_after_step`, orchestrated by `AgentEngine` inside `NexusLoop`.

Tier-3 **applications** that compose these agents are listed in [Applications — isolated environments](#applications--isolated-environments-you-can-ship).

---

## Runtime capabilities

- **NexusLoop** — task intake, agent selection, lifecycle (`Task` → `RUNNING` → `COMPLETED` / `WAITING_FOR_HUMAN`).
- **ExecutionGraph** — multi-agent workflows (e.g. Research → Summary).
- **UAEP + governance** — step boundaries, `AgentDecision`, human-in-the-loop pause/resume, policy interrupts.
- **ToolRuntime gateway** — unified tool access via `ToolRequest` / `RuntimeToolGateway` (§42.12).
- **Observability** — persisted run traces (SQLite), debug CLI (`python -m intergrax.debug`), debug HTTP API (`intergrax.debug.app`).
- **Tier-0 reuse** — one canonical path for LLM, RAG, tools, and tracing; agents compose rather than reimplement.

Legacy Supervisor / LangGraph orchestration is **deprecated** in favour of the Nexus runtime model.

---

## Repository layout

```text
intergrax/              # Tier-0 platform + Tier-1 Nexus runtime, AgentEngine, UAEP
agents/                 # Tier-2 specialized agents (echo, legal, research, …)
applications/           # Tier-3 isolated deployable environments (see applications/USAGE.md)
docs/                   # Architecture canon, implementation plan, agent guide (see docs/README.md)
notebooks/              # Runnable examples and integration demos
tests/                  # Unit and integration tests (gate: pytest -m gate)
```

---

## Platform building blocks (Tier-0)

Shared infrastructure used by all agents:

- **LLM adapters** — seventeen providers via `LLMAdapterRegistry` (`intergrax/llm_adapters/`) — see [LLM Adapters](#llm-adapters)
- **RAG** — embeddings, vector stores, document loaders (`intergrax/rag/`)
- **Tools** — registry, Tool Library catalog, MCP export (`intergrax/tools/`) — see [Tool Library](#tool-library)
- **Memory** — conversational and session storage (`intergrax/memory/`)
- **Integration Library** — modular catalog of external backends (DB, queues, search, vectors, cloud) — see [Integration Library](#integration-library)
- **FastAPI core** — shared API primitives for Tier-3 hosts (`intergrax/fastapi_core/`)

---

## Integration Library

Intergrax treats external systems as **pluggable, environment-specific infrastructure** — not hard-coded dependencies inside agents.

The **Integration Library** (`intergrax/integrations/`) provides:

| Property | Benefit |
|----------|---------|
| **Universal contracts** | Agents depend on small category protocols (`RelationalStore`, `VectorStore`, `MessageBus`, …), not vendor SDKs. |
| **Modular providers** | Each backend is a self-contained package (`providers/<slug>/`). Swap SQLite for PostgreSQL, Chroma for Qdrant, or log for Slack by changing configuration — not agent code. |
| **Portable across environments** | The same Tier-2 agent runs in a local lab (`IntegrationProfile.lab()`), a customer VPC, or a multi-cloud stack. Tier-3 applications compose the profile at startup. |
| **Safe boundaries** | Vendor SDKs live only in each provider’s `opens.py`. Tier-2 agents must not import provider slugs or third-party drivers. |

**72 providers** are registered today — relational stores (incl. Oracle/MSSQL/Azure SQL/Cloud SQL), document DBs (MongoDB, Cassandra, DynamoDB), Redis/Memcached/ElastiCache, Kafka/Celery/RabbitMQ/SQS/Service Bus/Pub/Sub, S3/Azure Blob/GCS, Pinecone/Qdrant/Chroma, web search (Google CSE, Bing, Brave, SerpAPI), Slack/Teams/SMTP, Jira/GitHub/Linear/Azure DevOps, Confluence/Notion/SharePoint, observability backends, Playwright, and AWS/Azure/GCP facades.

```python
# Tier-3 application — declare backends once
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

profile = IntegrationProfile(
    relational_store=IntegrationSlug.POSTGRESQL,
    vector_store=IntegrationSlug.QDRANT,
    notification_channel=IntegrationSlug.SLACK,
)
# Agents receive resolved contract instances — no boto3, no pymongo in agent code
```

**Full catalog (72 providers, env vars, links to per-slug `USAGE.md`):**  
**[docs/INTEGRATIONS.md](docs/INTEGRATIONS.md)**

Architecture rules: [runtime architecture §7.1](docs/intergrax_runtime_architecture.md) · implementation status: [Phase M](docs/INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Tool Library

Agents do not call Jira, Bing, or PostgreSQL directly. They invoke **tools** — named, schema-defined operations optimized for LLM planners and MCP clients. Tools compose the [Integration Library](#integration-library) (and RAG modules) underneath.

The **Tool Library** (`intergrax/tools/`) provides:

| Property | Benefit |
|----------|---------|
| **LLM-first contracts** | Each tool has `tool_id`, description, and JSON Schema parameters — for native tool-calling models and MCP. |
| **Composable semantics** | e.g. `jira.search_tasks(project, status)` builds JQL internally; agents never see raw integration APIs. |
| **Unified execution** | All invocations go through `ToolRuntime` — policy, trace, idempotency, and `allowed_tools` enforcement. |
| **Dual export** | Same catalog entry → OpenAI function schema, MCP tool, and UAEP `ToolRequest`. |
| **Unified model** | RAG and web search are catalog tools (`rag.retrieve`, `websearch.query`) — legacy `use_rag` / `use_websearch` map automatically. |

**Engine today:** `ToolContract`, `ToolRegistry`, `RuntimeToolInvoker`, `ToolsAgent`.  
**Catalog providers (Phase O Done):** full first-party catalog wired end-to-end in reference applications (`tool_wiring.py` → `ApplicationBuildContext` → agent `RuntimeConfig`). Legacy `use_rag` / `use_websearch` remain as compatibility shims.

```python
# Tier-2 agent — declare tool policy, not vendors
AgentContract(
    id="pm",
    allowed_tools=["jira.search_tasks", "jira.get_issue", "rag.retrieve", "confluence.search_pages"],
)

# Tier-3 application — wire integrations into tool handlers
from intergrax.tools.registry import ToolProfile, ToolWiringContext, build_registry_from_profile, register_default_tools

register_default_tools()
ctx = ToolWiringContext.from_integration_profile(integration_profile)
registry = build_registry_from_profile(ToolProfile(enabled_bundles=["jira", "confluence"]), ctx=ctx)
```

**Full catalog (tool_ids, status, migration from legacy flags):**  
**[docs/TOOLS.md](docs/TOOLS.md)** · **end-to-end wiring:** **[intergrax/tools/USAGE.md](intergrax/tools/USAGE.md)**

Architecture: [§7.1.6–§7.1.7](docs/intergrax_runtime_architecture.md) · implementation: [Phase O](docs/INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## LLM adapters

Agents call language models through **`LLMAdapter`** — not OpenAI/Anthropic SDKs directly. The adapter layer keeps chat, **streaming**, native **tools**, and **structured JSON** consistent across providers.

| Property | Benefit |
|----------|---------|
| **Unified contract** | `generate_messages`, `stream_messages`, optional `generate_with_tools` / `generate_structured`. |
| **Lazy registry** | `LLMAdapterRegistry.create("openai")` loads only the provider you need. |
| **Tool-ready** | OpenAI, Claude, Azure, Gemini, Mistral, and Bedrock (Anthropic) support native tool loops for `ToolsAgent`. |
| **Usage tracking** | Per-`run_id` token and latency stats for runtime observability. |
| **Outside integrations** | LLM providers are **not** Integration Library slugs (architecture §5.2.2). |

**Seventeen providers today:** `openai`, `claude`, `azure_openai`, `gemini`, `vertex_gemini`, `mistral`, `aws_bedrock`, `ollama`, `groq`, `vllm`, `together`, `fireworks`, `openrouter`, `deepseek`, `xai`, `llama_cpp`, `cohere`.

```python
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry

llm = LLMAdapterRegistry.create(LLMProvider.OPENAI, model="gpt-4o-mini")
text = llm.generate_messages([...], run_id=task_id)
```

**Full catalog (env vars, streaming/tools matrix, Bedrock Converse, optional `pyproject` extras):**  
**[docs/LLM_ADAPTERS.md](docs/LLM_ADAPTERS.md)**

Architecture: [§5.2.2](docs/intergrax_runtime_architecture.md)

---

## Status

Intergrax is under **active development** (private R&D). Phase status, priorities, and the business-agent readiness checklist:

**[`docs/INTERGRAX_IMPLEMENTATION_PLAN.md`](docs/INTERGRAX_IMPLEMENTATION_PLAN.md)**

Regression gate: `uv run pytest -m gate -q` (**314** tests in CI paths; includes `tests/unit/llm_adapters/`)

---

## Audience

Intergrax is for teams and developers who want to:

- build **specialized agents** on a shared Harness rather than one-off LLM scripts,
- orchestrate **multi-agent ecosystems** with traceable execution,
- validate agent hypotheses in a **controlled laboratory** before productizing,
- integrate AI capabilities into business systems via **Tier-3 application hosts** — isolated, template-based environments that compose agent specializations into separate deployable products.

---

## License

All rights reserved © Artur Czarnecki.

This repository is currently in private R&D stage. Commercial licensing and partnership opportunities are available upon request.

---

**Maintainer:** Artur Czarnecki  
**Repository:** [Intergrax](https://github.com/jakbuczarnecki/intergrax-ai)  
**Contact:** jakbu.czarnecki.83@gmail.com
