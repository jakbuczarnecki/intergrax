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

All platform documentation lives in [`docs/`](docs/). Four documents, one source of truth per topic:

| Document | Read when you want to… |
|----------|------------------------|
| [docs/README.md](docs/README.md) | Navigate docs, see current phase focus, update rules |
| [intergrax_runtime_architecture.md](docs/intergrax_runtime_architecture.md) | Full architecture canon (tiers, Nexus, UAEP §42) |
| [INTERGRAX_IMPLEMENTATION_PLAN.md](docs/INTERGRAX_IMPLEMENTATION_PLAN.md) | Phase status, gaps, priority, business-agent checklist (Appendix A) |
| [AGENT_CREATION_GUIDE.md](docs/AGENT_CREATION_GUIDE.md) | Create an agent: scaffold → register → run → inspect |

**Quick paths:** new agent → [AGENT_CREATION_GUIDE](docs/AGENT_CREATION_GUIDE.md) · current phase → [IMPLEMENTATION_PLAN](docs/INTERGRAX_IMPLEMENTATION_PLAN.md) §1–§4 · deep architecture → [runtime_architecture](docs/intergrax_runtime_architecture.md) §1–§5

---

## Four-tier model

Intergrax is organized as a **four-tier stack**. Higher tiers compose lower tiers; they do not reimplement platform infrastructure.

| Tier | Role | Repository path |
|------|------|-----------------|
| **Tier-0 — Platform** | Universal, domain-agnostic building blocks: LLM adapters, RAG, tools, memory, logging, trace persistence | `intergrax/` (outside Nexus orchestration) |
| **Tier-1 — Nexus Runtime** | Agent OS: task lifecycle, `NexusLoop`, `AgentEngine`, UAEP, execution graph, governance, event bus | `intergrax/runtime/` |
| **Tier-2 — Agents** | Specialized capability modules: contracts, domain logic, pipelines, agent-local governance | `agents/` |
| **Tier-3 — Applications** | Product hosts: FastAPI entrypoints, env config, HTTP routes wiring Tier-2 agents | `applications/` |

**Dependency rules:**

- `intergrax/` must not import from `agents/` or `applications/`.
- `agents/` must not import from `applications/`.
- `applications/` may import from `agents/` and `intergrax/`.

This separation keeps the **Harness** (runtime) stable while agents and product surfaces evolve independently.

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

Run the regression gate:

```bash
uv run pytest -m gate -q
```

---

## Reference agents and applications

**Tier-2 agents** (capability modules):

| Agent | Capability | Path |
|-------|------------|------|
| Echo | Minimal UAEP reference | `agents/echo/` |
| Research | Web research pipeline | `agents/research/` |
| Research Summary | Summarization stage in multi-agent flow | `agents/research/` |
| Legal | Contract analysis and legal review | `agents/legal/` |

**Tier-3 applications** (HTTP / product hosts):

| Application | Purpose |
|-------------|---------|
| `applications/lab_application/` | Universal Agent OS lab (run agents, debug API) |
| `applications/legal_application/` | Legal agent FastAPI host |
| `applications/research_application/` | Research pipeline HTTP host |

Agents execute through **UAEP** (Unified Agent Execution Protocol): `get_steps` → `run_step` → `decide_after_step`, orchestrated by `AgentEngine` inside `NexusLoop`.

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
applications/           # Tier-3 product hosts (legal_application, research_application)
docs/                   # Architecture canon, implementation plan, agent guide (see docs/README.md)
notebooks/              # Runnable examples and integration demos
tests/                  # Unit and integration tests (gate: pytest -m gate)
```

---

## Platform building blocks (Tier-0)

Shared infrastructure used by all agents:

- **LLM adapters** — OpenAI, Anthropic, Ollama, Gemini, and others (`intergrax/llm_adapters/`)
- **RAG** — embeddings, vector stores, document loaders (`intergrax/rag/`)
- **Tools** — registry, MCP-oriented integrations (`intergrax/tools/`)
- **Memory** — conversational and session storage (`intergrax/memory/`)
- **FastAPI core** — shared API primitives for Tier-3 hosts (`intergrax/fastapi_core/`)

---

## Status

Intergrax is under **active development** (private R&D). Phase status, priorities, and the business-agent readiness checklist:

**[`docs/INTERGRAX_IMPLEMENTATION_PLAN.md`](docs/INTERGRAX_IMPLEMENTATION_PLAN.md)**

Regression gate: `uv run pytest -m gate -q` (247 tests; collects `tests/`, `applications/`, `agents/`)

---

## Audience

Intergrax is for teams and developers who want to:

- build **specialized agents** on a shared Harness rather than one-off LLM scripts,
- orchestrate **multi-agent ecosystems** with traceable execution,
- validate agent hypotheses in a **controlled laboratory** before productizing,
- integrate AI capabilities into business systems via Tier-3 application hosts.

---

## License

All rights reserved © Artur Czarnecki.

This repository is currently in private R&D stage. Commercial licensing and partnership opportunities are available upon request.

---

**Maintainer:** Artur Czarnecki  
**Repository:** [Intergrax](https://github.com/jakbuczarnecki/intergrax-ai)  
**Contact:** jakbu.czarnecki.83@gmail.com
