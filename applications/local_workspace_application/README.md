# Local Knowledge Workspace (LKW)

Tier-3 product host for local document indexing, semantic search, and synthesis.

**Architecture (canonical):** [ARCHITECTURE.md](ARCHITECTURE.md) · **Plan:** [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)  
**User journey:** [USER_JOURNEY.md](USER_JOURNEY.md) · **Platform proof loop:** [PLATFORM_PROOF_LOOP.md](PLATFORM_PROOF_LOOP.md)  
**Build & deploy:** [BUILD_AND_DEPLOY.md](BUILD_AND_DEPLOY.md)

## Agents

| Agent | Capability |
|-------|------------|
| `LocalIndexerAgent` | `local.workspace.index` |
| `LocalSearchAgent` | `local.workspace.search` (default) |
| `LocalSynthesizerAgent` | `local.workspace.synthesize` |

## What LKW proves

LKW is not only a local document assistant. It is the first product proof that Intergrax can repeatedly create, configure, run, package, deploy, observe, and evolve agent applications.

A new user should be able to follow [USER_JOURNEY.md](USER_JOURNEY.md): clone the repository, configure LKW, start the local backend, index a document, search with evidence, synthesize a draft into the shadow workspace, and inspect the trace/evidence for the run.

## Local stack

The local-first LKW stack is:

- **LKW backend:** FastAPI + Nexus + local agents;
- **Vector store:** Qdrant for persistent local RAG;
- **Relational/runtime data:** SQLite files under `INTERGRAX_SQLITE_DATA_DIR`;
- **Shadow artifacts:** `INTERGRAX_SHADOW_ROOT`;
- **LLM:** Ollama by default, vLLM optionally;
- **Redis:** optional until background ingest / queue workflows require it.

In-memory vector storage is only for tests or temporary development. It is not the real local product default.

## Quickstart

From repository root:

```bash
uv run pytest applications/local_workspace_application/local_workspace_application_tests -q
cp applications/local_workspace_application/.env.example applications/local_workspace_application/.env
uv run uvicorn local_workspace_application.host.main:app --host 127.0.0.1 --port 8020
```

Before indexing real files, set `INTERGRAX_ALLOWED_READ_ROOTS` in `.env` to one or more absolute folders that LKW may read.

## HTTP

```bash
curl -s http://127.0.0.1:8020/health
curl -s http://127.0.0.1:8020/v1/local_workspace/agents
curl -s -X POST http://127.0.0.1:8020/v1/local_workspace/run \
  -H "Content-Type: application/json" \
  -d '{"message":"find information about project X","capability":"local.workspace.search"}'
```

## MCP

`http://127.0.0.1:8020/mcp` — `list_agents`, `run_agent`, catalog tools.

## Runtime model

**Philosophy:** local **backend daemon** (Nexus + agents + index) + **thin frontends** (MCP, tray, optional Slack). See [ARCHITECTURE.md §3–§4](ARCHITECTURE.md#3-product-philosophy).

**Install & data paths:** [ARCHITECTURE.md §7](ARCHITECTURE.md#7-installation-lifecycle-and-on-disk-layout)

**Runtime:** [ARCHITECTURE.md §9](ARCHITECTURE.md#9-local-os-runtime-and-interaction-model) — Slack is **optional** (§9.4).

**Implementation waves:** [ARCHITECTURE.md §15](ARCHITECTURE.md#15-implementation-plan-derivation-canonical) · [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

## Platform stack

LKW uses the canonical **Integration → Tool → Skill → Agent** model ([ARCHITECTURE.md §8](ARCHITECTURE.md#8-integrations-tools-and-skills)):

- **Integrations:** LKW local product profile — SQLite, Qdrant, Docling, optional Redis, local LLM;
- **Tools:** `host/tool_wiring.py` — `rag.*`, `document.parse`, `workspace.*`, `memory.*`, `cache.*`;
- **Skills:** `harness` bundle (LKW.0); domain `local.workspace.*` skills planned (LKW.2).

## Docs

- Final user journey: [USER_JOURNEY.md](USER_JOURNEY.md)
- LKW architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- LKW hardening: [ARCHITECTURE_HARDENING.md](ARCHITECTURE_HARDENING.md)
- Platform proof loop: [PLATFORM_PROOF_LOOP.md](PLATFORM_PROOF_LOOP.md)
- Plan register: [docs/intergrax_runtime_architecture.md §6.3a](../../docs/intergrax_runtime_architecture.md#63a-business-backlog-register-consolidated)
- Agent workflow: [docs/guides/AGENT_CREATION_GUIDE.md](../../docs/guides/AGENT_CREATION_GUIDE.md)
- Application layout: [applications/USAGE.md](../USAGE.md)
