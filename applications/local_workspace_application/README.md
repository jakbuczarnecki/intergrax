# Local Knowledge Workspace (LKW)

Tier-3 product host for local document indexing, semantic search, and synthesis.

**Architecture (canonical):** [ARCHITECTURE.md](docs/ARCHITECTURE.md) · **Plan:** [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)  
**Live verification:** [LKW_1_LIVE_VERIFICATION.md](docs/LKW_1_LIVE_VERIFICATION.md)  
**User journey:** [USER_JOURNEY.md](docs/USER_JOURNEY.md) · **Platform proof loop:** [PLATFORM_PROOF_LOOP.md](docs/PLATFORM_PROOF_LOOP.md)  
**Build & deploy:** [BUILD_AND_DEPLOY.md](docs/BUILD_AND_DEPLOY.md)

## Agents

| Agent | Capability |
|-------|------------|
| `LocalIndexerAgent` | `local.workspace.index` |
| `LocalSearchAgent` | `local.workspace.search` (default) |
| `LocalSynthesizerAgent` | `local.workspace.synthesize` |

## What LKW proves

LKW is not only a local document assistant. It is the first product proof that Intergrax can repeatedly create, configure, run, package, deploy, observe, and evolve agent applications.

LKW.1 product proof is now passed for the live path:

```text
index -> search with tenant-scoped evidence -> synthesize with evidence -> shadow artifact only
```

**LKW.2** (graph pipeline + local workspace skills) is **in progress**. **LKW.2.1–LKW.2.4C** are complete (local skill bundle, agent `skill_ids`, environment profile, pipeline graph spec, search evidence handoff, live pipeline proof). **Next:** **LKW.2 closeout docs** — direct-capability smoke + platform proof checklist. LKW.2 is not fully closed until closeout lands.

A new user should be able to follow [USER_JOURNEY.md](docs/USER_JOURNEY.md): clone the repository, configure LKW, start the local backend, index a document, search with evidence, synthesize a draft into the shadow workspace, and inspect the trace/evidence for the run.

Current LKW.2 status: [IMPLEMENTATION_PLAN.md §5](docs/IMPLEMENTATION_PLAN.md#5-lkw2-graph-pipeline--local-workspace-skills). [LKW_1_LIVE_VERIFICATION.md](docs/LKW_1_LIVE_VERIFICATION.md) is the historical LKW.1/H1 live proof record, not the current LKW.2 execution status.

## Local stack

The local-first LKW stack is:

- **LKW backend:** FastAPI + Nexus + local agents;
- **Vector store:** Qdrant for persistent local RAG;
- **Relational/runtime data:** SQLite files under `INTERGRAX_SQLITE_DATA_DIR`;
- **Shadow artifacts:** `INTERGRAX_SHADOW_ROOT`;
- **LLM:** Ollama by default, vLLM optionally;
- **Redis:** optional until background ingest / queue workflows require it.

In-memory vector storage is only for tests or temporary development. It is not the real local product default.

## Docker quickstart

From `applications/local_workspace_application/`:

Windows:

```bat
scripts/build-local-docker.bat
```

Linux/macOS:

```bash
chmod +x scripts/build-local-docker.sh
./scripts/build-local-docker.sh
```

The scripts copy `.env.example` to `.env` when needed, build the Docker image, start Ollama, pull the model configured in `.env`, and start the local stack.

## Quickstart

From repository root:

```bash
uv run pytest applications/local_workspace_application/tests -q
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

**Philosophy:** local **backend daemon** (Nexus + agents + index) + **thin frontends** (MCP, tray, optional Slack). See [ARCHITECTURE.md §3–§4](docs/ARCHITECTURE.md#3-product-philosophy).

**Install & data paths:** [ARCHITECTURE.md §7](docs/ARCHITECTURE.md#7-installation-lifecycle-and-on-disk-layout)

**Runtime:** [ARCHITECTURE.md §9](docs/ARCHITECTURE.md#9-local-os-runtime-and-interaction-model) — Slack is **optional** (§9.4).

**Implementation waves:** [ARCHITECTURE.md §15](docs/ARCHITECTURE.md#15-implementation-plan-derivation-canonical) · [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)

## Platform stack

LKW uses the canonical **Integration → Tool → Skill → Agent** model ([ARCHITECTURE.md §8](docs/ARCHITECTURE.md#8-integrations-tools-and-skills)):

- **Integrations:** LKW local product profile — SQLite, Qdrant, Docling, optional Redis, local LLM;
- **Tools:** `host/tool_wiring.py` — `rag.*`, `document.parse`, `workspace.*`, `memory.*`, `cache.*`;
- **Skills:** `harness` + `local` bundles (LKW.0, LKW.2.1–LKW.2.3); pipeline capability `local.workspace.pipeline` registered and live proof passed (LKW.2.4A–LKW.2.4C).

## Docs

See [docs/README.md](docs/README.md) for the full local documentation index.

- Final user journey: [USER_JOURNEY.md](docs/USER_JOURNEY.md)
- LKW architecture: [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- LKW live verification: [LKW_1_LIVE_VERIFICATION.md](docs/LKW_1_LIVE_VERIFICATION.md)
- LKW hardening: [ARCHITECTURE_HARDENING.md](docs/ARCHITECTURE_HARDENING.md)
- Platform proof loop: [PLATFORM_PROOF_LOOP.md](docs/PLATFORM_PROOF_LOOP.md)
- Implementation plan: [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)
- Agent workflow: [docs/guides/AGENT_CREATION_GUIDE.md](../../docs/guides/AGENT_CREATION_GUIDE.md)
- Application layout: [applications/USAGE.md](../USAGE.md)
