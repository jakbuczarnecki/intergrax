# Local Knowledge Workspace (LKW) — architecture

**Status:** Architecture baseline v2 (2026-06-07) — implementation-plan source of truth  
**Tier:** Tier-3 application (`local_workspace_application`)  
**Agents:** Tier-2 `local_indexer`, `local_search`, `local_synthesizer`  
**Canonical plan row:** [`docs/intergrax_runtime_architecture.md` §6.3a LKW.*](../../docs/intergrax_runtime_architecture.md#63a-business-backlog-register-consolidated)  
**Derived plan:** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — generated from this document; do not fork scope elsewhere  
**Public product-validation narrative:** [`docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md`](../../docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md)

---

## 0. How to use this document

This file is the **single product architecture** for LKW. From it you derive:

| Need | Read section |
|------|----------------|
| Product philosophy, boundaries | §3 · §4 |
| What is frontend vs backend | §4 |
| Solution + trust zones | §5 |
| Agent roster | §6 |
| Install / upgrade / uninstall | §7 |
| Integrations, tools, skills · LKW.4 background jobs | §8 |
| Runtime + Slack (optional) | §9 |
| Request flows | §10 |
| Implementation waves + acceptance | §15 |
| Env vars and paths on disk | §7.3 · §12 |

**Rule:** change architecture first, then update [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) and platform [`§6.3a`](../../docs/intergrax_runtime_architecture.md#63a-business-backlog-register-consolidated). One coherent diff per wave.

---

## 1. Strategic purpose

**Local Knowledge Workspace (LKW)** is the first **business product environment** on Intergrax after harness platform maturity. Its role is dual:

1. **Product:** Give a user a local, safe assistant over their own files — search, gather context, produce structured outputs (reports, emails, estimates).
2. **Harness validation:** Exercise the Agent OS on a real, observable workload without external market APIs (unlike deferred K.1 Problem Radar / K.2 Vendor Discovery).

LKW validates: RAG ingest/retrieve/index lifecycle, document parsing, shadow workspace, multi-agent orchestration, memory, policy, trace, MCP/HTTP serving, and Tier-3 composition — while surfacing platform gaps early.

**Strategic frame:** [`docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md) — explicit product reprioritization after Appendix A sign-off.

---

## 2. Problem statement

Users store project knowledge across folders (PDF, DOCX, XLSX, TXT, email exports). They need to:

| Need | Example |
|------|---------|
| **Find** | "Find documents about project X / settlement Y" |
| **Gather** | "Gather data from folders A and B about the cost estimate" |
| **Synthesize** | „Przygotuj mail / sprawozdanie / kosztorys wg szablonu” |
| **Safety** | Do not delete or overwrite user original files |

LKW solves this with **read-heavy indexing + semantic retrieval + isolated write artifacts**, orchestrated by Nexus.

---

## 3. Product philosophy

### 3.1 What LKW is

LKW is a **personal Agent OS instance** on the user's computer:

- **Always-on backend** (local daemon) owns data, index, agents, and policy.
- **Thin frontends** (tray, Cursor MCP, Slack, scripts) only send **tasks** and show **results**.
- **Intergrax Nexus** is the only orchestrator — no ad-hoc agent loops in UI code.

### 3.2 What LKW is not

| Not this | Why |
|----------|-----|
| Slack bot that “is” the product | Slack is an **optional remote control**; execution stays on localhost |
| Cloud SaaS with file upload | Files never leave the machine by default |
| Single monolithic “chat agent” | Three bounded agents + graph pipeline |
| Unrestricted filesystem agent | Read allowlist + shadow-only writes |
| Replacement for Nexus / Tier-0 | Composition and wiring only — reuse platform mechanisms |

### 3.3 Design principles (non-negotiable)

1. **Compute local, control multi-channel** — index and embeddings on user disk; Slack/HTTP/MCP are equal task transports.
2. **Read user FS, write shadow only** — originals are never modified by agents.
3. **Integration → Tool → Skill → Agent** — no vendor SDKs in Tier-2; Tier-3 wires profiles.
4. **Every surface → one Task** — same trace, policy, and agents regardless of UI.
5. **Slack optional** — product must work with **localhost only** (LKW.1–5); Slack is LKW.6b enhancement.
6. **Harness honesty** — gaps discovered during LKW feed back to Tier-0 plan, not Nexus forks.

### 3.4 Primary vs optional user journeys

| Journey | Channel | Required wave |
|---------|---------|---------------|
| Developer at desk | MCP / HTTP | LKW.0–1 |
| Background index of folders | Daemon + watcher | LKW.7 |
| Quick search anytime | HTTP / tray | LKW.1 + LKW.8 |
| Remote command from phone | Slack slash | LKW.6b |
| Approve draft report | Slack HITL / HTTP | LKW.2 + notify |

---

## 4. Frontend vs backend boundaries

### 4.1 Layer map

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  FRONTEND (thin clients — no agent logic, no direct RAG)                 │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────┐ ┌─────────────────┐ │
│  │ LKW Tray    │ │ Cursor MCP  │ │ Slack client │ │ curl / scripts  │ │
│  │ (LKW.8)     │ │ (LKW.0)     │ │ (LKW.6b)     │ │ (LKW.0)         │ │
│  └──────┬──────┘ └──────┬──────┘ └──────┬───────┘ └────────┬────────┘ │
│         │               │               │                   │          │
│         └───────────────┴───────────────┴───────────────────┘          │
│                                    │ HTTP / MCP / interaction intake    │
└────────────────────────────────────┼────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  BACKEND — LKW Daemon (single product boundary on localhost)             │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Tier-3  local_workspace_application                                │  │
│  │  FastAPI Core · /health · /v1/local_workspace/* · /mcp           │  │
│  │  /v1/interactions/intake (LKW.6) · optional Socket Mode (LKW.6b) │  │
│  │  manifest · environment_profile · tool_wiring · factory          │  │
│  └───────────────────────────────┬───────────────────────────────────┘  │
│                                  ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Tier-1  Nexus Agent OS                                             │  │
│  │  NexusLoop · graph · HITL · trace · memory · policy                │  │
│  └───────────────────────────────┬───────────────────────────────────┘  │
│                                  ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Tier-2  local_indexer · local_search · local_synthesizer           │  │
│  └───────────────────────────────┬───────────────────────────────────┘  │
│                                  ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Tier-0  integrations · tools · skills · RAG · shadow workspace     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────┐  optional: indexer sidecar (LKW.7)        │
│  │ file watcher +         │  same host; enqueues via message_bus       │
│  │ background ingest path │  (platform TaskQueue — LKW.4)              │
│  └──────────────────────┘                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LOCAL DATA PLANE (backend-owned paths — §7)                            │
│  ~/.local/share/intergrax/lkw/  or  %LOCALAPPDATA%\Intergrax\LKW\       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Responsibility matrix

| Concern | Frontend | Backend (LKW application host) |
|---------|----------|----------------------|
| User message / command | Collects text, paths, template choice | Parses into `Task` |
| Capability routing | May suggest `capability` in JSON | Nexus selects agent / graph |
| RAG ingest / retrieve | **Never** | Agents + `rag.*` tools |
| LLM calls | **Never** | `RuntimeConfig` / agent pipeline |
| File read (user home) | May pick folder in tray | `filesystem.*` / ingest with allowlist |
| File write (deliverables) | May open exported file | `workspace.*` shadow only |
| Auth to localhost | Optional API key in tray config | `LocalWorkspaceBackendSettings` |
| Slack tokens | **Never** stored in tray | Daemon config / env |
| Trace / debug | May show run_id link | SQLite trace DB |

### 4.3 Frontend catalog (planned)

| Client | Technology | Talks to | Wave |
|--------|------------|----------|------|
| **HTTP API** | any HTTP client | `POST /v1/local_workspace/run` | LKW.0 Done |
| **MCP** | Cursor, Claude Desktop | `http://127.0.0.1:8020/mcp` | LKW.0 Done |
| **LKW Tray** | Tauri/Electron or native | localhost HTTP + folder picker | LKW.8 |
| **Slack** | Slack App (Socket Mode) | intake via platform interaction stack on LKW host | LKW.6b |
| **CLI operator** | `intergrax.debug` | trace DB | platform Done |

### 4.4 Backend process model

| Process | Role | Required |
|---------|------|----------|
| **`lkw-host`** | Uvicorn + FastAPI + NexusLoop + MCP + optional Slack socket | **Yes** — one per user session |
| **`lkw-indexer-worker`** | File watcher → `message_bus.enqueue` → ingest | LKW.7 optional |
| **External LLM API** | Inference only | Configurable (Ollama local or cloud) |

**Single-host rule:** one `lkw-host` per user account binds `127.0.0.1:<port>` (default `8020`). Tray and MCP are clients, not second runtimes.

---

## 5. Solution overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  User client (HTTP / MCP / future desktop shell)                        │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ POST /v1/local_workspace/run
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Tier-3  local_workspace_application                                    │
│  manifest · environment_profile · tool_wiring · factory · MCP           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Tier-1  Nexus Agent OS                                                 │
│  Intake → Plan → Graph (index → search → synthesize) → Trace → Result   │
└───────┬─────────────────┬─────────────────┬─────────────────────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ local_indexer│  │ local_search │  │ local_synthesizer│
│ Tier-2       │  │ Tier-2       │  │ Tier-2           │
└──────┬───────┘  └──────┬───────┘  └────────┬─────────┘
       │                 │                    │
       └─────────────────┴────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Tier-0  Platform — four-layer stack (canon §7.1.6–§7.1.8)              │
│  Integration → Tool → Skill → Agent                                     │
│  Docling · SQLite · vector store · rag.* · workspace.* · local.* skills │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Four-layer composition (Integration → Tool → Skill → Agent)

LKW follows the canonical Intergrax stack — agents never call vendor SDKs; Tier-3 selects backends and enables catalog surfaces:

```text
IntegrationProfile (Tier-3 host)
  ├── document_parser=docling     → DocumentParser backend
  ├── vector_store=inmemory|chroma → VectorStore for RAG index
  ├── relational_store=sqlite     → trace, session, task memory
  ├── rerank_provider=cohere_rerank → optional rerank on rag.retrieve
  └── observability_backend=otel  → traces (optional)

ToolProfile (Tier-3 host/tool_wiring.py)
  └── enabled tool_ids → ToolRegistry + ToolWiringContext

SkillProfile (Tier-3 environment)
  └── enabled_bundles: harness (LKW.0) · local (LKW.2 planned)
      → resolves skill_ids → allowed_tools + prompt refs on AgentContract

AgentContract (Tier-2)
  └── skill_ids[] + capabilities[] → UAEP steps invoke tools via ToolRuntime
```

**Rule:** Tier-3 **wires** integrations and tools; Tier-2 agents **declare** `skill_ids` on `AgentContract`; skills **compose** tool packs + prompts + policy fragments. See [`docs/architecture/SKILLS.md`](../../docs/architecture/SKILLS.md) · [`docs/architecture/TOOLS.md`](../../docs/architecture/TOOLS.md) · [`docs/architecture/INTEGRATIONS.md`](../../docs/architecture/INTEGRATIONS.md).

### 5.2 Trust zones (filesystem safety)

| Zone | Purpose | Mechanism | Mutations |
|------|---------|-----------|-----------|
| **Read zone** | User documents (allowlisted paths) | `rag.ingest_document`, `document.parse`; future `filesystem.*` read-only | **None** on user FS |
| **Artifact zone** | Reports, drafts, exports | `workspace.*` on **shadow workspace** | Only under `INTERGRAX_SHADOW_ROOT` |
| **Sandbox zone** | Risky experiments | `sandbox.exec` (opt-in per task) | Isolated under `INTERGRAX_SANDBOX_ROOT` |

**Rule:** LKW agents MUST NOT write to user home directories. All deliverables go to shadow workspace unless the user explicitly promotes an export path in a future Wave.

---

## 6. Agent roster and capabilities

| Agent | Module | Capability | Responsibility |
|-------|--------|------------|----------------|
| **LocalIndexerAgent** | `agents/local_indexer/` | `local.workspace.index` | Discover paths (Wave 1: explicit), parse, chunk, embed, index via `rag.ingest_document` |
| **LocalSearchAgent** | `agents/local_search/` | `local.workspace.search` | Semantic + metadata-filtered retrieval via `rag.retrieve`; rank and package evidence |
| **LocalSynthesizerAgent** | `agents/local_synthesizer/` | `local.workspace.synthesize` | LLM synthesis from retrieved context; write artifacts to shadow workspace |

**Pipeline capability (graph-level):** `local.workspace.pipeline` — multi-step intent routing index → search → synthesize (Wave 2). Documented here; wired via Nexus `AgentGraph` / delegation like `research.pipeline`.

Agent architecture docs:

- [`agents/local_indexer/docs/ARCHITECTURE.md`](../../agents/local_indexer/docs/ARCHITECTURE.md)
- [`agents/local_search/docs/ARCHITECTURE.md`](../../agents/local_search/docs/ARCHITECTURE.md)
- [`agents/local_synthesizer/docs/ARCHITECTURE.md`](../../agents/local_synthesizer/docs/ARCHITECTURE.md)

---

## 7. Installation, lifecycle, and on-disk layout

### 7.1 Installation philosophy

LKW installs as a **user-level background service** plus optional tray frontend. No system-wide server required. Python/uv environment ships with the product bundle (or uses existing Intergrax dev tree for engineering builds).

**Target personas:**

| Persona | Install path |
|---------|--------------|
| Developer | `uv sync` + `uvicorn` from repo (today) |
| End user (future) | Installer → `%LOCALAPPDATA%\Intergrax\LKW` or `~/.local/share/intergrax/lkw` |

### 7.2 Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.12 + uv | Dev; packaged install may embed runtime |
| LLM endpoint | Ollama local (`INTERGRAX_OLLAMA_*`) or cloud API key |
| Disk space | Index + trace (plan ~1–5 GB for typical corpus) |
| OS permissions | Read access to user-selected folders; macOS Full Disk Access if needed |

### 7.3 On-disk layout (canonical paths)

Default root: **`$LKW_DATA_HOME`** (env) with fallbacks:

| OS | Default `LKW_DATA_HOME` |
|----|-------------------------|
| Linux | `~/.local/share/intergrax/lkw` |
| macOS | `~/Library/Application Support/Intergrax/LKW` |
| Windows | `%LOCALAPPDATA%\Intergrax\LKW` |

```text
$LKW_DATA_HOME/
├── config/
│   ├── .env                    # LOCAL_WORKSPACE_* secrets (gitignored)
│   ├── allowed_read_roots.json # folder allowlist (LKW.3+)
│   └── integration_profile.json  # optional Chroma override
├── data/
│   ├── chroma/                   # vector index (when chroma enabled)
│   ├── sqlite/
│   │   ├── intergrax_trace.db
│   │   ├── intergrax_session.db
│   │   └── intergrax_task_memory.db
│   └── shadow_workspaces/        # INTERGRAX_SHADOW_ROOT override
├── logs/
│   └── lkw-host.log
└── run/
    └── lkw-host.pid
```

**Engineering default (repo dev):** `build/` under repository — override via env for product parity testing.

### 7.4 Install steps by OS (APP-HOST-7 — later operator/packaging targets)

**Ownership:** LKW declares product always-on requirements and adopts platform Application Hosting. Generic OS service integration, signal handling, restart supervision, and service-manager descriptors are **platform-owned** ([`APPLICATION_HOSTING`](../../../docs/architecture/APPLICATION_HOSTING.md)). The examples below are **operator-facing targets** for post-APP-HOST-7 packaging — **not** LKW.6B initial proof requirements.

**LKW.6B initial proof** does not require service-manager installation or reboot survival unless APP-HOST-7 is completed. Initial acceptance covers: foreground hosted start, READY state, real LKW request, single-instance rejection, graceful stop, supervisor restart, new instance identity, real request after restart.

```text
LKW application
  → LKW-specific HostedApplicationProfile
  → platform HostedApplicationEngine
  → platform supervisor / OS adapters when applicable (APP-HOST-7)
```

#### Windows

```text
1. Installer copies bundle → %LOCALAPPDATA%\Intergrax\LKW\
2. APP-HOST-7: platform Windows hosting adapter registers always-on service (LKW does not own generic service framework)
3. Optional: tray app in Startup folder → localhost:8020
4. First-run wizard (LKW.8): pick folders → writes allowed_read_roots.json
```

#### Linux (systemd user unit)

```ini
# ~/.config/systemd/user/lkw-host.service
[Unit]
Description=Intergrax Local Knowledge Workspace
After=network.target

[Service]
ExecStart=%h/.local/share/intergrax/lkw/bin/lkw-host
Restart=on-failure
Environment=LKW_DATA_HOME=%h/.local/share/intergrax/lkw

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now lkw-host
```

#### macOS (LaunchAgent)

```xml
<!-- ~/Library/LaunchAgents/com.intergrax.lkw.plist -->
Label: com.intergrax.lkw
ProgramArguments: ~/.local/share/intergrax/lkw/bin/lkw-host
RunAtLoad: true
KeepAlive: true
```

Grant **Full Disk Access** if indexing outside home directory.

### 7.5 Upgrade and uninstall

| Action | Behaviour |
|--------|-----------|
| **Upgrade** | Stop service → replace `bin/` + Python env → migrate sqlite/chroma if schema version bumps → start |
| **Uninstall** | Stop service → remove unit/plist → delete `$LKW_DATA_HOME` (user prompt: keep index?) |
| **Config only reset** | Delete `config/.env`; keep `data/chroma` |

### 7.6 Health and readiness

| Check | Endpoint / command |
|-------|-------------------|
| Process up | `GET http://127.0.0.1:8020/health` |
| Agents registered | `GET /v1/local_workspace/agents` |
| Index ready | `rag.check_index_status` via MCP or debug task |
| Integration health | host bootstrap probes at startup (log on failure) |

---

## 8. Integrations, tools, and skills

### 8.1 Integrations (`IntegrationProfile`)

**Baseline preset:** `IntegrationProfile.legal_product()` — RAG + document parsing without mandatory web search (unlike `research_product()`).

| `IntegrationCategory` slot | Slug (default) | Role in LKW | Wired via |
|--------------------------|----------------|-------------|-----------|
| `relational_store` | `sqlite` | Trace DB, session state, task memory persistence | `wire_application_environment` → `memory_wiring` |
| `vector_store` | `inmemory` | RAG chunk index (dev); replace with `chroma` for durable local index | `rag_runtime_bridge` → `ToolWiringContext.vectorstore_manager` |
| `document_parser` | `docling` | PDF/DOCX/XLSX parsing inside `rag.ingest_document` / `document.parse` | `CatalogDocumentParser` — infra slot, not auto-exposed as agent tool |
| `rerank_provider` | `cohere_rerank` | Optional rerank after hybrid retrieval in `rag.retrieve` | `RetrievalService` / `RagProfile` |
| `observability_backend` | `otel` (optional) | Export traces when OTLP enabled on environment profile | `host/environment_profile.py` |
| `object_storage` | `filesystem` (Wave 4) | Export shadow artifacts / checkpoint blobs | `storage.*` tools when enabled |
| `message_bus` | message_bus provider slug (LKW.4) | Platform background ingest jobs | `message_bus.*` when a message bus provider is configured |

**Override:** `INTERGRAX_INTEGRATION_PROFILE_JSON` — e.g. swap `vector_store` to `chroma` for persistent local index.

**Explicitly excluded in baseline:** `search_provider` (web), `collaboration` (mail APIs) — LKW is local-first.

Authoring: [`docs/guides/AGENT_CREATION_GUIDE.md` Appendix K](../../docs/guides/AGENT_CREATION_GUIDE.md#appendix-k--integration--rag-control-plane) · catalog: [`docs/architecture/INTEGRATIONS.md`](../../docs/architecture/INTEGRATIONS.md).

### 8.2 Tools (`ToolProfile` + `host/tool_wiring.py`)

Tier-3 enables tools; agents invoke them through `BoundToolGateway` / `ctx.invoke_tool()` — never direct integration imports.

#### Host-wide tool allowlist (`_LKW_BASE_TOOL_IDS`)

| tool_id | Bundle | Composes (integration slot) | LKW role |
|---------|--------|----------------------------|----------|
| `rag.ingest_document` | `rag` | `vector_store` + `document_parser` + embedding managers | Index local files |
| `rag.retrieve` | `rag` | `vector_store` + optional `rerank_provider` | Semantic search |
| `rag.list_collections` | `rag` | `vector_store` | Index diagnostics |
| `rag.list_documents` | `rag` | `vectorstore_manager` | Paginated index inventory |
| `rag.get_document` | `rag` | `vectorstore_manager` | Fetch indexed chunk by id |
| `rag.check_index_status` | `rag` | `vectorstore_manager` | Index readiness probe |
| `document.parse` | `document` | `document_parser` | Ad-hoc parse without full ingest |
| `document.parse_preview` | `document` | `document_parser` | Bounded parse preview (no ingest) |
| `workspace.read_file` | `workspace` | runtime `ShadowWorkspace` | Read shadow artifacts |
| `workspace.write_file` | `workspace` | runtime `ShadowWorkspace` | Write drafts/reports |
| `workspace.list_files` | `workspace` | runtime `ShadowWorkspace` | List artifacts |
| `workspace.snapshot` | `workspace` | runtime `ShadowWorkspace` | Point-in-time snapshot |
| `workspace.delete_file` | `workspace` | runtime `ShadowWorkspace` | Remove draft revisions in shadow only |
| `workspace.search` | `workspace` | runtime `ShadowWorkspace` | Grep across shadow artifacts |
| `memory.read` / `memory.write` / `memory.list_keys` | `memory` | `relational_store` + task memory | Session working state |
| `cache.get` / `cache.set` | `cache` | optional KV backend | Dedup parse/embedding keys |

**Env-gated (settings):** `LOCAL_WORKSPACE_ENABLE_RAG` → `rag.retrieve`; `LOCAL_WORKSPACE_ENABLE_RAG_INGEST` → `rag.ingest_document`.

**Filesystem browse (T6 / LKW.3 Done):** when `INTERGRAX_ALLOWED_READ_ROOTS` or `allowed_read_roots` is set, host auto-enables `filesystem.list`, `filesystem.glob`, `filesystem.read_text`, `filesystem.stat`.

**Explicitly disabled:** `websearch.*`, `openai.file_search.*` — external retrieval out of scope for LKW baseline.

Catalog reference: [`docs/architecture/TOOLS.md`](../../docs/architecture/TOOLS.md) · wiring: [`host/tool_wiring.py`](host/tool_wiring.py).

### 8.3 Skills (`SkillProfile` + `AgentContract.skill_ids`)

Skills are **composable packs** (tools + prompt instruction ids + optional policy fragments). The LLM does not call skills directly — Nexus resolves `skill_ids` into `allowed_tools` at register time.

#### Enabled today (LKW.0)

| Bundle | `skill_bundles` | Purpose |
|--------|-----------------|---------|
| `harness` | `["harness"]` on `ApplicationEnvironmentProfile` | Platform smoke packs (`harness.tool_smoke`, `harness.trace_read`, …) — harness validation only |

Environment: [`manifest.py`](manifest.py) · [`host/environment_profile.py`](host/environment_profile.py).

#### Planned domain bundle (LKW.2) — `intergrax/skills/providers/local/`

| `skill_id` | Agent | `tool_ids` | `prompt_instruction_ids` |
|------------|-------|------------|----------------------------|
| `local.workspace.index` | `local_indexer` | `rag.ingest_document`, `document.parse`, `rag.list_collections` | `local.workspace.index.system` |
| `local.workspace.search` | `local_search` | `rag.retrieve`, `rag.list_collections`, `cache.get`, `cache.set` | `local.workspace.search.system` |
| `local.workspace.synthesize` | `local_synthesizer` | `workspace.read_file`, `workspace.write_file`, `workspace.list_files`, `workspace.search`, `memory.read` | `local.workspace.synthesize.system` |
| `local.workspace.pipeline` | graph intent (all three) | union of above (via `requires_skills`) | orchestration prompt refs |

**Agent wiring (LKW.2):** each `AgentContract` gains `skill_ids=[...]`; register via `registry.register(agent, skill_registry=..., tool_registry=...)`. Until then, agents use scaffold `skills=[]` and rely on host `ToolProfile` only.

Skill authoring: [`docs/architecture/SKILLS.md`](../../docs/architecture/SKILLS.md) · Appendix J in [`docs/guides/AGENT_CREATION_GUIDE.md`](../../docs/guides/AGENT_CREATION_GUIDE.md#appendix-j--tools--skills-control-plane).

### 8.4 Per-agent Integration / Tool / Skill matrix

| Agent | Integrations consumed (indirect) | Primary tools | Skill (LKW.2) |
|-------|----------------------------------|---------------|---------------|
| **LocalIndexerAgent** | `document_parser`, `vector_store`, embedding managers | `rag.ingest_document`, `document.parse`, `rag.list_collections` | `local.workspace.index` |
| **LocalSearchAgent** | `vector_store`, `rerank_provider` | `rag.retrieve`, `cache.*`, `memory.*` | `local.workspace.search` |
| **LocalSynthesizerAgent** | runtime shadow workspace (not integration slug) | `workspace.*`, `memory.read` | `local.workspace.synthesize` |

### 8.5 Runtime wiring path (Tier-3 → Tier-1 → Tier-2)

```text
wire_application_environment(manifest, environment, settings)
  ├── bootstrap_application_integration_catalog()
  ├── probe_integration_profile_health()
  ├── resolve_rag_stack_for_environment()     # ContextProfile.enable_rag=true
  ├── build_application_tool_wiring()           # ToolProfile → ToolRegistry
  ├── build_application_skill_wiring()        # SkillProfile → SkillRegistry
  └── ApplicationBuildContext                 # passed to agent factories

build_application_registry(manifest, build_context, builders)
  └── register agents; resolve skill_ids → allowed_tools

NexusLoop → AgentEngine → UAEP ctx.invoke_tool(ToolRequest(tool_name="rag.retrieve", ...))
```

### 8.6 Environment profile summary

- `ApplicationEnvironmentProfile.product_defaults(profile_id="local_workspace.product")`
- `skill_bundles=["harness"]` (LKW.0); extend with `"local"` at LKW.2
- `integration_profile=IntegrationProfile.legal_product()`
- `ContextProfile(enable_rag=True, enable_websearch=False)`
- `with_harness_memory()` — STM/LTM hooks for long sessions
- OTLP optional on `observability_profile` + `IntegrationProfile` OTEL slot

See [`host/environment_profile.py`](host/environment_profile.py).

### 8.7 LKW.4 — Background jobs via platform MessageBus

LKW.4 is a **platform message-bus / background-jobs proof track**, not an LKW-owned queue implementation. LKW must **not** implement an application-specific queue, a new queue system, or provider-specific SDK wiring. **LKW is the proof workload; platform owns queue infrastructure.**

**Platform proof pattern** (same as observability):

```text
Application/domain job
  → platform TaskQueue / MessageBus contract
  → provider-neutral message_bus.* tools
  → provider integration
  → LKW background ingest proof workload
```

#### Ownership boundaries

| Layer | Owns |
|-------|------|
| **Platform** | `TaskQueue` / `MessageBus` contract (`intergrax/queueing/contracts/task_queue.py`, `intergrax/integrations/contracts/message_bus.py`); `MessageBusIntegrationContract` (`intergrax/runtime/integrations/categories/messaging.py`); provider integrations; provider-neutral `message_bus.*` tools (`message_bus.enqueue`, `message_bus.get_status`, `message_bus.get_result`, `message_bus.list_tasks`, …); lifecycle / status / result abstraction |
| **LKW (Tier-3)** | `LkwBackgroundIngestJob` (`background_ingest/contracts.py`); `task_name` (`lkw.background_ingest.v1`); payload schema; idempotency key convention; handler mapping; proof workload and reviewer runbook |
| **Agents (Tier-2)** | Tool/skill invocation only — **no** provider SDK imports; **no** Kafka / RabbitMQ / Celery imports |
| **Providers** | Backend implementation behind the common contract (examples only — LKW.4 does not require all): `kafka`, `rabbitmq`, `celery`, `redpanda`, `sqs`, `service_bus`, `pubsub`, `nats`, `pulsar`, `confluent`, `temporal` |

#### Platform background task model dependency

LKW.4 is aligned with the platform background task architecture in [`docs/architecture/BACKGROUND_TASKS.md`](../../../docs/architecture/BACKGROUND_TASKS.md). LKW background ingest is one concrete **TaskDefinition** in that model — not a separate queue design.

**LKW.4E must use the target concepts:**

- `TaskRequest` enqueue envelope
- `TaskDefinition` / handler mapping (`lkw.background_ingest.v1` → `handle_background_ingest_task_request`)
- `WorkerRuntime` with a **real local MessageBus provider** in the proof stack (BG-TASKS-7)
- Pull status/result via `message_bus.get_status` / `get_result`
- Lifecycle events and trace correlation (target model; proof may start minimal)

LKW.4E is a **platform proof through LKW**. It must demonstrate production-like platform behavior: real `message_bus.*` tools, a real local broker/provider in the proof stack, and asynchronous worker execution. Mocks, fake queues, in-memory-only bypasses, and unit-test-only handler invocation are **not** sufficient for platform proof. See [`docs/plan/BACKGROUND_TASKS.md`](../../../docs/plan/BACKGROUND_TASKS.md) and public reviewer Step 8 in [`docs/public-adoption/LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md).

#### Intended background ingest flow

Triggers (file watcher, scheduler, or explicit user background action) build a domain job and enqueue through the platform surface — **without** duplicating queue logic in LKW:

```text
File watcher / scheduler / user background action
  → build LkwBackgroundIngestJob
  → encode_background_ingest_job()
  → background_ingest_payload_base64()
  → message_bus.enqueue
  → TaskRequest(
       tenant_id,
       run_id,
       task_name="lkw.background_ingest.v1",
       payload=<json bytes>,
       idempotency_key=<stable key>
     )
  → MessageBus / TaskQueue
  → provider adapter
  → worker handler
  → decode_background_ingest_job()
  → execute local.workspace.index through platform execution path
  → TaskResult / status via message_bus.get_status / get_result / list_tasks
```

**Execution rules:**

- `local.workspace.index` remains the indexing capability — the worker runs the **existing** capability path and must **not** duplicate `LocalIndexerAgent` logic inline.
- Payload carries paths and scope only (`LkwBackgroundIngestJob`); no raw document content in the job envelope.
- Idempotency is platform-backed via `TaskRequest.idempotency_key` and LKW's stable key convention.

**Compact request-flow diagram:**

```text
Background action
  → LkwBackgroundIngestJob
  → message_bus.enqueue
  → TaskQueue / MessageBus
  → provider
  → worker handler
  → local.workspace.index (platform execution path)
  → message_bus.get_status / get_result
```

#### LKW.4 vs LKW.7

| Wave | Proves / adds |
|------|----------------|
| **LKW.4** | Domain job payload; enqueue via platform `message_bus.*`; inspect lifecycle through provider-neutral tools; handler executes index without changing agent logic; live proof via search/index evidence |
| **LKW.7** (later) | File watcher; incremental index trigger policy; directory change detection; batching/debounce; recurring filesystem-driven enqueue |

File watcher and incremental index are **LKW.7**, not LKW.4. OS daemon and interaction intake remain **LKW.6**. Slack notify remains optional later (**LKW.6b**), not LKW.4 core.

#### LKW.4 vs provider portability

LKW.4 starts with **one real local message bus provider** in the proof stack (for example RabbitMQ in Docker). Provider portability proof can happen later. Provider-specific SDKs stay behind platform provider integrations — LKW.4 does **not** implement every listed backend. Mocks and in-memory-only queue bypasses do **not** satisfy LKW.4E platform proof.

#### Message bus tool exposure (LKW.4B guardrail — implemented)

When a `message_bus` integration is configured on the host integration profile, `message_bus.*` tools **may** be exposed to the relevant host/tool profile. When `message_bus` is **not** configured, `message_bus.*` tools remain **disabled** for LKW. Shared application wiring (`apply_resolved_integration_tool_guardrails` in `intergrax/applications/_shared/integration_tool_profile.py`) enforces the resolved `ToolWiringContext.message_bus` guardrail; LKW host (`host/tool_wiring.py`) consumes that helper — **LKW.4B closed** · **LKW.4B-PROP-1 closed**.

Code references: [`background_ingest/contracts.py`](../background_ingest/contracts.py) · [`background_ingest/enqueue.py`](../background_ingest/enqueue.py) (LKW.4C enqueue helper) · [`background_ingest/handler.py`](../background_ingest/handler.py) (LKW.4D worker handler contract) · platform [`BACKGROUND_TASKS.md`](../../../docs/architecture/BACKGROUND_TASKS.md) · [`INTEGRATIONS.md`](../../docs/architecture/INTEGRATIONS.md) · [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §6.

---

## 9. Local OS runtime and interaction model

LKW is a **local execution environment** (Tier-3 host + Nexus) that runs **in the background on the user's machine** (Windows, Linux, macOS). The user can submit work **at any time**; agents are spawned by Nexus on demand. Chat apps (Slack, Teams) are **interaction surfaces** — not the runtime — they deliver commands and receive summaries.

### 9.1 Design principle: compute local, control multi-channel

| Layer | Where it runs | What it holds |
|-------|---------------|---------------|
| **Execution** | User OS (localhost) | Nexus, agents, RAG index, shadow workspace, file access |
| **Interaction** | Slack / Teams / HTTP / MCP / tray (optional) | Commands, status, HITL prompts, short answers |
| **Cloud** | Optional (Slack API, LLM API) | Messaging + inference only — **not** user file storage |

**Privacy default:** document chunks and embeddings stay on disk under user control. Slack receives **commands and condensed answers**, not full file dumps.

### 9.2 Background process topology

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  User workstation (Windows / Linux / macOS)                              │
│                                                                          │
│  ┌─────────────────────┐    ┌──────────────────────┐                   │
│  │ LKW Host (always-on) │    │ Indexer sidecar (opt)│                   │
│  │ local_workspace_app  │    │ file watcher +       │                   │
│  │ :8020 localhost      │    │ background ingest    │                   │
│  │                      │    │ enqueue (LKW.7)      │                   │
│  └──────────┬──────────┘    └──────────┬───────────┘                   │
│             │                          │                                 │
│             ▼                          ▼                                 │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ Local data plane                                             │        │
│  │ Chroma/SQLite index · shadow_workspaces/ · trace DB          │        │
│  └─────────────────────────────────────────────────────────────┘        │
│             ▲                                                            │
│             │ POST /v1/local_workspace/run                               │
│             │ POST /v1/interactions/intake  (Slack / Teams / JSON)       │
│             │ MCP /v1/...                                                │
│  ┌──────────┴──────────┐   ┌─────────────┐   ┌──────────────────┐     │
│  │ Tray / CLI (LKW.8)  │   │ Cursor MCP  │   │ Slack / Teams    │     │
│  │ (optional UI)       │   │ (local)     │   │ (remote surface) │     │
│  └─────────────────────┘   └─────────────┘   └────────┬─────────┘     │
└──────────────────────────────────────────────────────────┼───────────────┘
                                                           │
                                              Slack Socket Mode or HTTPS tunnel
                                              (outbound from daemon — no public IP required)
```

#### OS service packaging (APP-HOST-7 — later operator/packaging targets)

**Platform ownership:** generic always-on hosting, lifecycle state machine, readiness aggregation, instance lock, signal handling, restart loop, and OS adapters (`systemd`, `launchd`, Windows Service) are owned by [`APPLICATION_HOSTING`](../../../docs/architecture/APPLICATION_HOSTING.md). LKW is the first adopter and proof — it supplies an LKW-specific `HostedApplicationProfile`, hooks, and components only.

**LKW.6B initial proof** does not require the OS adapters below unless APP-HOST-7 is completed. Keep these as later operator/packaging targets.

| OS | Platform adapter target (APP-HOST-7) | Notes |
|----|--------------------------------------|-------|
| **Windows** | Platform Windows hosting adapter | User-session for file access; avoid SYSTEM account for home-folder indexing |
| **Linux** | Platform `systemd` user-unit integration | `After=network.target`; restart on failure via platform supervisor |
| **macOS** | Platform `launchd` LaunchAgent integration | Full Disk Access may be required for user folders |

Host entrypoint (today): `uvicorn local_workspace_application.host.main:app`. **LKW.6B** adopts platform hosting around this factory; LKW does not implement generic daemon engine or OS hosting mechanics in the application tree.

#### Always-on responsibilities

1. **Listen** for user tasks (HTTP, MCP, interaction intake).
2. **Maintain** local RAG index (platform message-bus background ingest — LKW.4; filesystem triggers — LKW.7).
3. **Run** Nexus graph on demand (search now, synthesize on request).
4. **Notify** on completion / HITL (`notification_channel=slack` on long-running tasks).
5. **Persist** checkpoints for pause/resume ([`docs/intergrax_runtime_architecture.md` Appendix F.4](../../docs/intergrax_runtime_architecture.md)).

### 9.3 Interaction surfaces (how the user talks to LKW)

| Surface | Status | Endpoint / mechanism | Best for |
|---------|--------|----------------------|----------|
| **Local HTTP** | Scaffold **Done** | `POST /v1/local_workspace/run` | Scripts, tray, local integrations |
| **Local MCP** | Scaffold **Done** | `/mcp` on same host | Cursor / IDE at desk |
| **Interaction intake** | Platform **Done**; LKW host **LKW.6A Done** | `POST /v1/interactions/intake` | Slack slash commands, Teams, lab JSON |
| **Slack outbound** | Platform **Done** | `INTERGRAX_SLACK_WEBHOOK_URL`, HITL templates | Alerts, approvals, result snippets |
| **Debug CLI** | Platform **Done** | `python -m intergrax.debug` | Operators |
| **Tray / native UI** | **Deferred LKW.8** | Calls localhost HTTP/MCP | Folder picker, status icon |

**Rule:** every surface normalizes to a Nexus `Task` — same agents, same policy, same trace. **LKW.6A** unifies `/v1/local_workspace/run` and `/v1/interactions/intake` through one `LocalWorkspaceTaskExecutor` before `NexusLoop`. See [`applications/USAGE.md` §4b](../USAGE.md) · canon §18.

### 9.3a LKW.6A — unified application execution boundary (closed)

Platform interaction intake exists and LKW host wiring exists; **LKW.6A** unifies execution and application-level readiness semantics (temporary until **LKW.6B** adopts platform Application Hosting).

```text
POST /v1/local_workspace/run
POST /v1/interactions/intake
(future tray / Slack / OS sources)
  → platform adapter (interaction only) / HTTP request model (/run)
  → Task
  → LocalWorkspaceTaskExecutor.prepare()  [capability policy + LKW defaults + reliability + orchestration ACP]
  → LocalWorkspaceTaskExecutor.execute_prepared()
  → NexusLoop.handle_task
  → TaskResult
```

| Concern | Owner | Notes |
|---------|-------|-------|
| Transport normalization | Platform `InteractionAdapter` / HTTP schemas | No LKW interaction models |
| Application execution prep | `LocalWorkspaceTaskExecutor` | Allowlisted capabilities: `local.workspace.search` / `.index` / `.synthesize` (+ graph triggers) |
| Reliability enrichment | Shared `build_reliability_task_enricher` | Applied once per execution |
| Application readiness (temporary) | `LocalWorkspaceHostLifecycle` (LKW.6A) | `STARTING` → `READY` → `STOPPING` → `STOPPED`; `FAILED` on startup errors — **not** canonical platform hosting; awaits LKW.6B migration to `HostedApplicationEngine` |
| Liveness | `GET /health` | Unchanged: `{"status":"ok"}` |
| Readiness | `GET /v1/local_workspace/readiness` | Requires `READY` + executor available + required components healthy |
| Work rejection | Both execution surfaces | HTTP 503 `lkw_host_not_ready` / `lkw_host_stopping` when not accepting work |
| Background extension point | Documented only | `execute=false` returns prepared `Task`; future background routing via platform message bus (LKW.4) |

**LKW.6A does not include:** platform Application Hosting adoption (**LKW.6B**), Socket Mode (**LKW.6b**), file watcher (**LKW.7**), or OS interaction adapters (**LKW.6C**).

### 9.3b LKW.6C — Windows PowerShell interaction adapter

**Status: Closed** after the live MongoDB-backed reviewer proof passes.

LKW owns a thin Windows PowerShell product client that serializes the supported `lab_json` payload and posts it to the existing platform interaction intake. No new platform interaction channel is introduced.

```text
invoke-lkw-interaction.ps1
  → POST /v1/interactions/intake?execute=true
  → LabJsonInteractionAdapter (lab_json / channel = lab)
  → InteractionIntakeService
  → LocalWorkspaceTaskExecutor
  → NexusLoop
  → real LKW capability execution
```

| Concern | Owner | Notes |
|---------|-------|-------|
| Product adapter script | `scripts/invoke-lkw-interaction.ps1` | Adapter identity `lkw.windows_powershell`; source `windows_powershell` |
| Platform channel | `LabJsonInteractionAdapter` | `interaction_channel` remains `lab` |
| Task / enrichment / Nexus | Existing LKW host + platform | No Task, agent, or RAG logic in PowerShell |
| Hosting / instance lock / signals | Platform Application Hosting | No generic OS hosting behavior in LKW |
| Windows Service | APP-HOST-7 | Not LKW.6C |
| Slack Socket Mode | LKW.6b (optional) | Not LKW.6C |
| File watcher | LKW.7 | Not LKW.6C |
| Tray | LKW.8 | Not LKW.6C |

Enable intake with existing settings: `LOCAL_WORKSPACE_INCLUDE_INTERACTIONS=true`, `LOCAL_WORKSPACE_INTERACTION_SURFACE=lab_json`, `LOCAL_WORKSPACE_INTERACTION_EXECUTE_DEFAULT=true`.

Reviewer command: `applications\local_workspace_application\scripts\run-lkw-windows-interaction-proof.bat` — see [`LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md) Steps 12–13.

### 9.4 Slack as optional interaction channel

Slack is **supported and professional** as an **optional** channel — not the product core. Use existing Intergrax **interaction + notification** integrations (`slack` slug). Execution remains on the **local LKW application host** (always-available backend on localhost).

**Decision record:** Primary UX = localhost (HTTP/MCP/tray). Slack = remote/mobile/team + HITL. Product must pass acceptance tests **without** Slack configured.

#### Reference flow (slash command)

```text
User in Slack:  /lkw search dokumenty o projekcie Alpha
       │
       ▼
Slack Events API  ──►  (A) Socket Mode client in LKW host process   [preferred: no inbound port]
                    or (B) HTTPS tunnel → localhost:8020/v1/interactions/intake
       │
       ▼
InteractionIntakeService  +  SlackInteractionAdapter
       │  verify signature · parse slash payload · map text → Task
       ▼
LocalWorkspaceTaskExecutor  →  NexusLoop.handle_task(capability=local.workspace.search, message=...)
       │
       ▼
LocalSearchAgent → rag.retrieve (local Chroma index)
       │
       ▼
Reply to Slack (response_url / chat.postMessage) — citations + short summary only
```

**Platform primitives to reuse (no Nexus fork):**

| Primitive | Module / doc | LKW use |
|-----------|--------------|---------|
| `InteractionIntakeService` | `runtime/interactions/intake_service.py` | Inbound Slack → `Task` |
| `SlackInteractionAdapter` | `integrations/providers/notification_channel/slack/` | Channel id `slack` |
| `wire_interaction_intake_service` | `applications/_shared/interaction_wiring.py` | Enable on `local_workspace_application` factory |
| `TaskLongRunningOptions.notify_channel="slack"` | plan Appendix F.4 | HITL + long ingest jobs |
| Organization worker runbook | plan §H.6 | Prior art for slash → Nexus → resume |

**Example intake (lab-equivalent, today):**

```bash
curl -s -X POST "http://127.0.0.1:8020/v1/interactions/intake?execute=true&tenant=U1" \
  -H "Content-Type: application/json" \
  -d '{"command":"/lkw","text":"search projekt Alpha","user_id":"U1","team_id":"T1"}'
```

LKW.6A wires interaction intake through the shared executor (see §9.3a). Enable with `LOCAL_WORKSPACE_INCLUDE_INTERACTIONS=true` and `LOCAL_WORKSPACE_INTERACTION_SURFACE=lab_json` for Slack-free proof.

#### Slack connectivity modes

| Mode | Pros | Cons | LKW recommendation |
|------|------|------|---------------------|
| **Socket Mode** | No public URL; daemon initiates outbound WebSocket | Requires Slack app + bot token in local config | **Default for desktop daemon** |
| **HTTPS tunnel** (ngrok, Cloudflare Tunnel) | Quick dev | Extra dependency; URL rotation | Dev / demo only |
| **Slack notifications only** | Simple webhook | User cannot command from Slack | Phase 1 fallback — local MCP/HTTP for commands, Slack for HITL |

#### Slack command mapping (convention)

| User text (after `/lkw`) | `Task.context.capability` | Agent |
|--------------------------|----------------------------|-------|
| `index <path>` | `local.workspace.index` | `local_indexer` |
| `search <query>` | `local.workspace.search` | `local_search` |
| `draft email\|report\|estimate …` | `local.workspace.synthesize` | `local_synthesizer` |
| free text (default) | `local.workspace.pipeline` | graph (LKW.2) |

`tenant_id` / `user_id` from Slack identity map to Intergrax task scope for memory and index partitions.

#### What must NOT go through Slack

- Raw file uploads containing full document corpora (use local index instead).
- Shadow workspace binary artifacts (link to local export path or summary).
- Unredacted secrets from parsed files.

### 9.5 Task timing: foreground vs background

| Pattern | Trigger | Nexus behaviour |
|---------|---------|-----------------|
| **Interactive** | User message (Slack, HTTP, MCP) | Sync or async run; reply when `COMPLETED` or `WAITING_FOR_HUMAN` |
| **Background index** | File watcher / cron / explicit enqueue (LKW.4 + LKW.7) | `message_bus.enqueue` → platform queue lifecycle → worker runs `local.workspace.index`; optional Slack notify on batch complete (LKW.6b) |
| **Long-running synthesize** | Large report | `TaskLongRunningOptions` + checkpoint; user resumes via Slack `approve` / HTTP |

User can **always** submit a new interactive task while background indexing runs — platform message-bus/task-queue idempotency prevents duplicate ingests (LKW.4 payload key + LKW.7 watcher policy).

### 9.6 Integration profile extension for Slack (LKW.6b)

Extend `IntegrationProfile` on LKW host (in addition to `legal_product()` RAG slots):

```text
notification_channel = slack    # HITL + completion alerts
interaction_surface  = slack    # inbound slash / events (via intake router)
```

Env (mirror legal/lab): `LOCAL_WORKSPACE_INCLUDE_INTERACTIONS=true`, `LOCAL_WORKSPACE_INTERACTION_SURFACE=slack`, Slack signing secret + bot token for Socket Mode.

---

## 10. Request and data flows

### 10.1 Index flow

```text
Task(capability=local.workspace.index, metadata={source_paths: [...]})
  → LocalIndexerAgent UAEP steps
  → invoke rag.ingest_document per path
  → ParserPipeline + chunk + embed + vector store
  → StepOutput(metadata: {num_chunks, collection_id, parser_trace})
```

### 10.2 Search flow

```text
Task(capability=local.workspace.search, message="find documents about project X")
  → LocalSearchAgent
  → rag.retrieve(query, metadata filters)
  → Package evidence chunks + citations (path, page, chunk_id)
```

### 10.3 Synthesize flow

```text
Task(capability=local.workspace.synthesize, metadata={template: "email"|"report"|...})
  → LocalSynthesizerAgent
  → LLM with retrieved context (from graph handoff or prior step)
  → workspace.write_file("draft.md", content)
  → metadata: {shadow_workspace_id, artifact_paths}
```

### 10.4 Pipeline flow (Wave 2)

```text
Task(capability=local.workspace.pipeline, intent=local_workspace_full)
  → Nexus graph: DELEGATES_TO indexer? → search → synthesizer
  → SharedTaskContext carries evidence + artifact refs
```

---

## 11. Tier-3 composition map

| File | Role |
|------|------|
| [`manifest.py`](manifest.py) | Roster, capabilities, `LOCAL_WORKSPACE_APPLICATION_MANIFEST` |
| [`host/environment_profile.py`](host/environment_profile.py) | RAG-on, websearch-off product profile |
| [`host/tool_wiring.py`](host/tool_wiring.py) | LKW tool allowlist |
| [`host/settings.py`](host/settings.py) | `LOCAL_WORKSPACE_*` env, RAG flags |
| [`host/wiring.py`](host/wiring.py) | Registry + `wire_application_environment` |
| [`host/factory.py`](host/factory.py) | FastAPI Core + MCP |
| [`serving/fastapi_router.py`](serving/fastapi_router.py) | `/run`, `/agents` |
| [`mcp/server.py`](mcp/server.py) | FastMCP mount |

**No agent logic in Tier-3** — only wiring. Domain steps live in `agents/*/steps/`.

Authoring guide: [`docs/guides/AGENT_CREATION_GUIDE.md`](../../docs/guides/AGENT_CREATION_GUIDE.md) · Tier-3: [`applications/USAGE.md`](../USAGE.md).

---

## 12. Configuration

### 12.1 Environment variables (`.env.example`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `LOCAL_WORKSPACE_BACKEND_PORT` | `8020` | HTTP port |
| `LOCAL_WORKSPACE_DEFAULT_AGENT_ID` | `local_search` | Default roster agent |
| `LOCAL_WORKSPACE_ENABLE_RAG` | `true` | Enable `rag.retrieve` |
| `LOCAL_WORKSPACE_ENABLE_RAG_INGEST` | `true` | Enable `rag.ingest_document` |
| `INTERGRAX_SHADOW_ROOT` | `build/shadow_workspaces` | Artifact isolation root |
| `LKW_DATA_HOME` | OS default (§7.3) | Product data root |
| `INTERGRAX_ALLOWED_READ_ROOTS` | user config | Comma-separated read allowlist |
| `LOCAL_WORKSPACE_INCLUDE_INTERACTIONS` | `false` | Enable `/v1/interactions/intake` (LKW.6) |
| `LOCAL_WORKSPACE_INTERACTION_SURFACE` | `auto` | `slack` \| `teams` \| `lab` |

### 12.2 Task metadata conventions (Wave 1+)

```python
Task(
    metadata={
        "shadow_workspace": True,
        "source_paths": ["D:/Docs/project_a/report.pdf"],
        "collection_id": "user_u1_workspace",
        "synthesis_template": "email",
    }
)
```

---

## 13. Security and governance

- **Read-only user FS** in Waves 1–2 (ingest reads; no writes)
- **Shadow workspace** mandatory for synthesizer outputs
- **HITL** optional for sensitive exports (`REQUEST_HUMAN`) — [`docs/guides/AGENT_CREATION_GUIDE.md` Appendix A](../../docs/guides/AGENT_CREATION_GUIDE.md#appendix-a--human-in-the-loop)
- **Cost governance:** `CostProfile` on environment; embedding batch limits per ingest job
- **Trace:** all tool calls via Nexus trace DB — debug with `intergrax.debug` CLI

---

## 14. Observability and verification

```bash
# Host smoke
uv run pytest applications/local_workspace_application/tests -q

# Agent smoke
uv run pytest agents/local_indexer/tests agents/local_search/tests agents/local_synthesizer/tests -q

# Run host
uv run uvicorn local_workspace_application.host.main:app --host 127.0.0.1 --port 8020
```

Deploy triad: `docker/`, `BUILD_AND_DEPLOY.md` — gate `test_application_deploy_triad.py`.

---

## 15. Implementation plan derivation (canonical)

Each row is one implementable **wave**. Copy to [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) when scheduling work. **Depends** = prior waves. **Acceptance** = objective done criteria.

### 15.1 Wave summary

| ID | Wave | Title | Layer | Depends | Status |
|----|------|-------|-------|---------|--------|
| **LKW.0** | 0 | Scaffold + architecture v2 | Tier-2/3 docs | — | **Done** |
| **LKW.1** | 1 | Domain UAEP: ingest + search | Tier-2 agents | LKW.0 | Planned |
| **LKW.2** | 2 | Graph pipeline + local skills | Tier-1 graph + Tier-0 skills | LKW.1 | Planned |
| **LKW.3** | 3 | Filesystem browse + allowlist | Tier-0 tools + Tier-3 policy | LKW.0 | **Done** (T6) |
| **LKW.4** | 4 | Platform message-bus background ingest proof | Tier-0 message_bus + Tier-3 proof workload | LKW.1 | Planned |
| **LKW.5** | 5 | Chroma persistent index + `LKW_DATA_HOME` | Tier-3 config | LKW.1 | Planned |
| **LKW.6** | 6 | OS daemon packaging + interaction intake | Tier-3 host | LKW.1 | **Closed** (LKW.6A/6B/6C) |
| **LKW.6b** | 6b | Slack Socket Mode (optional) | Tier-3 + slack integration | LKW.6 | Planned / optional |
| **LKW.7** | 7 | File watcher + incremental index | Tier-3 sidecar + enqueue path | LKW.4, LKW.5 | **In progress** (LKW.7A/7B1/7B2A Done; LKW.7B2B next) |
| **LKW.8** | 8 | Tray frontend (thin client) | Frontend | LKW.6 | Deferred |

### 15.2 Wave detail (tasks + acceptance)

#### LKW.1 — Domain UAEP: ingest + search

| Task | Owner module | Deliverable |
|------|--------------|-------------|
| LKW.1.1 | `agents/local_indexer/steps/` | Ingest pipeline: validate paths → `rag.ingest_document` loop |
| LKW.1.2 | `agents/local_search/steps/` | Search pipeline: `rag.retrieve` → evidence package |
| LKW.1.3 | `agents/local_synthesizer/steps/` | Stub synthesize → `workspace.write_file` in shadow |
| LKW.1.4 | tests | Acceptance: ingest fixture PDF → search returns citation |

**Acceptance:** `POST /run` with `source_paths` + `local.workspace.search` returns grounded answer; shadow artifact on synthesize; pytest green.

**Frontend:** HTTP/MCP only. **Backend:** all logic in agents.

---

#### LKW.2 — Graph pipeline + skills

| Task | Owner module | Deliverable |
|------|--------------|-------------|
| LKW.2.1 | `intergrax/skills/providers/local/` | Skill manifests `local.workspace.*` |
| LKW.2.2 | `agents/*/contract.py` | `skill_ids` on each agent |
| LKW.2.3 | `host/environment_profile.py` | `skill_bundles=["harness","local"]` |
| LKW.2.4 | `manifest` / graph_spec | `local.workspace.pipeline` graph |

**Acceptance:** Single `POST /run` with pipeline capability runs index→search→synthesize without manual capability selection.

---

#### LKW.5 — Persistent index + data home

| Task | Owner module | Deliverable |
|------|--------------|-------------|
| LKW.5.1 | `host/settings.py` | `LKW_DATA_HOME` resolution (§7.3) |
| LKW.5.2 | env / profile | Chroma under `data/chroma/` |
| LKW.5.3 | `BUILD_AND_DEPLOY.md` | Document paths per OS |

**Acceptance:** Restart host → prior index still retrievable.

---

#### LKW.6 — OS daemon + interaction intake (backend productization)

**Platform ownership:** generic always-on hosting is owned by [`APPLICATION_HOSTING`](../../../docs/architecture/APPLICATION_HOSTING.md); LKW is the first adopter and proof ([`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../../docs/architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md) §34). LKW.6A delivered the application execution boundary; LKW.6B adopts platform hosting — it does not implement `HostedApplicationEngine`, supervisors, or generic OS adapters in the application tree.

| Task | Owner module | Deliverable |
|------|--------------|-------------|
| LKW.6.1 | `scripts/lkw-host.*` | Start/stop wrapper for uvicorn (dev/operator convenience — not generic hosting engine) |
| LKW.6.2 | `hosting/` (LKW profile) + platform adoption | LKW-specific `HostedApplicationProfile` / hooks; platform OS adapter integration via APP-HOST-7 (§7.4 targets) |
| LKW.6.3 | `host/factory.py` | `wire_interaction_intake_service` + router |
| LKW.6.4 | `host/settings.py` | `LOCAL_WORKSPACE_INCLUDE_INTERACTIONS` |

**Acceptance (LKW.6B initial proof):** foreground hosted start → READY → real LKW request → single-instance rejection → graceful stop → supervisor restart → new instance identity → real request after restart. No Slack required. Service-manager installation and reboot survival are **APP-HOST-7** targets, not LKW.6B initial proof.

**Frontend:** none new. **Backend:** host only.

**ORCH-MAINT-02 — CFG-14 hybrid daemon enablement (operator runbook):**

1. Copy `.env.example` → `.env` in `applications/local_workspace_application/`.
2. Set `LOCAL_WORKSPACE_INCLUDE_SCHEDULER=true`, `LOCAL_WORKSPACE_INCLUDE_INTERACTIONS=true`, `LOCAL_WORKSPACE_INCLUDE_TASK_CONTROL=true`.
3. Optional background-jobs path: `LOCAL_WORKSPACE_INCLUDE_QUEUE_WORKER=true` when a message_bus provider is configured (see ORCH-MAINT-01 lab scaffold default).
4. Start host: `uv run uvicorn local_workspace_application.host.main:app --port 8090`.
5. Verify: `GET /health` → 200; `POST /v1/local_workspace/run` with `echo.basic` completes; scheduler poll logs when `INTERGRAX_SCHEDULER_POLL_SECONDS` set.

**Platform audit (2026-06-09):** CFG-14 hybrid daemon E2E remains **deferred** (Band 3 / §6.3). Harness reference for task control + scheduler: `poc_template_application`, `legal_application`, `research_application` with `INCLUDE_TASK_CONTROL` — see [`docs/architecture/ORCHESTRATION.md`](../../docs/architecture/ORCHESTRATION.md) §59.2 · Phase **H-APP-WIRING.4**.

---

#### LKW.6b — Slack optional channel

| Task | Owner module | Deliverable |
|------|--------------|-------------|
| LKW.6b.1 | `host/slack_socket.py` (new) | Socket Mode client → local intake |
| LKW.6b.2 | mapping | Slash `/lkw` → capability table (§9.4) |
| LKW.6b.3 | profile | `notification_channel=slack` for HITL |

**Acceptance:** `/lkw search foo` in Slack returns summary; **LKW.1 acceptance still passes with Slack disabled.**

---

#### LKW.7 — File watcher + incremental index

**Status:** **In progress** — LKW.7A **Done**; LKW.7B **In progress**; LKW.7B1 **Done**; LKW.7B2 **In progress**; LKW.7B2A **Done**; LKW.7B2B **Planned — next**; LKW.7C **Planned**.

| ID | Scope | Status |
|----|-------|--------|
| **LKW.7A** | Incremental file-change contract and idempotent batches | **Done** |
| **LKW.7B** | Watcher runtime + sidecar process | **In progress** |
| **LKW.7B1** | Runtime state machine, bounded debounce, existing enqueue boundary | **Done** |
| **LKW.7B2** | Cross-platform sidecar process, settings, checkpoint, graceful shutdown | **In progress** |
| **LKW.7B2A** | Durable checkpoint and restart recovery | **Done** |
| **LKW.7B2B** | Sidecar settings, process loop, signals and automatic checkpoint lifecycle | Planned — next |
| **LKW.7C** | Persistent-index live proof and ProofReceipt | Planned |

**LKW.7A flow (contract only):**

```text
allowed roots
  → metadata snapshot (path + size_bytes + modified_time_ns)
  → snapshot diff
  → IncrementalFileChangeBatch
  → change_token
  → LkwBackgroundIngestJob(change_token=...)
```

**LKW.7B1 flow (runtime state machine — no OS process yet):**

```text
snapshot
  → diff
  → pending final state per canonical path
  → quiet debounce or maximum wait
  → IncrementalFileChangeBatch
  → LkwBackgroundIngestJob
  → enqueue_background_ingest_job
  → platform message bus
```

**LKW.7B2A flow (durable checkpoint — no process loop yet):**

```text
runtime baseline + pending final changes
  → versioned FileWatcherCheckpoint
  → deterministic JSON
  → atomic replace under data home
  → process restart
  → fail-closed load
  → runtime restore
  → first poll detects downtime changes
  → existing LKW.7B1 debounce / enqueue flow
```

| Concern | Notes |
|---------|-------|
| Version identity | Metadata-based only (`path` + `size_bytes` + `modified_time_ns`); not content hashing |
| `change_token` | Deterministic identity of final actionable `source_snapshots` in one batch |
| Initial files | Baseline only — not emitted as `created` and not enqueued at start |
| Pending state | Bounded by changed path count; last change wins per canonical path |
| Debounce | Quiet period on `last_change_at`, plus bounded `max_batch_wait_seconds` |
| Deletions | Deletion-only batches do not enqueue; not automatically removed from the index |
| Enqueue failure | Pending changes retained; retry uses deterministic batch/job identity |
| Checkpoint | Durable baseline + final pending `FileChange` values; no file content |
| Monotonic timestamps | Never persisted; restored pending work starts a new debounce window |
| Missing checkpoint | Valid fresh-start condition for the future sidecar |
| Invalid checkpoint | Fail closed — not silently treated as missing |
| Identity mismatch | Fail closed when tenant/workspace/collection/roots disagree |
| Checkpoint save | Not yet automatically driven by a process loop (LKW.7B2B) |
| Runtime process | No watcher process yet (LKW.7B2B) |
| Content | No raw file content enters the job or checkpoint |

**Acceptance (full LKW.7):** Drop file in watched folder → indexed within N minutes without user command (requires LKW.7B2B/7C).

---

#### LKW.8 — Tray frontend (thin client)

| Task | Owner module | Deliverable |
|------|--------------|-------------|
| LKW.8.1 | `clients/lkw-tray/` (new repo folder or app) | Status icon + search box |
| LKW.8.2 | | Folder picker → `allowed_read_roots.json` |
| LKW.8.3 | | Calls only `localhost:8020` API |

**Acceptance:** No Python agent code in tray; uninstall tray does not remove index.

---

### 15.3 End-to-end scenarios (validation scripts)

| # | Scenario | Channels | Waves required |
|---|----------|----------|----------------|
| E1 | First install → pick folders → index | Tray + HTTP | LKW.5, LKW.6, LKW.8 |
| E2 | "Find documents about X" at desk | MCP | LKW.1 |
| E3 | Full report draft | HTTP pipeline | LKW.2 |
| E4 | New file auto-indexed | background | LKW.7 |
| E5 | Search from phone | Slack | LKW.6b (optional) |

---

## 16. Known platform gaps (honest audit)

| Gap | Impact | Mitigation (Wave) |
|-----|--------|-------------------|
| No file watcher | No auto re-index | LKW.7 file watcher + enqueue path |
| In-memory vector store default | Index lost on restart | Chroma + `INTEGRATION_PROFILE_JSON` |
| Windows path / OneDrive edge cases | Parser failures | Test matrix in LKW.1 acceptance |
| Qdrant/Chroma lack `list_document_ids` | `rag.list_documents` empty/unsupported on some backends | Use InMemory for dev; extend provider bindings in follow-up |

These gaps are **expected** — LKW exists to discover and close them without Nexus forks.

---

## 17. References

| Topic | Document |
|-------|----------|
| Agent workflow | [`docs/guides/AGENT_CREATION_GUIDE.md`](../../docs/guides/AGENT_CREATION_GUIDE.md) |
| Integration catalog | [`docs/architecture/INTEGRATIONS.md`](../../docs/architecture/INTEGRATIONS.md) |
| Tools catalog | [`docs/architecture/TOOLS.md`](../../docs/architecture/TOOLS.md) |
| Skill Library | [`docs/architecture/SKILLS.md`](../../docs/architecture/SKILLS.md) |
| Tools & skills control plane | [`docs/guides/AGENT_CREATION_GUIDE.md` Appendix J](../../docs/guides/AGENT_CREATION_GUIDE.md#appendix-j--tools--skills-control-plane) |
| RAG control plane | [`docs/guides/AGENT_CREATION_GUIDE.md` Appendix K](../../docs/guides/AGENT_CREATION_GUIDE.md#appendix-k--integration--rag-control-plane) |
| Shadow workspace | [`docs/guides/AGENT_CREATION_GUIDE.md` Appendix B](../../docs/guides/AGENT_CREATION_GUIDE.md#appendix-b--shadow-workspace-and-sandbox) |
| Multi-agent graphs | [`docs/guides/AGENT_CREATION_GUIDE.md` Appendix C](../../docs/guides/AGENT_CREATION_GUIDE.md#appendix-c--multi-agent-graphs) |
| Nexus execution flow | [`docs/architecture/NEXUS_EXECUTION_FLOW.md`](../../docs/architecture/NEXUS_EXECUTION_FLOW.md) |
| Implementation plan | [`docs/intergrax_runtime_architecture.md`](../../docs/intergrax_runtime_architecture.md) |
| Quickstart | [`README.md`](../README.md) · [`BUILD_AND_DEPLOY.md`](BUILD_AND_DEPLOY.md) |

---

## 18. Runtime recovery (APP-EVOL-5)

| Scenario | Host action |
|----------|-------------|
| Host restart | `resume_scheduler` via `ReliabilityProfile.recovery_contract` |
| Task interrupted | `resume` with checkpoint + idempotency store |
| Graph node failure | `retry_node` via Nexus orchestration retries |
| Corrupt checkpoint | `replay_from_snapshot` using `environment_snapshot.v1` |

- **Checkpoint store:** SQLite task checkpoints (see `.env.example` / `BUILD_AND_DEPLOY.md`)
- **Scheduler:** `long_running_scheduler_enabled` for async and HITL paths
- **In-flight tasks on deploy:** drain via checkpoint + `resume_token`; do not abort without operator ack
