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
| Integrations, tools, skills | §8 |
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
│  │ file watcher + queue   │  same host, enqueues ingest tasks          │
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

| Concern | Frontend | Backend (LKW daemon) |
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
| **Slack** | Slack App (Socket Mode) | intake inside daemon | LKW.6b |
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

### 7.4 Install steps by OS (LKW.6 target)

#### Windows

```text
1. Installer copies bundle → %LOCALAPPDATA%\Intergrax\LKW\
2. Register Windows Service OR Login Task → runs scripts/lkw-host.ps1
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
| `message_bus` | queue slug (Wave 4) | Background ingest jobs | `message_bus.*` when worker enabled |

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
│  │ LKW Host (always-on) │    │ Indexer worker (opt) │                   │
│  │ local_workspace_app  │    │ file watcher + queue │                   │
│  │ :8020 localhost      │    │ Wave LKW.7           │                   │
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

#### OS service packaging (see also §7.4)

| OS | Recommended mechanism | Notes |
|----|----------------------|-------|
| **Windows** | Windows Service or scheduled task + tray helper | User-session for file access; avoid SYSTEM account for home-folder indexing |
| **Linux** | `systemd` user unit (`lkw.service`) | `After=network.target`; restart on failure |
| **macOS** | `launchd` LaunchAgent (`~/Library/LaunchAgents/`) | Full Disk Access may be required for user folders |

Host entrypoint (today): `uvicorn local_workspace_application.host.main:app`. Production packaging: single binary or `uv run` wrapper in service unit.

#### Always-on responsibilities

1. **Listen** for user tasks (HTTP, MCP, interaction intake).
2. **Maintain** local RAG index (background ingest — LKW.7).
3. **Run** Nexus graph on demand (search now, synthesize on request).
4. **Notify** on completion / HITL (`notification_channel=slack` on long-running tasks).
5. **Persist** checkpoints for pause/resume ([`docs/intergrax_runtime_architecture.md` Appendix F.4](../../docs/intergrax_runtime_architecture.md)).

### 9.3 Interaction surfaces (how the user talks to LKW)

| Surface | Status | Endpoint / mechanism | Best for |
|---------|--------|----------------------|----------|
| **Local HTTP** | Scaffold **Done** | `POST /v1/local_workspace/run` | Scripts, tray, local integrations |
| **Local MCP** | Scaffold **Done** | `/mcp` on same host | Cursor / IDE at desk |
| **Interaction intake** | Platform **Done**; LKW host **planned LKW.6** | `POST /v1/interactions/intake` | Slack slash commands, Teams, lab JSON |
| **Slack outbound** | Platform **Done** | `INTERGRAX_SLACK_WEBHOOK_URL`, HITL templates | Alerts, approvals, result snippets |
| **Debug CLI** | Platform **Done** | `python -m intergrax.debug` | Operators |
| **Tray / native UI** | **Deferred LKW.8** | Calls localhost HTTP/MCP | Folder picker, status icon |

**Rule:** every surface normalizes to a Nexus `Task` — same agents, same policy, same trace. See [`applications/USAGE.md` §4b](../USAGE.md) · canon §18.

### 9.4 Slack as optional interaction channel

Slack is **supported and professional** as an **optional** channel — not the product core. Use existing Intergrax **interaction + notification** integrations (`slack` slug). Execution remains on the **local LKW daemon**.

**Decision record:** Primary UX = localhost (HTTP/MCP/tray). Slack = remote/mobile/team + HITL. Product must pass acceptance tests **without** Slack configured.

#### Reference flow (slash command)

```text
User in Slack:  /lkw search dokumenty o projekcie Alpha
       │
       ▼
Slack Events API  ──►  (A) Socket Mode client in LKW daemon   [preferred: no inbound port]
                    or (B) HTTPS tunnel → localhost:8020/v1/interactions/intake
       │
       ▼
InteractionIntakeService  +  SlackInteractionAdapter
       │  verify signature · parse slash payload · map text → Task
       ▼
NexusLoop.handle_task(capability=local.workspace.search, message=...)
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

LKW.6 wires this on the product host (mirror `lab_application` / `legal_application` interaction flags).

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
| **Background index** | File watcher / cron (LKW.7) | `message_bus.enqueue` → worker runs `local.workspace.index`; Slack notify on batch complete |
| **Long-running synthesize** | Large report | `TaskLongRunningOptions` + checkpoint; user resumes via Slack `approve` / HTTP |

User can **always** submit a new interactive task while background indexing runs — Nexus queue + idempotency prevent duplicate ingests (LKW.7).

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
| **LKW.4** | 4 | Background ingest queue | Tier-0 message_bus | LKW.1 | Planned |
| **LKW.5** | 5 | Chroma persistent index + `LKW_DATA_HOME` | Tier-3 config | LKW.1 | Planned |
| **LKW.6** | 6 | OS daemon packaging + interaction intake | Tier-3 host | LKW.1 | Planned |
| **LKW.6b** | 6b | Slack Socket Mode (optional) | Tier-3 + slack integration | LKW.6 | Planned |
| **LKW.7** | 7 | File watcher + incremental index | Tier-3 worker | LKW.4, LKW.5 | Planned |
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

| Task | Owner module | Deliverable |
|------|--------------|-------------|
| LKW.6.1 | `scripts/lkw-host.*` | Start/stop wrapper for uvicorn |
| LKW.6.2 | `packaging/` | systemd / launchd / Windows scripts (§7.4) |
| LKW.6.3 | `host/factory.py` | `wire_interaction_intake_service` + router |
| LKW.6.4 | `host/settings.py` | `LOCAL_WORKSPACE_INCLUDE_INTERACTIONS` |

**Acceptance:** Service survives reboot; `GET /health` OK; intake JSON → completed task (no Slack required).

**Frontend:** none new. **Backend:** host only.

**ORCH-MAINT-02 — CFG-14 hybrid daemon enablement (operator runbook):**

1. Copy `.env.example` → `.env` in `applications/local_workspace_application/`.
2. Set `LOCAL_WORKSPACE_INCLUDE_SCHEDULER=true`, `LOCAL_WORKSPACE_INCLUDE_INTERACTIONS=true`, `LOCAL_WORKSPACE_INCLUDE_TASK_CONTROL=true`.
3. Optional queue path: `LOCAL_WORKSPACE_INCLUDE_QUEUE_WORKER=true` (see ORCH-MAINT-01 lab scaffold default).
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

#### LKW.7 — Background indexer

| Task | Owner module | Deliverable |
|------|--------------|-------------|
| LKW.7.1 | `host/indexer_worker.py` | File watcher on allowlist roots |
| LKW.7.2 | queue | `message_bus.enqueue` ingest jobs |
| LKW.7.3 | notify | Optional Slack batch complete |

**Acceptance:** Drop file in watched folder → indexed within N minutes without user command.

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
| No file watcher | No auto re-index | LKW.4 worker |
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
